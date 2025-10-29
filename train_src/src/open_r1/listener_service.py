#!/usr/bin/env python3
"""
Standalone listener service to run on GPU 0.
Endpoints:
 - POST /score         -> score a single (candidate_paths, question)
 - POST /batch_score   -> score a batch of such requests in one call (fast)

Requirements:
 - transformers, bitsandbytes, fastapi, uvicorn, pillow
 - Put this file on the same machine that can access image paths.
"""

import os
import time
import threading
import argparse
from typing import List, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import torch, gc
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # your util that prepares vision inputs
import torch.nn.functional as F

# configure concurrency via env (default 1)
INFERENCE_CONCURRENCY = int(os.environ.get("INFERENCE_CONCURRENCY", "1"))
INFERENCE_ACQUIRE_TIMEOUT = float(os.environ.get("INFERENCE_ACQUIRE_TIMEOUT", "300"))  # seconds
LISTENER_BATCH_SIZE = int(os.environ.get("LISTENER_BATCH_SIZE", "5"))  # per-model-call inner batch size (lower default)

_infer_semaphore = threading.Semaphore(INFERENCE_CONCURRENCY)


class ScoreRequest(BaseModel):
    candidate_paths: List[str]
    question: str
    topk: int = 1

class BatchScoreRequest(BaseModel):
    batch: List[ScoreRequest]

# ---- Listener class ----
class ListenerService:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", use_8bit: bool = False, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_8bit = use_8bit

        # Load model (standalone process - no DeepSpeed here)
        load_kwargs = {}
        if use_8bit:
            # bitsandbytes must be installed and supported
            load_kwargs.update({"load_in_8bit": True, "device_map": {"": device}})
        else:
            # fallback to fp16
            load_kwargs.update({"torch_dtype": torch.float16, "device_map": {"": device}})

        print(f"[listener] loading model {model_name} with kwargs: {load_kwargs}")
        # Load model (trust_remote_code in case of custom Qwen code)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True, **load_kwargs
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Precompute yes/no token ids robustly (some tokenizers may behave differently)
        tok = self.processor.tokenizer
        # prefer convert_tokens_to_ids if available, fallback to encode
        try:
            self.yes_token_id = tok.convert_tokens_to_ids("yes")
            self.no_token_id = tok.convert_tokens_to_ids("no")
            # convert_tokens_to_ids may return None if token doesn't exist as single token
            if self.yes_token_id is None or self.no_token_id is None:
                raise Exception("convert_tokens_to_ids returned None")
        except Exception:
            # fallback to encode then take first id
            try:
                self.yes_token_id = tok.encode("yes", add_special_tokens=False)[0]
                self.no_token_id = tok.encode("no", add_special_tokens=False)[0]
            except Exception as e:
                print(f"[listener WARN] failed to compute yes/no token ids: {e}")
                self.yes_token_id = None
                self.no_token_id = None

        print("[listener] loaded. Model device:", next(self.model.parameters()).device)
        print(f"[listener] yes_token_id={self.yes_token_id}, no_token_id={self.no_token_id}")

    def _prepare_image(self, path_or_obj):
        """Return either PIL.Image or path (processor accepts both)."""
        if isinstance(path_or_obj, Image.Image):
            return path_or_obj
        if isinstance(path_or_obj, str):
            if not os.path.exists(path_or_obj):
                raise FileNotFoundError(f"Image not found: {path_or_obj}")
            return path_or_obj
        raise ValueError("candidate must be file path or PIL.Image")

    def _build_messages_and_inputs(self, images: List[Any], question: str):
        """
        Build processor inputs for a *batch of images* with the same question.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
            for img in images
        ]

        text_list = [ self.processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
                     for m in messages ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # ensure tensors on same device as model
        model_device = next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model_device)
        return inputs

    def _generate_with_confidence(self, inputs, max_new_tokens=10):
        """
        Run model.generate with defensive flags:
        - use torch.inference_mode (safe here, standalone)
        - explicitly disable sync-related flags so generate won't call dist.collectives
        - try one fallback on error
        """
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        self.model.eval()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            # fallback with safer settings
            print(f"[LISTENER WARN] generate() failed: {e}. Retrying lightweight fallback.")
            try:
                fallback_kwargs = dict(
                    max_new_tokens=min(8, max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                # for flag in ("synced_gpus", "synchronized_gpus", "sync_gpus"):
                #     fallback_kwargs[flag] = False
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **fallback_kwargs)
            except Exception as e2:
                print(f"[LISTENER ERROR] fallback generate also failed: {e2}. Reraising.")
                raise

        # free caches
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        return outputs
        
    def score_candidates(self, candidate_paths: List[str], question: str, max_new_tokens=10, batch_size: int = 8):
        """
        Score a list of candidate_paths. Returns yes_probabilities list aligned with candidate_paths.
        Uses fp32 log_softmax on CPU for stable numeric results and decodes fallback if scores missing.
        """
        prepared = [self._prepare_image(p) for p in candidate_paths]
        yes_probs = []

        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i:i+batch_size]
            inputs = self._build_messages_and_inputs(chunk, question)
            gen_out = self._generate_with_confidence(inputs, max_new_tokens=max_new_tokens)
            seqs = getattr(gen_out, "sequences", None)
            if seqs is not None:
                try:
                    seqs_cpu = seqs.detach().cpu()
                except Exception:
                    seqs_cpu = seqs
            else:
                seqs_cpu = None
            scores_list = getattr(gen_out, "scores", None)
            first_logits = scores_list[0]  # (batch, vocab)
            fl_cpu = first_logits.detach().cpu().to(dtype=torch.float32)
            vocab_size = fl_cpu.shape[-1]
            yid = self.yes_token_id
            nid = self.no_token_id

            if yid is None or nid is None or yid >= vocab_size or nid >= vocab_size or yid < 0 or nid < 0:
                # fallback decision from decoded text (use CPU seqs)
                if seqs_cpu is None:
                    yes_probs.extend([0.0] * len(chunk))
                else:
                    decoded = self.processor.batch_decode(seqs_cpu, skip_special_tokens=True)
                    for txt in decoded:
                        yes_probs.append(1.0 if txt.strip().lower().startswith("yes") else 0.0)
            else:
                # compute log_softmax on CPU and normalize between yes/no
                logp = F.log_softmax(fl_cpu, dim=-1)  # (batch, vocab)
                yes_logps = logp[:, yid]
                no_logps = logp[:, nid]
                yes_vs_no = torch.softmax(torch.stack([yes_logps, no_logps], dim=1), dim=1)[:, 0].tolist()
                yes_probs.extend([float(x) for x in yes_vs_no])
            try:
                # remove references that may hold GPU memory (gen_out carries scores/sequences)
                del gen_out
            except Exception:
                pass
            try:
                del first_logits
            except Exception:
                pass
            try:
                del fl_cpu
            except Exception:
                pass
            try:
                del seqs, seqs_cpu, scores_list
            except Exception:
                pass
            # Also remove inputs we created earlier (BatchEncoding with tensors)
            try:
                del inputs
            except Exception:
                pass

            # enforce python GC and free cached GPU memory
            gc.collect()
            torch.cuda.empty_cache()
        return yes_probs

# ---- FastAPI wrapper ----
app = FastAPI()
listener: Optional[ListenerService] = None

@app.on_event("startup")
def startup():
    global listener
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args, _ = parser.parse_known_args()
    print(f"[listener_service] starting up, loading model {args.model_name}")
    listener = ListenerService(model_name=args.model_name, use_8bit=args.use_8bit, device=args.device)
    print("[listener_service] ready.")


    
@app.post("/score")
def score(req: ScoreRequest):
    start = time.time()
    # Use semaphore to prevent concurrent forward passes
    acquired = _infer_semaphore.acquire(timeout=INFERENCE_ACQUIRE_TIMEOUT)
    if not acquired:
        raise HTTPException(status_code=503, detail="Listener busy; try again later")
    try:
        try:
            yes_probs = listener.score_candidates(req.candidate_paths, req.question, batch_size=1)
            # max(1, len(req.candidate_paths)))
        except Exception as e:
            print(f"[LISTENER ERROR] score() failed: {e}")
            return {"error": str(e)}
        predicted = int(torch.tensor(yes_probs).argmax().item()) if len(yes_probs)>0 else -1
        return {"yes_probabilities": yes_probs, "predicted_index": predicted, "took": time.time()-start}
    finally:
        try:
            _infer_semaphore.release()
        except Exception:
            pass
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

@app.post("/batch_score")
def batch_score(req: BatchScoreRequest):
    """
    Robust batch_score that:
      - serializes concurrent whole-request inferences using a semaphore
      - groups items with the same question and flattens candidate lists so we can call the model once per unique question
      - splits flattened results back to per-item outputs
    """
    start = time.time()

    # Try to acquire semaphore (block up to INFERENCE_ACQUIRE_TIMEOUT)
    acquired = _infer_semaphore.acquire(timeout=INFERENCE_ACQUIRE_TIMEOUT)
    if not acquired:
        # Couldn't acquire; tell the client to retry later
        raise HTTPException(status_code=503, detail="Listener busy; try again later")

    try:
        items = req.batch or []
        n_items = len(items)
        if n_items == 0:
            return {"results": [], "took": time.time() - start}

        # Group items by question to allow a single scoring call per unique question
        question_groups = {}  # question -> list of (original_index, candidate_paths)
        for idx, item in enumerate(items):
            q = item.question
            question_groups.setdefault(q, []).append((idx, item.candidate_paths))

        # Prepare results placeholder
        results = [None] * n_items

        # Process each group (one model call per unique question)
        for question, group in question_groups.items():
            # Flatten all candidate paths for this question
            flat_paths = []
            counts = []  # how many candidates per original item
            for (_, candidate_paths) in group:
                counts.append(len(candidate_paths))
                flat_paths.extend(candidate_paths)

            if len(flat_paths) == 0:
                for (orig_idx, _) in group:
                    results[orig_idx] = {"yes_probabilities": [], "predicted_index": -1}
                continue

            # call listener.score_candidates once for this entire flattened list
            try:
                yes_probs_flat = listener.score_candidates(flat_paths, question, batch_size=min(1, max(1, len(flat_paths))))
            except Exception as e:
                print(f"[LISTENER ERROR] score_candidates failed: {e}")
                yes_probs_flat = [0.0] * len(flat_paths)

            # split flattened probabilities back to per-item lists
            cur = 0
            for (orig_idx, _), cnt in zip(group, counts):
                sub = yes_probs_flat[cur: cur + cnt] if cnt > 0 else []
                cur += cnt
                predicted = int(torch.tensor(sub).argmax().item()) if len(sub) > 0 else -1
                results[orig_idx] = {"yes_probabilities": sub, "predicted_index": predicted}
        print(f"[results]->{results}")
        took = time.time() - start
        return {"results": results, "took": took}

    finally:
        # Always release semaphore and try to free GPU cache
        try:
            _infer_semaphore.release()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

# ---- run server (if executed directly) ----
if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # start uvicorn programmatically (single worker, single process)
    uvicorn.run("listener_service:app", host="0.0.0.0", port=args.port, log_level="info", access_log=False)
