#!/usr/bin/env python3
"""
Ablation listener service for the two-image reference-matching reward.

Each request supplies a *list of (query, reference) image pairs* and a question.
For every pair the model sees both images and answers yes/no to the question.
The pair with the highest yes-probability is the predicted concept.

Endpoints:
 - POST /score         -> score a single (query_paths, reference_paths, question)
 - POST /batch_score   -> score a batch of such requests in one call (fast)

Differences from listener_service.py
-------------------------------------
- ScoreRequest uses `query_paths` + `reference_paths` instead of `candidate_paths`.
- Each scored unit is a two-image message (query image + reference image).
- Default port is 9001 (set via --port or LISTENER_ABLATION_PORT env var).

Requirements:
 - transformers, bitsandbytes, fastapi, uvicorn, pillow
"""

# Standard library
import argparse
import gc
import os
import threading
import time
from typing import List, Any, Optional

# Third-party
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, field_validator
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Local/project-specific
from qwen_vl_utils import process_vision_info

# configure concurrency via env (default 1)
INFERENCE_CONCURRENCY = int(os.environ.get("INFERENCE_CONCURRENCY", "1"))
INFERENCE_ACQUIRE_TIMEOUT = float(os.environ.get("INFERENCE_ACQUIRE_TIMEOUT", "300"))
LISTENER_BATCH_SIZE = int(os.environ.get("LISTENER_BATCH_SIZE", "5"))

_infer_semaphore = threading.Semaphore(INFERENCE_CONCURRENCY)


class ScoreRequest(BaseModel):
    query_paths: List[str]
    reference_paths: List[str]
    question: str
    topk: int = 1

    # Coerce a bare string to a single-element list so the service accepts
    # both {"query_paths": "path/img.jpg", ...} and {"query_paths": ["path/img.jpg"], ...}
    @field_validator("query_paths", "reference_paths", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class BatchScoreRequest(BaseModel):
    batch: List[ScoreRequest]


# ---- Listener class ----

class ListenerAblationService:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", use_8bit: bool = False, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.use_8bit = use_8bit

        load_kwargs = {}
        if use_8bit:
            load_kwargs.update({"load_in_8bit": True, "device_map": {"": device}})
        else:
            load_kwargs.update({"torch_dtype": torch.float16, "device_map": {"": device}})

        print(f"[listener_ablation] loading model {model_name} with kwargs: {load_kwargs}")
        if 'Qwen/Qwen2.5-VL' in model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, trust_remote_code=True, **load_kwargs
            )
        elif 'Qwen/Qwen2-VL' in model_name:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, trust_remote_code=True, **load_kwargs
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

        tok = self.processor.tokenizer
        try:
            self.yes_token_id = tok.convert_tokens_to_ids("yes")
            self.no_token_id = tok.convert_tokens_to_ids("no")
            if self.yes_token_id is None or self.no_token_id is None:
                raise Exception("convert_tokens_to_ids returned None")
        except Exception:
            try:
                self.yes_token_id = tok.encode("yes", add_special_tokens=False)[0]
                self.no_token_id = tok.encode("no", add_special_tokens=False)[0]
            except Exception as e:
                print(f"[listener_ablation WARN] failed to compute yes/no token ids: {e}")
                self.yes_token_id = None
                self.no_token_id = None

        print("[listener_ablation] loaded. Model device:", next(self.model.parameters()).device)
        print(f"[listener_ablation] yes_token_id={self.yes_token_id}, no_token_id={self.no_token_id}")

    def _prepare_image(self, path_or_obj):
        if isinstance(path_or_obj, Image.Image):
            return path_or_obj
        if isinstance(path_or_obj, str):
            if not os.path.exists(path_or_obj):
                raise FileNotFoundError(f"Image not found: {path_or_obj}")
            return path_or_obj
        raise ValueError("image must be a file path or PIL.Image")

    def _build_pair_messages_and_inputs(self, query_images: List[Any], reference_images: List[Any], question: str):
        """
        Build processor inputs for a batch of (query, reference) image pairs.
        Each message contains two images followed by the question text.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": q_img},
                    {"type": "image", "image": r_img},
                    {"type": "text", "text": question},
                ],
            }
            for q_img, r_img in zip(query_images, reference_images)
        ]

        text_list = [
            self.processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model_device)
        return inputs

    def _generate_with_confidence(self, inputs, max_new_tokens=10):
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
            print(f"[LISTENER_ABLATION WARN] generate() failed: {e}. Retrying lightweight fallback.")
            try:
                fallback_kwargs = dict(
                    max_new_tokens=min(8, max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **fallback_kwargs)
            except Exception as e2:
                print(f"[LISTENER_ABLATION ERROR] fallback generate also failed: {e2}. Reraising.")
                raise
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        return outputs

    def score_pairs(self, query_paths: List[str], reference_paths: List[str], question: str,
                    max_new_tokens=10, batch_size: int = 4):
        """
        Score a list of (query, reference) image pairs.
        Returns yes_probabilities list aligned with the input pairs.
        """
        if len(query_paths) != len(reference_paths):
            raise ValueError(
                f"query_paths and reference_paths must have the same length "
                f"(got {len(query_paths)} vs {len(reference_paths)})"
            )
        prepared_q = [self._prepare_image(p) for p in query_paths]
        prepared_r = [self._prepare_image(p) for p in reference_paths]
        yes_probs = []

        for i in range(0, len(prepared_q), batch_size):
            q_chunk = prepared_q[i:i + batch_size]
            r_chunk = prepared_r[i:i + batch_size]
            inputs = self._build_pair_messages_and_inputs(q_chunk, r_chunk, question)
            gen_out = self._generate_with_confidence(inputs, max_new_tokens=max_new_tokens)

            seqs = getattr(gen_out, "sequences", None)
            seqs_cpu = seqs.detach().cpu() if seqs is not None else None
            scores_list = getattr(gen_out, "scores", None)
            first_logits = scores_list[0]  # (batch, vocab)
            fl_cpu = first_logits.detach().cpu().to(dtype=torch.float32)
            vocab_size = fl_cpu.shape[-1]
            yid = self.yes_token_id
            nid = self.no_token_id

            if yid is None or nid is None or yid >= vocab_size or nid >= vocab_size or yid < 0 or nid < 0:
                if seqs_cpu is None:
                    yes_probs.extend([0.0] * len(q_chunk))
                else:
                    decoded = self.processor.batch_decode(seqs_cpu, skip_special_tokens=True)
                    for txt in decoded:
                        yes_probs.append(1.0 if txt.strip().lower().startswith("yes") else 0.0)
            else:
                logp = F.log_softmax(fl_cpu, dim=-1)
                yes_logps = logp[:, yid]
                no_logps = logp[:, nid]
                yes_vs_no = torch.softmax(torch.stack([yes_logps, no_logps], dim=1), dim=1)
                yes_probs.extend([x for x in yes_vs_no])
            # cleanup
            for obj in (gen_out, first_logits, fl_cpu, seqs, seqs_cpu, scores_list, inputs):
                try:
                    del obj
                except Exception:
                    pass
            gc.collect()
            torch.cuda.empty_cache()

        return yes_probs


# ---- FastAPI wrapper ----
app = FastAPI()
listener: Optional[ListenerAblationService] = None


@app.on_event("startup")
def startup():
    global listener
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args, _ = parser.parse_known_args()
    print(f"[listener_service_ablation] starting up, loading model {args.model_name}")
    listener = ListenerAblationService(model_name=args.model_name, use_8bit=args.use_8bit, device=args.device)
    print("[listener_service_ablation] ready.")


@app.post("/score")
def score(req: ScoreRequest):
    start = time.time()
    acquired = _infer_semaphore.acquire(timeout=INFERENCE_ACQUIRE_TIMEOUT)
    if not acquired:
        raise HTTPException(status_code=503, detail="Listener busy; try again later")
    try:
        try:
            yes_probs = listener.score_pairs(req.query_paths, req.reference_paths, req.question, batch_size=1)
            max_prob = max(yes_probs) if yes_probs else 0.0
            total = sum(yes_probs) if sum(yes_probs) > 0 else 1.0
        except Exception as e:
            print(f"[LISTENER_ABLATION ERROR] score() failed: {e}")
            return {"error": str(e)}
        predicted = int(torch.tensor(yes_probs).argmax().item()) if len(yes_probs) > 0 else -1
        soft = yes_probs[predicted] / total if predicted < len(yes_probs) else 0.0
        reward_score = soft if max_prob >= 0.3 else soft * 0.5
        return {
            "yes_probabilities": yes_probs,
            "predicted_index": predicted,
            "reward_score": reward_score,
            "took": time.time() - start,
        }
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
    Batch scoring for two-image pairs.
    Groups items with the same question, flattens their pairs, and calls score_pairs once
    per unique question to maximise GPU utilisation.
    """
    start = time.time()
    acquired = _infer_semaphore.acquire(timeout=INFERENCE_ACQUIRE_TIMEOUT)
    if not acquired:
        raise HTTPException(status_code=503, detail="Listener busy; try again later")
    try:
        items = req.batch or []
        n_items = len(items)
        if n_items == 0:
            return {"results": [], "took": time.time() - start}

        # Group by question
        question_groups = {}
        for idx, item in enumerate(items):
            question_groups.setdefault(item.question, []).append((idx, item.query_paths, item.reference_paths))

        results = [None] * n_items

        for question, group in question_groups.items():
            # Flatten (query, reference) pairs for this question
            flat_query = []
            flat_reference = []
            counts = []
            for (_, q_paths, r_paths) in group:
                n_pairs = len(q_paths)
                counts.append(n_pairs)
                flat_query.extend(q_paths)
                flat_reference.extend(r_paths)

            if len(flat_query) == 0:
                for (orig_idx, _, _) in group:
                    results[orig_idx] = {"yes_probabilities": [], "predicted_index": -1, "reward_score": 0.0}
                continue

            try:
                yes_probs_flat = listener.score_pairs(
                    flat_query, flat_reference, question,
                    batch_size=min(LISTENER_BATCH_SIZE, max(1, len(flat_query)))
                )
            except Exception as e:
                print(f"[LISTENER_ABLATION ERROR] score_pairs failed: {e}")
                yes_probs_flat = [0.0] * len(flat_query)

            # Split back to per-item lists
            cur = 0
            for (orig_idx, _, _), cnt in zip(group, counts):
                yes_no_probs = yes_probs_flat[cur: cur + cnt] if cnt > 0 else []
                predicted = int(torch.tensor(yes_no_probs[0].detach()).argmax().item()) if len(yes_no_probs) > 0 else -1
                cur += cnt
                # predicted = int(torch.tensor(yes_probs).argmax().item()) if len(yes_probs) > 0 else -1
                max_prob = max(yes_no_probs[0]) if yes_no_probs[0] else 0.0
                # total = sum(yes_probs) if sum(yes_probs) > 0 else 1.0
                # soft = yes_probs[predicted] / total if predicted < len(yes_probs) else 0.0
                # reward_score = soft if max_prob >= 0.3 else soft * 0.5
                results[orig_idx] = {
                    "yes_no_probabilities": yes_no_probs[0] if len(yes_no_probs) > 0 else [],
                    "predicted_index": predicted,
                }

        took = time.time() - start
        return {"results": results, "took": took}
    finally:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("LISTENER_ABLATION_PORT", "9001")))
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    uvicorn.run(
        "listener_service_ablation:app",
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
