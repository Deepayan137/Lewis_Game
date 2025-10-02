import os
import re
import threading
import time
import torch, gc
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
from PIL import Image
import json
from typing import List, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import nullcontext
from deepspeed.runtime.zero import GatheredParameters
import sys

sys.path.insert(0, 'src/virft/src/')
from open_r1.listener_service import ListenerService, BaseModel

# configure concurrency via env (default 1)
INFERENCE_CONCURRENCY = int(os.environ.get("INFERENCE_CONCURRENCY", "1"))
INFERENCE_ACQUIRE_TIMEOUT = float(os.environ.get("INFERENCE_ACQUIRE_TIMEOUT", "300"))  # seconds
SPEAKER_BATCH_SIZE = int(os.environ.get("SPEAKER_BATCH_SIZE", "5"))  # per-model-call inner batch size (lower default)

_infer_semaphore = threading.Semaphore(INFERENCE_CONCURRENCY)

class DescribeRequest(BaseModel):
    candidate_paths: List[str]
    question: str
    topk: int = 1

class BatchDescribeRequest(BaseModel):
    batch: List[DescribeRequest]

def extract_speaker_answer_term(text: str) -> str:
    """
    Return the content between <answer>...</answer> if present,
    otherwise fall back to everything after an opening <answer> or the original text.
    """
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # fallback: opening tag present but no closing tag
    fallback_pattern = r'<answer>\s*(.*?)\s*$'
    fallback_match = re.search(fallback_pattern, text, re.DOTALL)
    if fallback_match:
        return fallback_match.group(1).strip()

    return text


class SpeakerService(ListenerService):
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        load_kwargs = {"torch_dtype": torch.float16, "device_map": {"": device}}
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, trust_remote_code=True, **load_kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.baseline = 0.0  # Just a scalar!
        self.baseline_momentum = 0.9
        # super().__init__(model_name=model_name, use_8bit=False, device=device)
        print(f"speaker loaded from {model_name}")
        print("[speaker] loaded. Model device:", next(self.model.parameters()).device)
    
    def _generate(self, inputs, max_new_tokens=10):
        """
        Run model.generate with defensive flags:
        - use torch.inference_mode (safe here, standalone)
        - explicitly disable sync-related flags so generate won't call dist.collectives
        - try one fallback on error
        """
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for consistency
            use_cache=True,   # Enable KV cache for speed
            pad_token_id=self.processor.tokenizer.eos_token_id,
            temperature=0.4,  # Add explicit temperature
            top_p=0.9,
            repetition_penalty=1.0
        )
        self.model.eval()
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
        except Exception as e:
            # fallback with safer settings
            print(f"[SPEAKER WARN] generate() failed: {e}. Retrying lightweight fallback.")
            try:
                fallback_kwargs = dict(
                    max_new_tokens=min(8, max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    # return_dict_in_generate=True,
                    # output_scores=True,
                )
                # for flag in ("synced_gpus", "synchronized_gpus", "sync_gpus"):
                #     fallback_kwargs[flag] = False
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **fallback_kwargs)
            except Exception as e2:
                print(f"[SPEAKER ERROR] fallback generate also failed: {e2}. Reraising.")
                raise

        # free caches
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

        return outputs
    
    def describe_images(self, candidate_paths: List[str], question: str, max_new_tokens=128, batch_size: int = 8):
        """
        Score a list of candidate_paths. Returns yes_probabilities list aligned with candidate_paths.
        Uses fp32 log_softmax on CPU for stable numeric results and decodes fallback if scores missing.
        """
        prepared = [self._prepare_image(p) for p in candidate_paths]
        responses = []

        for i in range(0, len(prepared), batch_size):
            chunk = prepared[i:i+batch_size]
            inputs = self._build_messages_and_inputs(chunk, question)
            generated_ids = self._generate(inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            response_clean = extract_speaker_answer_term(response)
            responses.append(response_clean)
            try:
                # remove references that may hold GPU memory (gen_out carries scores/sequences)
                del generated_ids
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
        return responses

# ---- FastAPI wrapper ----
app = FastAPI()
speaker: Optional[SpeakerService] = None

@app.on_event("startup")
def startup():
    global speaker
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="share_models/Qwen2.5-VL-7B-Instruct_GRPO_lewis_YoLLaVA_all_train_seed_23")
    parser.add_argument("--device", type=str, default="cuda:0")
    args, _ = parser.parse_known_args()
    print("[speaker_service] starting up, loading model...")
    speaker= SpeakerService(model_name=args.model_name, device=args.device)
    print("[speaker_service] ready.")

@app.post("/describe")
def describe(req: DescribeRequest):
    start = time.time()
    acquired = _infer_semaphore.acquire(timeout=INFERENCE_ACQUIRE_TIMEOUT)
    if not acquired:
        # Couldn't acquire; tell the client to retry later
        raise HTTPException(status_code=503, detail="Speaker busy; try again later")
    try:
        try:
            description = speaker.describe_image(req.candidate_paths, req.question, batch_size=1)

            # max(1, len(req.candidate_paths)))
        except Exception as e:
            print(f"[SPEAKER ERROR] score() failed: {e}")
            return {"error": str(e)}
        return {"description": description, "took": time.time()-start}
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
    
@app.post("/batch_describe")
def batch_describe(req: BatchDescribeRequest):
    start = time.time()
    print("")
    # Try to acquire semaphore (block up to INFERENCE_ACQUIRE_TIMEOUT)
    acquired = _infer_semaphore.acquire(timeout=INFERENCE_ACQUIRE_TIMEOUT)
    if not acquired:
        # Couldn't acquire; tell the client to retry later
        raise HTTPException(status_code=503, detail="Speaker busy; try again later")
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
                    results[orig_idx] = {"description": ""}
                continue
            # call listener.score_candidates once for this entire flattened list
            try:
                descriptions = speaker.describe_images(flat_paths, question, batch_size=min(1, max(1, len(flat_paths))))
            except Exception as e:
                print(f"[SPEAKER ERROR] score_candidates failed: {e}")
                descriptions = [""] * len(flat_paths)

            # split flattened probabilities back to per-item lists
            cur = 0
            for (orig_idx, _), cnt in zip(group, counts):
                sub = descriptions[cur: cur + cnt] if cnt > 0 else []
                cur += cnt
                results[orig_idx] = {"descriptions":descriptions}

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
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # start uvicorn programmatically (single worker, single process)
    uvicorn.run("speaker_service:app", host="0.0.0.0", port=args.port, log_level="info", access_log=False, reload=True)