# Standard library
import argparse
import gc
import json
import os
import re
import sys
import threading
import time
from contextlib import nullcontext
from typing import List, Any, Optional

# Modify path BEFORE other imports
sys.path.insert(0, 'src/virft/src/')

# Third-party
import torch
import torch.nn.functional as F
import uvicorn
from deepspeed.runtime.zero import GatheredParameters
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Local/project-specific
from open_r1.listener_service import ListenerService
from qwen_vl_utils import process_vision_info

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

def parse_descriptions(output, category=None):
    """
    Extract coarse and detailed descriptions from model output with fallback strategies.
    
    Args:
        output: Model generation output string
        category: Optional category name for better fallback parsing
    
    Returns:
        dict with 'thinking', 'coarse', 'detailed' keys
    """
    
    result = {
        "thinking": "",
        "coarse": "",
        "detailed": ""
    }
    
    # ========== Extract Thinking ==========
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', output, re.DOTALL)
    if thinking_match:
        result["thinking"] = thinking_match.group(1).strip()
    else:
        # Fallback: Extract thinking without closing tag
        thinking_fallback = re.search(r'<thinking>(.*?)(?=<coarse|<detailed|$)', output, re.DOTALL)
        if thinking_fallback:
            result["thinking"] = thinking_fallback.group(1).strip()
    
    # ========== Extract Coarse Description ==========
    # Strategy 1: Try with both tags
    coarse_match = re.search(r'<coarse>(.*?)</coarse>', output, re.DOTALL)
    
    if coarse_match:
        result["coarse"] = coarse_match.group(1).strip()
    else:
        # Strategy 2: Opening tag exists but no closing tag
        coarse_fallback = re.search(r'<coarse>(.*?)(?=<detailed|<thinking|$)', output, re.DOTALL)
        if coarse_fallback:
            content = coarse_fallback.group(1).strip()
            # Take first line or until newline
            result["coarse"] = content.split('\n')[0].strip()
        else:
            # Strategy 3: Look for "A photo of a" pattern
            photo_pattern = re.search(r'(A photo of a [^\n.]{5,50})', output, re.IGNORECASE)
            if photo_pattern:
                result["coarse"] = photo_pattern.group(1).strip()
    
    # ========== Extract Detailed Description ==========
    # Strategy 1: Try with both tags
    detailed_match = re.search(r'<detailed>(.*?)</detailed>', output, re.DOTALL)
    
    if detailed_match:
        result["detailed"] = detailed_match.group(1).strip()
    else:
        # Strategy 2: Opening tag exists but no closing tag
        detailed_fallback = re.search(r'<detailed>(.*?)(?=<coarse|<thinking|$)', output, re.DOTALL)
        if detailed_fallback:
            content = detailed_fallback.group(1).strip()
            # Take first sentence or until newline
            result["detailed"] = content.split('\n')[0].strip()
        else:
            # Strategy 3: Look for "The {category}" pattern
            if category:
                category_pattern = re.search(
                    rf'(The {re.escape(category)}[^\n]*?(?:\.|$))', 
                    output, 
                    re.IGNORECASE
                )
                if category_pattern:
                    result["detailed"] = category_pattern.group(1).strip()
            else:
                # Strategy 4: Look for any "The X" pattern
                the_pattern = re.search(r'(The [A-Za-z]+[^\n]{20,200}?\.)', output)
                if the_pattern:
                    result["detailed"] = the_pattern.group(1).strip()
    
    # ========== Cleanup ==========
    # Remove any remaining XML tags
    result["coarse"] = re.sub(r'</?[^>]+>', '', result["coarse"]).strip()
    result["detailed"] = re.sub(r'</?[^>]+>', '', result["detailed"]).strip()
    
    # Remove extra whitespace
    result["coarse"] = ' '.join(result["coarse"].split())
    result["detailed"] = ' '.join(result["detailed"].split())
    str_result = result['coarse'] + ' ' +result['detailed']
    return str_result


class SpeakerService(ListenerService):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-2B-Instruct", device: str = "cuda:0", use_peft=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        load_kwargs = {"torch_dtype": torch.float16, "device_map": {"": device}}
        self.processor = AutoProcessor.from_pretrained(model_name)
        if use_peft:
            from peft import PeftConfig, PeftModel
            config = PeftConfig.from_pretrained(self.model_name)
            print("Loading LoRA model from {}".format(self.model_name))
            if 'Qwen/Qwen2-VL' in model_name:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    device_map=device,)
            elif 'Qwen/Qwen2.5-VL' in model_name:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    config.base_model_name_or_path, 
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                    device_map=device,)
            else:
                raise ValueError(f"Incorrect model name: {model_name}")
            self.model = PeftModel.from_pretrained(model, self.model_name)
        else:
            print(f"Loading Model from {self.model_name}")
            if 'Qwen/Qwen2.5-VL' in model_name:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name, 
                    attn_implementation="flash_attention_2", 
                    trust_remote_code=True, 
                    **load_kwargs)
            elif 'Qwen/Qwen2-VL' in model_name:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name, 
                    attn_implementation="flash_attention_2", 
                    trust_remote_code=True, 
                    **load_kwargs)
            else:
                raise ValueError(f"Incorrect model name: {model_name}")
        self.model.eval()
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
            # response_clean = extract_speaker_answer_term(response)
            response_clean = parse_descriptions(response)
            responses.append(response_clean) # <-- change here
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    args, _ = parser.parse_known_args()
    print("[speaker_service] starting up, loading model...")
    speaker= SpeakerService(model_name=args.model_name, device=args.device, use_peft=False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # store args globally so startup() can access
    service_args = args

    uvicorn.run("speaker_service:app",
                host=args.host,
                port=args.port,
                reload=True,
                access_log=False,
                log_level="info")