# Standard library
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

# Modify path BEFORE other third-party imports
sys.path.insert(0, 'train_src/')

# Third-party
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import DatasetDict, Dataset, load_dataset, load_from_disk
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# Local/project-specific
from dist_helpers import *
from math_verify import parse, verify
from open_r1.logger import PredictionLogger
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
# from speaker_service import parse_descriptions
from rouge_helpers import to_str, split_sentences, rouge_2_sw

logger = PredictionLogger(log_path=os.getenv("LOG_PATH"))

def extract_answer_content(text):
    """
    Extract the content between <answer> tags from the given text.
    
    Args:
        text (str): Input text containing <answer> tags
        
    Returns:
        str: Content between answer tags, or empty string if not found
    """
    # Pattern to match content between <answer> and </answer> tags
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return ""

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
    # ========== Extract State ==========
    # Strategy 1: Try with both tags
    detailed_match = re.search(r'<state>(.*?)</state>', output, re.DOTALL)
    
    if detailed_match:
        result["state"] = detailed_match.group(1).strip()
    else:
        result["state"] = ""
    
    # ========== Extract Location ==========
    # Strategy 1: Try with both tags
    detailed_match = re.search(r'<location>(.*?)</location>', output, re.DOTALL)
    if detailed_match:
        result["location"] = detailed_match.group(1).strip()
    else:
        result["location"] = ""
    # ========== Cleanup ==========
    # Remove any remaining XML tags
    result["coarse"] = re.sub(r'</?[^>]+>', '', result["coarse"]).strip()
    result["detailed"] = re.sub(r'</?[^>]+>', '', result["detailed"]).strip()
    result["state"] = re.sub(r'</?[^>]+>', '', result["state"]).strip()
    result["location"] = re.sub(r'</?[^>]+>', '', result["location"]).strip()
    # Remove extra whitespace
    result["coarse"] = ' '.join(result["coarse"].split())
    result["detailed"] = ' '.join(result["detailed"].split())
    result["state"] = ' '.join(result["state"].split())
    result["location"] = ' '.join(result["location"].split())
    str_result = result['coarse'] + '. ' +result['detailed']
    return str_result, result

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    listener_reward_mode: str = field(
        default="soft_gated",
        metadata={"help": "Listener reward mode: 'binary' (1/0), "
                        "'soft_gated' (soft only if correct), "
                        "'soft_always' (soft regardless of correctness)."},
    )

LISTENER_URL = os.environ.get("LISTENER_URL", "http://127.0.0.1:9000/batch_score")
LISTENER_REWARD_MODE = os.environ.get("LISTENER_REWARD_MODE", "soft_gated")
LISTENER_TIMEOUT = float(os.environ.get("LISTENER_TIMEOUT", 30.0))  # seconds
_listener_cache = {}
# Defaults you can tune via environment variables:
DEFAULT_LISTENER_CHUNK_SIZE = int(os.environ.get("LISTENER_CHUNK_SIZE", "4"))
DEFAULT_CHUNK_DELAY = float(os.environ.get("LISTENER_CHUNK_DELAY", "0.05"))  # seconds between chunks
DEFAULT_MAX_RETRIES = int(os.environ.get("LISTENER_MAX_RETRIES", "3"))
DEFAULT_BACKOFF = float(os.environ.get("LISTENER_BACKOFF_FACTOR", "1.0"))

def _call_listener_batch(batch_requests, timeout=LISTENER_TIMEOUT,
                         max_retries=DEFAULT_MAX_RETRIES,
                         backoff_factor=DEFAULT_BACKOFF,
                         chunk_size=DEFAULT_LISTENER_CHUNK_SIZE,
                         chunk_delay=DEFAULT_CHUNK_DELAY):
    """
    Robust client-side call to listener with chunking and retries.

    batch_requests: list of dicts: {"candidate_paths": [...], "question": "..."}
    Returns: list of responses aligned with batch_requests (each response is dict).
    """
    if not batch_requests:
        return []

    url = os.environ.get("LISTENER_URL", "http://127.0.0.1:9000/batch_score")
    parsed = urlparse(url)
    host = parsed.hostname or ""

    # Decide whether to honor environment proxies. Default: do NOT trust env proxies
    # unless the host is explicitly present in NO_PROXY/no_proxy.
    no_proxy_env = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    no_proxy_hosts = [h.strip() for h in no_proxy_env.split(",") if h.strip()]
    use_trust_env = host in no_proxy_hosts

    sess = requests.Session()
    sess.trust_env = bool(use_trust_env)

    results_all = []
    total = len(batch_requests)
    connect_timeout = min(5.0, timeout)
    read_timeout = timeout

    # helper to build neutral fallback responses for a slice
    def neutral_resp_slice(slice_requests):
        return [{"yes_probabilities": [0.0]*len(req.get("candidate_paths", [])), "predicted_index": -1}
                for req in slice_requests]

    # iterate chunk by chunk
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = batch_requests[start:end]
        payload = {"batch": chunk}
        attempt = 0
        chunk_success = False
        r = sess.post(url, json=payload, timeout=180)
        r.raise_for_status()
        resp = r.json()
        if isinstance(resp, dict) and "results" in resp:
            res = resp["results"]
        elif isinstance(resp, list):
            res = resp
        else:
            print(f"[WARN] unexpected JSON format for chunk {start}:{end}: {type(resp)}; resp keys: {list(resp.keys()) if isinstance(resp, dict) else 'N/A'}")
            res = neutral_resp_slice(chunk)
        results_all.extend(res)
    if len(results_all) != total:
        print(f"[WARN] results length mismatch: got {len(results_all)} expected {total}; filling remainder with neutral responses")
        # pad if needed
        while len(results_all) < total:
            i = len(results_all)
            req = batch_requests[i]
            results_all.append({"yes_probabilities": [0.0]*len(req.get("candidate_paths", [])), "predicted_index": -1})
        # trim if somehow longer
        results_all = results_all[:total]

    return results_all

def accuracy_reward(completions, solution, logger=None, **kwargs):
    """
    Distributed-aware accuracy reward with rank-0 chunked listener calls and local caching.
    - Only rank 0 calls the external listener service.
    - Rank 0 dedups unique payloads, sends them in chunks to the listener (safe for memory),
      and broadcasts only the newly-fetched cache entries to all ranks.
    - Each rank then computes per-example rewards from the (now-updated) local cache.
    """
    global _listener_cache

    print(f"[accuracy_reward] listener_reward_mode={LISTENER_REWARD_MODE}")

    contents = [completion[0]["content"] for completion in completions]
    n = len(contents)

    path_candidates = kwargs.get('ret_paths')
    names_list = kwargs.get('names')
    categories = kwargs.get('category')
    if path_candidates is None or categories is None:
        return [0.0] * n
    # Build local requests and keys
    local_requests = []      # payloads this rank needs (for gather)
    local_indices = []       # indices into the local batch that correspond to each payload
    local_keys = []          # key for each local example (used to lookup rewards later)
    for i, content in enumerate(contents):
        cat = categories[i]
        # content_clean = extract_answer_content(content)
        content_clean, _ = parse_descriptions(content)
        # question = f'Does the description -> "{content_clean}" match the {cat} in the image? Answer in yes or no.'
        question = f'Does this description "{content_clean}" accurately describe the main subject in the image? Answer yes or no.'
        candidate_paths = path_candidates[i]
        key = (tuple(candidate_paths), question)
        local_keys.append(key)
        if key not in _listener_cache:
            payload = {"candidate_paths": candidate_paths, "question": question}
            local_requests.append(payload)
            local_indices.append(i)

    # Distributed gather: collect all ranks' local_requests to everyone (fallback helper handles non-dist)
    gathered_requests = dist_all_gather_object_fallback(local_requests)  # list-of-lists (len=world_size)

    world_rank, world_size = get_world_info()

    # Rank 0: dedupe and call listener in safe chunks
    if world_rank == 0:
        seen = set()
        unique_payloads = []
        unique_keys = []
        # flatten gathered requests and dedupe
        for rank_list in gathered_requests:
            for payload in rank_list or []:
                # Suggestion for a better key: use a tuple of (question, tuple of sorted candidate_paths)
                # This ensures the key is order-invariant for candidate_paths and robust to accidental list/tuple differences.
                key = (payload["question"], tuple(sorted(payload["candidate_paths"])))
                if key not in seen:
                    seen.add(key)
                    unique_payloads.append(payload)
                    unique_keys.append(key)

        # If there are unique payloads, call listener in chunks to avoid huge single request
        new_entries = {}
        if unique_payloads:
            # choose chunk size (can be tuned via env)
            chunk_size = int(os.environ.get("LISTENER_CHUNK_SIZE", "4"))
            # If the listener-caller already does internal chunking, this outer chunking is defensive.
            for i in range(0, len(unique_payloads), chunk_size):
                chunk = unique_payloads[i: i + chunk_size]
                # Call the client helper which includes its own retry/inner-chunking logic
                try:
                    chunk_results = _call_listener_batch(chunk)
                except Exception as e:
                    print(f"[WARN] _call_listener_batch failed for chunk {i}:{i+len(chunk)}: {e}")
                    # fallback neutral results for this chunk
                    chunk_results = [{"yes_probabilities": [0.0] * len(p["candidate_paths"]), "predicted_index": -1} for p in chunk]

                # map chunk results to keys and update local rank0 cache
                for payload_item, res in zip(chunk, chunk_results):
                    k = (tuple(payload_item["candidate_paths"]), payload_item["question"])
                    # store result in global cache (rank0's view)
                    _listener_cache[k] = res
                    new_entries[k] = res
        else:
            new_entries = {}  # nothing new

    else:
        # non-zero ranks expect to receive new_entries broadcasted from rank 0
        new_entries = None

    # Broadcast only the new entries (small dict) from rank 0 to all ranks
    # Use the safe object-broadcast helper (handles older PyTorch with pickle fallback)
    new_entries = dist_broadcast_object_fallback(new_entries, src=0)

    # Merge received new entries into local cache
    if new_entries:
        _listener_cache.update(new_entries)

    # Now compute rewards locally using the (now-consistent) cache
    rewards = []
    for i, content in enumerate(contents):
        names = names_list[i]
        key = local_keys[i]
        res = _listener_cache.get(key, None)
        print(f"[RESULTS] -> {res}")
        soft_reward_score = 0.0  # safe default for all reward modes
        if res is None:
            yes_probs = [0.0] * len(path_candidates[i])
            predicted_index = -1
        else:
            yes_probs = res.get("yes_probabilities", res.get("yes_probs", []))
            # coerce to list if needed
            if not isinstance(yes_probs, list):
                try:
                    import numpy as _np
                    yes_probs = list(_np.array(yes_probs).tolist())
                except Exception:
                    yes_probs = [0.0] * len(path_candidates[i])
            predicted_index = int(res.get("predicted_index", -1))
            soft_reward_score = res.get("reward_score", 0.0)
        prediction = names[predicted_index] if predicted_index >=0  else "<none>"
        target = solution[i]
        correct = (prediction == target and predicted_index >= 0)
        if LISTENER_REWARD_MODE == "binary":
            reward = 1.0 if correct else 0.0
        elif LISTENER_REWARD_MODE == "soft_always":
            reward = soft_reward_score
        else:  # "soft_gated" (default)
            reward = soft_reward_score if correct else 0.0
        rewards.append(reward)
        # optional logging
        if logger is not None:
            # content   _clean = extract_answer_content(content)
            content_clean, _ = parse_descriptions(content)
            logger.log(reward, content_clean, target)

        # debug file logging (only rank0's LOCAL_RANK==0 writes)
        if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = f'debug_files/debug_speaker_{os.getenv("LOCAL_RANK", "0")}_job_{os.getenv("SLURM_JOB_ID", "unknown")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} | reward_mode: {LISTENER_REWARD_MODE} -------------\n")
                    f.write(f"content: {content_clean}\n")
                    f.write(f"sol: {solution[i]}\n")
                    f.write(f"prediction: {prediction}, target_index: {target}, correct: {correct}, soft_score: {soft_reward_score:.4f}\n")
                    f.flush()
            except Exception as e:
                if logger is not None:
                    logger.log(f"Logging error: {e}", content_clean, target)

    return rewards

def overlap_penalty(completions, solution, logger=None, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        try:
            _, concept = parse_descriptions(content)
        except Exception:
            rewards.append(0.0)
            continue
        detailed_text = to_str(concept.get("detailed", ""))
        state_text    = to_str(concept.get("state", ""))
        location_text = to_str(concept.get("location", ""))
        reference   = (state_text + " " + location_text).strip()
        sentences = split_sentences(detailed_text)
        sentence_scores = []

        for sent in sentences:
            r2_sw  = rouge_2_sw(sent, reference)
            sentence_scores.append({
                "sentence":   sent,
                "rouge_2_sw": round(r2_sw, 4)
            })

        sw_scores = [s["rouge_2_sw"] for s in sentence_scores]
        max_sw_score = round(max(sw_scores), 4) if sw_scores else 0.0
        if logger and max_sw_score > 0:
            top = max(sentence_scores, key=lambda x: x["rouge_2_sw"])
            logger.log(f"overlap_penalty fired: {max_sw_score:.3f} | '{top['sentence'][:80]}'")
        reward = -max_sw_score  # negative reward for high overlap
        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """
    Reward function that checks if the model's response contains all required fields:
    1. <thinking>...</thinking> tag (reasoning process)
    2. <coarse>...</coarse> tag (5-6 word description)
    3. <detailed>...</detailed> tag (permanent identity features)
    4. <state>...</state> tag (pose and body position)
    5. <location>...</location> tag (background and positioning)

    Rewards:
    - Partial credit: 0.2 per field with proper open/close tags
    - Total: 1.0 if all five fields are present and properly closed
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content in completion_contents:
        # Isolate the assistant's response (comes after "assistant\n")
        parts = content.split('assistant\n')
        model_output = parts[-1] if len(parts) > 1 else content

        # Check each field individually for partial credit
        score = 0.0

        if re.search(r'<thinking>.*?</thinking>', model_output, re.DOTALL):
            score += 0.34
        if re.search(r'<coarse>.*?</coarse>', model_output, re.DOTALL):
            score += 0.33
        if re.search(r'<detailed>.*?</detailed>', model_output, re.DOTALL):
            score += 0.33
        # if re.search(r'<state>.*?</state>', model_output, re.DOTALL):
        #     score += 0.2
        # if re.search(r'<location>.*?</location>', model_output, re.DOTALL):
        #     score += 0.2

        rewards.append(round(score, 2))

    return rewards

def length_reward(completions, logger=None, **kwargs):
    """
    Penalizes <detailed> longer than 1 sentence and <coarse> outside 5-7 words.
    """
    rewards = []
    completion_contents = [c[0]["content"] for c in completions]

    for content in completion_contents:
        _, parsed = parse_descriptions(content)
        score = 0.0

        # Penalize multi-sentence <detailed>
        detailed = parsed.get("detailed", "").strip()
        sentences = [s.strip() for s in re.split(r'[.!?](?:\s|$)', detailed) if s.strip()]
        detailed_words = len(detailed.split())
        if len(sentences) > 1:
            score -= 0.2
        elif detailed_words < 20: 
            score -= 0.2
        # Penalize <coarse> outside expected word count
        coarse = parsed.get("coarse", "").strip()
        if not (6 <= len(coarse.split()) <= 8):
            score -= 0.1

        rewards.append(score)

        if logger is not None:
            logger.log({
                "length_reward": score,
                "detailed_sentence_count": len(sentences),
                "coarse_word_count": len(coarse.split()),
            })

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward,
    "overlap": overlap_penalty,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def count_trainable_parameters_detailed(model: torch.nn.Module):
    """
    Counts and prints the details of trainable parameters in a PyTorch model.

    Args:
        model: A PyTorch nn.Module.
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(f"Trainable: {name} - size: {param.size()} - dtype: {param.dtype}")
        else:
            non_trainable_params += num_params
            print(f"Frozen: {name} - size: {param.size()} - dtype: {param.dtype}")

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    return trainable_params

def make_conversation_lewis_game(example):
    return {
        "prompt": [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},  # This will be filled with example["image"]
                    {"type": "text", "text": example["speaker_problem"]},
                ],
            },
        ],
    }

def main(script_args, training_args, model_args, lora_args):
    global LISTENER_REWARD_MODE
    script_args.reward_funcs = ['accuracy', 'format', 'length']
    print(script_args)
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    from datasets import load_dataset
    dataset_path = script_args.dataset_name
    dataset = DatasetDict.load_from_disk(dataset_path)
    dataset = dataset.map(make_conversation_lewis_game)
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)
    if model_args.use_peft:
        print("Training in LORA mode")
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora_args.lo_rank,  # the rank of the LoRA matrices
            lora_alpha=lora_args.lo_alpha, # the weight
            lora_dropout=lora_args.lo_dropout, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules="all-linear", # the name of the layers to add LoRA
            modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
        )
    else:
        peft_config = None
    # global listener_model
    # listener_model = None
    LISTENER_URL = os.environ.get("LISTENER_URL", "http://127.0.0.1:9000/batch_score")
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

@dataclass
class LoRAArgs:
    """Local dataclass so HF parser will accept LoRA CLI flags."""
    lo_rank: int = field(default=64, metadata={"help": "LoRA rank"})
    lo_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lo_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig, LoRAArgs))
    script_args, training_args, model_args, lora_args = parser.parse_args_and_config()
    LISTENER_REWARD_MODE = script_args.listener_reward_mode
    main(script_args, training_args, model_args, lora_args)
