#!/usr/bin/env python3
"""
evaluate_with_qwen.py

Usage:
    python evaluate_with_qwen.py --input data/descriptions.json --prompt prompts/evaluator_prompt.json \
        --out results/eval_results.jsonl --batch-size 64

What it does:
- Loads a JSON dataset (structure described in the README below).
- Builds concatenated descriptions (general + distinguishing / coarse features).
- Uses a PyTorch DataLoader to produce batches.
- Loads a prompt template JSON (editable) and injects each batch as the "BEGIN INPUT".
- Calls Qwen-8B via the `call_qwen` function (pluggable: you must implement it for your client).
- Parses and validates the returned JSON, with one retry on parse failure.
- Writes per-item results to a JSONL output file.

Notes:
- Replace the call_qwen(...) implementation with your Qwen-8B client call.
- Use temperature=0 and deterministic decoding for the evaluator.
"""

import argparse
import json
import math
import statistics as st
import os
import sys
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
sys.path.insert(0, '.')

# Optional: import torch only for DataLoader utilities
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    Dataset = object
    DataLoader = None

# -------------------------
# Dataset
# -------------------------
class JsonDescriptionsDataset(Dataset):
    """
    PyTorch-style dataset that loads JSON of the form:
    {
      "id1": { "name": "...", "category": "...", "general": [...], "distinguishing features": [...] },
      "id2": ...
    }
    Each item returned is a dict: {"id": id, "text": concatenated_text, "meta": {...}}
    The text is built from `general` + `distinguishing features` (or other coarse keys).
    """
    def __init__(self, json_path: str, general_key: str = "general",
                 coarse_key_candidates: List[str] = None):
        self.json_path = json_path
        if coarse_key_candidates is None:
            coarse_key_candidates = ["coarse", "distinguishing features", "coarse_features", "distinguishing", "distinct features"]
        self.path = Path(json_path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSON input not found: {json_path}")
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if "descriptions" in json_path:
            self.ids = list(self.data.keys())
        elif "database" in json_path:
            self.ids = list(self.data["concept_dict"].keys())
        self.general_key = general_key
        self.coarse_key_candidates = coarse_key_candidates

    def __len__(self):
        return len(self.ids)

    def _get_coarse_key(self, obj: Dict[str, Any]) -> Optional[str]:
        for k in self.coarse_key_candidates:
            if k in obj:
                return k
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.ids[idx]
        if "descriptions" in self.json_path:
            obj = self.data[key]
        elif "database" in self.json_path:
            obj = self.data['concept_dict'][key]['info']
        if not isinstance(obj, str):
            general_list = obj.get(self.general_key, [])
            coarse_key = self._get_coarse_key(obj)
            coarse_list = obj.get(coarse_key, []) if coarse_key else []
            general_text = " ".join(general_list) if isinstance(general_list, list) else str(general_list or "")
            coarse_text = " ".join(coarse_list) if isinstance(coarse_list, list) else str(coarse_list or "")
        else:
            general_text = obj
            coarse_text = None
        if coarse_text:
            text = (general_text + " " + coarse_text).strip() if general_text else coarse_text.strip()
        else:
            text = general_text.strip()
        return {"id": key, "text": text}

# def refine_descriptions(json_path, responses):
#     coarse_key_candidates = ["coarse", "distinguishing features", "coarse_features", "distinguishing", "distinct features"]
#     path = Path(json_path)
#     if not path.exists():
#         raise FileNotFoundError(f"JSON input not found: {json_path}")
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if "descriptions" in json_path:
#         ids = list(data.keys())
#     elif "database" in json_path:
#         ids = list(data["concept_dict"].keys())
#     general_key = general_key
#     coarse_key_candidates = coarse_key_candidates
#     for key in ids:

def collate_batch(batch: List[Dict[str, Any]], feat_key=None) -> List[Dict[str, str]]:
    """
    Receive list of items from dataset and return a list of {"id","text"} dicts ready for prompt injection.
    """
    if not feat_key:
        return [{"id": item["id"], "text": item["text"]} for item in batch]
    else:
         return [{"id": item["id"], "text": item[feat_key]} for item in batch]
# -------------------------
# Prompt handling
# -------------------------
def load_prompt_template(prompt_json_path: str) -> Dict[str, Any]:
    p = Path(prompt_json_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt JSON not found: {prompt_json_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(prompt_template: Dict[str, Any], items: List[Dict[str, str]]) -> str:
    """
    The prompt_template should have keys:
      - prefix: str (system + instructions + few-shot examples)
      - suffix: str (task suffix that expects a JSON array)
    We'll insert the JSON array into a placeholder token <<JSON_ARRAY>> in the suffix or template.
    """
    prefix = prompt_template.get("prefix", "")
    suffix = prompt_template.get("suffix", "")
    # Build JSON array of inputs
    json_array = json.dumps([{"id": it["id"], "text": it["text"]} for it in items], ensure_ascii=False)
    if "<<JSON_ARRAY>>" in suffix:
        body = suffix.replace("<<JSON_ARRAY>>", json_array)
        prompt = prefix + "\n\n" + body
    else:
        # fallback: append at end
        prompt = prefix + "\n\n" + json_array
    return prompt

# -------------------------
# Parsing + validation
# -------------------------
def validate_output_item(obj: Dict[str, Any]) -> bool:
    required = ["id","has_state","state_conf","has_background","background_conf","has_location","location_conf","length_tokens","length_words","discriminative_score"]
    for k in required:
        if k not in obj:
            return False
    if not isinstance(obj["id"], str):
        return False
    for b in ["has_state","has_background","has_location"]:
        if not isinstance(obj[b], bool):
            return False
    for c in ["state_conf","background_conf","location_conf","discriminative_score"]:
        if not (isinstance(obj[c], float) or isinstance(obj[c], int)):
            return False
        if not (0.0 <= float(obj[c]) <= 1.0):
            return False
    for ln in ["length_tokens","length_words"]:
        if not isinstance(obj[ln], int):
            return False
    return True


def setup_model(model_path: str):
    # Try using HuggingFace transformers as a common fallback
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # heuristics: for very large models you may want device_map="auto" and torch_dtype="auto"
        print(f"[setup_model] Loading HF model/tokenizer for: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # attempt to load model with device_map=auto if available
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        except Exception:
            # fallback to default load (may be large)
            model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        # No transformers installed or model loading failed. Provide clear instructions for Qwen SDK.
        msg = (
            "Could not load a HuggingFace-style model (transformers not available or failed).\n"
            "If you are using a Qwen SDK, modify setup_model() to initialize the Qwen client and model.\n"
            "For example (pseudo):\n"
            "    from qwen import QwenClient\n"
            "    client = QwenClient(api_key=...)\n"
            "    model = client.load_model('Qwen/Qwen3-8B')\n"
            "    return model, client\n\n"
            f"Original error: {e}"
        )
        raise NotImplementedError(msg)

def _build_prompt_from_template(prompt_template: Dict[str, str], items: List[Dict[str,str]], feat_key=None) -> str:
    prefix = prompt_template.get("prefix", "")
    suffix = prompt_template.get("suffix", "")
    if not feat_key:
        json_array = json.dumps([{"id": it["id"], "text": it["text"]} for it in items], ensure_ascii=False)
    else:
        json_array = json.dumps([{"id": it["id"], "text": it[feat_key]} for it in items], ensure_ascii=False)
    if "<<JSON_ARRAY>>" in suffix:
        body = suffix.replace("<<JSON_ARRAY>>", json_array)
        prompt = prefix + "\n\n" + body
    else:
        prompt = prefix + "\n\n" + json_array
    return prompt

def _generate_with_hf(model, tokenizer, prompt: str, temperature: float = 0.0, max_new_tokens: int = 1024) -> str:
    """
    Simple HF generate wrapper. Decoding is deterministic by setting do_sample=False (temperature ignored).
    If you want sampling-based deterministic generation, set temperature=0 and do_sample=False gives greedy.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content




def infer_batch(model, processor, descriptions: List[Dict[str,str]], prompt_prefix, prompt_suffix,
                temperature: float = 0.0, max_new_tokens: int = 1024, retry_on_fail: bool = True, refine=False, feat_key=None) -> List[Dict]:
    """
    descriptions: list of {"id": str, "text": str}
    prompt_template_path: path to JSON prompt template (prefix/suffix with <<JSON_ARRAY>>)
    Returns: list of parsed JSON objects as returned by the model (one per input item)
    """
    # load prompt
    prompt_template = {"prefix":prompt_prefix, "suffx":prompt_suffix}
    prompt = _build_prompt_from_template(prompt_template, descriptions, feat_key=feat_key)
    raw = None
    raw = _generate_with_hf(model, processor, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    import re
    match = re.search(r'(\[\\s*\\{.*\\}\\s*\\])', raw, flags=re.S)
    parsed = None
    try:
        if match:
            json_text = match.group(1)
        else:
            json_text = raw.strip()
        parsed = json.loads(json_text)
    except Exception as e:
        if retry_on_fail:
            clar_prompt = prompt + "\\n\\nIMPORTANT: Output only the JSON array, nothing else."
            raw2 = _generate_with_hf(model, processor, clar_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            try:
                parsed = json.loads(raw2)
            except Exception as e2:
                raise RuntimeError(f"Failed to parse model output as JSON. Raw1:\\n{raw}\\n\\nRaw2:\\n{raw2}") from e2
        else:
            raise RuntimeError(f"Failed to parse model output as JSON. Raw:\\n{raw}") from e

    # Basic validation: ensure list and same length as input
    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list.")
    if len(parsed) != len(descriptions):
        # allow but warn
        print(f"[warn] parsed length {len(parsed)} != input length {len(descriptions)}")
    elif refine:
        for idx in range(len(parsed)):
            parsed[idx]['length']=len(parsed[idx]['text'].split(' '))
    else:
        for idx in range(len(descriptions)):
            parsed[idx]['length']=len(descriptions[idx]['text'].split(' '))
            parsed[idx]['text']=descriptions[idx]['text']
    return parsed

def aggregate_stats(batch_out):
    has_state = [item['has_state'] for item in batch_out]
    has_location = [item['has_location'] for item in batch_out]
    lengths = [item['length'] for item in batch_out]
    mean_state = st.mean(has_state)
    mean_location = st.mean(has_location)
    mean_length = st.mean(lengths)
    return {"mean_state":mean_state, "mean_location":mean_location, "mean_length":mean_length}

def refine(args, model, tokenizer, prefix, suffix, batch_out, feat_key, refined):
    ds = JsonDescriptionsDataset(args.input)
    def batch_iterator(batch_out, batch_size=6):
        batch = []
        for item in batch_out:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    for batch in batch_iterator(batch_out):
        items = collate_batch(batch, feat_key=feat_key)
        parsed = infer_batch(model, tokenizer, batch, prefix, suffix, refine=True, feat_key=feat_key)
        for item in parsed:
            key = item['id']
            if key not in refined:
                refined[key] = ds.data[key]
            if 'text' in item:
                if feat_key == 'gen_text':
                    refined[key]["general"] = [item['text']]
                else:
                    refined[key]["distinguishing features"] = [item['text']]
        print(f"batch num {len(refined)} is processed")
    return refined

def evaluate(args, model, tokenizer, prefix, suffix, outpath):
    ds = JsonDescriptionsDataset(args.input)
    if torch is None:
        # create a simple iterator fallback if PyTorch is unavailable
        def batches():
            batch = []
            for i in range(len(ds)):
                batch.append(ds[i])
                if len(batch) >= args.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b)
        def batches():
            for b in dl:
                yield b
    total = 0
    start_time = time.time()
    batch_out = []
    for batch_idx, batch in enumerate(batches()):
        items = collate_batch(batch)
        try:
            parsed = infer_batch(model, tokenizer, batch, prefix, suffix)
            batch_out.extend(parsed)
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx} failed to parse: {e}", file=sys.stderr)
            # write raw output for debugging
            debug_file = out_path.parent / f"debug_batch_{batch_idx}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(parsed)
            # skip this batch or raise, depending on fail behavior
            if args.fail_on_error:
                raise
            else:
                print(f"[WARN] Skipping batch {batch_idx} (writing raw to {debug_file})", file=sys.stderr)
                continue
                total += 1

        # optional sleep to respect rate limits
        if args.sleep_between_batches > 0:
            time.sleep(args.sleep_between_batches)

        print(f"[INFO] Completed batch {batch_idx+1}, items_processed={total}")

    elapsed = time.time() - start_time
    stats = aggregate_stats(batch_out)
    overall = {"stats":stats, "response":batch_out}
    with open(outpath, "w", encoding="utf-8") as fout:
        json.dump(overall, fout, indent=2)
    
    print(f"[DONE] Evaluated {total} items in {elapsed:.1f}s. Results saved to {outpath}")

def main(args):
    model, tokenizer = setup_model("Qwen/Qwen3-8B")   
    parsed_input = args.input.split('/')
    data_name, category_name, seed = parsed_input[1], parsed_input[2], parsed_input[3]
    if data_name == 'PerVA':
        filename = data_name + '_' + f'{category_name}_' + seed + '_' + '_'.join(parsed_input[-1].split('_')[1:])
    else:
        filename = data_name + '_'+ seed + '_' + '_'.join(parsed_input[-1].split('_')[1:])
    out_path = Path(args.out) / filename
    if args.refine == 'none':
        if 'descriptions' in args.input or 'database' in args.input:
            from src.eval_utils.qwen_prompt import prefix, suffix
            # if not os.path.exists(args.input):
            evaluate(args, model, tokenizer, prefix, suffix, out_path)
    else:
        from src.eval_utils.qwen_prompt import prefix_state, prefix_location, prefix_location_and_state, suffix
        with open(out_path, 'r') as f:
            batch_out = json.load(f)
        if args.refine == 'state':
            prefix = prefix_state
        elif args.refine == 'location':
            prefix = prefix_location
        elif args.refine == 'location_and_state':
            prefix = prefix_location_and_state
        for i, response in enumerate(batch_out['response']):
            gen_text = response['text'].split('.')[0]
            dist_text = '.'.join(response['text'].split('.')[1:])
            temp_dict = {"gen_text":gen_text, "dist_text":dist_text}
            for key, val in temp_dict.items():
                batch_out['response'][i][key] = val
        # batch_out['response'] = [response['text'].split('.')[1] for response in batch_out['response']]
        parsed={}
        for feat_key in ["gen_text", "dist_text"]:
            parsed = refine(args, model, tokenizer, prefix, suffix, batch_out['response'], feat_key, parsed)
        out_path = args.input.split('.')[0] + f'_{args.refine}_refined.json'
        with open(out_path, 'w') as f:
            json.dump(parsed, f, indent=2)
        print(f'Refined version saved in {out_path}')

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate descriptions using Qwen-8B evaluator prompt")
    p.add_argument("--input", required=True, help="Path to input JSON dataset")
    p.add_argument("--refine", type=str)
    p.add_argument("--out", required=True, help="Output JSONL file")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (number of descriptions per model call)")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens for Qwen call")
    p.add_argument("--sleep-between-batches", type=float, default=0.0, help="Optional sleep between batches (seconds)")
    p.add_argument("--fail-on-error", action="store_true", help="Raise exception on parse/validation error")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # evaluate(args)
    main(args)

