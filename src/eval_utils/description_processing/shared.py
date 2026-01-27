"""
Shared utilities for description processing.

This module contains common functionality used by both the evaluator and refiner:
- Dataset classes for loading description data
- Model setup and inference functions
- Batch processing utilities
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

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


def collate_batch(batch: List[Dict[str, Any]], feat_key=None) -> List[Dict[str, str]]:
    """
    Receive list of items from dataset and return a list of {"id","text"} dicts ready for prompt injection.
    """
    if not feat_key:
        return [{"id": item["id"], "text": item["text"]} for item in batch]
    else:
         return [{"id": item["id"], "text": item[feat_key]} for item in batch]


# -------------------------
# Model Setup
# -------------------------
def setup_model(model_path: str):
    """
    Load the model and tokenizer for inference.

    Args:
        model_path: Path or identifier for the model (e.g., "Qwen/Qwen3-8B")

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        NotImplementedError: If the model cannot be loaded
    """
    # Try using HuggingFace transformers as a common fallback
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # heuristics: for very large models you may want device_map="auto" and torch_dtype="auto"
        print(f"[setup_model] Loading HF model/tokenizer for: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # attempt to load model with device_map=auto if available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
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


# -------------------------
# Prompt Building
# -------------------------
def _build_prompt_from_template(prompt_template: Dict[str, str], items: List[Dict[str, str]], feat_key=None) -> str:
    """
    Build a prompt from a template by injecting JSON array of items.

    Args:
        prompt_template: Dict with "prefix" and "suffix" keys
        items: List of dicts with "id" and "text" keys
        feat_key: Optional key to use instead of "text"

    Returns:
        Complete prompt string
    """
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


# -------------------------
# Model Inference
# -------------------------
def _generate_with_hf(model, tokenizer, prompt: str, temperature: float = 0.0, max_new_tokens: int = 1024) -> str:
    """
    Simple HF generate wrapper. Decoding is deterministic by setting do_sample=False (temperature ignored).
    If you want sampling-based deterministic generation, set temperature=0 and do_sample=False gives greedy.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: The prompt string
        temperature: Temperature for generation (not used with greedy decoding)
        max_new_tokens: Maximum number of new tokens to generate

    Returns:
        Generated text (excluding thinking content if present)
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
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


def infer_batch(model, processor, descriptions: List[Dict[str, str]], prompt_prefix, prompt_suffix,
                temperature: float = 0.0, max_new_tokens: int = 1024, retry_on_fail: bool = True,
                refine=False, feat_key=None) -> List[Dict]:
    """
    Perform inference on a batch of descriptions.

    Args:
        model: The loaded model
        processor: The tokenizer/processor
        descriptions: List of {"id": str, "text": str} dicts
        prompt_prefix: The prefix part of the prompt template
        prompt_suffix: The suffix part of the prompt template
        temperature: Temperature for generation
        max_new_tokens: Maximum new tokens to generate
        retry_on_fail: Whether to retry once on parsing failure
        refine: Whether this is a refinement task (affects output processing)
        feat_key: Optional feature key to use instead of "text"

    Returns:
        List of parsed JSON objects from the model

    Raises:
        RuntimeError: If JSON parsing fails
        ValueError: If output format is invalid
    """
    # load prompt
    prompt_template = {"prefix": prompt_prefix, "suffix": prompt_suffix}
    prompt = _build_prompt_from_template(prompt_template, descriptions, feat_key=feat_key)
    raw = None
    raw = _generate_with_hf(model, processor, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    match = re.search(r'(\[\s*\{.*\}\s*\])', raw, flags=re.S)
    parsed = None
    try:
        if match:
            json_text = match.group(1)
        else:
            json_text = raw.strip()
        parsed = json.loads(json_text)
    except Exception as e:
        if retry_on_fail:
            clar_prompt = prompt + "\n\nIMPORTANT: Output only the JSON array, nothing else."
            raw2 = _generate_with_hf(model, processor, clar_prompt, temperature=temperature, max_new_tokens=max_new_tokens)
            try:
                parsed = json.loads(raw2)
            except Exception as e2:
                raise RuntimeError(f"Failed to parse model output as JSON. Raw1:\n{raw}\n\nRaw2:\n{raw2}") from e2
        else:
            raise RuntimeError(f"Failed to parse model output as JSON. Raw:\n{raw}") from e

    # Basic validation: ensure list and same length as input
    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list.")
    if len(parsed) != len(descriptions):
        # allow but warn
        print(f"[warn] parsed length {len(parsed)} != input length {len(descriptions)}")

    # Add length information to each item
    if refine:
        for idx in range(len(parsed)):
            parsed[idx]['length'] = len(parsed[idx]['text'].split(' '))
    else:
        for idx in range(len(descriptions)):
            parsed[idx]['length'] = len(descriptions[idx]['text'].split(' '))
            parsed[idx]['text'] = descriptions[idx]['text']

    return parsed
