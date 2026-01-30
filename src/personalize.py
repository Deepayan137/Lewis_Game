#!/usr/bin/env python3
"""
personalize.py

Personalized identification task: Given a query image and multiple reference
descriptions retrieved via CLIP, identify which reference matches the query.

Usage:
    python personalize.py --data_name PerVA --model_type original_7b --category all
"""

from __future__ import annotations

import json
import logging
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference_utils.common import (
    set_seed,
    get_model_config,
    add_common_args,
    save_results,
    get_category_for_concept,
    get_device,
    clear_cuda_cache,
    PERVA_CATEGORY_MAP,
)
from inference_utils.dataset import SimpleImageDataset, create_data_loader, DictListDataset, dict_collate_fn
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.retriever import SimpleClipRetriever
from inference_utils.cleanup import extract_reasoning_answer_term

LOG = logging.getLogger(__name__)


# ============================================================================
# Prompt Generation
# ============================================================================

def get_prompt(descriptions: List[str], category: str, names: List[str], query_desc: str = "") -> str:
    """
    Build the prompt for personalized identification.

    Args:
        descriptions: List of reference description strings
        category: Object category (e.g., 'toy', 'pet animal')
        names: List of option names (letters A, B, C, ...)
        query_desc: Optional query description (unused in current implementation)

    Returns:
        Formatted prompt string
    """
    n = len(names)
    letters = [string.ascii_uppercase[i] for i in range(n)]

    answer_format = {
        "Reasoning": "<Brief justification>",
        "Answer": f"one of {letters}"
    }

    descriptions_block = json.dumps(descriptions, indent=2, ensure_ascii=False)
    prompt = (
        f"You are provided with a query image containing a {category} "
        f"along with the name and detailed distinguishing features of several other {category}s.\n\n"
        "Below are the name and their descriptions:\n"
        f"{descriptions_block}\n\n"
        "Your Task:\n"
        f"1. Generate an attribute-focused description of the {category} in the query image. "
        "Focus on its distinguishing features rather than superficial details such as background, pose, lighting, clothes or accessories.\n"
        f"2. Compare your generated description of the query image with the provided descriptions of the other {category}s.\n"
        f"3. Identify the name of the {category} in the query image from the best match.\n\n"
        "Output Requirements:\n"
        f"- Your response MUST be a valid JSON exactly matching the format:\n{json.dumps(answer_format)}\n"
        "- Do not include any extra text, explanations, or formatting outside of the JSON.\n"
    )

    return prompt


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_test_retrieval_items(
    args,
    description_json: Path,
    catalog_json: str,
    category: str,
    retriever: SimpleClipRetriever,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Build a list of items for evaluation using CLIP retrieval.

    Each item contains:
      - 'path': path to query image
      - 'problem': prompt string
      - 'solution': ground-truth object name
      - 'solution_desc': description string for the true object
      - 'ret_path': list of retrieved image paths
      - 'letter2name': mapping from letter options to concept names

    Args:
        args: Parsed arguments
        description_json: Path to descriptions JSON
        catalog_json: Path to catalog JSON
        category: Category to evaluate
        retriever: CLIP retriever instance
        k: Number of references to retrieve

    Returns:
        List of prepared item dicts
    """
    # Check for cached results
    savedir = Path('outputs') / args.data_name / args.category / f'seed_{args.seed}'
    savedir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading dataset from: %s", catalog_json)
    dataset = SimpleImageDataset(
        json_path=catalog_json,
        category=category,
        split="test",
        seed=args.seed,
        data_name=args.data_name
    )

    # Load description dictionary
    with description_json.open("r", encoding="utf-8") as fh:
        desc_lookup: Dict[str, Any] = json.load(fh)

    concept_dict_format = 'concept_dict' in desc_lookup
    if concept_dict_format:
        desc_lookup = desc_lookup['concept_dict']

    items: List[Dict[str, Any]] = []

    for item in tqdm(dataset, desc="Preparing retrieval items"):
        query_path = item["path"]
        query_name = item['name']

        # Retrieve K nearest neighbors
        results = retriever.hybrid_search(
            query_path,
            k=k,
            alpha=0.5,
            normalization='minmax',
            aggregation='weighted_sum',
            image_k=len(desc_lookup),
            text_k=len(desc_lookup)
        )

        descriptions: List[str] = []
        ret_paths: List[str] = []
        names = []
        letters = list(string.ascii_uppercase)[:len(results)]
        letter2name = {}

        for i, r in enumerate(results):
            name = r.get("name")
            letter = letters[i]
            letter2name[letter] = name
            names.append(name)

            # Get description
            lookup_key = f'<{name}>' if concept_dict_format else name
            if lookup_key in desc_lookup:
                if concept_dict_format:
                    desc_info = desc_lookup[lookup_key]['info']['general']
                    descriptions.append(f"Name: {letter}, Info: {desc_info}")
                else:
                    descriptions.append(f"Name: {letter}, Info: {desc_lookup[lookup_key]}")

            if "path" in r:
                ret_paths.append(r["path"])

        if not descriptions:
            descriptions = ["No description available."]

        # Get category for prompt
        item_category = get_category_for_concept(query_name, args.data_name)
        if item_category == query_name and args.data_name == 'PerVA':
            # Try path-based lookup for PerVA
            concept_name = query_path.split('/')[-2]
            item_category = PERVA_CATEGORY_MAP.get(concept_name, concept_name)

        prompt = get_prompt(descriptions, item_category, names)

        items.append({
            "path": query_path,
            "problem": prompt,
            "solution": item["name"],
            "solution_desc": desc_lookup.get(item["name"], ""),
            "ret_path": ret_paths,
            "letter2name": letter2name
        })

    return items


# ============================================================================
# Inference Loop
# ============================================================================

def run_inference_loop(
    model,
    processor,
    items: Iterable[Dict[str, Any]],
    temperature: float = 1e-6,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: torch.device = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run model on the prepared items and return results and metrics.

    Args:
        model: The loaded model
        processor: The model processor
        items: Iterable of prepared item dicts
        temperature: Generation temperature
        batch_size: Batch size for inference
        max_new_tokens: Maximum tokens to generate
        device: Torch device

    Returns:
        Tuple of (results_list, metrics_dict)
    """
    dataset = DictListDataset(list(items))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=dict_collate_fn, num_workers=4
    )

    results: List[Dict[str, Any]] = []
    correct_count = 0
    total_count = 0

    if device is None:
        device = get_device()

    for batch in tqdm(loader, desc="Running inference"):
        # Load images
        images = []
        for it in batch:
            if "image" in it and it["image"] is not None:
                images.append(it["image"])
            else:
                images.append(Image.open(it["path"]).convert("RGB"))

        problems = [it["problem"] for it in batch]
        gt_names = [it["solution"] for it in batch]
        sol_descs = [it.get("solution_desc", "") for it in batch]
        paths = [it.get("path") for it in batch]
        ret_paths = [it.get("ret_path", []) for it in batch]
        letter2names = [it.get("letter2name", {}) for it in batch]

        try:
            responses = speaker_describes_batch(
                model, processor, problems, images,
                temperature=temperature, max_new_tokens=max_new_tokens
            )
        except Exception:
            LOG.exception("Failed generating responses for batch; skipping.")
            for path, gt, sol_desc, rp in zip(paths, gt_names, sol_descs, ret_paths):
                results.append({
                    "image_path": path,
                    "problem": None,
                    "solution": gt,
                    "solution_desc": sol_desc,
                    "ret_paths": rp,
                    "response": "",
                    "pred_name": "",
                    "correct": False,
                })
                total_count += 1
            continue

        # Normalize responses
        if isinstance(responses, str):
            responses = [responses]

        # Extract predictions
        pred_names = []
        for resp in responses:
            try:
                if isinstance(resp, list):
                    resp = resp[0]
                term = extract_reasoning_answer_term(resp, "Answer")
                pred_names.append(term.strip() if term else "")
            except Exception:
                LOG.exception("Failed to extract answer; using empty string")
                pred_names.append("")

        # Accumulate results
        for path, gt, sol_desc, rp, resp, pred, lt2nm in zip(
            paths, gt_names, sol_descs, ret_paths, responses, pred_names, letter2names
        ):
            # Map letter prediction to concept name
            pred_name = lt2nm.get(pred, pred)
            is_correct = pred_name.lower() == gt.lower()

            if is_correct:
                correct_count += 1
            total_count += 1

            results.append({
                "image_path": path,
                "problem": resp if isinstance(resp, str) else resp[0] if resp else None,
                "solution": gt,
                "solution_desc": sol_desc,
                "ret_paths": rp,
                "response": resp[0] if isinstance(resp, list) else resp,
                "pred_name": pred_name,
                "correct": is_correct,
            })

        clear_cuda_cache()

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    metrics = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
    }

    return results, metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Personalized identification evaluation")

    add_common_args(parser)

    # Task-specific args
    parser.add_argument("--k_retrieval", type=int, default=3,
                        help="Number of references to retrieve")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()

    set_seed(args.seed)
    LOG.info("Args: %s", args)

    # Set up paths
    manifests_dir = Path("manifests") / args.data_name
    description_json = (
        Path("outputs") / args.data_name / args.category /
        f"seed_{args.seed}" / f"descriptions_{args.db_type}.json"
    )
    catalog_json = str(manifests_dir / args.catalog_file)

    if not description_json.exists():
        LOG.error("Descriptions file not found at %s", description_json)
        raise FileNotFoundError(f"Descriptions file not found: {description_json}")

    # Initialize retriever
    retriever = SimpleClipRetriever(
        dataset=args.data_name,
        category=args.category,
        json_path=catalog_json,
        create_index=True,
        seed=args.seed,
        db_type=args.db_type
    )
    LOG.info("Retriever created")

    # Prepare test items
    items = prepare_test_retrieval_items(
        args,
        description_json,
        catalog_json,
        args.category,
        retriever,
        k=args.k_retrieval
    )
    LOG.info("Prepared %d test items", len(items))

    # Filter by concept if specified
    if args.concept_name:
        items = [item for item in items if item.get("solution") == args.concept_name]
        LOG.info("Filtered to %d items for concept '%s'", len(items), args.concept_name)

    # Load model
    try:
        model_config = get_model_config(
            args.model_type, dataset=args.data_name, seed=args.seed
        )
    except ValueError:
        LOG.warning("Model type '%s' not in config, using as direct path", args.model_type)
        model_config = {
            'path': args.model_type,
            'use_peft': 'lora' in args.model_type.lower(),
        }

    model_path = model_config['path']
    use_peft = model_config['use_peft']

    LOG.info("Loading model from %s", model_path)
    model, processor = setup_model(model_path, use_peft=use_peft)

    # Run inference
    results, metrics = run_inference_loop(
        model,
        processor,
        items,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    outdir = Path('results') / args.data_name / args.category
    if args.concept_name:
        outdir = outdir / args.concept_name
    outdir = outdir / f'seed_{args.seed}'
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"results_model_{args.model_type}_db_{args.db_type}_k_{args.k_retrieval}.json"
    save_results(results, metrics, vars(args), outpath)

    LOG.info("Accuracy: %.4f (%d/%d)", metrics['accuracy'], metrics['correct'], metrics['total'])


if __name__ == "__main__":
    main()
