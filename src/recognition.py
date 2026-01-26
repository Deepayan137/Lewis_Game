#!/usr/bin/env python3
"""
recognition.py

Binary recognition task: Given a query image (Image 1) and a reference image
(Image 2) with its description, determine if they show the same object.

Usage:
    python recognition.py --data_name PerVA --model_type original_7b --category all
"""

from __future__ import annotations

import json
import logging
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from inference_utils.common import (
    set_seed,
    get_model_config,
    add_common_args,
    save_results,
    load_database,
    get_database_path,
    get_device,
    clear_cuda_cache,
)
from inference_utils.dataset import DictListDataset, dict_collate_fn
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.cleanup import extract_reasoning_answer_term

LOG = logging.getLogger(__name__)


# ============================================================================
# Prompt Generation
# ============================================================================

def get_prompt(description: Dict[str, Any], test_question: str, vqa: bool = False) -> str:
    """
    Build the prompt for binary recognition or VQA task.

    Args:
        description: Dict with reference info (category, general, distinct features)
        test_question: The question to answer
        vqa: Whether this is a VQA task (changes answer format)

    Returns:
        Formatted prompt string
    """
    if vqa:
        answer_format = {
            "Reasoning": "<Brief comparison based on key attributes>",
            "Answer": "<A or B>"
        }
    else:
        answer_format = {
            "Reasoning": "<Brief comparison based on key attributes>",
            "Answer": "<yes or no>"
        }

    test_question = test_question.replace('the image', 'the first image')

    prompt = (
        f"You are a helpful AI agent specializing in image analysis and object recognition\n\n"
        f"You are given two images, additionally, the name and a textual description of the subject "
        f"in the second image is also provided below:\n\n"
        f"{json.dumps(description, indent=2)}\n"
        f"Your Task:\n"
        f"- Compare the first image with the second image and answer the following question: "
        f"{test_question}\n"
        f"- **Ignore superficial details** such as clothing, accessories, pose variations, or "
        f"surrounding elements (e.g., people in the background).\n"
        f"- Focus only on non-variant/permanent features such as color, shape, pattern, text for "
        f"objects/buildings and facial features for people.\n"
        f"- If you are uncertain then you can refer the textual description of the second image "
        f"to make a more informed decision.\n"
        f"**Output (JSON only):**\n{json.dumps(answer_format, indent=2)}"
    )

    return prompt


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_test_recognition_items(
    retrieval_json_path: str,
    database: Dict[str, Any],
    data_name: str = "PerVA",
) -> List[Dict[str, Any]]:
    """
    Prepare items for recognition evaluation.

    Each item contains a query-reference pair with a Yes/No ground truth.

    Args:
        retrieval_json_path: Path to retrieval JSON
        database: Database with concept descriptions
        data_name: Dataset name

    Returns:
        List of prepared item dicts
    """
    with open(retrieval_json_path, "r") as f:
        retrieval_data = json.load(f)

    items: List[Dict[str, Any]] = []

    for entry in tqdm(retrieval_data, desc="Preparing recognition items"):
        query_path = entry.get("query_path") or entry.get("query")
        ret_paths = entry.get("retrieved_paths") or entry.get("ret_paths", [])
        label = entry.get("label", 0)
        category = entry.get("category", "object")

        if not query_path or not ret_paths:
            continue

        # Create positive pair (query matches correct reference)
        correct_ref_path = ret_paths[label] if label < len(ret_paths) else ret_paths[0]

        # Get reference info from database
        concept_key = database.get("path_to_concept", {}).get(correct_ref_path)
        if concept_key and concept_key in database.get("concept_dict", {}):
            ref_info = database["concept_dict"][concept_key].get("info", {})
        else:
            ref_info = {
                "category": category,
                "general": ["Unknown"],
                "distinct features": ["Unknown"]
            }

        # Positive sample
        question = f"Is the {category} in Image 1 the same as the {category} in Image 2?"
        items.append({
            "query_path": query_path,
            "ret_path": correct_ref_path,
            "ret_info": ref_info,
            "question": question,
            "solution": "Yes",
        })

        # Optionally add negative samples (different reference)
        for i, rp in enumerate(ret_paths):
            if i == label:
                continue
            # Get info for this (incorrect) reference
            neg_concept_key = database.get("path_to_concept", {}).get(rp)
            if neg_concept_key and neg_concept_key in database.get("concept_dict", {}):
                neg_ref_info = database["concept_dict"][neg_concept_key].get("info", {})
            else:
                neg_ref_info = ref_info  # Fallback

            items.append({
                "query_path": query_path,
                "ret_path": rp,
                "ret_info": neg_ref_info,
                "question": question,
                "solution": "No",
            })
            break  # Only add one negative per query

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
    Run recognition inference over items.

    Args:
        model: The loaded model
        processor: The model processor
        items: Iterable of prepared item dicts
        temperature: Generation temperature
        batch_size: Batch size
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
    correct_yes, correct_no = 0, 0
    total_yes, total_no = 0, 0

    if device is None:
        device = get_device()

    for batch in tqdm(loader, desc="Running recognition inference"):
        # Load images
        images, ret_images = [], []
        for it in batch:
            try:
                images.append(Image.open(it["query_path"]).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
            try:
                ret_images.append(Image.open(it["ret_path"]).convert("RGB"))
            except Exception:
                ret_images.append(Image.new("RGB", (224, 224)))

        problems = [get_prompt(it['ret_info'], it["question"]) for it in batch]
        solutions = [it["solution"] for it in batch]
        query_paths = [it["query_path"] for it in batch]
        ref_paths = [it["ret_path"] for it in batch]

        try:
            responses = speaker_describes_batch(
                model, processor, problems, images, ret_images,
                temperature=temperature, max_new_tokens=max_new_tokens
            )
        except Exception:
            LOG.exception("Failed generating responses for batch; skipping.")
            for qp, sol, rp in zip(query_paths, solutions, ref_paths):
                results.append({
                    "query_path": qp,
                    "ref_path": rp,
                    "problem": None,
                    "solution": sol,
                    "response": "",
                    "pred": "",
                    "correct": False,
                })
                if sol.lower() == "yes":
                    total_yes += 1
                else:
                    total_no += 1
            continue

        # Normalize responses
        if isinstance(responses, str):
            responses = [responses]

        # Extract predictions
        predictions = []
        for resp in responses:
            try:
                if isinstance(resp, list):
                    resp = resp[0]
                term = extract_reasoning_answer_term(resp, "Answer")
                predictions.append(term.strip() if term else "")
            except Exception:
                predictions.append("")

        # Accumulate results
        for qp, sol, rp, prob, resp, pred in zip(
            query_paths, solutions, ref_paths, problems, responses, predictions
        ):
            pred_lower = pred.lower().strip()
            sol_lower = sol.lower().strip()

            is_correct = pred_lower == sol_lower

            if sol_lower == "yes":
                total_yes += 1
                if is_correct:
                    correct_yes += 1
            else:
                total_no += 1
                if is_correct:
                    correct_no += 1

            results.append({
                "query_path": qp,
                "ref_path": rp,
                "problem": prob,
                "solution": sol,
                "response": resp[0] if isinstance(resp, list) else resp,
                "pred": pred,
                "correct": is_correct,
            })

        clear_cuda_cache()

    total = total_yes + total_no
    correct = correct_yes + correct_no
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "accuracy_yes": correct_yes / total_yes if total_yes > 0 else 0.0,
        "accuracy_no": correct_no / total_no if total_no > 0 else 0.0,
        "total_yes": total_yes,
        "total_no": total_no,
    }

    return results, metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Binary recognition evaluation")

    add_common_args(parser)

    # Task-specific args
    parser.add_argument("--retrieval_json", type=str, default=None,
                        help="Path to retrieval JSON. If not set, auto-generated from args.")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()

    set_seed(args.seed)
    LOG.info("Args: %s", args)

    # Load database
    database_path = get_database_path(
        args.data_name, args.category, args.seed, args.db_type
    )
    LOG.info("Loading database from %s", database_path)
    database = load_database(database_path)

    # Get model configuration
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

    # Determine retrieval JSON path
    if args.retrieval_json:
        retrieval_json_path = args.retrieval_json
    else:
        retrieval_json_path = (
            Path("outputs") / args.data_name / args.category /
            f"seed_{args.seed}" / "retrieval_top3.json"
        )

    LOG.info("Loading retrieval data from %s", retrieval_json_path)

    # Prepare items
    items = prepare_test_recognition_items(
        retrieval_json_path=str(retrieval_json_path),
        database=database,
        data_name=args.data_name,
    )
    LOG.info("Prepared %d recognition items", len(items))

    # Run inference
    results, metrics = run_inference_loop(
        model, processor, items,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    outdir = Path(args.output_dir) / args.data_name / args.category
    if args.concept_name:
        outdir = outdir / args.concept_name
    outdir = outdir / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"recognition_model_{args.model_type}_db_{args.db_type}.json"
    save_results(results, metrics, vars(args), outpath)

    LOG.info("Overall Accuracy: %.4f (%d/%d)", metrics['accuracy'], metrics['correct'], metrics['total'])
    LOG.info("Accuracy (Yes): %.4f, Accuracy (No): %.4f", metrics['accuracy_yes'], metrics['accuracy_no'])


if __name__ == "__main__":
    main()
