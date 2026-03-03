#!/usr/bin/env python3
"""
personalize_skip_retrieval.py

Ablation: skip the retriever entirely and directly give the listener
the ground-truth description of a candidate concept.

For every (query image of concept Q, description of concept C) pair the
listener sees:
  - the query image
  - the personalised description of concept C (from the database)

and answers yes / no: does this image match the description?

  solution = 'yes'  if Q == C   (positive pair)
  solution = 'no'   if Q != C   (negative pair)

This enables full P / R / F1 computation per concept and tests whether
the speaker's descriptions are discriminative enough for the zero-shot
listener to correctly accept positives and reject negatives.

The script is designed to run as a SLURM array job where
SLURM_ARRAY_TASK_ID selects the concept_name (inner/reference concept C).
Each array job processes all query images paired with one reference
description, keeping jobs balanced and independent.

Usage (single concept):
    python src/personalize_skip_retrieval.py \\
        --data_name PerVA --model_type original_7b \\
        --category all --concept_name bo

Usage (SLURM array — handled externally via --concept_name):
    sbatch --array=0-N scripts/run_skip_ret.sh
"""

from __future__ import annotations

import json
import logging
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
)
from inference_utils.dataset import SimpleImageDataset, DictListDataset, dict_collate_fn
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.cleanup import extract_reasoning_answer_term

LOG = logging.getLogger(__name__)


# ============================================================================
# Prompt Generation
# ============================================================================

def get_skip_ret_prompt(descriptions: List[str], category: str) -> str:
    """
    Build the yes/no matching prompt for the skip-retrieval ablation.

    Args:
        descriptions: Single-element list containing "Name: <name>, Info: <desc>"
        category: Object category (e.g., 'toy', 'pet animal')

    Returns:
        Formatted prompt string
    """
    answer_format = {
        "Reasoning": "<Brief justification>",
        "Answer": "yes or no"
    }

    descriptions_block = json.dumps(descriptions, indent=2, ensure_ascii=False)
    prompt = (
        f"You are provided with a query image containing a {category} "
        f"along with the name and detailed distinguishing features of another {category}.\n\n"
        "Below is its name and its description:\n"
        f"{descriptions_block}\n\n"
        "Your Task:\n"
        f"1. Generate an attribute-focused description of the {category} in the query image. "
        "Focus on its distinguishing features rather than superficial details such as "
        "background, pose, lighting, clothes or accessories.\n"
        f"2. Compare your generated description of the query image with the provided "
        f"description of the other {category}.\n"
        f"3. Does the description of the query image match the provided description? "
        "Answer yes or no.\n\n"
        "Output Requirements:\n"
        f"- Your response MUST be a valid JSON exactly matching the format:\n"
        f"{json.dumps(answer_format)}\n"
        "- Do not include any extra text, explanations, or formatting outside of the JSON.\n"
    )

    return prompt


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_test_items(
    args,
    description_json: Path,
    catalog_json: str,
    category: str,
) -> List[Dict[str, Any]]:
    """
    Build evaluation items using a double loop:
      outer: every query image in the test split
      inner: every concept C in the description database

    Each item contains:
      - 'query_path'   : path to query image
      - 'name'         : query concept Q
      - 'concept_name' : reference concept C  (used for SLURM array filtering)
      - 'problem'      : prompt (query image + description of C)
      - 'solution'     : 'yes' if Q == C else 'no'

    Args:
        args            : parsed arguments
        description_json: path to speaker-generated descriptions JSON
        catalog_json    : path to catalog JSON
        category        : category to evaluate

    Returns:
        List of item dicts
    """
    LOG.info("Loading dataset from: %s", catalog_json)
    dataset = SimpleImageDataset(
        json_path=catalog_json,
        category=category,
        split="test",
        seed=args.seed,
        data_name=args.data_name
    )

    # Load description dictionary (all concepts)
    with description_json.open("r", encoding="utf-8") as fh:
        desc_lookup: Dict[str, Any] = json.load(fh)

    concept_dict_format = 'concept_dict' in desc_lookup
    if concept_dict_format:
        desc_lookup = desc_lookup['concept_dict']

    # Build lookup: concept_name -> description string
    desc_by_concept: Dict[str, str] = {}
    for key, value in desc_lookup.items():
        # strip angle brackets if present: '<bo>' -> 'bo'
        concept_name = key.strip('<>') if key.startswith('<') else key
        if concept_dict_format:
            desc_by_concept[concept_name] = value['info']['general']
        else:
            desc_by_concept[concept_name] = value

    concept_names = sorted(desc_by_concept.keys())
    LOG.info("Found %d concepts in description database", len(concept_names))

    items: List[Dict[str, Any]] = []

    for sub_item in tqdm(dataset, desc="Preparing items"):
        query_path = sub_item["path"]
        query_name = sub_item["name"]

        # Inner loop: pair this query image with every concept's description
        for concept_name in concept_names:
            description = desc_by_concept[concept_name]
            desc_entry = [f"Name: {concept_name}, Info: {description}"]
            item_category = get_category_for_concept(concept_name, args.data_name)
            prompt = get_skip_ret_prompt(desc_entry, item_category)

            items.append({
                "query_path": query_path,
                "name": query_name,              # outer query concept Q
                "concept_name": concept_name,    # inner reference concept C
                "problem": prompt,
                "solution": "yes" if query_name == concept_name else "no",
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
    Run the zero-shot listener on each (query image, description) pair.

    Tracks yes/no accuracy separately so P / R / F1 can be derived.

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

    for batch in tqdm(loader, desc="Running inference"):
        images = []
        for it in batch:
            try:
                images.append(Image.open(it["query_path"]).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224)))

        problems  = [it["problem"]      for it in batch]
        solutions = [it["solution"]     for it in batch]
        query_paths   = [it["query_path"]   for it in batch]
        names         = [it["name"]         for it in batch]
        concept_names = [it["concept_name"] for it in batch]

        try:
            responses = speaker_describes_batch(
                model, processor, problems, images,
                temperature=temperature, max_new_tokens=max_new_tokens
            )
        except Exception:
            LOG.exception("Failed generating responses for batch; skipping.")
            for qp, name, cname, sol in zip(query_paths, names, concept_names, solutions):
                results.append({
                    "query_path": qp,
                    "name": name,
                    "concept_name": cname,
                    "problem": None,
                    "solution": sol,
                    "response": "",
                    "prediction": "",
                    "correct": False,
                })
                if sol == "yes":
                    total_yes += 1
                else:
                    total_no += 1
            continue

        if isinstance(responses, str):
            responses = [responses]

        # Extract yes/no predictions
        predictions = []
        for resp in responses:
            try:
                if isinstance(resp, list):
                    resp = resp[0]
                term = extract_reasoning_answer_term(resp, "Answer")
                predictions.append(term.strip().lower() if term else "")
            except Exception:
                LOG.exception("Failed to extract answer; using empty string")
                predictions.append("")

        for qp, name, cname, problem, sol, resp, pred in zip(
            query_paths, names, concept_names, problems, solutions, responses, predictions
        ):
            sol_lower = sol.lower()
            is_correct = pred == sol_lower

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
                "name": name,
                "concept_name": cname,
                "problem": problem,
                "solution": sol,
                "response": resp[0] if isinstance(resp, list) else resp,
                "prediction": pred,
                "correct": is_correct,
            })

        clear_cuda_cache()

    total   = total_yes + total_no
    correct = correct_yes + correct_no

    # Precision = TP / (TP + FP) = correct_yes / (correct_yes + (total_no - correct_no))
    tp = correct_yes
    fp = total_no - correct_no     # negative pairs predicted "yes"
    fn = total_yes - correct_yes   # positive pairs predicted "no"

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    metrics = {
        "accuracy":     correct / total if total > 0 else 0.0,
        "correct":      correct,
        "total":        total,
        "correct_yes":  correct_yes,
        "correct_no":   correct_no,
        "total_yes":    total_yes,
        "total_no":     total_no,
        "accuracy_yes": correct_yes / total_yes if total_yes > 0 else 0.0,
        "accuracy_no":  correct_no  / total_no  if total_no  > 0 else 0.0,
        "precision":    precision,
        "recall":       recall,
        "f1":           f1,
    }

    return results, metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Skip-retrieval ablation: yes/no match between query image and concept description"
    )
    add_common_args(parser)
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()

    set_seed(args.seed)
    LOG.info("Args: %s", args)

    # Paths
    manifests_dir    = Path("manifests") / args.data_name
    description_json = (
        Path("outputs") / args.data_name / args.category /
        f"seed_{args.seed}" / f"descriptions_{args.db_type}.json"
    )
    catalog_json = str(manifests_dir / args.catalog_file)

    if not description_json.exists():
        LOG.error("Descriptions file not found at %s", description_json)
        raise FileNotFoundError(f"Descriptions file not found: {description_json}")

    # Prepare all (query image, concept description) pairs
    items = prepare_test_items(
        args,
        description_json,
        catalog_json,
        args.category,
    )
    LOG.info("Prepared %d test items", len(items))

    # Filter to a single reference concept if requested (SLURM array hook)
    if args.concept_name:
        items = [item for item in items if item.get("concept_name") == args.concept_name]
        LOG.info(
            "Filtered to %d items for concept_name='%s'",
            len(items), args.concept_name
        )

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
    use_peft   = model_config['use_peft']

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

    # Save results — one file per concept when running as SLURM array
    outdir = Path('results') / args.data_name / args.category
    if args.concept_name:
        outdir = outdir / args.concept_name
    outdir = outdir / f'seed_{args.seed}'
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"results_model_{args.model_type}_db_{args.db_type}_skip_ret.json"
    save_results(results, metrics, vars(args), outpath)

    LOG.info(
        "Accuracy: %.4f (%d/%d) | Yes: %.4f | No: %.4f | P: %.4f | R: %.4f | F1: %.4f",
        metrics['accuracy'], metrics['correct'], metrics['total'],
        metrics['accuracy_yes'], metrics['accuracy_no'],
        metrics['precision'], metrics['recall'], metrics['f1'],
    )


if __name__ == "__main__":
    main()
