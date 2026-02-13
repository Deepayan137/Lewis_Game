#!/usr/bin/env python3
"""
test_hf_recognition.py

Test recognition performance on HuggingFace dataset format.

Usage:
    python tests/test_hf_recognition.py --dataset_path share_data/PerVA_seed_23_K_3_subset_30_sampled_500 \
        --model_type original_7b --batch_size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import DatasetDict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, 'src')

from inference_utils.common import (
    set_seed,
    get_model_config,
    save_results,
    get_device,
    clear_cuda_cache,
)
from inference_utils.dataset import DictListDataset, dict_collate_fn
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.cleanup import extract_reasoning_answer_term

LOG = logging.getLogger(__name__)


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_hf_dataset_items(
    dataset_path: str,
    split: str = "train",
    max_samples: int = None,
) -> List[Dict[str, Any]]:
    """
    Load HuggingFace dataset and prepare items for inference.

    Args:
        dataset_path: Path to HF dataset directory
        split: Dataset split to use (default: "train")
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        List of prepared item dicts
    """
    LOG.info("Loading HF dataset from: %s", dataset_path)
    dataset_dict = DatasetDict.load_from_disk(dataset_path)

    if split not in dataset_dict:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(dataset_dict.keys())}")

    dataset = dataset_dict[split]
    LOG.info("Loaded %d samples from split '%s'", len(dataset), split)

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        LOG.info("Limited to %d samples", max_samples)

    # Convert to list of dicts for easier processing
    items = []
    for idx, sample in enumerate(tqdm(dataset, desc="Preparing items")):
        item = {
            'idx': idx,
            'query_image': sample['query_image'],
            'reference_image': sample['reference_image'],
            'listener_problem': sample['listener_problem'],
            'listener_solution': sample['listener_solution'],
            'ret_paths': sample.get('ret_paths', []),
            'names': sample.get('names', []),
            'category': sample.get('category', 'unknown'),
            'example_idx': sample.get('example_idx', idx),
        }
        items.append(item)

    return items


# ============================================================================
# Inference Loop
# ============================================================================

def run_inference_loop(
    model,
    processor,
    items: List[Dict[str, Any]],
    temperature: float = 1e-6,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: torch.device = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run recognition inference over HF dataset items.

    Args:
        model: The loaded model
        processor: The model processor
        items: List of prepared item dicts
        temperature: Generation temperature
        batch_size: Batch size
        max_new_tokens: Maximum tokens to generate
        device: Torch device

    Returns:
        Tuple of (results_list, metrics_dict)
    """
    dataset = DictListDataset(items)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=dict_collate_fn, num_workers=4
    )

    results: List[Dict[str, Any]] = []
    correct_count = 0
    total_count = 0

    # Track by solution type (for positive/negative accuracy)
    solution_stats = {}

    if device is None:
        device = get_device()

    for batch in tqdm(loader, desc="Running recognition inference"):
        # Extract images (already PIL images from HF dataset)
        query_images = [item['query_image'] for item in batch]
        reference_images = [item['reference_image'] for item in batch]
        # Extract prompts and solutions
        problems = [item['listener_problem'] for item in batch]
        solutions = [item['listener_solution'] for item in batch]

        try:
            responses = speaker_describes_batch(
                model, processor, problems, query_images, reference_images,
                temperature=temperature, max_new_tokens=max_new_tokens
            )
        except Exception:
            LOG.exception("Failed generating responses for batch; skipping.")
            # Record failures
            for idx, (prob, sol) in enumerate(zip(problems, solutions)):
                results.append({
                    "idx": batch[idx]['idx'],
                    "category": batch[idx]['category'],
                    "problem": prob,
                    "solution": sol,
                    "response": "",
                    "pred": "",
                    "correct": False,
                })
                total_count += 1

                # Track by solution type
                sol_key = sol.strip().lower()
                if sol_key not in solution_stats:
                    solution_stats[sol_key] = {'correct': 0, 'total': 0}
                solution_stats[sol_key]['total'] += 1
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
                # Try to extract Answer field from JSON-like response
                term = extract_reasoning_answer_term(resp, "Answer")
                if term:
                    predictions.append(term.strip())
                else:
                    # Fallback: look for yes/no in response
                    resp_lower = resp.lower()
                    if 'yes' in resp_lower:
                        predictions.append('yes')
                    elif 'no' in resp_lower:
                        predictions.append('no')
                    else:
                        predictions.append('')
            except Exception:
                LOG.exception("Error extracting prediction from response")
                predictions.append('')

        # Accumulate results
        for idx, (prob, sol, resp, pred) in enumerate(zip(problems, solutions, responses, predictions)):
            pred_lower = pred.lower().strip()
            sol_lower = sol.lower().strip()

            is_correct = pred_lower == sol_lower

            if is_correct:
                correct_count += 1
            total_count += 1

            # Track by solution type
            sol_key = sol_lower
            if sol_key not in solution_stats:
                solution_stats[sol_key] = {'correct': 0, 'total': 0}
            solution_stats[sol_key]['total'] += 1
            if is_correct:
                solution_stats[sol_key]['correct'] += 1
            results.append({
                "idx": batch[idx]['idx'],
                "category": batch[idx]['category'],
                "example_idx": batch[idx]['example_idx'],
                "problem": prob,
                "solution": sol,
                "response": resp[0] if isinstance(resp, list) else resp,
                "pred": pred,
                "correct": is_correct,
                "query_image_path": batch[idx]['query_path'],
                "reference_image": batch[idx]['reference_path'],
            })

        clear_cuda_cache()

    # Compute overall metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
    }

    # Add per-solution-type metrics
    for sol_key, stats in solution_stats.items():
        prefix = f"solution_{sol_key}"
        metrics[f"{prefix}_correct"] = stats['correct']
        metrics[f"{prefix}_total"] = stats['total']
        metrics[f"{prefix}_accuracy"] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    # Compute positive/negative accuracy if yes/no present
    if 'yes' in solution_stats and 'no' in solution_stats:
        metrics['positive_accuracy'] = metrics['solution_yes_accuracy']
        metrics['negative_accuracy'] = metrics['solution_no_accuracy']
        metrics['positive_total'] = metrics['solution_yes_total']
        metrics['negative_total'] = metrics['solution_no_total']

    return results, metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test recognition performance on HuggingFace dataset"
    )

    # Dataset args
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to HuggingFace dataset directory"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to process (default: all)"
    )

    # Model args
    parser.add_argument(
        "--model_type", type=str, default="original_7b",
        help="Model type or path"
    )
    parser.add_argument(
        "--data_name", type=str, default="YoLLaVA",
        help="Dataset name for model config lookup"
    )
    parser.add_argument(
        "--seed", type=int, default=23,
        help="Random seed"
    )

    # Inference args
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--temperature", type=float, default=1e-6,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="Maximum new tokens to generate"
    )

    # Output args
    parser.add_argument(
        "--output_dir", type=str, default="results/hf_recognition",
        help="Output directory for results"
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Output filename (default: auto-generated)"
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    args = parse_args()
    set_seed(args.seed)

    LOG.info("=" * 80)
    LOG.info("HF Recognition Test")
    LOG.info("=" * 80)
    LOG.info("Dataset path:   %s", args.dataset_path)
    LOG.info("Split:          %s", args.split)
    LOG.info("Model type:     %s", args.model_type)
    LOG.info("Batch size:     %d", args.batch_size)
    LOG.info("Max samples:    %s", args.max_samples or "all")
    LOG.info("=" * 80)

    # Load dataset items
    items = prepare_hf_dataset_items(
        args.dataset_path,
        split=args.split,
        max_samples=args.max_samples,
    )
    LOG.info("Prepared %d items for inference", len(items))

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

    LOG.info("Loading model from %s (use_peft=%s)", model_path, use_peft)
    model, processor = setup_model(model_path, use_peft=use_peft)

    # Run inference
    results, metrics = run_inference_loop(
        model, processor, items,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        outpath = outdir / args.output_name
    else:
        dataset_name = Path(args.dataset_path).name
        outpath = outdir / f"recognition_{dataset_name}_model_{args.model_type}.json"

    save_results(results, metrics, vars(args), outpath)

    # Print summary
    LOG.info("=" * 80)
    LOG.info("Results Summary")
    LOG.info("=" * 80)
    LOG.info("Overall Accuracy: %.4f (%d/%d)",
             metrics['accuracy'], metrics['correct'], metrics['total'])

    if 'positive_accuracy' in metrics:
        LOG.info("Positive Accuracy (yes): %.4f (%d/%d)",
                 metrics['positive_accuracy'],
                 metrics['solution_yes_correct'],
                 metrics['positive_total'])
        LOG.info("Negative Accuracy (no):  %.4f (%d/%d)",
                 metrics['negative_accuracy'],
                 metrics['solution_no_correct'],
                 metrics['negative_total'])

    # Show per-solution-type breakdown
    solution_types = [k.replace('solution_', '').replace('_accuracy', '')
                     for k in metrics.keys() if k.startswith('solution_') and k.endswith('_accuracy')]

    if len(solution_types) > 2:  # More than just yes/no
        LOG.info("")
        LOG.info("Per-solution-type breakdown:")
        for sol_type in sorted(solution_types):
            prefix = f"solution_{sol_type}"
            LOG.info("  %s: %.4f (%d/%d)",
                     sol_type,
                     metrics[f"{prefix}_accuracy"],
                     metrics[f"{prefix}_correct"],
                     metrics[f"{prefix}_total"])

    LOG.info("=" * 80)
    LOG.info("Results saved to: %s", outpath)
    LOG.info("=" * 80)


if __name__ == "__main__":
    main()
