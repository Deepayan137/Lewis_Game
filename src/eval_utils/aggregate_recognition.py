#!/usr/bin/env python3
"""
Aggregate Recognition Metrics

Computes positive and negative accuracy from per-concept binary recognition results.
This script aggregates concept-level Yes/No accuracy into dataset/category summaries.

Usage:
    python aggregate_recognition.py --dataset YoLLaVA --model_type original_7b --seed 23

Input: Per-concept JSON files with correct_yes, total_yes, correct_no, total_no metrics
Output: Aggregated metrics JSON with pos_accuracy and neg_accuracy scores
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

from utils import (
    add_common_eval_args,
    get_categories,
    get_results_base_path,
    get_concept_result_path,
    load_json,
    log_debug,
    safe_percentage,
    save_json,
    scan_concepts,
)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate concept-level recognition metrics into dataset/category summaries."
    )
    add_common_eval_args(parser)
    parser.add_argument(
        "--eval_type",
        default="recognition",
        choices=["recognition", "recall"],
        help="Evaluation type (determines input filename format)"
    )
    parser.add_argument("--k", type=int, default=3, help="Number of retrieval candidates (for recall eval_type)")
    return parser.parse_args()


def get_result_filename(args: argparse.Namespace) -> str:
    """Generate the expected result filename based on evaluation type."""
    if args.eval_type == "recall":
        return f"results_model_{args.model_type}_db_{args.db_type}_k_{args.k}.json"
    return f"recognition_model_{args.model_type}_db_{args.db_type}.json"


def main() -> int:
    args = parse_args()
    dataset = args.dataset

    print(f"# Aggregating recognition metrics for dataset={dataset}, seed={args.seed}")

    categories = get_categories(dataset)

    # Overall aggregation (for PerVA)
    overall: Dict = {
        "metrics": {
            "correct_yes": 0,
            "total_yes": 0,
            "correct_no": 0,
            "total_no": 0,
            "pos_accuracy": 0.0,
            "neg_accuracy": 0.0,
        },
        "category": {},
    }

    # Counters
    total_concepts_seen = 0
    total_concepts_with_file = 0

    # Last category result (for non-PerVA datasets)
    results_per_cat_last = None

    result_filename = get_result_filename(args)
    for category in categories:
        base = get_results_base_path(dataset, category)

        # Category-level aggregation
        results_per_cat: Dict = {
            "metrics": {
                "correct_yes": 0,
                "correct_no": 0,
                "total_yes": 0,
                "total_no": 0,
            },
            "concepts": {}
        }

        concept_names = scan_concepts(base)
        total_concepts_seen += len(concept_names)

        for name in concept_names:
            concept_path = get_concept_result_path(base, name, args.seed, result_filename)

            if not concept_path.exists():
                log_debug(f"{category},{name},missing_file")
                continue

            total_concepts_with_file += 1
            data = load_json(concept_path)

            # Extract metrics
            correct_yes = int(data["metrics"]["correct_yes"])
            total_yes = int(data["metrics"]["total_yes"])
            correct_no = int(data["metrics"]["correct_no"])
            total_no = int(data["metrics"]["total_no"])

            # Check for zero samples
            if total_yes == 0 and total_no == 0:
                log_debug(f"{category},{name},zero_samples")

            # Accumulate
            results_per_cat["metrics"]["correct_yes"] += correct_yes
            results_per_cat["metrics"]["total_yes"] += total_yes
            results_per_cat["metrics"]["correct_no"] += correct_no
            results_per_cat["metrics"]["total_no"] += total_no

            # Store per-concept metrics
            results_per_cat["concepts"][name] = {
                "pos_accuracy": safe_percentage(correct_yes, total_yes),
                "neg_accuracy": safe_percentage(correct_no, total_no),
                "total_yes": total_yes,
                "total_no": total_no,
            }

        # Finalize per-category accuracy
        m = results_per_cat["metrics"]
        m["pos_accuracy"] = safe_percentage(m["correct_yes"], m["total_yes"])
        m["neg_accuracy"] = safe_percentage(m["correct_no"], m["total_no"])

        # For PerVA: accumulate into overall
        if dataset == "PerVA":
            overall["metrics"]["correct_yes"] += m["correct_yes"]
            overall["metrics"]["total_yes"] += m["total_yes"]
            overall["metrics"]["correct_no"] += m["correct_no"]
            overall["metrics"]["total_no"] += m["total_no"]
            overall["category"][category] = {
                "pos_accuracy": m["pos_accuracy"],
                "neg_accuracy": m["neg_accuracy"],
            }

        results_per_cat_last = results_per_cat

    # Determine final results object
    if dataset in ["YoLLaVA", "MyVLM", "DreamBooth"]:
        results = results_per_cat_last if results_per_cat_last is not None else {
            "metrics": {
                "correct_yes": 0,
                "total_yes": 0,
                "pos_accuracy": 0.0,
                "correct_no": 0,
                "total_no": 0,
                "neg_accuracy": 0.0,
            },
            "concepts": {}
        }
    else:
        # PerVA: finalize overall metrics
        results = overall
        results["metrics"]["pos_accuracy"] = safe_percentage(
            results["metrics"]["correct_yes"],
            results["metrics"]["total_yes"]
        )
        results["metrics"]["neg_accuracy"] = safe_percentage(
            results["metrics"]["correct_no"],
            results["metrics"]["total_no"]
        )
    results["metrics"]["weigthted_accuracy"] = (results["metrics"]["pos_accuracy"] + results["metrics"]["neg_accuracy"]) / 2
    # Report and save
    print(f"total concepts: {total_concepts_seen}")
    print(f"results showing for concepts: {total_concepts_with_file}")

    save_path = (
        Path("results") / dataset /
        f"recognition_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}.json"
    )
    save_json(results, save_path)
    print(f"Saved metrics at {save_path}")
    print(results["metrics"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
