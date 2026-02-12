#!/usr/bin/env python3
"""
Aggregate Identification Metrics

Computes macro Precision/Recall/F1 from per-concept personalized identification results.
This script aggregates concept-level predictions into dataset/category summaries.

Usage:
    python aggregate_identification.py --dataset YoLLaVA --model_type original_7b --seed 23 --k 3

Input: Per-concept JSON files with 'pred_name' and 'solution' fields
Output: Aggregated metrics JSON with macro P/R/F1 scores
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from utils import (
    add_common_eval_args,
    get_categories,
    get_results_base_path,
    get_concept_result_path,
    log_debug,
    read_records,
    safe_div,
    save_json,
    scan_concepts,
)


# =============================================================================
# Precision/Recall/F1 Computation
# =============================================================================

def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Tuple of (precision, recall, f1) as percentages (0-100)
    """
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision * 100.0, recall * 100.0, f1 * 100.0


def update_confusions_for_pair(
    per_class: Dict[str, Dict[str, int]],
    gold: str,
    pred: str
) -> None:
    """
    Update confusion matrix for a single prediction (one-vs-rest per class).

    Args:
        per_class: Dictionary tracking TP/FP/FN/support per class
        gold: Ground truth label
        pred: Predicted label
    """
    if pred == gold:
        per_class.setdefault(gold, {"TP": 0, "FP": 0, "FN": 0, "support": 0})
        per_class[gold]["TP"] += 1
        per_class[gold]["support"] += 1
    else:
        per_class.setdefault(pred, {"TP": 0, "FP": 0, "FN": 0, "support": 0})
        per_class.setdefault(gold, {"TP": 0, "FP": 0, "FN": 0, "support": 0})
        per_class[pred]["FP"] += 1
        per_class[gold]["FN"] += 1
        per_class[gold]["support"] += 1


def macro_from_per_class(per_class: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    Compute macro precision/recall/F1 across all classes.

    Args:
        per_class: Dictionary with per-class TP/FP/FN counts

    Returns:
        Dictionary with macro precision, recall, and f1 scores
    """
    # Filter classes with any activity
    active_classes = [
        cls for cls, v in per_class.items()
        if v.get("TP", 0) + v.get("FP", 0) + v.get("FN", 0) > 0
    ]

    if not active_classes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precs, recs, f1s = [], [], []
    for cls in active_classes:
        v = per_class[cls]
        p, r, f = prf1(v.get("TP", 0), v.get("FP", 0), v.get("FN", 0))
        precs.append(p)
        recs.append(r)
        f1s.append(f)

    return {
        "precision": sum(precs) / len(precs),
        "recall": sum(recs) / len(recs),
        "f1": sum(f1s) / len(f1s),
    }


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate macro P/R/F1 from concept prediction files."
    )
    add_common_eval_args(parser)
    parser.add_argument("--k", type=int, default=3, help="Number of retrieval candidates (k)")
    return parser.parse_args()


def get_result_filename(args: argparse.Namespace) -> str:
    """Generate the expected result filename for identification task."""
    return f"results_model_{args.model_type}_db_{args.db_type}_k_{args.k}.json"

def get_test_concepts(dataset, category, seed):
    if dataset in ['YoLLaVA', 'MyVLM']:
        filename = f'{dataset}_concept_list.txt'
    else:
        filename = f'OSC_subset_seed_{seed}.txt'
    with open(filename) as f:
        concepts = [line.strip().split(',')[1] for line in f.readlines() if line.strip().split(',')[0]==category]
    return concepts

def main() -> int:
    args = parse_args()
    dataset = args.dataset

    print(f"# Computing MACRO metrics for dataset={dataset}, seed={args.seed}")

    categories = get_categories(dataset)

    # Counters
    total_concepts_seen = 0
    total_concepts_with_file = 0
    total_examples_processed = 0

    # Accumulators
    per_concept_metrics: Dict[str, Dict[str, float]] = {}
    per_class_global: Dict[str, Dict[str, int]] = {}

    result_filename = get_result_filename(args)
    for category in categories:
        base = get_results_base_path(dataset, category)
        concepts = get_test_concepts(dataset, category, args.seed)
        total_concepts_seen += len(concepts)
        for concept in concepts:
            concept_path = get_concept_result_path(base, concept, args.seed, result_filename)
            if not concept_path.exists():
                log_debug(f"{category},{concept}")
                continue
            records = read_records(concept_path)
            if not records:
                log_debug(f"{category},{concept}")
            else:
                total_concepts_with_file += 1
            # Update global per-class totals
            for r in records:
                pred = str(r.get("pred_name", "")).strip()
                gold = str(r.get("solution", "")).strip()
                if pred == "" and gold == "":
                    continue
                update_confusions_for_pair(per_class_global, gold, pred)

            total_examples_processed += sum(
                1 for r in records
                if not (str(r.get("pred_name", "")).strip() == "" and
                        str(r.get("solution", "")).strip() == "")
            )
            # Per-concept metrics for single-category datasets
            if dataset in ("YoLLaVA", "MyVLM", "DreamBooth"):
                tp = fp = fn = 0
                for r in records:
                    pred = str(r.get("pred_name", "")).strip()
                    gold = str(r.get("solution", "")).strip()
                    if gold == concept and pred == concept:
                        tp += 1
                    elif gold != concept and pred == concept:
                        fp += 1
                    elif gold == concept and pred != concept:
                        fn += 1
                p, r, f = prf1(tp, fp, fn)
                per_concept_metrics[concept] = {
                    "precision": p,
                    "recall": r,
                    "f1": f,
                    "support": tp + fn
                }

    # Final aggregation
    # import pdb;pdb.set_trace()
    macro_metrics = macro_from_per_class(per_class_global)
    # Sum TP/FP/FN across classes
    sum_tp = sum(v.get("TP", 0) for v in per_class_global.values())
    sum_fp = sum(v.get("FP", 0) for v in per_class_global.values())
    sum_fn = sum(v.get("FN", 0) for v in per_class_global.values())

    # Compute global TN using one-vs-rest identity
    num_classes = len([
        c for c, v in per_class_global.items()
        if (v.get("TP", 0) + v.get("FP", 0) + v.get("FN", 0)) > 0
    ])
    global_tn = (
        num_classes * total_examples_processed - (sum_tp + sum_fp + sum_fn)
        if num_classes > 0 else 0
    )

    # Build results object
    if dataset in ("YoLLaVA", "MyVLM", "DreamBooth"):
        results = {
            "metrics": {
                "macro": {
                    "precision": macro_metrics["precision"],
                    "recall": macro_metrics["recall"],
                    "f1": macro_metrics["f1"],
                },
                "tp": sum_tp,
                "fp": sum_fp,
                "fn": sum_fn,
                "tn": global_tn,
            },
            "concepts": per_concept_metrics,
            "classes": {
                cls: {
                    "TP": v["TP"],
                    "FP": v["FP"],
                    "FN": v["FN"],
                    "support": v.get("support", 0)
                }
                for cls, v in per_class_global.items()
                if v.get("TP", 0) + v.get("FP", 0) + v.get("FN", 0) > 0
            },
        }
    else:
        # PerVA: global macro across categories
        results = {
            "metrics": {
                "precision": macro_metrics["precision"],
                "recall": macro_metrics["recall"],
                "f1": macro_metrics["f1"],
                "tp": sum_tp,
                "fp": sum_fp,
                "fn": sum_fn,
                "tn": global_tn,
            },
            "category": {}
        }

    # Report and save
    print(f"total concepts: {total_concepts_seen}")
    print(f"results showing for concepts: {total_concepts_with_file}")

    save_path = (
        Path("results") / dataset /
        f"results_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}_k_{args.k}.json"
    )
    save_json(results, save_path)
    print(f"Saved metrics at {save_path}")
    print(results["metrics"])

    return 0


if __name__ == "__main__":
    sys.exit(main())

#python src/eval_utils/aggregate_identification.py --db_type original_7b --model_type original_7b --seed 23 --out reports/ 