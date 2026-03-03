#!/usr/bin/env python3
"""
Aggregate Skip-Retrieval Metrics

Aggregates per-concept results from the skip-retrieval ablation
(personalize_skip_retrieval.py) into dataset/category summaries.

Two additions over aggregate_recognition.py:
  1. Filename is fixed to results_model_{model}_db_{db}_skip_ret.json
  2. P/R/F1 is aggregated in two ways:
       - micro: recomputed from the summed TP/FP/FN across all concepts
       - macro: mean of per-concept precision/recall/F1

Usage:
    python aggregate_skip_ret.py --dataset YoLLaVA --model_type ls_soft_gated --db_type sp_concise_soft_gated --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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
# Helpers
# =============================================================================

def safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


def get_result_filename(model_type: str, db_type: str) -> str:
    """Fixed filename pattern for skip-retrieval results."""
    return f"results_model_{model_type}_db_{db_type}_skip_ret.json"


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-concept skip-retrieval metrics into dataset/category summaries."
    )
    add_common_eval_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    result_filename = get_result_filename(args.model_type, args.db_type)

    print(f"# Aggregating skip-retrieval metrics for dataset={dataset}, seed={args.seed}")
    print(f"# Reading files named: {result_filename}")

    categories = get_categories(dataset)

    # Overall aggregation (for PerVA)
    overall: Dict = {
        "metrics": _empty_metrics(),
        "category": {},
    }

    total_concepts_seen = 0
    total_concepts_with_file = 0
    results_per_cat_last = None

    for category in categories:
        base = get_results_base_path(dataset, category)

        results_per_cat: Dict = {
            "metrics": _empty_metrics(),
            "concepts": {},
        }

        # Per-concept P/R/F1 values for macro averaging
        per_concept_precision: List[float] = []
        per_concept_recall: List[float] = []
        per_concept_f1: List[float] = []

        concept_names = scan_concepts(base)
        total_concepts_seen += len(concept_names)

        for name in concept_names:
            concept_path = get_concept_result_path(base, name, args.seed, result_filename)

            if not concept_path.exists():
                log_debug(f"{category},{name},missing_file")
                continue

            total_concepts_with_file += 1
            try:
                data = load_json(concept_path)
            except Exception as e:
                log_debug(f"{category},{name},load_error,{str(e)}")
                continue

            m = data["metrics"]
            correct_yes = int(m["correct_yes"])
            total_yes   = int(m["total_yes"])
            correct_no  = int(m["correct_no"])
            total_no    = int(m["total_no"])

            if total_yes == 0 and total_no == 0:
                log_debug(f"{category},{name},zero_samples")

            # Accumulate raw counts (for micro P/R/F1 and pos/neg accuracy)
            cat_m = results_per_cat["metrics"]
            cat_m["correct_yes"] += correct_yes
            cat_m["total_yes"]   += total_yes
            cat_m["correct_no"]  += correct_no
            cat_m["total_no"]    += total_no

            # Accumulate TP/FP/FN for micro aggregation
            tp = correct_yes
            fp = total_no - correct_no
            fn = total_yes - correct_yes
            cat_m["tp"] += tp
            cat_m["fp"] += fp
            cat_m["fn"] += fn

            # Collect per-concept P/R/F1 for macro averaging
            # Re-derive from counts so we're consistent even if stored values differ
            p  = safe_div(tp, tp + fp)
            r  = safe_div(tp, tp + fn)
            f1 = safe_div(2 * p * r, p + r)
            per_concept_precision.append(p)
            per_concept_recall.append(r)
            per_concept_f1.append(f1)

            # Store per-concept entry
            results_per_cat["concepts"][name] = {
                "pos_accuracy": safe_percentage(correct_yes, total_yes),
                "neg_accuracy": safe_percentage(correct_no, total_no),
                "precision":    round(p,  4),
                "recall":       round(r,  4),
                "f1":           round(f1, 4),
                "total_yes":    total_yes,
                "total_no":     total_no,
            }

        # Finalise category-level metrics
        _finalize_metrics(results_per_cat["metrics"], per_concept_precision, per_concept_recall, per_concept_f1)

        # For PerVA: accumulate into overall
        if dataset == "PerVA":
            om = overall["metrics"]
            cat_m = results_per_cat["metrics"]
            for key in ("correct_yes", "total_yes", "correct_no", "total_no", "tp", "fp", "fn"):
                om[key] += cat_m[key]
            overall["category"][category] = {
                "pos_accuracy":    cat_m["pos_accuracy"],
                "neg_accuracy":    cat_m["neg_accuracy"],
                "micro_precision": cat_m["micro_precision"],
                "micro_recall":    cat_m["micro_recall"],
                "micro_f1":        cat_m["micro_f1"],
                "macro_precision": cat_m["macro_precision"],
                "macro_recall":    cat_m["macro_recall"],
                "macro_f1":        cat_m["macro_f1"],
            }

        results_per_cat_last = results_per_cat

    # Determine final results object
    if dataset in ["YoLLaVA", "MyVLM", "DreamBooth"]:
        results = results_per_cat_last if results_per_cat_last is not None else {
            "metrics": _empty_metrics(),
            "concepts": {},
        }
    else:
        # PerVA: finalise overall (macro is not meaningful here — use category-level macro)
        results = overall
        _finalize_metrics(results["metrics"], [], [], [])   # micro-only for overall

    # Report
    print(f"total concepts: {total_concepts_seen}")
    print(f"results showing for concepts: {total_concepts_with_file}")
    print(results["metrics"])

    # Save
    save_path = (
        Path("results") / dataset /
        f"skip_ret_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}.json"
    )
    save_json(results, save_path)
    print(f"Saved metrics at {save_path}")

    return 0


# =============================================================================
# Internal helpers
# =============================================================================

def _empty_metrics() -> Dict:
    return {
        "correct_yes": 0, "total_yes": 0,
        "correct_no":  0, "total_no":  0,
        "tp": 0, "fp": 0, "fn": 0,
        # filled by _finalize_metrics
        "pos_accuracy":    0.0, "neg_accuracy":    0.0,
        "micro_precision": 0.0, "micro_recall":    0.0, "micro_f1": 0.0,
        "macro_precision": 0.0, "macro_recall":    0.0, "macro_f1": 0.0,
        "weighted_accuracy": 0.0,
    }


def _finalize_metrics(
    m: Dict,
    per_concept_precision: List[float],
    per_concept_recall: List[float],
    per_concept_f1: List[float],
) -> None:
    """Compute derived metrics in-place from accumulated counts."""
    # pos / neg accuracy
    m["pos_accuracy"] = safe_percentage(m["correct_yes"], m["total_yes"])
    m["neg_accuracy"] = safe_percentage(m["correct_no"],  m["total_no"])
    m["weighted_accuracy"] = (m["pos_accuracy"] + m["neg_accuracy"]) / 2

    # Micro P/R/F1 from aggregated TP/FP/FN
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    micro_p  = safe_div(tp, tp + fp)
    micro_r  = safe_div(tp, tp + fn)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r)
    m["micro_precision"] = round(micro_p,  4)
    m["micro_recall"]    = round(micro_r,  4)
    m["micro_f1"]        = round(micro_f1, 4)

    # Macro P/R/F1 — mean of per-concept values
    n = len(per_concept_precision)
    if n > 0:
        m["macro_precision"] = round(sum(per_concept_precision) / n, 4)
        m["macro_recall"]    = round(sum(per_concept_recall)    / n, 4)
        m["macro_f1"]        = round(sum(per_concept_f1)        / n, 4)


if __name__ == "__main__":
    sys.exit(main())
