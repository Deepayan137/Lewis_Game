#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PERVA_CATEGORIES = [
    "bag","book","bottle","bowl","clothe","cup","decoration","headphone","pillow",
    "plant","plate","remote","retail","telephone","tie","towel","toy","tro_bag",
    "tumbler","umbrella","veg"
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute macro P/R/F1 from concept prediction files (list of dicts).")
    p.add_argument("--dataset", default="YoLLaVA", help="YoLLaVA, MyVLM, DreamBooth, or PerVA")
    p.add_argument("--db_type", default="original")
    p.add_argument("--model_type", default="original_7b")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--category", default="all")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()

def safe_div(a: float, b: float) -> float:
    return (a / b) if b > 0 else 0.0

def prf1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return precision, recall, f1 in PERCENT."""
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p * 100.0, r * 100.0, f * 100.0

def scan_concepts(base: Path) -> List[str]:
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and any(d.iterdir()))

def update_confusions_for_pair(per_class: Dict[str, Dict[str, int]], gold: str, pred: str):
    """
    Multiclass, single-label bookkeeping (one-vs-rest per class):
      - correct: TP for that class (gold)
      - error: FP for pred class, FN for gold class
    'support' tracks positives for the class (TP+FN).
    """
    if pred == gold:
        per_class.setdefault(gold, {"TP":0,"FP":0,"FN":0,"support":0})
        per_class[gold]["TP"] += 1
        per_class[gold]["support"] += 1
    else:
        per_class.setdefault(pred, {"TP":0,"FP":0,"FN":0,"support":0})
        per_class.setdefault(gold, {"TP":0,"FP":0,"FN":0,"support":0})
        per_class[pred]["FP"] += 1
        per_class[gold]["FN"] += 1
        per_class[gold]["support"] += 1

def read_records(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("records", "data", "items", "predictions", "results"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    return []

def macro_from_per_class(per_class: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Compute macro precision/recall/F1 across classes that had any TP/FP/FN activity."""
    classes = []
    for cls, v in per_class.items():
        if v.get("TP",0) + v.get("FP",0) + v.get("FN",0) > 0:
            classes.append(cls)
    if not classes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precs, recs, f1s = [], [], []
    for cls in classes:
        v = per_class[cls]
        P, R, F = prf1(v.get("TP",0), v.get("FP",0), v.get("FN",0))
        precs.append(P); recs.append(R); f1s.append(F)
    return {
        "precision": sum(precs) / len(precs),
        "recall":    sum(recs)  / len(recs),
        "f1":        sum(f1s)   / len(f1s),
    }

def main():
    args = parse_args()
    dataset = args.dataset
    print(f"# Computing MACRO metrics for dataset={dataset}, seed={args.seed}")

    categories = PERVA_CATEGORIES if dataset == "PerVA" else ["all"]

    # Global counters
    total_concepts_seen = 0
    total_concepts_with_file = 0
    total_examples_processed = 0  # N (per category for non-PerVA; across cats for PerVA)

    # Per-dataset accumulators
    per_concept_metrics: Dict[str, Dict[str, float]] = {}  # kept for non-PerVA (debug/inspection)
    per_class_global: Dict[str, Dict[str, int]] = {}       # accumulates over 'all' (or all PerVA categories)

    for category in categories:
        base = Path("results") / dataset / category
        concepts = scan_concepts(base)
        total_concepts_seen += len(concepts)

        for concept in concepts:
            concept_path = (
                base / concept / f"seed_{args.seed}" /
                f"results_model_{args.model_type}_db_{args.db_type}_k_{args.k}.json"
            )
            if not concept_path.exists():
                with open("debug.txt", "a", encoding="utf-8") as dbg:
                    dbg.write(f"{category},{concept}\n")
                continue

            total_concepts_with_file += 1
            records = read_records(concept_path)

            # Update global per-class totals
            for r in records:
                pred = str(r.get("pred_name","")).strip()
                gold = str(r.get("solution","")).strip()
                if pred == "" and gold == "":
                    continue
                update_confusions_for_pair(per_class_global, gold, pred)
            total_examples_processed += sum(
                1 for r in records if not (str(r.get("pred_name","")).strip() == "" and str(r.get("solution","")).strip() == "")
            )

            # Per-concept (one-vs-rest) metrics for inspection (non-PerVA datasets)
            if dataset in ("YoLLaVA","MyVLM","DreamBooth"):
                tp = fp = fn = 0
                for r in records:
                    pred = str(r.get("pred_name","")).strip()
                    gold = str(r.get("solution","")).strip()
                    if gold == concept and pred == concept: tp += 1
                    elif gold != concept and pred == concept: fp += 1
                    elif gold == concept and pred != concept: fn += 1
                P, R, F = prf1(tp, fp, fn)
                per_concept_metrics[concept] = {"precision": P, "recall": R, "f1": F, "support": tp + fn}

    # ---- Final aggregation ----
    # Macro P/R/F1 across classes
    macro_metrics = macro_from_per_class(per_class_global)

    # Sums of TP/FP/FN across classes (not used for macro, but reported)
    sum_tp = sum(v.get("TP",0) for v in per_class_global.values())
    sum_fp = sum(v.get("FP",0) for v in per_class_global.values())
    sum_fn = sum(v.get("FN",0) for v in per_class_global.values())

    # Compute a coherent global TN using one-vs-rest identity:
    # For each class c, TN_c = N - TP_c - FP_c - FN_c; summing => Σ TN_c = C*N - Σ(TP+FP+FN)
    num_classes = len([c for c,v in per_class_global.items() if (v.get("TP",0)+v.get("FP",0)+v.get("FN",0)) > 0])
    global_tn = num_classes * total_examples_processed - (sum_tp + sum_fp + sum_fn) if num_classes > 0 else 0

    if dataset in ("YoLLaVA","MyVLM","DreamBooth"):
        results = {
            "metrics": {
                "macro": {
                    "precision": macro_metrics["precision"],
                    "recall": macro_metrics["recall"],
                    "f1": macro_metrics["f1"],
                },
                "tp": sum_tp, "fp": sum_fp, "fn": sum_fn, "tn": global_tn,
            },
            "concepts": per_concept_metrics,   # keep for drill-down; remove if you want a lighter file
            "classes": {                       # per-class one-vs-rest counts (optional)
                cls: {"TP":v["TP"], "FP":v["FP"], "FN":v["FN"], "support":v.get("support",0)}
                for cls, v in per_class_global.items()
                if v.get("TP",0)+v.get("FP",0)+v.get("FN",0) > 0
            },
        }
    else:
        # PerVA: global macro across classes (and include global confusion sums)
        results = {
            "metrics": {
                "precision": macro_metrics["precision"],
                "recall": macro_metrics["recall"],
                "f1": macro_metrics["f1"],
                "tp": sum_tp, "fp": sum_fp, "fn": sum_fn, "tn": global_tn,
            },
            "category": {}
        }

    print(f"total concepts:{total_concepts_seen}")
    print(f"results showing for concepts:{total_concepts_with_file}")

    save_path = Path("results") / dataset / f"results_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}_k_{args.k}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving metrics at {save_path}")
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(results["metrics"])

if __name__ == "__main__":
    sys.exit(main())
