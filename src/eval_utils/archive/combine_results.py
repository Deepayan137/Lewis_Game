#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PERVA_CATEGORIES = [
    "bag", "book", "bottle", "bowl", "clothe", "cup", "decoration", "headphone",
    "pillow", "plant", "plate", "remote", "retail", "telephone", "tie", "towel",
    "toy", "tro_bag", "tumbler", "umbrella", "veg"
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate concept-level metrics into dataset/category summaries.")
    p.add_argument("--dataset", default="YoLLaVA", help="Dataset name: YoLLaVA, MyVLM, DreamBooth, or PerVA")
    p.add_argument("--db_type", default="original")
    p.add_argument("--model_type", default="base_qwen")
    p.add_argument("--eval_type", default="recall")
    p.add_argument("--category", default="all")  # preserved for compatibility; not used when scanning
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", type=str, default=None)  # preserved for compatibility; file name is standardized
    return p.parse_args()


def safe_accuracy(correct: int, total: int) -> float:
    """Return accuracy in percent (0–100)."""
    return (correct / total) * 100.0 if total > 0 else 0.0


def scan_concepts(base: Path) -> list[str]:
    """
    Discover concept names under `base`, keeping only non-empty directories.
    A concept dir is considered 'valid' if it has at least one child.
    """
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and any(d.iterdir())
    )


def save_list(items: list[str], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.suffix.lower() == ".json":
        outpath.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        outpath.write_text("\n".join(items) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    dataset = args.dataset  # keep original for paths/printing
    dataset_norm = dataset.strip()  # do not lower-case paths; keep exact name as provided
    print(f"# Concepts for dataset={dataset}, seed={args.seed}")

    # Choose category list
    if dataset == "PerVA":
        categories = PERVA_CATEGORIES
    else:
        # For other datasets the code expects a single 'all' bucket on disk
        categories = ["all"]

    overall = {
        "metrics": {"correct": 0, "total": 0, "accuracy": 0.0},
        "category": {},  # For PerVA: category -> accuracy; for others not used in final output
    }

    # Debug counters
    in_total_concepts = 0
    concept_count = 0

    # We’ll reuse this for the final object in non-PerVA cases (it will end up as the last category's aggregation,
    # which is correct here because categories=['all']).
    results_per_cat_last = None

    for category in categories:
        outpath = Path("results") / dataset_norm / category

        # Aggregate object for this category
        
        results_per_cat = {
            "metrics": {"correct": 0, "total": 0, "accuracy": 0.0},
            "concepts": {}
        }

        concept_names = scan_concepts(outpath)

        for name in concept_names:
            # Expected file:
            eval_json = f'results_model_{args.model_type}_db_{args.db_type}_k_{args.k}.json' if args.eval_type == 'recall' else f"recognition_model_{args.model_type}_db_{args.db_type}.json"
            concept_path = (
                outpath / name / f"seed_{args.seed}" / eval_json
            )
            in_total_concepts += 1

            if concept_path.exists():
                concept_count += 1
                with concept_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                correct = int(data["metrics"]["correct count"])
                total = int(data["metrics"]["total samples"])
                if total ==0:
                    with open("debug.txt", "a", encoding="utf-8") as f:
                        f.write(f"{category},{name}\n")
                    print(f"{category},{name}\n")
                results_per_cat["metrics"]["correct"] += correct
                results_per_cat["metrics"]["total"] += total
                results_per_cat["concepts"][name] = {
                    "accuracy": safe_accuracy(correct, total),
                    "total": total,
                }
            else:
                # Log missing concept to debug file
                with open("debug.txt", "a", encoding="utf-8") as f:
                    f.write(f"{category},{name}\n")

        # finalize per-category accuracy
        m = results_per_cat["metrics"]
        m["accuracy"] = safe_accuracy(m["correct"], m["total"])

        # For PerVA we keep an overall aggregation and store per-category accuracy
        if dataset == "PerVA":
            overall["metrics"]["correct"] += m["correct"]
            overall["metrics"]["total"] += m["total"]
            overall["category"][category] = m["accuracy"]

        # Keep the last (only) category block for non-PerVA output
        results_per_cat_last = results_per_cat

    # Determine final results object to save
    if dataset in ["YoLLaVA", "MyVLM", "DreamBooth"]:
        results = results_per_cat_last if results_per_cat_last is not None else {
            "metrics": {"correct": 0, "total": 0, "accuracy": 0.0},
            "concepts": {}
        }
    else:
        results = overall
        results["metrics"]["accuracy"] = safe_accuracy(
            results["metrics"]["correct"], results["metrics"]["total"]
        )
    # Report and save
    print(f"total concepts:{in_total_concepts}")
    print(f"results showing for concepts:{concept_count}")
    report_json = f"recognition_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}.json" if args.eval_type == 'recognition' else f"recall_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}.json"
    save_path = Path("results") / dataset_norm / report_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model at {save_path}")

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(results["metrics"])


if __name__ == "__main__":
    sys.exit(main())
