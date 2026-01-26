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
    p.add_argument("--model_type", default="original_7b")
    p.add_argument("--eval_type", default="recognition")
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
        "metrics": {"correct_yes": 0, "total_yes": 0, "correct_no": 0, "total_no": 0, "pos_accuracy": 0.0, "neg_accuracy": 0.0},  # FIX: Added missing keys
        "category": {},  # For PerVA: category -> accuracy; for others not used in final output
    }

    # Debug counters
    in_total_concepts = 0
    concept_count = 0

    # We'll reuse this for the final object in non-PerVA cases (it will end up as the last category's aggregation,
    # which is correct here because categories=['all']).
    results_per_cat_last = None

    for category in categories:
        outpath = Path("results") / dataset_norm / category

        # Aggregate object for this category

        results_per_cat = {
            "metrics": {"correct_yes": 0, "correct_no": 0, "total_yes": 0, "total_no": 0},  # FIX: fixed spacing in correct_no
            "concepts": {}
        }

        concept_names = scan_concepts(outpath)

        for name in concept_names:
            # Expected file:
            eval_json = f'results_model_{args.model_type}_db_{args.db_type}_k_{args.k}.json' if args.eval_type == 'recall' else f"recognition_model_{args.model_type}_db_{args.db_type}.json"
            # import pdb;pdb.set_trace()
            concept_path = (
                outpath / name / f"seed_{args.seed}" / eval_json
            )
            in_total_concepts += 1

            if concept_path.exists():
                concept_count += 1
                with concept_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                correct_yes = int(data["metrics"]["correct_yes"])
                total_yes = int(data["metrics"]["total_yes"])
                correct_no = int(data["metrics"]["correct_no"])
                total_no = int(data["metrics"]["total_no"])
                
                # FIX: Check individual totals instead of undefined 'total' variable
                if total_yes == 0 and total_no == 0:
                    with open("debug.txt", "a", encoding="utf-8") as f:
                        f.write(f"{category},{name},zero_samples\n")
                    print(f"{category},{name},zero_samples")
                
                results_per_cat["metrics"]["correct_yes"] += correct_yes
                results_per_cat["metrics"]["total_yes"] += total_yes
                results_per_cat["metrics"]["correct_no"] += correct_no
                results_per_cat["metrics"]["total_no"] += total_no
                results_per_cat["concepts"][name] = {
                    "pos_accuracy": safe_accuracy(correct_yes, total_yes),
                    "neg_accuracy": safe_accuracy(correct_no, total_no),
                    "total_yes": total_yes,
                    "total_no": total_no
                }
            else:
                # Log missing concept to debug file
                print(f"{category},{name},missing_file\n")
                with open("debug.txt", "a", encoding="utf-8") as f:
                    f.write(f"{category},{name},missing_file\n")  # FIX: Added label for clarity

        # finalize per-category accuracy
        m = results_per_cat["metrics"]
        m["pos_accuracy"] = safe_accuracy(m["correct_yes"], m["total_yes"])
        m["neg_accuracy"] = safe_accuracy(m["correct_no"], m["total_no"])
        # For PerVA we keep an overall aggregation and store per-category accuracy
        if dataset == "PerVA":
            overall["metrics"]["correct_yes"] += m["correct_yes"]
            overall["metrics"]["total_yes"] += m["total_yes"]
            overall["metrics"]["correct_no"] += m["correct_no"]
            overall["metrics"]["total_no"] += m["total_no"]
            overall["category"][category] = {"pos_accuracy": m["pos_accuracy"], "neg_accuracy": m["neg_accuracy"]}  # FIX: Fixed key name consistency

        # Keep the last (only) category block for non-PerVA output
        results_per_cat_last = results_per_cat

    # Determine final results object to save
    if dataset in ["YoLLaVA", "MyVLM", "DreamBooth"]:
        results = results_per_cat_last if results_per_cat_last is not None else {
            "metrics": {"correct_yes": 0, "total_yes": 0, "pos_accuracy": 0.0, "correct_no": 0, "total_no": 0, "neg_accuracy": 0.0},
            "concepts": {}
        }
    else:
        results = overall
        results["metrics"]["pos_accuracy"] = safe_accuracy(
            results["metrics"]["correct_yes"], results["metrics"]["total_yes"]
        )
        results["metrics"]["neg_accuracy"] = safe_accuracy(
            results["metrics"]["correct_no"], results["metrics"]["total_no"]
        )
    # Report and save
    print(f"total concepts: {in_total_concepts}")  # FIX: Added space after colon
    print(f"results showing for concepts: {concept_count}")  # FIX: Added space after colon
    report_json = f"recognition_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}.json"
    save_path = Path("results") / dataset_norm / report_json
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model at {save_path}")

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(results["metrics"])


if __name__ == "__main__":
    sys.exit(main())