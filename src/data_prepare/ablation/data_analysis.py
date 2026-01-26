#!/usr/bin/env python3
"""
data_stats.py

Reads a dataset JSON file of the form:

{
  "bag": {
    "alx": {
      "train": [ "data/PerVA/train_/bag/alx/1.jpg", ...],
      "test": [ "data/PerVA/test_/bag/alx/1.jpg", ...]
    },
    "ash": {
      "train": [...],
      "test": [...]
    }
  },
  "clothe": {
    "train": [],
    "test": []
  },
  ...
}

and prints dataset statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def get_data_stats(data: Dict[str, Any]):
    """
    Compute statistics for the dataset JSON structure.
    """
    stats = {}
    for category, content in data.items():
        cat_stats = {"n_classes": 0, "train_count": 0, "test_count": 0, "classes": {}}

        # two possibilities:
        # 1. category -> dict of classes (with train/test keys)
        # 2. category -> dict with direct train/test lists
        if all(isinstance(v, dict) for v in content.values()):
            # case 1: multiple sub-classes
            for class_name, splits in content.items():
                train_count = len(splits.get("train", []))
                test_count = len(splits.get("test", []))
                cat_stats["classes"][class_name] = {
                    "train": train_count,
                    "test": test_count,
                    "total": train_count + test_count,
                }
                cat_stats["train_count"] += train_count
                cat_stats["test_count"] += test_count
            cat_stats["n_classes"] = len(content)
        else:
            # case 2: no sub-classes, just split lists
            train_count = len(content.get("train", []))
            test_count = len(content.get("test", []))
            cat_stats["classes"][category] = {
                "train": train_count,
                "test": test_count,
                "total": train_count + test_count,
            }
            cat_stats["train_count"] = train_count
            cat_stats["test_count"] = test_count
            cat_stats["n_classes"] = 1

        cat_stats["total_count"] = cat_stats["train_count"] + cat_stats["test_count"]
        stats[category] = cat_stats

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics from JSON file")
    parser.add_argument("json_file", type=str, help="Path to dataset JSON file")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stats = get_data_stats(data)

    print("\n=== Dataset Statistics ===")
    grand_train, grand_test, grand_total = 0, 0, 0
    for category, cat_stats in stats.items():
        print(f"\nCategory: {category}")
        print(f"  Classes: {cat_stats['n_classes']}")
        print(f"  Train:   {cat_stats['train_count']}")
        print(f"  Test:    {cat_stats['test_count']}")
        print(f"  Total:   {cat_stats['total_count']}")
        # if cat_stats["n_classes"] > 1:
        #     print("  Per-class breakdown:")
        #     for cname, cstats in cat_stats["classes"].items():
        #         print(f"    {cname}: train={cstats['train']}, test={cstats['test']}, total={cstats['total']}")

        grand_train += cat_stats["train_count"]
        grand_test += cat_stats["test_count"]
        grand_total += cat_stats["total_count"]

    print("\n=== Overall Totals ===")
    print(f"Train: {grand_train}")
    print(f"Test:  {grand_test}")
    print(f"Total: {grand_total}")


if __name__ == "__main__":
    main()
