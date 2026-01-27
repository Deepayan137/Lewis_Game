"""Shared utilities for data preparation scripts."""

from pathlib import Path
import json
from typing import Dict, Any, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file from disk."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save data to JSON file with pretty formatting."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def summarize_catalog(data: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Summarize a catalog structure.

    Returns:
        (n_concepts_total, total_train_images, total_test_images)
    """
    total_concepts = 0
    total_train_images = 0
    total_test_images = 0
    for category, concepts in data.items():
        if not isinstance(concepts, dict):
            continue
        total_concepts += len(concepts)
        for cname, splits in concepts.items():
            total_train_images += len(splits.get("train", []))
            total_test_images += len(splits.get("test", []))
    return total_concepts, total_train_images, total_test_images
