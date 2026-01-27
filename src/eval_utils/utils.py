#!/usr/bin/env python3
"""
Shared utilities for evaluation metric aggregation.

This module provides common functions used by both identification (precision/recall/F1)
and recognition (yes/no accuracy) evaluation scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================================
# Constants
# =============================================================================

PERVA_CATEGORIES = [
    "bag", "book", "bottle", "bowl", "clothe", "cup", "decoration", "headphone",
    "pillow", "plant", "plate", "remote", "retail", "telephone", "tie", "towel",
    "toy", "tro_bag", "tumbler", "umbrella", "veg"
]

SINGLE_CATEGORY_DATASETS = ["YoLLaVA", "MyVLM", "DreamBooth"]


# =============================================================================
# Math Utilities
# =============================================================================

def safe_div(numerator: float, denominator: float) -> float:
    """Safe division that returns 0.0 when denominator is zero."""
    return (numerator / denominator) if denominator > 0 else 0.0


def safe_percentage(numerator: float, denominator: float) -> float:
    """Safe division returning percentage (0-100)."""
    return safe_div(numerator, denominator) * 100.0


# =============================================================================
# File System Utilities
# =============================================================================

def scan_concepts(base: Path) -> List[str]:
    """
    Discover concept names under `base`, keeping only non-empty directories.

    Args:
        base: Directory to scan for concept subdirectories

    Returns:
        Sorted list of concept directory names
    """
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and any(d.iterdir())
    )


def load_json(path: Path) -> Any:
    """Load and return JSON data from a file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON to a file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_records(json_path: Path) -> List[dict]:
    """
    Load JSON file and extract list of records.

    Handles various JSON structures:
    - Direct list: returns as-is
    - Dict with 'records', 'data', 'items', 'predictions', or 'results' key

    Args:
        json_path: Path to JSON file

    Returns:
        List of record dictionaries
    """
    data = load_json(json_path)
    if isinstance(data, list):
        return data
    for key in ("records", "data", "items", "predictions", "results"):
        if isinstance(data, dict) and isinstance(data.get(key), list):
            return data[key]
    return []


# =============================================================================
# Logging Utilities
# =============================================================================

def log_debug(message: str, debug_file: str = "debug.txt") -> None:
    """Append a debug message to the debug file and print it."""
    print(message)
    with open(debug_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


# =============================================================================
# Argument Parsing Utilities
# =============================================================================

def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add common evaluation arguments to a parser."""
    parser.add_argument(
        "--dataset",
        default="YoLLaVA",
        choices=["YoLLaVA", "MyVLM", "DreamBooth", "PerVA"],
        help="Dataset name"
    )
    parser.add_argument("--db_type", default="original", help="Database type")
    parser.add_argument("--model_type", default="original_7b", help="Model type identifier")
    parser.add_argument("--seed", type=int, default=23, help="Random seed used in evaluation")
    parser.add_argument("--category", default="all", help="Category (for compatibility, not used when scanning)")
    parser.add_argument("--out", type=str, default=None, help="Output path (for compatibility)")


def get_categories(dataset: str) -> List[str]:
    """Get the list of categories for a dataset."""
    if dataset == "PerVA":
        return PERVA_CATEGORIES
    return ["all"]


# =============================================================================
# Path Utilities
# =============================================================================

def get_results_base_path(dataset: str, category: str) -> Path:
    """Get the base path for results of a dataset/category combination."""
    return Path("results") / dataset / category


def get_concept_result_path(
    base: Path,
    concept: str,
    seed: int,
    filename: str
) -> Path:
    """
    Construct the path to a concept's result file.

    Args:
        base: Base results directory (results/dataset/category)
        concept: Concept name
        seed: Random seed
        filename: Result filename (e.g., 'results_model_X_db_Y_k_Z.json')

    Returns:
        Full path to the concept result file
    """
    return base / concept / f"seed_{seed}" / filename
