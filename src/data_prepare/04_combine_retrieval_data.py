"""
Combine retrieval data from multiple categories into a single file.

This script is designed to run AFTER all per-category retrieval jobs
(03_build_retrieval_per_category.py) have completed.
"""

import os
import argparse
import random
import json
from pathlib import Path

from utils import load_json, save_json


# Default categories for PerVA dataset
PERVA_CATEGORIES = [
    'bag', 'book', 'clothe', 'cup', 'decoration',
    'pillow', 'plant', 'retail', 'toy', 'tumbler', 'veg'
]


def get_categories_from_catalog(catalog_path: str) -> list:
    """Extract category names from a catalog JSON file."""
    catalog = load_json(Path(catalog_path))
    return sorted(catalog.keys())


def combine_retrieval_data(
    input_dir: str,
    categories: list,
    seed: int,
    input_filename: str,
    num_samples: int = None,
) -> list:
    """
    Combine retrieval data from multiple category directories.

    Args:
        input_dir: Base directory containing category subdirectories
        categories: List of category names to combine
        seed: Seed used in retrieval (for locating files)
        input_filename: Name of the retrieval JSON file in each category
        num_samples: If set, randomly sample this many entries from combined data

    Returns:
        Combined (and optionally sampled) retrieval data
    """
    all_data = []
    found_categories = []
    missing_categories = []

    for category in categories:
        path = Path(input_dir) / category / f'seed_{seed}' / input_filename
        if path.exists():
            data = load_json(path)
            all_data.extend(data)
            found_categories.append(category)
            print(f"  Loaded {len(data)} entries from {category}")
        else:
            missing_categories.append(category)
            print(f"  Warning: File not found for category '{category}': {path}")

    print(f"\nFound {len(found_categories)}/{len(categories)} categories")
    print(f"Total entries before sampling: {len(all_data)}")

    if missing_categories:
        print(f"Missing categories: {missing_categories}")

    # Sample if requested
    if num_samples is not None and num_samples < len(all_data):
        random.seed(seed)
        all_data = random.sample(all_data, num_samples)
        print(f"Sampled {num_samples} entries")

    return all_data


def main():
    parser = argparse.ArgumentParser(
        description="Combine retrieval data from multiple categories into a single file."
    )
    parser.add_argument(
        '--input_dir', type=str, default='outputs/PerVA',
        help='Base directory containing category subdirectories (default: outputs/PerVA)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory. If not set, uses {input_dir}/all/seed_{seed}/'
    )
    parser.add_argument(
        '--categories', type=str, nargs='+', default=None,
        help='List of categories to combine. If not set, uses default PerVA categories.'
    )
    parser.add_argument(
        '--catalog', type=str, default=None,
        help='Path to catalog JSON to extract categories from (alternative to --categories)'
    )
    parser.add_argument(
        '--input_filename', type=str, default='retrieval_top3.json',
        help='Name of retrieval JSON file in each category directory (default: retrieval_top3.json)'
    )
    parser.add_argument(
        '--output_filename', type=str, default=None,
        help='Output filename. If not set, auto-generated based on input and num_samples.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Seed used in retrieval jobs (for locating files) (default: 42)'
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Number of samples to randomly select from combined data. If not set, keeps all.'
    )
    args = parser.parse_args()

    # Determine categories
    if args.catalog:
        categories = get_categories_from_catalog(args.catalog)
        print(f"Loaded {len(categories)} categories from catalog: {args.catalog}")
    elif args.categories:
        categories = args.categories
    else:
        categories = PERVA_CATEGORIES
        print(f"Using default PerVA categories: {categories}")

    # Combine data
    print(f"\nCombining retrieval data from: {args.input_dir}")
    combined_data = combine_retrieval_data(
        input_dir=args.input_dir,
        categories=categories,
        seed=args.seed,
        input_filename=args.input_filename,
        num_samples=args.num_samples,
    )

    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input_dir) / 'all' / f'seed_{args.seed}'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if args.output_filename:
        output_filename = args.output_filename
    else:
        # Auto-generate based on input filename and num_samples
        base = Path(args.input_filename).stem
        if args.num_samples:
            output_filename = f"{base}_combined_{args.num_samples}.json"
        else:
            output_filename = f"{base}_combined_all.json"

    output_path = output_dir / output_filename

    # Save
    save_json(combined_data, output_path)
    print(f"\nSaved {len(combined_data)} entries to: {output_path}")


if __name__ == "__main__":
    main()
