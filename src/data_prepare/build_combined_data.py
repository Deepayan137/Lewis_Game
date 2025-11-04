
from pathlib import Path
import argparse
import json
import math
import random
from typing import Dict, Any, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def create_sampled_combined_splits(
    data: Dict[str, Any],
    concept_frac: float = 0.10,
    min_concepts_threshold: int = 10,
    test_sample_frac: float = 0.5,
    max_test_images: int = 10,
    seed: int = 23,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns (train_combined, test_combined, metadata)

    - train_combined: contains only the selected concepts; for each selected concept:
        - 'train' copied unchanged
        - 'test' is sampled subset of original test list (controlled by test_sample_frac and max_test_images)
    - test_combined: contains the remaining concepts unchanged
    - metadata: records selections and sampled test images per concept
    """
    if not (0.0 <= test_sample_frac <= 1.0):
        raise ValueError("test_sample_frac must be between 0 and 1")

    train_combined: Dict[str, Any] = {}
    test_combined: Dict[str, Any] = {}
    metadata = {
        "seed": seed,
        "concept_frac": concept_frac,
        "min_concepts_threshold": min_concepts_threshold,
        "test_sample_frac": test_sample_frac,
        "max_test_images": max_test_images,
        "categories": {},
    }

    for category, concepts in data.items():
        if not isinstance(concepts, dict):
            train_combined[category] = {}
            test_combined[category] = {}
            metadata["categories"][category] = {"note": "unexpected_structure", "n_concepts": 0}
            continue

        concept_names = sorted(list(concepts.keys()))
        n_concepts = len(concept_names)
        if n_concepts >= min_concepts_threshold:
            selected_count = math.ceil(concept_frac * n_concepts)
            # Deterministic per-category shuffle
            per_cat_rng = random.Random(seed + (hash(category) & 0xFFFFFFFF))
            shuffled = concept_names[:]
            per_cat_rng.shuffle(shuffled)
            selected = sorted(shuffled[:selected_count])
        else:
            selected_count = 0
            selected = []

        train_combined[category] = {}
        test_combined[category] = {}
        cat_meta = {
            "n_concepts": n_concepts,
            "selected_count": selected_count,
            "selected_concepts": selected,
            "sampled_test_images_by_concept": {},
        }

        # Build train_combined entries for selected concepts (sample test lists)
        for cname in selected:
            splits = concepts[cname]
            orig_train = list(splits.get("train", []))
            orig_test = list(splits.get("test", []))

            # Determine number to sample
            n_orig_test = len(orig_test)
            n_to_sample = math.floor(test_sample_frac * n_orig_test)
            # Bound by max_test_images
            # n_to_sample = min(n_to_sample, max_test_images)
            # If fraction=1 but n_to_sample==0 because floor -> allow at least 1 if orig_test>0 and test_sample_frac>0
            if test_sample_frac > 0 and n_to_sample == 0 and n_orig_test > 0:
                n_to_sample = 1

            sampled = []
            if n_to_sample > 0:
                per_concept_rng = random.Random(seed + (hash(category + cname) & 0xFFFFFFFF))
                tmp = orig_test[:]
                per_concept_rng.shuffle(tmp)
                sampled = tmp[:n_to_sample]

            # Keep train unchanged; test becomes sampled subset (may be empty)
            train_combined[category][cname] = {
                "train": orig_train+sampled,
                "test": [],
            }
            cat_meta["sampled_test_images_by_concept"][cname] = {
                "n_original_test": n_orig_test,
                "n_sampled": len(sampled),
                "sampled": sampled,
            }

        # Build test_combined entries: all other concepts unchanged
        for cname in concept_names:
            if cname in selected:
                continue
            splits = concepts[cname]
            test_combined[category][cname] = {
                "train": list(splits.get("train", [])),
                "test": list(splits.get("test", [])),
            }

        metadata["categories"][category] = cat_meta

    return train_combined, test_combined, metadata


def summarize_split(data: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Return (n_concepts_total, total_train_images, total_test_images)
    """
    total_concepts = 0
    total_train_images = 0
    total_test_images = 0
    categories = []
    for category, concepts in data.items():
        if not isinstance(concepts, dict):
            continue
        categories.append(category)
        total_concepts += len(concepts)
        for cname, splits in concepts.items():
            total_train_images += len(splits.get("train", []))
            total_test_images += len(splits.get("test", []))
    return total_concepts, total_train_images, total_test_images


def main():
    parser = argparse.ArgumentParser(description="Create train/test combined concept files with sampled test images in train file.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to original dataset JSON")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--concept_frac", type=float, default=0.20, help="Fraction of concepts to select for large categories")
    parser.add_argument("--min_concepts_threshold", type=int, default=8, help="Apply selection only to categories with at least this many concepts")
    parser.add_argument("--test_sample_frac", type=float, default=1., help="Fraction (0-1) of original test images to include per selected concept in train_combined")
    parser.add_argument("--max_test_images", type=int, default=100, help="Hard cap on sampled test images per selected concept")
    parser.add_argument("--seed", type=int, default=23, help="Random seed for deterministic selection/sampling")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(input_path)

    train_combined, test_combined, metadata = create_sampled_combined_splits(
        data,
        concept_frac=args.concept_frac,
        min_concepts_threshold=args.min_concepts_threshold,
        test_sample_frac=args.test_sample_frac,
        max_test_images=args.max_test_images,
        seed=args.seed,
    )

    train_name = out_dir / f"train_combined_seed_{args.seed}.json"
    test_name = out_dir / f"test_combined_seed_{args.seed}.json"
    meta_name = out_dir / f"train_test_combined_metadata_seed_{args.seed}.json"

    save_json(train_combined, train_name)
    save_json(test_combined, test_name)
    save_json(metadata, meta_name)

    # Print summaries
    train_concepts, train_tcount, train_testcount = summarize_split(train_combined)
    test_concepts, test_tcount, test_testcount = summarize_split(test_combined)

    print("\n=== Output summary ===")
    print(f"Train combined file: {train_name}")
    print(f"  Catgories: {[c for c in train_combined.keys() if train_combined[c]]}")
    print(f"  Categories with selected concepts: {len([c for c in train_combined.keys() if train_combined[c]])}")
    print(f"  Selected (train) concepts total: {train_concepts}")
    print(f"  Train images in train_combined (copied from original train lists): {train_tcount}")
    print(f"  Test images in train_combined (sampled subset): {train_testcount}")

    print(f"\nTest combined file: {test_name}")
    print(f"  Remaining concepts total: {test_concepts}")
    print(f"  Train images in test_combined: {test_tcount}")
    print(f"  Test images in test_combined: {test_testcount}")

    print(f"\nMetadata saved to: {meta_name}")
    print("Notes:")
    print(" - train_combined contains only the selected concepts; their 'test' lists are sampled (no images moved).")
    print(" - test_combined contains the remaining concepts with unchanged lists.")
    print(" - Sampling is deterministic using the provided seed.")
    print(" - This script does NOT alter your original JSON file.")

if __name__ == "__main__":
    main()
