"""
Compute recognition metrics for different methods across datasets.

This script computes overall, in-category, and cross-category recognition
accuracy for methods like R2P, RePIC, RAP, and custom approaches.

Usage:
    python compute_recognition_metrics.py \
        --method R2P \
        --dataset YoLLaVA \
        --results_dir results_R2P/QWEN_YoLLaVA_seed_23/all \
        --file_identifier reco_results.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ============================================================================
# Category Mappings (copied from src/inference_utils/common.py)
# ============================================================================

DREAMBOOTH_CATEGORY_MAP = {
    "backpack": 'bag',
    'backpack_dog': 'bag',
    'bear_plushie': 'toy',
    'berry_bowl': 'household object',
    'can': 'household object',
    'candle': 'household object',
    'cat': 'pet animal',
    'cat2': 'pet animal',
    'clock': 'household object',
    'colorful_sneaker': 'shoe',
    'dog': 'pet animal',
    'dog2': 'pet animal',
    'dog3': 'pet animal',
    'dog5': 'pet animal',
    'dog6': 'pet animal',
    'dog7': 'pet animal',
    'dog8': 'pet animal',
    'duck_toy': 'toy',
    'fancy_boot': 'shoe',
    'grey_sloth_plushie': 'toy',
    'monster_toy': 'toy',
    'pink_sunglasses': 'glasses',
    'poop_emoji': 'toy',
    'rc_car': 'toy',
    'red_cartoon': 'cartoon character',
    'robot_toy': 'toy',
    'shiny_sneaker': 'shoe',
    'teapot': 'household object',
    'vase': 'household object',
    'wolf_plushie': 'toy'
}

YOLLAVA_CATEGORY_MAP = {
    'ciin': 'person',
    'denisdang': 'person',
    'khanhvy': 'person',
    'oong': 'person',
    'phuc-map': 'person',
    'thao': 'person',
    'thuytien': 'person',
    'viruss': 'person',
    'yuheng': 'person',
    'willinvietnam': 'person',
    'chua-thien-mu': 'building',
    'nha-tho-hanoi': 'building',
    'nha-tho-hcm': 'building',
    'thap-but': 'building',
    'thap-cham': 'building',
    'dug': 'cartoon character',
    'fire': 'cartoon character',
    'marie-cat': 'cartoon character',
    'toodles-galore': 'cartoon character',
    'water': 'cartoon character',
    'bo': 'pet animal',
    'butin': 'pet animal',
    'henry': 'pet animal',
    'mam': 'pet animal',
    'mydieu': 'pet animal',
    'shiba-yellow': 'toy',
    'pusheen-cup': 'mug',
    'neurips-cup': 'toy',
    'tokyo-keyboard': 'electronic',
    'cat-cup': 'cup',
    'brown-duck': 'toy',
    'lamb': 'toy',
    'duck-banana': 'toy',
    'shiba-black': 'toy',
    'pig-cup': 'cup',
    'shiba-sleep': 'toy',
    'yellow-duck': 'toy',
    'elephant': 'toy',
    'shiba-gray': 'toy',
    'dragon': 'toy'
}

MYVLM_CATEGORY_MAP = {
    'asian_doll': 'toy',
    'boy_funko_pop': 'toy',
    'bull': 'figurine',
    'cat_statue': 'figurine',
    'ceramic_head': 'figurine',
    'chicken_bean_bag': 'toy',
    'colorful_teapot': 'tea pot',
    'dangling_child': 'toy',
    'elephant_sphere': 'figurine',
    'elephant_statue': 'figurine',
    'espresso_cup': 'cup',
    'gengar_toy': 'toy',
    'gold_pineapple': 'household object',
    'iverson_funko_pop': 'toy',
    'green_doll': 'toy',
    'maeve_dog': 'pet animal',
    'minion_toy': 'toy',
    'rabbit_toy': 'toy',
    'red_chicken': 'figurine',
    'red_piggy_bank': 'piggy bank',
    'robot_toy': 'toy',
    'running_shoes': 'shoe',
    'sheep_pillow': 'pillow',
    'sheep_plush': 'toy',
    'sheep_toy': 'toy',
    'skulls_mug': 'mug',
    'small_penguin': 'toy',
    'billy_dog': 'pet animal',
    'my_cat': 'pet animal'
}

PERVA_CATEGORY_MAP = {
    'veg': 'vegetable',
    'decoration': 'decoration object',
    'retail': 'retail object',
    'tro_bag': 'trolley bag',
}

DATASET_CATEGORY_MAPS = {
    'DreamBooth': DREAMBOOTH_CATEGORY_MAP,
    'YoLLaVA': YOLLAVA_CATEGORY_MAP,
    'MyVLM': MYVLM_CATEGORY_MAP,
    'PerVA': PERVA_CATEGORY_MAP,
}


def get_category_for_concept(concept_name: str, dataset: str) -> str:
    """
    Get the category for a given concept name in a dataset.

    Args:
        concept_name: The concept/object name
        dataset: Dataset name (YoLLaVA, MyVLM, PerVA, DreamBooth)

    Returns:
        Category string, or concept_name as fallback
    """
    category_map = DATASET_CATEGORY_MAPS.get(dataset, {})
    return category_map.get(concept_name, concept_name)


def extract_concept_from_path(image_path: str) -> str:
    """
    Extract concept name from image path.

    Example:
        'data/YoLLaVA/test/all/bo/0.png' -> 'bo'
        'data/YoLLaVA/test/all/thao/0.png' -> 'thao'
    """
    path = Path(image_path)
    # The concept is the parent directory of the image file
    return path.parent.name


def extract_concept_from_info(info_text: str) -> str:
    """
    Extract concept name from info text (RAP/RePIC format).

    Example:
        '1. Name: <bo>, Info: ...' -> 'bo'
        '1. Name: <brown-duck>, Info: ...' -> 'brown-duck'
    """
    # Pattern to match: Name: <concept-name>
    match = re.search(r'Name:\s*<([^>]+)>', info_text)
    if match:
        return match.group(1)
    return None


def process_rap_entry(entry: Dict[str, Any], dataset: str) -> Tuple[str, str, str, str]:
    """
    Process RAP format entry.

    Returns:
        (query_concept, query_category, ref_concept, ref_category)
    """
    if 'imge_path' not in entry:
        query_path = entry['query_path']
    else:
        query_path = entry['image_path']
    query_concept = extract_concept_from_path(query_path)
    query_category = get_category_for_concept(query_concept, dataset)

    info_text = entry['info']
    ref_concept = extract_concept_from_info(info_text)
    ref_category = get_category_for_concept(ref_concept, dataset)

    return query_concept, query_category, ref_concept, ref_category


def process_repic_entry(entry: Dict[str, Any], dataset: str) -> Tuple[str, str, str, str]:
    """
    Process RePIC format entry (same as RAP).

    Returns:
        (query_concept, query_category, ref_concept, ref_category)
    """
    return process_rap_entry(entry, dataset)


def process_r2p_entry(entry: Dict[str, Any], dataset: str) -> Tuple[str, str, str, str]:
    """
    Process R2P format entry.

    Returns:
        (query_concept, query_category, ref_concept, ref_category)
    """
    query_path = entry['image']
    query_concept = extract_concept_from_path(query_path)
    query_category = get_category_for_concept(query_concept, dataset)

    # Use first retrieved concept
    ref_concept = entry['ret_concepts'][0] if entry['ret_concepts'] else None
    ref_category = get_category_for_concept(ref_concept, dataset) if ref_concept else None

    return query_concept, query_category, ref_concept, ref_category


def process_ours_entry(entry: Dict[str, Any], dataset: str) -> Tuple[str, str, str, str]:
    """
    Process 'ours' format entry (adjust based on your format).

    Returns:
        (query_concept, query_category, ref_concept, ref_category)
    """
    # Assuming similar format to RAP/RePIC for now
    # Adjust this based on your actual format
    return process_rap_entry(entry, dataset)


PROCESSORS = {
    'RAP': process_rap_entry,
    'RePIC': process_repic_entry,
    'R2P': process_r2p_entry,
    'ours': process_ours_entry,
}


def compute_metrics(results_dir: Path, file_identifier: str, method: str, dataset: str, seed: int = None) -> Dict[str, Any]:
    """
    Compute recognition metrics across all concepts.

    Args:
        results_dir: Directory containing per-concept result folders
        file_identifier: JSON filename to read (e.g., 'reco_results.json')
        method: Method name (R2P, RePIC, RAP, ours)
        dataset: Dataset name (YoLLaVA, MyVLM, PerVA, DreamBooth)
        seed: Optional seed number for methods with seed subdirectories

    Returns:
        Dictionary with metrics
    """
    if method not in PROCESSORS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(PROCESSORS.keys())}")

    processor = PROCESSORS[method]

    # Metrics tracking
    total_samples = 0
    total_correct = 0

    # Positive/Negative accuracy tracking
    positive_samples = 0
    positive_correct = 0
    negative_samples = 0
    negative_correct = 0

    in_category_samples = 0
    in_category_correct = 0
    in_category_wrong = 0

    cross_category_samples = 0
    cross_category_correct = 0
    cross_category_wrong = 0

    # Iterate through all concept folders
    for concept_dir in sorted(results_dir.iterdir()):
        if not concept_dir.is_dir():
            continue

        # Try to find the JSON file in multiple possible locations
        json_file = None

        # Format 1: Direct (R2P style)
        # results_R2P/QWEN_YoLLaVA_seed_23/all/{concept}/{file_identifier}
        candidate1 = concept_dir / file_identifier
        if candidate1.exists():
            json_file = candidate1

        # Format 2: With seed subdirectory (RePIC/RAP/ours style)
        # results_{method}/{dataset}/all/{concept}/seed_{seed}/{file_identifier}
        elif seed is not None:
            candidate2 = concept_dir / f"seed_{seed}" / file_identifier
            if candidate2.exists():
                json_file = candidate2

        # Format 3: Auto-detect seed subdirectory
        if json_file is None:
            # Look for any seed_* subdirectory
            seed_dirs = list(concept_dir.glob("seed_*"))
            if seed_dirs:
                # Use the first seed directory found
                candidate3 = seed_dirs[0] / file_identifier
                if candidate3.exists():
                    json_file = candidate3

        if json_file is None:
            print(f"Warning: {file_identifier} not found in {concept_dir}, skipping...")
            continue

        # Load results
        with json_file.open('r') as f:
            data = json.load(f)

        results = data.get('results', [])

        for entry in results:
            # Extract concepts and categories
            query_concept, query_category, ref_concept, ref_category = processor(entry, dataset)

            # Get prediction and ground truth
            pred = entry.get('pred', '').lower().strip()
            solution = entry.get('solution', '').lower().strip()

            # Skip if missing data
            if not pred or not solution or not query_category or not ref_category:
                continue

            # Check correctness
            is_correct = (pred == solution)

            # Determine if in-category or cross-category
            is_in_category = (query_category == ref_category)

            # Update counters
            total_samples += 1
            if is_correct:
                total_correct += 1

            # Track positive/negative accuracy
            if solution == "yes":
                positive_samples += 1
                if is_correct:
                    positive_correct += 1
            else:  # solution == "no"
                negative_samples += 1
                if is_correct:
                    negative_correct += 1

            if is_in_category:
                in_category_samples += 1
                if is_correct:
                    in_category_correct += 1
                else:
                    in_category_wrong += 1
            else:
                cross_category_samples += 1
                if is_correct:
                    cross_category_correct += 1
                else:
                    cross_category_wrong += 1

    # Compute accuracies
    positive_accuracy = positive_correct / positive_samples if positive_samples > 0 else 0.0
    negative_accuracy = negative_correct / negative_samples if negative_samples > 0 else 0.0
    weighted_accuracy = (positive_accuracy + negative_accuracy) / 2.0  # Balanced accuracy
    in_category_accuracy = in_category_correct / in_category_samples if in_category_samples > 0 else 0.0
    cross_category_accuracy = cross_category_correct / cross_category_samples if cross_category_samples > 0 else 0.0

    return {
        "method": method,
        "dataset": dataset,
        "positive_accuracy": round(positive_accuracy, 4),
        "negative_accuracy": round(negative_accuracy, 4),
        "weighted_accuracy": round(weighted_accuracy, 4),
        "in_category_accuracy": round(in_category_accuracy, 4),
        "cross_category_accuracy": round(cross_category_accuracy, 4),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "total_wrong": total_samples - total_correct,
        "positive_samples": positive_samples,
        "positive_correct": positive_correct,
        "positive_wrong": positive_samples - positive_correct,
        "negative_samples": negative_samples,
        "negative_correct": negative_correct,
        "negative_wrong": negative_samples - negative_correct,
        "in_category_samples": in_category_samples,
        "in_category_correct": in_category_correct,
        "in_category_wrong": in_category_wrong,
        "cross_category_samples": cross_category_samples,
        "cross_category_correct": cross_category_correct,
        "cross_category_wrong": cross_category_wrong,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute recognition metrics for different methods"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=['R2P', 'RePIC', 'RAP', 'ours'],
        help="Method name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['YoLLaVA', 'MyVLM', 'PerVA', 'DreamBooth'],
        help="Dataset name"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to directory containing per-concept results"
    )
    parser.add_argument(
        "--file_identifier",
        type=str,
        required=True,
        help="JSON filename to read (e.g., 'reco_results.json')"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output filename (default: {method}_{dataset}_metrics.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed number for methods with seed subdirectories (optional)"
    )

    args = parser.parse_args()

    # Compute metrics
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    metrics = compute_metrics(
        results_dir=results_dir,
        file_identifier=args.file_identifier,
        method=args.method,
        dataset=args.dataset,
        seed=args.seed
    )

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Save metrics
    output_name = args.output_name or f"{args.method}_{args.dataset}_metrics.json"
    output_path = reports_dir / output_name

    with output_path.open('w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Recognition Metrics: {args.method} on {args.dataset}")
    print(f"{'='*60}")
    print(f"Positive Accuracy:       {metrics['positive_accuracy']:.2%} ({metrics['positive_correct']}/{metrics['positive_samples']})")
    print(f"Negative Accuracy:       {metrics['negative_accuracy']:.2%} ({metrics['negative_correct']}/{metrics['negative_samples']})")
    print(f"Weighted Accuracy:       {metrics['weighted_accuracy']:.2%}")
    print(f"In-Category Accuracy:    {metrics['in_category_accuracy']:.2%} ({metrics['in_category_correct']}/{metrics['in_category_samples']})")
    print(f"Cross-Category Accuracy: {metrics['cross_category_accuracy']:.2%} ({metrics['cross_category_correct']}/{metrics['cross_category_samples']})")
    print(f"\nBreakdown:")
    print(f"  Positive:       {metrics['positive_correct']} correct, {metrics['positive_wrong']} wrong")
    print(f"  Negative:       {metrics['negative_correct']} correct, {metrics['negative_wrong']} wrong")
    print(f"  In-Category:    {metrics['in_category_correct']} correct, {metrics['in_category_wrong']} wrong")
    print(f"  Cross-Category: {metrics['cross_category_correct']} correct, {metrics['cross_category_wrong']} wrong")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
