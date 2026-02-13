#!/usr/bin/env python3
"""
test_dataset.py

Analyze the distribution of positive/negative samples and cross-category vs within-category
negatives in the HF dataset.

Usage:
    python tests/test_dataset.py
"""

from datasets import DatasetDict
from collections import defaultdict
import argparse


def analyze_dataset(dataset_path: str):
    """
    Analyze dataset to count:
    - Positive samples (same concept)
    - Negative samples (different concepts)
      - Within-category negatives
      - Cross-category negatives
    """
    print(f"Loading dataset from: {dataset_path}")
    ds = DatasetDict.load_from_disk(dataset_path)

    # Get the train split
    train_ds = ds['train']

    # Statistics counters
    stats = {
        'total': 0,
        'positive': 0,
        'negative': 0,
        'negative_within_category': 0,
        'negative_cross_category': 0,
    }

    # Track per-category statistics
    category_stats = defaultdict(lambda: {
        'positive': 0,
        'negative_within': 0,
        'negative_cross': 0
    })

    # Track cross-category pairs
    cross_category_pairs = defaultdict(int)

    print("Analyzing samples...")

    # Analyze each sample
    for sample in train_ds:
        stats['total'] += 1

        # Extract concept names from paths
        query_path = sample['query_path']
        ref_path = sample['reference_path']

        query_concept = query_path.split('/')[-2]
        ref_concept = ref_path.split('/')[-2]

        query_category = query_path.split('/')[-3]
        ref_category = ref_path.split('/')[-3]

        # Check if positive (same concept)
        is_positive = (query_concept == ref_concept)

        if is_positive:
            stats['positive'] += 1
            category_stats[query_category]['positive'] += 1
        else:
            stats['negative'] += 1

            # Check if same category or cross-category
            if query_category == ref_category:
                stats['negative_within_category'] += 1
                category_stats[query_category]['negative_within'] += 1
            else:
                stats['negative_cross_category'] += 1
                category_stats[query_category]['negative_cross'] += 1

                # Track which category pairs appear
                pair_key = tuple(sorted([query_category, ref_category]))
                cross_category_pairs[pair_key] += 1

    # Print overall statistics
    print("\n" + "="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    print(f"\nTotal samples: {stats['total']}")
    print(f"\nPositive samples (same concept): {stats['positive']} ({stats['positive']/stats['total']*100:.2f}%)")
    print(f"Negative samples (different concept): {stats['negative']} ({stats['negative']/stats['total']*100:.2f}%)")

    if stats['negative'] > 0:
        print(f"\nNegative breakdown:")
        print(f"  - Within-category: {stats['negative_within_category']} ({stats['negative_within_category']/stats['negative']*100:.2f}% of negatives)")
        print(f"  - Cross-category:  {stats['negative_cross_category']} ({stats['negative_cross_category']/stats['negative']*100:.2f}% of negatives)")

    # Print per-category statistics
    print("\n" + "="*80)
    print("PER-CATEGORY BREAKDOWN")
    print("="*80)
    for category, cat_stats in sorted(category_stats.items()):
        total_cat = cat_stats['positive'] + cat_stats['negative_within'] + cat_stats['negative_cross']
        print(f"\n{category}:")
        print(f"  Total: {total_cat}")
        print(f"  Positive: {cat_stats['positive']} ({cat_stats['positive']/total_cat*100:.1f}%)")
        print(f"  Negative (within-category): {cat_stats['negative_within']} ({cat_stats['negative_within']/total_cat*100:.1f}%)")
        print(f"  Negative (cross-category):  {cat_stats['negative_cross']} ({cat_stats['negative_cross']/total_cat*100:.1f}%)")

    # Print cross-category pair statistics
    if cross_category_pairs:
        print("\n" + "="*80)
        print("CROSS-CATEGORY PAIR DISTRIBUTION")
        print("="*80)
        print(f"\nTotal unique category pairs: {len(cross_category_pairs)}")
        print("\nMost common cross-category pairs:")
        for (cat1, cat2), count in sorted(cross_category_pairs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat1} <-> {cat2}: {count} samples")

    # Verify with listener_solution field
    print("\n" + "="*80)
    print("VERIFICATION WITH listener_solution FIELD")
    print("="*80)
    yes_count = sum(1 for sample in train_ds if sample['listener_solution'] == 'yes')
    no_count = sum(1 for sample in train_ds if sample['listener_solution'] == 'no')
    print(f"listener_solution='yes': {yes_count}")
    print(f"listener_solution='no':  {no_count}")

    matches = (yes_count == stats['positive'] and no_count == stats['negative'])
    print(f"\nCounts match concept-based analysis: {'✓ YES' if matches else '✗ NO'}")

    if not matches:
        print("WARNING: Mismatch detected! Check dataset consistency.")

    print("\n" + "="*80)

    return stats, category_stats, cross_category_pairs


def main():
    parser = argparse.ArgumentParser(description="Analyze HF dataset statistics")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="share_data/PerVA_seed_23_K_3_subset_30_sampled_500reco_no_desc",
        help="Path to HF dataset directory"
    )

    args = parser.parse_args()

    analyze_dataset(args.dataset_path)


if __name__ == "__main__":
    main()
