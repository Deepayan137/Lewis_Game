#!/usr/bin/env python3
"""
Post-process a retrieval JSON file to inject cross-category distractors.

For each entry, with probability `--cross_cat_prob`, ALL distractor slots
(i.e. every ret_paths position that is NOT the label/correct answer) are
replaced with (path, desc) pairs sampled from a DIFFERENT category.

The correct answer (at index `label`) is never touched.
The `label` field therefore remains valid after processing.

Pool construction: only distractor slots from other entries are added to the
pool, so GT-concept images are never accidentally used as distractors.

Usage examples:
  # 50% of entries get cross-category distractors
  python add_cross_category_distractors.py \\
      --input  outputs/PerVA/all/seed_23/retrieval_top3_subset_30_original_7b_sampled_500.json \\
      --cross_cat_prob 0.5

  # all entries get cross-category distractors (simulates easy_1.0)
  python add_cross_category_distractors.py \\
      --input  outputs/PerVA/all/seed_23/retrieval_top3_subset_30_original_7b_sampled_500.json \\
      --cross_cat_prob 1.0 --seed 42

  # explicit output path
  python add_cross_category_distractors.py \\
      --input  outputs/PerVA/all/seed_23/retrieval_top3_subset_30_original_7b_sampled_500.json \\
      --cross_cat_prob 0.5 --out my_custom_output.json
"""

import json
import random
import argparse
import copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def build_distractor_pool(
    data: List[dict],
) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    """
    Build a pool of (path, desc_or_None) pairs for each category,
    using only the distractor slots (non-label positions) of each entry.

    Returns:
        pool[category] = [(path, desc), ...]
    """
    pool: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)

    for entry in data:
        cat   = entry.get('category', '')
        label = entry.get('label', -1)
        paths = entry.get('ret_paths', [])
        descs = entry.get('ret_descs', None)  # may be absent

        for i, path in enumerate(paths):
            if i == label:
                continue  # skip the correct answer
            desc = descs[i] if (descs is not None and i < len(descs)) else None
            pool[cat].append((path, desc))

    return pool


def apply_cross_category_noise(
    data: List[dict],
    pool: Dict[str, List[Tuple[str, Optional[str]]]],
    cross_cat_prob: float,
    rng: random.Random,
) -> Tuple[List[dict], int]:
    """
    For each entry, with probability cross_cat_prob replace all distractor
    slots with samples drawn from a different category's pool.

    Returns:
        (modified_data, num_entries_modified)
    """
    all_categories = list(pool.keys())
    modified = 0

    result = copy.deepcopy(data)

    for entry in result:
        if rng.random() >= cross_cat_prob:
            continue  # keep original distractors

        cat   = entry.get('category', '')
        label = entry.get('label', -1)
        paths = entry.get('ret_paths', [])
        has_descs = 'ret_descs' in entry
        descs = entry.get('ret_descs', [None] * len(paths))

        # Candidate categories: everything except this entry's own category
        other_cats = [c for c in all_categories if c != cat and pool[c]]
        if not other_cats:
            continue  # no other categories available, skip

        swapped = False
        for i in range(len(paths)):
            if i == label:
                continue  # never touch the correct answer

            # Sample a random other category, then a random item from it
            chosen_cat  = rng.choice(other_cats)
            chosen_path, chosen_desc = rng.choice(pool[chosen_cat])

            paths[i] = chosen_path
            if has_descs and chosen_desc is not None:
                descs[i] = chosen_desc
            swapped = True

        if swapped:
            entry['ret_paths'] = paths
            if has_descs:
                entry['ret_descs'] = descs
            modified += 1

    return result, modified


def derive_output_path(input_path: Path, cross_cat_prob: float) -> Path:
    """
    Auto-generate output filename by appending _cross_{prob} before .json.
    e.g. retrieval_top3_..._sampled_500.json
      -> retrieval_top3_..._sampled_500_cross_0.5.json
    """
    prob_str = str(cross_cat_prob).rstrip('0').rstrip('.')  # '0.50' -> '0.5'
    stem = input_path.stem
    return input_path.parent / f'{stem}_cross_{prob_str}.json'


def main():
    parser = argparse.ArgumentParser(
        description='Inject cross-category distractors into a retrieval JSON.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to the input retrieval JSON file.',
    )
    parser.add_argument(
        '--cross_cat_prob', type=float, required=True,
        help='Probability (0.0–1.0) that an entry gets cross-category distractors.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).',
    )
    parser.add_argument(
        '--out', default=None,
        help='Output file path. If omitted, auto-generated from input filename.',
    )
    args = parser.parse_args()

    if not (0.0 <= args.cross_cat_prob <= 1.0):
        parser.error('--cross_cat_prob must be between 0.0 and 1.0')

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'Error: input file not found: {input_path}')
        return

    output_path = Path(args.out) if args.out else derive_output_path(
        input_path, args.cross_cat_prob)

    # ── Load ────────────────────────────────────────────────────────────────
    with open(input_path, 'r') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} entries from {input_path}')

    # ── Summarise categories in the file ────────────────────────────────────
    from collections import Counter
    cat_counts = Counter(e.get('category', 'unknown') for e in data)
    print('Category distribution:')
    for cat, cnt in sorted(cat_counts.items()):
        print(f'  {cat}: {cnt} entries')
    print(f'Has ret_descs: {any("ret_descs" in e for e in data)}')

    # ── Build pool ──────────────────────────────────────────────────────────
    pool = build_distractor_pool(data)
    print('\nDistractor pool sizes per category:')
    for cat, items in sorted(pool.items()):
        print(f'  {cat}: {len(items)} items')

    # ── Apply noise ─────────────────────────────────────────────────────────
    rng = random.Random(args.seed)
    result, n_modified = apply_cross_category_noise(
        data, pool, args.cross_cat_prob, rng)

    # ── Save ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nDone.')
    print(f'  Entries total:    {len(result)}')
    print(f'  Entries modified: {n_modified}  '
          f'({100 * n_modified / max(len(result), 1):.1f}%)')
    print(f'  cross_cat_prob:   {args.cross_cat_prob}')
    print(f'  seed:             {args.seed}')
    print(f'  Output:           {output_path}')


if __name__ == '__main__':
    main()
