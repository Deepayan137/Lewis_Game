#!/usr/bin/env python3
"""
select_concepts_subset.py

Select a subset of concepts from a JSON of structure:
{
  "bag": {
    "bkq": { "train": ["file1", "file2", ...] },
    "xyz": { "train": [...] }
  },
  "headphone": {
    "concept1": { "train": [...] }, ...
  },
  ...
}

Goal: pick k concepts (e.g., k=20 or k=30) so that the total number of images across
selected concepts is as close as possible to target_images (e.g., 1000).

Outputs:
 - subset_{k}_best.json  (same nested structure but only selected concepts)
 - subset_{k}_best_list.txt (list of selected files, one per line)
 - printed summary
"""

import json
import random
import argparse
import sys
from collections import defaultdict
from typing import List, Tuple

def flatten_concepts(data):
    """
    Return list of tuples: (category, concept, n_images, filenames_list)
    """
    flat = []
    for category, concepts in data.items():
        if not isinstance(concepts, dict):
            continue
        for concept, entry in concepts.items():
            if not isinstance(entry, dict):
                continue
            # prefer 'train' key; if absent, try other keys or assume empty
            files = []
            if 'train' in entry and isinstance(entry['train'], list):
                files = entry['train']
            else:
                # attempt to collect lists inside entry
                for v in entry.values():
                    if isinstance(v, list):
                        files = v
                        break
            n = len(files)
            flat.append((category, concept, n, files))
    return flat

def build_nested_from_selection(selection: List[Tuple[str,str,List[str]]]):
    out = {}
    for category, concept, files in selection:
        out.setdefault(category, {})[concept] = {"train": files}
    return out

def randomized_search(flat, k, target, iterations=20000, rng=None, require_category_coverage=False):
    """
    Try `iterations` random draws of k concepts and track subset whose total
    images is closest to target. Returns best selection (list of tuples).
    """
    if rng is None:
        rng = random.Random()

    n_total = len(flat)
    if k > n_total:
        raise ValueError(f"k={k} larger than total concepts {n_total}")

    # Precompute items w/o zero images
    indices = list(range(n_total))

    best = None
    best_diff = None

    # If require_category_coverage, ensure at least one concept from each category is present
    categories = list({c for c,_,_,_ in flat})
    n_categories = len(categories)
    import pdb;pdb.set_trace()
    for it in range(iterations):
        # sample k unique indices
        chosen_idx = rng.sample(indices, k)
        chosen = [flat[i] for i in chosen_idx]
        if require_category_coverage:
            chosen_cats = {c for c,_,_,_ in chosen}
            if len(chosen_cats) < n_categories:
                # skip this draw if it doesn't cover all categories
                continue
        total_images = sum(item[2] for item in chosen)
        diff = abs(total_images - target)
        if best is None or diff < best_diff:
            best = chosen
            best_diff = diff
            if best_diff == 0:
                break
    return best, best_diff

def greedy_closest(flat, k, target):
    """
    Greedy heuristic: sort by file count descending, take top k, then try small swaps
    to move closer to target.
    """
    items = sorted(flat, key=lambda x: x[2], reverse=True)
    if k > len(items):
        raise ValueError("k larger than available concepts")

    chosen = items[:k]
    chosen_set = set((c,concept) for c,concept,_,_ in chosen)
    total = sum(x[2] for x in chosen)
    improved = True
    while improved:
        improved = False
        # try swap one chosen with one not chosen
        for i, ch in enumerate(chosen):
            for candidate in items[k:]:
                if (candidate[0], candidate[1]) in chosen_set:
                    continue
                new_total = total - ch[2] + candidate[2]
                if abs(new_total - target) < abs(total - target):
                    # perform swap
                    chosen_set.remove((ch[0], ch[1]))
                    chosen_set.add((candidate[0], candidate[1]))
                    chosen[i] = candidate
                    total = new_total
                    improved = True
                    break
            if improved:
                break
    return chosen, abs(total - target)

def write_outputs(selection, k, target, out_prefix="subset"):
    # selection: list of tuples (category,concept,n,filenames)
    # Build nested structure but include only filenames (train)
    nested = {}
    all_files = []
    for category, concept, n, files in selection:
        nested.setdefault(category, {})[concept] = {"train": files}
        all_files.extend(files)

    out_json = f"{out_prefix}_{k}_best.json"
    out_list = f"{out_prefix}_{k}_best_list.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False)

    with open(out_list, "w", encoding="utf-8") as f:
        for p in all_files:
            f.write(p + "\n")

    return out_json, out_list, len(all_files)

def main():
    parser = argparse.ArgumentParser(description="Select subset of concepts to hit a target image count.")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--target_images", "-t", type=int, default=1000, help="Target total images (default 1000)")
    parser.add_argument("--k_values", "-k", nargs="+", type=int, default=[30], help="List of k concept counts to produce")
    parser.add_argument("--iterations", "-it", type=int, default=20000, help="Random sampling iterations per k (default 20000)")
    parser.add_argument("--random_seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--require_category_coverage", action="store_true", help="Enforce at least one concept from every category (may be impossible depending on k)")
    parser.add_argument("--out_prefix", type=str, default="subset", help="Prefix for output files")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat_raw = flatten_concepts(data)
    # convert to (category, concept, n, files_list)
    flat = [(categ, conc, n, files) for categ, conc, n, files in flat_raw if n > 0]
    if not flat:
        print("No concepts with train images found. Exiting.", file=sys.stderr)
        sys.exit(1)

    total_concepts = len(flat)
    total_images = sum(x[2] for x in flat)
    print(f"Total concepts available (with >0 images): {total_concepts}")
    print(f"Total images across all concepts: {total_images}")

    rng = random.Random(args.random_seed)

    for k in args.k_values:
        print("\n" + "="*40)
        print(f"Selecting k={k} concepts to target {args.target_images} images...")

        if k > total_concepts:
            print(f"Requested k={k} > available concepts {total_concepts}. Skipping.")
            continue

        # randomized search first
        best, best_diff = randomized_search(flat, k, args.target_images, iterations=args.iterations, rng=rng, require_category_coverage=args.require_category_coverage)
        if best is not None:
            best_total = sum(x[2] for x in best)
        else:
            best_total = None

        print(f"Random search best diff = {best_diff}, total_images = {best_total}")

        # If random search failed or not good enough, run greedy and compare
        greedy_sel, greedy_diff = greedy_closest(flat, k, args.target_images)
        greedy_total = sum(x[2] for x in greedy_sel)
        print(f"Greedy best diff = {greedy_diff}, total_images = {greedy_total}")

        # choose better of the two (lower diff). If tie, prefer one with total_images <= target (optional)
        chosen = None
        if best is None or greedy_diff < best_diff:
            chosen = greedy_sel
            chosen_diff = greedy_diff
            chosen_total = greedy_total
            method = "greedy"
        else:
            chosen = best
            chosen_diff = best_diff
            chosen_total = best_total
            method = "randomized"

        # final selection: convert to (cat,concept,n,files)
        final_sel = [(c, concept, len(files), files) for c, concept, n, files in chosen]
        import pdb;pdb.set_trace()
        out_json, out_list, out_count = write_outputs(final_sel, k, args.target_images, out_prefix=args.out_prefix)
        print(f"Chosen method: {method}")
        print(f"Wrote JSON -> {out_json}")
        print(f"Wrote file list -> {out_list}")
        print(f"Selected concepts: {k}, total images in selection: {out_count}, diff={chosen_diff}")
        # show per-category breakdown
        per_cat = defaultdict(int)
        for c, concept, n, files in final_sel:
            per_cat[c] += n
        print("Per-category image counts for selection:")
        for c, cnt in sorted(per_cat.items(), key=lambda x: -x[1]):
            print(f"  {c}: {cnt} images")

if __name__ == "__main__":
    main()
