#!/usr/bin/env python3
"""
Find qualitative examples where RRG is correct but zs-speaker and R2P both fail.

For each matching triplet the output captures the reasoning text from all three
methods so the entries can be directly used to build a paper figure.

  zs-speaker  → json.loads(entry["response"])["Reasoning"]
  R2P         → json.loads(entry["reasoning"])["Reasoning"]
  RRG         → json.loads(entry["response"])["Reasoning"]

File path templates (relative to the respective results dirs)
-------------------------------------------------------------
zs-speaker : {ours_dir}/{dataset}/{category}/{concept}/seed_{seed}/results_model_original_7b_db_original_7b_k_{k}.json
RRG        : {ours_dir}/{dataset}/{category}/{concept}/seed_{seed}/results_model_original_7b_db_sp_concise_soft_gated_k_{k}.json
R2P        : {r2p_dir}/QWEN_{dataset}_seed_{seed}/{category}/{concept}/recall_results.json

Usage
-----
python find_qualitative_examples.py \\
    --ours_results_dir /path/to/our/results \\
    --r2p_results_dir  /path/to/r2p/results \\
    --seed 42 \\
    --output qualitative_examples.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# ── Constants ──────────────────────────────────────────────────────────────────

DATASETS = ["YoLLaVA", "MyVLM", "PerVA"]

PERVA_CATEGORIES = [
    "bag", "book", "bottle", "bowl", "clothe", "cup", "decoration", "headphone",
    "pillow", "plant", "plate", "remote", "retail", "telephone", "tie", "towel",
    "toy", "tro_bag", "tumbler", "umbrella", "veg",
]

ZS_MODEL = "original_7b"
ZS_DB    = "original_7b"
RRG_MODEL = "original_7b"
RRG_DB    = "sp_concise_soft_gated"

# ── Path helpers ───────────────────────────────────────────────────────────────

def our_path(results_dir: str, dataset: str, category: str,
             concept: str, seed: int, model: str, db: str, k: int) -> Path:
    return (
        Path(results_dir) / dataset / category / concept /
        f"seed_{seed}" / f"results_model_{model}_db_{db}_k_{k}.json"
    )

def r2p_path(results_dir: str, dataset: str, category: str,
             concept: str, seed: int) -> Path:
    return (
        Path(results_dir) / f"QWEN_{dataset}_seed_{seed}" /
        category / concept / "recall_results.json"
    )

# ── File helpers ───────────────────────────────────────────────────────────────

def load_results(path: Path) -> Optional[List[Dict]]:
    """Load results list from a JSON file; returns None if file missing/broken."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        entries = data.get("results", data) if isinstance(data, dict) else data
        return entries if isinstance(entries, list) else None
    except Exception as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None

def get_categories(dataset: str) -> List[str]:
    return PERVA_CATEGORIES if dataset == "PerVA" else ["all"]

def discover_concepts(base: Path) -> List[str]:
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir())

# ── Matching / extraction helpers ─────────────────────────────────────────────

def stem(path_str: str) -> str:
    """Filename stem used as the matching key across methods."""
    return Path(path_str).stem

def build_lookup(entries: List[Dict], key: str) -> Dict[str, Dict]:
    """Index entries by the filename stem of a given path key."""
    out = {}
    for e in entries:
        val = e.get(key, "")
        if val:
            out[stem(val)] = e
    return out

def r2p_correct(entry: Dict) -> bool:
    return entry.get("pred_name", "").lower() == entry.get("solution", "").lower()

def extract_reasoning(entry: Dict, field: str) -> str:
    """Parse a JSON string field and return the 'Reasoning' value inside it."""
    raw = entry.get(field, "")
    if not raw:
        return ""
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        return parsed.get("Reasoning", "")
    except Exception:
        return raw

# ── Core collection loop ───────────────────────────────────────────────────────

def collect_examples(ours_dir: str, r2p_dir: str, seed: int, k: int) -> List[Dict]:
    examples = []

    for dataset in DATASETS:
        categories = get_categories(dataset)

        for category in categories:
            base = Path(ours_dir) / dataset / category
            concepts = discover_concepts(base)

            if not concepts:
                print(f"  [INFO] {dataset}/{category}: no concepts found, skipping.")
                continue

            print(f"  [INFO] {dataset}/{category}: {len(concepts)} concepts")

            for concept in concepts:
                zs_file  = our_path(ours_dir, dataset, category, concept, seed, ZS_MODEL,  ZS_DB,  k)
                rrg_file = our_path(ours_dir, dataset, category, concept, seed, RRG_MODEL, RRG_DB, k)
                r2p_file = r2p_path(r2p_dir,  dataset, category, concept, seed)

                zs_entries  = load_results(zs_file)
                rrg_entries = load_results(rrg_file)
                r2p_entries = load_results(r2p_file)

                if not zs_entries or not rrg_entries or not r2p_entries:
                    continue

                # Build stem-keyed lookups
                zs_lookup  = build_lookup(zs_entries,  "image_path")
                rrg_lookup = build_lookup(rrg_entries, "image_path")
                r2p_lookup = build_lookup(r2p_entries, "image")

                # Only examine images present in all three
                common = set(zs_lookup) & set(rrg_lookup) & set(r2p_lookup)

                for img_stem in sorted(common):
                    zs_e  = zs_lookup[img_stem]
                    rrg_e = rrg_lookup[img_stem]
                    r2p_e = r2p_lookup[img_stem]

                    zs_ok  = bool(zs_e.get("correct", False))
                    rrg_ok = bool(rrg_e.get("correct", False))
                    r2p_ok = r2p_correct(r2p_e)

                    # Keep only: zs wrong, R2P wrong, RRG correct
                    if zs_ok or r2p_ok or not rrg_ok:
                        continue

                    examples.append({
                        "image_path": zs_e.get("image_path", ""),
                        "concept":    zs_e.get("solution", ""),
                        "dataset":    dataset,
                        "category":   category,
                        "seed":       seed,
                        "zs_speaker": extract_reasoning(zs_e,  "response"),
                        "r2p":        extract_reasoning(r2p_e, "reasoning"),
                        "rrg":        extract_reasoning(rrg_e, "response"),
                    })

    return examples

# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find examples where RRG succeeds but zs-speaker and R2P both fail."
    )
    parser.add_argument("--ours_results_dir", required=True,
                        help="Root results dir for zs-speaker and RRG (e.g. /path/to/results)")
    parser.add_argument("--r2p_results_dir",  required=True,
                        help="Root results dir for R2P (e.g. /path/to/results_R2P)")
    parser.add_argument("--seed",   type=int, default=42, help="Seed to use (default: 42)")
    parser.add_argument("--k",      type=int, default=3,  help="Number of retrieved candidates (default: 3)")
    parser.add_argument("--output", default="qualitative_examples.json",
                        help="Output JSON file path (default: qualitative_examples.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Searching seed={args.seed}, k={args.k}")
    print(f"  Ours dir : {args.ours_results_dir}")
    print(f"  R2P  dir : {args.r2p_results_dir}")

    examples = collect_examples(args.ours_results_dir, args.r2p_results_dir, args.seed, args.k)

    print(f"\nFound {len(examples)} qualifying examples across all datasets.")

    # Group by dataset for a quick summary
    by_dataset: Dict[str, int] = {}
    for ex in examples:
        by_dataset[ex["dataset"]] = by_dataset.get(ex["dataset"], 0) + 1
    for ds, count in sorted(by_dataset.items()):
        print(f"  {ds}: {count}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
