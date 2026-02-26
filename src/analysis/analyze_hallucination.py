"""
Hallucination analysis for description-based recognition methods.

Two analysis types, selected automatically by method:

  1. Reasoning Trace Analysis  (R2P, zero_shot)
     For each FP: extracts colors from the reference description and from
     the model's reasoning about the query. Colors echoed from the reference
     description that are absent from the query concept's description are
     flagged as hallucinated.

  2. Description Overlap Analysis  (RAP, RePIC, ours)
     For each FP: compares colors in the reference description against
     colors in the query concept's description. Color attributes present
     in the reference but absent from the query expose the mismatch that
     the model failed to resolve visually.

All metrics are reported overall and split by in-category / cross-category FPs.

Usage examples
--------------
# R2P — local Qwen3B
python analysis/analyze_hallucination.py \\
    --method R2P --dataset YoLLaVA \\
    --results_dir results_R2P/QWEN_YoLLaVA_seed_23/all \\
    --file_identifier reco_results.json \\
    --ref_db_path example_database_R2P/YoLLaVA_seed_23/all/database_user_defined_cat_no_template.json \\
    --query_db_path outputs/YoLLaVA/all/seed_23/database_original_7b.json \\
    --extraction_mode local --local_model_path /path/to/qwen3b \\
    --seed 23

# RAP — API
python analysis/analyze_hallucination.py \\
    --method RAP --dataset MyVLM \\
    --results_dir results_RAP/MyVLM/all \\
    --file_identifier reco_results.json \\
    --ref_db_path outputs/MyVLM/all/seed_23/database_original_7b.json \\
    --query_db_path outputs/MyVLM/all/seed_23/database_original_7b.json \\
    --seed 23
"""

import sys
import argparse
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from within analysis/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))                   # analysis/ (for attribute_extraction)
sys.path.insert(0, str(_HERE.parent))            # project root (for compute_recognition_metrics)

from attribute_extraction import (
    extract_colors_rule_based,
    colors_overlap,
    colors_mismatch,
    AttributeExtractor,
)
from compute_recognition_metrics import (
    DATASET_CATEGORY_MAPS,
    get_category_for_concept,
    extract_concept_from_path,
    extract_concept_from_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Methods that expose reasoning traces
REASONING_TRACE_METHODS = {"R2P", "zero_shot"}
# Methods for which only description overlap is possible
DESCRIPTION_OVERLAP_METHODS = {"RAP", "RePIC", "ours"}


# ============================================================================
# Database loading
# ============================================================================

def load_database(db_path: str, is_r2p_format: bool = False, is_rap_format: bool=False) -> Dict[str, str]:
    """
    Load concept → description mapping from a database JSON.

    Standard format  (RAP / RePIC / zero_shot):
        info['general']          is a list; use [0]
        info['distinct features'] is a list; use [0]

    R2P format:
        info['general']           is a plain string
        info['distinct features'] is a bracket-notation string, e.g.
            "[Hand-painted, colorful design, yellow lid]"
        The brackets are stripped before use.

    Args:
        db_path       : Path to the database JSON file.
        is_r2p_format : Set True when loading the R2P-specific database.

    Returns:
        Dict mapping concept_name (no angle brackets) → description string.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with path.open("r") as f:
        data = json.load(f)

    concept_dict = data.get("concept_dict", {})
    descriptions: Dict[str, str] = {}

    for key, val in concept_dict.items():
        # Normalise key: strip angle brackets
        concept_name = val.get("name", key.strip("<>"))
        info = val.get("info", {})

        try:
            if is_r2p_format:
                general  = info.get("general", "")
                distinct = info.get("distinct features", "")
                # Strip surrounding brackets from the bracket-notation string
                distinct = distinct.strip("[]")
                description = f"{general}. {distinct}".strip(". ")
            elif is_rap_format:
                description = info
            else:
                general  = info.get("general",          [""])[0]
                distinct = info.get("distinct features", [""])[0]
                description = f"{general}. {distinct}".strip(". ")
        except (IndexError, AttributeError, TypeError) as exc:
            logger.warning(f"Could not parse description for '{concept_name}': {exc}")
            description = ""

        descriptions[concept_name] = description

    logger.info(f"Loaded {len(descriptions)} concepts from {path.name}")
    return descriptions


# ============================================================================
# Entry-level helpers
# ============================================================================

def get_ref_concept(entry: Dict, method: str) -> Optional[str]:
    """Extract the reference concept name from a result entry."""
    if method == "R2P":
        concepts = entry.get("ret_concepts", [])
        return concepts[0] if concepts else None
    elif method in ("RAP", "RePIC", "ours"):
        return extract_concept_from_info(entry.get("info", ""))
    elif method == "zero_shot":
        ref_path = entry.get("ref_path", "")
        return extract_concept_from_path(ref_path) if ref_path else None
    return None


def get_query_concept(entry: Dict, method: str) -> Optional[str]:
    """Extract the query concept name from a result entry."""
    if method == "R2P":
        return extract_concept_from_path(entry.get("image", ""))
    elif method in ("RAP", "RePIC", "ours"):
        return extract_concept_from_path(entry.get("image_path", ""))
    elif method == "zero_shot":
        return extract_concept_from_path(entry.get("query_path", ""))
    return None


def parse_reasoning(entry: Dict, method: str) -> Optional[str]:
    """
    Extract the model's reasoning text from an entry.
    Only applicable for R2P and zero_shot.

    R2P       : 'reasoning' field is a JSON string → extract 'Reasoning' key.
    zero_shot : 'response'  field is a JSON string → extract 'Reasoning' key.

    Returns:
        Reasoning string, or None if unavailable.
    """
    if method == "R2P":
        raw = entry.get("reasoning", "")
        if not raw:
            return None
        try:
            return json.loads(raw).get("Reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            return raw  # Fallback: use raw string

    elif method == "zero_shot":
        raw = entry.get("response", "")
        if not raw:
            return None
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip("`").strip()
        try:
            return json.loads(raw).get("Reasoning", "")
        except (json.JSONDecodeError, AttributeError):
            return raw

    return None


# ============================================================================
# Result file loading
# ============================================================================

def load_all_entries(
    args,
    results_dir: Path,
    file_identifier: str,
    method: str,
    seed: Optional[int] = None,
) -> List[Dict]:
    """
    Walk results_dir and collect all entries from per-concept JSON files.
    Mirrors the file-discovery logic in compute_recognition_metrics.py.

    Returns:
        Flat list of all entry dicts across all concepts.
    """
    all_entries: List[Dict] = []
    for concept_dir in sorted(results_dir.iterdir()):
        if not concept_dir.is_dir():
            continue

        json_file = None
        # Format 1: Direct (R2P style)
        candidate = concept_dir / file_identifier
        if candidate.exists():
            json_file = candidate

        # Format 2: Explicit seed subdirectory
        elif seed is not None:
            candidate = concept_dir / f"seed_{seed}" / file_identifier
            if candidate.exists():
                json_file = candidate

        # Format 3: Auto-detect any seed_* subdirectory
        if json_file is None:
            seed_dirs = sorted(concept_dir.glob("seed_*"))
            if seed_dirs:
                candidate = seed_dirs[0] / file_identifier
                if candidate.exists():
                    json_file = candidate

        if json_file is None:
            logger.warning(f"  {file_identifier} not found in {concept_dir.name}, skipping.")
            continue

        with json_file.open("r") as f:
            data = json.load(f)
        all_entries.extend(data.get("results", []))

    return all_entries


# ============================================================================
# Analysis 1: Reasoning Trace (R2P / zero_shot)
# ============================================================================

def _collect_fp_candidates(
    entries: List[Dict],
    ref_db: Dict[str, str],
    query_db: Dict[str, str],
    method: str,
    dataset: str,
) -> List[Dict]:
    """
    First pass: filter all entries down to FP candidates and pre-compute
    the description-based color sets. No LLM calls here.

    Returns:
        List of dicts with entry metadata and description colors populated,
        ready for the LLM reasoning-trace step.
    """
    candidates: List[Dict] = []

    for entry in entries:
        pred     = entry.get("pred",     "").lower().strip()
        solution = entry.get("solution", "").lower().strip()

        if pred != "yes" or solution != "no":
            continue  # Only FPs

        query_concept = get_query_concept(entry, method)
        ref_concept   = get_ref_concept(entry, method)

        if not query_concept or not ref_concept:
            continue

        query_category = get_category_for_concept(query_concept, dataset)
        ref_category   = get_category_for_concept(ref_concept,   dataset)
        is_in_category = query_category == ref_category

        ref_desc   = ref_db.get(ref_concept,    "")
        query_desc = query_db.get(query_concept, "")

        if not ref_desc or not query_desc:
            logger.debug(
                f"Missing description: ref={ref_concept!r}, query={query_concept!r} — skipping."
            )
            continue

        candidates.append({
            "_entry":          entry,
            "query_concept":   query_concept,
            "ref_concept":     ref_concept,
            "query_category":  query_category,
            "ref_category":    ref_category,
            "is_in_category":  is_in_category,
            "ref_desc":        ref_desc,
            "query_desc":      query_desc,
            "ref_desc_colors":   extract_colors_rule_based(ref_desc),
            "query_true_colors": extract_colors_rule_based(query_desc),
        })
    return candidates


def analyze_reasoning_traces(
    entries: List[Dict],
    ref_db: Dict[str, str],
    query_db: Dict[str, str],
    extractor: "AttributeExtractor",
    method: str,
    dataset: str,
) -> List[Dict]:
    """
    Analyse FP instances using LLM-based reasoning trace color extraction.

    Two-pass design:
      Pass 1 — fast: filter FP entries and pre-compute description colors.
      Pass 2 — slow: call the LLM for each FP to extract reasoning colors,
               with a tqdm progress bar showing X / N processed.

    For each FP entry:
      ref_desc_colors    : colors from the reference concept description (rule-based)
      query_true_colors  : colors from the query concept description     (rule-based)
      reasoning_colors   : colors the model attributes to the query image (LLM-based)
      echoed_colors      : ref_desc_colors ∩ reasoning_colors
      hallucinated_colors: echoed colors absent from query_true_colors

    Returns:
        List of per-FP dicts with analysis fields populated.
    """
    # --- Pass 1: collect FP candidates (no LLM) ---
    candidates = _collect_fp_candidates(entries, ref_db, query_db, method, dataset)
    n_fps = len(candidates)
    logger.info(f"Found {n_fps} FP instances — starting LLM reasoning extraction.")

    # Progress bar: use tqdm if available, fall back to plain logging
    try:
        from tqdm import tqdm
        iterator = tqdm(candidates, desc="Extracting reasoning colors", unit="FP")
    except ImportError:
        logger.info("tqdm not found; falling back to log-based progress.")
        iterator = candidates

    # --- Pass 2: LLM extraction with progress tracking ---
    fp_instances: List[Dict] = []

    for i, candidate in enumerate(iterator, start=1):
        # Plain-log fallback (only active when tqdm is absent)
        if not hasattr(iterator, "update"):
            logger.info(f"  FP {i}/{n_fps}  —  {candidate['query_concept']} vs {candidate['ref_concept']}")

        entry         = candidate.pop("_entry")
        reasoning_text = parse_reasoning(entry, method)
        reasoning_colors = (
            extractor.extract_colors_from_reasoning(reasoning_text)
            if reasoning_text else set()
        )

        echoed_colors       = colors_overlap(candidate["ref_desc_colors"], reasoning_colors)
        hallucinated_colors = colors_mismatch(echoed_colors, candidate["query_true_colors"])

        fp_instances.append({
            **candidate,
            "reasoning_text":      reasoning_text,
            "reasoning_colors":    reasoning_colors,
            "echoed_colors":       echoed_colors,
            "hallucinated_colors": hallucinated_colors,
            "is_hallucinated":     len(hallucinated_colors) > 0,
        })

    return fp_instances


# ============================================================================
# Analysis 2: Description Overlap (RAP / RePIC / ours)
# ============================================================================

def analyze_description_overlap(
    entries: List[Dict],
    ref_db: Dict[str, str],
    query_db: Dict[str, str],
    method: str,
    dataset: str,
) -> List[Dict]:
    """
    Analyse FP instances by comparing reference and query concept description colors.

    For each FP entry:
      ref_desc_colors   : colors from the reference concept description
      query_true_colors : colors from the query concept description (ground truth)
      mismatched_colors : ref colors that have no counterpart in query colors
                          (attributes that should have disqualified a YES prediction)

    Returns:
        List of per-FP dicts with analysis fields populated.
    """
    fp_instances: List[Dict] = []

    for entry in entries:
        pred     = entry.get("pred",     "").lower().strip()
        solution = entry.get("solution", "").lower().strip()

        if pred != "yes" or solution != "no":
            continue

        query_concept = get_query_concept(entry, method)
        ref_concept   = get_ref_concept(entry, method)

        if not query_concept or not ref_concept:
            continue

        query_category = get_category_for_concept(query_concept, dataset)
        ref_category   = get_category_for_concept(ref_concept,   dataset)
        is_in_category = query_category == ref_category

        ref_desc   = ref_db.get(ref_concept,    "")
        query_desc = query_db.get(query_concept, "")

        if not ref_desc or not query_desc:
            logger.debug(
                f"Missing description: ref={ref_concept!r}, query={query_concept!r} — skipping."
            )
            continue

        ref_desc_colors   = extract_colors_rule_based(ref_desc)
        query_true_colors = extract_colors_rule_based(query_desc)
        mismatched_colors  = colors_mismatch(ref_desc_colors, query_true_colors)

        fp_instances.append({
            "query_concept":        query_concept,
            "ref_concept":          ref_concept,
            "query_category":       query_category,
            "ref_category":         ref_category,
            "is_in_category":       is_in_category,
            "ref_desc":             ref_desc,
            "query_desc":           query_desc,
            "ref_desc_colors":      ref_desc_colors,
            "query_true_colors":    query_true_colors,
            "mismatched_colors":    mismatched_colors,
            "is_mismatched":        len(mismatched_colors) > 0,
        })

    return fp_instances


# ============================================================================
# Metrics computation
# ============================================================================

def _safe_rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator > 0 else 0.0


def _subset_metrics(subset: List[Dict], analysis_type: str) -> Dict:
    n = len(subset)
    if analysis_type == "reasoning_trace":
        n_hallucinated   = sum(1 for x in subset if x["is_hallucinated"])
        n_with_ref_colors = sum(1 for x in subset if x["ref_desc_colors"])
        n_echoed          = sum(1 for x in subset if x["echoed_colors"])
        return {
            "n_fps":              n,
            "n_hallucinated":     n_hallucinated,
            "hallucination_rate": _safe_rate(n_hallucinated, n),
            "n_with_ref_colors":  n_with_ref_colors,
            "n_echoed":           n_echoed,
            "text_dominance_rate": _safe_rate(n_echoed, n_with_ref_colors),
        }
    else:  # description_overlap
        n_mismatched      = sum(1 for x in subset if x["is_mismatched"])
        n_with_ref_colors = sum(1 for x in subset if x["ref_desc_colors"])
        return {
            "n_fps":              n,
            "n_mismatched":       n_mismatched,
            "mismatch_rate":      _safe_rate(n_mismatched, n),
            "n_with_ref_colors":  n_with_ref_colors,
        }


def compute_metrics(fp_instances: List[Dict], analysis_type: str) -> Dict:
    """
    Compute hallucination / mismatch metrics overall and by category split.

    Args:
        fp_instances  : Output of analyze_reasoning_traces or analyze_description_overlap.
        analysis_type : 'reasoning_trace' or 'description_overlap'.

    Returns:
        Dict with keys 'overall', 'in_category', 'cross_category'.
    """
    in_cat    = [x for x in fp_instances if     x["is_in_category"]]
    cross_cat = [x for x in fp_instances if not x["is_in_category"]]

    return {
        "overall":        _subset_metrics(fp_instances, analysis_type),
        "in_category":    _subset_metrics(in_cat,       analysis_type),
        "cross_category": _subset_metrics(cross_cat,    analysis_type),
    }


# ============================================================================
# Reporting
# ============================================================================

def print_report(
    metrics: Dict,
    method: str,
    dataset: str,
    analysis_type: str,
):
    """Print a formatted analysis report to stdout."""
    label = analysis_type.replace("_", " ").title()
    print(f"\n{'='*65}")
    print(f"  Hallucination Analysis  |  {method} on {dataset}")
    print(f"  Analysis type: {label}")
    print(f"{'='*65}")

    splits = [
        ("Overall",             metrics["overall"]),
        ("In-Category FPs",     metrics["in_category"]),
        ("Cross-Category FPs",  metrics["cross_category"]),
    ]

    for split_name, m in splits:
        print(f"\n  [{split_name}]  n_fps = {m['n_fps']}")
        if analysis_type == "reasoning_trace":
            print(
                f"    Hallucination Rate   : {m['hallucination_rate']:.2%}"
                f"  ({m['n_hallucinated']} / {m['n_fps']})"
            )
            print(
                f"    Text Dominance Rate  : {m['text_dominance_rate']:.2%}"
                f"  ({m['n_echoed']} / {m['n_with_ref_colors']}"
                f" FPs with ref colors)"
            )
        else:
            print(
                f"    Attribute Mismatch Rate : {m['mismatch_rate']:.2%}"
                f"  ({m['n_mismatched']} / {m['n_fps']})"
            )

    print(f"\n{'='*65}\n")


# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hallucination / attribute-mismatch analysis for recognition methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method", required=True,
        choices=["R2P", "RAP", "RePIC", "zero_shot", "ours"],
        help="Method whose results to analyse.",
    )
    p.add_argument(
        "--dataset", required=True,
        choices=["YoLLaVA", "MyVLM", "PerVA", "DreamBooth"],
    )
    p.add_argument(
        "--results_dir", required=True,
        help="Directory containing per-concept result folders.",
    )
    p.add_argument(
        "--file_identifier", required=True,
        help="JSON filename inside each concept folder (e.g. reco_results.json).",
    )
    p.add_argument(
        "--ref_db_path", required=True,
        help=(
            "Database JSON for reference concept descriptions. "
            "Use the R2P-specific DB for --method R2P, "
            "the standard DB otherwise."
        ),
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--extraction_mode", choices=["local", "api"], default="api",
        help="LLM backend for reasoning trace extraction (R2P / zero_shot only).",
    )
    p.add_argument(
        "--local_model_path", default=None,
        help="Path to local Qwen3B weights (required when --extraction_mode local).",
    )
    p.add_argument(
        "--output_name", default=None,
        help="Output JSON filename. Defaults to {method}_{dataset}_hallucination.json.",
    )
    return p


def main():
    args = build_arg_parser().parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    is_reasoning_trace = args.method in REASONING_TRACE_METHODS
    analysis_type = "reasoning_trace" if is_reasoning_trace else "description_overlap"

    # ------------------------------------------------------------------
    # Load databases
    # ------------------------------------------------------------------
    is_r2p_format = (args.method == "R2P")
    is_rap_format = (args.method == "RAP" or args.method == "RePIC")
    logger.info("Loading reference description database…")
    ref_db   = load_database(args.ref_db_path,   is_r2p_format=is_r2p_format, is_rap_format=is_rap_format)
    logger.info("Loading query description database…")
    query_db = ref_db

    # ------------------------------------------------------------------
    # Load result entries
    # ------------------------------------------------------------------
    logger.info("Loading result entries…")
    all_entries = load_all_entries(
        args, results_dir, args.file_identifier, args.method, args.seed
    )
    logger.info(f"Total entries loaded: {len(all_entries)}")

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------
    if is_reasoning_trace:
        extractor = AttributeExtractor(
            mode=args.extraction_mode,
            local_model_path=args.local_model_path,
            api_key="sk-c94d1738ffb84c4ba7ca16c30c724d5d"
        )
        logger.info("Running reasoning trace analysis…")
        fp_instances = analyze_reasoning_traces(
            all_entries, ref_db, query_db, extractor, args.method, args.dataset
        )
    else:
        logger.info("Running description overlap analysis…")
        fp_instances = analyze_description_overlap(
            all_entries, ref_db, query_db, args.method, args.dataset
        )

    logger.info(f"FP instances analysed: {len(fp_instances)}")

    # ------------------------------------------------------------------
    # Compute metrics and report
    # ------------------------------------------------------------------
    metrics = compute_metrics(fp_instances, analysis_type)
    print_report(metrics, args.method, args.dataset, analysis_type)

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    output_name = args.output_name or f"{args.method}_{args.dataset}_{args.file_identifier}_hallucination.json"
    output_path = reports_dir / output_name

    # Serialise sets → lists for JSON
    serialisable = []
    for inst in fp_instances:
        serialisable.append(
            {k: (sorted(v) if isinstance(v, set) else v) for k, v in inst.items()}
        )

    with output_path.open("w") as f:
        json.dump({"metrics": metrics, "fp_instances": serialisable}, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
