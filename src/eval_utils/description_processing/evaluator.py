"""
Description Evaluator

This module handles the evaluation of descriptions to identify whether they contain
state-specific or location-specific attributes.

Main functionality:
- evaluate(): Evaluate all descriptions in a dataset for state/location attributes
- aggregate_stats(): Compute aggregate statistics from evaluation results
"""

import json
import time
import statistics as st
import sys
from pathlib import Path
from typing import List, Dict, Any

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:
    torch = None
    DataLoader = None

from .shared import JsonDescriptionsDataset, collate_batch, infer_batch


def aggregate_stats(batch_out: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute aggregate statistics from evaluation results.

    Args:
        batch_out: List of evaluation results, each containing:
            - has_state: bool indicating presence of state attributes
            - has_location: bool indicating presence of location attributes
            - length: int indicating description length in words

    Returns:
        Dictionary with aggregate statistics:
            - mean_state: proportion of descriptions with state attributes
            - mean_location: proportion of descriptions with location attributes
            - mean_length: average description length
    """
    has_state = [item['has_state'] for item in batch_out]
    has_location = [item['has_location'] for item in batch_out]
    lengths = [item['length'] for item in batch_out]
    mean_state = st.mean(has_state)
    mean_location = st.mean(has_location)
    mean_length = st.mean(lengths)
    return {"mean_state": mean_state, "mean_location": mean_location, "mean_length": mean_length}


def evaluate(args, model, tokenizer, prefix: str, suffix: str, outpath: Path) -> None:
    """
    Evaluate all descriptions in a dataset for state and location attributes.

    This function:
    1. Loads the dataset from args.input
    2. Processes descriptions in batches
    3. Calls the model to evaluate each batch
    4. Aggregates results and saves to output file

    Args:
        args: Argument object containing:
            - input: Path to input JSON dataset
            - batch_size: Number of descriptions per batch
            - sleep_between_batches: Optional delay between batches (seconds)
            - fail_on_error: Whether to raise exception on errors
        model: The loaded model for inference
        tokenizer: The tokenizer for the model
        prefix: Evaluation prompt prefix
        suffix: Evaluation prompt suffix
        outpath: Path where results will be saved

    Output format:
        JSON file containing:
        {
            "stats": {
                "mean_state": float,
                "mean_location": float,
                "mean_length": float
            },
            "response": [
                {
                    "id": str,
                    "has_state": bool,
                    "has_location": bool,
                    "length": int,
                    "text": str
                },
                ...
            ]
        }

    Raises:
        Exception: If fail_on_error is True and batch processing fails
    """
    ds = JsonDescriptionsDataset(args.input)

    # Create batch iterator
    if torch is None:
        # create a simple iterator fallback if PyTorch is unavailable
        def batches():
            batch = []
            for i in range(len(ds)):
                batch.append(ds[i])
                if len(batch) >= args.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b)
        def batches():
            for b in dl:
                yield b

    total = 0
    start_time = time.time()
    batch_out = []

    # Process each batch
    for batch_idx, batch in enumerate(batches()):
        items = collate_batch(batch)
        try:
            parsed = infer_batch(model, tokenizer, batch, prefix, suffix)
            batch_out.extend(parsed)
            total += len(parsed)
        except Exception as e:
            print(f"[ERROR] Batch {batch_idx} failed to parse: {e}", file=sys.stderr)
            # write raw output for debugging
            debug_file = outpath.parent / f"debug_batch_{batch_idx}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(str(e))
            # skip this batch or raise, depending on fail behavior
            if args.fail_on_error:
                raise
            else:
                print(f"[WARN] Skipping batch {batch_idx} (writing error to {debug_file})", file=sys.stderr)
                continue

        # optional sleep to respect rate limits
        if args.sleep_between_batches > 0:
            time.sleep(args.sleep_between_batches)

        print(f"[INFO] Completed batch {batch_idx+1}, items_processed={total}")

    elapsed = time.time() - start_time
    stats = aggregate_stats(batch_out)
    overall = {"stats": stats, "response": batch_out}

    # Save results
    with open(outpath, "w", encoding="utf-8") as fout:
        json.dump(overall, fout, indent=2)

    print(f"[DONE] Evaluated {total} items in {elapsed:.1f}s. Results saved to {outpath}")
