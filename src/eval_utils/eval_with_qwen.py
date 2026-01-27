#!/usr/bin/env python3
"""
evaluate_with_qwen.py

Usage:
    python evaluate_with_qwen.py --input data/descriptions.json \
        --out results/eval_results.jsonl --batch-size 64

    python evaluate_with_qwen.py --input data/descriptions.json --refine state \
        --out results/ --batch-size 64

What it does:
- Loads a JSON dataset of descriptions
- Evaluates descriptions for state and location attributes (default mode)
- Refines descriptions by removing state/location attributes (with --refine flag)
- Uses Qwen-8B model via HuggingFace transformers

Refine modes:
- none: Evaluation only (default)
- state: Remove state attributes
- location: Remove location attributes
- location_and_state: Remove both state and location attributes

Notes:
- Uses temperature=0 and deterministic decoding
- Supports batch processing for efficiency
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')

# Import from the new modular structure
from src.eval_utils.description_processing.shared import setup_model
from src.eval_utils.description_processing.evaluator import evaluate
from src.eval_utils.description_processing.refiner import refine_descriptions
from src.eval_utils.description_processing import prompts


def main(args):
    """
    Main entry point for description evaluation and refinement.

    Args:
        args: Parsed command line arguments containing:
            - input: Path to input JSON dataset
            - out: Output directory or file path
            - refine: Refinement mode ('none', 'state', 'location', 'location_and_state')
            - batch_size: Batch size for processing
            - Other optional parameters
    """
    # Load the model
    model, tokenizer = setup_model("Qwen/Qwen3-8B")

    # Parse input path to generate appropriate output filename
    parsed_input = args.input.split('/')
    data_name, category_name, seed = parsed_input[1], parsed_input[2], parsed_input[3]
    if data_name == 'PerVA':
        filename = data_name + '_' + f'{category_name}_' + seed + '_' + '_'.join(parsed_input[-1].split('_')[1:])
    else:
        filename = data_name + '_' + seed + '_' + '_'.join(parsed_input[-1].split('_')[1:])
    out_path = Path(args.out) / filename

    # Choose operation based on refine mode
    if args.refine == 'none':
        # Evaluation mode: identify state and location attributes
        if 'descriptions' in args.input or 'database' in args.input:
            evaluate(args, model, tokenizer, prompts.prefix, prompts.suffix, out_path)
    else:
        # Refinement mode: remove state and/or location attributes
        # Select appropriate prompt based on refinement type
        if args.refine == 'state':
            prefix = prompts.prefix_state
        elif args.refine == 'location':
            prefix = prompts.prefix_location
        elif args.refine == 'location_and_state':
            prefix = prompts.prefix_location_and_state
        else:
            raise ValueError(f"Unknown refine mode: {args.refine}")

        # Load evaluation results
        with open(out_path, 'r') as f:
            batch_out = json.load(f)

        # Split text into general and distinguishing parts for refinement
        for i, response in enumerate(batch_out['response']):
            gen_text = response['text'].split('.')[0]
            dist_text = '.'.join(response['text'].split('.')[1:])
            batch_out['response'][i]['gen_text'] = gen_text
            batch_out['response'][i]['dist_text'] = dist_text

        # Refine the descriptions
        refined = refine_descriptions(args, model, tokenizer, prefix, prompts.suffix, batch_out)

        # Save refined results
        out_path = args.input.split('.')[0] + f'_{args.refine}_refined.json'
        with open(out_path, 'w') as f:
            json.dump(refined, f, indent=2)
        print(f'Refined version saved in {out_path}')


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Evaluate descriptions using Qwen-8B evaluator prompt")
    p.add_argument("--input", required=True, help="Path to input JSON dataset")
    p.add_argument("--refine", type=str, default='none',
                   choices=['none', 'state', 'location', 'location_and_state'],
                   help="Refinement mode: 'none' for evaluation only, or specify attributes to remove")
    p.add_argument("--out", required=True, help="Output directory or file path")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (number of descriptions per model call)")
    p.add_argument("--max-tokens", type=int, default=1024, help="Max new tokens for Qwen call")
    p.add_argument("--sleep-between-batches", type=float, default=0.0,
                   help="Optional sleep between batches (seconds)")
    p.add_argument("--fail-on-error", action="store_true",
                   help="Raise exception on parse/validation error")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
