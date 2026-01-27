"""
Description Processing Module

This module provides functionality for evaluating and refining object descriptions,
specifically focusing on identifying and removing state-specific and location-specific
attributes.

Main components:
- evaluator: Evaluate descriptions to identify state/location attributes
- refiner: Refine descriptions by removing unwanted attributes
- prompts: Prompt templates for the Qwen model
- shared: Common utilities (dataset, model setup, inference)

Usage examples:

    # Evaluation
    from src.eval_utils.description_processing.evaluator import evaluate
    from src.eval_utils.description_processing.prompts import prefix, suffix
    from src.eval_utils.description_processing.shared import setup_model

    model, tokenizer = setup_model("Qwen/Qwen3-8B")
    evaluate(args, model, tokenizer, prefix, suffix, output_path)

    # Refinement
    from src.eval_utils.description_processing.refiner import refine_descriptions
    from src.eval_utils.description_processing.prompts import prefix_state, suffix

    refined = refine_descriptions(args, model, tokenizer, prefix_state, suffix, batch_out)
"""

from .evaluator import evaluate, aggregate_stats
from .refiner import refine, refine_descriptions
from .shared import (
    JsonDescriptionsDataset,
    collate_batch,
    setup_model,
    infer_batch
)

# Import prompts for convenience
from . import prompts

__all__ = [
    # Evaluator
    'evaluate',
    'aggregate_stats',

    # Refiner
    'refine',
    'refine_descriptions',

    # Shared utilities
    'JsonDescriptionsDataset',
    'collate_batch',
    'setup_model',
    'infer_batch',

    # Prompts module
    'prompts',
]

__version__ = '1.0.0'
