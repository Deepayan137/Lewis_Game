"""
Description Refiner

This module handles the refinement of descriptions by removing state-specific
and/or location-specific attributes.

Main functionality:
- refine(): Refine descriptions by removing unwanted attributes
- Supports removing state attributes, location attributes, or both
"""

import json
from typing import List, Dict, Any

from .shared import JsonDescriptionsDataset, collate_batch, infer_batch


def refine(args, model, tokenizer, prefix: str, suffix: str, batch_out: List[Dict[str, Any]],
           feat_key: str, refined: Dict[str, Any]) -> Dict[str, Any]:
    """
    Refine descriptions by removing state and/or location attributes.

    This function processes evaluation results and uses the model to generate
    refined versions of descriptions with unwanted attributes removed.

    Args:
        args: Argument object containing:
            - input: Path to input JSON dataset
        model: The loaded model for inference
        tokenizer: The tokenizer for the model
        prefix: Refinement prompt prefix (specifies what to remove)
        suffix: Refinement prompt suffix
        batch_out: List of evaluation results to refine
        feat_key: Which feature to refine ('gen_text' or 'dist_text')
        refined: Dictionary to accumulate refined results (may already contain partial results)

    Returns:
        Updated refined dictionary with new refinements added

    Processing details:
    - Processes items in small batches (default 6)
    - Updates the original data structure with refined text
    - For 'gen_text': updates the "general" field
    - For 'dist_text': updates the "distinguishing features" field
    """
    ds = JsonDescriptionsDataset(args.input)

    def batch_iterator(batch_out: List[Dict[str, Any]], batch_size: int = 6):
        """
        Yield batches of specified size from batch_out.

        Args:
            batch_out: List of items to batch
            batch_size: Size of each batch

        Yields:
            Lists of items, each of size batch_size (except possibly the last)
        """
        batch = []
        for item in batch_out:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # Process each batch
    for batch in batch_iterator(batch_out):
        items = collate_batch(batch, feat_key=feat_key)
        parsed = infer_batch(model, tokenizer, batch, prefix, suffix, refine=True, feat_key=feat_key)

        # Update refined dictionary with results
        for item in parsed:
            key = item['id']
            if key not in refined:
                refined[key] = ds.data[key]
            if 'text' in item:
                if feat_key == 'gen_text':
                    refined[key]["general"] = [item['text']]
                else:
                    refined[key]["distinguishing features"] = [item['text']]

        print(f"batch num {len(refined)} is processed")

    return refined


def refine_descriptions(args, model, tokenizer, prefix: str, suffix: str,
                        batch_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for description refinement.

    This function:
    1. Splits descriptions into general and distinguishing features
    2. Refines each part separately using the refine() function
    3. Returns the complete refined dataset

    Args:
        args: Argument object containing input path and other settings
        model: The loaded model for inference
        tokenizer: The tokenizer for the model
        prefix: Refinement prompt prefix
        suffix: Refinement prompt suffix
        batch_out: Dictionary containing:
            - 'response': List of evaluation results with text to refine

    Returns:
        Dictionary mapping IDs to refined description objects

    Note:
        Assumes descriptions can be split at the first period:
        - Before first period: general text (gen_text)
        - After first period: distinguishing text (dist_text)
    """
    # Split descriptions into general and distinguishing parts
    for i, response in enumerate(batch_out['response']):
        gen_text = response['text'].split('.')[0]
        dist_text = '.'.join(response['text'].split('.')[1:])
        batch_out['response'][i]['gen_text'] = gen_text
        batch_out['response'][i]['dist_text'] = dist_text

    # Refine both parts
    parsed = {}
    for feat_key in ["gen_text", "dist_text"]:
        parsed = refine(args, model, tokenizer, prefix, suffix, batch_out['response'], feat_key, parsed)

    return parsed
