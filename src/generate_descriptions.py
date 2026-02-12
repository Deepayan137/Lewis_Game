#!/usr/bin/env python3
"""
generate_descriptions.py

Generate attribute-focused descriptions for reference images using a
(optionally LoRA-finetuned) vision-language model.

Usage (as script):
    python generate_descriptions.py --model_type original_7b --data_name PerVA ...

Usage (imported):
    from generate_descriptions import run_description_generation
    results_list = run_description_generation(model, processor, data_loader)
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm

from inference_utils.common import (
    set_seed,
    get_model_config,
    get_category_for_concept,
    add_common_args,
    save_results,
    clear_cuda_cache,
    DATASET_CATEGORY_MAPS,
)
from inference_utils.dataset import SimpleImageDataset, create_data_loader
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.cleanup import parse_descriptions

LOG = logging.getLogger(__name__)


# ============================================================================
# Core Generation Functions
# ============================================================================

def process_batch_efficiently(
    speaker_model,
    processor,
    batch_items: List[Dict[str, Any]],
    max_new_tokens: int = 128,
    num_return_sequences: int = 1,
    temperature: float = 1e-6,
) -> List[Tuple[str, str, Any]]:
    """
    Efficiently describe a batch of items using the speaker model.

    Args:
        speaker_model: The loaded model
        processor: The model processor
        batch_items: List of dicts with 'name', 'image', 'problem', 'path' keys
        max_new_tokens: Maximum tokens to generate
        num_return_sequences: Number of descriptions per image
        temperature: Sampling temperature

    Returns:
        List of (name, path, description) tuples
    """
    names = [item['name'] for item in batch_items]
    images = [item['image'] for item in batch_items]
    problems = [item.get('problem', '') for item in batch_items]
    paths = [item['path'] for item in batch_items]

    with torch.no_grad():
        contents = speaker_describes_batch(
            speaker_model,
            processor,
            problems,
            images,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
        )
    # Normalize to list if single response
    if isinstance(contents, str):
        contents = [contents]

    return list(zip(names, paths, contents))


def run_description_generation(
    speaker_model,
    processor,
    data_loader: Iterable,
    temperature: float = 1e-6,
    max_new_tokens: int = 128,
    num_return_sequences: int = 1,
    log_every: int = 5,
) -> List[Tuple[str, str, Any]]:
    """
    Run description generation over a data loader.

    Args:
        speaker_model: The loaded model
        processor: The model processor
        data_loader: DataLoader yielding batches of items
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        num_return_sequences: Number of descriptions per image
        log_every: Log progress every N batches

    Returns:
        List of (name, path, description) tuples
    """
    results: List[Tuple[str, str, Any]] = []
    eval_start_time = time.time()

    for batch_idx, batch_items in enumerate(tqdm(data_loader, desc="Generating descriptions")):
        batch_start_time = time.time()

        try:
            batch_results = process_batch_efficiently(
                speaker_model,
                processor,
                batch_items,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )
            results.extend(batch_results)
        except Exception:
            LOG.exception("Failed processing batch %d; skipping.", batch_idx)
            continue

        batch_time = time.time() - batch_start_time
        n_in_batch = len(batch_items) if hasattr(batch_items, '__len__') else 0
        samples_per_second = n_in_batch / batch_time if batch_time > 0 else float('inf')

        if batch_idx % log_every == 0:
            LOG.info(
                "Batch %d: %.2f samples/sec (batch_time=%.2fs)",
                batch_idx, samples_per_second, batch_time
            )

        clear_cuda_cache()

    total_eval_time = time.time() - eval_start_time
    LOG.info(
        "Finished generation for %d items in %.2f s (%.2f s/item)",
        len(results), total_eval_time, total_eval_time / max(1, len(results))
    )

    return results


# ============================================================================
# Result Processing
# ============================================================================

def process_raw_results(
    raw_results: List[Tuple[str, str, Any]],
    data_name: str,
    default_category: str,
    num_return_sequences: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process raw generation results into structured output dicts.

    Args:
        raw_results: List of (name, path, description) tuples
        data_name: Dataset name for category lookup
        default_category: Fallback category if not found in mapping
        num_return_sequences: Number of descriptions generated per image

    Returns:
        Tuple of (descriptions_dict, database_dict)
    """
    descriptions_dict: Dict[str, Any] = {}
    database_dict = {"concept_dict": {}, "path_to_concept": {}}

    for name, image_path, desc in raw_results:
        # Get category for this concept
        category = get_category_for_concept(name, data_name)
        if category == name:  # Fallback if not in mapping
            category = default_category

        # Parse descriptions
        if num_return_sequences > 1:
            desc_clean = {
                "coarse": [parse_descriptions(d)["coarse"] for d in desc],
                "detailed": [parse_descriptions(d)["detailed"] for d in desc],
            }
        else:
            parsed = parse_descriptions(desc[0])
            desc_clean = {
                "coarse": [parsed["coarse"] or f"A photo of a {category}"],
                "detailed": [parsed["detailed"]],
            }

        # Build descriptions dict
        descriptions_dict[name] = {
            "name": name,
            "category": category,
            "general": desc_clean["coarse"],
            "distinguishing features": desc_clean["detailed"],
        }

        # Build database dict
        resolved_path = str(Path(image_path).resolve())
        database_dict["concept_dict"][f'<{name}>'] = {
            "name": name,
            "category": category,
            "image": resolved_path,
            "info": {
                "category": category,
                "general": desc_clean["coarse"],
                "distinct features": desc_clean["detailed"],
            }
        }
        database_dict["path_to_concept"][resolved_path] = f'<{name}>'

    return descriptions_dict, database_dict


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate attribute-focused descriptions')

    # Use common args
    add_common_args(parser)

    # Description-specific args
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Number of descriptions to generate per image")
    parser.add_argument("--copy_to_rap", action="store_true",
                        help="Copy database file to RAP example_database directory")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    LOG.info("Arguments: %s", args)

    set_seed(args.seed)

    # Get model configuration
    try:
        model_config = get_model_config(
            args.model_type,
            dataset=args.data_name,
            seed=args.seed,
        )
    except ValueError:
        # Fallback for custom model types not in config
        LOG.warning("Model type '%s' not in config, using as direct path", args.model_type)
        model_config = {
            'path': args.model_type,
            'use_peft': 'lora' in args.model_type.lower(),
        }

    model_path = model_config['path']
    use_peft = model_config['use_peft']

    LOG.info("Loading model from %s (use_peft=%s)", model_path, use_peft)
    start_time = time.time()
    speaker_model, processor = setup_model(model_path, use_peft=use_peft)
    LOG.info("Model loaded in %.1f s", time.time() - start_time)

    # Create dataset and loader
    catalog_path = Path('manifests') / args.data_name / args.catalog_file
    dataset = SimpleImageDataset(
        json_path=str(catalog_path),
        category=args.category,
        split="train",
        seed=args.seed,
        data_name=args.data_name,
    )
    data_loader = create_data_loader(dataset, batch_size=args.batch_size)

    # Run generation
    raw_results = run_description_generation(
        speaker_model,
        processor,
        data_loader,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        log_every=5,
    )

    # Process results
    descriptions_dict, database_dict = process_raw_results(
        raw_results,
        data_name=args.data_name,
        default_category=args.category,
        num_return_sequences=args.num_return_sequences,
    )

    # Save outputs
    savedir = Path(args.output_dir) / args.data_name / args.category / f'seed_{args.seed}'
    savedir.mkdir(parents=True, exist_ok=True)

    desc_file = savedir / f"descriptions_{args.model_type}.json"
    with desc_file.open('w', encoding='utf-8') as f:
        json.dump(descriptions_dict, f, indent=2, ensure_ascii=False)

    db_file = savedir / f"database_{args.model_type}.json"
    with db_file.open('w', encoding='utf-8') as f:
        json.dump(database_dict, f, indent=2, ensure_ascii=False)

    LOG.info("Descriptions saved to: %s", desc_file)
    LOG.info("Database saved to: %s", db_file)

    # Optionally copy to RAP directory
    if args.copy_to_rap:
        dest_dir = Path('../RAP/example_database') / f'{args.data_name}_seed_{args.seed}'
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / db_file.name
        shutil.copy(str(db_file), str(dest_file))
        LOG.info("Copied database file to: %s", dest_file)

    LOG.info("=== GENERATION COMPLETE ===")
    LOG.info("Total samples: %d", len(raw_results))


if __name__ == "__main__":
    main()
