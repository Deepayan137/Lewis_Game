#!/usr/bin/env python3
"""
generate_descriptions.py

Refactored: the main batch loop has been moved into an importable function
`run_description_generation(...)` so other scripts can reuse the generation logic.

Usage (as script):
    python generate_descriptions.py --model_type original ...

Usage (imported):
    from generate_descriptions import run_description_generation
    results_list = run_description_generation(model, processor, data_loader, max_new_tokens=128)
"""

import re
import json
import time
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable

import torch
from tqdm import tqdm

# Keep these imports but prefer explicit imports in your project:
# from inference_utils.dataset import SimpleImageDataset, create_data_loader
# from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.dataset import *
from inference_utils.model import *
from inference_utils.cleanup import extract_speaker_answer_term, parse_descriptions
from defined import yollava_reverse_category_dict, myvlm_reverse_category_dict, dbooth_reverse_category_dict
LOG = logging.getLogger(__name__)


def process_batch_efficiently(
    speaker_model,
    processor,
    batch_items: List[Dict[str, Any]],
    max_new_tokens: int = 128,
    num_return_sequences: int = 1,
    temperature: float = 1e-6
) -> List[Tuple[str, str]]:
    """
    Efficiently describe a batch of items using the speaker model.

    Expects `batch_items` to be an iterable of dicts with keys:
      - 'name'
      - 'image'
      - 'problem' (or prompt)
    Returns list[(name, description)].
    """
    names = [item['name'] for item in batch_items]
    images = [item['image'] for item in batch_items]
    problems = [item.get('problem', '') for item in batch_items]
    paths = [item['path'] for item in batch_items]
    with torch.no_grad():
        contents = speaker_describes_batch(
            speaker_model,
            processor,
            images,
            problems,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences
        )
    # If single response returned, normalize to list
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
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    eval_start_time = time.time()

    for batch_idx, batch_items in enumerate(tqdm(data_loader, desc="Processing batches")):
        batch_start_time = time.time()
        try:
            batch_results = process_batch_efficiently(
                speaker_model,
                processor,
                batch_items,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences
            )
            results.extend(batch_results)
        except Exception:
            LOG.exception("Failed processing batch %d; skipping.", batch_idx)
            continue

        batch_time = time.time() - batch_start_time
        try:
            n_in_batch = len(batch_items)
        except Exception:
            n_in_batch = 0
        samples_per_second = n_in_batch / batch_time if batch_time > 0 else float('inf')

        if batch_idx % log_every == 0:
            LOG.info("Batch %d: %.2f samples/sec (batch_time=%.2fs)", batch_idx, samples_per_second, batch_time)

    total_eval_time = time.time() - eval_start_time
    LOG.info("Finished generation for %d items in %.2f s (%.2f s/item)",
             len(results), total_eval_time, total_eval_time / max(1, len(results)))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optimized Lewis Game Evaluation')
    parser.add_argument("--data_name", type=str, default='PerVA',
                       help='name of the dataset')
    parser.add_argument("--catalog_file", type=str, default="main_catalog_seed_23.json",
                       help="Path to the catalog JSON file")
    parser.add_argument("--category", type=str, default='clothe',
                       help='Model category')
    parser.add_argument("--model_type", type=str, default='original', choices=['original_3b', 'original_7b', 'lora_finetuned_3b', 'lora_finetuned_7b'],
                       help='Model type: original or finetuned')
    parser.add_argument("--seed", type=int, default=42,
                       help='random seed')
    parser.add_argument("--batch_size", type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument("--max_new_tokens", type=int, default=128,
                       help='max tokens')
    parser.add_argument("--temperature", type=float, default=1e-6,
                       help='generation temperature')
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help='max tokens')
    parser.add_argument("--output_dir", type=str, default='outputs',
                       help='where to save outputs')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    LOG.info("Arguments: %s", args)

    # Choose model path
    if args.model_type == 'original_2b':
        model_path = "Qwen/Qwen2-VL-2B-Instruct"
    elif args.model_type == 'original_7b':
        model_path = "Qwen/Qwen2-VL-7B-Instruct"
    elif args.model_type == 'lora_finetuned_2b':
        model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2-VL-2B-Instruct_GRPO_lewis_LoRA_SPEAKER_PerVA_all_train_seed_{args.seed}"
    elif args.model_type == 'lora_finetuned_7b':
        model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2-VL-7B-Instruct_GRPO_lewis_LoRA_SPEAKER_PerVA_all_train_seed_{args.seed}"
    LOG.info("Loading model from %s", model_path)
    start_time = time.time()
    use_peft = False
    if args.model_type.startswith('lora_finetuned_7b') or args.model_type.startswith('lora_finetuned_2b'):
        use_peft = True
    speaker_model, processor = setup_model(model_path, use_peft=use_peft)
    LOG.info("Model loaded in %.1f s", time.time() - start_time)

    # Dataset + loader
    catalog_path = Path('manifests') / args.data_name / args.catalog_file
    dataset = SimpleImageDataset(
        json_path=str(catalog_path),
        category=args.category,
        split="train",
        seed=args.seed,
        data_name=args.data_name
    )
    data_loader = create_data_loader(dataset, batch_size=args.batch_size)
    # Run the now-importable generation function
    raw_results = run_description_generation(
        speaker_model,
        processor,
        data_loader,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        log_every=5,
    )

    # Build result dict and save cleaned descriptions
    savedir = Path(args.output_dir) / args.data_name / args.category / f'seed_{args.seed}'
    savedir.mkdir(parents=True, exist_ok=True)
    result_dict: Dict[str, Dict[str, str]] = {}
    response_dict = {"concept_dict":{}, "path_to_concept":{}}
    for name, image_path, desc in raw_results:
        if args.data_name == "YoLLaVA":
            category = yollava_reverse_category_dict[name]
        elif args.data_name == "MyVLM":
            category = myvlm_reverse_category_dict[name]
        elif args.data_name == 'DreamBooth':
            category = dbooth_reverse_category_dict[name]
        else:
            category = args.category
        if args.num_return_sequences > 1:
            desc_clean = {"coarse":[], "detailed":[]}
            desc_clean["coarse"] = [parse_descriptions(d)["coarse"] for d in desc]
            desc_clean["detailed"] = [parse_descriptions(d)["detailed"] for d in desc]
        else:
            desc_clean = parse_descriptions(desc[0])
            for key in ("coarse", "detailed"):
                if key == 'coarse':
                    desc_clean['coarse'] = desc_clean["coarse"] if desc_clean["coarse"] else f"A photo of a {category}"
                desc_clean[key] = [desc_clean[key]]
        result_dict[name] = {
            "name": name,
            "category":category,
            "general": desc_clean["coarse"],
            "distinguishing features": desc_clean["detailed"],
        }
        response_dict["concept_dict"][f'<{name}>'] = {
            "name": name,
            "category":category,
            "image": str(Path(image_path).resolve()),
            "info": {
                "category":category,
                "general": desc_clean["coarse"],
                "distinct features":desc_clean["detailed"],
            }
        }
        response_dict["path_to_concept"][str(Path(image_path).resolve())]=f'<{name}>'
    output_file = savedir / f"descriptions_{args.model_type}.json"
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    response_file = savedir / f"database_{args.model_type}.json"
    with response_file.open('w', encoding='utf-8') as f:
        json.dump(response_dict, f, indent=2, ensure_ascii=False)
    dest_dir = Path('../RAP/') / 'example_database' / f'{args.data_name}_seed_{args.seed}'
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Copy response_file to dest_dir and log success
    dest_file = dest_dir / response_file.name
    shutil.copy(str(response_file), str(dest_file))
    LOG.info("Copied database file to: %s", dest_file)
    # Print final statistics
    total_eval_time = 0.0
    if raw_results:
        # If needed you can compute timing stats more precisely by returning timings
        # from run_description_generation; for now we log simple counts.
        pass

    LOG.info("=== EVALUATION RESULTS ===")
    LOG.info("Total samples: %d", len(raw_results))
    LOG.info("Results saved to: %s", output_file)


if __name__ == "__main__":
    main()
