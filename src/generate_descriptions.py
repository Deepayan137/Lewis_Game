import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from tqdm import tqdm

# Keep these imports but prefer explicit imports in your project:
# from inference_utils.dataset import SimpleImageDataset, create_data_loader
# from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.dataset import *
from inference_utils.model import *
from inference_utils.clean import extract_speaker_answer_term

LOG = logging.getLogger(__name__)

def process_batch_efficiently(
    speaker_model,
    processor,
    batch_items: List[Dict[str, Any]],
    max_new_tokens: int = 128,
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

    with torch.no_grad():
        contents = speaker_describes_batch(
            speaker_model,
            processor,
            images,
            problems,
            max_new_tokens=max_new_tokens,
        )

    return list(zip(names, contents))


def save_results(result_dict: Dict[str, Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as fh:
        json.dump(result_dict, fh, indent=2, ensure_ascii=False)


def compute_stats(total_time: float, n_samples: int) -> Dict[str, float]:
    avg_time = total_time / n_samples if n_samples else float('nan')
    s_per_sec = n_samples / total_time if total_time > 0 else float('nan')
    return {
        "total_seconds": total_time,
        "avg_time_per_sample": avg_time,
        "samples_per_second": s_per_sec,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optimized Lewis Game Evaluation')
    parser.add_argument("--data_name", type=str, default='PerVA', help='name of the dataset')
    parser.add_argument("--catalog_file", type=str, default="test_catalog_seed_23.json", help="Path to the catalog JSON file")
    parser.add_argument("--category", type=str, default='clothe', help='dataset category')
    parser.add_argument("--model_type", type=str, default='original', choices=['original', 'finetuned'], help='Model type')
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    parser.add_argument("--batch_size", type=int, default=4, help='Batch size for processing')
    parser.add_argument("--max_new_tokens", type=int, default=128, help='max tokens')
    parser.add_argument("--output_dir", type=str, default='outputs', help='where to save outputs')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    LOG.info("Arguments: %s", args)

    # Choose model path
    if args.model_type == 'original':
        model_path = "Qwen/Qwen2-VL-2B-Instruct"
    else:
        model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2.5-VL-2B-Instruct_GRPO_lewis_{args.category}_test_subset"

    LOG.info("Loading model from %s", model_path)
    start_time = time.time()
    speaker_model, processor = setup_model(model_path)
    LOG.info("Model loaded in %.1f s", time.time() - start_time)

    # Dataset + loader
    catalog_path = Path('manifests') / args.data_name / args.catalog_file
    dataset = SimpleImageDataset(
        json_path=str(catalog_path),
        category=args.category,
        split="train",
        seed=args.seed
    )
    data_loader = create_data_loader(dataset, batch_size=args.batch_size)

    results: List[Tuple[str, str]] = []
    eval_start_time = time.time()

    # iterate batches
    for batch_idx, batch_items in enumerate(tqdm(data_loader, desc="Processing batches")):
        batch_start_time = time.time()

        # Ensure batch_items is a list-like of dicts. If your loader returns dict-of-tensors,
        # wrap/convert it to the structure expected by process_batch_efficiently.
        try:
            batch_results = process_batch_efficiently(
                speaker_model,
                processor,
                batch_items,
                max_new_tokens=args.max_new_tokens,
            )
            results.extend(batch_results)
        except Exception:
            LOG.exception("Failed processing batch %d", batch_idx)
            continue

        batch_time = time.time() - batch_start_time
        n_in_batch = len(batch_items) if hasattr(batch_items, '__len__') else args.batch_size
        samples_per_second = n_in_batch / batch_time if batch_time > 0 else float('inf')

        if (batch_idx + 1) % 5 == 0:
            LOG.info("Batch %d: %.2f samples/sec (batch_time=%.2fs)", batch_idx, samples_per_second, batch_time)

    total_eval_time = time.time() - eval_start_time

    # build dict and save
    savedir = Path(args.output_dir) / args.data_name / args.category
    output_file = savedir / f"descriptions_{args.model_type}.json"

    result_dict: Dict[str, Dict[str, str]] = {}
    for name, desc in results:
        desc_clean = extract_speaker_answer_term(desc)
        result_dict[name] = {
            "name": name,
            "distinguishing features": desc_clean
        }

    save_results(result_dict, output_file)

    # stats
    stats = compute_stats(total_eval_time, len(results))
    LOG.info("=== EVALUATION RESULTS ===")
    LOG.info("Total samples: %d", len(results))
    LOG.info("Total evaluation time: %.2f seconds", stats["total_seconds"])
    LOG.info("Average time per sample: %.4f seconds", stats["avg_time_per_sample"])
    LOG.info("Samples per second: %.4f", stats["samples_per_second"])
    LOG.info("Results saved to: %s", output_file)

    # example speedup calculation (keep original value if desired)
    original_time_per_sample = 23.33
    try:
        speedup = original_time_per_sample / stats["avg_time_per_sample"]
    except Exception:
        speedup = float('nan')
    LOG.info("Speedup over original: %.2fx", speedup)


if __name__ == "__main__":
    main()
