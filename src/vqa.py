#!/usr/bin/env python3
"""
vqa.py

Visual Question Answering task for personalized concepts: Answer questions
about personalized objects/entities (e.g., "What is <bo> doing in Image 1?").

Usage:
    python vqa.py --data_name YoLLaVA --model_type original_7b --category all
"""

from __future__ import annotations

import json
import logging
import string
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from inference_utils.common import (
    set_seed,
    get_model_config,
    add_common_args,
    save_results,
    load_database,
    get_database_path,
    get_device,
    clear_cuda_cache,
)
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.cleanup import extract_reasoning_answer_term
from recognition import get_prompt

LOG = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class VQADataset(Dataset):
    """
    Dataset for Visual Question Answering on personalized concepts.

    Loads VQA data where questions reference personalized concepts
    (e.g., "Is <sks> happy in the image?").
    """

    def __init__(
        self,
        json_path: str,
        database: Dict[str, Any],
        data_name: str = "YoLLaVA",
    ):
        """
        Args:
            json_path: Path to VQA JSON file
            database: Database dict with concept descriptions
            data_name: Dataset name
        """
        self.database = database
        self.data_name = data_name
        self.data = []

        with open(json_path, "r") as f:
            json_content = json.load(f)

        # Parse nested structure: {concept: {image_path: vqa_info}}
        for concept, samples in json_content.items():
            for image_path, entry in samples.items():
                item = {
                    "image_path": image_path,
                    "question": entry.get("question", ""),
                    "options": entry.get("options", {}),
                    "correct_answer": entry.get("correct_answer", None),
                    "concept": concept,
                }
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get("question", "")
        concept_name = item.get("concept", "")
        options = item.get("options", {})

        # Replace <sks> placeholder with concept name
        question = question.replace('<sks>', f'<{concept_name}>')
        question = question.strip('?') + ' in Image 1?'
        question = (
            f'{question} Here are the options {options}. '
            'Read the options carefully before answering. '
            'Your answer must be either A or B.'
        )

        # Get reference image and info from database
        concept_key = f'<{concept_name}>'
        if concept_key in self.database.get("concept_dict", {}):
            concept_data = self.database["concept_dict"][concept_key]
            ret_image_path = concept_data.get("image")
            ret_info = concept_data.get("info", {})
        else:
            ret_image_path = None
            ret_info = {"category": "object", "general": ["Unknown"], "distinct features": ["Unknown"]}

        # Handle list vs string for image path
        if isinstance(ret_image_path, list):
            ret_path = ret_image_path[0]
        else:
            ret_path = ret_image_path

        # Build prompt
        problem = get_prompt(ret_info, question, vqa=True)

        # Fix image path for YoLLaVA dataset
        image_path = item.get("image_path", "")
        if self.data_name == "YoLLaVA":
            image_path = image_path.replace('./yollava-data/test', 'data/YoLLaVA/test/all')

        return {
            "problem": problem,
            "solution": item.get("correct_answer", None),
            "image_path": image_path,
            "ret_path": ret_path,
            "concept": concept_name,
            "options": options,
        }


def vqa_collate_fn(batch: List[Dict]) -> Dict[str, List]:
    """Collate function for VQADataset."""
    return {
        "problems": [item["problem"] for item in batch],
        "solutions": [item["solution"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "ret_paths": [item["ret_path"] for item in batch],
        "concepts": [item["concept"] for item in batch],
        "options": [item["options"] for item in batch],
    }


# ============================================================================
# Inference Loop
# ============================================================================

def run_inference_loop(
    model,
    processor,
    qa_file: str,
    database: Dict[str, Any],
    data_name: str = "YoLLaVA",
    temperature: float = 1e-6,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: torch.device = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Run VQA inference.

    Args:
        model: The loaded model
        processor: The model processor
        qa_file: Path to VQA JSON file
        database: Database with concept descriptions
        data_name: Dataset name
        temperature: Generation temperature
        batch_size: Batch size
        max_new_tokens: Maximum tokens to generate
        device: Torch device

    Returns:
        Tuple of (results_list, metrics_dict)
    """
    dataset = VQADataset(qa_file, database, data_name=data_name)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=vqa_collate_fn, num_workers=4
    )

    results: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    if device is None:
        device = get_device()

    for batch in tqdm(loader, desc="Running VQA inference"):
        # Load images
        images, ret_images = [], []
        for img_path, ret_path in zip(batch["image_paths"], batch["ret_paths"]):
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as e:
                LOG.warning("Failed to load image %s: %s", img_path, e)
                images.append(Image.new("RGB", (224, 224)))

            try:
                if ret_path:
                    ret_images.append(Image.open(ret_path).convert("RGB"))
                else:
                    ret_images.append(Image.new("RGB", (224, 224)))
            except Exception as e:
                LOG.warning("Failed to load ref image %s: %s", ret_path, e)
                ret_images.append(Image.new("RGB", (224, 224)))

        problems = batch['problems']
        solutions = batch['solutions']
        paths = batch["image_paths"]
        ret_paths = batch["ret_paths"]

        try:
            responses = speaker_describes_batch(
                model, processor, problems, images, ret_images,
                temperature=temperature, max_new_tokens=max_new_tokens
            )
        except Exception:
            LOG.exception("Failed generating responses for batch; skipping.")
            for path, gt, rp in zip(paths, solutions, ret_paths):
                results.append({
                    "image_path": path,
                    "problem": None,
                    "solution": gt,
                    "ret_path": rp,
                    "response": "",
                    "pred": "",
                    "correct": False,
                })
                total += 1
            continue

        # Normalize responses
        if isinstance(responses, str):
            responses = [responses]

        # Extract predictions
        predictions = []
        for resp in responses:
            try:
                if isinstance(resp, list):
                    resp = resp[0]
                term = extract_reasoning_answer_term(resp, "Answer")
                predictions.append(term.strip() if term else "")
            except Exception:
                LOG.exception("Failed to extract answer; using empty string")
                predictions.append("")

        # Accumulate results
        for path, gt, rp, prob, resp, pred in zip(
            paths, solutions, ret_paths, problems, responses, predictions
        ):
            is_correct = gt and pred and gt.lower() == pred.lower()
            if is_correct:
                correct += 1
            total += 1

            results.append({
                "image_path": path,
                "problem": prob,
                "solution": gt,
                "ret_path": rp,
                "response": resp[0] if isinstance(resp, list) else resp,
                "pred": pred,
                "correct": is_correct,
            })

        clear_cuda_cache()

    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

    return results, metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="VQA evaluation for personalized concepts")

    add_common_args(parser)

    # Task-specific args
    parser.add_argument("--qa_file", type=str, default=None,
                        help="Path to VQA JSON file. If not set, uses default for data_name.")

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = parse_args()

    set_seed(args.seed)
    LOG.info("Args: %s", args)

    # Load database
    database_path = get_database_path(
        args.data_name, args.category, args.seed, args.db_type
    )
    if not database_path.exists():
        LOG.error("Database file not found at %s", database_path)
        raise FileNotFoundError(f"Database file not found: {database_path}")

    LOG.info("Loading database from %s", database_path)
    database = load_database(database_path)

    # Get model configuration
    try:
        model_config = get_model_config(
            args.model_type, dataset=args.data_name, seed=args.seed
        )
    except ValueError:
        LOG.warning("Model type '%s' not in config, using as direct path", args.model_type)
        model_config = {
            'path': args.model_type,
            'use_peft': 'lora' in args.model_type.lower(),
        }

    model_path = model_config['path']
    use_peft = model_config['use_peft']

    LOG.info("Loading model from %s", model_path)
    model, processor = setup_model(model_path, use_peft=use_peft)

    # Determine QA file path
    if args.qa_file:
        qa_file = args.qa_file
    else:
        # Default paths per dataset
        qa_file_defaults = {
            "YoLLaVA": "data/YoLLaVA/yollava-visual-qa.json",
            "MyVLM": "data/MyVLM/myvlm-visual-qa.json",
        }
        qa_file = qa_file_defaults.get(args.data_name, f"data/{args.data_name}/visual-qa.json")

    LOG.info("Loading QA data from %s", qa_file)

    # Run inference
    results, metrics = run_inference_loop(
        model, processor, qa_file, database,
        data_name=args.data_name,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    outdir = Path(args.output_dir) / args.data_name / args.category
    if args.concept_name:
        outdir = outdir / args.concept_name
    outdir = outdir / f'seed_{args.seed}'
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"vqa_model_{args.model_type}_db_{args.db_type}.json"
    save_results(results, metrics, vars(args), outpath)

    LOG.info("Accuracy: %.4f (%d/%d)", metrics['accuracy'], metrics['correct'], metrics['total'])


if __name__ == "__main__":
    main()
