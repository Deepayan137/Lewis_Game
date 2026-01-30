#!/usr/bin/env python3
"""
recognition.py

Binary recognition task: Given a query image (Image 1) and a reference image
(Image 2) with its description, determine if they show the same object.

Usage:
    python recognition.py --data_name PerVA --model_type original_7b --category all
"""

from __future__ import annotations

import json
import logging
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from inference_utils.dataset import SimpleImageDataset
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
from inference_utils.dataset import DictListDataset, dict_collate_fn
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.cleanup import extract_reasoning_answer_term

LOG = logging.getLogger(__name__)


# ============================================================================
# Image Marker Utility
# ============================================================================

def add_marker_to_image(image: Image.Image, marker_text: str, position: str = "top_right",
                        font_size: int = 32, bg_color: Tuple[int, int, int] = (255, 0, 0),
                        text_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Add a numerical marker (circle with number) to an image corner.

    Args:
        image: PIL Image object
        marker_text: Text to display (e.g., "1" or "2")
        position: Where to place the marker ("top_right", "top_left")
        font_size: Size of the marker text
        bg_color: Background color of the marker circle (RGB tuple)
        text_color: Color of the text (RGB tuple)

    Returns:
        PIL Image with marker added
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Try to load a font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Calculate text dimensions
    bbox = draw.textbbox((0, 0), marker_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 8
    circle_radius = max(text_width, text_height) // 2 + padding

    img_width, img_height = img.size
    margin = 10

    if position == "top_right":
        center_x = img_width - margin - circle_radius
        center_y = margin + circle_radius
    else:  # top_left
        center_x = margin + circle_radius
        center_y = margin + circle_radius

    # Draw circle background
    draw.ellipse(
        [center_x - circle_radius, center_y - circle_radius,
         center_x + circle_radius, center_y + circle_radius],
        fill=bg_color
    )

    # Draw text centered in circle
    text_x = center_x - text_width // 2
    text_y = center_y - text_height // 2
    draw.text((text_x, text_y), marker_text, font=font, fill=text_color)

    return img


# ============================================================================
# Prompt Generation
# ============================================================================

def get_prompt(description: Dict[str, Any], test_question: str, vqa: bool = False) -> str:
    """
    Build the prompt for binary recognition or VQA task.

    Args:
        description: Dict with reference info (category, general, distinct features)
        test_question: The question to answer
        vqa: Whether this is a VQA task (changes answer format)

    Returns:
        Formatted prompt string
    """
    if vqa:
        answer_format = {
            "Reasoning": "<Brief comparison based on key attributes>",
            "Answer": "<A or B>"
        }
    else:
        answer_format = {
            "Reasoning": "<Brief comparison based on key attributes>",
            "Answer": "<yes or no>"
        }

    test_question = test_question.replace('the image', 'the first image')

    prompt = (
        f"You are a helpful AI agent specializing in image analysis and object recognition\n\n"
        f"You are given two images (marked with numbers 1 and 2 in the top-right corner). "
        f"Additionally, the name and a textual description of the subject "
        f"in Image 2 is also provided below:\n\n"
        f"{json.dumps(description, indent=2)}\n"
        f"Your Task:\n"
        f"- Compare Image 1 with Image 2 and answer the following question: "
        f"{test_question}\n"
        f"- **Ignore superficial details** such as clothing, accessories, pose variations, or "
        f"surrounding elements (e.g., people in the background).\n"
        f"- Focus only on non-variant/permanent features such as color, shape, pattern, text for "
        f"objects/buildings and facial features for people.\n"
        f"- If you are uncertain then you can refer the textual description of Image 2 "
        f"to make a more informed decision.\n"
        f"**Output (JSON only):**\n{json.dumps(answer_format, indent=2)}"
    )

    return prompt


# ============================================================================
# Data Preparation
# ============================================================================

def copy_descriptions_to_database(
    database: Dict[str, Any],
    descriptions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Copy description fields from descriptions to database.

    Args:
        database: The database dict with concept_dict structure
        descriptions: Dict with concept descriptions containing "general" and "distinguishing features"

    Returns:
        Modified database with copied description fields
    """
    for concept_name in descriptions:
        concept_key = f"<{concept_name}>"
        if concept_key in database["concept_dict"]:
            if "info" not in database["concept_dict"][concept_key]:
                database["concept_dict"][concept_key]["info"] = {}
            database["concept_dict"][concept_key]["info"]["general"] = descriptions[concept_name]["general"]
            database["concept_dict"][concept_key]["info"]["distinct features"] = descriptions[concept_name]["distinguishing features"]
    return database


def prepare_test_recognition_items(
    args,
    database,
    catalog_json: Path,
    category: str,
    concept: str = None,
    ):
    # LOG.info("Loading database from: %s", database_json)  # FIX: changed catalog_json to database_json
    # with open(database_json, 'r') as f:
    #     database = json.load(f)
    LOG.info("Loading dataset from: %s", catalog_json)
    dataset = SimpleImageDataset(
        json_path=catalog_json,
        category=category,
        split="test",
        seed=args.seed,
        data_name=args.data_name
    )
    recognition_samples = []
    concept_names = sorted({item['name'] for item in dataset})
    for sub_item in dataset:
        query_path = sub_item["path"]
        query_name = sub_item["name"]
        for concept_name in concept_names:
            # if concept_name == query_name:
            ret_image_path = database["concept_dict"][f'<{concept_name}>']["image"]
            if isinstance(database["concept_dict"][f'<{concept_name}>']["image"], list):  # FIX: changed concept to concept_name
                ret_path = database["concept_dict"][f'<{concept_name}>']["image"][0]  # FIX: changed concept to concept_name
            else:  # FIX: added else clause for non-list case
                ret_path = database["concept_dict"][f'<{concept_name}>']["image"]
            ret_info = database["concept_dict"][f'<{concept_name}>']["info"]  # FIX: changed tag to concept_name, info to ret_info
            sample = {
                'concept_name': concept_name,
                "name": query_name,
                "query_path": query_path,
                "ret_path": ret_path,  # FIX: changed comma to colon
                "ret_info": ret_info,    
                "question": f"Is <{concept_name}> in the first image? Answer in yes or no.",
                "solution": 'yes' if query_name == concept_name else 'no'
            }
            recognition_samples.append(sample)
    return recognition_samples


# ============================================================================
# Inference Loop
# ============================================================================

def run_inference_loop(
    model,
    processor,
    items: Iterable[Dict[str, Any]],
    temperature: float = 1e-6,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: torch.device = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run recognition inference over items.

    Args:
        model: The loaded model
        processor: The model processor
        items: Iterable of prepared item dicts
        temperature: Generation temperature
        batch_size: Batch size
        max_new_tokens: Maximum tokens to generate
        device: Torch device

    Returns:
        Tuple of (results_list, metrics_dict)
    """
    dataset = DictListDataset(list(items))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=dict_collate_fn, num_workers=4
    )

    results: List[Dict[str, Any]] = []
    correct_yes, correct_no = 0, 0
    total_yes, total_no = 0, 0

    if device is None:
        device = get_device()

    for batch in tqdm(loader, desc="Running recognition inference"):
        # Load images
        images, ret_images = [], []
        for it in batch:
            try:
                query_img = Image.open(it["query_path"]).convert("RGB")
                images.append(add_marker_to_image(query_img, "1"))
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
            try:
                ret_img = Image.open(it["ret_path"]).convert("RGB")
                ret_images.append(add_marker_to_image(ret_img, "2"))
            except Exception:
                ret_images.append(Image.new("RGB", (224, 224)))
        questions = [it["question"] for it in batch]
        problems = [get_prompt(it['ret_info'], it["question"]) for it in batch]
        solutions = [it["solution"] for it in batch]
        query_paths = [it["query_path"] for it in batch]
        ref_paths = [it["ret_path"] for it in batch]

        try:
            responses = speaker_describes_batch(
                model, processor, problems, images, ret_images,
                temperature=temperature, max_new_tokens=max_new_tokens
            )
        except Exception:
            LOG.exception("Failed generating responses for batch; skipping.")
            for qp, ques, sol, rp in zip(query_paths, questions, solutions, ref_paths):
                results.append({
                    "query_path": qp,
                    "question": ques,
                    "ref_path": rp,
                    "problem": None,
                    "solution": sol,
                    "response": "",
                    "pred": "",
                    "correct": False,
                })
                if sol.lower() == "yes":
                    total_yes += 1
                else:
                    total_no += 1
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
                predictions.append("")

        # Accumulate results
        for qp, ques, sol, rp, prob, resp, pred in zip(
            query_paths, questions, solutions, ref_paths, problems, responses, predictions
        ):
            pred_lower = pred.lower().strip()
            sol_lower = sol.lower().strip()

            is_correct = pred_lower == sol_lower

            if sol_lower == "yes":
                total_yes += 1
                if is_correct:
                    correct_yes += 1
            else:
                total_no += 1
                if is_correct:
                    correct_no += 1

            results.append({
                "query_path": qp,
                "question": ques,
                "ref_path": rp,
                "problem": prob,
                "solution": sol,
                "response": resp[0] if isinstance(resp, list) else resp,
                "pred": pred,
                "correct": is_correct,
            })

        clear_cuda_cache()

    total = total_yes + total_no
    correct = correct_yes + correct_no
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "correct_yes": correct_yes,
        "correct_no": correct_no,
        "total_yes": total_yes,
        "total_no": total_no,
        "accuracy_yes": correct_yes / total_yes if total_yes > 0 else 0.0,
        "accuracy_no": correct_no / total_no if total_no > 0 else 0.0,
    }

    return results, metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Binary recognition evaluation")

    add_common_args(parser)

    # Task-specific args
    parser.add_argument("--retrieval_json", type=str, default=None,
                        help="Path to retrieval JSON. If not set, auto-generated from args.")

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
    if 'refined' in args.db_type:
        database_path = get_database_path(
            args.data_name, args.category, args.seed, 'original_7b'
        )

    else:
        database_path = get_database_path(
            args.data_name, args.category, args.seed, args.db_type
        )
    LOG.info("Loading database from %s", database_path)
    database = load_database(database_path)
    if 'refined' in args.db_type:
        descriptions = load_database(Path(f"outputs/{args.data_name}/all/seed_{args.seed}/descriptions_original_7b_location_and_state_refined.json"))
        database = copy_descriptions_to_database(database, descriptions)
        # database_path = get_database_path(
        #     args.data_name, args.category, args.seed, 'original_7b_location_and_state_refined'
        # )
        # with open(database_path, 'w') as f:
        #     json.dump(database, f, indent=2)
        
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

    # Determine retrieval JSON path
    manifests_dir = Path("manifests") / args.data_name
    catalog_path = str(manifests_dir / f'main_catalog_seed_{args.seed}.json')

    LOG.info("Loading retrieval data from %s", catalog_path)
    # Prepare items
    items = prepare_test_recognition_items(
        args,
        database, 
        catalog_path,
        args.category)
    if args.concept_name != '':
        items = [item for item in items if item.get("concept_name") == args.concept_name]  # FIX: changed "solution" to "concept_name"
    LOG.info("Prepared %d test items", len(items))
    # Run inference
    results, metrics = run_inference_loop(
        model, processor, items,
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # Save results
    outdir = Path('results') / args.data_name / args.category
    if args.concept_name:
        outdir = outdir / args.concept_name
    outdir = outdir / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"recognition_model_{args.model_type}_db_{args.db_type}.json"
    save_results(results, metrics, vars(args), outpath)

    LOG.info("Overall Accuracy: %.4f (%d/%d)", metrics['accuracy'], metrics['correct'], metrics['total'])
    LOG.info("Accuracy (Yes): %.4f, Accuracy (No): %.4f", metrics['accuracy_yes'], metrics['accuracy_no'])


if __name__ == "__main__":
    main()

