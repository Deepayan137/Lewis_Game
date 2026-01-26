from __future__ import annotations
import re
import json
import logging
import os
import string
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# local project imports (assume these exist and are correct)
from inference_utils.dataset import SimpleImageDataset, create_data_loader, DictListDataset, dict_collate_fn
from inference_utils.model import setup_model, speaker_describes_batch
from inference_utils.retriever import SimpleClipRetriever
from inference_utils.cleanup import extract_reasoning_answer_term, extract_speaker_answer_term
from generate_descriptions import run_description_generation
from defined import yollava_reverse_category_dict, myvlm_reverse_category_dict
from recognition import get_prompt
LOG = logging.getLogger(__name__)
os.environ["HF_HUB_OFFLINE"] = "1"


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class YoLLaVA_VQADataset(Dataset):
    def __init__(self, json_path, database, transform=None):
        # Read the JSON file and store the content
        self.data = []
        self.database = database
        with open(json_path, "r") as f:
            json_content = json.load(f)
        # The outer keys are concept names, with each value a dict of {img_path: vqa_info}
        for concept, samples in json_content.items():
            for image_path, entry in samples.items():
                # Each entry contains: question, options, correct_answer (plus possibly others)
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
        question = question.replace('<sks>', f'<{concept_name}>')  # FIX: concep_name -> concept_name
        question = question.strip('?') + ' in Image 1?'
        question = f'{question} Here are the options {options}. Read the options carefully before answering. Your answer must be either A or B.'
        ret_image_path = self.database["concept_dict"][f'<{concept_name}>']["image"]
        if isinstance(ret_image_path, list):  # FIX: use ret_image_path variable
            ret_path = ret_image_path[0]
        else:
            ret_path = ret_image_path
        ret_info = self.database["concept_dict"][f'<{concept_name}>']["info"]
        problem = get_prompt(ret_info, question, vqa=True)
        sample = {
            "problem": problem,
            "solution": item.get("correct_answer", None),
            "image_path": item.get("image_path", "").replace('./yollava-data/test', 'data/YoLLaVA/test/all'),
            "ret_path": ret_path,
            "concept": concept_name
        }
        return sample

def vqa_collate_fn(batch):
    """
    Collate function for YoLLaVA_VQADataset.
    Args:
        batch: list of samples. Each sample is a dict from __getitem__.
    Returns:
        A dict of lists/tensors with keys: image, question, options, correct_answer, image_path
    """
    problems = [item["problem"] for item in batch]
    solutions = [item["solution"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    ret_paths = [item["ret_path"] for item in batch]
    concepts = [item["concept"] for item in batch]
    return {
        "problems": problems,
        "solutions": solutions,
        "image_paths": image_paths,
        "ret_paths": ret_paths,
        "concepts": concepts
    }


def run_inference_loop(
    model,
    processor,
    qa_file,  # FIX: changed json_path to json_data (receives dict, not path)
    database,  # FIX: added database parameter
    temperature: float = 1e-6,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: torch.device | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:  # FIX: return type should be Dict not float
    """
    Run model on the prepared items and return (results_list, accuracy).
    results_list contains dicts with keys: image_path, problem, solution, solution_desc,
    ret_paths, response (raw model text), pred_name (cleaned)
    """
    dataset = YoLLaVA_VQADataset(qa_file, database)  # FIX: pass both json_data and database
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=vqa_collate_fn, num_workers=4
    )

    results: List[Dict[str, Any]] = []
    correct = 0  # FIX: simplified to single counter
    total = 0

    # device inference guidance (model may already be on device)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for batch in tqdm(loader, desc="Generating model responses"):
        # load images lazily if not provided
        images, ret_images = [], []
        for img_path, ret_path in zip(batch["image_paths"], batch["ret_paths"]):  # FIX: iterate over paths from batch dict
            images.append(Image.open(img_path).convert("RGB"))
            ret_images.append(Image.open(ret_path).convert("RGB"))
            
        problems = batch['problems']  # FIX: added problems extraction
        solutions = batch['solutions']
        paths = batch["image_paths"]
        ret_paths = batch["ret_paths"]
        try:
            responses = speaker_describes_batch(model, processor, problems, images, ret_images, temperature=temperature, max_new_tokens=max_new_tokens)
        except Exception:
            LOG.exception("Failed generating model responses for current batch; skipping.")
            # append placeholders for each item in batch and continue
            for path, gt, rp in zip(paths, solutions, ret_paths):
                results.append(
                    {
                        "image_path": path,
                        "problem": None,
                        "solution": gt,
                        "ret_path": rp,
                        "response": "",
                        "pred": "",  # FIX: changed pred_name to pred for consistency
                    }
                )
            continue

        # speaker_describes_batch may return a single string when batch_size==1
        if isinstance(responses, str):
            responses = [responses]
        # clean predicted "Answer" using provided utility
        predictions = [] 
        for resp in responses:
            try:
                if isinstance(resp, list):
                    resp = resp[0]
                term = extract_reasoning_answer_term(resp, "Answer")
                predictions.append(term.strip())
            except Exception:
                LOG.exception("Failed to extract answer term from model response; using empty string")
                predictions.append("")
        
        for path, gt, rp, prob, resp, pred in zip(paths, solutions, ret_paths, problems, responses, predictions):  # FIX: added prob to zip
            if gt.lower() == pred.lower():
                correct += 1
            total += 1
            results.append(
                {
                    "image_path": path,
                    "problem": prob,  # FIX: use actual problem, not response
                    "solution": gt,
                    "ret_path": rp,
                    "response": resp,
                    "pred": pred,
                }
            )

        # free cuda mem (harmless if CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "acc": accuracy,
        "correct": correct,
        "total": total,
    }
    return results, metrics


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning-based personalization evaluation")
    parser.add_argument("--data_name", type=str, default="YoLLaVA")
    parser.add_argument("--catalog_file", type=str, default="main_catalog_seed_23.json")
    parser.add_argument("--category", type=str, default="all")
    parser.add_argument("--concept_name", type=str, default="bo")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--db_type", type=str, default="original_7b")
    parser.add_argument("--model_type", type=str, default="original_7b")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7,
                       help='generation temperature')
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    set_seed(args.seed)
    LOG.info("Args: %s", args)
    database_json = Path("outputs") / args.data_name / args.category / f"seed_{args.seed}" / f"database_{args.db_type}.json"
    if not database_json.exists():
        LOG.error("Descriptions file not found at %s", database_json)
        raise FileNotFoundError(f"Descriptions file not found: {database_json}")

    # FIX: Load database
    with open(database_json, "r") as f:
        database = json.load(f)

    model_paths = {
        'original_2b': "Qwen/Qwen2-VL-2B-Instruct",
        'original_7b': "Qwen/Qwen2-VL-7B-Instruct",
    }

    # FIX: Define model_path and use_peft from args
    model_path = model_paths.get(args.model_type, model_paths['original_7b'])
    use_peft = 'lora' in args.model_type.lower()  # FIX: determine if PEFT should be used

    LOG.info("Loading model from %s", model_path)
    model, processor = setup_model(model_path, use_peft=use_peft)
    
    # run inference
    qa_file = f'data/YoLLaVA/yollava-visual-qa.json'
    # with open(qa_file, "r") as f:
    #     json_data = json.load(f)
    
    results, metrics = run_inference_loop(
        model,
        processor,
        qa_file,
        database,  # FIX: pass database
        temperature=args.temperature,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # save results & stats
    outdir = Path(args.output_dir) / args.data_name / args.category
    if args.concept_name != '':
        outdir = Path(outdir) / args.concept_name / f'seed_{str(args.seed)}'
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"recognition_model_{args.model_type}_db_{args.db_type}.json"
    output = {
        "metrics": metrics,
        "results": results
    }
    with outpath.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    
    LOG.info("Saved results to %s", outpath)
    LOG.info("Accuracy: %.4f", metrics['acc'])  # FIX: access metrics dict properly


if __name__ == "__main__":
    main()