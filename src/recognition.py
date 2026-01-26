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
LOG = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_prompt(description: str, test_question: str, vqa=False) -> str:
    """
    Build the prompt used to ask the model to reason and return a JSON
    with keys "Reasoning" and "Answer".
    Uses the description to ground attributes of both images before comparison.
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
        f"You are given two images, additionally, the name and a textual description of the subject in the second image is also provided below:\n\n"
        f"{json.dumps(description, indent=2)}\n"
        f"Your Task:\n"
        f"- Compare the first image with the second image and answer the following question:"
        f"{test_question}"
        f"-**Ignore superficial details** such as clothing, accessories, pose variations, or surrounding elements (e.g., people in the background).\n"
        f"- Focus only on non-variant/permanent features such as color, shape, pattern, text for objects/buildings and facial features for people.\n"
        f"- If you are uncertain then you can refer the textual description of the second image to make a more informed decision.\n"
        f"**Output (JSON only):**\n{json.dumps(answer_format, indent=2)}"
    )
    # prompt = (
    #     f"You are a helpful AI agent specializing in image analysis and object recognition\n\n"
    #     f"You are provided with a query image along with the name and description of a subject detailed below:\n\n"
    #     f"{json.dumps(description, indent=2)}\n"
    #     f"Your Task:\n"
    #     f"1. Generate an attribute-focused description of the subject in the query image. "
    #     "Focus on its distinguishing features rather than superficial details such as background, pose, lighting, clothes or accessories.\n"
    #     f"2. Compare your generated description of the query image with the provided description of the subject and answer the following question:\n"
    #     f"{test_question}"
    #     "Output Requirements:\n"
    #     f"- Your response MUST be a valid JSON exactly matching the format:\n{json.dumps(answer_format)}\n"
    #     "- Do not include any extra text, explanations, or formatting outside of the JSON.\n"
    # )
    return prompt


def run_inference_loop(
    model,
    processor,
    items: Iterable[Dict[str, Any]],
    temperature: float = 1e-6,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: torch.device | None = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run model on the prepared items and return (results_list, accuracy).
    results_list contains dicts with keys: image_path, problem, solution, solution_desc,
    ret_paths, response (raw model text), pred_name (cleaned)
    """
    dataset = DictListDataset(list(items))
    # dataset = dataset[:10]
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dict_collate_fn, num_workers=4
    )

    results: List[Dict[str, Any]] = []
    correct_yes, correct_no, total_yes, total_no = 0, 0, 0, 0

    # device inference guidance (model may already be on device)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for batch in tqdm(loader, desc="Generating model responses"):
        # load images lazily if not provided
        images, ret_images = [], []
        for it in batch:
            images.append(Image.open(it["query_path"]).convert("RGB"))
            ret_images.append(Image.open(it["ret_path"]).convert("RGB"))  # FIX: was query_path, should be ret_path
            
        problems = [get_prompt(it['ret_info'], it["question"]) for it in batch]  # FIX: added missing comma
        solutions = [it["solution"] for it in batch]
        paths = [it.get("query_path") for it in batch]  # FIX: changed "path" to "query_path"
        ret_paths = [it.get("ret_path", []) for it in batch]
        try:
            responses = speaker_describes_batch(model, processor, problems, images, ret_images, temperature=temperature, max_new_tokens=max_new_tokens)
        except Exception:
            LOG.exception("Failed generating model responses for current batch; skipping.")
            # append placeholders for each item in batch and continue
            for path, gt, rp in zip(paths, solutions, ret_paths):  # FIX: removed sol_desc from unpacking
                results.append(
                    {
                        "image_path": path,
                        "problem": None,
                        "solution": gt,
                        "ret_path": rp,  # FIX: added ret_path
                        "response": "",
                        "pred_name": "",
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

        for path, gt, rp, resp, pred in zip(paths, solutions, ret_paths, responses, predictions):  # FIX: added ret_paths to zip
            if gt.lower() == 'yes':
                correct_yes += int(pred.lower() == gt.lower())
                total_yes +=1
            elif gt.lower() == 'no':
                correct_no += int(pred.lower() == gt.lower())
                total_no +=1
            is_correct = gt.lower() in pred.lower()
            # correct_count += is_correct
            results.append(
                {
                    "image_path": path,
                    "problem": resp if resp else None,
                    "solution": gt,
                    "ret_path": rp,  # FIX: added ret_path
                    "response": resp,
                    "pred": pred,
                    "is_correct": bool(is_correct),
                }
            )

        # free cuda mem (harmless if CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    positive_accuracy = correct_yes / total_yes if total_yes > 0 else 0.0
    negative_accuracy = correct_no / total_no if total_no > 0 else 0.0
    weighted_accuracy = 0.5*(positive_accuracy + negative_accuracy)
    metrics = {
        "pos_acc":positive_accuracy,
        "neg_acc":negative_accuracy,
        "weighted":weighted_accuracy,
        "correct_yes":correct_yes,
        "correct_no":correct_no,
        "total_yes":total_yes,
        "total_no":total_no
    }
    return results, metrics


def prepare_test_recognition_items(
    args,
    database_json: Path,
    catalog_json: Path,
    category: str,
    ):
    LOG.info("Loading database from: %s", database_json)  # FIX: changed catalog_json to database_json
    with open(database_json, 'r') as f:
        database = json.load(f)
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

def parse_args():  # FIX: removed -> argparse.Namespace type hint (argparse not imported yet)
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

    manifests_dir = Path("manifests") / args.data_name
    database_json = Path("outputs") / args.data_name / args.category / f"seed_{args.seed}" / f"database_{args.db_type}.json"
    catalog_json = str(manifests_dir / args.catalog_file)
    if not database_json.exists():
        LOG.error("Descriptions file not found at %s", database_json)
        raise FileNotFoundError(f"Descriptions file not found: {database_json}")

    # FIX: retriever was never defined but used in prepare_test_recognition_items
    # You need to initialize it here, e.g.:
    
    # prepare items to run inference on
    items = prepare_test_recognition_items(
        args,
        database_json, 
        catalog_json,
        args.category)
    LOG.info("Prepared %d test items", len(items))
    if args.concept_name != '':
        items = [item for item in items if item.get("concept_name") == args.concept_name]  # FIX: changed "solution" to "concept_name"
    # load model
    model_paths = {
        'original_2b': "Qwen/Qwen2-VL-2B-Instruct",
        'original_7b': "Qwen/Qwen2-VL-7B-Instruct",
        # 'lora_finetuned_3b_base': f"../Visual-RFT/share_models/Qwen2.5-VL-3B-Instruct_GRPO_lewis_LoRA_LISTENER_PerVA_all_train_seed_{args.seed}_K_3_base_3b",
        # 'lora_finetuned_7b_base': f"../Visual-RFT/share_models/Qwen2.5-VL-7B-Instruct_GRPO_lewis_LoRA_LISTENER_PerVA_all_train_seed_{args.seed}_K_3_base_7b"    
    }

    if args.model_type in ['original_2b', 'original_7b']:
        model_path = model_paths[args.model_type]
    else:
        model_path = model_paths.get((args.model_type, args.k_retrieval))
        # model_path = f"../Visual-RFT/share_models/Qwen2.5-VL-7B-Instruct_GRPO_lewis_LISTENER_prompt2_epoch2_YoLLaVA_all_train_seed_23"
    LOG.info("Loading model from %s", model_path)
    use_peft = False
    if args.model_type.startswith('lora_finetuned'):
        use_peft = True
    model, processor = setup_model(model_path, use_peft=use_peft)
    # run inference
    results, metrics = run_inference_loop(
        model,
        processor,
        items,
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
        "metrics":metrics,
        "results": results
        }
    with outpath.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)
    
    LOG.info("Saved results to %s", outpath)
    # LOG.info("metrics: %.4f", metrics)


if __name__ == "__main__":
    main()