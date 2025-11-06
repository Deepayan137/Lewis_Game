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

def format_extra_info(extra_info, names) -> str:
    """Format name→description pairs as lettered options (A., B., ...)."""
    letters = list(string.ascii_uppercase)
    lines = []
    for i, (info) in enumerate(extra_info):
        label = letters[i] if i < len(letters) else f"Option {i+1}"
        info = info.split(":")[1]
        lines.append(f"{label}. {info.strip()}")
    return "\n".join(lines)

def get_prompt(descriptions: Sequence[str], category: str,  names, query_desc="",) -> str:
    """
    Build the prompt used to ask the model to reason and return a JSON
    with keys "Reasoning" and "Answer".
    `descriptions` is typically a list of strings (distinguishing features).
    """
    n = len(names)
    letters = [string.ascii_uppercase[i] for i in range(n)]
    test_question = (
        f"Which description matches the {category} in the image? "
        f"Answer in {', '.join([chr(65 + i) for i in range(n)])}."
    )
    # answer_format = {chr(65 + i): f"[Matching attributes for option {chr(65 + i)}]" for i in range(len(names))}
    answer_format = {}
    answer_format.update({
        "Reasoning": "<Brief justification>",
        # "Answer": f"<one of {names}>",
        "Answer": f"one of {letters}"
    })

    descriptions_block = json.dumps(descriptions, indent=2, ensure_ascii=False)
    prompt = (
        f"You are provided with a query image containing a {category} "
        f"along with the name and detailed distinguishing features of several other {category}s.\n\n"
        "Below are the name and their descriptions:\n"
        f"{descriptions_block}\n\n"
        "Your Task:\n"
        f"1. Generate an attribute-focused description of the {category} in the query image. "
        "Focus on its distinguishing features rather than superficial details such as background, pose, lighting, clothes or accessories.\n"
        f"2. Compare your generated description of the query image with the provided descriptions of the other {category}s.\n"
        f"3. Identify the name of the {category} in the query image from the best match.\n\n"
        "Output Requirements:\n"
        f"- Your response MUST be a valid JSON exactly matching the format:\n{json.dumps(answer_format)}\n"
        "- Do not include any extra text, explanations, or formatting outside of the JSON.\n"
    )
    
    return prompt

# def generate_query_description(args, data_loader):
#     if args.db_type == 'original':
#         model_path = "Qwen/Qwen2-VL-2B-Instruct"
#     else:
#         model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2-VL-2B-Instruct_GRPO_lewis_{args.category}_test_subset"
#     speaker_model, processor = setup_model(model_path)
#     raw_results = run_description_generation(
#         speaker_model,
#         processor,
#         data_loader)
#     query_descs = []
#     for name, desc in raw_results:
#         desc_clean = extract_speaker_answer_term(desc)
#         query_descs.append(desc_clean)
#     return query_descs

def prepare_test_retrieval_items(
    args,
    description_json: Path,
    catalog_json: Path,
    category: str,
    retriever: SimpleClipRetriever,
    k: int = 5,
    use_query_desc=False
) -> List[Dict[str, Any]]:
    """
    Build a list of items (dicts) for evaluation. Each item contains:
      - 'path' : path to query image
      - 'problem' : prompt (string) constructed from retrieved descriptions
      - 'solution' : ground-truth object name
      - 'solution_desc' : the description string for the true object (if available)
      - 'ret_path' : list of retrieved image paths (may be empty)
    """
    savedir = Path('outputs') / args.data_name / args.category / f'seed_{args.seed}'
    savedir.mkdir(parents=True, exist_ok=True)
    test_ret_path = savedir / f'{args.db_type}_test_set.json'
    # INSERT_YOUR_CODE
    if test_ret_path.exists() and use_query_desc:
        LOG.info(f"{test_ret_path} already exists. Loading and returning existing data.")
        with test_ret_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        LOG.info("Loading dataset from: %s", catalog_json)
        dataset = SimpleImageDataset(
            json_path=catalog_json,
            category=category,
            split="test",
            seed=retriever.seed if hasattr(retriever, "seed") else 0,
            data_name=args.data_name
        )
        data_loader = create_data_loader(dataset, batch_size=args.batch_size)
        query_descs = []
        # generate query descriptions
        # if use_query_desc:
        #     query_descs = generate_query_description(args, data_loader)
        # load description dictionary produced earlier by description generation step
        with description_json.open("r", encoding="utf-8") as fh:
            desc_lookup: Dict[str, str] = json.load(fh)
        concept_dict_format = False
        if 'concept_dict' in desc_lookup:
            concept_dict_format = True
            desc_lookup = desc_lookup['concept_dict']
        items: List[Dict[str, Any]] = []
        for idx, item in enumerate(tqdm(dataset, desc="Preparing retrieval items")):
            query_path = item["path"]
            query_name = item['name']
            query_desc = query_descs[idx] if len(query_descs) > 0 else ""
            # retrieve K nearest neighbors (clip search)
            results = retriever.hybrid_search(query_path,
                     k=k, 
                    alpha=0.5,
                    normalization='minmax',
                    aggregation='weighted_sum',
                    image_k=len(desc_lookup),
                    text_k=len(desc_lookup))
            descriptions: List[str] = []
            ret_paths: List[str] = []
            names = []
            import string
            letters = list(string.ascii_uppercase)[:len(results)]
            letter2name = {}
            for i, r in enumerate(results):
                name = r.get("name")
                letter = letters[i]
                letter2name[letter] = name
                names.append(name)
                if concept_dict_format: name = f'<{name}>'
                if name and name in desc_lookup:
                    if concept_dict_format:
                        descriptions.append(f"Name: {letter}, Info: {desc_lookup[name]['info']['general']}")
                    else:
                        descriptions.append(f"Name: {letter}, Info: {desc_lookup[name]}")
                # keep the returned path for diagnostics
                if "path" in r:
                    ret_paths.append(r["path"])

            if not descriptions:
                descriptions = ["No description available."]
            if args.data_name == "YoLLaVA":
                category = yollava_reverse_category_dict[query_name]
            elif args.data_name == "MyVLM":
                category = myvlm_reverse_category_dict[query_name]
            elif args.data_name == 'PerVA':
                category_dict = {
                    'veg': 'vegetable',
                    'decoration': 'decoration object',
                    'retail': 'retail object',
                    'tro_bag': 'trolley bag',
                }
                concept_name = query_path.split('/')[-2]
                category = category_dict.get(concept_name, concept_name)
            prompt = get_prompt(descriptions, category, names, query_desc)
            items.append(
                {
                    "path": query_path,
                    "problem": prompt,
                    "solution": item["name"],
                    "solution_desc": desc_lookup.get(item["name"], ""),
                    "ret_path": ret_paths,
                    "query_desc": query_desc,
                    "letter2name": letter2name
                }
            )
        if use_query_desc:
            with test_ret_path.open("w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, ensure_ascii=False)

        return items

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
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dict_collate_fn, num_workers=4
    )

    results: List[Dict[str, Any]] = []
    correct_count = 0

    # device inference guidance (model may already be on device)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for batch in tqdm(loader, desc="Generating model responses"):
        # load images lazily if not provided
        images = []
        for it in batch:
            if "image" in it and it["image"] is not None:
                images.append(it["image"])
            else:
                # load and convert to RGB
                images.append(Image.open(it["path"]).convert("RGB"))

        problems = [it["problem"] for it in batch]
        gt_names = [it["solution"] for it in batch]
        sol_descs = [it.get("solution_desc", "") for it in batch]
        paths = [it.get("path") for it in batch]
        ret_paths = [it.get("ret_path", []) for it in batch]
        letter2names = [it.get("letter2name", []) for it in batch]
        try:
            responses = speaker_describes_batch(model, processor, images, problems, temperature=temperature, max_new_tokens=max_new_tokens)
        except Exception:
            LOG.exception("Failed generating model responses for current batch; skipping.")
            # append placeholders for each item in batch and continue
            for path, gt, sol_desc, rp in zip(paths, gt_names, sol_descs, ret_paths):
                results.append(
                    {
                        "image_path": path,
                        "problem": None,
                        "solution": gt,
                        "solution_desc": sol_desc,
                        "ret_paths": rp,
                        "response": "",
                        "pred_name": "",
                    }
                )
            continue

        # speaker_describes_batch may return a single string when batch_size==1
        if isinstance(responses, str):
            responses = [responses]
        # clean predicted "Answer" using provided utility
        pred_names = [] 
        for resp in responses:
            try:
                if isinstance(resp, list):
                    resp = resp[0]
                term = extract_reasoning_answer_term(resp, "Answer")
                pred_names.append(term.strip())
            except Exception:
                LOG.exception("Failed to extract answer term from model response; using empty string")
                pred_names.append("")

        for path, gt, sol_desc, rp, resp, pred, lt2nm in zip(paths, gt_names, sol_descs, ret_paths, responses, pred_names, letter2names):
            pred_name = lt2nm[pred] if pred in lt2nm else pred
            is_correct = int(pred_name.lower() == gt.lower())
            # is_correct = gt.lower() in pred.lower()
            correct_count += is_correct
            results.append(
                {
                    "image_path": path,
                    "problem": resp if resp else None,
                    "solution": gt,
                    "solution_desc": sol_desc,
                    "ret_paths": rp,
                    "response": resp,
                    "pred_name": pred_name,
                    "correct": bool(is_correct),
                }
            )

        # free cuda mem (harmless if CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    accuracy = correct_count / len(results) if results else 0.0
    return results, accuracy, correct_count, len(results)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--temperature", type=float, default=1e-6,
                       help='generation temperature')
    parser.add_argument("--k_retrieval", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    set_seed(args.seed)
    LOG.info("Args: %s", args)

    manifests_dir = Path("manifests") / args.data_name
    description_json = Path("outputs") / args.data_name / args.category / f"seed_{args.seed}" / f"descriptions_{args.db_type}.json"
    catalog_json = str(manifests_dir / args.catalog_file)
    if not description_json.exists():
        LOG.error("Descriptions file not found at %s", description_json)
        raise FileNotFoundError(f"Descriptions file not found: {description_json}")

    # initialize retriever
    retriever = SimpleClipRetriever(
        dataset=args.data_name,
        category=args.category,
        json_path=catalog_json,
        create_index=True,
        seed=args.seed,
        db_type=args.db_type
    )
    LOG.info("Retriever created")

    # prepare items to run inference on
    items = prepare_test_retrieval_items(
        args,
        description_json, 
        catalog_json, 
        args.category, 
        retriever, 
        k=args.k_retrieval)
    LOG.info("Prepared %d test items", len(items))
    if args.concept_name != '':
        items = [item for item in items if item.get("solution") == args.concept_name]
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
    results, accuracy, correct_count, total_samples = run_inference_loop(
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
    outpath = outdir / f"results_model_{args.model_type}_db_{args.db_type}_k_{args.k_retrieval}.json"
    output = {
        "metrics":{
            "accuracy":accuracy,
            "correct count": correct_count,
            "total samples": total_samples 
            }, 
        "results": results}
    with outpath.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    LOG.info("Saved results to %s", outpath)
    LOG.info("Accuracy: %.4f", accuracy)


if __name__ == "__main__":
    main()
