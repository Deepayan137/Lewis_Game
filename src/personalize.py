from __future__ import annotations

import json
import logging
import os
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

LOG = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_prompt(descriptions: Sequence[str], category: str,  query_desc="") -> str:
    """
    Build the prompt used to ask the model to reason and return a JSON
    with keys "Reasoning" and "Answer".
    `descriptions` is typically a list of strings (distinguishing features).
    """
    answer_format = {
        "Reasoning": "<Your reasoning in 1-2 sentences.>",
        "Answer": "<name of the object>",
    }

    descriptions_block = json.dumps(descriptions, indent=2, ensure_ascii=False)
    if query_desc != "":
        prompt = (
            f"You are provided with a query image containing a {category} object, "
            f"the detailed distinguishing features (description) of this query object, "
            f"and the name and descriptions of several other {category} objects.\n\n"
            "Query object description:\n"
            f"{query_desc}\n\n"
            "Other objects (name and description):\n"
            f"{descriptions_block}\n\n"
            "Your Task:\n"
            "- Carefully compare the query object (image + its provided description) "
            "with the provided descriptions of the other objects.\n"
            "- Identify the name of the object in the query image.\n"
            "- Your response MUST be a JSON exactly matching the format:\n"
            f"{json.dumps(answer_format)}\n"
            "- Output only the JSON response, with no extra text. "
            "Ignore superficial details such as background, pose, or lighting.\n"
            )
    else:
        prompt = (
            f"You are provided with a query image containing a {category} object, "
            f"as well as the names and distinguishing descriptions of several other {category} objects.\n\n"
            "Other objects:\n"
            f"{descriptions_block}\n\n"
            "Your Task:\n"
            "1. Generate an attribute-focused description of the query image object. "
            "Focus on its distinguishing features rather than superficial details such as background, pose, or lighting.\n"
            "2. Compare your generated description of the query image with the provided descriptions of the other objects.\n"
            "3. Identify which object from the provided list matches the query image.\n\n"
            "Output Requirements:\n"
            f"- Your response MUST be a valid JSON exactly matching the format:\n{json.dumps(answer_format)}\n"
            "- Do not include any extra text, explanations, or formatting outside of the JSON.\n"
        )

    return prompt

def generate_query_description(args, data_loader):
    if args.model_type == 'original':
        model_path = "Qwen/Qwen2-VL-2B-Instruct"
    else:
        model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2.5-VL-2B-Instruct_GRPO_lewis_{args.category}_test_subset"
    speaker_model, processor = setup_model(model_path)
    raw_results = run_description_generation(
        speaker_model,
        processor,
        data_loader)
    query_descs = []
    for name, desc in raw_results:
        desc_clean = extract_speaker_answer_term(desc)
        query_descs.append(desc_clean)
    return query_descs

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
    test_ret_path = savedir / f'{args.model_type}_test_set.json'
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
        )
        data_loader = create_data_loader(dataset, batch_size=args.batch_size)
        query_descs = []
        # generate query descriptions
        if use_query_desc:
            query_descs = generate_query_description(args, data_loader)
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
            query_desc = query_descs[idx] if len(query_descs) > 0 else ""
            # retrieve K nearest neighbors (clip search)
            results = retriever.image_search(query_path, k=k)
            descriptions: List[str] = []
            ret_paths: List[str] = []
            for r in results:
                name = r.get("name")
                if concept_dict_format: name = f'<{name}>'
                if name and name in desc_lookup:
                    if concept_dict_format:
                        descriptions.append(f"{name}: {desc_lookup[name]['info']['general']}")
                    else:
                        descriptions.append(f"{name}: {desc_lookup[name]}")
                # keep the returned path for diagnostics
                if "path" in r:
                    ret_paths.append(r["path"])

            if not descriptions:
                descriptions = ["No description available."]
            prompt = get_prompt(descriptions, category, query_desc)
            items.append(
                {
                    "path": query_path,
                    "problem": prompt,
                    "solution": item["name"],
                    "solution_desc": desc_lookup.get(item["name"], ""),
                    "ret_path": ret_paths,
                    "query_desc": query_desc
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
        try:
            responses = speaker_describes_batch(model, processor, images, problems, max_new_tokens=max_new_tokens)
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
                term = extract_reasoning_answer_term(resp, "Answer")
                
                pred_names.append(term.lower().strip())
            except Exception:
                LOG.exception("Failed to extract answer term from model response; using empty string")
                pred_names.append("")

        for path, gt, sol_desc, rp, resp, pred in zip(paths, gt_names, sol_descs, ret_paths, responses, pred_names):
            is_correct = int(pred == gt)
            correct_count += is_correct
            results.append(
                {
                    "image_path": path,
                    "problem": resp if resp else None,
                    "solution": gt,
                    "solution_desc": sol_desc,
                    "ret_paths": rp,
                    "response": resp,
                    "pred_name": pred,
                    "correct": bool(is_correct),
                }
            )

        # free cuda mem (harmless if CPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accuracy = correct_count / len(results) if results else 0.0
    return results, accuracy


def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description="Reasoning-based personalization evaluation")
    parser.add_argument("--data_name", type=str, default="PerVA")
    parser.add_argument("--catalog_file", type=str, default="main_catalog_seed_23.json")
    parser.add_argument("--category", type=str, default="clothe")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--model_type", type=str, default="original", choices=["original", "finetuned"])
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--k_retrieval", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    set_seed(args.seed)
    LOG.info("Args: %s", args)

    manifests_dir = Path("manifests") / args.data_name
    description_json = Path("outputs") / args.data_name / args.category / f"seed_{args.seed}" / f"descriptions_{args.model_type}.json"
    catalog_json = str(manifests_dir / args.catalog_file)
    if not description_json.exists():
        LOG.error("Descriptions file not found at %s", description_json)
        raise FileNotFoundError(f"Descriptions file not found: {description_json}")

    # initialize retriever
    retriever = SimpleClipRetriever(
        category=args.category,
        json_path=catalog_json,
        create_index=True,
        seed=args.seed,
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

    # load model
    LOG.info("Loading model from %s", args.model_path)
    model, processor = setup_model(args.model_path)

    # run inference
    results, accuracy = run_inference_loop(
        model,
        processor,
        items,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    # save results & stats
    outdir = Path(args.output_dir) / args.data_name / args.category
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"results_{args.model_type}_seed_{args.seed}.json"

    output = {"accuracy": accuracy, "results": results}
    with outpath.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    LOG.info("Saved results to %s", outpath)
    LOG.info("Accuracy: %.4f", accuracy)


if __name__ == "__main__":
    main()
