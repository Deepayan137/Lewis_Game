import os
import random
import string
import json
import argparse
import re
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from PIL import Image
import sys
sys.path.insert(0, 'src/')
from defined import yollava_reverse_category_dict, myvlm_reverse_category_dict
# fixed base save directory per your request
BASE_SAVE_DIR = "share_data"


def _resolve_path(root: str, p: str) -> str:
    """If p is absolute, return it; otherwise join with root."""
    if os.path.isabs(p):
        return p
    if root is None:
        return p
    return os.path.join(root, p)


def json_to_dataset_dict(input_filename, with_negative):
    retrieval_json_path = input_filename 
    with open(retrieval_json_path, "r") as f:
        data = json.load(f)
    rows = []
    
    for idx, item in enumerate(tqdm(data, desc="Processing retrieval entries")):
        # accept either field name
        ret_list = item.get("retrieved_paths") or item.get("ret_paths") or item.get("ret_paths", [])
        query_path_raw = item.get("query_path") or item.get("query")
        if query_path_raw is None:
            # skip malformed entry
            continue
        query_abs = _resolve_path('./', query_path_raw)
        # make retrieved absolute where possible
        retrieved_abs = [_resolve_path('./', rp) for rp in ret_list]
        # if with_negative:
        # names = [''.join(random.choices(string.ascii_uppercase, k=3)) for _ in range(len(retrieved_abs))]
        names = [string.ascii_uppercase[i] for i in range(len(retrieved_abs))]
        # else:
        #     names = [Path(p).parent.name for p in retrieved_abs]
        # simple validation: warn if query path missing
        if not os.path.exists(query_abs):
            # keep entry, but warn
            print(f"Warning: query image not found: {query_abs}")

        # speaker problem prompt (kept similar to your original)
        # if dataset == 'PerVA':
        concept_name = query_path_raw.split('/')[-3]
        # else:
        #     concept_name = query_path_raw.split('/')[-2]
        
        # if dataset == 'YoLLaVA':
        #     category = yollava_reverse_category_dict[concept_name]
        # elif dataset == "MyVLM":
        #     category = myvlm_reverse_category_dict[concept_name]
        # elif dataset == 'PerVA':
        perva_category_map = {
            'veg': 'vegetable',
            'decoration': 'decoration object',
            'retail': 'retail object',
            'tro_bag': 'trolley bag',
        }
        category = perva_category_map.get(concept_name, concept_name)
        # else:
        #     category = item.get("category", "object")
        speaker_problem = (
            f'Provide two descriptions of the {category} in the image:\n'
            f'1. A coarse 5-6 word description starting with "A photo of a "\n'
            f'2. A detailed description: Describe the {category} so it can be distinguished from other {category}s. '
            "Do NOT mention background, location or state. "
            "If the image contains a person, avoid mentioning clothing or accessories. "
            f'Write exactly one fluent sentence beginning with "The " and highlighting 3-4 visible distinguishing attributes. '
            "Keep it concise and natural, without lists or brackets.\n\n"
            "Output format:\n"
            "<thinking>Your reasoning</thinking>\n"
            f"<coarse>A photo of a ...</coarse>\n"
            f"<detailed>The ...</detailed>"
        )
        uid = idx
        rows.append({
            "image": Image.open(query_abs).convert('RGB'),
            "ret_paths": retrieved_abs,
            "names":names,
            "speaker_problem": speaker_problem,
            "solution": names[item.get("label")],
            "category": category,
            "example_idx": uid,
        })

    # create HF Dataset and DatasetDict
    dataset = Dataset.from_list(rows)
    ds_dict = DatasetDict({"train": dataset})
    return ds_dict

def print_dataset_statistics(ds_dict: DatasetDict):
    """
    Print statistics for each split in the DatasetDict.
    Shows number of examples, and unique values for key fields.
    """
    for split, ds in ds_dict.items():
        print(f"=== Split: {split} ===")
        print(f"Number of examples: {len(ds)}")
        if len(ds) == 0:
            continue
        # Print unique categories
        if "category" in ds.column_names:
            categories = set(ds["category"])
            print(f"Unique categories: {categories}")
        # Print number of unique solutions (labels)
        if "solution" in ds.column_names:
            unique_solutions = set(ds["solution"])
            print(f"Number of unique solutions: {len(unique_solutions)}")
        # Print example of a row
        print("Example row:")
        print(ds[0])
        print("-" * 40)


def save_dataset_dict(ds_dict: DatasetDict, save_dirname: str):
    """Save dataset dict to BASE_SAVE_DIR / save_dirname"""
    out_dir = os.path.join(BASE_SAVE_DIR, save_dirname)
    os.makedirs(out_dir, exist_ok=True)
    ds_dict.save_to_disk(out_dir)
    print(f"Saved dataset to {out_dir}")
    return out_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Convert retrieval JSON to HF Dataset and save to disk.")
    ap.add_argument("--input_filename", default="outputs/PerVA/all/seed_23/retrieval_top3_subset_30.json")
    ap.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility.")
    ap.add_argument("--K", type=int, default=3, help="distracors + 1")
    return ap.parse_args()


def main():
    args = parse_args()
    ds = json_to_dataset_dict(args.input_filename, with_negative=False)
    print_dataset_statistics(ds)
    subset_match = re.search(r'subset_(\d+)', args.input_filename)
    subset = f"subset_{subset_match.group(1)}" if subset_match else None
    sampled_match = re.search(r'sampled_(\d+)', args.input_filename)
    sampled = f"sampled_{sampled_match.group(1)}" if sampled_match else None
    save_dirname = f"PerVA_seed_{args.seed}_K_{args.K}"
    if subset:
        save_dirname += f"_{subset}"
    if sampled:
        save_dirname += f"_{sampled}"
    out = save_dataset_dict(ds, save_dirname)
    print(f"Dataset saved in: {save_dirname}")
    # quick sanity check load
    loaded = DatasetDict.load_from_disk(out)
    print("Loaded dataset summary:")
    for k, v in loaded.items():
        print(f" - split: {k}, n = {len(v)}")
    # print column names
    print("Columns:", loaded["train"].column_names)


if __name__ == "__main__":
    main()
