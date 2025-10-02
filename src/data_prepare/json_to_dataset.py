import os
import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from PIL import Image
import sys
sys.path.insert(0, 'src/')
from defined import yollava_reverse_category_dict, myvlm_reverse_category_dict
# fixed base save directory per your request
BASE_SAVE_DIR = "/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_data"


def _resolve_path(root: str, p: str) -> str:
    """If p is absolute, return it; otherwise join with root."""
    if os.path.isabs(p):
        return p
    if root is None:
        return p
    return os.path.join(root, p)


def json_to_dataset_dict(ret_json, root, category, seed, dataset):
    retrieval_json_path = os.path.join(root, category, f'seed_{seed}', ret_json)
    with open(retrieval_json_path, "r") as f:
        data = json.load(f)
    rows = []
    for item in tqdm(data, desc="Processing retrieval entries"):
        # accept either field name
        ret_list = item.get("retrieved_paths") or item.get("ret_paths") or item.get("ret_paths", [])
        query_path_raw = item.get("query_path") or item.get("query")
        if query_path_raw is None:
            # skip malformed entry
            continue
        query_abs = _resolve_path('/gpfs/projects/ehpc171/ddas/projects/Lewis_Game', query_path_raw)
        # make retrieved absolute where possible
        retrieved_abs = [_resolve_path('/gpfs/projects/ehpc171/ddas/projects/Lewis_Game', rp) for rp in ret_list]
        names = [Path(p).parent.name for p in retrieved_abs]
        # simple validation: warn if query path missing
        if not os.path.exists(query_abs):
            # keep entry, but warn
            print(f"Warning: query image not found: {query_abs}")

        # speaker problem prompt (kept similar to your original)
        concept_name = query_path_raw.split('/')[-2]
        if dataset == 'YoLLaVA':
            category = yollava_reverse_category_dict[concept_name]
        elif dataset == "MyVLM":
            category = myvlm_reverse_category_dict[concept_name]
        else:
            category = item.get("category", "object")
        speaker_problem = (
            f'Describe the {category} in the image so that it can be distinguished from other {category} objects. '
            "Do NOT mention background, location or state of the object. "
            "If the image contains a person, avoid mentioning the clothing or accesories."
            f'Write exactly one fluent sentence that begins with "The {category}" and highlights 3–4 visible distinguishing attributes. '
            "Keep the description concise and natural, without using lists or brackets. "
            "Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags."
        )

        uid = Path(query_abs).stem

        rows.append({
            "image": Image.open(query_abs).convert('RGB'),
            "ret_paths": retrieved_abs,
            "names":names,
            "speaker_problem": speaker_problem,
            "solution": item.get("label"),
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
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--dataset", default="YoLLaVA")
    ap.add_argument("--category", default="all")
    ap.add_argument("--ret_json", default="retrieval_top5.json")
    ap.add_argument("--seed", type=int, default=23, help="Random seed for reproducibility.")
    
    return ap.parse_args()


def main():
    args = parse_args()
    root = os.path.join(args.root, args.dataset)
    ds = json_to_dataset_dict(args.ret_json, root, args.category, args.seed, args.dataset)
    # optionally you could include args.dataset into the save_dirname if that helps naming consistency
    print_dataset_statistics(ds)
    save_dirname = f"{args.dataset}_{args.category}_test_subset_seed_{args.seed}"
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
