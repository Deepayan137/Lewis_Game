import os
import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm


# fixed base save directory per your request
BASE_SAVE_DIR = "/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_data"


def _resolve_path(root: str, p: str) -> str:
    """If p is absolute, return it; otherwise join with root."""
    if os.path.isabs(p):
        return p
    if root is None:
        return p
    return os.path.join(root, p)


def json_to_dataset_dict(retrieval_json_path: str, root: str = None):
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

        query_abs = _resolve_path(root, query_path_raw)
        # make retrieved absolute where possible
        retrieved_abs = [_resolve_path(root, rp) for rp in ret_list]

        # simple validation: warn if query path missing
        if not os.path.exists(query_abs):
            # keep entry, but warn
            print(f"Warning: query image not found: {query_abs}")

        # speaker problem prompt (kept similar to your original)
        category = item.get("category", "object")
        speaker_problem = (
            f'Describe the {category} in the image so that it can be distinguished from other {category} objects. '
            "Do NOT mention background, location or state of the object. "
            f'Write exactly one fluent sentence that begins with "The {category}" and highlights 3–4 visible distinguishing attributes. '
            "Keep the description concise and natural, without using lists or brackets. "
            "Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags."
        )

        uid = Path(query_abs).stem

        rows.append({
            "image_path": query_abs,
            "retrieved_paths": retrieved_abs,
            "speaker_problem": speaker_problem,
            "solution": item.get("label"),
            "category": category,
            "example_idx": uid,
        })

    # create HF Dataset and DatasetDict
    dataset = Dataset.from_list(rows)
    ds_dict = DatasetDict({"train": dataset})
    return ds_dict


def save_dataset_dict(ds_dict: DatasetDict, save_dirname: str):
    """Save dataset dict to BASE_SAVE_DIR / save_dirname"""
    out_dir = os.path.join(BASE_SAVE_DIR, save_dirname)
    os.makedirs(out_dir, exist_ok=True)
    ds_dict.save_to_disk(out_dir)
    print(f"Saved dataset to {out_dir}")
    return out_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Convert retrieval JSON to HF Dataset and save to disk.")
    ap.add_argument("--ret_json", default="clothe_retrieval_top5_subset.json")
    ap.add_argument("--root", default="/gpfs/projects/ehpc171/ddas/projects/YoLLaVA/yollava-data/train_/",
                    help="Optional root to prefix relative image paths found in the retrieval JSON.")
    ap.add_argument("--dataset", default="PerVA",
                    help="Short name for dataset (used as part of folder naming if you like).")
    ap.add_argument("--save_dirname", default="PerVA_clothe_test_subset",
                    help="Final folder name (under the fixed share_data base) where dataset will be saved.")
    return ap.parse_args()


def main():
    args = parse_args()
    ds = json_to_dataset_dict(args.ret_json, root=args.root)
    # optionally you could include args.dataset into the save_dirname if that helps naming consistency
    out = save_dataset_dict(ds, args.save_dirname)
    # quick sanity check load
    loaded = DatasetDict.load_from_disk(out)
    print("Loaded dataset summary:")
    for k, v in loaded.items():
        print(f" - split: {k}, n = {len(v)}")
    # print column names
    print("Columns:", loaded["train"].column_names)


if __name__ == "__main__":
    main()
