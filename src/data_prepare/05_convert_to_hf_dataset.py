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
from recognition import add_marker_to_image
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
        retrieved_descriptions = item.get("ret_descs", [])
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
        concept_name = query_path_raw.split('/')[-2]
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
            f'Provide descriptions of the {category} in the image in four parts:\n'
            f'1. Coarse: A 5-6 word description starting with "A photo of a {category}"\n'
            f'2. Detailed: Describe ONLY permanent identity features (color, patterns, markings, shape, facial features, eye color, build, etc.). '
            f'Write one sentence with "The {category}" highlighting 3-4 permanent attributes.\n'
            f'3. State: Describe pose and position of the {category} in the image (eg. lying, open, closed, sitting, standing, running, hanging, folded, etc.).\n'
            f'4. Location: Describe positioning and background (outside, inside, on the floor, on the shelf/table, near objects, background elements etc.).\n\n'
            'Examples:\n\n'
            'Example 1 (cat):\n'
            '<thinking>I need to separate permanent features from temporary state. The cat has white and brown fur with green eyes - these are identity features. It is sitting with paws tucked - this is state. It is on a wooden floor - this is location.</thinking>\n'
            '<coarse>A photo of a cat</coarse>\n'
            '<detailed>The cat has a white chest and face with brown fur on its back and ears, bright green eyes, and a distinctive pink nose.</detailed>\n'
            '<state>Sitting upright with front paws tucked under its body</state>\n'
            '<location>On a wooden floor near a window</location>\n\n'
            'Example 2 figurine:\n'
            '<thinking>Permanent features include shape, color and pattern on the figurine. Where and how it is placed are not permanent, so they go in location and state respectively.</thinking>\n'
            '<coarse>A photo of a deer shaped ceramic figurine</coarse>\n'
            '<detailed>The figurine is shaped like a deer with four legs and two antlers, featuring brown coloring on its upper body with yellow specks and white on its underside.</detailed>\n'
            '<state>lying hrizontally</state>\n'
            '<location>on the shelf</location>\n\n'
            f'Now describe the {category} in the image following this format:\n'
            "<thinking>Your reasoning</thinking>\n"
            f"<coarse>A photo of a {category}</coarse>\n"
            f"<detailed>The {category} ...</detailed>\n"
            "<state>...</state>\n"
            "<location>...</location>"
        )
        uid = idx
        rand_idx = random.randint(0, len(ret_list))
        ref_path = retrieved_abs[rand_idx]
        ref_concept_name = ref_path.split('/')[-2]
        ref_concept_description = retrieved_descriptions[rand_idx] if rand_idx < len(retrieved_descriptions) else "No description available."
        answer_format = {
            "Reasoning": "<Brief comparison based on key attributes>",
            "Answer": "<yes or no>"
        }
        test_questions_examples =[f'Is <{ref_concept_name}> in Image 1? Answer yes or no.', 
            f'Is <{ref_concept_name}> in the first image? Answer yes or no.',
            f'Does Image 1 contain <{ref_concept_name}>? Answer yes or no.',
            f'Does the first image contain <{ref_concept_name}>? Answer yes or no.',
            f"Can you find <{ref_concept_name}> in Image 1? Answer yes or no.",
            f"Can you see <{ref_concept_name}> in the first image? Answer yes or no."]
        
        test_question = random.choice(test_questions_examples)
        listener_problem = (
            f"You are a helpful AI agent specializing in image analysis and object recognition\n\n"
            f"You are given two images (marked with numbers 1 and 2 in the top-right corner). "
            f"Additionally, the name and a textual description of the subject "
            f"in Image 2 is also provided below:\n\n"
            f"{json.dumps(ref_concept_description, indent=2)}\n"
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
        listener_solution = 'yes' if concept_name == ref_concept_name else 'no'
        rows.append({
            "image": Image.open(query_abs).convert('RGB'),
            "ret_paths": retrieved_abs,
            "names":names,
            "speaker_problem": speaker_problem,
            "solution": names[item.get("label")],
            "category": category,
            "example_idx": uid,
            "query_image": add_marker_to_image(Image.open(query_abs).convert('RGB'), marker_text="1"),
            "reference_image": add_marker_to_image(Image.open(ref_path).convert('RGB'), marker_text="2"),
            "listener_problem": listener_problem,
            "listener_solution": listener_solution,
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
