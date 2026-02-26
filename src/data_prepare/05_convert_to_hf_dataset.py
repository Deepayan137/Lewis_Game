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

from inference_utils.prompts import get_description_prompt
from inference_utils.common  import PERVA_CATEGORY_MAP
# from inference_utils.model import setup_model, speaker_describes_batch
# from inference_utils.cleanup import parse_descriptions
# from generate_descriptions import process_batch_efficiently
# fixed base save directory per your request
BASE_SAVE_DIR = "share_data"

class PromptTemplates:
    """Centralized prompt templates for speaker and listener tasks."""
    
    @staticmethod
    def get_detailed_speaker_prompt(category: str) -> str:
        """Generate speaker task prompt for describing an object."""
        return (
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
            '<state>lying horizontally</state>\n'
            '<location>on the shelf</location>\n\n'
            f'Now describe the {category} in the image following this format:\n'
            "<thinking>Your reasoning</thinking>\n"
            f"<coarse>A photo of a {category}</coarse>\n"
            f"<detailed>The {category} ...</detailed>\n"
            "<state>...</state>\n"
            "<location>...</location>"
        )
    @staticmethod
    def get_speaker_prompt(category: str) -> str:
        return (
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

    @staticmethod
    def get_test_question(concept_name: str) -> str:
        """Generate a random test question for the listener task."""
        questions = [
            f'Is <{concept_name}> in Image 1? Answer yes or no.',
            f'Is <{concept_name}> in the first image? Answer yes or no.',
            f'Does Image 1 contain <{concept_name}>? Answer yes or no.',
            f'Does the first image contain <{concept_name}>? Answer yes or no.',
            f"Can you find <{concept_name}> in Image 1? Answer yes or no.",
            f"Can you see <{concept_name}> in the first image? Answer yes or no."
        ]
        return random.choice(questions)
    
    @staticmethod
    def get_listener_prompt_no_description(ref_concept_name: str, ref_category:str, test_question: str) -> str:
        """Generate listener task prompt for comparing images."""
        answer_format = {
            "Reasoning": "<Brief comparison based on key attributes>",
            "Answer": "<yes or no>"
        }
        
        return (
            f"You are a helpful AI agent specializing in image analysis and object recognition\n\n"
            f"You are given two images (marked with numbers 1 and 2 in the top-right corner). "
            f"Additionally, the name and category information of the subject "
            f"in Image 2 is also provided below:\n\n"
            f"Name: {ref_concept_name}, Category: {ref_category}\n\n"
            f"Your Task:\n"
            f"- Compare Image 1 with Image 2 and answer the following question: "
            f"{test_question}\n"
            "Think step by step and provide your reasoning before giving the final answer.\n\n"
            f"**Output (JSON only):**\n{json.dumps(answer_format, indent=2)}"
        )
    @staticmethod
    def get_listener_prompt_with_description(ref_description, test_question: str) -> str:
        """Generate listener task prompt for comparing images."""
        answer_format = {
            "Reasoning": "<Brief comparison based on key attributes>",
            "Answer": "<yes or no>"
        }
        
        return (
            f"You are a helpful AI agent specializing in image analysis and object recognition\n\n"
            f"You are given two images (marked with numbers 1 and 2 in the top-right corner). "
            f"Additionally, the name and a textual description of the subject in Image 2 is provided below:\n\n"
            f"\"{ref_description}\"\n\n"
            f"Your Task:\n"
            f"- Compare Image 1 with Image 2 and answer the following question: {test_question}\n"
            f"- **Ignore superficial details** such as clothing, accessories, pose variations, "
            f"or surrounding elements (e.g., people in the background).\n"
            f"- **Focus only on non-variant/permanent features** such as color, shape, pattern, "
            f"distinctive markings for objects/buildings and facial features for people\n"
            f"- Determine if the EXACT SAME subject appears in both images.\n"
            f"- If you are uncertain then you can refer the textual description of Image 2 "
            f"to make a more informed decision.\n\n"
            f"Think step by step and provide your reasoning before giving the final answer.\n\n"
            f"**Output (JSON only):**\n{json.dumps(answer_format, indent=2)}"
        )


def _resolve_path(root: str, p: str) -> str:
    """If p is absolute, return it; otherwise join with root."""
    if os.path.isabs(p):
        return p
    if root is None:
        return p
    return os.path.join(root, p)

def sample_random_image_from_different_category(ret_data, query_category):
    data_subset = []
    for item in ret_data:
        if item['category'] != query_category:
            data_subset.append(item)
    entry = random.choice(data_subset)
    reference_path = entry['ret_paths'][0]
    reference_description = entry['ret_descs'][0] if entry.get('ret_descs') else "No description available."
    # reference_description = entry['for_recognition']['descriptions']['coarse'][0] + '. ' + entry['for_recognition']['descriptions']['detailed'][0]
    return reference_path, reference_description

def json_to_dataset_dict(input_filename, with_negative, cross_category, use_description, task='speaker', multi_view=False, sp_detailed=False):
    """Build an HF DatasetDict from a combined retrieval JSON.

    Args:
        task: 'speaker' — emit only fields needed by train_speaker_dist.py
                          [image, ret_paths, names, speaker_problem, solution, category]
              'listener' — emit only fields needed by train_listener_dist.py
                          [query_image_path, reference_image_path,
                           query_image, reference_image,
                           listener_problem, listener_solution]
    """
    retrieval_json_path = input_filename
    with open(retrieval_json_path, "r") as f:
        data = json.load(f)
    rows = []

    perva_category_map = {
        'veg': 'vegetable',
        'decoration': 'decoration object',
        'retail': 'retail object',
        'tro_bag': 'trolley bag',
    }

    for idx, item in enumerate(tqdm(data, desc="Processing retrieval entries")):
        # accept either field name for backward compat
        ret_list = item.get("retrieved_paths") or item.get("ret_paths") or []
        query_path_raw = item.get("query_path") or item.get("query")
        if query_path_raw is None:
            continue

        view1_abs = _resolve_path('./', query_path_raw)
        query_concept_name = query_path_raw.split('/')[-2]
        category_name      = query_path_raw.split('/')[-3]
        query_category     = perva_category_map.get(category_name, category_name)

        retrieved_abs = [_resolve_path('./', rp) for rp in ret_list]
        names = [string.ascii_uppercase[i] for i in range(len(retrieved_abs))]

        if not os.path.exists(view1_abs):
            print(f"Warning: query image not found: {view1_abs}")

        # ── Speaker branch ──────────────────────────────────────────────────
        if task == 'speaker':
            view1_image    = Image.open(view1_abs).convert('RGB')
            speaker_problem = PromptTemplates.get_speaker_prompt(query_category)
            rows.append({
                "image":            view1_image,
                "ret_paths":        retrieved_abs,
                "names":            names,
                "speaker_problem":  speaker_problem,
                "solution":         names[item.get("label")],
                "category":         query_category,
            })
            continue  # skip all listener logic

        # ── Listener branch ─────────────────────────────────────────────────
        view1_image = Image.open(view1_abs).convert('RGB')

        # Target distribution: 50% positive, 50% negative
        is_positive = random.random() < 0.5
        ref_concept_description = "No description available."  # default

        if is_positive:
            positive_candidates = [p for p in retrieved_abs if p.split('/')[-2] == query_concept_name]
            candidates = positive_candidates if positive_candidates else retrieved_abs
            ref_path   = random.choice(candidates)
            ref_index  = retrieved_abs.index(ref_path)
            ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
            if use_description:
                ref_concept_description = (
                    item['ret_descs'][ref_index]
                    if ref_index < len(item.get('ret_descs', []))
                    else "No description available."
                )
        else:
            # cross-category negative (20% overall when cross_category=True)
            if cross_category and random.random() < 0.4:
                ref_path, ref_concept_description = sample_random_image_from_different_category(data, category_name)
                ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
            else:
                # within-category negative
                negative_candidates = [p for p in retrieved_abs if p.split('/')[-2] != query_concept_name]
                candidates = negative_candidates if negative_candidates else retrieved_abs
                ref_path   = random.choice(candidates)
                ref_index  = retrieved_abs.index(ref_path)
                ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
                if use_description:
                    ref_concept_description = (
                        item['ret_descs'][ref_index]
                        if ref_index < len(item.get('ret_descs', []))
                        else "No description available."
                    )

        reference_image  = Image.open(ref_path).convert('RGB')
        reference_image  = reference_image.resize(view1_image.size, Image.LANCZOS)
        ref_concept_name = ref_path.split('/')[-2]
        test_question    = PromptTemplates.get_test_question(ref_concept_name)

        if use_description:
            listener_problem = PromptTemplates.get_listener_prompt_with_description(ref_concept_description, test_question)
        else:
            listener_problem = PromptTemplates.get_listener_prompt_no_description(ref_concept_name, ref_category, test_question)

        listener_solution = 'yes' if query_concept_name == ref_concept_name else 'no'

        rows.append({
            "query_image_path":     query_path_raw,
            "reference_image_path": ref_path,
            "query_image":          add_marker_to_image(view1_image,    marker_text="1"),
            "reference_image":      add_marker_to_image(reference_image, marker_text="2"),
            "listener_problem":     listener_problem,
            "listener_solution":    listener_solution,
        })

    dataset = Dataset.from_list(rows)
    ds_dict  = DatasetDict({"train": dataset})
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
    ap.add_argument("--seed",  type=int, default=23, help="Random seed for reproducibility.")
    ap.add_argument("--K",     type=int, default=3,  help="Distractors + 1 (retrieval pool size).")
    ap.add_argument("--task",  choices=['speaker', 'listener'], default='speaker',
                    help="'speaker': emit [image, ret_paths, names, speaker_problem, solution, category]. "
                         "'listener': emit [query_image_path, reference_image_path, query_image, "
                         "reference_image, listener_problem, listener_solution].")
    ap.add_argument("--cross_category", action='store_true',
                    help="(Listener only) Allow cross-category negatives (20%% of training examples).")
    ap.add_argument("--use_description", action='store_true',
                    help="(Listener only) Use speaker descriptions in the listener prompt.")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    ds = json_to_dataset_dict(
        args.input_filename,
        with_negative=False,
        cross_category=args.cross_category,
        use_description=args.use_description,
        task=args.task,
    )
    print_dataset_statistics(ds)

    # ── Extract subset / sampled tags from filename ────────────────────────
    subset_match  = re.search(r'subset_(\d+)',  args.input_filename)
    sampled_match = re.search(r'sampled_(\d+)', args.input_filename)
    sub_tag     = f"sub{subset_match.group(1)}"  if subset_match  else None
    sampled_tag = f"n{sampled_match.group(1)}"   if sampled_match else None

    # ── Build canonical save dirname ───────────────────────────────────────
    if args.task == 'speaker':
        # PerVA_speaker_train_seed{seed}_K{K}[_sub{S}][_n{N}]
        save_dirname = f"PerVA_speaker_train_seed{args.seed}_K{args.K}"
        if sub_tag:
            save_dirname += f"_{sub_tag}"
        if sampled_tag:
            save_dirname += f"_{sampled_tag}"
    else:
        # PerVA_listener_train_seed{seed}_K{K}[_sub{S}][_n{N}][_with_desc|_no_desc][_cross_cat][_{model_tag}]
        save_dirname = f"PerVA_listener_train_seed{args.seed}_K{args.K}"
        if sub_tag:
            save_dirname += f"_{sub_tag}"
        if sampled_tag:
            save_dirname += f"_{sampled_tag}"
        save_dirname += "_with_desc" if args.use_description else "_no_desc"
        if args.cross_category:
            save_dirname += "_cross_cat"
        # Append model tag extracted from input filename (token between 'subset_NN_' and '_sampled_N')
        # e.g. "retrieval_top3_subset_30_original_7b_sampled_500.json" → "original_7b"
        try:
            after_subset = re.split(r'subset_\d+_', args.input_filename, maxsplit=1)[1]
            model_tag = re.sub(r'_sampled_\d+.*', '', after_subset.split('.json')[0])
            if model_tag:
                save_dirname += f"_{model_tag}"
        except IndexError:
            pass  # no model tag in filename (e.g. plain subset_30.json)

    out = save_dataset_dict(ds, save_dirname)
    print(f"Dataset saved in: {save_dirname}")

    # quick sanity-check load
    loaded = DatasetDict.load_from_disk(out)
    print("Loaded dataset summary:")
    for k, v in loaded.items():
        print(f"  split: {k},  n = {len(v)}")
    print("Columns:", loaded["train"].column_names)


if __name__ == "__main__":
    main()
