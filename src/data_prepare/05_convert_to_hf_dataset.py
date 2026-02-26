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

def json_to_dataset_dict(input_filename, with_negative, cross_category, use_description, multi_view=False, sp_detailed=False):
    retrieval_json_path = input_filename
    with open(retrieval_json_path, "r") as f:
        data = json.load(f)
    rows = []
    # model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
    # speaker_model, processor = setup_model(model_name_or_path)
    for idx, item in enumerate(tqdm(data, desc="Processing retrieval entries")):
        # accept either field name
        ret_list = item.get("retrieved_paths") or item.get("ret_paths") or item.get("ret_paths", [])
        query_path_raw = item.get("query_path") or item.get("query")
        if query_path_raw is None:
            continue
        view1_abs = _resolve_path('./', query_path_raw)
        query_concept_name = query_path_raw.split('/')[-2]
        category_name      = query_path_raw.split('/')[-3]

        # make retrieved absolute where possible
        retrieved_abs = [_resolve_path('./', rp) for rp in ret_list]
        names = [string.ascii_uppercase[i] for i in range(len(retrieved_abs))]

        if not os.path.exists(view1_abs):
            print(f"Warning: query image not found: {view1_abs}")

        perva_category_map = {
            'veg': 'vegetable',
            'decoration': 'decoration object',
            'retail': 'retail object',
            'tro_bag': 'trolley bag',
        }
        query_category = perva_category_map.get(category_name, category_name)

        # Load speaker image(s)
        view1_image = Image.open(view1_abs).convert('RGB')
        speaker_images = view1_image       # single image, original behaviour
        speaker_problem = PromptTemplates.get_speaker_prompt(query_category)
        uid = idx
        # Target distribution: 50% positive, 50% negative
        is_positive = random.random() < 0.5
        # import pdb;pdb.set_trace()
        if is_positive:
            # Sample positive: same concept as query
            positive_candidates = [p for p in retrieved_abs if p.split('/')[-2] == query_concept_name]
            if positive_candidates:
                ref_path = random.choice(positive_candidates)
                ref_index = retrieved_abs.index(ref_path)
                ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
                # Try to get description from for_recognition if it matches
                if use_description:
                    ref_concept_description = item['ret_descs'][ref_index] if ref_index < len(item.get('ret_descs', [])) else "No description available."
                    # if ref_path == item['for_recognition']['ref_path']:
                    #     ref_concept_description = item['for_recognition']['descriptions']['coarse'][0] + '. ' + item['for_recognition']['descriptions']['detailed'][0]
                    # else:
                    #     ref_concept_description = "No description available."
                        # name = ref_path.split('/')[-2]
                        # batch_items = [{
                        #     'image': Image.open(ref_path).convert('RGB'),
                        #     'name': name,
                        #     'problem': get_description_prompt(ref_category),
                        #     'path': ref_path
                        # }]
                        # raw_results = process_batch_efficiently(speaker_model, processor, batch_items=batch_items, max_new_tokens=100)
                        # _, _, desc = raw_results[0]
                        # parsed = parse_descriptions(desc[0])
                        # coarse = parsed["coarse"] or f"A photo of a {ref_category}"
                        # detailed = parsed["detailed"]
                        # ref_concept_description = coarse + ". " + detailed
            else:
                # Fallback if no positive candidates (shouldn't happen)
                ref_path = random.choice(retrieved_abs)
                ref_index = retrieved_abs.index(ref_path)
                ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
                if use_description:
                    ref_concept_description = item['ret_descs'][ref_index] if ref_index < len(item.get('ret_descs', [])) else "No description available."
        else:
            # Sample negative: different concept from query
            if cross_category and random.random() < 0.4:  # 40% of negatives are cross-category (20% overall)
                # Sample from different category
                ref_path, ref_concept_description = sample_random_image_from_different_category(data, category_name)
                ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
            else:
                # Sample within-category negative (60% of negatives = 30% overall if cross_category, 100% if not)
                negative_candidates = [p for p in retrieved_abs if p.split('/')[-2] != query_concept_name]

                if negative_candidates:
                    ref_path = random.choice(negative_candidates)
                    ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
                    ref_index = retrieved_abs.index(ref_path)
                    # Try to get description from for_recognition if it matches
                    if use_description:
                        ref_concept_description = item['ret_descs'][ref_index] if ref_index < len(item.get('ret_descs', [])) else "No description available."
                        # if ref_path == item['for_recognition']['ref_path']:
                        #     ref_concept_description = item['for_recognition']['descriptions']['coarse'][0] + '. ' + item['for_recognition']['descriptions']['detailed'][0]
                        # else:
                        #     name = ref_path.split('/')[-2]
                        #     batch_items = [{
                        #         'image': Image.open(ref_path).convert('RGB'),
                        #         'name': name,
                        #         'problem': get_description_prompt(ref_category),
                        #         'path': ref_path
                        #     }]
                        # raw_results = process_batch_efficiently(speaker_model, processor, batch_items=batch_items, max_new_tokens=100)
                        # _, _, desc = raw_results[0]
                        # parsed = parse_descriptions(desc[0])
                        # coarse = parsed["coarse"] or f"A photo of a {ref_category}"
                        # detailed = parsed["detailed"]
                        # ref_concept_description = coarse + ". " + detailed
                else:
                    # Fallback: use for_recognition even if it might be positive
                    ref_path = random.choice(retrieved_abs)
                    ref_index = retrieved_abs.index(ref_path)
                    ref_category = perva_category_map.get(ref_path.split('/')[-3], ref_path.split('/')[-3])
                    if use_description:
                        ref_concept_description = item['ret_descs'][ref_index] if ref_index < len(item.get('ret_descs', [])) else "No description available."
        reference_image = Image.open(ref_path).convert('RGB')
        reference_image = reference_image.resize(view1_image.size, Image.LANCZOS)
        ref_concept_name = ref_path.split('/')[-2]
        test_question = PromptTemplates.get_test_question(ref_concept_name)
        if use_description:
            listener_problem = PromptTemplates.get_listener_prompt_with_description(ref_concept_description, test_question)
        else:
            listener_problem = PromptTemplates.get_listener_prompt_no_description(ref_concept_name, ref_category, test_question)
        listener_solution = 'yes' if query_concept_name == ref_concept_name else 'no'
        rows.append({
            "image": speaker_images,
            "ret_paths": retrieved_abs,
            "names": names,
            "speaker_problem": speaker_problem,
            "solution": names[item.get("label")],
            "category": query_category,
            "example_idx": uid,
            "query_image_path": query_path_raw,
            "reference_image_path": ref_path,
            "query_image": add_marker_to_image(view1_image, marker_text="1"),
            "reference_image": add_marker_to_image(reference_image, marker_text="2"),
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
    ap.add_argument("--cross_category", action='store_true', help="Whether to sample negatives from different categories (cross-category) or same category (within-category).")
    ap.add_argument("--use_description", action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    ds = json_to_dataset_dict(args.input_filename, with_negative=False, cross_category=args.cross_category,
                              use_description=args.use_description)
    print_dataset_statistics(ds)
    subset_match = re.search(r'subset_(\d+)', args.input_filename)
    subset = f"subset_{subset_match.group(1)}" if subset_match else None
    sampled_match = re.search(r'sampled_(\d+)', args.input_filename)
    sampled = f"sampled_{sampled_match.group(1)}" if sampled_match else None
    save_dirname = f"PerVA_for_Listener_seed_{args.seed}_K_{args.K}"
    if subset:
        save_dirname += f"_{subset}"
    if sampled:
        save_dirname += f"_{sampled}"
    if args.use_description:
        save_dirname += "_reco_with_desc"
    else:
        save_dirname += "_reco_no_desc"
    if args.cross_category:
        save_dirname += "_cross_category"
    file_identifier = args.input_filename.split('subset_30_')[1].split('.json')[0]
    save_dirname += f"_{file_identifier}"
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
