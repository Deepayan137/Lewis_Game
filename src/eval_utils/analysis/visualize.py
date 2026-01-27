#!/usr/bin/env python3
"""
compare_model_outputs.py

Compare outputs of two models and report cases where:
  - model1 makes a mistake (pred != gt)
  - model2 is correct (pred == gt)

Usage:
  python compare_model_outputs.py <model1_root> <model2_root> --dataset PerVA \
      [--categories bag book ...] [--seeds 23 42 63] [--out out.json]

The script finds JSON files recursively under each model root, loads those that contain
a "results" list (or are a list), and indexes entries by the `image` field inside each result.
It extracts (category, concept, filename) from the image path (e.g. data/PerVA/test_/bag/bkq/1.jpg)
so that entries are matched even if file-system layouts differ.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import csv
import sys
import os
# Default categories (use your provided list)
DEFAULT_CATEGORIES = [
  "bag","book","bottle","bowl","clothe","cup","decoration","headphone","pillow","plant",
  "plate","remote","retail","telephone","tie","towel","toy","tro_bag","tumbler","umbrella","veg"
]

def read_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rrg(args, concept, path):
    if os.path.exists(path):
        data = read_json_file(path)
        return data["results"]
    return None

# def load_r2p(args, concept):
#     path = f"/gpfs/projects/ehpc171/ddas/projects/R2P/outputs/QWEN_{args.dataset}_seed_{args.seed}/{args.category}/{concept}/recall_results.json"
#     if os.path.exists(path):
#         data = read_json_file(path)
#         return data['results']
#     return None

def main():
    parser = argparse.ArgumentParser(description="Compare model outputs")
    parser.add_argument("--seed", type=int, default=23, help="Seed value")
    # parser.add_argument("--concept", type=str, default='fzn', help="Concept name")
    parser.add_argument("--category", type=str, default='all', help="Category name")
    parser.add_argument("--dataset", type=str, default='YoLLaVA', help="Dataset name")
    args = parser.parse_args()
    refined_db_path = f'outputs/{args.dataset}/{args.category}/seed_{args.seed}/descriptions_original_7b_location_and_state_refined.json'
    refined_db = read_json_file(refined_db_path)
    ft_db_path = f'outputs/{args.dataset}/{args.category}/seed_{args.seed}/descriptions_r64_a1024.json'
    ft_db = read_json_file(ft_db_path)
    concepts = os.listdir(f'results/{args.dataset}/{args.category}/')
    final=[]
    for concept in concepts:
        data_refined = load_rrg(args, concept, f'results/{args.dataset}/{args.category}/{concept}/seed_{args.seed}/results_model_original_7b_db_r64_a1024_k_3.json')
        data_ft = load_rrg(args, concept, f'results/{args.dataset}/{args.category}/{concept}/seed_{args.seed}/results_model_original_7b_db_original_7b_location_and_state_refined_k_3.json')
        concept_dict = {}
        # if data_refined and data_ft:
        for item in data_refined:
            refined_rets = [ret_path.split('/')[-2] for ret_path in item['ret_paths']]
            refined_reason = item['response'][0]
            image_key = item['image_path']
            if args.dataset != 'MyVLM':
                image_key = image_key.replace('/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/', '')
            is_correct = item['correct']
            refined_pred_name = item['pred_name']
            for item2 in data_ft:
                if item2['image_path'] == image_key:
                    ft_pred_name = item2["pred_name"]
                    ft_reason = item2['response'][0]
                    if not is_correct and (item2["pred_name"] == item2["solution"]):
                        ft_rets = [ret_path.split('/')[-2] for ret_path in item2['ret_paths']]
                        if 'ft_rets' in locals() and 'refined_rets' in locals():
                            common_rets = set(refined_rets).intersection(set(ft_rets))
                            if ft_pred_name in refined_rets:
                                mydict={
                                    "query_image_path":image_key,
                                    "ret_paths": item2['ret_paths'],
                                    "refined_pred":refined_pred_name,
                                    "refined_pred_desc":refined_db[refined_pred_name],
                                    "refined_reasoning":refined_reason,
                                    "ft_pred":ft_pred_name,
                                    "ft_pred_desc":ft_db[ft_pred_name],
                                    "ft_reasoning":ft_reason
                                }
                                final.append(mydict)

    output_filename = f"{args.dataset}_for_visualize_{args.category}.json"
    with open(output_filename, "w") as outfile:
        json.dump(final, outfile, indent=2)
    print(f"Saved output to {output_filename}")
                                        
if __name__ == "__main__":
    main()
