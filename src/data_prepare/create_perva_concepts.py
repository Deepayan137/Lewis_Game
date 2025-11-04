import os
import json
import argparse

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PerVA concepts script.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    seed = args.seed

    catalog_file = os.path.join(f'manifests/PerVA/train_combined_seed_{args.seed}.json')
    data = read_json_file(catalog_file)
    
    for key in data.keys():
        for concept in data[key].keys():
            with open(f'OSC_train_subset_seed_{seed}.txt', 'a') as f:
                f.write(f'{key},{concept}\n')
    