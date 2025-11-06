import os
import argparse
import random
import json
categories = ['bag', 'book', 'clothe', 'cup', 'decoration', 'pillow', 'plant', 'retail', 'toy', 'tumbler' ,'veg']

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        import json
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine retrieval data for PerVA categories")
    parser.add_argument('--with_negative', action='store_true', help='Include negative samples in the data')
    parser.add_argument('--seed', type=int, default=42, help='Include negative samples in the data')
    parser.add_argument('--num_samples', type=int, default=500, help='Include negative samples in the data')
    args = parser.parse_args()
    if args.with_negative:
        file_name = 'retrieval_top3_with_negative.json'
    else:
        file_name = 'retrieval_top3.json'
    all_data = []
    for category in categories:
        path = f'outputs/PerVA/{category}/seed_{args.seed}/{file_name}'
        data = load_json(path)
        all_data.extend(data)
        seed = args.seed
        # Only do this after the last category
        
    random.seed(seed)
    sampled_data = random.sample(all_data, min(args.num_samples, len(all_data)))
    print(len(all_data))
    out_dir = f'outputs/PerVA/all/seed_{seed}'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)