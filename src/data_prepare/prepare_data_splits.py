#!/usr/bin/env python3
import os
import json
import random
import argparse

def split_data(root, 
        catalog_name,
        seed=23, 
        train_fraction=0.20, 
        num_train_per_category=None, 
        out_dir=None):
    random.seed(seed)
    json_path = os.path.join(root, catalog_name)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Catalog JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    if out_dir is None:
        out_dir = root
    os.makedirs(out_dir, exist_ok=True)
    categories = list(data.keys())
    tr_data, te_data = {}, {}
    for category in categories:
        tr_data[category] = {}
        te_data[category] = {}
        concepts = list(data[category].keys())
        num_concepts = len(concepts)
        # Shuffle deterministically using seed
        random.shuffle(concepts)

        if num_concepts == 0:
            train_concepts, test_concepts = [], []
        else:
            if num_train_per_category is not None:
                k = max(1, min(num_train_per_category, num_concepts))
            else:
                # compute via fraction; ensure at least 1 (when num_concepts >= 1)
                k = max(1, int(round(train_fraction * num_concepts))) if num_concepts > 0 else 0
            train_concepts = concepts[:k]
            test_concepts = concepts[k:]

        for con in train_concepts:
            tr_data[category][con] = data[category][con]
        for con in test_concepts:
            te_data[category][con] = data[category][con]

    train_json_path = os.path.join(out_dir, f'train_catalog_seed_{seed}.json')
    test_json_path  = os.path.join(out_dir, f'test_catalog_seed_{seed}.json')

    with open(train_json_path, 'w') as f:
        json.dump(tr_data, f, indent=2)
    with open(test_json_path, 'w') as f:
        json.dump(te_data, f, indent=2)

    print(f"Train split saved to {train_json_path}")
    print(f"Test  split saved to {test_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Divide train/test concepts for Lewis Game")
    parser.add_argument('--root', type=str,
                        default="manifests/PerVA/",
                        help='Root directory for data')
    parser.add_argument('--catalog_name', type=str,
                        default="main_catalog.json",
                        help='JSON filename with data splits (catalogue)')
    parser.add_argument('--seed', type=int, default=23, help='Random seed for shuffling')
    parser.add_argument('--train-fraction', type=float, default=0.20,
                        help='Fraction of concepts per category to use for training (default 0.20)')
    parser.add_argument('--num-train', type=int, default=None,
                        help='If set, override fraction and use this many train concepts per category (min 1)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Optional output directory (defaults to --root)')
    args = parser.parse_args()

    split_data(
        root=args.root,
        catalog_name=args.catalog_name,
        seed=args.seed,
        train_fraction=args.train_fraction,
        num_train_per_category=args.num_train,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()

#USAGE
#python src/data_prepare/prepare_data_splits.py  --root manifests/PerVA/ --catalog_name main_catalog.json --seed 23 --out_dir manifests/PerVA/