#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import json
import sys
sys.path.insert(0, 'src')
# Import dicts from your repo (handle both 'defined.py' and the possible typo 'definened.py')
try:
    from defined import yollava_reverse_category_dict, myvlm_reverse_category_dict
except Exception:
    print("ERROR: Could not import dictionaries from src/defined.py or src/definened.py", file=sys.stderr)
    raise

def get_concept_names(dataset: str) -> list[str]:
    ds = dataset.strip().lower()
    if ds == "yollava":
        keys = yollava_reverse_category_dict.keys()
    elif ds == "myvlm":
        keys = myvlm_reverse_category_dict.keys()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected 'YoLLaVA' or 'MyVLM'.")
    # Ensure everything is a string and stable order
    return sorted(str(k) for k in keys)

def save_list(items: list[str], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if outpath.suffix.lower() == ".json":
        import json
        outpath.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        outpath.write_text("\n".join(items) + "\n", encoding="utf-8")

def parse_args():
    p = argparse.ArgumentParser(description="Extract concept names for a dataset.")
    p.add_argument("--dataset", default='YoLLaVA', help="Dataset name: YoLLaVA or MyVLM")
    p.add_argument("--db_type", default='original')
    p.add_argument("--model_type", default='base_qwen')
    p.add_argument("--category", default='all')
    p.add_argument("--seed", default=23)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    concept_names = get_concept_names(args.dataset)

    # Default output path if not provided
    if args.out is None:
        outpath = Path("results") / args.dataset / args.category / 'concepts'
    else:
        outpath = Path(args.out)

    # Print to stdout
    print(f"# Concepts for dataset={args.dataset}, seed={args.seed} (count={len(concept_names)})")
    correct_count, total = 0, 0
    results = {
        "metrics":{
            "correct":0, 
            "total":0,
            "accuracy":0.},
        "concepts":{}}
    for name in concept_names:
        concept_path = Path(outpath) / name / f"seed_{str(args.seed)}" / f"results_model_{args.model_type}_db_{args.db_type}_k_3.json"
        if concept_path.exists():
            import json
            with concept_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        results['metrics']["correct"] += data['metrics']["correct count"]
        results['metrics']["total"] += data['metrics']['total samples']
        results['concepts'][name] = {"acc":(data['metrics']["correct count"] / data['metrics']['total samples'])*100,
        "total":data['metrics']['total samples']}
    results["metrics"]["accuracy"] = (results['metrics']["correct"]/results['metrics']["total"])*100.
    save_path = Path("results") / args.dataset / f"results_model_{args.model_type}_db_{args.db_type}_seed_{args.seed}_k_3.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model at {save_path}")
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(results["metrics"])
if __name__ == "__main__":
    main()
