import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def gather_images(folder: Path, relative_to: Path = None) -> List[str]:
    imgs = sorted([p for p in folder.iterdir() if is_image_file(p)])
    if relative_to:
        return [str(p.resolve()) for p in imgs]
    return [str(p) for p in imgs]


def build_catalog_from_explicit_splits(data_root: Path, train_dirname: str, test_dirname: str, relative: bool, num_train:int):
    """
    Build catalog using explicit train_/test_ subfolders (your layout).
    """
    catalog = {}
    train_dir = data_root / train_dirname
    test_dir = data_root / test_dirname
    base = data_root if relative else None

    # gather set of categories appearing in either train_ or test_
    categories = set()
    if train_dir.exists():
        categories.update([p.name for p in train_dir.iterdir() if p.is_dir()])
    if test_dir.exists():
        categories.update([p.name for p in test_dir.iterdir() if p.is_dir()])

    for cat in sorted(categories):
        catalog[cat] = {}
        train_cat_dir = train_dir / cat
        test_cat_dir = test_dir / cat
        # import pdb;pdb.set_trace()
        # unify set of concepts under category
        concepts = set()
        if train_cat_dir.exists():
            concepts.update([p.name for p in train_cat_dir.iterdir() if p.is_dir()])
        if test_cat_dir.exists():
            concepts.update([p.name for p in test_cat_dir.iterdir() if p.is_dir()])

        for concept in sorted(concepts):
            train_list = []
            test_list = []
            neg_list = []
            tdir = train_cat_dir / concept
            if tdir.exists():
                train_list = gather_images(tdir, relative_to=base)
                n = len(train_list)
                k = max(1, min(num_train, n))
                shuffled = train_list.copy()
                random.shuffle(shuffled)
                train_list = shuffled[:k]
            sdir = test_cat_dir / concept
            if sdir.exists():
                test_list = gather_images(sdir, relative_to=base)
            ndir = tdir / 'laion'
            if ndir.exists():
                neg_list = gather_images(ndir, relative_to=base)
            catalog[cat][concept] = {"train": train_list, "test": test_list, "negative":neg_list}

    return catalog


def build_catalog_by_splitting(data_root: Path, relative: bool, train_fraction: float, num_train: int, seed: int):
    """
    Build catalog by scanning data_root/<category>/<concept> and splitting images per concept.
    """
    random.seed(seed)
    catalog = {}
    base = data_root if relative else None
    # categories are immediate subdirectories of data_root
    for cat_path in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        # skip folders that are obviously the explicit split folders
        if cat_path.name in {"train_", "test_"}:
            continue
        cat = cat_path.name
        catalog[cat] = {}
        # concepts are immediate subdirectories of category
        concept_dirs = [p for p in cat_path.iterdir() if p.is_dir()]
        for concept_path in sorted(concept_dirs):
            concept = concept_path.name
            images = gather_images(concept_path, relative_to=base)
            n = len(images)
            if n == 0:
                # skip empty concepts
                continue

            if num_train is not None:
                k = max(1, min(num_train, n))
            else:
                k = max(1, int(round(train_fraction * n)))
            k = 1
            shuffled = images.copy()
            random.shuffle(shuffled)
            train_list =  shuffled[:k]
            test_list = shuffled[k:]
            catalog[cat][concept] = {"train": train_list, "test": test_list}

    return catalog


def save_json(obj, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(obj, f, indent=2)
    print(f"Wrote catalog JSON to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Create a catalogue JSON from a data folder (handles train_/test_ layout).")
    p.add_argument("--data_root", required=True, help="Path to dataset root (e.g., data/PerVA).")
    p.add_argument("--out", required=True, help="Output JSON path (e.g., manifests/catalogue.json).")
    p.add_argument("--relative", action="store_true",
                   help="Store image paths relative to --data-root (recommended).")
    p.add_argument("--train-fraction", type=float, default=0.2,
                   help="Fraction of images per concept to use for train when splitting (default 0.2).")
    p.add_argument("--num_train", type=int, default=None,
                   help="If set, use exact number of images per concept for train (overrides fraction).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits.")
    p.add_argument("--train_dirname", type=str, default="train", help="Name of train split folder (default 'train_').")
    p.add_argument("--test_dirname", type=str, default="test", help="Name of test split folder (default 'test_').")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    # prefer relative paths by default (user asked earlier for relative paths)
    relative = args.relative

    # check for explicit train_/test_ layout you described
    train_dir = data_root / args.train_dirname
    test_dir = data_root / args.test_dirname

    if train_dir.exists() or test_dir.exists():
        print(f"Detected explicit split layout under {data_root} using '{args.train_dirname}' and '{args.test_dirname}'.")
        catalog = build_catalog_from_explicit_splits(data_root, args.train_dirname, args.test_dirname, relative, args.num_train)
    else:
        print(f"No explicit train_/test_ directories found. Scanning categories under {data_root} and splitting per concept.")
        catalog = build_catalog_by_splitting(
            data_root,
            relative=relative,
            train_fraction=args.train_fraction,
            num_train=args.num_train,
            seed=args.seed
        )

    save_json(catalog, out_path)


if __name__ == "__main__":
    main()

#USAGE
#python src/data_prepare/build_catalog.py --data_root data/PerVA/ --out manifests/PerVA/catalog.json