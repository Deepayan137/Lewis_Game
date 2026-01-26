# Data Preparation Pipeline

Scripts for creating and managing dataset catalogs, train/validation splits, and retrieval datasets.

## Overview

The data preparation pipeline has two main workflows:

**Catalog & Splits Pipeline:**
1. **Catalog Creation** — Scan raw image directories and create a structured JSON catalog
2. **Split Creation** — Take the catalog and create concept-level train/validation splits

**Retrieval Dataset Pipeline:**
3. **Build Retrieval (per category)** — Run CLIP-based hard negative mining (parallelizable via SLURM)
4. **Combine Retrieval Data** — Merge per-category retrieval results into a single file
5. **Convert to HuggingFace** — Create a training-ready HuggingFace Dataset

---

## Usage

### Step 1: Create Catalog from Raw Images

```bash
python src/data_prepare/01_build_image_catalog.py \
    --data_root data/PerVA/ \
    --out manifests/PerVA/catalog.json \
    --num_train 5 \
    --seed 42
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_root` | Yes | — | Path to dataset root directory |
| `--out` | Yes | — | Output JSON path |
| `--relative` | No | False | Store paths relative to data_root |
| `--train-fraction` | No | 0.2 | Fraction of images for train (when auto-splitting) |
| `--num_train` | No | None | Exact number of train images per concept (overrides fraction) |
| `--seed` | No | 42 | Random seed for reproducibility |
| `--train_dirname` | No | "train" | Name of train split folder |
| `--test_dirname` | No | "test" | Name of test split folder |

**Supported directory layouts:**
- **Explicit splits:** `data_root/train/<category>/<concept>/` and `data_root/test/<category>/<concept>/`
- **Auto-split:** `data_root/<category>/<concept>/` (images split automatically)

---

### Step 2: Create Concept-Level Train/Validation Splits

```bash
python src/data_prepare/02_create_concept_splits.py \
    --input_json manifests/PerVA/catalog.json \
    --out_dir manifests/PerVA \
    --concept_frac 0.65 \
    --min_concepts_threshold 3 \
    --seed 23
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_json` | Yes | — | Path to catalog JSON (from Step 1) |
| `--out_dir` | No | manifests/PerVA | Output directory |
| `--concept_frac` | No | 0.65 | Fraction of concepts to select for training |
| `--min_concepts_threshold` | No | 3 | Only split categories with at least this many concepts |
| `--test_sample_frac` | No | 1.0 | Fraction of test images to sample per concept |
| `--max_test_images` | No | 200 | Max test images per concept |
| `--seed` | No | 23 | Random seed |

**Outputs:**
- `train_combined_concepts_subset_*.json` — Training concepts with sampled test images merged into train
- `validation_combined_concepts_subset_*.json` — Remaining concepts (for validation)
- `train_val_combined_metadata_seed_*.json` — Metadata recording selections

---

### Step 3: Build Retrieval Data (Per Category)

This step runs CLIP-based retrieval to find hard negatives for each query image. Designed to run in parallel via SLURM array jobs.

```bash
# Single category
python src/data_prepare/03_build_retrieval_per_category.py \
    --category bag \
    --catalog manifests/PerVA/catalog.json \
    --out_dir outputs/PerVA \
    --seed 23 \
    --top_k 3

# SLURM array job (see scripts/run_retrieval_array.sh)
sbatch scripts/run_retrieval_array.sh
```

**Output:** `outputs/PerVA/{category}/seed_{seed}/retrieval_top{K}.json`

---

### Step 4: Combine Retrieval Data

After all per-category jobs complete, combine them into a single file.

```bash
python src/data_prepare/04_combine_retrieval_data.py \
    --input_dir outputs/PerVA \
    --input_filename retrieval_top3.json \
    --num_samples 1000 \
    --seed 23
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_dir` | No | outputs/PerVA | Base directory with category subdirectories |
| `--output_dir` | No | {input_dir}/all/seed_{seed}/ | Output directory |
| `--categories` | No | PerVA defaults | List of categories to combine |
| `--catalog` | No | None | Path to catalog JSON (alternative way to get categories) |
| `--input_filename` | No | retrieval_top3.json | Name of retrieval file in each category |
| `--output_filename` | No | auto-generated | Output filename |
| `--seed` | No | 42 | Seed (for locating files and sampling) |
| `--num_samples` | No | None | Random sample size (if set) |

**Output:** `outputs/PerVA/all/seed_{seed}/retrieval_top3_combined_{num_samples}.json`

---

### Step 5: Convert to HuggingFace Dataset

Convert the combined retrieval JSON to a HuggingFace Dataset for training.

```bash
python src/data_prepare/05_convert_to_hf_dataset.py \
    --root outputs \
    --dataset PerVA \
    --category all \
    --ret_json retrieval_top3_combined_1000.json \
    --seed 23
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--root` | No | outputs | Base output directory |
| `--dataset` | No | YoLLaVA | Dataset name (PerVA, YoLLaVA, MyVLM) |
| `--category` | No | all | Category name |
| `--ret_json` | No | retrieval_top5.json | Retrieval JSON filename |
| `--seed` | No | 23 | Random seed |

**Output:** HuggingFace Dataset saved to `share_data/{dataset}_{category}_test_subset_seed_{seed}_K_{K}/`

---

## File Structure

```
src/data_prepare/
├── utils.py                          # Shared utilities (JSON I/O, summarization)
│
│  # Catalog & Splits Pipeline
├── 01_build_image_catalog.py         # Step 1: Raw images → catalog JSON
├── 02_create_concept_splits.py       # Step 2: Catalog → train/val concept splits
│
│  # Retrieval Dataset Pipeline
├── 03_build_retrieval_per_category.py  # Step 3: CLIP retrieval (per category, SLURM-friendly)
├── 04_combine_retrieval_data.py        # Step 4: Merge per-category retrieval results
├── 05_convert_to_hf_dataset.py         # Step 5: Convert to HuggingFace Dataset
│
│  # Utilities
├── select_perva_subset.py            # Select k concepts to hit target image count
├── create_perva_concepts.py          # Create PerVA concept definitions
└── data_analysis.py                  # Dataset analysis utilities
```

---

## SLURM Array Job Example

For running retrieval on multiple categories in parallel:

```bash
#!/bin/bash
#SBATCH --job-name=retrieval
#SBATCH --array=0-10
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

CATEGORIES=(bag book clothe cup decoration pillow plant retail toy tumbler veg)
CATEGORY=${CATEGORIES[$SLURM_ARRAY_TASK_ID]}

python src/data_prepare/03_build_retrieval_per_category.py \
    --category $CATEGORY \
    --catalog manifests/PerVA/catalog.json \
    --out_dir outputs/PerVA \
    --seed 23 \
    --top_k 3
```

After all jobs complete:
```bash
python src/data_prepare/04_combine_retrieval_data.py --input_dir outputs/PerVA --seed 23
python src/data_prepare/05_convert_to_hf_dataset.py --dataset PerVA --category all --seed 23
```
