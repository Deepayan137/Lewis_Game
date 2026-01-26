# Inference Pipeline

This directory contains the core inference scripts for MLLM personalization tasks.

## Overview

The pipeline consists of two main stages:

1. **Description Generation** — Generate attribute-focused descriptions for reference images
2. **Evaluation Tasks** — Run personalization, recognition, or VQA evaluation

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐      ┌──────────────────────────────────────────┐ │
│  │  Reference       │      │  Description Generation                  │ │
│  │  Images          │ ───► │  (generate_descriptions.py)              │ │
│  │  (train split)   │      │  Outputs: database_{model}.json          │ │
│  └──────────────────┘      └────────────────┬─────────────────────────┘ │
│                                             │                            │
│                                             ▼                            │
│  ┌──────────────────┐      ┌──────────────────────────────────────────┐ │
│  │  Query Images    │      │  Evaluation Tasks                        │ │
│  │  (test split)    │ ───► │  • personalize.py (identification)       │ │
│  │                  │      │  • recognition.py (binary yes/no)        │ │
│  └──────────────────┘      │  • vqa.py (visual QA)                    │ │
│                            └──────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before running inference, you need:

1. **Data catalog** — Created using the data preparation pipeline
2. **Model weights** — Either base Qwen2-VL or LoRA-finetuned checkpoints

For data preparation, see: [Data Preparation README](data_prepare/README.md)

---

## Step 1: Generate Descriptions

Generate attribute-focused descriptions for all reference images in the training set.

```bash
python src/generate_descriptions.py \
    --data_name PerVA \
    --catalog_file main_catalog_seed_23.json \
    --category all \
    --model_type original_7b \
    --batch_size 4 \
    --seed 23 \
    --output_dir outputs
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_name` | YoLLaVA | Dataset name (PerVA, YoLLaVA, MyVLM) |
| `--catalog_file` | main_catalog_seed_23.json | Catalog JSON filename |
| `--category` | clothe | Category to process (or "all") |
| `--model_type` | original_7b | Model type (original_2b, original_7b, or LoRA path) |
| `--batch_size` | 4 | Batch size for inference |
| `--max_new_tokens` | 128 | Maximum tokens to generate |
| `--temperature` | 1e-6 | Generation temperature |
| `--num_return_sequences` | 1 | Number of descriptions per image |
| `--seed` | 42 | Random seed |
| `--output_dir` | outputs | Output directory |
| `--copy_to_rap` | False | Copy database to RAP directory |

**Outputs:**
- `outputs/{data_name}/{category}/seed_{seed}/descriptions_{model_type}.json`
- `outputs/{data_name}/{category}/seed_{seed}/database_{model_type}.json`

---

## Step 2: Evaluation Tasks

### Task A: Personalized Identification

Given a query image and multiple reference descriptions (retrieved via CLIP), identify which reference matches the query.

```bash
python src/personalize.py \
    --data_name PerVA \
    --catalog_file main_catalog_seed_23.json \
    --category all \
    --model_type original_7b \
    --db_type original_7b \
    --k_retrieval 3 \
    --batch_size 4 \
    --seed 23 \
    --output_dir results
```

**Additional Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--db_type` | original_7b | Model type used for database descriptions |
| `--k_retrieval` | 3 | Number of references to retrieve |

**Output:** `results/{data_name}/{category}/seed_{seed}/results_model_{model}_db_{db}_k_{k}.json`

---

### Task B: Binary Recognition

Given a query image and a reference image with description, determine if they show the same object (Yes/No).

```bash
python src/recognition.py \
    --data_name PerVA \
    --category all \
    --model_type original_7b \
    --db_type original_7b \
    --batch_size 4 \
    --seed 23 \
    --output_dir results
```

**Additional Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--retrieval_json` | None | Path to retrieval JSON (auto-generated if not set) |

**Output:** `results/{data_name}/{category}/seed_{seed}/recognition_model_{model}_db_{db}.json`

**Metrics:** Overall accuracy, Yes accuracy, No accuracy

---

### Task C: Visual Question Answering (VQA)

Answer questions about personalized concepts (e.g., "What is \<bo\> doing?").

```bash
python src/vqa.py \
    --data_name YoLLaVA \
    --category all \
    --model_type original_7b \
    --db_type original_7b \
    --batch_size 4 \
    --seed 23 \
    --output_dir results
```

**Additional Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--qa_file` | None | Path to VQA JSON (uses default per dataset if not set) |

**Output:** `results/{data_name}/{category}/seed_{seed}/vqa_model_{model}_db_{db}.json`

---

## Common Arguments

All scripts share these common arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_name` | YoLLaVA | Dataset name |
| `--catalog_file` | main_catalog_seed_23.json | Catalog JSON filename |
| `--category` | all | Category to evaluate |
| `--concept_name` | "" | Specific concept (optional filter) |
| `--seed` | 23 | Random seed |
| `--model_type` | original_7b | Inference model type |
| `--db_type` | original_7b | Database model type |
| `--batch_size` | 4 | Batch size |
| `--max_new_tokens` | 128 | Max generation tokens |
| `--temperature` | 0.7 | Generation temperature |
| `--output_dir` | results | Output directory |

---

## Model Types

### Base Models
- `original_2b` — Qwen/Qwen2-VL-2B-Instruct
- `original_7b` — Qwen/Qwen2-VL-7B-Instruct

### LoRA-Finetuned Models
- `lora_7b_grpo` — GRPO-finetuned 7B model (requires dataset and seed)
- `lora_2b_grpo` — GRPO-finetuned 2B model

Custom model paths can also be passed directly to `--model_type`.

---

## File Structure

```
src/
├── README.md                    # This file
│
│  # Main Task Scripts
├── generate_descriptions.py     # Step 1: Generate descriptions
├── personalize.py               # Task A: Personalized identification
├── recognition.py               # Task B: Binary recognition
├── vqa.py                       # Task C: Visual QA
│
│  # Shared Utilities
├── inference_utils/
│   ├── common.py                # Shared utilities (seed, model paths, etc.)
│   ├── prompts.py               # Prompt templates
│   ├── cleanup.py               # Response parsing utilities
│   ├── dataset.py               # Dataset classes
│   ├── model.py                 # Model loading utilities
│   └── retriever.py             # CLIP retrieval
│
│  # Data Preparation (separate pipeline)
├── data_prepare/                # See data_prepare/README.md
│   ├── 01_build_image_catalog.py
│   ├── 02_create_concept_splits.py
│   ├── 03_build_retrieval_per_category.py
│   ├── 04_combine_retrieval_data.py
│   └── 05_convert_to_hf_dataset.py
│
│  # Other
├── defined.py                   # Category mappings (legacy)
└── analysis.py                  # Result analysis utilities
```

---

## Example: Full Pipeline

```bash
# 1. Create data catalog (see data_prepare/README.md)
python src/data_prepare/01_build_image_catalog.py \
    --data_root data/PerVA \
    --out manifests/PerVA/catalog.json

# 2. Generate descriptions for training images
python src/generate_descriptions.py \
    --data_name PerVA \
    --category all \
    --model_type original_7b \
    --seed 23

# 3. Run personalization evaluation
python src/personalize.py \
    --data_name PerVA \
    --category all \
    --model_type original_7b \
    --db_type original_7b \
    --seed 23

# 4. Run recognition evaluation
python src/recognition.py \
    --data_name PerVA \
    --category all \
    --model_type original_7b \
    --db_type original_7b \
    --seed 23

# 5. Run VQA evaluation (YoLLaVA dataset)
python src/vqa.py \
    --data_name YoLLaVA \
    --category all \
    --model_type original_7b \
    --db_type original_7b \
    --seed 23
```

---

## Output Format

All evaluation scripts save results in a standardized JSON format:

```json
{
  "config": {
    "data_name": "PerVA",
    "model_type": "original_7b",
    "seed": 23,
    ...
  },
  "metrics": {
    "accuracy": 0.85,
    "correct": 170,
    "total": 200
  },
  "results": [
    {
      "image_path": "...",
      "solution": "A",
      "pred": "A",
      "correct": true,
      ...
    }
  ]
}
```
