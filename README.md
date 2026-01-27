# Lewis Game - Vision-Language Model Training & Evaluation

A comprehensive framework for training and evaluating vision-language models on personalized concept understanding using the Lewis Game paradigm with GRPO (Generalized Reward Policy Optimization).

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipelines](#pipelines)
  - [Data Preparation](#1-data-preparation-pipeline)
  - [Training Pipeline](#2-training-pipeline-train_src)
  - [Inference & Evaluation](#3-inference--evaluation-pipeline-src)
  - [Description Processing](#4-description-processing-eval_utils)
- [Installation](#installation)
- [Configuration](#configuration)
- [Documentation Index](#documentation-index)
- [Recent Updates](#recent-updates)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project implements a multi-agent communication framework for training vision-language models to understand personalized concepts. The system consists of:

- **Speaker Agent**: Generates descriptions of images
- **Listener Agent**: Identifies images based on descriptions
- **GRPO Training**: Reinforcement learning with custom reward functions
- **Evaluation Suite**: Comprehensive testing across multiple tasks

### Key Features

✨ **Multi-Task Evaluation**
- Personalized concept identification
- Binary recognition (Yes/No matching)
- Visual question answering (VQA)

🚀 **Scalable Training**
- Distributed training with DeepSpeed
- LoRA efficient fine-tuning
- Flash Attention 2 support
- Multi-GPU coordination

🎯 **Custom Rewards**
- Accuracy-based rewards (listener feedback)
- Format rewards (structured outputs)
- Length rewards (conciseness)

📊 **Data Pipeline**
- Automated catalog creation
- CLIP-based retrieval
- Train/test split management
- HuggingFace dataset conversion

---

## Quick Start

### Prerequisites

```bash
# Clone repository
git clone <repository-url>
cd Lewis_Game

# Install dependencies
pip install -r requirements.txt

# Install flash attention
pip install flash-attn --no-build-isolation
```

### Minimal Example

```bash
# 1. Prepare data catalog
python src/data_prepare/01_build_image_catalog.py \
    --data_root data/PerVA \
    --out manifests/PerVA/catalog.json \
    --num_train 5 \
    --seed 23

# 2. Start listener service (Terminal 1)
CUDA_VISIBLE_DEVICES=0 python train_src/open_r1/listener_service.py \
    --host $(hostname -s) \
    --port 9000

# 3. Run training (Terminal 2)
./scripts/run_speaker_training.sh $(hostname -s) 9000 4 PerVA 23

# 4. Generate descriptions with trained model
python src/generate_descriptions.py \
    --data_name PerVA \
    --model_type lora_7b_grpo \
    --seed 23

# 5. Evaluate on personalization task
python src/personalize.py \
    --data_name PerVA \
    --model_type lora_7b_grpo \
    --db_type lora_7b_grpo \
    --k_retrieval 3 \
    --seed 23
```

---

## Project Structure

```
Lewis_Game/
│
├── README.md                          # This file - main documentation
├── requirements.txt                   # Python dependencies
├── git_.sh                           # Git helper script
│
├── src/                              # Inference & Evaluation Pipeline
│   ├── README.md                     # → Inference pipeline guide
│   │
│   ├── generate_descriptions.py      # Generate descriptions for reference images
│   ├── personalize.py                # Task A: Personalized identification
│   ├── recognition.py                # Task B: Binary recognition
│   ├── vqa.py                       # Task C: Visual QA
│   │
│   ├── data_prepare/                # Data preparation scripts
│   │   ├── README.md                # → Data pipeline guide
│   │   ├── 01_build_image_catalog.py
│   │   ├── 02_create_concept_splits.py
│   │   ├── 03_build_retrieval_per_category.py
│   │   ├── 04_combine_retrieval_data.py
│   │   └── 05_convert_to_hf_dataset.py
│   │
│   ├── inference_utils/             # Shared inference utilities
│   │   ├── common.py
│   │   ├── prompts.py
│   │   ├── cleanup.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── retriever.py
│   │
│   └── eval_utils/                  # Evaluation & analysis utilities
│       ├── description_processing/  # Description evaluation & refinement
│       │   ├── README.md           # → Description processing guide
│       │   ├── __init__.py
│       │   ├── prompts.py          # Prompt templates for Qwen
│       │   ├── shared.py           # Dataset & model utilities
│       │   ├── evaluator.py        # Evaluate state/location attributes
│       │   └── refiner.py          # Remove unwanted attributes
│       │
│       ├── eval_with_qwen.py       # CLI for description evaluation
│       ├── aggregate_identification.py
│       ├── aggregate_recognition.py
│       └── analysis/               # Result visualization
│
├── train_src/                       # Training Pipeline
│   ├── README.md                    # → Training pipeline guide (comprehensive)
│   │
│   └── open_r1/                    # GRPO training implementation
│       ├── __init__.py
│       ├── dist_helpers.py         # Distributed training utilities
│       ├── logger.py               # Training logging
│       │
│       ├── listener_service.py     # Listener model inference service
│       ├── speaker_service.py      # Speaker model inference service
│       │
│       ├── train_listener_dist.py  # Train listener with GRPO
│       ├── train_speaker_dist.py   # Train speaker with GRPO
│       │
│       └── trainer/                # Custom GRPO trainer implementations
│           ├── qwen_grpo_trainer.py
│           └── qwen_grpo_vllm_trainer.py
│
├── scripts/                         # Helper scripts
│   └── run_speaker_training.sh     # Training orchestration script
│
├── tests/                          # Unit tests
│
├── REFACTORING_SUMMARY.md          # Description processing refactoring
└── IMPORT_FIX_SUMMARY.md           # Import order fixes (PEP 8)
```

---

## Pipelines

### 1. Data Preparation Pipeline

**Purpose**: Create structured catalogs, splits, and retrieval datasets from raw images.

**Location**: `src/data_prepare/`

**Documentation**: 📖 [Data Preparation README](src/data_prepare/README.md)

**Steps**:
1. **Build Catalog** - Scan image directories → JSON catalog
2. **Create Splits** - Concept-level train/validation splits
3. **Build Retrieval** - CLIP-based hard negative mining (parallel)
4. **Combine Data** - Merge per-category results
5. **Convert to HF** - Create HuggingFace Dataset

**Example**:
```bash
# Step 1: Create catalog
python src/data_prepare/01_build_image_catalog.py \
    --data_root data/PerVA \
    --out manifests/PerVA/catalog.json \
    --num_train 5 \
    --seed 23

# Step 2: Create splits
python src/data_prepare/02_create_concept_splits.py \
    --input_json manifests/PerVA/catalog.json \
    --out_dir manifests/PerVA \
    --concept_frac 0.65 \
    --seed 23
```

**Outputs**:
- `catalog.json` - Full image catalog with metadata
- `train_combined_concepts_subset_*.json` - Training concepts
- `validation_combined_concepts_subset_*.json` - Validation concepts
- `retrieval_top{K}.json` - CLIP retrieval results

---

### 2. Training Pipeline (`train_src/`)

**Purpose**: Train speaker and listener models using GRPO with distributed training.

**Location**: `train_src/open_r1/`

**Documentation**: 📖 [Training Pipeline README](train_src/README.md) ⭐ *Comprehensive guide*

#### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   TRAINING ARCHITECTURE                     │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐                                   │
│  │  Listener Service   │◄──────────────┐                  │
│  │  (GPU 0, Port 9000) │               │                  │
│  │  • Batch scoring    │               │                  │
│  │  • FastAPI          │               │                  │
│  └─────────────────────┘               │                  │
│                                         │ HTTP              │
│                                         │ Request           │
│  ┌─────────────────────────────────────┴─────────────────┐ │
│  │  Speaker Training (Multi-GPU, DeepSpeed)              │ │
│  │  ┌────────────────────────────────────────────────┐   │ │
│  │  │  1. Generate descriptions                      │   │ │
│  │  │     (thinking + coarse + detailed)             │   │ │
│  │  └──────────────────┬─────────────────────────────┘   │ │
│  │                     ▼                                  │ │
│  │  ┌────────────────────────────────────────────────┐   │ │
│  │  │  2. Calculate Rewards                          │   │ │
│  │  │     • Accuracy (call listener)                 │   │ │
│  │  │     • Format (XML tags)                        │   │ │
│  │  │     • Length (conciseness)                     │   │ │
│  │  └──────────────────┬─────────────────────────────┘   │ │
│  │                     ▼                                  │ │
│  │  ┌────────────────────────────────────────────────┐   │ │
│  │  │  3. Policy Update (GRPO)                       │   │ │
│  │  │     • LoRA fine-tuning                         │   │ │
│  │  │     • Gradient accumulation                    │   │ │
│  │  └────────────────────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

#### Components

**Services**:
- `listener_service.py` - FastAPI service for listener inference
- `speaker_service.py` - FastAPI service for speaker inference (optional)

**Training Scripts**:
- `train_speaker_dist.py` - Train speaker with GRPO
- `train_listener_dist.py` - Train listener with GRPO

**Utilities**:
- `dist_helpers.py` - Distributed training helpers
- `logger.py` - WandB & JSON logging

#### Quick Start

**Terminal 1 - Start Listener**:
```bash
HOSTNAME=$(hostname -s)
PORT=9000
CUDA_VISIBLE_DEVICES=0 python train_src/open_r1/listener_service.py \
    --host ${HOSTNAME} \
    --port ${PORT}
```

**Terminal 2 - Run Training**:
```bash
./scripts/run_speaker_training.sh <LISTENER_HOST> [PORT] [EPOCHS] [DATASET] [SEED]

# Example:
./scripts/run_speaker_training.sh gpu-node-01 9000 4 PerVA 23
```

#### Training Configuration

**LoRA Settings** (via environment or flags):
```bash
export LORA_RANK=64
export LORA_ALPHA=128

# Or pass directly:
--lo_rank 64 --lo_alpha 128 --lo_dropout 0.001
```

**DeepSpeed ZeRO-3**:
```bash
--deepspeed configs/zero3.json
```

**Environment Variables**:
```bash
export LISTENER_URL="http://gpu-node-01:9000/batch_score"
export LISTENER_TIMEOUT=60
export DEBUG_MODE=true
export WANDB_PROJECT="GRPO_Lewis_Qwen_2VL"
```

#### Outputs

```
share_models/
└── Qwen2.5-VL-7B_GRPO_lewis_PerVA_seed_23_r64_a128/
    ├── adapter_config.json       # LoRA config
    ├── adapter_model.bin          # LoRA weights
    ├── trainer_state.json
    └── lewis_<job_id>.json       # Training logs
```

---

### 3. Inference & Evaluation Pipeline (`src/`)

**Purpose**: Generate descriptions and evaluate trained models on multiple tasks.

**Location**: `src/`

**Documentation**: 📖 [Inference Pipeline README](src/README.md)

#### Workflow

```
Reference Images → generate_descriptions.py → database_{model}.json
                                                      ↓
Query Images → Task Scripts → Results JSON
               ├─ personalize.py (identification)
               ├─ recognition.py (binary yes/no)
               └─ vqa.py (visual QA)
```

#### Tasks

**Task A: Personalized Identification**
```bash
python src/personalize.py \
    --data_name PerVA \
    --model_type original_7b \
    --db_type original_7b \
    --k_retrieval 3 \
    --seed 23
```

**Task B: Binary Recognition**
```bash
python src/recognition.py \
    --data_name PerVA \
    --model_type original_7b \
    --db_type original_7b \
    --seed 23
```

**Task C: Visual QA**
```bash
python src/vqa.py \
    --data_name YoLLaVA \
    --model_type original_7b \
    --db_type original_7b \
    --seed 23
```

#### Model Types

| Type | Description | Path |
|------|-------------|------|
| `original_2b` | Qwen2-VL-2B-Instruct | Qwen/Qwen2-VL-2B-Instruct |
| `original_7b` | Qwen2-VL-7B-Instruct | Qwen/Qwen2-VL-7B-Instruct |
| `lora_7b_grpo` | GRPO-finetuned 7B | share_models/Qwen2.5-VL-7B_GRPO_... |

---

### 4. Description Processing (`eval_utils/`)

**Purpose**: Evaluate and refine descriptions for state/location attributes.

**Location**: `src/eval_utils/description_processing/`

**Documentation**: 📖 [Description Processing README](src/eval_utils/description_processing/README.md)

#### Modular Structure

```
description_processing/
├── prompts.py           # Qwen prompt templates
├── shared.py            # Dataset, model, inference utilities
├── evaluator.py         # Identify state/location attributes
└── refiner.py           # Remove unwanted attributes
```

#### Usage

**Evaluation** (detect state/location attributes):
```bash
python src/eval_utils/eval_with_qwen.py \
    --input outputs/PerVA/all/seed_23/descriptions_original_7b.json \
    --out results/ \
    --refine none \
    --batch-size 2
```

**Refinement** (remove state attributes):
```bash
python src/eval_utils/eval_with_qwen.py \
    --input outputs/PerVA/all/seed_23/descriptions_original_7b.json \
    --out results/ \
    --refine state \
    --batch-size 2
```

**Refinement modes**:
- `state` - Remove state attributes (standing, sitting, wearing, etc.)
- `location` - Remove location attributes (on table, in kitchen, etc.)
- `location_and_state` - Remove both

#### Output Format

```json
{
  "stats": {
    "mean_state": 0.45,
    "mean_location": 0.32,
    "mean_length": 25.3
  },
  "response": [
    {
      "id": "obj_001",
      "has_state": true,
      "has_location": false,
      "length": 28,
      "text": "The dog has a fluffy coat..."
    }
  ]
}
```

---

## Installation

### System Requirements

**Hardware**:
- GPU: 2+ NVIDIA A100 40GB (or equivalent)
- RAM: 256GB+ recommended
- Storage: 500GB+ for datasets and models

**Software**:
- Python 3.10+
- CUDA 12.1+
- PyTorch 2.0+

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
pip install -r requirements.txt

# Install Flash Attention 2 (recommended)
pip install flash-attn --no-build-isolation

# Install DeepSpeed (for distributed training)
pip install deepspeed

# Install development tools (optional)
pip install black isort flake8 pytest
```

### Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check Flash Attention
python -c "import flash_attn; print('Flash Attention OK')"

# Check Transformers
python -c "from transformers import Qwen2VLForConditionalGeneration; print('Transformers OK')"
```

---

## Configuration

### Directory Setup

```bash
# Create necessary directories
mkdir -p data/PerVA/train
mkdir -p data/PerVA/test
mkdir -p manifests/PerVA
mkdir -p outputs/PerVA
mkdir -p results/PerVA
mkdir -p share_models
mkdir -p debug_files
```

### Environment Variables

Create `.env` file:

```bash
# Paths
export DATA_ROOT="data"
export MANIFEST_ROOT="manifests"
export OUTPUT_ROOT="outputs"
export MODEL_CACHE="/path/to/model/cache"

# Training
export LISTENER_URL="http://gpu-node-01:9000/batch_score"
export LISTENER_TIMEOUT=60
export LORA_RANK=64
export LORA_ALPHA=128

# Logging
export DEBUG_MODE=true
export WANDB_PROJECT="GRPO_Lewis_Qwen_2VL"
export WANDB_MODE="offline"

# Distributed Training
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="INFO"
```

### Model Paths

Edit model paths in `src/inference_utils/common.py`:

```python
MODEL_PATHS = {
    "original_2b": "Qwen/Qwen2-VL-2B-Instruct",
    "original_7b": "Qwen/Qwen2-VL-7B-Instruct",
    "lora_7b_grpo": "share_models/Qwen2.5-VL-7B_GRPO_lewis_{dataset}_seed_{seed}_r64_a128",
}
```

---

## Documentation Index

### Core Documentation

| Document | Description | Location |
|----------|-------------|----------|
| **Main README** | This file - project overview | `README.md` |
| **Training Guide** | Comprehensive training pipeline | `train_src/README.md` ⭐ |
| **Inference Guide** | Evaluation tasks & workflows | `src/README.md` |
| **Data Preparation** | Catalog & dataset creation | `src/data_prepare/README.md` |
| **Description Processing** | Evaluation & refinement | `src/eval_utils/description_processing/README.md` |

### Technical Documentation

| Document | Description | Location |
|----------|-------------|----------|
| **Refactoring Summary** | Description processing modularization | `REFACTORING_SUMMARY.md` |
| **Import Fix Summary** | PEP 8 import order compliance | `IMPORT_FIX_SUMMARY.md` |

### Quick Reference

- **Start Training**: See [Training README](train_src/README.md)
- **Run Evaluation**: See [Inference README](src/README.md)
- **Prepare Data**: See [Data Prep README](src/data_prepare/README.md)
- **Process Descriptions**: See [Description Processing README](src/eval_utils/description_processing/README.md)

---

## Recent Updates

### Latest Changes (January 2026)

✅ **Code Quality Improvements**
- Fixed import order in `train_src/open_r1/` to comply with PEP 8
- Properly ordered: stdlib → third-party → local imports
- Fixed `sys.path.insert` placement before dependent imports
- Split multiple imports on single line

✅ **Documentation Overhaul**
- Created comprehensive training pipeline README
- Added description processing module documentation
- Created main project README (this file)
- Added import fix and refactoring summaries

✅ **Module Refactoring**
- Refactored `eval_with_qwen.py` into modular structure
- Separated evaluation and refinement logic
- Created reusable components (prompts, shared utilities)
- Improved code maintainability and testability

### Migration Notes

If you're updating from an older version:

1. **Import paths changed** for description processing:
   ```python
   # Old
   from src.eval_utils.eval_with_qwen import evaluate

   # New
   from src.eval_utils.description_processing.evaluator import evaluate
   ```

2. **Training scripts** now use properly ordered imports - no functional changes

3. **All CLIs remain unchanged** - scripts work as before

---

## Common Workflows

### Workflow 1: Train from Scratch

```bash
# 1. Prepare data
python src/data_prepare/01_build_image_catalog.py \
    --data_root data/PerVA --out manifests/PerVA/catalog.json

# 2. Start listener service
CUDA_VISIBLE_DEVICES=0 python train_src/open_r1/listener_service.py \
    --host $(hostname -s) --port 9000

# 3. Train speaker (in another terminal)
./scripts/run_speaker_training.sh $(hostname -s) 9000 4 PerVA 23

# 4. Generate descriptions with trained model
python src/generate_descriptions.py \
    --data_name PerVA --model_type lora_7b_grpo --seed 23

# 5. Evaluate
python src/personalize.py \
    --data_name PerVA --model_type lora_7b_grpo --seed 23
```

### Workflow 2: Evaluate Existing Model

```bash
# 1. Generate descriptions
python src/generate_descriptions.py \
    --data_name PerVA --model_type original_7b --seed 23

# 2. Run all evaluation tasks
python src/personalize.py --data_name PerVA --model_type original_7b --seed 23
python src/recognition.py --data_name PerVA --model_type original_7b --seed 23
python src/vqa.py --data_name YoLLaVA --model_type original_7b --seed 23
```

### Workflow 3: Process Descriptions

```bash
# 1. Evaluate for state/location attributes
python src/eval_utils/eval_with_qwen.py \
    --input descriptions.json --out results/ --refine none

# 2. Refine (remove state attributes)
python src/eval_utils/eval_with_qwen.py \
    --input descriptions.json --out results/ --refine state

# 3. Refine (remove both state and location)
python src/eval_utils/eval_with_qwen.py \
    --input descriptions.json --out results/ --refine location_and_state
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution:
- Reduce batch size: --per_device_train_batch_size 1
- Increase gradient accumulation: --gradient_accumulation_steps 32
- Enable DeepSpeed ZeRO-3 offloading
- Reduce max_pixels: --max_pixels 200704
```

**2. Listener Connection Fails**
```
Solution:
- Check listener service is running: curl http://<host>:<port>/
- Verify NO_PROXY settings: export NO_PROXY="<host>,localhost"
- Check firewall rules
```

**3. Import Errors**
```
Solution:
- Ensure sys.path.insert is at top of file
- Run from project root: cd Lewis_Game
- Check PYTHONPATH: export PYTHONPATH="${PYTHONPATH}:."
```

**4. Model Loading Fails**
```
Solution:
- Check model cache: echo $MODEL_CACHE
- Verify model exists: ls ~/.cache/huggingface/hub/
- Try manual download: huggingface-cli download Qwen/Qwen2-VL-7B-Instruct
```

### Debug Mode

Enable comprehensive logging:

```bash
export DEBUG_MODE=true
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Check debug logs
tail -f debug_files/debug_speaker_0_job_*.txt
```

---

## Performance Tips

### Training Optimization

1. **Use Flash Attention 2** (2-4x speedup)
   ```bash
   --attn_implementation flash_attention_2
   ```

2. **Tune Batch Sizes** (effective batch = per_device × accumulation × gpus)
   ```bash
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 16 \
   # Effective batch = 1 × 16 × 2 = 32
   ```

3. **Enable Gradient Checkpointing** (trade compute for memory)
   ```bash
   --gradient_checkpointing true
   ```

4. **Optimize LoRA Settings**
   ```bash
   # Faster but lower quality
   --lo_rank 32 --lo_alpha 64

   # Slower but higher quality
   --lo_rank 128 --lo_alpha 256
   ```

### Inference Optimization

1. **Batch Processing**
   ```bash
   --batch_size 8  # Increase for faster throughput
   ```

2. **Reduce Generation Length**
   ```bash
   --max_new_tokens 64  # For shorter descriptions
   ```

3. **Use Smaller Models for Testing**
   ```bash
   --model_type original_2b  # Faster than 7B
   ```

---

## Contributing

### Code Style

This project follows:
- **PEP 8** for Python style
- **Black** for code formatting
- **isort** for import sorting
- **Type hints** where applicable

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Import Order

Follow this structure (see [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md)):

```python
# Standard library
import os
import sys
from typing import Optional

# Third-party
import torch
from transformers import AutoModel

# Local
from inference_utils.common import setup_model
```

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{lewis_game_grpo_2024,
  title={Training Vision-Language Models for Personalized Concept Understanding with GRPO},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@domain.com

---

## Acknowledgments

- **Qwen Team** for the vision-language models
- **DeepSpeed** for distributed training infrastructure
- **HuggingFace** for transformers and datasets
- **Contributors** to this project

---

## Additional Resources

### External Links

- [Qwen2-VL Documentation](https://huggingface.co/Qwen)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Related Projects

- [GRPO Implementation](https://github.com/huggingface/trl)
- [Vision-Language Benchmarks](https://github.com/...)

---

**Last Updated**: January 27, 2026

For the most up-to-date information, always refer to the specific README files in each subdirectory.
