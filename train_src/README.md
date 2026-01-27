# Train Source - Lewis Game Training Pipeline

This directory contains the training infrastructure for the Lewis Game project using GRPO (Generalized Reward Policy Optimization) with Vision-Language Models (Qwen2-VL and Qwen2.5-VL).

## Directory Structure

```
train_src/
├── open_r1/
│   ├── __init__.py
│   ├── dist_helpers.py           # Distributed training utilities
│   ├── listener_service.py       # Listener model service (GPU inference)
│   ├── logger.py                 # Training logging utilities
│   ├── speaker_service.py        # Speaker model service (GPU inference)
│   ├── train_listener_dist.py   # Distributed training for listener
│   ├── train_speaker_dist.py    # Distributed training for speaker
│   └── trainer/                  # Custom GRPO trainer implementations
└── README.md                     # This file
```

## Overview

The Lewis Game training pipeline implements a two-agent communication game:
- **Speaker**: Generates descriptions of images
- **Listener**: Selects the correct image based on descriptions

Both agents are trained using GRPO with custom reward functions.

## Quick Start

### 1. Start the Listener Service

First, start the listener service on a GPU:

```bash
HOSTNAME=$(hostname -s)
PORT=9000
echo "Starting listener on ${HOSTNAME}:${PORT}"

# Run listener_service (ensure it binds to HOSTNAME or 0.0.0.0)
CUDA_VISIBLE_DEVICES=0 python train_src/open_r1/listener_service.py \
    --host ${HOSTNAME} \
    --port ${PORT}
```

**What this does:**
- Loads the Qwen2-VL listener model on GPU 0
- Starts a FastAPI server on the specified host and port
- Provides `/score` and `/batch_score` endpoints for inference
- Uses a semaphore to control concurrent inference requests

### 2. Run Distributed Training

In a separate terminal, run the training script:

```bash
./run_speaker_training.sh <LISTENER_HOST> [LISTENER_PORT] [EPOCHS] [DATASET] [SEED]
```

**Example:**
```bash
./run_speaker_training.sh gpu-node-01 9000 4 YoLLaVA 23
```

**Parameters:**
- `LISTENER_HOST`: Hostname where listener service is running (required)
- `LISTENER_PORT`: Port for listener service (default: 9000)
- `EPOCHS`: Number of training epochs (default: 4)
- `DATASET`: Dataset name (default: YoLLaVA)
- `SEED`: Random seed (default: 23)

## Components

### 1. Listener Service (`listener_service.py`)

**Purpose:** Provides inference endpoints for the listener model during training.

**Key Features:**
- FastAPI-based REST service
- Batch processing support
- Configurable concurrency control
- Memory-efficient inference with cleanup

**Endpoints:**
- `POST /score`: Score a single (candidate_paths, question) pair
- `POST /batch_score`: Score multiple requests in one call (optimized)

**Configuration (via environment variables):**
```bash
INFERENCE_CONCURRENCY=1          # Max concurrent inference calls
INFERENCE_ACQUIRE_TIMEOUT=300    # Timeout for acquiring semaphore (seconds)
LISTENER_BATCH_SIZE=5            # Per-model-call inner batch size
```

**Example Request:**
```python
{
    "batch": [
        {
            "candidate_paths": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
            "question": "Does the description match the dog in the image?",
            "topk": 1
        }
    ]
}
```

**Example Response:**
```python
{
    "results": [
        {
            "yes_probabilities": [0.95, 0.23],
            "predicted_index": 0
        }
    ],
    "took": 0.234
}
```

### 2. Speaker Service (`speaker_service.py`)

**Purpose:** Provides inference endpoints for the speaker model.

**Key Features:**
- Generates image descriptions with thinking process
- Parses structured output (coarse + detailed descriptions)
- Batch processing with memory management

**Endpoints:**
- `POST /describe`: Describe a single image
- `POST /batch_describe`: Describe multiple images in batch

**Output Format:**
The speaker generates structured descriptions:
```
<thinking>reasoning process here</thinking>
<coarse>A photo of a golden retriever</coarse>
<detailed>The dog has fluffy golden fur, a black nose, and is wearing a red collar.</detailed>
```

### 3. Distributed Training Scripts

#### `train_speaker_dist.py`

Trains the speaker model using GRPO with distributed training support.

**Key Features:**
- Multi-GPU training with DeepSpeed
- LoRA fine-tuning support
- Custom reward functions (accuracy, format, length)
- Integration with listener service for reward calculation

**Reward Functions:**

1. **Accuracy Reward** (`accuracy_reward`)
   - Checks if listener correctly identifies target image
   - Calls listener service via HTTP
   - Reward: 1.0 if correct, 0.0 otherwise
   - Uses distributed caching to avoid redundant API calls

2. **Format Reward** (`format_reward`)
   - Checks if output contains proper XML tags
   - Required: `<thinking>`, `<coarse>`, `<detailed>`
   - Reward: 1.0 if all tags present and closed, 0.0 otherwise

3. **Length Reward** (`length_reward`)
   - Encourages concise descriptions (single sentence)
   - Reward: 0.1 if exactly one sentence, 0.0 otherwise

**Configuration:**
```bash
# Model settings
--model_name_or_path "Qwen/Qwen2-VL-7B-Instruct"
--attn_implementation "flash_attention_2"
--max_pixels 401408

# Training settings
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
--num_train_epochs 4
--num_generations 4

# LoRA settings
--use_peft true
--lo_rank 64
--lo_alpha 128
--lo_dropout 0.001

# DeepSpeed
--deepspeed zero3.json
```

#### `train_listener_dist.py`

Similar structure to speaker training but for the listener model.

### 4. Distributed Helpers (`dist_helpers.py`)

**Purpose:** Utility functions for distributed training.

**Key Functions:**
- `get_world_info()`: Get rank and world size
- `dist_all_gather_object_fallback()`: Gather objects across ranks (with fallback)
- `dist_broadcast_object_fallback()`: Broadcast objects from rank 0 (with fallback)

**Features:**
- Graceful fallback to non-distributed mode
- Compatible with older PyTorch versions
- Handles edge cases (uninitialized dist, NCCL issues)

### 5. Logger (`logger.py`)

**Purpose:** Logging utilities for predictions and metrics.

**Features:**
- JSON-based logging
- WandB integration
- Prediction tracking with timestamps

## Training Pipeline

### Complete Workflow

```
┌─────────────────────┐
│  1. Start Listener  │
│     Service         │
│  (GPU 0, Port 9000) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Load Dataset    │
│  (Lewis Game pairs) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Initialize      │
│  Speaker Model      │
│  (Multi-GPU)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  4. Training Loop (GRPO)            │
│  ┌───────────────────────────────┐  │
│  │ a. Generate descriptions      │  │
│  │    (thinking + coarse +       │  │
│  │     detailed)                 │  │
│  └───────────────┬───────────────┘  │
│                  ▼                   │
│  ┌───────────────────────────────┐  │
│  │ b. Call Listener Service      │  │
│  │    (HTTP request)             │  │
│  └───────────────┬───────────────┘  │
│                  ▼                   │
│  ┌───────────────────────────────┐  │
│  │ c. Calculate Rewards          │  │
│  │    - Accuracy (listener)      │  │
│  │    - Format (XML tags)        │  │
│  │    - Length (conciseness)     │  │
│  └───────────────┬───────────────┘  │
│                  ▼                   │
│  ┌───────────────────────────────┐  │
│  │ d. Update Policy (GRPO)       │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  5. Save Model      │
│  (LoRA adapters)    │
└─────────────────────┘
```

## Configuration

### Environment Variables

**Listener Service:**
```bash
export LISTENER_URL="http://gpu-node-01:9000/batch_score"
export LISTENER_TIMEOUT=60
export LISTENER_CHUNK_SIZE=4
export LISTENER_CHUNK_DELAY=0.05
export LISTENER_MAX_RETRIES=3
export LISTENER_BACKOFF_FACTOR=1.0
```

**Training:**
```bash
export DEBUG_MODE=true
export LOG_PATH="share_models/model_name/lewis_job123.json"
export WANDB_DIR="/path/to/wandb"
export WANDB_MODE="offline"
export WANDB_PROJECT="GRPO_Lewis_Qwen_2VL"
```

**Network:**
```bash
export NO_PROXY="gpu-node-01,localhost,127.0.0.1"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="INFO"
```

### LoRA Configuration

**Recommended Settings:**

| Model Size | Rank | Alpha | Dropout | Batch Size | Grad Accumulation |
|------------|------|-------|---------|------------|-------------------|
| 2B         | 64   | 128   | 0.001   | 1          | 8                 |
| 7B         | 64   | 128   | 0.001   | 1          | 16                |
| 14B+       | 128  | 256   | 0.001   | 1          | 32                |

### DeepSpeed Configuration

Example `zero3.json`:
```json
{
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-7,
            "betas": [0.9, 0.95],
            "weight_decay": 0.1
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-7,
            "warmup_num_steps": 100
        }
    },
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}
```

## Example Training Script

Here's the complete training script (`run_speaker_training.sh`):

```bash
#!/bin/bash
set -euo pipefail

# ===========================
# CONFIGURATION
# ===========================

# Usage check
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 LISTENER_HOST [LISTENER_PORT] [EPOCHS] [DATASET] [SEED]"
  exit 2
fi

# Activate environment
source /path/to/your/env/bin/activate

# Unset proxies for internal network
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Parse arguments
LISTENER_HOST="$1"
LISTENER_PORT="${2:-9000}"
EPOCHS="${3:-4}"
DATASET="${4:-YoLLaVA}"
SEED="${5:-23}"

# LoRA configuration
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

# ===========================
# NETWORK & ENVIRONMENT
# ===========================

export NO_PROXY="${LISTENER_HOST},localhost,127.0.0.1"
export no_proxy="${NO_PROXY}"
export LISTENER_URL="http://${LISTENER_HOST}:${LISTENER_PORT}/batch_score"
export LISTENER_TIMEOUT="${LISTENER_TIMEOUT:-60}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"

# ===========================
# PATHS & MODEL CONFIG
# ===========================

BASE_MODEL_NAME="Qwen2.5-VL-7B"
CONFIG_SUFFIX="r${LORA_RANK}_a${LORA_ALPHA}"

DATA_PATH="share_data/PerVA_all_test_subset_seed_${SEED}_K_3_subset_30"
CKPT_PATH="/path/to/Qwen2-VL-7B-Instruct/"
DEEPSPEED_CONFIG="src/virft/local_scripts/zero3.json"

# Verify paths exist
if [ ! -d "${CKPT_PATH}" ]; then
  echo "ERROR: CKPT_PATH does not exist: ${CKPT_PATH}"
  exit 3
fi

# Output directory
TS="$(date +%Y%m%d-%H%M%S)"
SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
SAVE_PATH="share_models/${BASE_MODEL_NAME}_GRPO_lewis_${DATASET}_seed_${SEED}_${CONFIG_SUFFIX}"
mkdir -p "${SAVE_PATH}"

# Logging & wandb
export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${SAVE_PATH}/lewis_${SLURM_JOB_ID}.json"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-GRPO_Lewis_Qwen_2VL}"
RUN_NAME="${BASE_MODEL_NAME}-GRPO-${CONFIG_SUFFIX}_${TS}"

echo "--- CONFIGURATION ---"
echo "Listener: ${LISTENER_HOST}:${LISTENER_PORT}"
echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA}"
echo "Data: ${DATA_PATH}"
echo "Model: ${CKPT_PATH}"
echo "Save: ${SAVE_PATH}"
echo "Run: ${RUN_NAME}"
echo "---------------------"

# ===========================
# TRAINING
# ===========================

torchrun --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=12345 \
  train_src/open_r1/train_speaker_dist.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --output_dir "${SAVE_PATH}" \
  --model_name_or_path "${CKPT_PATH}" \
  --dataset_name "${DATA_PATH}" \
  --max_prompt_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --logging_steps 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --attn_implementation flash_attention_2 \
  --max_pixels 401408 \
  --num_train_epochs "${EPOCHS}" \
  --report_to wandb \
  --run_name "${RUN_NAME}" \
  --save_only_model true \
  --num_generations 4 \
  --use_peft true \
  --lo_rank "${LORA_RANK}" \
  --lo_alpha "${LORA_ALPHA}" \
  --lo_dropout 0.001

echo "Training finished at $(date)"

# ===========================
# CLEANUP
# ===========================

# Remove intermediate checkpoints (keep only final)
shopt -s nullglob
for d in "${SAVE_PATH}"/checkpoint-*; do
    if [ -d "$d" ]; then
        echo "Removing $d"
        rm -rf -- "$d"
    fi
done
shopt -u nullglob

echo "Done!"
```

## Debugging

### Enable Debug Logging

```bash
export DEBUG_MODE=true
```

Debug logs are written to:
- `debug_files/debug_speaker_{rank}_{job_id}.txt`
- `debug_files/debug_one_sentence_rank{rank}.txt`

### Common Issues

**1. Listener connection fails:**
```
[WARN] listener chunk 0:4 request failed
```
**Solution:**
- Check listener service is running: `curl http://<host>:<port>/health`
- Verify NO_PROXY settings
- Check firewall rules

**2. Out of memory:**
```
CUDA out of memory
```
**Solution:**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable DeepSpeed ZeRO stage 3 offloading
- Reduce `max_pixels`

**3. NCCL timeout:**
```
NCCL timeout in collective
```
**Solution:**
- Set `export NCCL_ASYNC_ERROR_HANDLING=1`
- Increase timeout: `export NCCL_IB_TIMEOUT=50`
- Check network connectivity between nodes

**4. Import errors:**
```
ModuleNotFoundError: No module named 'open_r1'
```
**Solution:**
- Ensure `sys.path.insert(0, 'src/virft/src/')` is before imports
- Run from project root directory

## Performance Tips

### 1. Optimize Batch Sizes

**For Listener Service:**
```bash
export LISTENER_BATCH_SIZE=8  # Higher = better throughput
export INFERENCE_CONCURRENCY=2  # Allow concurrent batches
```

**For Training:**
```bash
# Effective batch size = per_device * accumulation * num_gpus
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
# Effective batch size = 1 * 16 * 2 = 32
```

### 2. Use Flash Attention 2

Always use Flash Attention for Qwen models:
```bash
--attn_implementation flash_attention_2
```

### 3. Tune LoRA Settings

**For faster training:**
- Lower rank: `--lo_rank 32`
- Fewer trainable parameters

**For better quality:**
- Higher rank: `--lo_rank 128`
- Higher alpha: `--lo_alpha 256`

### 4. Enable Gradient Checkpointing

Trades compute for memory:
```bash
--gradient_checkpointing true
```

## Monitoring

### WandB Integration

Training metrics are logged to WandB:
- Loss curves
- Reward values (accuracy, format, length)
- Generation examples
- Learning rate schedule

### Local Logs

Check logs in:
```bash
tail -f ${SAVE_PATH}/train.log
```

## Output

### Model Checkpoints

Saved in `${SAVE_PATH}`:
```
share_models/Qwen2.5-VL-7B_GRPO_lewis_YoLLaVA_seed_23_r64_a128/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin          # LoRA weights
├── trainer_state.json         # Training state
├── training_args.bin          # Training arguments
└── lewis_<job_id>.json       # Prediction logs
```

### Loading Trained Model

```python
from peft import PeftModel, PeftConfig
from transformers import Qwen2VLForConditionalGeneration

# Load config
config = PeftConfig.from_pretrained("path/to/checkpoint")

# Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "path/to/checkpoint")
model.eval()
```

## Requirements

### Python Packages

```
torch>=2.0.0
transformers>=4.40.0
deepspeed>=0.12.0
peft>=0.7.0
trl>=0.7.0
datasets>=2.14.0
fastapi>=0.100.0
uvicorn>=0.23.0
qwen-vl-utils
flash-attn>=2.3.0
wandb
pillow
requests
```

### Hardware

**Minimum:**
- 2x A100 40GB (or equivalent)
- 256GB RAM
- Fast network for distributed training

**Recommended:**
- 4x A100 80GB
- 512GB RAM
- InfiniBand network

## Citation

If you use this training pipeline, please cite:

```bibtex
@article{lewis_game_grpo_2024,
  title={Training Vision-Language Models for Lewis Game with GRPO},
  author={Your Name},
  year={2024}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please contact [your-email@domain.com]
