# Quick Reference Guide

Fast lookup for common commands and configurations.

## 🚀 Quick Start Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Create directories
mkdir -p data/{PerVA,YoLLaVA}/{train,test} manifests outputs results share_models
```

### Training (Most Common)
```bash
# Terminal 1: Start listener
CUDA_VISIBLE_DEVICES=0 python train_src/open_r1/listener_service.py --host $(hostname -s) --port 9000

# Terminal 2: Train speaker
./scripts/run_speaker_training.sh $(hostname -s) 9000 4 PerVA 23
```

### Evaluation (Most Common)
```bash
# Generate descriptions
python src/generate_descriptions.py --data_name PerVA --model_type original_7b --seed 23

# Run personalization
python src/personalize.py --data_name PerVA --model_type original_7b --db_type original_7b --k_retrieval 3 --seed 23
```

---

## 📁 File Locations

### Data
```
data/PerVA/train/<category>/<concept>/      # Training images
data/PerVA/test/<category>/<concept>/       # Test images
manifests/PerVA/catalog.json                # Image catalog
```

### Models
```
~/.cache/huggingface/hub/                   # Base models
share_models/Qwen2.5-VL-7B_GRPO_*/         # Trained LoRA models
```

### Outputs
```
outputs/PerVA/<category>/seed_*/            # Generated descriptions
results/PerVA/<category>/seed_*/            # Evaluation results
debug_files/debug_speaker_*.txt             # Training debug logs
```

---

## ⚙️ Common Configurations

### LoRA Settings by Model Size

| Model | Rank | Alpha | Dropout | Batch Size | Grad Accum |
|-------|------|-------|---------|------------|------------|
| 2B    | 64   | 128   | 0.001   | 1          | 8          |
| 7B    | 64   | 128   | 0.001   | 1          | 16         |
| 14B+  | 128  | 256   | 0.001   | 1          | 32         |

### Environment Variables (Training)

```bash
# Essential
export LISTENER_URL="http://$(hostname -s):9000/batch_score"
export NO_PROXY="$(hostname -s),localhost,127.0.0.1"

# LoRA
export LORA_RANK=64
export LORA_ALPHA=128

# Optimization
export LISTENER_CHUNK_SIZE=4
export LISTENER_TIMEOUT=60

# Logging
export DEBUG_MODE=true
export WANDB_PROJECT="GRPO_Lewis_Qwen_2VL"
export WANDB_MODE="offline"
```

### Model Types

```python
# Built-in
"original_2b"      # Qwen/Qwen2-VL-2B-Instruct
"original_7b"      # Qwen/Qwen2-VL-7B-Instruct

# LoRA (auto-constructed path)
"lora_7b_grpo"     # share_models/Qwen2.5-VL-7B_GRPO_lewis_{dataset}_seed_{seed}_r64_a128

# Custom (direct path)
"/path/to/model"   # Custom model or checkpoint
```

---

## 🔧 Debugging Commands

### Check Services
```bash
# Listener health check
curl http://$(hostname -s):9000/

# Check GPU usage
nvidia-smi

# Check processes
ps aux | grep listener_service
ps aux | grep train_speaker
```

### View Logs
```bash
# Training logs (real-time)
tail -f share_models/*/train.log

# Debug logs
tail -f debug_files/debug_speaker_0_job_*.txt

# WandB sync
ls -lh wandb/latest-run/
```

### Test Imports
```bash
# Test all imports
python -c "from src.inference_utils.common import *; print('OK')"
python -c "from train_src.open_r1.dist_helpers import *; print('OK')"
python -c "import flash_attn; print('Flash Attention OK')"
```

---

## 📊 Result Files

### Description Generation Output
```json
{
  "obj_001": {
    "name": "golden_retriever_1",
    "category": "dog",
    "general": ["The dog has golden fur"],
    "distinguishing features": ["wearing a red collar"],
    "train_paths": ["data/PerVA/train/dog/golden_retriever_1/img1.jpg"],
    "test_paths": ["data/PerVA/test/dog/golden_retriever_1/img1.jpg"]
  }
}
```

### Evaluation Results Output
```json
{
  "config": {
    "data_name": "PerVA",
    "model_type": "original_7b",
    "seed": 23
  },
  "metrics": {
    "accuracy": 0.85,
    "correct": 170,
    "total": 200
  },
  "results": [...]
}
```

---

## 🎯 Common Task Patterns

### Pattern 1: Full Training Pipeline
```bash
# 1. Prepare data
python src/data_prepare/01_build_image_catalog.py --data_root data/PerVA --out manifests/PerVA/catalog.json

# 2. Start listener
CUDA_VISIBLE_DEVICES=0 python train_src/open_r1/listener_service.py --host $(hostname -s) --port 9000 &

# 3. Train (4 epochs, seed 23)
./scripts/run_speaker_training.sh $(hostname -s) 9000 4 PerVA 23

# 4. Kill listener when done
pkill -f listener_service
```

### Pattern 2: Quick Evaluation
```bash
# Generate + evaluate in sequence
python src/generate_descriptions.py --data_name PerVA --model_type original_7b --seed 23 && \
python src/personalize.py --data_name PerVA --model_type original_7b --db_type original_7b --seed 23
```

### Pattern 3: Batch Evaluation (All Tasks)
```bash
MODEL="original_7b"
SEED=23

# Run all three tasks
for TASK in personalize recognition vqa; do
    python src/${TASK}.py \
        --data_name PerVA \
        --model_type ${MODEL} \
        --db_type ${MODEL} \
        --seed ${SEED}
done
```

### Pattern 4: Description Processing
```bash
DESC_FILE="outputs/PerVA/all/seed_23/descriptions_original_7b.json"

# Evaluate
python src/eval_utils/eval_with_qwen.py --input ${DESC_FILE} --out results/ --refine none

# Refine (remove state)
python src/eval_utils/eval_with_qwen.py --input ${DESC_FILE} --out results/ --refine state

# Refine (remove both)
python src/eval_utils/eval_with_qwen.py --input ${DESC_FILE} --out results/ --refine location_and_state
```

---

## 🐛 Quick Fixes

### Out of Memory
```bash
# Reduce batch size
--per_device_train_batch_size 1 --gradient_accumulation_steps 32

# Reduce image resolution
--max_pixels 200704

# Enable offloading (DeepSpeed)
# Edit zero3.json: "offload_optimizer": {"device": "cpu"}
```

### Listener Not Responding
```bash
# Check if running
pgrep -af listener_service

# Restart
pkill -f listener_service
CUDA_VISIBLE_DEVICES=0 python train_src/open_r1/listener_service.py --host $(hostname -s) --port 9000 &

# Test connection
curl -X POST http://$(hostname -s):9000/score -H "Content-Type: application/json" -d '{"candidate_paths":["test.jpg"],"question":"test"}'
```

### Import Errors
```bash
# From project root
cd /path/to/Lewis_Game

# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify
python -c "import sys; print('\n'.join(sys.path))"
```

### Model Not Found
```bash
# Check cache
ls ~/.cache/huggingface/hub/ | grep Qwen

# Download manually
huggingface-cli login  # If using gated models
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct

# Or set cache location
export HF_HOME=/path/to/cache
```

---

## 📖 Documentation Quick Links

| Topic | Location |
|-------|----------|
| **Main Overview** | `README.md` |
| **Training Setup** | `train_src/README.md` |
| **Evaluation Tasks** | `src/README.md` |
| **Data Preparation** | `src/data_prepare/README.md` |
| **Description Processing** | `src/eval_utils/description_processing/README.md` |
| **This Guide** | `QUICK_REFERENCE.md` |

---

## 💡 Pro Tips

1. **Use tmux/screen for long runs**
   ```bash
   tmux new -s training
   # Run training commands
   # Detach: Ctrl+B, D
   # Reattach: tmux attach -t training
   ```

2. **Monitor GPU in real-time**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save environment for reproducibility**
   ```bash
   pip freeze > requirements_frozen.txt
   ```

4. **Use absolute paths in scripts**
   ```bash
   PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   cd "${PROJECT_ROOT}"
   ```

5. **Log everything with tee**
   ```bash
   python src/train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
   ```

---

## 🔗 URLs & Endpoints

### Services (Default)
- Listener: `http://127.0.0.1:9000`
- Speaker: `http://127.0.0.1:9001`

### WandB
- Dashboard: `https://wandb.ai/<username>/GRPO_Lewis_Qwen_2VL`

### Model Hubs
- HuggingFace: `https://huggingface.co/Qwen`
- Model Cache: `~/.cache/huggingface/hub/`

---

## 🎓 Learning Resources

### Order of Learning
1. **Start Here**: Main `README.md`
2. **Data Setup**: `src/data_prepare/README.md`
3. **Evaluation**: `src/README.md`
4. **Training**: `train_src/README.md` (comprehensive)
5. **Advanced**: Description processing README

### Key Concepts
- **Lewis Game**: Two-agent communication (speaker generates, listener identifies)
- **GRPO**: Generalized Reward Policy Optimization (RL for language models)
- **LoRA**: Low-Rank Adaptation (parameter-efficient fine-tuning)
- **DeepSpeed**: Distributed training framework (ZeRO stages)
- **Flash Attention**: Memory-efficient attention mechanism

---

**Last Updated**: January 27, 2026

Keep this file handy for quick lookups during development and debugging!
