# Description Processing Module

This module provides functionality for evaluating and refining object descriptions, specifically focusing on identifying and removing state-specific and location-specific attributes.

## Structure

```
description_processing/
├── __init__.py          # Module initialization and exports
├── prompts.py           # Prompt templates for Qwen model
├── shared.py            # Common utilities (dataset, model, inference)
├── evaluator.py         # Evaluation functionality
├── refiner.py           # Refinement functionality
└── README.md            # This file
```

## Components

### 1. `prompts.py`
Contains all prompt templates used by the Qwen model:
- `prefix` and `suffix`: Evaluation prompts for detecting state/location attributes
- `prefix_state`: Refinement prompt for removing state attributes
- `prefix_location`: Refinement prompt for removing location attributes
- `prefix_location_and_state`: Refinement prompt for removing both

### 2. `shared.py`
Common utilities used by both evaluator and refiner:
- `JsonDescriptionsDataset`: PyTorch dataset for loading description JSON files
- `collate_batch()`: Batch collation function
- `setup_model()`: Load Qwen model and tokenizer
- `infer_batch()`: Perform batch inference with the model

### 3. `evaluator.py`
Evaluation functionality:
- `evaluate()`: Main evaluation function that processes all descriptions
- `aggregate_stats()`: Compute aggregate statistics from results

### 4. `refiner.py`
Refinement functionality:
- `refine()`: Refine descriptions by removing specified attributes
- `refine_descriptions()`: Main entry point for refinement

## Usage

### Evaluation

Evaluate descriptions to identify state and location attributes:

```python
from src.eval_utils.description_processing.evaluator import evaluate
from src.eval_utils.description_processing.prompts import prefix, suffix
from src.eval_utils.description_processing.shared import setup_model

# Setup
model, tokenizer = setup_model("Qwen/Qwen3-8B")

# Evaluate
evaluate(args, model, tokenizer, prefix, suffix, output_path)
```

Output format:
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

### Refinement

Refine descriptions by removing unwanted attributes:

```python
from src.eval_utils.description_processing.refiner import refine_descriptions
from src.eval_utils.description_processing.prompts import prefix_state, suffix
from src.eval_utils.description_processing.shared import setup_model

# Setup
model, tokenizer = setup_model("Qwen/Qwen3-8B")

# Load evaluation results
with open('eval_results.json', 'r') as f:
    batch_out = json.load(f)

# Refine (remove state attributes)
refined = refine_descriptions(args, model, tokenizer, prefix_state, suffix, batch_out)
```

### Command Line Usage

The main script `eval_with_qwen.py` provides a convenient CLI:

```bash
# Evaluation only
python src/eval_utils/eval_with_qwen.py \
    --input data/descriptions.json \
    --out results/ \
    --batch-size 64

# Refinement (remove state attributes)
python src/eval_utils/eval_with_qwen.py \
    --input data/descriptions.json \
    --refine state \
    --out results/ \
    --batch-size 64

# Refinement (remove both state and location)
python src/eval_utils/eval_with_qwen.py \
    --input data/descriptions.json \
    --refine location_and_state \
    --out results/ \
    --batch-size 64
```

## Key Definitions

### State Attributes
Transient or changeable conditions/actions:
- Examples: running, open, folded, wagging, standing, sitting, wearing
- These describe temporary states rather than permanent characteristics

### Location Attributes
Unrelated scene/background or spatial context:
- Examples: "on a table", "surrounded by trees", "in a kitchen", "in the garden"
- These describe where the object is rather than what it is

## Benefits of Modular Structure

1. **Separation of Concerns**: Evaluation and refinement logic are cleanly separated
2. **Reusability**: Components can be imported and used independently
3. **Testability**: Each module can be tested in isolation
4. **Maintainability**: Changes to one component don't affect others
5. **Clarity**: Clear responsibility for each file

## Migration from Old Code

The old `eval_with_qwen.py` file contained all functionality in one file. The new structure:
- Moved prompt templates to `prompts.py`
- Moved shared utilities to `shared.py`
- Moved evaluation logic to `evaluator.py`
- Moved refinement logic to `refiner.py`
- Updated main script to import from new modules

All functionality remains the same, just better organized!
