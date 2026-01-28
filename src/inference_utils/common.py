"""
Common utilities shared across inference tasks.

This module provides:
- set_seed(): Reproducibility helper
- get_model_path(): Centralized model path resolution
- CATEGORY_MAPPINGS: All dataset category mappings
- save_results(): Standardized result saving
- BaseInferenceConfig: Shared configuration dataclass
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

LOG = logging.getLogger(__name__)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Category Mappings
# ============================================================================

# DreamBooth dataset
DREAMBOOTH_CATEGORY_MAP = {
    "backpack": 'bag',
    'backpack_dog': 'bag',
    'bear_plushie': 'toy',
    'berry_bowl': 'household object',
    'can': 'household object',
    'candle': 'household object',
    'cat': 'pet animal',
    'cat2': 'pet animal',
    'clock': 'household object',
    'colorful_sneaker': 'shoe',
    'dog': 'pet animal',
    'dog2': 'pet animal',
    'dog3': 'pet animal',
    'dog5': 'pet animal',
    'dog6': 'pet animal',
    'dog7': 'pet animal',
    'dog8': 'pet animal',
    'duck_toy': 'toy',
    'fancy_boot': 'shoe',
    'grey_sloth_plushie': 'toy',
    'monster_toy': 'toy',
    'pink_sunglasses': 'glasses',
    'poop_emoji': 'toy',
    'rc_car': 'toy',
    'red_cartoon': 'cartoon character',
    'robot_toy': 'toy',
    'shiny_sneaker': 'shoe',
    'teapot': 'household object',
    'vase': 'household object',
    'wolf_plushie': 'toy'
}

# YoLLaVA dataset
YOLLAVA_CATEGORY_MAP = {
    'ciin': 'person',
    'denisdang': 'person',
    'khanhvy': 'person',
    'oong': 'person',
    'phuc-map': 'person',
    'thao': 'person',
    'thuytien': 'person',
    'viruss': 'person',
    'yuheng': 'person',
    'willinvietnam': 'person',
    'chua-thien-mu': 'building',
    'nha-tho-hanoi': 'building',
    'nha-tho-hcm': 'building',
    'thap-but': 'building',
    'thap-cham': 'building',
    'dug': 'cartoon character',
    'fire': 'cartoon character',
    'marie-cat': 'cartoon character',
    'toodles-galore': 'cartoon character',
    'water': 'cartoon character',
    'bo': 'pet animal',
    'butin': 'pet animal',
    'henry': 'pet animal',
    'mam': 'pet animal',
    'mydieu': 'pet animal',
    'shiba-yellow': 'toy',
    'pusheen-cup': 'mug',
    'neurips-cup': 'toy',
    'tokyo-keyboard': 'electronic',
    'cat-cup': 'cup',
    'brown-duck': 'toy',
    'lamb': 'toy',
    'duck-banana': 'toy',
    'shiba-black': 'toy',
    'pig-cup': 'cup',
    'shiba-sleep': 'toy',
    'yellow-duck': 'toy',
    'elephant': 'toy',
    'shiba-gray': 'toy',
    'dragon': 'toy'
}

# MyVLM dataset
MYVLM_CATEGORY_MAP = {
    'asian_doll': 'toy',
    'boy_funko_pop': 'toy',
    'bull': 'figurine',
    'cat_statue': 'figurine',
    'ceramic_head': 'figurine',
    'chicken_bean_bag': 'toy',
    'colorful_teapot': 'tea pot',
    'dangling_child': 'toy',
    'elephant_sphere': 'figurine',
    'elephant_statue': 'figurine',
    'espresso_cup': 'cup',
    'gengar_toy': 'toy',
    'gold_pineapple': 'household object',
    'iverson_funko_pop': 'toy',
    'green_doll': 'toy',
    'maeve_dog': 'pet animal',
    'minion_toy': 'toy',
    'rabbit_toy': 'toy',
    'red_chicken': 'figurine',
    'red_piggy_bank': 'piggy bank',
    'robot_toy': 'toy',
    'running_shoes': 'shoe',
    'sheep_pillow': 'pillow',
    'sheep_plush': 'toy',
    'sheep_toy': 'toy',
    'skulls_mug': 'mug',
    'small_penguin': 'toy',
    'billy_dog': 'pet animal',
    'my_cat': 'pet animal'
}

# PerVA short-to-long category names
PERVA_CATEGORY_MAP = {
    'veg': 'vegetable',
    'decoration': 'decoration object',
    'retail': 'retail object',
    'tro_bag': 'trolley bag',
}

# Combined lookup by dataset name
DATASET_CATEGORY_MAPS = {
    'DreamBooth': DREAMBOOTH_CATEGORY_MAP,
    'YoLLaVA': YOLLAVA_CATEGORY_MAP,
    'MyVLM': MYVLM_CATEGORY_MAP,
    'PerVA': PERVA_CATEGORY_MAP,
}


def get_category_for_concept(concept_name: str, dataset: str) -> str:
    """
    Get the category for a given concept name in a dataset.

    Args:
        concept_name: The concept/object name
        dataset: Dataset name (YoLLaVA, MyVLM, PerVA, DreamBooth)

    Returns:
        Category string, or 'object' as fallback
    """
    category_map = DATASET_CATEGORY_MAPS.get(dataset, {})
    return category_map.get(concept_name, concept_name)


# ============================================================================
# Model Path Resolution
# ============================================================================

# Base paths (can be overridden via environment variables)
DEFAULT_SHARE_MODELS_DIR = os.environ.get(
    "SHARE_MODELS_DIR",
    "./share_models"
)

# Model configurations
MODEL_CONFIGS = {
    # Base models
    'original_2b': {
        'path': "Qwen/Qwen2-VL-2B-Instruct",
        'use_peft': False,
    },
    'original_7b': {
        'path': "Qwen/Qwen2-VL-7B-Instruct",
        'use_peft': False,
    },
    # LoRA-finetuned models (path templates)
    'lora_7b_grpo': {
        'path_template': "{share_models}/Qwen2-VL-7B_GRPO_lewis_PerVA_seed_{seed}_r1024_a64_K_3_subset30",
        'use_peft': True,
    },
    'lora_2b_grpo': {
        'path_template': "{share_models}/Qwen2-VL-2B-Instruct_GRPO_lewis_PerVA_seed_{seed}",
        'use_peft': True,
    },
}


def get_model_config(
    model_type: str,
    dataset: str = None,
    seed: int = None,
    share_models_dir: str = None,
) -> Dict[str, Any]:
    """
    Get model path and configuration for a given model type.

    Args:
        model_type: Model identifier (e.g., 'original_7b', 'lora_7b_grpo')
        dataset: Dataset name (required for LoRA models)
        seed: Random seed (required for LoRA models)
        share_models_dir: Override for shared models directory

    Returns:
        Dict with 'path' and 'use_peft' keys
    """
    if model_type not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")

    config = MODEL_CONFIGS[model_type].copy()
    share_models = share_models_dir or DEFAULT_SHARE_MODELS_DIR

    # Handle template paths for finetuned models
    if 'path_template' in config:
        if dataset is None or seed is None:
            raise ValueError(f"model_type '{model_type}' requires 'dataset' and 'seed' arguments")
        config['path'] = config['path_template'].format(
            share_models=share_models,
            dataset=dataset,
            seed=seed,
        )
        del config['path_template']

    return config


def get_model_path(model_type: str, **kwargs) -> str:
    """Convenience function to get just the model path."""
    return get_model_config(model_type, **kwargs)['path']


def uses_peft(model_type: str) -> bool:
    """Check if a model type uses PEFT/LoRA."""
    if model_type not in MODEL_CONFIGS:
        # Fallback heuristic
        return 'lora' in model_type.lower()
    return MODEL_CONFIGS[model_type].get('use_peft', False)


# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class BaseInferenceConfig:
    """Base configuration for inference tasks."""
    # Dataset settings
    data_name: str = "YoLLaVA"
    catalog_file: str = "main_catalog_seed_23.json"
    category: str = "all"
    concept_name: str = ""

    # Model settings
    model_type: str = "original_7b"
    db_type: str = "original_7b"  # Database/description model type

    # Generation settings
    seed: int = 23
    batch_size: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.7

    # Output settings
    output_dir: str = "results"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_args(cls, args) -> "BaseInferenceConfig":
        """Create config from argparse namespace."""
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__})


def add_common_args(parser) -> None:
    """Add common arguments to an argument parser."""
    parser.add_argument("--data_name", type=str, default="YoLLaVA",
                        help="Dataset name (YoLLaVA, MyVLM, PerVA)")
    parser.add_argument("--catalog_file", type=str, default="main_catalog_seed_23.json",
                        help="Catalog JSON filename")
    parser.add_argument("--category", type=str, default="all",
                        help="Category to process")
    parser.add_argument("--concept_name", type=str, default="",
                        help="Specific concept name (optional)")
    parser.add_argument("--seed", type=int, default=23,
                        help="Random seed")
    parser.add_argument("--db_type", type=str, default="original_7b",
                        help="Database/description model type")
    parser.add_argument("--model_type", type=str, default="original_7b",
                        help="Inference model type")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for results")


# ============================================================================
# Result Saving
# ============================================================================

def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save inference results in a standardized format.

    Args:
        results: List of per-sample results
        metrics: Aggregated metrics (accuracy, etc.)
        config: Configuration used for the run
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": config,
        "metrics": metrics,
        "results": results,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    LOG.info("Saved results to %s", output_path)


def load_database(database_path: Path) -> Dict[str, Any]:
    """Load a database JSON file."""
    if not database_path.exists():
        raise FileNotFoundError(f"Database file not found: {database_path}")

    with database_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_database_path(
    data_name: str,
    category: str,
    seed: int,
    db_type: str,
    base_dir: str = "outputs",
) -> Path:
    """Construct the standard database path."""
    return Path(base_dir) / data_name / category / f"seed_{seed}" / f"database_{db_type}.json"


# ============================================================================
# Device Utilities
# ============================================================================

def get_device() -> torch.device:
    """Get the appropriate torch device."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def clear_cuda_cache() -> None:
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
