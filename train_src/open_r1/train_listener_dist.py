# Standard library
import json
import os
import re
import string
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Modify path BEFORE other third-party imports
sys.path.insert(0, 'train_src/')

# Third-party
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import DatasetDict, Dataset, load_dataset, load_from_disk
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# Local/project-specific
# from listener import Listener
from math_verify import parse, verify
from open_r1.logger import PredictionLogger
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer

logger = PredictionLogger(log_path=os.getenv("LOG_PATH"))

def extract_answer_content(text):
    """
    Extract the content between <answer> tags from the given text.
    
    Args:
        text (str): Input text containing <answer> tags
        
    Returns:
        str: Content between answer tags, or empty string if not found
    """
    # Pattern to match content between <answer> and </answer> tags
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return ""

def extract_yes_no_answer(text: str) -> str:
    """
    Extract yes/no answer from JSON or text.
    Returns 'yes', 'no', or None.
    """
    # First try JSON extraction
    answer = extract_reasoning_answer_term(text)
    
    if answer:
        answer = answer.lower().strip()
        # Handle variations
        if answer in ['yes', 'y', 'true', '1']:
            return 'yes'
        elif answer in ['no', 'n', 'false', '0']:
            return 'no'
    
    # Fallback: search for yes/no in text
    text_lower = text.lower()
    if 'yes' in text_lower and 'no' not in text_lower:
        return 'yes'
    elif 'no' in text_lower and 'yes' not in text_lower:
        return 'no'
    
    return None

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

def extract_reasoning_answer_term(text: str) -> str:
    """
    Extracts the value for a given term from the text.
    It first tries to match a quoted value, then an unquoted word.
    """
    patterns = {
        'Answer': r'"Answer":\s*(?:"([^"]+)"|([\w-]+))',
    }
    pattern = patterns.get('Answer')
    if not pattern:
        return None
    match = re.search(pattern, text)
    if match:
        return (match.group(1) or match.group(2)).strip()
    else:
        # Fallback if regex doesn't match.
        parts = text.split('Answer')
        if parts:
            return re.sub(r'[^a-zA-Z0-9\s]', '', parts[-1]).strip()
        return None

def accuracy_reward(completions, solution, task_type=None, **kwargs):
    """Selection task reward - only process selection examples"""
    rewards = []
    selection_count = 0
    correct_count = 0
    for i, (completion, sol, t_type) in enumerate(zip(completions, solution, task_type)):
        if t_type != 'selection':
            # This is a consistency task completion, skip it
            rewards.append(0.0)  # Or you could use None and filter later
            continue
        selection_count += 1
        # Original logic for selection task
        content = completion[0]["content"]
        pred_letter = extract_reasoning_answer_term(content)
        is_correct = (pred_letter == sol)
        if is_correct:
            correct_count += 1
        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)
        # print(f"predicted_option: {pred_letter}, target_option: {sol}")
        if selection_count > 0:
            true_accuracy = correct_count / selection_count
            # print(f"[SELECTION] True accuracy: {correct_count}/{selection_count} = {true_accuracy:.3f} (logged reward will be lower)")
    
        if logger is not None:
            logger.log(reward, content, sol)

        if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = f'debug_files/debug_listener_selection_{os.getenv("LOCAL_RANK", "0")}_job_{os.getenv("SLURM_JOB_ID", "unknown")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"predicted_option: {pred_letter}, target_option: {sol}\n")
                    f.flush()
            except Exception as e:
                if logger is not None:
                    logger.log(f"Logging error: {e}", content, sol)
    return rewards


def consistency_reward(completions, listener_solution, task_type=None, **kwargs):
    """Consistency task reward - only process consistency examples"""
    rewards = []
    consistency_count = 0
    correct_count = 0
    for completion, l_sol, t_type in zip(completions, listener_solution, task_type):
        if t_type != 'consistency':
            # This is a selection task completion, skip it
            rewards.append(0.0)
            continue
        
        # Extract yes/no from JSON
        consistency_count += 1
        content = completion[0]["content"]
        predicted_answer = extract_yes_no_answer(content)
        if predicted_answer:
            predicted_answer = predicted_answer.lower().strip()
        if l_sol:
            l_sol = l_sol.lower().strip()
        is_correct = (predicted_answer == l_sol)
        if is_correct:
            correct_count += 1
        
        reward = 1.0 if is_correct else 0.0
        rewards.append(reward)
        if consistency_count > 0:
            true_accuracy = correct_count / consistency_count
            # print(f"[CONSISTENCY] True accuracy: {correct_count}/{consistency_count} = {true_accuracy:.3f} (logged reward will be lower due to selection tasks)")
        if logger is not None:
            logger.log(reward, content, l_sol)
        if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = f'debug_files/debug_listener_consistency_{os.getenv("LOCAL_RANK", "0")}_job_{os.getenv("SLURM_JOB_ID", "unknown")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"predicted_option: {predicted_answer}, target_option: {l_sol}\n")
                    f.flush()
            except Exception as e:
                if logger is not None:
                    logger.log(f"Logging error: {e}", content, l_sol)
    return rewards

def format_reward(completions, **kwargs):
    """
    Reward = 1.0 if the model output follows the strict JSON format:
      - JSON inside ```json ... ``` fences
      - Contains keys for options + "Reasoning" + "Answer"
      - "Answer" must be one of A–J
    Otherwise reward = 0.0
    """
    rewards = []
    option_letters = list(string.ascii_uppercase[:10])  # A–J

    for completion in completions:
        content = completion[0]["content"]
        reward = 0.0

        # Try to extract fenced JSON
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict):
                    # Must contain Reasoning + Answer
                    if "Reasoning" in obj and "Answer" in obj:
                        ans = str(obj["Answer"]).strip().upper()
                        # if ans in option_letters:
                        reward = 1.0
            except Exception:
                pass

        rewards.append(reward)

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "consistency": consistency_reward,
    
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def count_trainable_parameters_detailed(model: torch.nn.Module):
    """
    Counts and prints the details of trainable parameters in a PyTorch model.

    Args:
        model: A PyTorch nn.Module.
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            print(f"Trainable: {name} - size: {param.size()} - dtype: {param.dtype}")
        else:
            non_trainable_params += num_params
            print(f"Frozen: {name} - size: {param.size()} - dtype: {param.dtype}")

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    return trainable_params

def make_conversation_lewis_game(example):
    return {
        "prompt": [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},  # This will be filled with example["image"]
                    {"type": "text", "text": example["speaker_problem"]},
                ],
            },
        ],
    }

def main(script_args, training_args, model_args, lora_args):
    script_args.reward_funcs = ['consistency', 'format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    from datasets import load_dataset
    dataset_path = script_args.dataset_name
    dataset = DatasetDict.load_from_disk(dataset_path)
    dataset = dataset.map(make_conversation_lewis_game)
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)
    if model_args.use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=lora_args.lo_rank,  # the rank of the LoRA matrices
            lora_alpha=lora_args.lo_alpha, # the weight
            lora_dropout=lora_args.lo_dropout, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules="all-linear", # the name of the layers to add LoRA
            modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
        )
    else:
        peft_config = None
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        train_listener=True
    )
    trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

@dataclass
class LoRAArgs:
    """Local dataclass so HF parser will accept LoRA CLI flags."""
    lo_rank: int = field(default=64, metadata={"help": "LoRA rank"})
    lo_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lo_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig, LoRAArgs))
    script_args, training_args, model_args, lora_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, lora_args)
