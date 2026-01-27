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
sys.path.insert(0, 'src/virft/src/')

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
from listener import Listener
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


# def accuracy_reward(completions, solution, logger=None, **kwargs):
#     # global listener_model
#     contents = [completion[0]["content"] for completion in completions]    
#     rewards = []    
#     for i, (content, sol) in enumerate(zip(contents, solution)):
#         reward = 0.0
#         ground_truth = chr(ord('A') + int(sol))
#         parts = content.split('assistant\n')
#         search_area = parts[-1] if len(parts) > 1 else content
#         content_match = re.search(r'<answer[^>]*>(.*?)</answer>', search_area, re.DOTALL)
#         if not content_match:
#             # Second try: Handle potential whitespace issues
#             content_match = re.search(r'<answer\s*>(.*?)</answer\s*>', content, re.DOTALL)
        
#         if not content_match:
#             # Third try: More flexible pattern
#             content_match = re.search(r'<answer[^>]*>(.*?)</answer[^>]*>', content, re.DOTALL)
        
#         if content_match:
#             student_answer = content_match.group(1).strip()
#         else:
#             answer_start = content.find('<answer>')
#             answer_end = content.find('</answer>')
#             if answer_start != -1 and answer_end != -1:
#                 manual_extract = content[answer_start + 8:answer_end]
#                 # print(f"DEBUG: Manual extraction found: {repr(manual_extract)}")
#                 student_answer = manual_extract.strip()
#             else:
#                 # print(f"DEBUG: Could not find answer tags manually either")
#                 student_answer = content.strip()
#         ground_truth = ground_truth.lower()
#         student_answer = student_answer.lower()
#         reward = 1.0 if ground_truth == student_answer else 0.0
#         rewards.append(reward)
#         # print(f"predicted_option: {student_answer}, target_option: {ground_truth}\n")
#         if logger is not None:
#             logger.log(reward, content, sol)
        
#         if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
#             current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add this import
#             log_path = f'debug_files/debug_listener_{os.getenv("LOCAL_RANK", "0")}_job_{os.getenv("SLURM_JOB_ID", "unknown")}.txt'
#             print(content)
#             try:
#                 with open(log_path, "a", encoding="utf-8") as f:
#                     f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} -------------\n")
#                     f.write(f"content: {content}\n")
#                     f.write(f"predicted_option: {student_answer}, target_option: {ground_truth}\n")
#                     f.flush()
#             except Exception as e:
#                 if logger is not None:
#                     logger.log(f"Logging error: {e}", content, sol)
    
#     return rewards

# def format_reward(completions, **kwargs):
#     """
#     Reward function that checks the format of the MODEL'S RESPONSE,
#     not the entire prompt-response string.
#     """
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content in completion_contents:
#         # --- FIX: Isolate the assistant's response ---
#         # The model's actual output comes after "assistant\n"
#         parts = content.split('assistant\n')
        
#         # Take the last part, which is the model's generation
#         model_output = parts[-1] if len(parts) > 1 else content
        
#         # Now, check the format on the isolated output
#         match = re.fullmatch(pattern, model_output.strip(), re.DOTALL)
        
#         reward = 1.0 if match else 0.0
#         rewards.append(reward)

#     return rewards

# def _extract_answer_letter_from_jsonish(text: str) -> str | None:
#     # 1) Prefer fenced JSON blocks: ```json ... ```
#     for m in re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE):
#         block = m.group(1)
#         try:
#             obj = json.loads(block)
#             if isinstance(obj, dict) and "Answer" in obj:
#                 m2 = re.search(r"([A-Ja-j])", str(obj["Answer"]))
#                 if m2:
#                     return m2.group(1).upper()
#         except Exception:
#             pass

#     # 2) Generic fenced code blocks without the 'json' tag: ``` ... ```
#     for m in re.finditer(r"```\s*(\{.*?\})\s*```", text, re.DOTALL):
#         block = m.group(1)
#         try:
#             obj = json.loads(block)
#             if isinstance(obj, dict) and "Answer" in obj:
#                 m2 = re.search(r"([A-Ja-j])", str(obj["Answer"]))
#                 if m2:
#                     return m2.group(1).upper()
#         except Exception:
#             pass

#     # 3) Any JSON-looking dict anywhere in the text
#     for m in re.finditer(r"\{.*?\}", text, re.DOTALL):
#         block = m.group(0)
#         try:
#             obj = json.loads(block)
#             if isinstance(obj, dict) and "Answer" in obj:
#                 m2 = re.search(r"([A-Ja-j])", str(obj["Answer"]))
#                 if m2:
#                     return m2.group(1).upper()
#         except Exception:
#             pass

#     # 4) Fallback regex for `"Answer": "X"` patterns
#     m = re.search(r'"Answer"\s*:\s*"?\s*([A-Ja-j])', text)
#     if m:
#         return m.group(1).upper()

#     # 5) Last resort: first standalone A–J letter
#     m = re.search(r"\b([A-J])\b", text)
#     if m:
#         return m.group(1).upper()

#     return None

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

def accuracy_reward(completions, solution, logger=None, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for i, (content, sol) in enumerate(zip(contents, solution)):
        # Ground truth letter (0->A, 1->B, ...)
        # try:
        #     gt_letter = chr(ord('A') + int(sol))
        # except Exception:
        #     gt_letter = None
        gt_letter = sol
        # Isolate assistant generation if your logs include role prefixes
        parts = content.split('assistant\n')
        search_area = parts[-1] if len(parts) > 1 else content

        # Extract predicted letter from (fenced) JSON
        pred_letter = extract_reasoning_answer_term(search_area)

        reward = 1.0 if (gt_letter is not None and pred_letter == gt_letter) else 0.0
        rewards.append(reward)

        print(f"predicted_option: {pred_letter}, target_option: {gt_letter}")

        if logger is not None:
            logger.log(reward, content, sol)

        if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = f'debug_files/debug_listener_{os.getenv("LOCAL_RANK", "0")}_job_{os.getenv("SLURM_JOB_ID", "unknown")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"predicted_option: {pred_letter}, target_option: {gt_letter}\n")
                    f.flush()
            except Exception as e:
                if logger is not None:
                    logger.log(f"Logging error: {e}", content, sol)

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
    "format": format_reward
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

def main(script_args, training_args, model_args):
    script_args.reward_funcs = ['accuracy', 'format']
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
            r=64,  # the rank of the LoRA matrices
            lora_alpha=128, # the weight
            lora_dropout=0.1, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules="all-linear", # the name of the layers to add LoRA
            # target_modules=[
            # "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            # "gate_proj", "up_proj", "down_proj",      # MLP
            # "mm_projector", "visual_proj"             # Vision bridge
            # ],
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


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
