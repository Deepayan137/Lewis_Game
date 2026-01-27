import os
import re
import torch
import torch.nn as nn
from PIL import Image
import requests

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from datasets import DatasetDict, Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
import sys
sys.path.insert(0, 'src/virft/src/')
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from open_r1.logger import PredictionLogger
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from listener import Listener
import json

logger = PredictionLogger(log_path=os.getenv("LOG_PATH"))

import torch.distributed as dist

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

listener_model = None
def accuracy_reward(completions, solution, logger=None, **kwargs):
    global listener_model
    contents = [completion[0]["content"] for completion in completions]    
    rewards = []            
    path_candidates = kwargs.get('ret_paths')  # List of candidate image paths for each example
    category = kwargs.get('category')
    for i, (content, sol) in enumerate(zip(contents, solution)):
        cat = category[i]
        content = extract_answer_content(content)
        listener_question = f'Does the description -> {content} match the {cat} in the image? Answer in yes or no.'
        candidate_paths = path_candidates[i]  # or however you index
        candidate_images = [Image.open(item).convert('RGB') for item in candidate_paths]
        target_index = sol  # Index of correct image
        listener_results = listener_model.listener_process_multiple_images_batched(candidate_images, listener_question, batch_size=1,
        return_confidence=True, for_contrastive_loss=True)
        yes_probabilities = listener_results['yes_probabilities']  # Tensor of shape [num_candidates]
        predicted_index = yes_probabilities.argmax().item()
        reward = 1.0 if predicted_index == target_index else 0.0
        rewards.append(reward)
        
        if logger is not None:
            logger.log(reward, content, sol)
        
        if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Add this import
            log_path = f'debug_files/debug_lewis_rank{os.getenv("LOCAL_RANK", "0")}_job{os.getenv("SLURM_JOB_ID", "unknown")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"sol: {sol}\n")
                    f.write(f"predicted_index: {predicted_index}, target_index: {target_index}\n")
                    f.flush()
            except Exception as e:
                if logger is not None:
                    logger.log(f"Logging error: {e}", content, sol)
    
    return rewards

def format_reward(completions, **kwargs):
    """
    Reward function that checks the format of the MODEL'S RESPONSE,
    not the entire prompt-response string.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        # --- FIX: Isolate the assistant's response ---
        # The model's actual output comes after "assistant\n"
        parts = content.split('assistant\n')
        
        # Take the last part, which is the model's generation
        model_output = parts[-1] if len(parts) > 1 else content
        
        # Now, check the format on the isolated output
        match = re.fullmatch(pattern, model_output.strip(), re.DOTALL)
        
        reward = 1.0 if match else 0.0
        rewards.append(reward)

    return rewards

def length_reward(completions, logger=None, **kwargs):
    """
    Reward if the model's <answer> contains exactly one sentence.
    Otherwise, reward = 0.
    """
    rewards = []
    answer_re = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

    completion_contents = [c[0]["content"] for c in completions]
    for i, content in enumerate(completion_contents):
        # Isolate the assistant-generated part (same trick as format_reward)
        parts = content.split('assistant\n')
        model_output = parts[-1] if len(parts) > 1 else content

        # Extract <answer> block
        m = answer_re.search(model_output.strip())
        if m:
            answer_text = m.group(1).strip()
        else:
            try:
                answer_text = extract_answer_content(model_output)  # your helper
            except Exception:
                answer_text = model_output.strip()

        # --- Sentence counting heuristic ---
        # Split by . ! ? followed by whitespace or end-of-string
        sentences = re.split(r'[.!?](?:\s|$)', answer_text)
        # Remove empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        reward = 0.1 if len(sentences) == 1 else 0.0
        rewards.append(reward)

        # Optional logging
        if logger is not None:
            logger.log({"one_sentence_reward": reward,
                        "answer_text": answer_text,
                        "n_sentences": len(sentences)})

        if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_path = f'debug_files/debug_one_sentence_rank{os.getenv("LOCAL_RANK", "0")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"--- {current_time} | Rank {os.getenv('LOCAL_RANK','0')} | Reward: {reward} ---\n")
                    f.write(f"Answer: {answer_text}\n")
                    f.write(f"Sentence count: {len(sentences)}\n\n")
                    f.flush()
            except Exception as e:
                if logger is not None:
                    logger.log(f"One-sentence logging error: {e}", answer_text)

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward
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
            lora_alpha=16, # the weight
            lora_dropout=0.1, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
            modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
        )
    else:
        peft_config = None
    from deepspeed.runtime.zero import Init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global listener_model
    with Init(enabled=False):
        listener_model = Listener()
    if hasattr(listener_model, 'model'):
        listener_model.model = listener_model.model.to(f"cuda:{local_rank}")
        listener_model.model.eval()
        for param in listener_model.model.parameters():
            param.requires_grad = False
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
