# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import torch
import torch.nn as nn

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from datasets import DatasetDict, Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
import sys
sys.path.insert(0, 'src/virft/src/')
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from open_r1.logger import PredictionLogger
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

logger = PredictionLogger(log_path=os.getenv("LOG_PATH"))

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

def caption_accuracy_reward(completions, solution, logger=None, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        ground_truth = sol[0]
        # Extract answer from content if it has think/answer tags
        # Debug: Print the content to see what we're working with
        # if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
        #     print(f"DEBUG: Raw content: {repr(content)}")
        #     print(f"DEBUG: Content length: {len(content)}")
        #     print(f"DEBUG: Looking for answer tags...")
        #     print(f"DEBUG: '<answer>' found at position: {content.find('<answer>')}")
        #     print(f"DEBUG: '</answer>' found at position: {content.find('</answer>')}")
        
        # Try multiple regex patterns to handle different cases
        # First try: Standard pattern with DOTALL
        parts = content.split('assistant\n')
        search_area = parts[-1] if len(parts) > 1 else content
        content_match = re.search(r'<answer[^>]*>(.*?)</answer>', search_area, re.DOTALL)
        
        if not content_match:
            # Second try: Handle potential whitespace issues
            content_match = re.search(r'<answer\s*>(.*?)</answer\s*>', content, re.DOTALL)
        
        if not content_match:
            # Third try: More flexible pattern
            content_match = re.search(r'<answer[^>]*>(.*?)</answer[^>]*>', content, re.DOTALL)
        
        if content_match:
            student_answer = content_match.group(1).strip()
            # if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            #     print(f"DEBUG: Found answer tags, raw match: {repr(content_match.group(1))}")
            #     print(f"DEBUG: Found answer tags, extracted: {repr(student_answer)}")
        else:
            # If no answer tags found, try alternative patterns or use full content
            if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
                # print(f"DEBUG: No answer tags found with any pattern")
                # Let's try to manually find the answer tags
                answer_start = content.find('<answer>')
                answer_end = content.find('</answer>')
                if answer_start != -1 and answer_end != -1:
                    manual_extract = content[answer_start + 8:answer_end]
                    # print(f"DEBUG: Manual extraction found: {repr(manual_extract)}")
                    student_answer = manual_extract.strip()
                else:
                    # print(f"DEBUG: Could not find answer tags manually either")
                    student_answer = content.strip()
            else:
                student_answer = content.strip()
        
        ground_truth = ground_truth.lower()
        
        # Debug: Print before processing
        # if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
        #     print(f"DEBUG: Student answer before processing: {repr(student_answer)}")
        
        student_answer = student_answer.lower()
        
        # Debug: Print after processing
        # if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
        #     print(f"DEBUG: Student answer after processing: {repr(student_answer)}")
        #     print(f"DEBUG: Ground truth: {repr(ground_truth)}")
        #     print(f"DEBUG: Checking if '{ground_truth}' in '{student_answer}'")
        
        # Compare the extracted answers
        # if ground_truth in student_answer or student_answer in ground_truth:
        if ground_truth in student_answer:
            reward = 1.0
        rewards.append(reward)
        if logger is not None:
            logger.log(reward, content, sol)
        if os.getenv("DEBUG_MODE", "false").lower() == "true" and os.getenv("LOCAL_RANK", "0") == "0":
            job_id = os.getenv("JOB_ID", "nojobid")
            slurm_job_id = os.getenv("SLURM_JOB_ID", "nojobid")
            log_path = f'debug_files/debug_mcllava_{slurm_job_id}_rank{os.getenv("LOCAL_RANK", "0")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"pred:{student_answer}\n")
                    f.write(f"sol: {ground_truth}\n")
                    f.flush()
            except Exception as e:
                if logger is not None:
                    logger.log(f"Logging error: {e}", content, ground_truth)
    return rewards

def accuracy_reward(completions, solution, logger=None, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                ground_truth = ground_truth.replace(' ','').replace('_','').lower()
                student_answer = student_answer.replace(' ','').replace('_','').lower()

                # Compare the extracted answers
                # if ground_truth in student_answer or student_answer in ground_truth:
                if ground_truth == student_answer:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if logger is not None:
            logger.log(reward, content, sol)
        # import pdb; pdb.set_trace()
        # if os.getenv("DEBUG_MODE") == "true":
        # Improved debug logging: only log if DEBUG_MODE is set, include local_rank, and flush for safety
        if os.getenv("DEBUG_MODE", "false").lower() == "true":
            log_path = f'debug_mcllava_rank{os.getenv("LOCAL_RANK", "0")}.txt'
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"------------- {current_time} | Rank: {os.getenv('LOCAL_RANK', '0')} | Accuracy reward: {reward} -------------\n")
                    f.write(f"content: {content}\n")
                    f.write(f"sol: {sol}\n")
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

reward_funcs_registry = {
    "accuracy": caption_accuracy_reward,
    "format": format_reward,
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

    # # Assuming your model is loaded as 'model'
    
    # if hasattr(model, 'module'):
    #     print("Inspecting parameters of model.module:")
    #     num_trainable = count_trainable_parameters_detailed(model.module)
    #     print(f"\nTotal trainable parameters in the wrapped model: {num_trainable}")
    # else:
    #     print("Inspecting parameters of the top-level model:")
    #     num_trainable = count_trainable_parameters_detailed(model)
    #     print(f"\nTotal trainable parameters in the model: {num_trainable}")
   

def main(script_args, training_args, model_args):
    # Get reward functions
    # debug_dir = "debug_files"
    # os.makedirs(debug_dir, exist_ok=True)
    
    # # Extract dataset name (get the base name without path)
    # dataset_name = os.path.basename(script_args.dataset_name)
    # # Remove extension if present
    # dataset_name = os.path.splitext(dataset_name)[0]
    
    # # Try to get SLURM job ID first
    # job_id = os.getenv("SLURM_JOB_ID")
    # if not job_id:
    #     # If not running under SLURM, create a stable ID based on date (no seconds)
    #     job_id = datetime.now().strftime("%Y%m%d_%H%M")
    
    # # Get local rank for distributed training
    # local_rank = os.getenv("LOCAL_RANK", "0")
    
    # # Create the log path
    # log_path = os.path.join(debug_dir, f"{dataset_name}_{job_id}_rank{local_rank}.log")
    
    # # Set environment variables
    # os.environ["DEBUG_MODE"] = "true"  # Always true
    # os.environ["LOG_PATH"] = log_path
    
    # # Print for user information
    # print(f"Debug mode enabled. Logs will be saved to: {log_path}")
    script_args.reward_funcs = ['accuracy','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    ### lzy modified
    from datasets import load_dataset
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    print(script_args.dataset_name)
    if 'YoLLaVA' in script_args.dataset_name or 'MC_LLaVa' in script_args.dataset_name:
        dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    else:
        dataset = load_dataset('parquet', data_files=script_args.dataset_name)
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    # def make_conversation_multi_image(example):
    #     return {
    #         "prompt": [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    def make_conversation_multi_image(example):
        """
        Fixed version that properly loads images and creates the images field
        """
        conversation = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Placeholder for first image
                        {"type": "image"},  # Placeholder for second image
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }
        return conversation
    
    if "grid_image" in dataset[script_args.dataset_train_split].features and "image" in dataset[script_args.dataset_train_split].features:
        print("has multiple images in dataset")
        # Use the fixed function
        dataset = dataset.map(make_conversation_multi_image)
        # Filter out None values (failed image loads)
        dataset = dataset.filter(lambda x: x is not None)

    elif "image" in dataset[script_args.dataset_train_split].features and not "grid_image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
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

    # Initialize the GRPO trainer
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
