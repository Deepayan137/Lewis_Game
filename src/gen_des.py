import os
import re
import torch
import json
import argparse
import torch.nn as nn
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from tqdm import *
from transformers import AutoModelForVision2Seq, AutoProcessor
from datasets import load_dataset, load_from_disk
from datasets import DatasetDict, Dataset
from qwen_vl_utils import process_vision_info
import sys
sys.path.insert(0, '../Visual-RFT/src/virft/src/open_r1/')
from listener import Listener
from torch.utils.data import DataLoader
import time

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

class SimpleImageDataset(Dataset):
    """
    Simple dataset that returns PIL images based on index
    """
    
    def __init__(self, category, json_path=None, split='train', seed=42):
        """
        Initialize the dataset
        
        Args:
            category: Category name (e.g., 'animals', 'vehicles', etc.)
            json_path: Path to the JSON file containing image paths
            split: Data split to use ('train', 'val', 'test')
        """
        if json_path is None:
            json_path = "/gpfs/projects/ehpc171/ddas/projects/YoLLaVA/yollava-data/train_/train_test_val_seed_42_num_train_1.json"
        
        self.category = category
        self.split = split
        self.json_path = json_path
        
        # Load and process the JSON data
        self.image_paths = self._load_image_paths()
        random.seed(seed)
        print(f"Dataset initialized with {len(self.image_paths)} images from category '{category}' ({split} split)")
    
    def _load_image_paths(self):
        """Load image paths from JSON file"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        if self.category not in data:
            raise ValueError(f"Category '{self.category}' not found in JSON data. Available categories: {list(data.keys())}")
        
        dir_names = data[self.category].keys()
        image_paths = []
        
        for dir_name in tqdm(dir_names, desc=f"Loading {self.split} image paths"):
            if self.split not in data[self.category][dir_name]:
                print(f"Warning: Split '{self.split}' not found for directory '{dir_name}'. Skipping.")
                continue
                
            filepaths = data[self.category][dir_name][self.split]
            if filepaths and self.split == 'train':
                file_path = random.choice(filepaths)
                image_paths.append(file_path)
            else:
                image_paths.extend(filepaths)
        
        return image_paths
    
    def __len__(self):
        """Return the total number of images"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_paths)}")
        
        image_path = self.image_paths[idx]
        name = image_path.split('/')[-2]
        # Load image as PIL Image
        image = Image.open(image_path).convert('RGB')
        # problem =  f"""Describe the {self.category} in the image so that it can be distinguished from other {self.category} objects. DO NOT mention the background or location of the object. Output in the format: "{self.category}: [concise distinguishing qualities]"
        # Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags. The output answer format should be as follows: <think> ... </think> <answer> ... </answer>. Please strictly follow the format.
        # """
        category = self.category
        # problem = f"""Describe the {category} in the image so that it can be distinguished from other {category} objects. Do NOT mention background, location or state of the object. Write exactly one fluent sentence that begins with "The {category}" and highlights 3–4 visible distinguishing attributes (e.g., color, pattern, shape, unique marks, material if obvious). Keep the description concise and natural, without using lists or brackets.</answer>. 
        # Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags. The output answer format should be as follows: <think> ... </think> <answer> ... </answer>. Please strictly follow the format."""
        problem = (
            f'Describe the {category} in the image so that it can be distinguished from other {category} objects. '
            "Do NOT mention background, location or state of the object. "
            f'Write exactly one fluent sentence that begins with "The {category}" and highlights 3–4 visible distinguishing attributes. '
            "Keep the description concise and natural, without using lists or brackets. "
            "Output the thinking process in <think> </think> and the personalized caption in <answer> </answer> tags."
        )
        return {
            'image': image,
            'problem': problem,
            'path': image_path,
            'name':name,
            'index': idx
        }

def extract_answer_content(text):
    # Pattern to match content between <answer> and </answer> tags
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # Try a fallback pattern: <answer> ... (no closing tag)
        fallback_pattern = r'<answer>\s*(.*?)\s*$'
        fallback_match = re.search(fallback_pattern, text, re.DOTALL)
        if fallback_match:
            return fallback_match.group(1).strip()
        else:
            return text

def setup_model(model_name_or_path, device="cuda"):
    if model_name_or_path == "Qwen/Qwen3-8B":
        processor = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype="auto",
                device_map="auto")
    else:
        """
        Setup the Qwen 2.5 VL model and processor with optimizations.
        """
        processor = AutoProcessor.from_pretrained(model_name_or_path)  
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto"  # Let it automatically distribute
        )  
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except:
            print("Model compilation failed, continuing without compilation")
    
    model.eval()
    
    # Disable gradient computation globally for inference
    for param in model.parameters():
        param.requires_grad = False
        
    return model, processor

def speaker_describes(model, processor, image, problem, max_new_tokens=256):
    """
    Process a single speaker description.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": problem},
            ],
        }
    ]
    text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = processor(
        text=text,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistency
                use_cache=True,   # Enable KV cache for speed
                pad_token_id=processor.tokenizer.eos_token_id,
            )   
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts if len(output_texts) > 1 else output_texts[0]

def llm_judge_batch(model, tokenizer, problems, max_new_tokens=128):
    if not isinstance(problems, list):
        problems = [problems]
    messages_batch = [
        [
            {
                "role": "user",
                "content": problem,
            }
        ]
        for problem in problems
    ]
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        for messages in messages_batch
    ]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens
            )
    # For each item in the batch, extract the generated output after the input
    outputs = []
    input_lens = [len(input_ids) for input_ids in model_inputs.input_ids]
    for i, (gen_ids, in_len) in enumerate(zip(generated_ids, input_lens)):
        output_ids = gen_ids[in_len:].tolist()
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        # Remove pdb and debugging
        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        outputs.append(content)
    return outputs

def speaker_describes_batch(model, processor, images, problems, max_new_tokens=256):
    """
    Process multiple speaker descriptions in batch for better efficiency.
    """
    if not isinstance(images, list):
        images = [images]
    if not isinstance(problems, list):
        problems = [problems]
    
    # Prepare all messages
    all_messages = []
    for image, problem in zip(images, problems):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": problem},
                ],
            }
        ]
        all_messages.append(messages)
    
    # Process all texts
    texts = []
    all_image_inputs = []
    all_video_inputs = []
    
    for messages in all_messages:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        
        image_inputs, video_inputs = process_vision_info(messages)
        all_image_inputs.extend(image_inputs if image_inputs else [])
        all_video_inputs.extend(video_inputs if video_inputs else [])
    
    # Batch process
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Batch inference with optimizations
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistency
                use_cache=True,   # Enable KV cache for speed
                pad_token_id=processor.tokenizer.eos_token_id,
                temperature=0.4,  # Add explicit temperature
                top_p=0.9,
                repetition_penalty=1.0  # Add repetition penalty
            )
    
    # Decode outputs
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_texts if len(output_texts) > 1 else output_texts[0]

def process_batch_efficiently(speaker_model, processor, batch_items, batch_size=4):
    """
    Process a batch of items more efficiently.
    """
    results = []
    
    # Extract batch data
    problems = [item['problem'] for item in batch_items]
    images = [item['image'] for item in batch_items]
    names = [item['name'] for item in batch_items]

    # Batch speaker descriptions
    with torch.no_grad():
        contents = speaker_describes_batch(speaker_model, processor, images, problems, max_new_tokens=128)
    return list(zip(names, contents))
    
def create_data_loader(dataset, batch_size=4, shuffle=False, num_workers=4):
    def collate_fn(batch):
        return batch  # Return as-is since we handle batching manually
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn, 
        num_workers=num_workers
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized Lewis Game Evaluation')
    parser.add_argument("--data_name", type=str, default='PerVA',
                       help='name of the dataset')
    parser.add_argument("--catalog_file", type=str, default="test_catalog_seed_23.json",
                       help="Path to the catalog JSON file")
    parser.add_argument("--category", type=str, default='clothe',
                       help='Model type: original or finetuned')
    parser.add_argument("--model_type", type=str, default='original',
                       help='Model type: original or finetuned')
    parser.add_argument("--seed", type=int, default=42,
                       help='random seed')
    parser.add_argument("--batch_size", type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument("--speaker_batch_size", type=int, default=4,
                       help='Batch size for speaker model')
    parser.add_argument("--listener_batch_size", type=int, default=5,
                       help='Batch size for listener model')
    parser.add_argument("--max_samples", type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Model setup
    if args.model_type == 'original':
        model_path = "Qwen/Qwen2-VL-2B-Instruct"
    else:
        model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2.5-VL-2B-Instruct_GRPO_lewis_{args.category}_test_test_subset"
    
    print(f"Loading model from {model_path}")
    start_time = time.time()
    
    # Setup models with optimizations
    speaker_model, processor = setup_model(model_path)
    dataset = SimpleImageDataset(
        json_path=os.path.join(f'manifests/{args.data_name}', args.catalog_file),
        category=args.category,
        split="train",
        seed=args.seed
    )
    # Create data loader
    data_loader = create_data_loader(dataset, batch_size=4)
    
    results = []
    eval_start_time = time.time()
    
    # Process in batches with progress tracking
    for batch_idx, batch_items in enumerate(tqdm(data_loader, desc="Processing batches")):
        batch_start_time = time.time()
        
        # try:
        batch_results = process_batch_efficiently(
            speaker_model, processor, batch_items, 
            batch_size=args.batch_size
        )
        results.extend(batch_results)
        batch_time = time.time() - batch_start_time
        samples_per_second = len(batch_items) / batch_time
        
        if batch_idx % 5 == 0:  # Print every 5 batches
            print(f"Batch {batch_idx}: {samples_per_second:.2f} samples/sec")
                
        # except Exception as e:
        #     print(f"Error processing batch {batch_idx}: {e}")
        #     # Add dummy results to maintain count
        #     for _ in batch_items:
        #         results.append({"description": "Error", "accuracy": 0.0})
    
    total_eval_time = time.time() - eval_start_time
    savedir = os.path.join('outputs', args.data_name, args.category)
    os.makedirs(savedir, exist_ok=True)
    result_dict = {}
    for name, desc in results:
        desc = extract_answer_content(desc)
        # if ':' in desc:desc = desc.split(':')[1]
        result_dict[name] = {
            "name":name,
            "distinguishing features":desc
        }
    output_file = f"{savedir}/descriptions_{args.model_type}.json"
    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    # Print final statistics
    # accuracy = get_accuracy(results)
    avg_time_per_sample = total_eval_time / len(results)
    samples_per_second = len(results) / total_eval_time
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Total samples: {len(results)}")
    # print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")
    print(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
    print(f"Samples per second: {samples_per_second:.2f}")
    print(f"Results saved to: {output_file}")
    
    # Performance comparison with original
    original_time_per_sample = 23.33  # 35 minutes / 90 samples
    speedup = original_time_per_sample / avg_time_per_sample
    print(f"Speedup over original: {speedup:.2f}x")