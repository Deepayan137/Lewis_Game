import os
import re
import torch
import json
import argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor
import logging

def setup_model(model_name_or_path, use_peft=False, device="cuda"):
    logging.info("Loading model...")
    """
    Setup the Qwen 2.5 VL model and processor with optimizations.
    """
    from transformers import Qwen2VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    if use_peft:
        from peft import PeftConfig, PeftModel
        config = PeftConfig.from_pretrained(model_name_or_path)
        print("Loading LoRA model from {}".format(model_name_or_path))
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",)
        model = PeftModel.from_pretrained(model, model_name_or_path)
    else:
        logging.info(f"Loading model from {model_name_or_path}")  
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto"  # Let it automatically distribute
        )  
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model, processor

def speaker_describes_batch(model, processor, problems, images, ret_images=None, temperature=1e-6, max_new_tokens=128, num_return_sequences=1):
    """
    Process images one at a time with multiple generations per image to avoid OOM.
    If ret_images is provided, process two images (image + ret_image) per iteration.
    """
    if not isinstance(images, list):
        images = [images]
    if not isinstance(problems, list):
        problems = [problems]
    
    # Handle ret_images
    if ret_images is not None:
        if not isinstance(ret_images, list):
            ret_images = [ret_images]
    
    all_outputs = []
    
    # Process ONE image (or pair of images) at a time
    if ret_images is not None:
        iterator = zip(images, ret_images, problems)
    else:
        iterator = zip(images, problems)
    
    for item in iterator:
        if ret_images is not None:
            image, ret_image, problem = item
            # Two images per message
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "image", "image": ret_image},
                    {"type": "text", "text": problem},
                ],
            }]
        else:
            image, problem = item
            # Single image per message
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": problem},
                ],
            }]
        
        from qwen_vl_utils import process_vision_info
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Single or dual image input
        inputs = processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            # "temperature": temperature,
            # "top_p": 0.9,
            "num_return_sequences": num_return_sequences,  # Multiple sequences for THIS image
            "pad_token_id": processor.tokenizer.eos_token_id,
        }
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]
        
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        all_outputs.append(output_texts)  # Add all sequences from this image
        
        # Clean up immediately
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
    
    return all_outputs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optimized Lewis Game Evaluation')
    parser.add_argument("--model_type", type=str, default='original',
                       help='Model type: original or finetuned')
    parser.add_argument("--category", type=str, default='clothe',
                       help='Model type: original or finetuned')
    args = parser.parse_args()
    # Model setup
    if args.model_type == 'original':
        model_path = "Qwen/Qwen2-VL-2B-Instruct"
    else:
        model_path = f"/gpfs/projects/ehpc171/ddas/projects/Visual-RFT/share_models/Qwen2.5-VL-2B-Instruct_GRPO_lewis_{args.category}_test_test_subset"
    
    print(f"Loading model from {model_path}")
    # start_time = time.time()
    
    # Setup models with optimizations
    speaker_model, processor = setup_model(model_path)