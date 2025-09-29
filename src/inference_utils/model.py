import os
import re
import torch
import json
import argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor
import logging

def setup_model(model_name_or_path, device="cuda"):
    logging.info("Loading model...")
    if model_name_or_path == "Qwen/Qwen3-8B":
        processor = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype="auto",
                device_map="auto")
    elif model_name_or_path == "openbmb/MiniCPM-o-2_6":
        model = AutoModel.from_pretrained(
            model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True)
        processor = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    else:
        """
        Setup the Qwen 2.5 VL model and processor with optimizations.
        """
        from transformers import Qwen2VLForConditionalGeneration
        logging.info(f"Loading model from {model_name_or_path}")
        processor = AutoProcessor.from_pretrained(model_name_or_path)  
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto"  # Let it automatically distribute
        )  
    # if hasattr(torch, 'compile'):
    #     try:
    #         model = torch.compile(model, mode="reduce-overhead")
    #     except:
    #        logger.exception("Model compilation failed, continuing without compilation")
    
    model.eval()
    
    # Disable gradient computation globally for inference
    for param in model.parameters():
        param.requires_grad = False
        
    return model, processor

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
        if 'Qwen' in model.name_or_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": problem},
                    ],
                }
            ]
        elif 'MiniCPM' in model.name_or_path:
            messages = [{'role': 'user', 'content': [image, problem]}]
        all_messages.append(messages)
    
    # Process all texts
    if 'MiniCPM' in model.name_or_path:
       _, output_texts = model.chat(msgs=all_messages, tokenizer=processor)
    else:
        from qwen_vl_utils import process_vision_info
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
        device = next(model.parameters()).device
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(device)
        dtype = next(model.parameters()).dtype
        use_autocast = (model.device.type == "cuda") and (dtype == torch.float16)
        with torch.no_grad():
            if use_autocast:
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
            else:
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,)
        
        # Decode outputs
        input_ids = inputs.get("input_ids", None)
        trimmed = []
        if input_ids is not None:
            for in_ids, out_ids in zip(input_ids, generated_ids):
                in_len = in_ids.shape[0]
                trimmed.append(out_ids[in_len:].unsqueeze(0))
            generated_ids_trimmed = torch.cat(trimmed, dim=0)
        else:
            generated_ids_trimmed = generated_ids  # no trimming possible
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(getattr(processor, "name_or_path", model.config._name_or_path))
            except Exception:
                tokenizer = None
        if tokenizer is not None:
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            output_texts = [str(x.tolist()) for x in generated_ids_trimmed]
        del inputs
    torch.cuda.empty_cache()
    # return output_texts if len(output_texts) > 1 else output_texts[0]
    return list(output_texts)

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