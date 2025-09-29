import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json

from contextlib import nullcontext
from deepspeed.runtime.zero import GatheredParameters
def listener_generate_safe(model, **kwargs):
    # detect if any params are DS-sharded
    params = [p for p in model.parameters()
              if hasattr(p, "ds_status") or "deepspeed" in str(type(p)).lower()]
    ctx = GatheredParameters(params, modifier_rank=None) if params else nullcontext()

    with torch.no_grad():   # inference_mode(False) is safer w/ DS
        with ctx:
            return model.generate(**kwargs)

class Listener:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None):
        # if device is None:
        #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
        #     device = f"cuda:{local_rank}"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            # device_map=device,
            torch_dtype=torch.float16,
            # load_in_8bit=True,

            # attn_implementation="flash_attention_2",
            # use_cache=False
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Get token IDs for yes/no for confidence extraction
        self.yes_token_id = self.processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_token_id = self.processor.tokenizer.encode("no", add_special_tokens=False)[0]
        self.baseline = 0.0  # Just a scalar!
        self.baseline_momentum = 0.9

    def _prepare_image_input(self, image):
        """
        Helper method to prepare image input - accepts both PIL Image objects and file paths
        """
        if isinstance(image, str):
            # If it's a string, treat it as a file path
            return image
        elif isinstance(image, Image.Image):
            # If it's a PIL Image object, return it directly
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}. Expected str (file path) or PIL.Image.Image object.")

    def listener_process_single_image(self, image, question, return_confidence=False):
        """
        Process a single image with the given question
        
        Args:
            image: Either a file path (str) or PIL Image object
            question: Question text to ask about the image
            return_confidence: Whether to return confidence scores
        """
        # Prepare image input
        image_input = self._prepare_image_input(image)
        
        # Create message in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_input,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Prepare inputs using the official format
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generate with confidence scores if requested
        # with torch.no_grad():
        if return_confidence:
            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=10,
            #     do_sample=False,
            #     return_dict_in_generate=True,
            #     output_scores=True
            # )
            outputs = listener_generate_safe(
                self.model,
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
                 # input_ids, attention_mask, pixel_values, etc.
            )
            # import pdb;pdb.set_trace()
            generated_ids = outputs.sequences
            
            # Extract confidence from first token
            confidence_score = None
            yes_prob = None
            no_prob = None
            
            if outputs.scores:
                first_token_logits = outputs.scores[0][0]  # Shape: [vocab_size]
                
                # Get logits for yes/no tokens
                yes_logit = first_token_logits[self.yes_token_id].item()
                no_logit = first_token_logits[self.no_token_id].item()
                
                # Convert to probabilities
                probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
                yes_prob = probs[0].item()
                no_prob = probs[1].item()
                
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        
        if return_confidence:
            # Determine confidence based on actual response
            if response.lower().startswith('yes'):
                confidence_score = yes_prob
            else:
                confidence_score = no_prob
                
            return {
                'answer': response,
                'confidence': confidence_score,
                'yes_prob': yes_prob,
                'no_prob': no_prob
            }
        else:
            return response
    
    def listener_process_multiple_images_batched(self, images, question, batch_size=2, image_ids=None, return_confidence=True, for_contrastive_loss=False):
        """
        Process multiple images in small batches (balance between speed and memory)
        
        Args:
            images: List of images - can be file paths (str) or PIL Image objects
            question: Question text to ask about each image
            batch_size: Number of images to process at once
            image_ids: Optional list of image identifiers
            return_confidence: Whether to return confidence scores
            for_contrastive_loss: If True, returns probabilities suitable for softmax cross-entropy
        
        Returns:
            If for_contrastive_loss=True, returns probabilities suitable for softmax cross-entropy
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]
            
        results = {}
        yes_probabilities = []  # For contrastive loss
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_ids = image_ids[i:i+batch_size]
            # Process each image in the batch
            for image, img_id in zip(batch_images, batch_ids):
                result = self.listener_process_single_image(image, question, return_confidence)
                results[img_id] = result
                
                # Collect yes probabilities for contrastive loss
                if for_contrastive_loss and return_confidence:
                    yes_probabilities.append(result['yes_prob'])
            
            # Clear cache after each batch
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        # Return format suitable for contrastive learning
        if for_contrastive_loss:
            return {
                'individual_results': results,
                'yes_probabilities': torch.tensor(yes_probabilities),  # Shape: [num_images]
                'image_ids': image_ids
            }
                
        return results

    def listener_process_from_paths(self, image_paths, question, batch_size=2, image_ids=None, return_confidence=True, for_contrastive_loss=False):
        """
        Convenience method to maintain backward compatibility with path-based processing
        """
        return self.listener_process_multiple_images_batched(
            image_paths, question, batch_size, image_ids, return_confidence, for_contrastive_loss
        )

    def listener_process_from_pil_images(self, pil_images, question, batch_size=2, image_ids=None, return_confidence=True, for_contrastive_loss=False):
        """
        Convenience method for processing PIL Image objects
        """
        return self.listener_process_multiple_images_batched(
            pil_images, question, batch_size, image_ids, return_confidence, for_contrastive_loss
        )

if __name__ == "__main__":
    image_paths = [
        "samples/cup_blue_floral.jpg",
        "samples/cup_chocolate.jpg",
        "samples/cup_golden_floral.jpg",
        "samples/cup_leaves.jpg",
        "samples/cup_rim_blue.jpg"
    ]
    from PIL import Image
    images = [Image.open(image_path).convert('RGBA') for image_path in image_paths]
    question = "Does the description a white porcelain cup match the cup in the image? Answer in yes or no."
    listener = Listener()
    out = listener.listener_process_multiple_images_batched(images, question, return_confidence=True, for_contrastive_loss=True, batch_size=5)
    print(out)