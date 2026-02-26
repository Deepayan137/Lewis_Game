#!/usr/bin/env python3
"""
visual_comparison.py

Test if visual comparison (seeing all options in a grid) leads to better predictions
compared to text-based comparison. Processes failure cases and generates detailed
comparison reports.

Usage:
    python visual_comparison.py --json_path results/analysis/YoLLaVA_lora/high_confidence_wrong.json
"""

from __future__ import annotations

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_utils.common import get_model_config
from inference_utils.model import setup_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger(__name__)


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_type: str, db_type: str, dataset: str = None, seed: int = 23):
    """
    Load Qwen VLM model using same setup as personalize.py.

    Args:
        model_type: Model type identifier
        db_type: Database type for model config
        dataset: Optional dataset name
        seed: Random seed

    Returns:
        Tuple of (model, processor)
    """
    try:
        model_config = get_model_config(model_type, dataset=dataset, seed=seed)
    except ValueError:
        LOG.warning(f"Model type '{model_type}' not in config, using as direct path")
        model_config = {
            'path': model_type,
            'use_peft': 'lora' in model_type.lower(),
        }

    model_path = model_config['path']
    use_peft = model_config['use_peft']

    LOG.info(f"Loading model from {model_path}")
    model, processor = setup_model(model_path, use_peft=use_peft)

    return model, processor


def load_descriptions(dataset: str, seed: int = 23) -> Dict[str, Any]:
    """Load concept descriptions to get category information."""
    desc_path = Path(f"outputs/{dataset}/all/seed_{seed}/database_original_7b.json")

    if not desc_path.exists():
        raise FileNotFoundError(f"Description file not found: {desc_path}")

    with open(desc_path, 'r') as f:
        data = json.load(f)

    LOG.info(f"Loaded descriptions from {desc_path}")
    return data


def get_concept_category(desc_data: Dict[str, Any], concept_name: str) -> str:
    """Get category for a concept."""
    try:
        category = desc_data['concept_dict'][f'<{concept_name}>']['category']
        return category
    except (KeyError, IndexError):
        LOG.warning(f"Could not find category for concept '{concept_name}'")
        return "entity"  # Fallback


# ============================================================================
# Grid Creation
# ============================================================================

def create_labeled_grid(
    ret_image_paths: List[str],
    target_height: int = 512,
    badge_radius: int = 20,
    padding: int = 10
) -> Image.Image:
    """
    Create a horizontal grid of 3 images with A, B, C labels.

    Args:
        ret_image_paths: List of 3 image paths
        target_height: Target height for all images (width scaled proportionally)
        badge_radius: Radius of the label badge circle
        padding: Padding from edge for badge placement

    Returns:
        PIL Image of the grid
    """
    if len(ret_image_paths) != 3:
        raise ValueError("Expected exactly 3 retrieved images")

    # Load and resize images maintaining aspect ratio
    images = []
    for img_path in ret_image_paths:
        img = Image.open(img_path).convert('RGB')

        # Calculate new width maintaining aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)

        # Resize
        img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        images.append(img_resized)

    # Create horizontal grid
    total_width = sum(img.width for img in images)
    grid = Image.new('RGB', (total_width, target_height))

    # Paste images
    x_offset = 0
    for img in images:
        grid.paste(img, (x_offset, 0))
        x_offset += img.width

    # Add A, B, C labels
    draw = ImageDraw.Draw(grid, 'RGBA')

    # Badge colors (with alpha for semi-transparency)
    badge_colors = [
        (255, 50, 50, 200),   # Red for A
        (50, 150, 255, 200),  # Blue for B
        (50, 200, 50, 200),   # Green for C
    ]

    labels = ['A', 'B', 'C']

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    x_offset = 0
    for idx, img in enumerate(images):
        # Badge position (top-left with padding)
        badge_x = x_offset + padding + badge_radius
        badge_y = padding + badge_radius

        # Draw semi-transparent circle
        draw.ellipse(
            [badge_x - badge_radius, badge_y - badge_radius,
             badge_x + badge_radius, badge_y + badge_radius],
            fill=badge_colors[idx]
        )

        # Draw white letter
        text = labels[idx]
        # Get text bounding box to center it
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = badge_x - text_width // 2
        text_y = badge_y - text_height // 2 - 2  # Slight adjustment for visual centering

        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

        x_offset += img.width

    return grid


# ============================================================================
# Inference
# ============================================================================

def visual_comparison_inference(
    model,
    processor,
    query_image: Image.Image,
    grid_image: Image.Image,
    category: str,
    temperature: float = 1e-6,
    max_new_tokens: int = 128
) -> str:
    """
    Run inference with query image and labeled grid.

    Args:
        model: VLM model
        processor: Model processor
        query_image: Query image
        grid_image: Grid of 3 labeled reference images
        category: Category of the concept (e.g., "person", "pet animal")
        temperature: Generation temperature
        max_new_tokens: Maximum tokens to generate

    Returns:
        Model response string
    """
    # Construct prompt
    prompt = (
        f"You are shown a query image and three reference images arranged horizontally and labeled A, B, and C.\n\n"
        f"Task: Identify which reference image (A, B, or C) shows the same {category} as the query image.\n\n"
        f"Carefully compare the visual features and characteristics to determine the best match.\n\n"
        f"Output your answer in JSON format:\n"
        f"{{\n"
        f"  \"Reasoning\": \"Brief explanation of your choice\",\n"
        f"  \"Answer\": \"A\"  // Must be exactly one of: A, B, or C\n"
        f"}}"
    )

    # Prepare messages with two images (query + grid)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Query image:"},
            {"type": "image", "image": query_image},
            {"type": "text", "text": "\nReference images (labeled A, B, C):"},
            {"type": "image", "image": grid_image},
            {"type": "text", "text": f"\n{prompt}"},
        ],
    }]

    from qwen_vl_utils import process_vision_info

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Generate
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Decode
    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


# ============================================================================
# Processing
# ============================================================================

def parse_response(response: str) -> Tuple[str, str]:
    """
    Parse JSON response to extract reasoning and answer.

    Returns:
        Tuple of (reasoning, answer_letter)
    """
    import re

    # Try to extract JSON
    try:
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith('```'):
            response = re.sub(r'^```(?:json)?\s*', '', response)
            response = re.sub(r'\s*```$', '', response)

        data = json.loads(response)
        reasoning = data.get('Reasoning', '')
        answer = data.get('Answer', '').strip().upper()

        return reasoning, answer
    except:
        # Fallback: try to extract answer letter
        match = re.search(r'"Answer"\s*:\s*"([ABC])"', response, re.IGNORECASE)
        if match:
            return response, match.group(1).upper()

        # Last resort: look for isolated A, B, or C
        for letter in ['A', 'B', 'C']:
            if letter in response.upper():
                return response, letter

        return response, None


def process_failure_cases(
    json_path: Path,
    model,
    processor,
    desc_data: Dict[str, Any],
    output_dir: Path,
    temperature: float = 1e-6,
    max_new_tokens: int = 128
) -> Dict[str, Any]:
    """
    Process all failure cases and generate comparison report.

    Args:
        json_path: Path to failure cases JSON
        model: VLM model
        processor: Model processor
        desc_data: Description data for categories
        output_dir: Output directory
        temperature: Generation temperature
        max_new_tokens: Maximum tokens to generate

    Returns:
        Results dictionary
    """
    # Load failure cases
    with open(json_path, 'r') as f:
        cases = json.load(f)

    LOG.info(f"Loaded {len(cases)} failure cases from {json_path}")

    # Process each case
    results = {
        'source_file': str(json_path),
        'total_cases': len(cases),
        'summary': {
            'predictions_changed': 0,
            'changed_to_correct': 0,
            'changed_to_wrong': 0,
            'stayed_wrong': 0,
            'accuracy_original': 0.0,
            'accuracy_visual': 0.0,
        },
        'cases': []
    }

    correct_visual = 0

    for idx, case in enumerate(cases, 1):
        LOG.info(f"Processing case {idx}/{len(cases)}: {case['solution']}")

        try:
            # Load query image
            query_image = Image.open(case['image_path']).convert('RGB')

            # Create labeled grid
            grid_image = create_labeled_grid(case['ret_paths'])

            # Get category
            solution = case['solution']
            category = get_concept_category(desc_data, solution)

            # Run visual comparison inference
            response = visual_comparison_inference(
                model, processor, query_image, grid_image,
                category, temperature, max_new_tokens
            )

            # Parse response
            reasoning, answer_letter = parse_response(response)

            # Map answer letter to concept name
            # ret_paths order is A, B, C
            letter_to_concept = {
                'A': case['ret_paths'][0].split('/')[-2],
                'B': case['ret_paths'][1].split('/')[-2],
                'C': case['ret_paths'][2].split('/')[-2],
            }

            visual_prediction = letter_to_concept.get(answer_letter, '')
            visual_correct = visual_prediction.lower() == solution.lower()

            if visual_correct:
                correct_visual += 1

            # Determine if prediction changed
            original_prediction = case['predicted']
            changed = original_prediction.lower() != visual_prediction.lower()

            # Categorize the change
            if changed:
                results['summary']['predictions_changed'] += 1
                if visual_correct:
                    results['summary']['changed_to_correct'] += 1
                else:
                    results['summary']['changed_to_wrong'] += 1
            else:
                results['summary']['stayed_wrong'] += 1

            # Store case result
            case_result = {
                'case_id': case['case_id'],
                'image_path': case['image_path'],
                'solution': solution,
                'category': category,
                'original': {
                    'prediction': original_prediction,
                    'reasoning': case.get('response', ''),
                    'correct': False,  # All are wrong in failure cases
                    'entropy': case.get('entropy', 0.0),
                    'margin': case.get('margin', 0.0),
                },
                'visual_comparison': {
                    'prediction': visual_prediction,
                    'answer_letter': answer_letter,
                    'reasoning': reasoning,
                    'correct': visual_correct,
                    'full_response': response,
                },
                'changed': changed,
                'improved': changed and visual_correct,
            }

            results['cases'].append(case_result)

        except Exception as e:
            LOG.error(f"Error processing case {idx}: {e}")
            continue

    # Calculate accuracies
    results['summary']['accuracy_original'] = 0.0  # All are wrong by definition
    results['summary']['accuracy_visual'] = (correct_visual / len(cases)) * 100 if cases else 0.0

    return results


# ============================================================================
# Report Generation
# ============================================================================

def save_comparison_report(results: Dict[str, Any], output_path: Path):
    """Save detailed comparison report to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    LOG.info(f"Saved comparison report to {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print summary statistics."""
    summary = results['summary']

    print("\n" + "="*80)
    print("VISUAL COMPARISON ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nSource: {results['source_file']}")
    print(f"Total Cases: {results['total_cases']}")
    print(f"\nAccuracy:")
    print(f"  Original (Text-based): {summary['accuracy_original']:.2f}%")
    print(f"  Visual Comparison: {summary['accuracy_visual']:.2f}%")
    print(f"  Improvement: {summary['accuracy_visual'] - summary['accuracy_original']:.2f} percentage points")
    print(f"\nPrediction Changes:")
    print(f"  Total Changed: {summary['predictions_changed']} ({summary['predictions_changed']/results['total_cases']*100:.1f}%)")
    print(f"  Changed to Correct: {summary['changed_to_correct']}")
    print(f"  Changed to Wrong: {summary['changed_to_wrong']}")
    print(f"  Stayed Wrong: {summary['stayed_wrong']}")
    print(f"\nEffectiveness of Visual Comparison:")
    if summary['predictions_changed'] > 0:
        success_rate = (summary['changed_to_correct'] / summary['predictions_changed']) * 100
        print(f"  Success Rate (when changed): {success_rate:.1f}%")
    print("="*80 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visual comparison analysis for failure cases"
    )

    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to failure cases JSON file')
    parser.add_argument('--model_type', type=str, default='lora_7b_grpo_res_fixed_no_reco',
                        help='Model type')
    parser.add_argument('--db_type', type=str, default='original_7b',
                        help='Database type')
    parser.add_argument('--dataset', type=str, default='YoLLaVA',
                        help='Dataset name (for loading descriptions)')
    parser.add_argument('--seed', type=int, default=23,
                        help='Random seed')
    parser.add_argument('--temperature', type=float, default=1e-6,
                        help='Generation temperature')
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum tokens to generate')

    args = parser.parse_args()

    # Validate paths
    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Setup output directory
    output_dir = json_path.parent
    output_filename = json_path.stem + '_visual_comparison_results.json'
    output_path = output_dir / output_filename

    LOG.info(f"Starting visual comparison analysis")
    LOG.info(f"Input: {json_path}")
    LOG.info(f"Output: {output_path}")

    # Load model
    model, processor = load_model(args.model_type, args.db_type, args.dataset, args.seed)

    # Load descriptions for category information
    desc_data = load_descriptions(args.dataset, args.seed)

    # Process failure cases
    results = process_failure_cases(
        json_path, model, processor, desc_data, output_dir,
        args.temperature, args.max_new_tokens
    )

    # Save report
    save_comparison_report(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
