import torch
from transformers import AutoProcessor
from PIL import Image
import numpy as np

def create_dummy_image(width=224, height=224, color=(255, 0, 0)):
    """Create a dummy PIL image for testing."""
    img_array = np.full((height, width, 3), color, dtype=np.uint8)
    return Image.fromarray(img_array)

def test_mixed_image_counts():
    """Test if processor can handle different numbers of images per prompt."""

    print("=" * 80)
    print("Testing Qwen2VL Processor with Mixed Image Counts")
    print("=" * 80)

    # Try different model names in order of preference
    model_candidates = [
        "Qwen/Qwen2-VL-7B-Instruct",
        # Add your local model path if you have one
        # "/path/to/local/qwen2-vl-model",
    ]

    processor = None
    model_name = None

    print(f"\n1. Loading processor (trying {len(model_candidates)} candidates)")
    for candidate in model_candidates:
        try:
            print(f"   Trying: {candidate}...", end=" ")
            processor = AutoProcessor.from_pretrained(candidate, trust_remote_code=True)
            model_name = candidate
            print("✓")
            break
        except Exception as e:
            print(f"✗ ({type(e).__name__})")

    if processor is None:
        print("\n   ✗ Failed to load any processor!")
        print("\n   Please ensure you have:")
        print("   - Installed transformers with Qwen2-VL support: pip install transformers")
        print("   - Access to the model (may need HF token)")
        print("   - Or specify a local model path in the script")
        return

    print(f"   ✓ Successfully loaded processor: {model_name}")

    # Create dummy images
    print("\n2. Creating dummy images")
    red_img = create_dummy_image(color=(255, 0, 0))
    green_img = create_dummy_image(color=(0, 255, 0))
    blue_img = create_dummy_image(color=(0, 0, 255))
    yellow_img = create_dummy_image(color=(255, 255, 0))
    print("   ✓ Created 4 dummy images (red, green, blue, yellow)")

    # Test 1: Single prompt with 1 image (baseline)
    print("\n" + "=" * 80)
    print("TEST 1: Single prompt with 1 image (baseline)")
    print("=" * 80)
    try:
        prompts = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
        ]
        images = [red_img]

        # Apply chat template
        texts = [processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                 for prompt in prompts]

        print(f"   Prompts: 1, Images: {len(images)}")
        print(f"   Image markers per prompt: [1]")

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        print(f"   ✓ Processing successful")
        print(f"   - input_ids shape: {inputs['input_ids'].shape}")
        print(f"   - pixel_values shape: {inputs['pixel_values'].shape}")
        if 'image_grid_thw' in inputs:
            print(f"   - image_grid_thw shape: {inputs['image_grid_thw'].shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    # Test 2: Single prompt with 2 images (baseline)
    print("\n" + "=" * 80)
    print("TEST 2: Single prompt with 2 images (baseline)")
    print("=" * 80)
    try:
        prompts = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "Compare these two images."},
                    ],
                }
            ]
        ]
        images = [red_img, green_img]

        texts = [processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                 for prompt in prompts]

        print(f"   Prompts: 1, Images: {len(images)}")
        print(f"   Image markers per prompt: [2]")

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        print(f"   ✓ Processing successful")
        print(f"   - input_ids shape: {inputs['input_ids'].shape}")
        print(f"   - pixel_values shape: {inputs['pixel_values'].shape}")
        if 'image_grid_thw' in inputs:
            print(f"   - image_grid_thw shape: {inputs['image_grid_thw'].shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return

    # Test 3: CRITICAL TEST - Mixed image counts in same batch
    print("\n" + "=" * 80)
    print("TEST 3: MIXED IMAGE COUNTS (1, 2, 1) - THE KEY TEST")
    print("=" * 80)
    try:
        prompts = [
            # Prompt 0: 1 image
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the first image."},
                    ],
                }
            ],
            # Prompt 1: 2 images
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "Compare these two images."},
                    ],
                }
            ],
            # Prompt 2: 1 image
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the last image."},
                    ],
                }
            ],
        ]

        # Images: [red, green, blue, yellow]
        # Mapping: prompt0->red, prompt1->(green,blue), prompt2->yellow
        images = [red_img, green_img, blue_img, yellow_img]

        texts = [processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                 for prompt in prompts]

        print(f"   Prompts: 3, Images: {len(images)}")
        print(f"   Image markers per prompt: [1, 2, 1]")
        print(f"   Expected mapping:")
        print(f"     - Prompt 0 (1 marker) -> images[0] (red)")
        print(f"     - Prompt 1 (2 markers) -> images[1:3] (green, blue)")
        print(f"     - Prompt 2 (1 marker) -> images[3] (yellow)")

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        print(f"\n   ✓ ✓ ✓ PROCESSING SUCCESSFUL! ✓ ✓ ✓")
        print(f"   - input_ids shape: {inputs['input_ids'].shape}")
        print(f"   - pixel_values shape: {inputs['pixel_values'].shape}")
        if 'image_grid_thw' in inputs:
            print(f"   - image_grid_thw shape: {inputs['image_grid_thw'].shape}")

        print("\n   CONCLUSION: The processor CAN handle mixed image counts!")
        print("   ✓ Option A (single generate call) should work!")

    except Exception as e:
        print(f"\n   ✗ ✗ ✗ PROCESSING FAILED! ✗ ✗ ✗")
        print(f"   Error: {e}")
        print(f"   Error type: {type(e).__name__}")

        print("\n   CONCLUSION: The processor CANNOT handle mixed image counts")
        print("   ✗ Option A (single generate call) will NOT work")
        print("   → Must use Option B (two separate generate calls)")

        # Try to give more details
        import traceback
        print("\n   Full traceback:")
        traceback.print_exc()

    # Test 4: Uniform batch with 2 images each (for comparison)
    print("\n" + "=" * 80)
    print("TEST 4: Uniform batch with 2 images per prompt (for comparison)")
    print("=" * 80)
    try:
        prompts = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "Compare images A and B."},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "Compare images C and D."},
                    ],
                }
            ],
        ]
        images = [red_img, green_img, blue_img, yellow_img]

        texts = [processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                 for prompt in prompts]

        print(f"   Prompts: 2, Images: {len(images)}")
        print(f"   Image markers per prompt: [2, 2]")

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        print(f"   ✓ Processing successful")
        print(f"   - input_ids shape: {inputs['input_ids'].shape}")
        print(f"   - pixel_values shape: {inputs['pixel_values'].shape}")
        if 'image_grid_thw' in inputs:
            print(f"   - image_grid_thw shape: {inputs['image_grid_thw'].shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

if __name__ == "__main__":
    test_mixed_image_counts()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("If TEST 3 passed: Use Option A (single generate call)")
    print("If TEST 3 failed: Use Option B (two separate generate calls)")
    print("=" * 80)