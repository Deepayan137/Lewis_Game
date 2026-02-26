# Visual Comparison Analysis

Test whether visual comparison (seeing all options in a labeled grid) leads to better predictions compared to text-based comparison.

## Overview

This script takes failure cases (high/low confidence wrong predictions) and re-evaluates them using a **visual comparison approach** where the model sees:
- The query image
- All 3 retrieved images in a horizontal grid, labeled A, B, C

The goal is to determine if visual comparison can correct mistakes made by text-based comparison.

## Visual Grid Layout

- **Format**: Horizontal layout with 3 images side-by-side
- **Labels**: Small colored circle badges (red/blue/green) with white letters A, B, C
- **Position**: Top-left corner with padding (minimizes occlusion)
- **Aspect Ratio**: Maintained for all images (512px height, width scaled proportionally)

## Usage

```bash
cd /sessions/dreamy-admiring-ramanujan/mnt/Lewis_Game/src/analysis

# Analyze high confidence wrong cases
python visual_comparison.py \
    --json_path results/analysis/YoLLaVA_lora_7b_grpo_res_fixed_no_reco/high_confidence_wrong.json \
    --dataset YoLLaVA

# Analyze low confidence wrong cases
python visual_comparison.py \
    --json_path results/analysis/YoLLaVA_lora_7b_grpo_res_fixed_no_reco/low_confidence_wrong.json \
    --dataset YoLLaVA
```

## Arguments

- `--json_path`: Path to failure cases JSON (high_confidence_wrong.json or low_confidence_wrong.json) **[Required]**
- `--model_type`: Model type (default: "lora_7b_grpo_res_fixed_no_reco")
- `--db_type`: Database type (default: "original_7b")
- `--dataset`: Dataset name for loading category info (default: "YoLLaVA")
- `--seed`: Random seed (default: 23)
- `--temperature`: Generation temperature (default: 1e-6)
- `--max_new_tokens`: Max tokens to generate (default: 128)

## Output

### Output File
Saved to: `{json_path_stem}_visual_comparison_results.json`

Example: `high_confidence_wrong_visual_comparison_results.json`

### Output Structure

```json
{
  "source_file": "results/analysis/.../high_confidence_wrong.json",
  "total_cases": 38,
  "summary": {
    "predictions_changed": 15,
    "changed_to_correct": 10,
    "changed_to_wrong": 5,
    "stayed_wrong": 23,
    "accuracy_original": 0.0,
    "accuracy_visual": 26.3
  },
  "cases": [
    {
      "case_id": 1,
      "image_path": "data/YoLLaVA/test/all/khanhvy/0.png",
      "solution": "khanhvy",
      "category": "person",
      "original": {
        "prediction": "thuytien",
        "reasoning": "...",
        "correct": false,
        "entropy": 0.0037,
        "margin": 0.9994
      },
      "visual_comparison": {
        "prediction": "khanhvy",
        "answer_letter": "C",
        "reasoning": "...",
        "correct": true,
        "full_response": "{...}"
      },
      "changed": true,
      "improved": true
    },
    ...
  ]
}
```

### Console Output

```
================================================================================
VISUAL COMPARISON ANALYSIS SUMMARY
================================================================================

Source: results/analysis/.../high_confidence_wrong.json
Total Cases: 38

Accuracy:
  Original (Text-based): 0.00%
  Visual Comparison: 26.32%
  Improvement: 26.32 percentage points

Prediction Changes:
  Total Changed: 15 (39.5%)
  Changed to Correct: 10
  Changed to Wrong: 5
  Stayed Wrong: 23

Effectiveness of Visual Comparison:
  Success Rate (when changed): 66.7%
================================================================================
```

## Key Metrics

### Summary Statistics

- **Total Cases**: Number of failure cases processed
- **Accuracy Original**: Always 0% (by definition, these are wrong predictions)
- **Accuracy Visual**: Percentage correct with visual comparison
- **Improvement**: Percentage point improvement

### Prediction Changes

- **Total Changed**: Cases where prediction changed from original
- **Changed to Correct**: Changed predictions that became correct ✅
- **Changed to Wrong**: Changed predictions that are still wrong ❌
- **Stayed Wrong**: Predictions that didn't change (still wrong)

### Success Rate
When the model changes its prediction, what % of those changes are improvements?
- **Formula**: `changed_to_correct / predictions_changed × 100`
- **Ideal**: > 70% means visual comparison helps when it decides to change

## Interpretation

### Good Results Indicators:
1. **High Accuracy Visual** (> 30%): Visual comparison substantially improves predictions
2. **High Success Rate** (> 70%): When model changes prediction, it's usually correct
3. **Low "Changed to Wrong"**: Model rarely makes things worse

### Poor Results Indicators:
1. **Low Accuracy Visual** (< 10%): Visual comparison doesn't help
2. **Low Success Rate** (< 50%): Changes are random, no real improvement
3. **High "Changed to Wrong"**: Visual comparison confuses the model

## Next Steps

After running visual comparison:

1. **Compare Results**: Check `changed_to_correct` vs `changed_to_wrong` ratio
2. **Analyze Improved Cases**: Review cases where `improved: true`
3. **Investigate Degraded Cases**: Check cases where model changed to wrong answer
4. **Category Analysis**: Do certain categories benefit more from visual comparison?

## Technical Details

### Prompt Used
```
You are shown a query image and three reference images arranged horizontally
and labeled A, B, and C.

Task: Identify which reference image (A, B, or C) shows the same {category}
as the query image.

Carefully compare the visual features and characteristics to determine the
best match.

Output your answer in JSON format:
{
  "Reasoning": "Brief explanation of your choice",
  "Answer": "A"  // Must be exactly one of: A, B, or C
}
```

- **Category**: Dynamically set based on concept (e.g., "person", "pet animal", "object")
- **Pure Visual**: No text descriptions provided, only images

### Grid Creation
- Uses PIL (Pillow) for image manipulation
- Maintains aspect ratios to avoid distortion
- Semi-transparent badges for visibility without occlusion
- Colored badges (Red/Blue/Green) for clear distinction

## Dependencies

```bash
pip install torch pillow transformers qwen-vl-utils
```
