#!/usr/bin/env python3
"""
Compare VLM captioning methods with concept descriptions.
Shows cases where Refined is correct but LoRA is wrong.
Includes descriptions from outputs/{dataset}/all/seed_23/descriptions_{db_type}.json
"""

import json
import re
from pathlib import Path

def clean_path(path):
    """Clean the file path."""
    if not path:
        return path
    path = path.replace('/leonardo_work/IscrB_SMIALLM/ddas/projects/Lewis_Game/data/', 'data/')
    # path = path.replace('/all/', '/')
    return path

def clean_json_text(text):
    """Clean JSON formatting from text."""
    text = str(text)
    text = text.replace('{', '').replace('}', '').replace('[', '').replace(']', '')
    text = text.replace('"', '').replace("'", '')
    text = text.replace('\\n', ' ')
    text = re.sub(r'(Reasoning|Answer|name|category|general|distinguishing features):\s*', '', text)
    text = ' '.join(text.split())
    return text

def html_escape(text):
    """Escape HTML special characters."""
    if not text:
        return text
    return (str(text).replace('&', '&amp;')
                     .replace('<', '&lt;')
                     .replace('>', '&gt;')
                     .replace('"', '&quot;')
                     .replace("'", '&#39;'))

def load_descriptions(dataset, db_type):
    """Load concept descriptions from outputs directory."""
    desc_path = Path(f'outputs/{dataset}/all/seed_23/descriptions_{db_type}.json')
    try:
        with open(desc_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Description file not found: {desc_path}")
        return {}

def get_concept_description(descriptions, concept_name):
    """Get description for a concept."""
    if concept_name not in descriptions:
        return {
            'general': ['No description available'],
            'distinguishing features': ['No description available']
        }

    concept_desc = descriptions[concept_name]
    return {
        'general': concept_desc.get('general', ['No description available']),
        'distinguishing features': concept_desc.get('distinguishing features', ['No description available'])
    }



def compare_and_find_mismatches(dataset_name, all_path, refined_desc, lora_desc):
    """Find cases where Refined is correct but LoRA is wrong."""
    print(f"\nProcessing {dataset_name} captioning...")

    mismatches = []

    # Iterate through all concept directories
    for concept_dir in all_path.iterdir():
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name
        seed_path = concept_dir / 'seed_23'

        if not seed_path.exists():
            continue

        refined_file = seed_path / 'results_model_original_7b_db_original_7b_k_3.json'
        lora_file = seed_path / 'results_model_lora_7b_grpo_db_original_7b_k_3.json'

        if not (refined_file.exists() and lora_file.exists()):
            continue

        # Load both files
        with open(refined_file, 'r') as f:
            refined_data = json.load(f)
        with open(lora_file, 'r') as f:
            lora_data = json.load(f)

        refined_results = refined_data.get('results', [])
        lora_results = lora_data.get('results', [])

        # Create lookup by image_path
        refined_lookup = {r['image_path']: r for r in refined_results}
        lora_lookup = {r['image_path']: r for r in lora_results}

        # Find mismatches
        for image_path, refined_result in refined_lookup.items():
            if image_path not in lora_lookup:
                continue

            lora_result = lora_lookup[image_path]

            # Check if Refined correct but LoRA wrong
            refined_correct = refined_result.get('correct', False)
            lora_correct = lora_result.get('correct', False)

            if refined_correct and not lora_correct:
                # Get solution and predicted concept names
                solution_concept = refined_result.get('solution', '')
                refined_pred = refined_result.get('pred_name', refined_result.get('solution', ''))
                lora_pred = lora_result.get('pred_name', lora_result.get('solution', ''))

                # Get descriptions from BOTH refined and lora for BOTH concepts
                solution_refined_desc = get_concept_description(refined_desc, solution_concept)
                solution_lora_desc = get_concept_description(lora_desc, solution_concept)

                lora_pred_refined_desc = get_concept_description(refined_desc, lora_pred)
                lora_pred_lora_desc = get_concept_description(lora_desc, lora_pred)

                mismatch = {
                    'dataset': dataset_name,
                    'concept': concept_name,
                    'query_image': clean_path(refined_result.get('image_path', '')),
                    'ret_images': [clean_path(p) for p in refined_result.get('ret_paths', [])],
                    'solution': solution_concept,
                    'lora_prediction': lora_pred,
                    # Solution concept descriptions from both models
                    'solution_refined_general': solution_refined_desc['general'],
                    'solution_refined_features': solution_refined_desc['distinguishing features'],
                    'solution_lora_general': solution_lora_desc['general'],
                    'solution_lora_features': solution_lora_desc['distinguishing features'],
                    # LoRA's predicted concept descriptions from both models
                    'lora_pred_refined_general': lora_pred_refined_desc['general'],
                    'lora_pred_refined_features': lora_pred_refined_desc['distinguishing features'],
                    'lora_pred_lora_general': lora_pred_lora_desc['general'],
                    'lora_pred_lora_features': lora_pred_lora_desc['distinguishing features'],
                    # Model responses
                    'refined': {
                        'prediction': refined_pred,
                        'reasoning': refined_result.get('response', '')
                    },
                    'lora': {
                        'prediction': lora_pred,
                        'reasoning': lora_result.get('response', '')
                    }
                }
                mismatches.append(mismatch)

    print(f"  Found {len(mismatches)} mismatches (Refined correct, LoRA wrong)")
    return mismatches

def generate_html(mismatches, output_file, dataset_name):
    """Generate HTML visualization with descriptions."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} Captioning Comparison - With Descriptions</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .comparison-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .card-header {{
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}

        .images-row {{
            display: grid;
            grid-template-columns: 300px repeat(3, 1fr);
            gap: 20px;
            margin: 25px 0;
            align-items: start;
        }}

        .image-container {{
            text-align: center;
        }}

        .image-container img {{
            max-width: 100%;
            height: 200px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}

        .image-container.query img {{
            border: 4px solid #2196f3;
        }}

        .image-label {{
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 1em;
        }}

        .image-label.query {{
            color: #2196f3;
        }}

        .image-label.retrieved {{
            color: #9c27b0;
        }}

        .descriptions-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin: 25px 0;
        }}

        .description-box {{
            border-radius: 10px;
            padding: 20px;
            border: 2px solid;
        }}

        .description-box.refined {{
            background: #e8f5e9;
            border-color: #4caf50;
        }}

        .description-box.lora {{
            background: #fff3e0;
            border-color: #ff9800;
        }}

        .description-header {{
            font-weight: 700;
            font-size: 1.3em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid;
        }}

        .description-box.refined .description-header {{
            color: #2e7d32;
            border-bottom-color: #4caf50;
        }}

        .description-box.lora .description-header {{
            color: #e65100;
            border-bottom-color: #ff9800;
        }}

        .concept-section {{
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.5);
            border-radius: 8px;
        }}

        .concept-title {{
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 12px;
            color: #333;
        }}

        .concept-title.solution {{
            color: #2e7d32;
        }}

        .concept-title.prediction {{
            color: #c62828;
        }}

        .description-section {{
            margin-bottom: 12px;
        }}

        .description-label {{
            font-weight: 600;
            font-size: 0.85em;
            color: #555;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .description-text {{
            background: rgba(255,255,255,0.8);
            padding: 10px;
            border-radius: 6px;
            line-height: 1.6;
            font-size: 0.95em;
        }}

        .predictions-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-top: 25px;
        }}

        .prediction-box {{
            border-radius: 10px;
            padding: 20px;
            border: 2px solid;
        }}

        .prediction-box.refined {{
            background: #d4edda;
            border-color: #28a745;
        }}

        .prediction-box.lora {{
            background: #f8d7da;
            border-color: #dc3545;
        }}

        .prediction-header {{
            font-weight: 700;
            font-size: 1.2em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .status-badge {{
            font-size: 0.75em;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 600;
        }}

        .status-badge.correct {{
            background: #28a745;
            color: white;
        }}

        .status-badge.wrong {{
            background: #dc3545;
            color: white;
        }}

        .prediction-text {{
            font-size: 1.1em;
            margin-bottom: 15px;
            font-weight: 600;
        }}

        .reasoning {{
            background: rgba(255,255,255,0.6);
            padding: 12px;
            border-radius: 6px;
            font-size: 0.95em;
            line-height: 1.6;
        }}

        .back-button {{
            display: inline-block;
            background: white;
            color: #667eea;
            padding: 12px 25px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .back-button:hover {{
            transform: translateY(-2px);
        }}

        .concept-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}

        .highlight-green {{
            border: 3px solid #28a745 !important;
        }}

        .selection-indicator {{
            background: #ffc107;
            color: #000;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: 700;
            margin-top: 5px;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="index.html" class="back-button">← Back to Overview</a>
            <h1>{dataset_name} Captioning Comparison</h1>
            <p>Cases where <strong>Location & State Refined is CORRECT</strong> but <strong>LoRA 7B GRPO is WRONG</strong></p>
            <div style="background: #fff3cd; padding: 15px 20px; border-radius: 10px; border-left: 4px solid #ffc107; margin-top: 15px;">
                <strong>Total Mismatches:</strong> {len(mismatches)}
            </div>
        </div>
"""

    for idx, mismatch in enumerate(mismatches, 1):
        solution = mismatch['solution']
        lora_pred = mismatch['lora']['prediction']

        html += f"""
        <div class="comparison-card">
            <div class="card-header">
                <h2 style="color: #667eea; margin-bottom: 10px;">Comparison #{idx}</h2>
                <span class="concept-badge">Ground Truth: {mismatch['concept']}</span>
            </div>

            <div class="images-row">
                <div class="image-container query">
                    <div class="image-label query">Query Image</div>
                    <img src="{mismatch['query_image']}" alt="Query">
                </div>
"""

        # Add retrieved images
        for ret_idx, ret_img in enumerate(mismatch['ret_images'][:3], 1):
            is_solution = solution.lower() in ret_img.lower()
            border_class = ' class="highlight-green"' if is_solution else ''

            html += f"""
                <div class="image-container">
                    <div class="image-label retrieved">Retrieved #{ret_idx}</div>
                    <img src="{ret_img}" alt="Retrieved {ret_idx}"{border_class}>
                    {'<div class="selection-indicator">✓ Correct</div>' if is_solution else ''}
                </div>
"""

        html += """
            </div>

            <div class="descriptions-grid">
                <div class="description-box refined">
                    <div class="description-header">
                        🔴 Refined Descriptions
                    </div>

                    <div class="concept-section">
                        <div class="concept-title solution">✓ Solution: """ + solution + """</div>
                        <div class="description-section">
                            <div class="description-label">General</div>
                            <div class="description-text">""" + ' '.join(mismatch['solution_refined_general']) + """</div>
                        </div>
                        <div class="description-section">
                            <div class="description-label">Distinguishing Features</div>
                            <div class="description-text">""" + ' '.join(mismatch['solution_refined_features']) + """</div>
                        </div>
                    </div>

                    <div class="concept-section">
                        <div class="concept-title prediction">✗ LoRA Predicted: """ + mismatch['lora_prediction'] + """</div>
                        <div class="description-section">
                            <div class="description-label">General</div>
                            <div class="description-text">""" + ' '.join(mismatch['lora_pred_refined_general']) + """</div>
                        </div>
                        <div class="description-section">
                            <div class="description-label">Distinguishing Features</div>
                            <div class="description-text">""" + ' '.join(mismatch['lora_pred_refined_features']) + """</div>
                        </div>
                    </div>
                </div>

                <div class="description-box lora">
                    <div class="description-header">
                        🟢 LoRA 7B Descriptions
                    </div>

                    <div class="concept-section">
                        <div class="concept-title solution">✓ Solution: """ + solution + """</div>
                        <div class="description-section">
                            <div class="description-label">General</div>
                            <div class="description-text">""" + ' '.join(mismatch['solution_lora_general']) + """</div>
                        </div>
                        <div class="description-section">
                            <div class="description-label">Distinguishing Features</div>
                            <div class="description-text">""" + ' '.join(mismatch['solution_lora_features']) + """</div>
                        </div>
                    </div>

                    <div class="concept-section">
                        <div class="concept-title prediction">✗ LoRA Predicted: """ + mismatch['lora_prediction'] + """</div>
                        <div class="description-section">
                            <div class="description-label">General</div>
                            <div class="description-text">""" + ' '.join(mismatch['lora_pred_lora_general']) + """</div>
                        </div>
                        <div class="description-section">
                            <div class="description-label">Distinguishing Features</div>
                            <div class="description-text">""" + ' '.join(mismatch['lora_pred_lora_features']) + """</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="predictions-grid">
                <div class="prediction-box refined">
                    <div class="prediction-header">
                        🔴 Location & State Refined
                        <span class="status-badge correct">✓ CORRECT</span>
                    </div>
                    <div class="prediction-text">Prediction: """ + mismatch['refined']['prediction'] + """</div>
                    <div class="reasoning">
                        <strong>Reasoning:</strong><br>
                        """ + clean_json_text(mismatch['refined']['reasoning']) + """
                    </div>
                </div>

                <div class="prediction-box lora">
                    <div class="prediction-header">
                        🟢 LoRA 7B GRPO
                        <span class="status-badge wrong">✗ WRONG</span>
                    </div>
                    <div class="prediction-text">Prediction: """ + lora_pred + """</div>
                    <div class="reasoning">
                        <strong>Reasoning:</strong><br>
                        """ + clean_json_text(mismatch['lora']['reasoning']) + """
                    </div>
                </div>
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)
    print(f"  Generated {output_file}")

# Main execution
if __name__ == '__main__':
    base_path = Path('results')

    # Load descriptions
    myvlm_refined_desc = load_descriptions('MyVLM', 'original_7b')
    myvlm_lora_desc = load_descriptions('MyVLM', 'original_7b')
    yollava_refined_desc = load_descriptions('YoLLaVA', 'original_7b')
    yollava_lora_desc = load_descriptions('YoLLaVA', 'original_7b')

    # Process MyVLM
    myvlm_all_path = base_path / 'MyVLM' / 'all'
    myvlm_mismatches = compare_and_find_mismatches(
        'MyVLM', myvlm_all_path,
        myvlm_refined_desc, myvlm_lora_desc
    )
    generate_html(myvlm_mismatches, 'html/myvlm_captioning.html', 'MyVLM')

    # Process YoLLaVA
    yollava_all_path = base_path / 'YoLLaVA' / 'all'
    yollava_mismatches = compare_and_find_mismatches(
        'YoLLaVA', yollava_all_path,
        yollava_refined_desc, yollava_lora_desc
    )
    generate_html(yollava_mismatches, 'html/yollava_captioning.html', 'YoLLaVA')

    print(f"\n✅ Done! Generated HTML files for captioning comparison with descriptions")
