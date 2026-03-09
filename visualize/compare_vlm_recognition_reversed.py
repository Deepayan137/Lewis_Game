#!/usr/bin/env python3
"""
Compare VLM recognition methods - showing cases where Refined is correct but LoRA is wrong.
REVERSED from original: was LoRA correct + Refined wrong, now Refined correct + LoRA wrong.
"""

import json
from pathlib import Path
from collections import defaultdict

def clean_path(path):
    """Clean the file path by removing prefix and /all/ subdirectories."""
    if not path:
        return path
    # Remove the long prefix
    path = path.replace('/leonardo_work/IscrB_SMIALLM/ddas/projects/Lewis_Game/data/', 'data/')
    # Remove /all/ from paths
    # path = path.replace('/all/', '/')
    return path

def html_escape(text):
    """Escape HTML special characters."""
    if not text:
        return text
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

def extract_concept_from_path(path):
    """Extract concept name from path like data/YoLLaVA/test/duck-banana/0.png -> duck-banana"""
    if '/' in path:
        parts = path.split('/')
        # Get the second-to-last part (concept name)
        return parts[-2] if len(parts) >= 2 else 'Unknown'
    return 'Unknown'

def compare_and_find_mismatches(dataset_name, all_path, lora_suffix=None):
    """Find cases where Refined is correct but LoRA is wrong."""
    print(f"\nProcessing {dataset_name}...")

    mismatches = []

    # Load description files ONCE at dataset level
    # Path: outputs/{dataset_name}/all/seed_23/descriptions_*.json
    outputs_path = Path('outputs') / dataset_name / 'all' / 'seed_23'
    refined_desc_file = outputs_path / 'descriptions_original_7b_location_and_state_refined.json'
    lora_desc_file = outputs_path / 'descriptions_lora_7b_grpo.json'

    refined_descriptions = {}
    lora_descriptions = {}

    print(f"  Looking for refined descriptions at: {refined_desc_file}")
    if refined_desc_file.exists():
        with open(refined_desc_file, 'r') as f:
            refined_db = json.load(f)
            # Extract descriptions by concept name
            for concept_id, concept_data in refined_db.items():
                name = concept_data.get('name', concept_id)
                general = concept_data.get('general', [])
                distinguishing = concept_data.get('distinguishing features', [])
                desc = ' '.join(general + distinguishing) if general or distinguishing else ''
                refined_descriptions[name] = desc
        print(f"  ✓ Loaded {len(refined_descriptions)} refined descriptions")
    else:
        print(f"  ✗ Refined descriptions file not found!")

    print(f"  Looking for GRPO descriptions at: {lora_desc_file}")
    if lora_desc_file.exists():
        with open(lora_desc_file, 'r') as f:
            lora_db = json.load(f)
            # Extract descriptions by concept name
            for concept_id, concept_data in lora_db.items():
                name = concept_data.get('name', concept_id)
                general = concept_data.get('general', [])
                distinguishing = concept_data.get('distinguishing features', [])
                desc = ' '.join(general + distinguishing) if general or distinguishing else ''
                lora_descriptions[name] = desc
        print(f"  ✓ Loaded {len(lora_descriptions)} GRPO descriptions")
    else:
        print(f"  ✗ GRPO descriptions file not found!")

    # Iterate through all concept directories
    for concept_dir in all_path.iterdir():
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name
        seed_path = concept_dir / 'seed_23'

        if not seed_path.exists():
            continue

        refined_file = seed_path / 'recognition_model_original_7b_db_original_7b_location_and_state_refined.json'
        if not lora_suffix:
            lora_file = seed_path / 'recognition_model_original_7b_db_lora_7b_grpo.json'
        else:
            lora_file = seed_path / f'recognition_model_original_7b_db_{lora_suffix}.json'
        if not (refined_file.exists() and lora_file.exists()):
            continue

        # Load both recognition files
        with open(refined_file, 'r') as f:
            refined_data = json.load(f)
        with open(lora_file, 'r') as f:
            lora_data = json.load(f)

        # Create lookup by query_path
        refined_lookup = {item.get('query_path', ''): item for item in refined_data.get('results', [])}
        lora_lookup = {item.get('query_path', ''): item for item in lora_data.get('results', [])}

        # Find mismatches
        for query_path, refined_item in refined_lookup.items():
            if query_path not in lora_lookup:
                continue

            lora_item = lora_lookup[query_path]

            # Check if Refined is correct but LoRA is wrong (REVERSED!)
            refined_correct = refined_item.get('correct', False)
            lora_correct = lora_item.get('correct', False)

            if refined_correct and not lora_correct:
                # Clean paths
                query_fixed = clean_path(refined_item.get('query_path', ''))
                ref_fixed = clean_path(refined_item.get('ref_path', ''))

                # Extract concept names from paths
                query_concept = extract_concept_from_path(query_fixed)
                ref_concept = extract_concept_from_path(ref_fixed)

                # Get descriptions for both concepts
                query_refined_desc = refined_descriptions.get(query_concept, 'No description available')
                query_lora_desc = lora_descriptions.get(query_concept, 'No description available')
                ref_refined_desc = refined_descriptions.get(ref_concept, 'No description available')
                ref_lora_desc = lora_descriptions.get(ref_concept, 'No description available')

                mismatch = {
                    'dataset': dataset_name,
                    'concept': concept_name,
                    'query_path': query_fixed,
                    'ref_path': ref_fixed,
                    'query_concept': query_concept,
                    'ref_concept': ref_concept,
                    'question': refined_item.get('question', ''),
                    'ground_truth': refined_item.get('solution', ''),
                    'refined': {
                        'prediction': refined_item.get('pred', ''),
                        'reasoning': refined_item.get('response', '')
                    },
                    'lora': {
                        'prediction': lora_item.get('pred', ''),
                        'reasoning': lora_item.get('response', '')
                    },
                    'descriptions': {
                        'query_refined': query_refined_desc,
                        'query_lora': query_lora_desc,
                        'ref_refined': ref_refined_desc,
                        'ref_lora': ref_lora_desc
                    }
                }
                mismatches.append(mismatch)

    print(f"  Found {len(mismatches)} mismatches (Refined correct, LoRA wrong)")
    return mismatches

def generate_html(mismatches, output_file, dataset_name):
    """Generate HTML visualization for mismatches."""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} Recognition Comparison - Refined Correct, LoRA Wrong</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .header h1 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .header p {{
            color: #666;
            font-size: 1.1em;
        }}

        .stats {{
            background: #fff3cd;
            padding: 15px 20px;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
            margin-top: 15px;
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

        .concept-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}

        .images-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 25px 0;
        }}

        .image-container {{
            text-align: center;
        }}

        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            margin-bottom: 10px;
        }}

        .image-label {{
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
            font-size: 1.1em;
        }}

        .concept-label {{
            display: inline-block;
            background: #9b59b6;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            margin-top: 5px;
        }}

        .info-section {{
            margin: 20px 0;
        }}

        .info-label {{
            font-weight: 600;
            color: #555;
            margin-bottom: 8px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .info-value {{
            background: #f8f9fa;
            padding: 12px 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            font-size: 1em;
            line-height: 1.6;
        }}

        .info-value.question {{
            background: #e3f2fd;
            border-left-color: #2196f3;
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
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .prediction-box.refined .prediction-header {{
            color: #155724;
        }}

        .prediction-box.lora .prediction-header {{
            color: #721c24;
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
            color: #333;
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
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}

        .descriptions-section {{
            margin: 25px 0;
            border: 2px solid #667eea;
            border-radius: 10px;
            overflow: hidden;
        }}

        .descriptions-header {{
            background: #667eea;
            color: white;
            padding: 12px 20px;
            font-weight: 700;
            font-size: 1.1em;
            text-align: center;
        }}

        .descriptions-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}

        .description-column {{
            padding: 20px;
            border-right: 2px solid #667eea;
        }}

        .description-column:last-child {{
            border-right: none;
        }}

        .description-column.refined {{
            background: #f0f4ff;
        }}

        .description-column.lora {{
            background: #fff8f0;
        }}

        .column-title {{
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid;
        }}

        .description-column.refined .column-title {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}

        .description-column.lora .column-title {{
            color: #ff6b35;
            border-bottom-color: #ff6b35;
        }}

        .concept-description {{
            margin-bottom: 20px;
        }}

        .concept-description:last-child {{
            margin-bottom: 0;
        }}

        .concept-desc-label {{
            font-weight: 600;
            color: #555;
            margin-bottom: 6px;
            font-size: 0.9em;
        }}

        .concept-desc-text {{
            background: white;
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 0.95em;
            line-height: 1.5;
            color: #333;
            border-left: 3px solid #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="index.html" class="back-button">← Back to Overview</a>
            <h1>{dataset_name} Recognition Comparison</h1>
            <p>Cases where <strong>Location & State Refined is CORRECT</strong> but <strong>LoRA 7B GRPO is WRONG</strong></p>
            <div class="stats">
                <strong>Total Mismatches:</strong> {len(mismatches)}
            </div>
        </div>
"""

    for idx, mismatch in enumerate(mismatches, 1):
        html += f"""
        <div class="comparison-card">
            <div class="card-header">
                <h2 style="color: #667eea; margin-bottom: 10px;">Comparison #{idx}</h2>
                <span class="concept-badge">{mismatch['concept']}</span>
            </div>

            <div class="images-row">
                <div class="image-container">
                    <div class="image-label">Query Image</div>
                    <img src="{mismatch['query_path']}" alt="Query Image">
                    <div class="concept-label">{mismatch['query_concept']}</div>
                </div>
                <div class="image-container">
                    <div class="image-label">Reference Image</div>
                    <img src="{mismatch['ref_path']}" alt="Reference Image">
                    <div class="concept-label">{mismatch['ref_concept']}</div>
                </div>
            </div>

            <div class="descriptions-section">
                <div class="descriptions-header">DESCRIPTIONS</div>
                <div class="descriptions-grid">
                    <div class="description-column refined">
                        <div class="column-title">REFINED DESC</div>
                        <div class="concept-description">
                            <div class="concept-desc-label">Query Concept: {mismatch['query_concept']}</div>
                            <div class="concept-desc-text">{html_escape(mismatch['descriptions']['query_refined'])}</div>
                        </div>
                        <div class="concept-description">
                            <div class="concept-desc-label">Reference Concept: {mismatch['ref_concept']}</div>
                            <div class="concept-desc-text">{html_escape(mismatch['descriptions']['ref_refined'])}</div>
                        </div>
                    </div>
                    <div class="description-column lora">
                        <div class="column-title">GRPO DESC</div>
                        <div class="concept-description">
                            <div class="concept-desc-label">Query Concept: {mismatch['query_concept']}</div>
                            <div class="concept-desc-text">{html_escape(mismatch['descriptions']['query_lora'])}</div>
                        </div>
                        <div class="concept-description">
                            <div class="concept-desc-label">Reference Concept: {mismatch['ref_concept']}</div>
                            <div class="concept-desc-text">{html_escape(mismatch['descriptions']['ref_lora'])}</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="info-section">
                <div class="info-label">Question</div>
                <div class="info-value question">{html_escape(mismatch['question'])}</div>
            </div>

            <div class="info-section">
                <div class="info-label">Ground Truth</div>
                <div class="info-value"><strong>{mismatch['ground_truth']}</strong></div>
            </div>

            <div class="predictions-grid">
                <div class="prediction-box refined">
                    <div class="prediction-header">
                        🔴 Location & State Refined
                        <span class="status-badge correct">✓ CORRECT</span>
                    </div>
                    <div class="prediction-text">Prediction: {mismatch['refined']['prediction']}</div>
                    <div class="reasoning">
                        <strong>Reasoning:</strong><br>
                        {html_escape(mismatch['refined']['reasoning'])}
                    </div>
                </div>

                <div class="prediction-box lora">
                    <div class="prediction-header">
                        🟢 LoRA 7B GRPO
                        <span class="status-badge wrong">✗ WRONG</span>
                    </div>
                    <div class="prediction-text">Prediction: {mismatch['lora']['prediction']}</div>
                    <div class="reasoning">
                        <strong>Reasoning:</strong><br>
                        {html_escape(mismatch['lora']['reasoning'])}
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

    # Process MyVLM
    myvlm_all_path = base_path / 'MyVLM' / 'all'
    myvlm_mismatches = compare_and_find_mismatches('MyVLM', myvlm_all_path)
    generate_html(myvlm_mismatches, 'myvlm_results.html', 'MyVLM')

    # Process YoLLaVA
    yollava_all_path = base_path / 'YoLLaVA' / 'all'
    yollava_mismatches = compare_and_find_mismatches('YoLLaVA', yollava_all_path)
    generate_html(yollava_mismatches, 'yollava_results.html', 'YoLLaVA')

    print(f"\n✅ Done! Generated HTML files for recognition comparison (Refined correct, LoRA wrong)")
