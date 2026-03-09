#!/usr/bin/env python3
"""
Analyze captioning inaccuracies for a specific configuration.
This script loads results for a given model_type and db_type, and displays incorrect predictions.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any


def clean_path(path: str) -> str:
    """Clean the file path for display."""
    if not path:
        return path
    path = path.replace('/leonardo_work/IscrB_SMIALLM/ddas/projects/Lewis_Game/data/', 'data/')
    return path


def clean_json_text(text: str) -> str:
    """Clean JSON formatting from text."""
    text = str(text)
    text = text.replace('{', '').replace('}', '').replace('[', '').replace(']', '')
    text = text.replace('"', '').replace("'", '')
    text = text.replace('\\n', ' ')
    text = re.sub(r'(Reasoning|Answer|name|category|general|distinguishing features):\s*', '', text)
    text = ' '.join(text.split())
    return text


def html_escape(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return text
    return (str(text).replace('&', '&amp;')
                     .replace('<', '&lt;')
                     .replace('>', '&gt;')
                     .replace('"', '&quot;')
                     .replace("'", '&#39;'))


def load_descriptions(dataset: str, db_type: str) -> Dict:
    """
    Load concept descriptions from outputs directory.

    Args:
        dataset: Dataset name (e.g., 'MyVLM', 'YoLLaVA')
        db_type: Database type (e.g., 'original_7b', 'lora_7b_grpo')

    Returns:
        Dictionary of concept descriptions
    """
    desc_path = Path(f'outputs/{dataset}/all/seed_23/descriptions_{db_type}.json')
    try:
        with open(desc_path, 'r') as f:
            descriptions = json.load(f)
            print(f"✓ Loaded descriptions from: {desc_path}")
            return descriptions
    except FileNotFoundError:
        print(f"✗ Warning: Description file not found: {desc_path}")
        return {}


def get_concept_description(descriptions: Dict, concept_name: str) -> Dict:
    """
    Get description for a specific concept.

    Args:
        descriptions: Dictionary of all concept descriptions
        concept_name: Name of the concept to look up

    Returns:
        Dictionary with 'general' and 'distinguishing features' keys
    """
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


def find_incorrect_predictions(dataset_name: str, all_path: Path, desc: Dict,
                               model_type: str, db_type: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Find all incorrect predictions for the given configuration.

    Args:
        dataset_name: Name of the dataset
        all_path: Path to the results directory
        desc: Dictionary of concept descriptions
        model_type: Model type for loading results file
        db_type: Database type for loading results file
        k: Number of retrieved images (default: 3)

    Returns:
        List of dictionaries containing incorrect prediction information
    """
    incorrect_predictions = []

    # Construct results filename based on configuration
    results_filename = f'results_model_{model_type}_db_{db_type}_k_{k}_analysis.json'

    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_name}")
    print(f"Results file pattern: {results_filename}")
    print(f"{'='*70}\n")

    # Iterate through all concept directories
    for concept_dir in all_path.iterdir():
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name
        seed_path = concept_dir / 'seed_23'

        if not seed_path.exists():
            continue

        results_file = seed_path / results_filename
        if not results_file.exists():
            continue

        # Load results file
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            continue

        results = data.get('results', [])

        # Process each result
        for result in results:
            correct = result.get('correct', False)

            # Only process incorrect predictions
            if correct:
                continue

            image_path = result.get('image_path', '')
            solution_concept = result.get('solution', '')
            pred_concept = result.get('pred_name', result.get('solution', ''))

            # Get descriptions for both solution and prediction
            solution_desc = get_concept_description(desc, solution_concept)
            pred_desc = get_concept_description(desc, pred_concept)

            # Build error record
            error_record = {
                'dataset': dataset_name,
                'concept': concept_name,
                'image_path': image_path,
                'ret_paths': result.get('ret_paths', []),
                'solution': solution_concept,
                'prediction': pred_concept,
                'solution_desc': solution_desc,
                'prediction_desc': pred_desc,
                'reasoning': result.get('response', 'No reasoning provided')
            }

            incorrect_predictions.append(error_record)

    print(f"Found {len(incorrect_predictions)} incorrect predictions")
    return incorrect_predictions


def display_error(error: Dict[str, Any], index: int):
    """
    Display a single error in a formatted way.

    Args:
        error: Error record dictionary
        index: Index number for display
    """
    print(f"\n{'='*70}")
    print(f"ERROR #{index}")
    print(f"{'='*70}")
    print(f"Dataset:      {error['dataset']}")
    print(f"Concept:      {error['concept']}")
    print(f"Image:        {error['image_path']}")
    print(f"\n{'-'*70}")
    print(f"SOLUTION:     {error['solution']}")
    print(f"{'-'*70}")
    print(f"PREDICTION:   {error['prediction']} ✗")
    print(f"{'-'*70}")

    # Display solution description
    print(f"\n📝 SOLUTION DESCRIPTION: {error['solution']}")
    print(f"\nGeneral:")
    for item in error['solution_desc']['general']:
        print(f"  • {item}")
    print(f"\nDistinguishing Features:")
    for item in error['solution_desc']['distinguishing features']:
        print(f"  • {item}")

    # Display prediction description
    print(f"\n📝 PREDICTION DESCRIPTION: {error['prediction']}")
    print(f"\nGeneral:")
    for item in error['prediction_desc']['general']:
        print(f"  • {item}")
    print(f"\nDistinguishing Features:")
    for item in error['prediction_desc']['distinguishing features']:
        print(f"  • {item}")

    # Display model reasoning
    print(f"\n🤖 MODEL REASONING:")
    print(f"{error['reasoning']}")

    # Display retrieved images
    if error['ret_paths']:
        print(f"\n🖼️  RETRIEVED IMAGES:")
        for idx, path in enumerate(error['ret_paths'][:3], 1):
            print(f"  {idx}. {path}")


def generate_html(errors: List[Dict[str, Any]], output_file: str,
                  dataset_name: str, model_type: str, db_type: str):
    """
    Generate an interactive HTML visualization with infinite scroll.

    Args:
        errors: List of error records
        output_file: Path to output HTML file
        dataset_name: Name of the dataset
        model_type: Model type used
        db_type: Database type used
    """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} Captioning Errors - {model_type} / {db_type}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: sticky;
            top: 20px;
            z-index: 100;
        }}

        .header h1 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .config-info {{
            background: #f0f4ff;
            padding: 15px 20px;
            border-radius: 10px;
            margin-top: 15px;
            border-left: 4px solid #667eea;
        }}

        .config-info span {{
            display: inline-block;
            margin-right: 20px;
            font-weight: 600;
        }}

        .stats {{
            background: #fff3cd;
            padding: 15px 20px;
            border-radius: 10px;
            margin-top: 15px;
            border-left: 4px solid #ffc107;
        }}

        .error-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: fadeIn 0.5s ease-in;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .card-header {{
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-title {{
            color: #667eea;
            font-size: 1.5em;
            font-weight: 700;
        }}

        .concept-badge {{
            background: #667eea;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}

        .images-section {{
            margin: 25px 0;
        }}

        .images-grid {{
            display: grid;
            grid-template-columns: 350px repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}

        .image-container {{
            text-align: center;
            position: relative;
        }}

        .image-container img {{
            max-width: 100%;
            height: 220px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }}

        .image-container img:hover {{
            transform: scale(1.05);
        }}

        .image-container.query img {{
            border: 5px solid #2196f3;
            height: 250px;
        }}

        .image-label {{
            font-weight: 700;
            margin-bottom: 10px;
            font-size: 1.1em;
            padding: 5px 10px;
            border-radius: 5px;
        }}

        .image-label.query {{
            background: #2196f3;
            color: white;
        }}

        .image-label.retrieved {{
            background: #9c27b0;
            color: white;
        }}

        .predictions-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin: 25px 0;
        }}

        .prediction-box {{
            padding: 20px;
            border-radius: 10px;
            border: 3px solid;
        }}

        .prediction-box.solution {{
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-color: #28a745;
        }}

        .prediction-box.wrong {{
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-color: #dc3545;
        }}

        .prediction-header {{
            font-weight: 700;
            font-size: 1.5em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .prediction-header.solution {{
            color: #155724;
        }}

        .prediction-header.wrong {{
            color: #721c24;
        }}

        .concept-name {{
            font-size: 1.3em;
            font-weight: 700;
            margin: 15px 0;
            padding: 10px 15px;
            border-radius: 8px;
        }}

        .concept-name.solution {{
            background: #28a745;
            color: white;
        }}

        .concept-name.wrong {{
            background: #dc3545;
            color: white;
        }}

        .description-section {{
            margin: 15px 0;
            background: rgba(255, 255, 255, 0.7);
            padding: 15px;
            border-radius: 8px;
        }}

        .description-label {{
            font-weight: 700;
            font-size: 0.9em;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .description-text {{
            line-height: 1.6;
            color: #333;
        }}

        .reasoning-section {{
            margin-top: 15px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 8px;
            border-left: 4px solid;
        }}

        .solution .reasoning-section {{
            border-left-color: #28a745;
        }}

        .wrong .reasoning-section {{
            border-left-color: #dc3545;
        }}

        .reasoning-label {{
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 1.1em;
        }}

        .reasoning-text {{
            line-height: 1.6;
            color: #333;
            font-size: 0.95em;
        }}

        .loading {{
            text-align: center;
            padding: 20px;
            color: white;
            font-size: 1.2em;
        }}

        .scroll-to-top {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5em;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            opacity: 0;
            pointer-events: none;
        }}

        .scroll-to-top.visible {{
            opacity: 1;
            pointer-events: all;
        }}

        .scroll-to-top:hover {{
            background: #764ba2;
            transform: translateY(-5px);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 {dataset_name} Captioning Error Analysis</h1>
            <div class="config-info">
                <span>📊 Model: <strong>{model_type}</strong></span>
                <span>💾 Database: <strong>{db_type}</strong></span>
            </div>
            <div class="stats">
                <strong>Total Errors Found:</strong> {len(errors)}
            </div>
        </div>

        <div id="error-container">
"""

    # Generate cards for each error
    for idx, error in enumerate(errors, 1):
        solution = error['solution']
        prediction = error['prediction']

        html += f"""
        <div class="error-card" data-index="{idx}">
            <div class="card-header">
                <div class="card-title">Error #{idx}</div>
                <div class="concept-badge">Ground Truth: {error['concept']}</div>
            </div>

            <div class="images-section">
                <div class="images-grid">
                    <div class="image-container query">
                        <div class="image-label query">📷 QUERY IMAGE</div>
                        <img src="{clean_path(error['image_path'])}" alt="Query Image" loading="lazy">
                    </div>
"""

        # Add retrieved images
        for ret_idx, ret_path in enumerate(error['ret_paths'][:3], 1):
            html += f"""
                    <div class="image-container">
                        <div class="image-label retrieved">Retrieved #{ret_idx}</div>
                        <img src="{clean_path(ret_path)}" alt="Retrieved {ret_idx}" loading="lazy">
                    </div>
"""

        html += """
                </div>
            </div>

            <div class="predictions-section">
                <div class="prediction-box solution">
                    <div class="prediction-header solution">
                        ✓ SOLUTION (Correct)
                    </div>
                    <div class="concept-name solution">""" + solution + """</div>

                    <div class="description-section">
                        <div class="description-label">📝 General Description</div>
                        <div class="description-text">""" + ' '.join(error['solution_desc']['general']) + """</div>
                    </div>

                    <div class="description-section">
                        <div class="description-label">🔍 Distinguishing Features</div>
                        <div class="description-text">""" + ' '.join(error['solution_desc']['distinguishing features']) + """</div>
                    </div>
                </div>

                <div class="prediction-box wrong">
                    <div class="prediction-header wrong">
                        ✗ MODEL PREDICTION (Incorrect)
                    </div>
                    <div class="concept-name wrong">""" + prediction + """</div>

                    <div class="description-section">
                        <div class="description-label">📝 General Description</div>
                        <div class="description-text">""" + ' '.join(error['prediction_desc']['general']) + """</div>
                    </div>

                    <div class="description-section">
                        <div class="description-label">🔍 Distinguishing Features</div>
                        <div class="description-text">""" + ' '.join(error['prediction_desc']['distinguishing features']) + """</div>
                    </div>

                    <div class="reasoning-section">
                        <div class="reasoning-label">🤖 Model Reasoning:</div>
                        <div class="reasoning-text">""" + html_escape(clean_json_text(error['reasoning'])) + """</div>
                    </div>
                </div>
            </div>
        </div>
"""

    html += """
        </div>
    </div>

    <div class="scroll-to-top" id="scrollToTop" onclick="scrollToTop()">
        ↑
    </div>

    <script>
        // Infinite scroll (lazy loading images)
        let observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, {
            threshold: 0.1
        });

        document.querySelectorAll('.error-card').forEach(card => {
            observer.observe(card);
        });

        // Scroll to top button
        window.addEventListener('scroll', () => {
            const scrollButton = document.getElementById('scrollToTop');
            if (window.pageYOffset > 300) {
                scrollButton.classList.add('visible');
            } else {
                scrollButton.classList.remove('visible');
            }
        });

        function scrollToTop() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }

        // Image lazy loading enhancement
        document.querySelectorAll('img[loading="lazy"]').forEach(img => {
            img.addEventListener('load', function() {
                this.style.opacity = '1';
            });
        });
    </script>
</body>
</html>
"""

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ HTML file generated: {output_file}")


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze captioning inaccuracies for a specific configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze MyVLM with original model and database
  python analyze_captioning_errors.py --dataset MyVLM --model original_7b --db original_7b

  # Analyze YoLLaVA with LoRA model
  python analyze_captioning_errors.py --dataset YoLLaVA --model lora_7b_grpo --db original_7b

  # Generate HTML visualization
  python analyze_captioning_errors.py --dataset MyVLM --model original_7b --db original_7b --html errors.html

  # Limit output to first 10 errors and save JSON + HTML
  python analyze_captioning_errors.py --dataset MyVLM --model original_7b --db original_7b --limit 10 --output errors.json --html errors.html
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['MyVLM', 'YoLLaVA'],
        help='Dataset name (MyVLM or YoLLaVA)'
    )

    parser.add_argument(
        '--model',
        dest='model_type',
        type=str,
        required=True,
        help='Model type (e.g., original_7b, lora_7b_grpo)'
    )

    parser.add_argument(
        '--db',
        dest='db_type',
        type=str,
        required=True,
        help='Database type (e.g., original_7b, lora_7b_grpo)'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of retrieved images (default: 3)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of errors to display (default: all)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional: Save results to JSON file'
    )

    parser.add_argument(
        '--html',
        type=str,
        default=None,
        help='Optional: Generate HTML visualization file'
    )

    args = parser.parse_args()

    # Load descriptions based on db_type
    descriptions = load_descriptions(args.dataset, args.db_type)

    if not descriptions:
        print(f"\n✗ Error: Could not load descriptions for {args.dataset} with db_type={args.db_type}")
        return

    # Set up paths
    results_base_path = Path('results')
    all_path = results_base_path / args.dataset / 'all'

    if not all_path.exists():
        print(f"\n✗ Error: Results path does not exist: {all_path}")
        return

    # Find incorrect predictions
    errors = find_incorrect_predictions(
        args.dataset,
        all_path,
        descriptions,
        args.model_type,
        args.db_type,
        args.k
    )

    if not errors:
        print("\n✓ No incorrect predictions found!")
        return

    # Apply limit if specified
    display_errors = errors[:args.limit] if args.limit else errors

    # Display errors
    print(f"\n{'#'*70}")
    print(f"DISPLAYING {len(display_errors)} ERRORS")
    if args.limit and len(errors) > args.limit:
        print(f"(Showing first {args.limit} of {len(errors)} total errors)")
    print(f"{'#'*70}")

    for idx, error in enumerate(display_errors, 1):
        display_error(error, idx)

    # Save to file if requested
    # if args.output:
    #     output_path = Path('html') / f'{args.dataset}_model_{args.model}_db_{args.db}_errors.json'
    #     with open(output_path, 'w') as f:
    #         json.dump(errors, f, indent=2)
    #     print(f"\n✓ Saved {len(errors)} errors to: {output_path}")

    # Generate HTML if requested
    # if args.html:
    html_path = Path('html') / f'{args.dataset}_model_{args.model_type}_db_{args.db_type}_errors.html'
    generate_html(errors, html_path, args.dataset, args.model_type, args.db_type)
    print(f"\n✓ Saved {len(errors)} errors to: {html_path}")
    # Summary
    print(f"\n{'#'*70}")
    print(f"SUMMARY")
    print(f"{'#'*70}")
    print(f"Dataset:              {args.dataset}")
    print(f"Model Type:           {args.model_type}")
    print(f"Database Type:        {args.db_type}")
    print(f"K (Retrieved):        {args.k}")
    print(f"Total Errors Found:   {len(errors)}")
    print(f"Errors Displayed:     {len(display_errors)}")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
