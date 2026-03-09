#!/usr/bin/env python3
"""
Analyze recognition errors for a specific configuration.
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
    path = path.replace('/leonardo_work/IscrB_SMIALLM/ddas/projects/Lewis_Game/', '')
    return path


def html_escape(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return text
    return (str(text).replace('&', '&amp;')
                     .replace('<', '&lt;')
                     .replace('>', '&gt;')
                     .replace('"', '&quot;')
                     .replace("'", '&#39;'))


def extract_reasoning_from_response(response: str) -> Dict[str, str]:
    """
    Extract reasoning and answer from the model response.

    Args:
        response: The model's JSON response string

    Returns:
        Dictionary with 'reasoning' and 'answer' keys
    """
    try:
        # Try to parse as JSON if it contains json markers
        if '```json' in response:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                return {
                    'reasoning': data.get('Reasoning', data.get('reasoning', 'No reasoning provided')),
                    'answer': data.get('Answer', data.get('answer', 'No answer'))
                }

        # Fallback: try to parse the response as JSON directly
        data = json.loads(response)
        return {
            'reasoning': data.get('Reasoning', data.get('reasoning', 'No reasoning provided')),
            'answer': data.get('Answer', data.get('answer', 'No answer'))
        }
    except:
        # If parsing fails, return the raw response
        return {
            'reasoning': response,
            'answer': 'N/A'
        }


def concept_from_path(path: str) -> str:
    """Extract concept name as the parent directory of the image file."""
    return Path(path).parent.name if path else ''


def load_database(dataset: str, db_type: str, seed: int) -> Dict[str, str]:
    """
    Load speaker descriptions from the database JSON.

    Returns a dict mapping plain concept name -> description string
    (general[0] + distinct features[0]).
    """
    db_path = Path(f'outputs/{dataset}/all/seed_{seed}/database_{db_type}.json')
    if not db_path.exists():
        print(f"Warning: Database not found at {db_path}")
        return {}
    with open(db_path, 'r') as f:
        data = json.load(f)
    concept_dict = data.get('concept_dict', {})
    lookup = {}
    for key, entry in concept_dict.items():
        plain_name = key.strip('<>')
        info = entry.get('info', {})
        general = info.get('general', [])
        distinct = info.get('distinct features', [])
        parts = []
        if general:
            parts.append(general[0])
        if distinct:
            parts.append(distinct[0])
        lookup[plain_name] = ' '.join(parts)
    return lookup


def find_incorrect_predictions(dataset_name: str, results_path: Path,
                               model_type: str, db_type: str,
                               db_lookup: Dict[str, str],
                               seed: int = 23,
                               use_desc: bool = False) -> List[Dict[str, Any]]:
    """
    Find all incorrect predictions for the given configuration.

    Args:
        dataset_name: Name of the dataset
        results_path: Path to the results directory
        model_type: Model type for loading results file
        db_type: Database type for loading results file

    Returns:
        List of dictionaries containing incorrect prediction information
    """
    incorrect_predictions = []

    # Construct results filename based on configuration
    if use_desc:
        results_filename = f'recognition_model_{model_type}_db_{db_type}_with_desc.json'
    else:
        results_filename = f'recognition_model_{model_type}_db_{db_type}_no_desc.json'
    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_name}")
    print(f"Results file pattern: {results_filename}")
    print(f"{'='*70}\n")

    # Iterate through all concept directories
    for concept_dir in results_path.iterdir():
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name
        seed_path = concept_dir / f'seed_{seed}'

        if not seed_path.exists():
            continue

        results_file = seed_path / results_filename
        if not results_file.exists():
            print(f"Skipping {concept_name}: {results_filename} not found")
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

            query_path = result.get('query_path', '')
            ref_path = result.get('ref_path', '')
            question = result.get('question', '')
            response = result.get('response', '')
            pred = result.get('pred', '')
            solution = result.get('solution', '')

            # Extract reasoning from response
            parsed_response = extract_reasoning_from_response(response)

            # Look up speaker descriptions (silently skip if not found)
            ref_concept = concept_from_path(ref_path)
            query_desc = db_lookup.get(concept_name, '')
            ref_desc = db_lookup.get(ref_concept, '')

            # Build error record
            error_record = {
                'dataset': dataset_name,
                'concept': concept_name,
                'query_path': query_path,
                'ref_path': ref_path,
                'question': question,
                'response': response,
                'reasoning': parsed_response['reasoning'],
                'model_answer': parsed_response['answer'],
                'pred': pred,
                'solution': solution,
                'query_desc': query_desc,
                'ref_concept': ref_concept,
                'ref_desc': ref_desc,
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
    print(f"\n{'-'*70}")
    print(f"Query Image:  {error['query_path']}")
    print(f"Ref Image:    {error['ref_path']}")
    print(f"{'-'*70}")
    print(f"\n📖 QUERY DESCRIPTION ({error['concept']}):")
    print(f"{error['query_desc']}" if error['query_desc'] else "  (not found)")
    print(f"\n📖 REF DESCRIPTION ({error['ref_concept']}):")
    print(f"{error['ref_desc']}" if error['ref_desc'] else "  (not found)")
    print(f"\n{'-'*70}")
    print(f"\nQuestion: {error['question']}")
    print(f"\n{'-'*70}")
    print(f"SOLUTION:     {error['solution']}")
    print(f"PREDICTION:   {error['pred']} ✗")
    print(f"{'-'*70}")

    # Display model reasoning
    print(f"\n🤖 MODEL REASONING:")
    print(f"{error['reasoning']}")

    print(f"\n📝 MODEL ANSWER:")
    print(f"{error['model_answer']}")


def generate_html(errors: List[Dict[str, Any]], output_file: str,
                  dataset_name: str, model_type: str, db_type: str):
    """
    Generate an interactive HTML visualization.

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
    <title>{dataset_name} Recognition Errors - {model_type} / {db_type}</title>
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
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 15px;
        }}

        .image-container {{
            text-align: center;
            position: relative;
        }}

        .image-container img {{
            max-width: 100%;
            height: 300px;
            object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
            background: white;
            padding: 10px;
        }}

        .image-container img:hover {{
            transform: scale(1.05);
        }}

        .image-container.query img {{
            border: 5px solid #2196f3;
        }}

        .image-container.reference img {{
            border: 5px solid #9c27b0;
        }}

        .image-label {{
            font-weight: 700;
            margin-bottom: 10px;
            font-size: 1.1em;
            padding: 8px 15px;
            border-radius: 5px;
            display: inline-block;
        }}

        .image-label.query {{
            background: #2196f3;
            color: white;
        }}

        .image-label.reference {{
            background: #9c27b0;
            color: white;
        }}

        .description-section {{
            margin: 25px 0;
            padding: 20px;
            background: #e8f5e9;
            border-radius: 10px;
            border-left: 4px solid #43a047;
        }}

        .description-section-title {{
            font-weight: 700;
            font-size: 1.2em;
            color: #2e7d32;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .description-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}

        .description-box {{
            background: white;
            border-radius: 8px;
            padding: 12px 15px;
        }}

        .description-box-label {{
            font-weight: 700;
            font-size: 0.85em;
            text-transform: uppercase;
            color: #43a047;
            margin-bottom: 8px;
        }}

        .description-box-text {{
            font-size: 0.95em;
            line-height: 1.7;
            color: #333;
        }}

        .question-section {{
            margin: 25px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}

        .question-label {{
            font-weight: 700;
            font-size: 1.2em;
            color: #667eea;
            margin-bottom: 10px;
        }}

        .question-text {{
            font-size: 1.1em;
            line-height: 1.6;
            color: #333;
        }}

        .prediction-section {{
            margin: 25px 0;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .prediction-box {{
            padding: 15px 20px;
            border-radius: 10px;
            border: 3px solid;
            text-align: center;
        }}

        .prediction-box.solution {{
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-color: #28a745;
        }}

        .prediction-box.wrong {{
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border-color: #dc3545;
        }}

        .prediction-label {{
            font-weight: 700;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .prediction-label.solution {{
            color: #155724;
        }}

        .prediction-label.wrong {{
            color: #721c24;
        }}

        .prediction-value {{
            font-size: 1.5em;
            font-weight: 700;
        }}

        .prediction-value.solution {{
            color: #28a745;
        }}

        .prediction-value.wrong {{
            color: #dc3545;
        }}

        .reasoning-section {{
            margin: 25px 0;
            padding: 20px;
            background: #fff8e1;
            border-radius: 10px;
            border-left: 4px solid #ffc107;
        }}

        .reasoning-label {{
            font-weight: 700;
            font-size: 1.2em;
            color: #f57c00;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .reasoning-text {{
            line-height: 1.8;
            color: #333;
            font-size: 1em;
            background: white;
            padding: 15px;
            border-radius: 8px;
        }}

        .response-section {{
            margin: 25px 0;
            padding: 20px;
            background: #e3f2fd;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        }}

        .response-label {{
            font-weight: 700;
            font-size: 1.2em;
            color: #1976d2;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .response-text {{
            line-height: 1.8;
            color: #333;
            font-size: 1em;
            background: white;
            padding: 15px;
            border-radius: 8px;
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

        .image-path {{
            font-size: 0.85em;
            color: #666;
            margin-top: 8px;
            font-family: monospace;
            word-break: break-all;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 {dataset_name} Recognition Error Analysis</h1>
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
        solution = html_escape(str(error['solution']))
        prediction = html_escape(str(error['pred']))
        question = html_escape(error['question'])
        reasoning = html_escape(error['reasoning'])
        model_answer = html_escape(str(error['model_answer']))
        query_desc = html_escape(error.get('query_desc', ''))
        ref_desc = html_escape(error.get('ref_desc', ''))
        ref_concept = html_escape(error.get('ref_concept', ''))

        html += f"""
        <div class="error-card" data-index="{idx}">
            <div class="card-header">
                <div class="card-title">Error #{idx}</div>
                <div class="concept-badge">Concept: {html_escape(error['concept'])}</div>
            </div>

            <div class="images-section">
                <div class="images-grid">
                    <div class="image-container query">
                        <div class="image-label query">📷 QUERY IMAGE</div>
                        <img src="{clean_path(error['query_path'])}" alt="Query Image" loading="lazy">
                        <div class="image-path">{clean_path(error['query_path'])}</div>
                    </div>
                    <div class="image-container reference">
                        <div class="image-label reference">🎯 REFERENCE IMAGE</div>
                        <img src="{clean_path(error['ref_path'])}" alt="Reference Image" loading="lazy">
                        <div class="image-path">{clean_path(error['ref_path'])}</div>
                    </div>
                </div>
            </div>

            <div class="description-section">
                <div class="description-section-title">📖 Speaker Descriptions</div>
                <div class="description-grid">
                    <div class="description-box">
                        <div class="description-box-label">Query concept: {html_escape(error['concept'])}</div>
                        <div class="description-box-text">{query_desc if query_desc else '<em>not found</em>'}</div>
                    </div>
                    <div class="description-box">
                        <div class="description-box-label">Ref concept: {ref_concept}</div>
                        <div class="description-box-text">{ref_desc if ref_desc else '<em>not found</em>'}</div>
                    </div>
                </div>
            </div>

            <div class="question-section">
                <div class="question-label">❓ Question</div>
                <div class="question-text">{question}</div>
            </div>

            <div class="prediction-section">
                <div class="prediction-box solution">
                    <div class="prediction-label solution">✓ Expected Answer</div>
                    <div class="prediction-value solution">{solution}</div>
                </div>
                <div class="prediction-box wrong">
                    <div class="prediction-label wrong">✗ Model Prediction</div>
                    <div class="prediction-value wrong">{prediction}</div>
                </div>
            </div>

            <div class="reasoning-section">
                <div class="reasoning-label">🤖 Model Reasoning</div>
                <div class="reasoning-text">{reasoning}</div>
            </div>

            <div class="response-section">
                <div class="response-label">📝 Model Answer</div>
                <div class="response-text">{model_answer}</div>
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
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✓ HTML file generated: {output_file}")


def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze recognition errors for a specific configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze MyVLM with original model and database
  python analyze_recognition_error.py --dataset MyVLM --model original_7b --db original_7b

  # Analyze YoLLaVA with LoRA model
  python analyze_recognition_error.py --dataset YoLLaVA --model lora_7b_grpo --db original_7b

  # Limit output to first 10 errors
  python analyze_recognition_error.py --dataset MyVLM --model original_7b --db original_7b --limit 10
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
        '--limit',
        type=int,
        default=None,
        help='Limit number of errors to display (default: all)'
    )
    parser.add_argument(
        '--use_desc',
        action='store_true',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=23,
        help='Seed used for results and database paths (default: 23)'
    )
    args = parser.parse_args()

    # Set up paths
    results_base_path = Path('results')
    all_path = results_base_path / args.dataset / 'all'

    if not all_path.exists():
        print(f"\n✗ Error: Results path does not exist: {all_path}")
        return

    # Load speaker descriptions database
    db_lookup = load_database(args.dataset, args.db_type, args.seed)
    print(f"Loaded {len(db_lookup)} concept descriptions from database.")

    # Find incorrect predictions
    errors = find_incorrect_predictions(
        args.dataset,
        all_path,
        args.model_type,
        args.db_type,
        db_lookup=db_lookup,
        seed=args.seed,
        use_desc=args.use_desc,
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

    # Generate HTML
    if args.use_desc:
        html_path = Path('html') / f'{args.dataset}_recognition_model_{args.model_type}_db_{args.db_type}_errors_use_desc.html'
    else:
        html_path = Path('html') / f'{args.dataset}_recognition_model_{args.model_type}_db_{args.db_type}_errors_no_desc.html'
    generate_html(errors, html_path, args.dataset, args.model_type, args.db_type)

    # Summary
    print(f"\n{'#'*70}")
    print(f"SUMMARY")
    print(f"{'#'*70}")
    print(f"Dataset:              {args.dataset}")
    print(f"Model Type:           {args.model_type}")
    print(f"Database Type:        {args.db_type}")
    print(f"Total Errors Found:   {len(errors)}")
    print(f"Errors Displayed:     {len(display_errors)}")
    print(f"HTML Output:          {html_path}")
    print(f"{'#'*70}\n")


if __name__ == '__main__':
    main()
