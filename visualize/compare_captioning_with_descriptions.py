#!/usr/bin/env python3
"""
Compare captioning performance between two description databases.
Shows cases where the model using db_good descriptions is correct
but the model using db_bad descriptions is wrong.

Descriptions panel per entry:
  - db_good description of the solution (GT) concept
  - db_bad description of the solution (GT) concept
  - db_bad description of the wrongly predicted concept
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean_path(path: str) -> str:
    if not path:
        return path
    path = path.replace('/leonardo_work/IscrB_SMIALLM/ddas/projects/Lewis_Game/data/', 'data/')
    path = path.replace('/gpfs/projects/ehpc171/ddas/projects/Lewis_Game/data/', 'data/')
    return path


def html_escape(text: str) -> str:
    if not text:
        return str(text)
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def extract_reasoning(response: str) -> str:
    """Extract the Reasoning field from a JSON response string."""
    if not response:
        return ''
    try:
        if '```json' in response:
            m = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if m:
                data = json.loads(m.group(1))
                return data.get('Reasoning', data.get('reasoning', response))
        data = json.loads(response)
        return data.get('Reasoning', data.get('reasoning', response))
    except Exception:
        return response


# ---------------------------------------------------------------------------
# Database loading
# ---------------------------------------------------------------------------

def load_database(dataset: str, db_type: str, seed: int) -> Dict[str, str]:
    """
    Load outputs/{dataset}/all/seed_{seed}/database_{db_type}.json.
    Returns plain_concept_name -> "general[0] distinct_features[0]".
    """
    db_path = Path(f'outputs/{dataset}/all/seed_{seed}/database_{db_type}.json')
    if not db_path.exists():
        print(f"Warning: database not found at {db_path}")
        return {}
    with open(db_path, 'r') as f:
        data = json.load(f)
    concept_dict = data.get('concept_dict', {})
    lookup: Dict[str, str] = {}
    for key, entry in concept_dict.items():
        plain = key.strip('<>')
        info = entry.get('info', {})
        general = info.get('general', [])
        distinct = info.get('distinct features', [])
        parts = []
        if general:
            parts.append(general[0])
        if distinct:
            parts.append(distinct[0])
        lookup[plain] = ' '.join(parts)
    return lookup


# ---------------------------------------------------------------------------
# Mismatch finding
# ---------------------------------------------------------------------------

def find_mismatches(dataset_name: str, all_path: Path,
                    model_type: str, db_good: str, db_bad: str,
                    db_good_lookup: Dict[str, str], db_bad_lookup: Dict[str, str],
                    seed: int) -> List[Dict[str, Any]]:
    """
    Find entries where model+db_good is correct and model+db_bad is wrong.
    """
    mismatches: List[Dict[str, Any]] = []
    print(f"\nProcessing {dataset_name}  |  good={db_good}  bad={db_bad}")

    for concept_dir in sorted(all_path.iterdir()):
        if not concept_dir.is_dir():
            continue

        concept_name = concept_dir.name
        seed_path = concept_dir / f'seed_{seed}'
        if not seed_path.exists():
            continue

        good_file = seed_path / f'results_model_{model_type}_db_{db_good}_k_3.json'
        bad_file  = seed_path / f'results_model_{model_type}_db_{db_bad}_k_3.json'

        if not good_file.exists():
            print(f"  Skipping {concept_name}: {good_file.name} not found")
            continue
        if not bad_file.exists():
            print(f"  Skipping {concept_name}: {bad_file.name} not found")
            continue

        with open(good_file) as f:
            good_data = json.load(f)
        with open(bad_file) as f:
            bad_data = json.load(f)

        good_by_img = {r['image_path']: r for r in good_data.get('results', [])}
        bad_by_img  = {r['image_path']: r for r in bad_data.get('results', [])}

        for img_path, good_r in good_by_img.items():
            if img_path not in bad_by_img:
                continue
            bad_r = bad_by_img[img_path]

            if not good_r.get('correct', False) or bad_r.get('correct', False):
                continue

            solution_concept = good_r.get('solution', '')
            bad_pred_concept  = bad_r.get('pred_name', '')

            mismatches.append({
                'dataset':        dataset_name,
                'concept':        concept_name,
                'query_image':    clean_path(img_path),
                'ret_images':     [clean_path(p) for p in good_r.get('ret_paths', [])],
                'solution':       solution_concept,
                'bad_pred':       bad_pred_concept,
                # Descriptions
                'desc_good_sol':  db_good_lookup.get(solution_concept, ''),   # db_good GT desc
                'desc_bad_sol':   db_bad_lookup.get(solution_concept, ''),    # db_bad  GT desc
                'desc_bad_pred':  db_bad_lookup.get(bad_pred_concept, ''),    # db_bad  wrong concept desc
                # Reasoning
                'good_reasoning': extract_reasoning(good_r.get('response', '')),
                'bad_reasoning':  extract_reasoning(bad_r.get('response', '')),
                'good_pred':      good_r.get('pred_name', solution_concept),
            })

    print(f"  Found {len(mismatches)} mismatches")
    return mismatches


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(mismatches: List[Dict[str, Any]], output_file: str,
                  dataset_name: str, model_type: str,
                  db_good: str, db_bad: str) -> None:

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name} — {db_good} ✓ vs {db_bad} ✗</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            min-height: 100vh;
        }}

        .container {{ max-width: 1700px; margin: 0 auto; }}

        /* ---- header ---- */
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            position: sticky; top: 20px; z-index: 100;
        }}
        .header h1 {{ color: #667eea; margin-bottom: 12px; }}
        .config-row {{
            display: flex; gap: 20px; flex-wrap: wrap;
            margin-top: 12px;
        }}
        .config-pill {{
            padding: 6px 14px; border-radius: 20px;
            font-size: 0.85em; font-weight: 600;
        }}
        .pill-good  {{ background: #d4edda; color: #155724; border: 2px solid #28a745; }}
        .pill-bad   {{ background: #f8d7da; color: #721c24; border: 2px solid #dc3545; }}
        .pill-model {{ background: #e8eaf6; color: #3949ab; border: 2px solid #7986cb; }}
        .stats {{
            background: #fff3cd; padding: 12px 18px;
            border-radius: 8px; margin-top: 14px;
            border-left: 4px solid #ffc107;
            font-weight: 600;
        }}

        /* ---- card ---- */
        .card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .card-header {{
            border-bottom: 3px solid #667eea;
            padding-bottom: 14px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-title {{ color: #667eea; font-size: 1.4em; font-weight: 700; }}
        .concept-badge {{
            background: #667eea; color: white;
            padding: 7px 14px; border-radius: 20px;
            font-size: 0.9em; font-weight: 600;
        }}

        /* ---- images ---- */
        .images-row {{
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
            margin: 22px 0;
        }}
        .image-container {{
            display: flex; flex-direction: column;
            align-items: center; flex: 1; min-width: 150px; max-width: 260px;
        }}
        .image-label {{
            font-weight: 700; font-size: 0.85em;
            text-transform: uppercase; letter-spacing: 0.5px;
            padding: 5px 12px; border-radius: 5px;
            margin-bottom: 8px; display: inline-block;
        }}
        .image-label.query      {{ background: #2196f3; color: white; }}
        .image-label.retrieved  {{ background: #9c27b0; color: white; }}
        .image-label.solution   {{ background: #28a745; color: white; }}
        .image-label.wrong-pick {{ background: #dc3545; color: white; }}
        .image-container img {{
            width: 100%; height: 200px; object-fit: contain;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            background: #f5f5f5; padding: 6px;
        }}
        .image-container.query-wrap img  {{ border: 4px solid #2196f3; }}
        .image-container.sol-wrap img    {{ border: 4px solid #28a745; }}
        .image-container.wrong-wrap img  {{ border: 4px solid #dc3545; }}
        .image-path {{
            font-size: 0.72em; color: #666;
            margin-top: 6px; font-family: monospace;
            word-break: break-all; text-align: center;
            max-width: 100%;
        }}

        /* ---- descriptions ---- */
        .desc-section {{
            margin: 22px 0;
            padding: 20px;
            background: #f3e5f5;
            border-radius: 10px;
            border-left: 4px solid #8e24aa;
        }}
        .desc-section-title {{
            font-weight: 700; font-size: 1.15em;
            color: #6a1b9a; margin-bottom: 16px;
        }}
        .desc-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 14px;
        }}
        .desc-box {{
            border-radius: 8px; padding: 14px;
            border: 2px solid;
        }}
        .desc-box.good-sol  {{ background: #e8f5e9; border-color: #43a047; }}
        .desc-box.bad-sol   {{ background: #e3f2fd; border-color: #1e88e5; }}
        .desc-box.bad-pred  {{ background: #fce4ec; border-color: #e53935; }}
        .desc-box-label {{
            font-weight: 700; font-size: 0.78em;
            text-transform: uppercase; letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .desc-box.good-sol .desc-box-label  {{ color: #2e7d32; }}
        .desc-box.bad-sol  .desc-box-label  {{ color: #1565c0; }}
        .desc-box.bad-pred .desc-box-label  {{ color: #b71c1c; }}
        .desc-box-text {{
            font-size: 0.92em; line-height: 1.65; color: #333;
        }}
        .desc-box-text em {{ color: #999; }}

        /* ---- predictions ---- */
        .predictions-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
            margin-top: 22px;
        }}
        .pred-box {{
            border-radius: 10px; padding: 20px; border: 2px solid;
        }}
        .pred-box.good {{ background: #d4edda; border-color: #28a745; }}
        .pred-box.bad  {{ background: #f8d7da; border-color: #dc3545; }}
        .pred-header {{
            font-weight: 700; font-size: 1.1em;
            margin-bottom: 12px;
            display: flex; align-items: center; gap: 10px;
        }}
        .pred-box.good .pred-header {{ color: #155724; }}
        .pred-box.bad  .pred-header {{ color: #721c24; }}
        .status-badge {{
            font-size: 0.75em; padding: 3px 10px;
            border-radius: 12px; font-weight: 700;
        }}
        .status-badge.correct {{ background: #28a745; color: white; }}
        .status-badge.wrong   {{ background: #dc3545; color: white; }}
        .pred-name {{
            font-size: 1.05em; font-weight: 600;
            margin-bottom: 12px;
        }}
        .reasoning-box {{
            background: rgba(255,255,255,0.65);
            padding: 12px; border-radius: 7px;
            font-size: 0.9em; line-height: 1.65;
        }}
        .reasoning-label {{
            font-weight: 700; font-size: 0.8em;
            text-transform: uppercase; letter-spacing: 0.4px;
            margin-bottom: 6px; color: #555;
        }}

        /* ---- scroll-to-top ---- */
        .scroll-top {{
            position: fixed; bottom: 28px; right: 28px;
            background: #667eea; color: white;
            width: 48px; height: 48px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.4em; cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            opacity: 0; pointer-events: none;
            transition: opacity 0.3s, transform 0.3s;
        }}
        .scroll-top.visible {{ opacity: 1; pointer-events: all; }}
        .scroll-top:hover {{ transform: translateY(-4px); }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>🔍 {dataset_name} — Captioning Comparison</h1>
        <p style="color:#555; margin-top:6px;">
            Instances where <strong>{db_good}</strong> descriptions are correct
            but <strong>{db_bad}</strong> descriptions are wrong.
        </p>
        <div class="config-row">
            <span class="config-pill pill-model">Model: {model_type}</span>
            <span class="config-pill pill-good">✓ Good DB: {db_good}</span>
            <span class="config-pill pill-bad">✗ Bad DB: {db_bad}</span>
        </div>
        <div class="stats">Total mismatches: {len(mismatches)}</div>
    </div>

    <div id="cards">
"""

    for idx, m in enumerate(mismatches, 1):
        solution   = html_escape(m['solution'])
        bad_pred   = html_escape(m['bad_pred'])
        good_pred  = html_escape(m['good_pred'])
        concept    = html_escape(m['concept'])

        desc_good_sol  = html_escape(m['desc_good_sol'])  or '<em>not found</em>'
        desc_bad_sol   = html_escape(m['desc_bad_sol'])   or '<em>not found</em>'
        desc_bad_pred  = html_escape(m['desc_bad_pred'])  or '<em>not found</em>'

        good_reasoning = html_escape(m['good_reasoning']) or '<em>—</em>'
        bad_reasoning  = html_escape(m['bad_reasoning'])  or '<em>—</em>'

        html += f"""
    <div class="card">
        <div class="card-header">
            <div class="card-title">#{idx}</div>
            <div class="concept-badge">GT concept: {concept}</div>
        </div>

        <!-- Images -->
        <div class="images-row">
            <div class="image-container query-wrap">
                <div class="image-label query">📷 Query</div>
                <img src="{m['query_image']}" alt="query" loading="lazy">
                <div class="image-path">{m['query_image']}</div>
            </div>
"""
        for ret_idx, ret_img in enumerate(m['ret_images'][:3], 1):
            is_sol  = m['solution'] and m['solution'].lower() in ret_img.lower()
            is_bad  = m['bad_pred'] and m['bad_pred'].lower() in ret_img.lower() and not is_sol
            wrap_cls  = 'sol-wrap' if is_sol else ('wrong-wrap' if is_bad else '')
            label_cls = 'solution' if is_sol else ('wrong-pick' if is_bad else 'retrieved')
            label_txt = ('✓ GT' if is_sol else ('✗ Wrong pick' if is_bad else f'Retrieved #{ret_idx}'))
            html += f"""
            <div class="image-container {wrap_cls}">
                <div class="image-label {label_cls}">{label_txt}</div>
                <img src="{ret_img}" alt="ret {ret_idx}" loading="lazy">
                <div class="image-path">{ret_img}</div>
            </div>
"""

        html += f"""
        </div>

        <!-- Descriptions -->
        <div class="desc-section">
            <div class="desc-section-title">📖 Speaker Descriptions</div>
            <div class="desc-grid">
                <div class="desc-box good-sol">
                    <div class="desc-box-label">✓ {db_good} — GT concept: {solution}</div>
                    <div class="desc-box-text">{desc_good_sol}</div>
                </div>
                <div class="desc-box bad-sol">
                    <div class="desc-box-label">{db_bad} — GT concept: {solution}</div>
                    <div class="desc-box-text">{desc_bad_sol}</div>
                </div>
                <div class="desc-box bad-pred">
                    <div class="desc-box-label">✗ {db_bad} — wrong pick: {bad_pred}</div>
                    <div class="desc-box-text">{desc_bad_pred}</div>
                </div>
            </div>
        </div>

        <!-- Predictions + Reasoning -->
        <div class="predictions-grid">
            <div class="pred-box good">
                <div class="pred-header">
                    ✓ {db_good}
                    <span class="status-badge correct">CORRECT</span>
                </div>
                <div class="pred-name">Predicted: {good_pred}</div>
                <div class="reasoning-label">Reasoning</div>
                <div class="reasoning-box">{good_reasoning}</div>
            </div>
            <div class="pred-box bad">
                <div class="pred-header">
                    ✗ {db_bad}
                    <span class="status-badge wrong">WRONG</span>
                </div>
                <div class="pred-name">Predicted: {bad_pred}</div>
                <div class="reasoning-label">Reasoning</div>
                <div class="reasoning-box">{bad_reasoning}</div>
            </div>
        </div>
    </div>
"""

    html += """
    </div><!-- #cards -->
</div><!-- .container -->

<div class="scroll-top" id="scrollTop" onclick="window.scrollTo({top:0,behavior:'smooth'})">↑</div>

<script>
    window.addEventListener('scroll', () => {
        document.getElementById('scrollTop').classList.toggle('visible', window.pageYOffset > 300);
    });
</script>
</body>
</html>
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  HTML written to {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare captioning results between two description databases.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_captioning_with_descriptions.py --dataset MyVLM
  python compare_captioning_with_descriptions.py --dataset YoLLaVA --seed 42
  python compare_captioning_with_descriptions.py --dataset MyVLM \\
      --db_good sp_concise_soft_gated --db_bad original_7b --limit 50
"""
    )
    parser.add_argument('--dataset', required=True,
                        choices=['MyVLM', 'YoLLaVA', 'PerVA'],
                        help='Dataset name')
    parser.add_argument('--model', default='original_7b',
                        help='Model type (default: original_7b)')
    parser.add_argument('--db_good', default='sp_concise_soft_gated',
                        help='DB type that produces correct results (default: sp_concise_soft_gated)')
    parser.add_argument('--db_bad', default='original_7b',
                        help='DB type that produces wrong results (default: original_7b)')
    parser.add_argument('--seed', type=int, default=23,
                        help='Seed for results and database paths (default: 23)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of mismatches in HTML (default: all)')
    args = parser.parse_args()

    results_path = Path('results') / args.dataset / 'all'
    if not results_path.exists():
        print(f"Error: results path not found: {results_path}")
        return

    # Load both description databases
    print(f"Loading databases (seed={args.seed})...")
    db_good_lookup = load_database(args.dataset, args.db_good, args.seed)
    db_bad_lookup  = load_database(args.dataset, args.db_bad,  args.seed)
    print(f"  {args.db_good}: {len(db_good_lookup)} concepts")
    print(f"  {args.db_bad}:  {len(db_bad_lookup)} concepts")

    mismatches = find_mismatches(
        args.dataset, results_path,
        args.model, args.db_good, args.db_bad,
        db_good_lookup, db_bad_lookup,
        args.seed,
    )

    if not mismatches:
        print("No mismatches found.")
        return

    display = mismatches[:args.limit] if args.limit else mismatches

    html_path = (Path('html') /
                 f'{args.dataset}_{args.model}_{args.db_good}_vs_{args.db_bad}_seed{args.seed}.html')
    generate_html(display, str(html_path),
                  args.dataset, args.model, args.db_good, args.db_bad)

    print(f"\nSummary")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Model:         {args.model}")
    print(f"  Good DB:       {args.db_good}")
    print(f"  Bad DB:        {args.db_bad}")
    print(f"  Seed:          {args.seed}")
    print(f"  Mismatches:    {len(mismatches)}")
    print(f"  In HTML:       {len(display)}")
    print(f"  Output:        {html_path}")


if __name__ == '__main__':
    main()
