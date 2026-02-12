#!/usr/bin/env python3
"""
entropy_analysis.py

Analyze entropy and margin values from model predictions, correlating with
correct/incorrect predictions. Generates comprehensive reports and visualizations.

Usage:
    python entropy_analysis.py --dataset YoLLaVA --model_type lora_7b_grpo_res_fixed_no_reco_db
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import roc_curve, auc, precision_recall_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger(__name__)

import sys
sys.path.append('src')
from analysis.plot_funcs import *
from inference_utils.common import YOLLAVA_CATEGORY_MAP, MYVLM_CATEGORY_MAP
# ============================================================================
# Data Loading
# ============================================================================

def load_descriptions(dataset: str, seed: int = 23) -> Dict[str, Any]:
    """
    Load concept descriptions from the database JSON file.

    Args:
        dataset: Dataset name (YoLLaVA or MyVLM)
        seed: Random seed

    Returns:
        Dictionary containing concept descriptions
    """
    desc_path = Path(f"outputs/{dataset}/all/seed_{seed}/database_original_7b.json")
    if not desc_path.exists():
        raise FileNotFoundError(f"Description file not found: {desc_path}")

    with open(desc_path, 'r') as f:
        data = json.load(f)

    LOG.info(f"Loaded descriptions from {desc_path}")
    return data


def get_concept_description(desc_data: Dict[str, Any], concept_name: str) -> str:
    """
    Extract concept description from the description data.

    Args:
        desc_data: Description data dictionary
        concept_name: Name of the concept

    Returns:
        Combined description string
    """
    try:
        concept_info = desc_data['concept_dict'][f'<{concept_name}>']['info']
        general = concept_info.get('general', [''])[0]
        distinct = concept_info.get('distinct features', [''])[0]
        return f"{general}. {distinct}".strip()
    except (KeyError, IndexError) as e:
        LOG.warning(f"Could not find description for concept '{concept_name}': {e}")
        return ""


def load_concept_results(
    dataset: str,
    category: str,
    concept: str,
    model_type: str,
    db_type: str,
    k_retrieval: int,
    seed: int
) -> Dict[str, Any]:
    """
    Load analysis results for a specific concept.

    Args:
        dataset: Dataset name
        category: Category name
        concept: Concept name
        model_type: Model type
        db_type: Database type
        k_retrieval: Number of retrieved references
        seed: Random seed

    Returns:
        Results dictionary
    """
    results_path = Path(f"results/{dataset}/{category}/{concept}/seed_{seed}/"f"results_model_{model_type}_db_{db_type}_k_{k_retrieval}_analysis.json")
    if not results_path.exists():
        LOG.warning(f"Results file not found: {results_path}")
        return None

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data


# ============================================================================
# Metadata Accumulation
# ============================================================================

def accumulate_metadata(
    results: Dict[str, Any],
    desc_data: Dict[str, Any],
    metadata: Dict[str, Any]
):
    """
    Accumulate metadata from results entries.

    Args:
        results: Results dictionary for a concept
        desc_data: Description data
        metadata: Metadata accumulator dictionary (modified in place)
    """
    if results is None or 'results' not in results:
        return

    for entry in results['results']:
        # Extract confidence data
        confidence = entry.get('confidence')
        if confidence is None:
            LOG.warning(f"No confidence data for entry: {entry.get('image_path', 'unknown')}")
            continue

        entropy = confidence.get('entropy')
        margin = confidence.get('margin')
        probabilities = confidence.get('probabilities', {})
        predicted_letter = confidence.get('predicted_letter')

        if entropy is None or margin is None:
            continue

        # Get predicted probability
        pred_prob = probabilities.get(predicted_letter, 0.0) if predicted_letter else 0.0

        # Extract prediction info
        correct = entry.get('correct', False)
        solution = entry.get('solution', '')
        pred_name = entry.get('pred_name', '')

        # Get concept description
        if 'ret_paths' in entry and entry['ret_paths']:
            concept_name = entry['ret_paths'][0].split('/')[-2]
            description = get_concept_description(desc_data, concept_name)
        else:
            concept_name = solution
            description = get_concept_description(desc_data, concept_name)

        # Accumulate data
        if correct:
            metadata['correct']['entropies'].append(entropy)
            metadata['correct']['margins'].append(margin)
            metadata['correct']['probabilities'].append(pred_prob)
            metadata['correct']['concept_names'].append(solution)
        else:
            metadata['incorrect']['entropies'].append(entropy)
            metadata['incorrect']['margins'].append(margin)
            metadata['incorrect']['probabilities'].append(pred_prob)
            metadata['incorrect']['concept_names'].append(solution)
            metadata['incorrect']['solutions'].append(solution)
            metadata['incorrect']['predictions'].append(pred_name)

            # Store full entry details for failure case analysis
            metadata['incorrect']['full_entries'].append({
                'image_path': entry.get('image_path', ''),
                'ret_paths': entry.get('ret_paths', []),
                'solution': solution,
                'pred_name': pred_name,
                'confidence': confidence,
                'entropy': entropy,
                'margin': margin,
                'response': entry.get('response', ''),
            })

        # Per-concept accumulation
        if solution not in metadata['per_concept']:
            metadata['per_concept'][solution] = {
                'total': 0,
                'correct': 0,
                'entropies_correct': [],
                'entropies_incorrect': [],
                'margins_correct': [],
                'margins_incorrect': [],
                'description': description,
            }

        metadata['per_concept'][solution]['total'] += 1
        if correct:
            metadata['per_concept'][solution]['correct'] += 1
            metadata['per_concept'][solution]['entropies_correct'].append(entropy)
            metadata['per_concept'][solution]['margins_correct'].append(margin)
        else:
            metadata['per_concept'][solution]['entropies_incorrect'].append(entropy)
            metadata['per_concept'][solution]['margins_incorrect'].append(margin)


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistical metrics for a list of values.

    Args:
        values: List of numerical values

    Returns:
        Dictionary with statistics
    """
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'p25': 0.0,
            'p75': 0.0,
            'p95': 0.0,
        }

    values = np.array(values)
    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p95': float(np.percentile(values, 95)),
    }


def analyze_overall(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform overall analysis on correct vs incorrect predictions.

    Args:
        metadata: Accumulated metadata

    Returns:
        Analysis results
    """
    analysis = {
        'correct': {
            'entropy': compute_statistics(metadata['correct']['entropies']),
            'margin': compute_statistics(metadata['correct']['margins']),
            'probability': compute_statistics(metadata['correct']['probabilities']),
        },
        'incorrect': {
            'entropy': compute_statistics(metadata['incorrect']['entropies']),
            'margin': compute_statistics(metadata['incorrect']['margins']),
            'probability': compute_statistics(metadata['incorrect']['probabilities']),
        },
        'total_predictions': (
            len(metadata['correct']['entropies']) +
            len(metadata['incorrect']['entropies'])
        ),
        'total_correct': len(metadata['correct']['entropies']),
        'total_incorrect': len(metadata['incorrect']['entropies']),
        'overall_accuracy': (
            len(metadata['correct']['entropies']) /
            (len(metadata['correct']['entropies']) + len(metadata['incorrect']['entropies']))
            if (len(metadata['correct']['entropies']) + len(metadata['incorrect']['entropies'])) > 0
            else 0.0
        ),
    }

    return analysis


def analyze_per_concept(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform per-concept analysis.

    Args:
        metadata: Accumulated metadata

    Returns:
        Per-concept analysis results
    """
    per_concept_analysis = {}

    for concept, data in metadata['per_concept'].items():
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0.0

        per_concept_analysis[concept] = {
            'total': data['total'],
            'correct': data['correct'],
            'incorrect': data['total'] - data['correct'],
            'accuracy': accuracy,
            'description': data['description'],
            'entropy_correct': compute_statistics(data['entropies_correct']),
            'entropy_incorrect': compute_statistics(data['entropies_incorrect']),
            'margin_correct': compute_statistics(data['margins_correct']),
            'margin_incorrect': compute_statistics(data['margins_incorrect']),
        }

    return per_concept_analysis


def identify_failure_modes(metadata: Dict[str, Any], entropy_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Identify different failure modes based on entropy and correctness.

    Args:
        metadata: Accumulated metadata
        entropy_threshold: Threshold to separate high/low entropy

    Returns:
        Failure mode analysis
    """
    failure_modes = {
        'high_confidence_correct': 0,  # Low entropy, correct
        'low_confidence_correct': 0,   # High entropy, correct (got lucky)
        'high_confidence_wrong': 0,    # Low entropy, incorrect (confidently wrong)
        'low_confidence_wrong': 0,     # High entropy, incorrect
    }

    for entropy in metadata['correct']['entropies']:
        if entropy < entropy_threshold:
            failure_modes['high_confidence_correct'] += 1
        else:
            failure_modes['low_confidence_correct'] += 1

    for entropy in metadata['incorrect']['entropies']:
        if entropy < entropy_threshold:
            failure_modes['high_confidence_wrong'] += 1
        else:
            failure_modes['low_confidence_wrong'] += 1

    return failure_modes


def extract_failure_cases(args, metadata: Dict[str, Any], entropy_threshold: float = 0.05) -> Dict[str, List[Dict]]:
    """
    Extract and categorize failure cases.

    Args:
        metadata: Accumulated metadata
        entropy_threshold: Threshold to separate high/low confidence

    Returns:
        Dictionary with high_confidence_wrong and low_confidence_wrong lists
    """
    failure_cases = {
        'high_confidence_wrong': [],
        'low_confidence_wrong': [],
    }
    descs = load_descriptions(args.dataset, args.seed)
    for entry in metadata['incorrect']['full_entries']:
        entropy = entry['entropy']
        ret_concepts = [item.split('/')[-2] for item in entry['ret_paths']] if entry['ret_paths'] else []
        entry["concept_descriptions"] = {c:get_concept_description(descs, c) for c in ret_concepts}
        if entropy < entropy_threshold:
            failure_cases['high_confidence_wrong'].append(entry)
        else:
            failure_cases['low_confidence_wrong'].append(entry)

    # Sort by entropy (ascending for high confidence, descending for low confidence)
    failure_cases['high_confidence_wrong'].sort(key=lambda x: x['entropy'])
    failure_cases['low_confidence_wrong'].sort(key=lambda x: x['entropy'], reverse=True)

    LOG.info(f"Extracted {len(failure_cases['high_confidence_wrong'])} high confidence errors")
    LOG.info(f"Extracted {len(failure_cases['low_confidence_wrong'])} low confidence errors")

    return failure_cases


def save_failure_cases(args, failure_cases: Dict[str, List[Dict]], output_dir: Path):
    """
    Save failure cases to separate JSON files.

    Args:
        failure_cases: Dictionary with categorized failure cases
        output_dir: Output directory
    """
    for case_type, cases in failure_cases.items():
        output_path = output_dir / f"{case_type}.json"

        # Format for better readability
        formatted_cases = []
        for idx, case in enumerate(cases, 1):
            formatted_case = {
                'case_id': idx,
                'image_path': case['image_path'],
                'category': YOLLAVA_CATEGORY_MAP[case['solution']] if args.dataset == 'YoLLaVA' else MYVLM_CATEGORY_MAP[case['solution']],
                'ret_paths': case['ret_paths'],
                'ret_descriptions': case.get('concept_descriptions', {}),
                'solution': case['solution'],
                'predicted': case['pred_name'],
                'entropy': round(case['entropy'], 4),
                'margin': round(case['margin'], 4),
                'probabilities': {
                    k: round(v, 4) for k, v in case['confidence'].get('probabilities', {}).items()
                },
                'response': case['response'],
            }
            formatted_cases.append(formatted_case)

        with open(output_path, 'w') as f:
            json.dump(formatted_cases, f, indent=2)

        LOG.info(f"Saved {len(cases)} cases to {output_path}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    metadata: Dict[str, Any],
    overall_analysis: Dict[str, Any],
    per_concept_analysis: Dict[str, Any],
    failure_modes: Dict[str, int],
    roc_auc: float,
    output_dir: Path,
    args: argparse.Namespace
):
    """
    Generate comprehensive JSON report.
    """
    report = {
        'metadata': {
            'dataset': args.dataset,
            'model_type': args.model_type,
            'db_type': args.db_type,
            'seed': args.seed,
            'category': args.category,
            'k_retrieval': args.k_retrieval,
        },
        'overall_analysis': overall_analysis,
        'per_concept_analysis': per_concept_analysis,
        'failure_modes': failure_modes,
        'discriminative_power': {
            'roc_auc': roc_auc,
            'interpretation': (
                'AUC > 0.7: Good discrimination' if roc_auc > 0.7 else
                'AUC > 0.6: Moderate discrimination' if roc_auc > 0.6 else
                'AUC > 0.5: Poor discrimination' if roc_auc > 0.5 else
                'AUC = 0.5: Random (no discrimination)'
            )
        }
    }

    report_path = output_dir / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    LOG.info(f"Saved analysis report to {report_path}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Entropy and confidence analysis")

    parser.add_argument('--dataset', type=str, required=True, choices=['YoLLaVA', 'MyVLM'],
                        help='Dataset name')
    parser.add_argument('--model_type', type=str, default='lora_7b_grpo_res_fixed_no_reco',
                        help='Model type')
    parser.add_argument('--db_type', type=str, default='original_7b',
                        help='Database type')
    parser.add_argument('--seed', type=int, default=23,
                        help='Random seed')
    parser.add_argument('--category', type=str, default='all',
                        help='Category name')
    parser.add_argument('--k_retrieval', type=int, default=3,
                        help='Number of retrieved references')
    parser.add_argument('--entropy_threshold', type=float, default=0.05,
                        help='Entropy threshold for failure mode classification')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(f'results/analysis/{args.dataset}_{args.model_type}_db_{args.db_type}')
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Starting analysis for dataset: {args.dataset}, model: {args.model_type}")

    # Load descriptions
    desc_data = load_descriptions(args.dataset, args.seed)

    # Get list of all concepts
    concepts = list(desc_data['concept_dict'].keys())
    LOG.info(f"Found {len(concepts)} concepts")

    # Initialize metadata accumulator
    metadata = {
        'correct': {
            'entropies': [],
            'margins': [],
            'probabilities': [],
            'concept_names': [],
        },
        'incorrect': {
            'entropies': [],
            'margins': [],
            'probabilities': [],
            'concept_names': [],
            'solutions': [],
            'predictions': [],
            'full_entries': [],  # Store full entry details for failure analysis
        },
        'per_concept': {}
    }

    # Accumulate data from all concepts
    for concept in concepts:
        LOG.info(f"Processing concept: {concept}")
        results = load_concept_results(
            args.dataset, args.category, concept.replace('<', '').replace('>', ''),
            args.model_type, args.db_type, args.k_retrieval, args.seed
        )
        if results is not None:
            accumulate_metadata(results, desc_data, metadata)

    LOG.info(f"Accumulated data for {len(metadata['correct']['entropies']) + len(metadata['incorrect']['entropies'])} predictions")

    # Perform analysis
    LOG.info("Performing overall analysis...")
    overall_analysis = analyze_overall(metadata)

    LOG.info("Performing per-concept analysis...")
    per_concept_analysis = analyze_per_concept(metadata)

    LOG.info("Identifying failure modes...")
    failure_modes = identify_failure_modes(metadata, args.entropy_threshold)

    # Extract and save failure cases
    LOG.info("Extracting failure cases...")
    failure_cases = extract_failure_cases(args, metadata, args.entropy_threshold)
    save_failure_cases(args, failure_cases, output_dir)

    # Generate visualizations
    LOG.info("Generating visualizations...")
    plot_entropy_distributions(metadata, output_dir)
    plot_margin_distributions(metadata, output_dir)
    plot_entropy_vs_margin(metadata, output_dir)
    roc_auc = plot_roc_curve(metadata, output_dir)
    plot_per_concept_analysis(metadata, per_concept_analysis, output_dir)
    plot_failure_modes(failure_modes, output_dir)

    # Generate report
    LOG.info("Generating report...")
    generate_report(
        metadata, overall_analysis, per_concept_analysis,
        failure_modes, roc_auc, output_dir, args
    )

    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Model: {args.model_type}")
    print(f"Total Predictions: {overall_analysis['total_predictions']}")
    print(f"Overall Accuracy: {overall_analysis['overall_accuracy']:.2%}")
    print(f"\nCorrect Predictions:")
    print(f"  Count: {overall_analysis['total_correct']}")
    print(f"  Entropy: {overall_analysis['correct']['entropy']['mean']:.4f} ± {overall_analysis['correct']['entropy']['std']:.4f}")
    print(f"  Margin: {overall_analysis['correct']['margin']['mean']:.4f} ± {overall_analysis['correct']['margin']['std']:.4f}")
    print(f"\nIncorrect Predictions:")
    print(f"  Count: {overall_analysis['total_incorrect']}")
    print(f"  Entropy: {overall_analysis['incorrect']['entropy']['mean']:.4f} ± {overall_analysis['incorrect']['entropy']['std']:.4f}")
    print(f"  Margin: {overall_analysis['incorrect']['margin']['mean']:.4f} ± {overall_analysis['incorrect']['margin']['std']:.4f}")
    print(f"\nROC AUC (Entropy as Error Predictor): {roc_auc:.3f}")
    print(f"\nFailure Modes:")
    for mode, count in failure_modes.items():
        print(f"  {mode}: {count}")
    print(f"\nFailure Case Files:")
    print(f"  High confidence wrong: {len(failure_cases['high_confidence_wrong'])} cases saved to high_confidence_wrong.json")
    print(f"  Low confidence wrong: {len(failure_cases['low_confidence_wrong'])} cases saved to low_confidence_wrong.json")
    print(f"\nOutputs saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
