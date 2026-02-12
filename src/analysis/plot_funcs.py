import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# ============================================================================
# Visualization
# ============================================================================

def plot_entropy_distributions(metadata: Dict[str, Any], output_dir: Path):
    """
    Plot entropy distributions for correct vs incorrect predictions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    correct_entropies = metadata['correct']['entropies']
    incorrect_entropies = metadata['incorrect']['entropies']

    # Overlapping histograms
    axes[0, 0].hist(correct_entropies, bins=50, alpha=0.6, label='Correct', color='green', density=True)
    axes[0, 0].hist(incorrect_entropies, bins=50, alpha=0.6, label='Incorrect', color='red', density=True)
    axes[0, 0].set_xlabel('Entropy')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Entropy Distribution: Correct vs Incorrect')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    axes[0, 1].boxplot([correct_entropies, incorrect_entropies], labels=['Correct', 'Incorrect'])
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].set_title('Entropy Box Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Violin plot
    data_for_violin = [
        {'Entropy': e, 'Prediction': 'Correct'} for e in correct_entropies
    ] + [
        {'Entropy': e, 'Prediction': 'Incorrect'} for e in incorrect_entropies
    ]

    import pandas as pd
    df = pd.DataFrame(data_for_violin)
    sns.violinplot(data=df, x='Prediction', y='Entropy', ax=axes[1, 0])
    axes[1, 0].set_title('Entropy Violin Plot')
    axes[1, 0].grid(True, alpha=0.3)

    # CDF plot
    correct_sorted = np.sort(correct_entropies)
    incorrect_sorted = np.sort(incorrect_entropies)
    correct_cdf = np.arange(1, len(correct_sorted) + 1) / len(correct_sorted)
    incorrect_cdf = np.arange(1, len(incorrect_sorted) + 1) / len(incorrect_sorted)

    axes[1, 1].plot(correct_sorted, correct_cdf, label='Correct', color='green', linewidth=2)
    axes[1, 1].plot(incorrect_sorted, incorrect_cdf, label='Incorrect', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Entropy')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Entropy CDF')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # LOG.info(f"Saved entropy distribution plots to {output_dir / 'entropy_distributions.png'}")


def plot_margin_distributions(metadata: Dict[str, Any], output_dir: Path):
    """
    Plot margin distributions for correct vs incorrect predictions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    correct_margins = metadata['correct']['margins']
    incorrect_margins = metadata['incorrect']['margins']

    # Overlapping histograms
    axes[0, 0].hist(correct_margins, bins=50, alpha=0.6, label='Correct', color='green', density=True)
    axes[0, 0].hist(incorrect_margins, bins=50, alpha=0.6, label='Incorrect', color='red', density=True)
    axes[0, 0].set_xlabel('Margin')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Margin Distribution: Correct vs Incorrect')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    axes[0, 1].boxplot([correct_margins, incorrect_margins], labels=['Correct', 'Incorrect'])
    axes[0, 1].set_ylabel('Margin')
    axes[0, 1].set_title('Margin Box Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Violin plot
    data_for_violin = [
        {'Margin': m, 'Prediction': 'Correct'} for m in correct_margins
    ] + [
        {'Margin': m, 'Prediction': 'Incorrect'} for m in incorrect_margins
    ]

    import pandas as pd
    df = pd.DataFrame(data_for_violin)
    sns.violinplot(data=df, x='Prediction', y='Margin', ax=axes[1, 0])
    axes[1, 0].set_title('Margin Violin Plot')
    axes[1, 0].grid(True, alpha=0.3)

    # CDF plot
    correct_sorted = np.sort(correct_margins)
    incorrect_sorted = np.sort(incorrect_margins)
    correct_cdf = np.arange(1, len(correct_sorted) + 1) / len(correct_sorted)
    incorrect_cdf = np.arange(1, len(incorrect_sorted) + 1) / len(incorrect_sorted)

    axes[1, 1].plot(correct_sorted, correct_cdf, label='Correct', color='green', linewidth=2)
    axes[1, 1].plot(incorrect_sorted, incorrect_cdf, label='Incorrect', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Margin')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Margin CDF')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'margin_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # LOG.info(f"Saved margin distribution plots to {output_dir / 'margin_distributions.png'}")


def plot_entropy_vs_margin(metadata: Dict[str, Any], output_dir: Path):
    """
    Plot entropy vs margin scatter plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    correct_entropies = metadata['correct']['entropies']
    correct_margins = metadata['correct']['margins']
    incorrect_entropies = metadata['incorrect']['entropies']
    incorrect_margins = metadata['incorrect']['margins']

    ax.scatter(correct_entropies, correct_margins, alpha=0.5, label='Correct', color='green', s=20)
    ax.scatter(incorrect_entropies, incorrect_margins, alpha=0.5, label='Incorrect', color='red', s=20)

    ax.set_xlabel('Entropy')
    ax.set_ylabel('Margin')
    ax.set_title('Entropy vs Margin')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_margin.png', dpi=300, bbox_inches='tight')
    plt.close()

    # LOG.info(f"Saved entropy vs margin plot to {output_dir / 'entropy_vs_margin.png'}")


def plot_roc_curve(metadata: Dict[str, Any], output_dir: Path):
    """
    Plot ROC curve to assess if entropy can discriminate correct from incorrect.
    """
    # Create binary labels and scores
    # Label: 1 for incorrect (positive class), 0 for correct
    # Score: entropy (higher entropy should predict incorrect)

    labels = [0] * len(metadata['correct']['entropies']) + [1] * len(metadata['incorrect']['entropies'])
    scores = metadata['correct']['entropies'] + metadata['incorrect']['entropies']

    if len(set(labels)) < 2:
        LOG.warning("Cannot create ROC curve: only one class present")
        return

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Entropy as Error Predictor')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # LOG.info(f"Saved ROC curve to {output_dir / 'roc_curve.png'} (AUC = {roc_auc:.3f})")

    return roc_auc


def plot_per_concept_analysis(metadata: Dict[str, Any], per_concept_analysis: Dict[str, Any], output_dir: Path):
    """
    Plot per-concept accuracy and entropy analysis.
    """
    concepts = list(per_concept_analysis.keys())
    accuracies = [per_concept_analysis[c]['accuracy'] for c in concepts]
    avg_entropies = [
        np.mean(metadata['per_concept'][c]['entropies_correct'] +
                metadata['per_concept'][c]['entropies_incorrect'])
        if (metadata['per_concept'][c]['entropies_correct'] +
            metadata['per_concept'][c]['entropies_incorrect'])
        else 0.0
        for c in concepts
    ]

    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    concepts_sorted = [concepts[i] for i in sorted_indices]
    accuracies_sorted = [accuracies[i] for i in sorted_indices]
    avg_entropies_sorted = [avg_entropies[i] for i in sorted_indices]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Accuracy bar chart
    axes[0].barh(concepts_sorted, accuracies_sorted, color='steelblue')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Per-Concept Accuracy')
    axes[0].set_xlim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='x')

    # Average entropy bar chart
    axes[1].barh(concepts_sorted, avg_entropies_sorted, color='coral')
    axes[1].set_xlabel('Average Entropy')
    axes[1].set_title('Per-Concept Average Entropy')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'per_concept_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # LOG.info(f"Saved per-concept analysis to {output_dir / 'per_concept_analysis.png'}")


def plot_failure_modes(failure_modes: Dict[str, int], output_dir: Path):
    """
    Plot failure mode distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(failure_modes.keys())
    values = list(failure_modes.values())
    colors = ['green', 'lightgreen', 'red', 'orange']

    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Count')
    ax.set_title('Failure Mode Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_modes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # LOG.info(f"Saved failure modes plot to {output_dir / 'failure_modes.png'}")
