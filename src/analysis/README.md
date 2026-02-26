# Entropy Analysis

Comprehensive analysis of model confidence (entropy, margin) and its correlation with prediction correctness.

## Usage

```bash
# Analyze YoLLaVA dataset with default parameters
python entropy_analysis.py --dataset YoLLaVA

# Analyze MyVLM dataset
python entropy_analysis.py --dataset MyVLM

# Custom parameters
python entropy_analysis.py \
    --dataset YoLLaVA \
    --model_type lora_7b_grpo_res_fixed_no_reco_db \
    --db_type original_7b \
    --seed 23 \
    --category all \
    --k_retrieval 3 \
    --entropy_threshold 0.05
```

## Arguments

- `--dataset`: Dataset name (YoLLaVA or MyVLM) **[Required]**
- `--model_type`: Model type (default: "lora_7b_grpo_res_fixed_no_reco_db")
- `--db_type`: Database type (default: "original_7b")
- `--seed`: Random seed (default: 23)
- `--category`: Category name (default: "all")
- `--k_retrieval`: Number of retrieved references (default: 3)
- `--entropy_threshold`: Threshold for failure mode classification (default: 0.05)

## Outputs

All outputs are saved to: `results/analysis/{dataset}_{model_type}/`

### Generated Files

1. **analysis_report.json**: Comprehensive JSON report with:
   - Overall statistics (correct vs incorrect)
   - Per-concept breakdown
   - Failure mode counts
   - ROC AUC score

2. **Visualizations (PNG files)**:

   **entropy_distributions.png**: 4-panel plot showing:
   - Overlapping histograms (correct vs incorrect)
   - Box plots
   - Violin plots
   - Cumulative Distribution Functions (CDF)

   **margin_distributions.png**: Same 4-panel layout for margin values

   **entropy_vs_margin.png**: Scatter plot showing relationship between entropy and margin, colored by correctness

   **roc_curve.png**: ROC curve showing how well entropy predicts errors
   - AUC > 0.7: Good discrimination
   - AUC > 0.6: Moderate discrimination
   - AUC ≤ 0.5: Poor discrimination

   **per_concept_analysis.png**: 2-panel bar charts showing:
   - Per-concept accuracy (sorted)
   - Per-concept average entropy

   **failure_modes.png**: Bar chart showing distribution of:
   - High confidence correct (low entropy, correct)
   - Low confidence correct (high entropy, correct - "got lucky")
   - High confidence wrong (low entropy, wrong - "confidently wrong")
   - Low confidence wrong (high entropy, wrong)

## Interpreting Results

### Entropy
- **Low entropy (< 0.02)**: Model is very confident (~98%+ probability on one answer)
- **High entropy (> 0.1)**: Model is uncertain (more uniform distribution)
- For 3 options, maximum entropy ≈ 1.1 (uniform distribution)

### Margin
- **High margin (> 0.9)**: Large gap between top two probabilities (confident)
- **Low margin (< 0.5)**: Small gap (uncertain between top options)

### Key Insights to Look For

1. **Entropy Gap**: How different is entropy between correct and incorrect?
   - Small gap → Model is overconfident (can't distinguish hard cases)
   - Large gap → Model has good calibration

2. **High Confidence Errors**:
   - If many predictions fall in "high confidence wrong", the model is poorly calibrated
   - These are the most problematic cases (wrong but confident)

3. **ROC AUC**:
   - Measures if entropy can predict errors
   - Low AUC → Entropy doesn't help identify mistakes

4. **Per-Concept Patterns**:
   - Which concepts have low accuracy?
   - Do low-accuracy concepts have higher entropy?
   - Identify concepts that need better descriptions

## Example Output

```
================================================================================
ANALYSIS SUMMARY
================================================================================

Dataset: YoLLaVA
Model: lora_7b_grpo_res_fixed_no_reco_db
Total Predictions: 1500
Overall Accuracy: 85.20%

Correct Predictions:
  Count: 1278
  Entropy: 0.0152 ± 0.0089
  Margin: 0.9712 ± 0.0234

Incorrect Predictions:
  Count: 222
  Entropy: 0.0347 ± 0.0156
  Margin: 0.9534 ± 0.0312

ROC AUC (Entropy as Error Predictor): 0.682

Failure Modes:
  high_confidence_correct: 1245
  low_confidence_correct: 33
  high_confidence_wrong: 178
  low_confidence_wrong: 44

Outputs saved to: results/analysis/YoLLaVA_lora_7b_grpo_res_fixed_no_reco_db
================================================================================
```

## Dependencies

```bash
pip install numpy matplotlib seaborn scikit-learn pandas
```
