
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: 25.0**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **12** different stimulus strengths:
5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
| Pearson Correlation | 25.0 | 0.5639 |
| R-squared | 25.0 | 0.3180 |
| MAE | 25.0 | 0.1715 |
| RMSE | 25.0 | 0.2191 |
| Binary Accuracy | 25.0 | 0.6973 |
| Binary Precision | 25.0 | 0.6826 |
| Binary Recall | 100.0 | 0.6987 |


**Strength 25.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.5318 | 0.2828 | 0.1781 | 0.2240 | 0.6858 | 0.6794 | 0.5799 |
| 10.0 | 0.5529 | 0.3057 | 0.1745 | 0.2206 | 0.6922 | 0.6766 | 0.6151 |
| 20.0 | 0.5588 | 0.3122 | 0.1725 | 0.2199 | 0.6948 | 0.6768 | 0.6251 |
| 25.0 | 0.5639 | 0.3180 | 0.1715 | 0.2191 | 0.6973 | 0.6826 | 0.6296 |
| 50.0 | 0.5467 | 0.2989 | 0.1757 | 0.2235 | 0.6892 | 0.6599 | 0.6759 |
| 100.0 | 0.3724 | 0.1387 | 0.2082 | 0.2635 | 0.6272 | 0.5919 | 0.6987 |
| 250.0 | 0.2968 | 0.0881 | 0.2235 | 0.2847 | 0.6019 | 0.5701 | 0.6955 |
| 500.0 | 0.2724 | 0.0742 | 0.2278 | 0.2912 | 0.5956 | 0.5650 | 0.6891 |
| 1000.0 | 0.2680 | 0.0718 | 0.2285 | 0.2925 | 0.5947 | 0.5648 | 0.6839 |
| 1500.0 | 0.2678 | 0.0717 | 0.2288 | 0.2930 | 0.5954 | 0.5667 | 0.6793 |
| 2000.0 | 0.2693 | 0.0725 | 0.2288 | 0.2932 | 0.5986 | 0.5701 | 0.6783 |
| 3000.0 | 0.2692 | 0.0725 | 0.2291 | 0.2938 | 0.6010 | 0.5732 | 0.6723 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 25.0 (r = 0.5639)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **25.0**
  - Achieves best Pearson correlation: 0.5639
  - R²: 0.3180

- **Alternative option**: Stimulus strength **20.0**
  - Pearson correlation: 0.5588


### For Specific Applications:

- **Binary classification tasks**: Strength 25.0 (accuracy = 0.6973)
- **Minimizing absolute error**: Strength 25.0 (MAE = 0.1715)
- **High precision required**: Strength 25.0 (precision = 0.6826)
- **High recall required**: Strength 100.0 (recall = 0.6987)


---

## Files Generated

### Individual Strength Results:

#### Strength 5.0
- Location: `strength_5.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 10.0
- Location: `strength_10.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 20.0
- Location: `strength_20.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 25.0
- Location: `strength_25.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 50.0
- Location: `strength_50.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 100.0
- Location: `strength_100.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 250.0
- Location: `strength_250.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 500.0
- Location: `strength_500.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 1000.0
- Location: `strength_1000.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 1500.0
- Location: `strength_1500.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 2000.0
- Location: `strength_2000.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 3000.0
- Location: `strength_3000.0/`
- Files: predictions, correlations, plots, and summary report


### Comparison Results:
- `strength_comparison_metrics.csv`: Complete metrics table
- `best_strength_by_criterion.csv`: Best strength for each metric
- `strength_composite_scores.csv`: Normalized composite performance scores
- `best_strength_per_drug.csv`: Optimal strength for each drug
- `best_strength_per_tf.csv`: Optimal strength for each TF
- `metrics_comparison.png`: Visual comparison of all metrics
- `strength_ranking.png`: Overall performance ranking
- `best_strength_per_drug.png`: Distribution of best strengths across drugs
- `best_strength_per_tf.png`: Distribution of best strengths across TFs
- `dose_selection_comparison_across_strengths.png`: Panel view of dose selection per strength
- `dose_selection_stacked_comparison.png`: Stacked bar chart comparing dose selections

---

## Conclusion

Based on comprehensive analysis across multiple performance metrics, **stimulus strength {overall_best:.1f}** 
is recommended for LODO validation. This strength provides the best balance of correlation, accuracy, 
and predictive performance across drugs and transcription factors.

For specific use cases or if computational resources are limited, consider the alternative strengths 
suggested in the recommendations section above.

---

*Generated by Multi-Strength LODO Analysis Pipeline*
