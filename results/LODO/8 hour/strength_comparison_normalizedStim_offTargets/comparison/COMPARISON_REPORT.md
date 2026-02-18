
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: 50.0**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **15** different stimulus strengths:
5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
| Pearson Correlation | 50.0 | 0.5299 |
| R-squared | 50.0 | 0.2807 |
| MAE | 50.0 | 0.1807 |
| RMSE | 50.0 | 0.2277 |
| Binary Accuracy | 50.0 | 0.6838 |
| Binary Precision | 50.0 | 0.6862 |
| Binary Recall | 100.0 | 0.7602 |


**Strength 50.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.4361 | 0.1901 | 0.1963 | 0.2412 | 0.6490 | 0.6527 | 0.6704 |
| 10.0 | 0.4996 | 0.2496 | 0.1859 | 0.2316 | 0.6752 | 0.6750 | 0.7096 |
| 20.0 | 0.5133 | 0.2635 | 0.1838 | 0.2296 | 0.6788 | 0.6787 | 0.7136 |
| 25.0 | 0.5229 | 0.2734 | 0.1824 | 0.2279 | 0.6823 | 0.6841 | 0.7169 |
| 50.0 | 0.5299 | 0.2807 | 0.1807 | 0.2277 | 0.6838 | 0.6862 | 0.7392 |
| 100.0 | 0.4351 | 0.1893 | 0.1943 | 0.2495 | 0.6544 | 0.6502 | 0.7602 |
| 250.0 | 0.3753 | 0.1408 | 0.2038 | 0.2646 | 0.6356 | 0.6358 | 0.7535 |
| 500.0 | 0.3446 | 0.1187 | 0.2095 | 0.2728 | 0.6251 | 0.6276 | 0.7414 |
| 1000.0 | 0.3229 | 0.1042 | 0.2144 | 0.2788 | 0.6209 | 0.6256 | 0.7316 |
| 1500.0 | 0.3147 | 0.0990 | 0.2166 | 0.2814 | 0.6185 | 0.6246 | 0.7250 |
| 2000.0 | 0.3118 | 0.0972 | 0.2176 | 0.2827 | 0.6192 | 0.6262 | 0.7210 |
| 3000.0 | 0.3074 | 0.0945 | 0.2189 | 0.2845 | 0.6193 | 0.6280 | 0.7150 |
| 4000.0 | 0.3040 | 0.0924 | 0.2201 | 0.2858 | 0.6179 | 0.6277 | 0.7107 |
| 5000.0 | 0.3025 | 0.0915 | 0.2208 | 0.2865 | 0.6171 | 0.6277 | 0.7070 |
| 6000.0 | 0.3022 | 0.0914 | 0.2212 | 0.2869 | 0.6171 | 0.6278 | 0.7057 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 50.0 (r = 0.5299)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **50.0**
  - Achieves best Pearson correlation: 0.5299
  - R²: 0.2807

- **Alternative option**: Stimulus strength **25.0**
  - Pearson correlation: 0.5229


### For Specific Applications:

- **Binary classification tasks**: Strength 50.0 (accuracy = 0.6838)
- **Minimizing absolute error**: Strength 50.0 (MAE = 0.1807)
- **High precision required**: Strength 50.0 (precision = 0.6862)
- **High recall required**: Strength 100.0 (recall = 0.7602)


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

#### Strength 4000.0
- Location: `strength_4000.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 5000.0
- Location: `strength_5000.0/`
- Files: predictions, correlations, plots, and summary report

#### Strength 6000.0
- Location: `strength_6000.0/`
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
