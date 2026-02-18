
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
| Pearson Correlation | 50.0 | 0.5312 |
| R-squared | 50.0 | 0.2822 |
| MAE | 50.0 | 0.1802 |
| RMSE | 50.0 | 0.2268 |
| Binary Accuracy | 50.0 | 0.6814 |
| Binary Precision | 50.0 | 0.6809 |
| Binary Recall | 250.0 | 0.7587 |


**Strength 50.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.3394 | 0.1152 | 0.2092 | 0.2513 | 0.6160 | 0.6185 | 0.6460 |
| 10.0 | 0.4510 | 0.2034 | 0.1933 | 0.2386 | 0.6538 | 0.6560 | 0.6847 |
| 20.0 | 0.4971 | 0.2471 | 0.1864 | 0.2320 | 0.6707 | 0.6726 | 0.7039 |
| 25.0 | 0.5087 | 0.2588 | 0.1849 | 0.2303 | 0.6757 | 0.6783 | 0.7097 |
| 50.0 | 0.5312 | 0.2822 | 0.1802 | 0.2268 | 0.6814 | 0.6809 | 0.7375 |
| 100.0 | 0.4315 | 0.1862 | 0.1946 | 0.2494 | 0.6514 | 0.6478 | 0.7576 |
| 250.0 | 0.3901 | 0.1522 | 0.2017 | 0.2614 | 0.6413 | 0.6398 | 0.7587 |
| 500.0 | 0.3593 | 0.1291 | 0.2071 | 0.2691 | 0.6298 | 0.6313 | 0.7466 |
| 1000.0 | 0.3231 | 0.1044 | 0.2145 | 0.2788 | 0.6208 | 0.6256 | 0.7322 |
| 1500.0 | 0.3141 | 0.0987 | 0.2166 | 0.2814 | 0.6185 | 0.6243 | 0.7265 |
| 2000.0 | 0.3115 | 0.0970 | 0.2177 | 0.2826 | 0.6184 | 0.6251 | 0.7223 |
| 3000.0 | 0.3077 | 0.0947 | 0.2188 | 0.2843 | 0.6185 | 0.6270 | 0.7174 |
| 4000.0 | 0.3050 | 0.0930 | 0.2198 | 0.2855 | 0.6184 | 0.6275 | 0.7143 |
| 5000.0 | 0.3034 | 0.0920 | 0.2206 | 0.2863 | 0.6177 | 0.6270 | 0.7117 |
| 6000.0 | 0.3024 | 0.0914 | 0.2211 | 0.2869 | 0.6177 | 0.6272 | 0.7096 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 50.0 (r = 0.5312)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **50.0**
  - Achieves best Pearson correlation: 0.5312
  - R²: 0.2822

- **Alternative option**: Stimulus strength **25.0**
  - Pearson correlation: 0.5087


### For Specific Applications:

- **Binary classification tasks**: Strength 50.0 (accuracy = 0.6814)
- **Minimizing absolute error**: Strength 50.0 (MAE = 0.1802)
- **High precision required**: Strength 50.0 (precision = 0.6809)
- **High recall required**: Strength 250.0 (recall = 0.7587)


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
