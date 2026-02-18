
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: 5.0**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **15** different stimulus strengths:
5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
| Pearson Correlation | 5.0 | 0.8657 |
| R-squared | 5.0 | 0.7494 |
| MAE | 5.0 | 0.0966 |
| RMSE | 5.0 | 0.1266 |
| Binary Accuracy | 5.0 | 0.8439 |
| Binary Precision | 5.0 | 0.8469 |
| Binary Recall | 250.0 | 0.8414 |


**Strength 5.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.8657 | 0.7494 | 0.0966 | 0.1266 | 0.8439 | 0.8469 | 0.7937 |
| 10.0 | 0.8563 | 0.7332 | 0.0994 | 0.1310 | 0.8396 | 0.8355 | 0.7972 |
| 20.0 | 0.8568 | 0.7341 | 0.0997 | 0.1309 | 0.8390 | 0.8241 | 0.8147 |
| 25.0 | 0.8558 | 0.7324 | 0.1001 | 0.1314 | 0.8365 | 0.8189 | 0.8155 |
| 50.0 | 0.8490 | 0.7208 | 0.1026 | 0.1347 | 0.8320 | 0.8076 | 0.8205 |
| 100.0 | 0.8368 | 0.7002 | 0.1069 | 0.1410 | 0.8214 | 0.7874 | 0.8290 |
| 250.0 | 0.8107 | 0.6573 | 0.1155 | 0.1539 | 0.8083 | 0.7623 | 0.8414 |
| 500.0 | 0.7947 | 0.6316 | 0.1209 | 0.1614 | 0.7999 | 0.7515 | 0.8391 |
| 1000.0 | 0.7754 | 0.6012 | 0.1260 | 0.1688 | 0.7921 | 0.7463 | 0.8306 |
| 1500.0 | 0.7613 | 0.5795 | 0.1292 | 0.1740 | 0.7891 | 0.7442 | 0.8253 |
| 2000.0 | 0.7506 | 0.5633 | 0.1321 | 0.1782 | 0.7846 | 0.7381 | 0.8185 |
| 3000.0 | 0.7336 | 0.5381 | 0.1355 | 0.1845 | 0.7813 | 0.7355 | 0.8112 |
| 4000.0 | 0.7236 | 0.5236 | 0.1374 | 0.1881 | 0.7817 | 0.7373 | 0.8085 |
| 5000.0 | 0.7160 | 0.5126 | 0.1386 | 0.1908 | 0.7809 | 0.7382 | 0.8060 |
| 6000.0 | 0.7094 | 0.5032 | 0.1397 | 0.1931 | 0.7803 | 0.7393 | 0.8015 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 5.0 (r = 0.8657)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **5.0**
  - Achieves best Pearson correlation: 0.8657
  - R²: 0.7494

- **Alternative option**: Stimulus strength **20.0**
  - Pearson correlation: 0.8568


### For Specific Applications:

- **Binary classification tasks**: Strength 5.0 (accuracy = 0.8439)
- **Minimizing absolute error**: Strength 5.0 (MAE = 0.0966)
- **High precision required**: Strength 5.0 (precision = 0.8469)
- **High recall required**: Strength 250.0 (recall = 0.8414)


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
