
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: 5.0**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **12** different stimulus strengths:
5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
| Pearson Correlation | 5.0 | 0.8609 |
| R-squared | 5.0 | 0.7411 |
| MAE | 5.0 | 0.0978 |
| RMSE | 5.0 | 0.1288 |
| Binary Accuracy | 5.0 | 0.8425 |
| Binary Precision | 5.0 | 0.8402 |
| Binary Recall | 250.0 | 0.8395 |


**Strength 5.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.8609 | 0.7411 | 0.0978 | 0.1288 | 0.8425 | 0.8402 | 0.7960 |
| 10.0 | 0.8558 | 0.7323 | 0.0995 | 0.1313 | 0.8367 | 0.8330 | 0.7925 |
| 20.0 | 0.8548 | 0.7307 | 0.1005 | 0.1318 | 0.8372 | 0.8195 | 0.8150 |
| 25.0 | 0.8535 | 0.7285 | 0.1010 | 0.1324 | 0.8355 | 0.8162 | 0.8155 |
| 50.0 | 0.8474 | 0.7181 | 0.1030 | 0.1354 | 0.8318 | 0.8072 | 0.8212 |
| 100.0 | 0.8343 | 0.6961 | 0.1074 | 0.1420 | 0.8212 | 0.7865 | 0.8299 |
| 250.0 | 0.8080 | 0.6529 | 0.1157 | 0.1547 | 0.8108 | 0.7676 | 0.8395 |
| 500.0 | 0.7915 | 0.6264 | 0.1213 | 0.1625 | 0.8016 | 0.7545 | 0.8377 |
| 1000.0 | 0.7725 | 0.5968 | 0.1266 | 0.1702 | 0.7942 | 0.7457 | 0.8333 |
| 1500.0 | 0.7578 | 0.5743 | 0.1299 | 0.1757 | 0.7881 | 0.7394 | 0.8252 |
| 2000.0 | 0.7465 | 0.5573 | 0.1323 | 0.1799 | 0.7850 | 0.7380 | 0.8180 |
| 3000.0 | 0.7310 | 0.5344 | 0.1353 | 0.1856 | 0.7852 | 0.7403 | 0.8135 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 5.0 (r = 0.8609)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **5.0**
  - Achieves best Pearson correlation: 0.8609
  - R²: 0.7411

- **Alternative option**: Stimulus strength **10.0**
  - Pearson correlation: 0.8558


### For Specific Applications:

- **Binary classification tasks**: Strength 5.0 (accuracy = 0.8425)
- **Minimizing absolute error**: Strength 5.0 (MAE = 0.0978)
- **High precision required**: Strength 5.0 (precision = 0.8402)
- **High recall required**: Strength 250.0 (recall = 0.8395)


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
