
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: 50.0**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **12** different stimulus strengths:
5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
| Pearson Correlation | 50.0 | 0.5268 |
| R-squared | 50.0 | 0.2775 |
| MAE | 50.0 | 0.1810 |
| RMSE | 25.0 | 0.2284 |
| Binary Accuracy | 50.0 | 0.6831 |
| Binary Precision | 50.0 | 0.6859 |
| Binary Recall | 100.0 | 0.7594 |


**Strength 50.0** wins the most criteria (5 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.4551 | 0.2071 | 0.1937 | 0.2386 | 0.6555 | 0.6562 | 0.6798 |
| 10.0 | 0.4991 | 0.2491 | 0.1859 | 0.2317 | 0.6749 | 0.6744 | 0.7101 |
| 20.0 | 0.5116 | 0.2617 | 0.1838 | 0.2300 | 0.6789 | 0.6792 | 0.7128 |
| 25.0 | 0.5205 | 0.2709 | 0.1825 | 0.2284 | 0.6816 | 0.6843 | 0.7160 |
| 50.0 | 0.5268 | 0.2775 | 0.1810 | 0.2284 | 0.6831 | 0.6859 | 0.7384 |
| 100.0 | 0.4310 | 0.1857 | 0.1950 | 0.2505 | 0.6526 | 0.6485 | 0.7594 |
| 250.0 | 0.3726 | 0.1389 | 0.2043 | 0.2654 | 0.6349 | 0.6352 | 0.7533 |
| 500.0 | 0.3433 | 0.1179 | 0.2098 | 0.2732 | 0.6252 | 0.6277 | 0.7414 |
| 1000.0 | 0.3225 | 0.1040 | 0.2145 | 0.2790 | 0.6206 | 0.6254 | 0.7308 |
| 1500.0 | 0.3139 | 0.0985 | 0.2168 | 0.2818 | 0.6190 | 0.6253 | 0.7243 |
| 2000.0 | 0.3108 | 0.0966 | 0.2179 | 0.2831 | 0.6191 | 0.6264 | 0.7197 |
| 3000.0 | 0.3064 | 0.0939 | 0.2191 | 0.2848 | 0.6189 | 0.6276 | 0.7148 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 50.0 (r = 0.5268)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **50.0**
  - Achieves best Pearson correlation: 0.5268
  - R²: 0.2775

- **Alternative option**: Stimulus strength **25.0**
  - Pearson correlation: 0.5205


### For Specific Applications:

- **Binary classification tasks**: Strength 50.0 (accuracy = 0.6831)
- **Minimizing absolute error**: Strength 50.0 (MAE = 0.1810)
- **High precision required**: Strength 50.0 (precision = 0.6859)
- **High recall required**: Strength 100.0 (recall = 0.7594)


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
