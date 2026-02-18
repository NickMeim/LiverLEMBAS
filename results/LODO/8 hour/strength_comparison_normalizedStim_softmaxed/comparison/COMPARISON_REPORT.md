
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: 250.0**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **15** different stimulus strengths:
5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
| Pearson Correlation | 250.0 | 0.3713 |
| R-squared | 250.0 | 0.1378 |
| MAE | 250.0 | 0.2033 |
| RMSE | 50.0 | 0.2559 |
| Binary Accuracy | 250.0 | 0.6343 |
| Binary Precision | 250.0 | 0.6336 |
| Binary Recall | 250.0 | 0.7531 |


**Strength 250.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.2621 | 0.0687 | 0.2172 | 0.2582 | 0.5911 | 0.5915 | 0.6457 |
| 10.0 | 0.2595 | 0.0673 | 0.2167 | 0.2589 | 0.5918 | 0.5919 | 0.6472 |
| 20.0 | 0.2653 | 0.0704 | 0.2156 | 0.2584 | 0.5942 | 0.5949 | 0.6463 |
| 25.0 | 0.2790 | 0.0778 | 0.2146 | 0.2575 | 0.5957 | 0.5983 | 0.6437 |
| 50.0 | 0.3105 | 0.0964 | 0.2110 | 0.2559 | 0.6129 | 0.6274 | 0.6693 |
| 100.0 | 0.3414 | 0.1166 | 0.2055 | 0.2596 | 0.6217 | 0.6253 | 0.7296 |
| 250.0 | 0.3713 | 0.1378 | 0.2033 | 0.2628 | 0.6343 | 0.6336 | 0.7531 |
| 500.0 | 0.3602 | 0.1297 | 0.2061 | 0.2669 | 0.6316 | 0.6332 | 0.7460 |
| 1000.0 | 0.3429 | 0.1176 | 0.2105 | 0.2725 | 0.6254 | 0.6292 | 0.7345 |
| 1500.0 | 0.3339 | 0.1115 | 0.2129 | 0.2755 | 0.6220 | 0.6273 | 0.7280 |
| 2000.0 | 0.3308 | 0.1094 | 0.2142 | 0.2770 | 0.6218 | 0.6277 | 0.7249 |
| 3000.0 | 0.3260 | 0.1063 | 0.2157 | 0.2790 | 0.6200 | 0.6282 | 0.7194 |
| 4000.0 | 0.3142 | 0.0987 | 0.2180 | 0.2823 | 0.6189 | 0.6284 | 0.7173 |
| 5000.0 | 0.3068 | 0.0941 | 0.2196 | 0.2846 | 0.6172 | 0.6269 | 0.7147 |
| 6000.0 | 0.3016 | 0.0910 | 0.2207 | 0.2860 | 0.6156 | 0.6268 | 0.7117 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a increasing trend as stimulus strength increases
2. **Peak performance** at strength 250.0 (r = 0.3713)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **250.0**
  - Achieves best Pearson correlation: 0.3713
  - R²: 0.1378

- **Alternative option**: Stimulus strength **500.0**
  - Pearson correlation: 0.3602


### For Specific Applications:

- **Binary classification tasks**: Strength 250.0 (accuracy = 0.6343)
- **Minimizing absolute error**: Strength 250.0 (MAE = 0.2033)
- **High precision required**: Strength 250.0 (precision = 0.6336)
- **High recall required**: Strength 250.0 (recall = 0.7531)


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
