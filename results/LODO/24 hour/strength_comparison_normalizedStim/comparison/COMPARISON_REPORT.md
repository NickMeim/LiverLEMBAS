
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: 25.0**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **15** different stimulus strengths:
5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
| Pearson Correlation | 25.0 | 0.5616 |
| R-squared | 25.0 | 0.3154 |
| MAE | 25.0 | 0.1728 |
| RMSE | 25.0 | 0.2191 |
| Binary Accuracy | 25.0 | 0.6962 |
| Binary Precision | 25.0 | 0.6824 |
| Binary Recall | 100.0 | 0.7021 |


**Strength 25.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.4697 | 0.2207 | 0.1888 | 0.2329 | 0.6603 | 0.6662 | 0.4995 |
| 10.0 | 0.5429 | 0.2947 | 0.1766 | 0.2216 | 0.6875 | 0.6805 | 0.5826 |
| 20.0 | 0.5567 | 0.3099 | 0.1738 | 0.2198 | 0.6927 | 0.6781 | 0.6133 |
| 25.0 | 0.5616 | 0.3154 | 0.1728 | 0.2191 | 0.6962 | 0.6824 | 0.6174 |
| 50.0 | 0.5565 | 0.3096 | 0.1748 | 0.2211 | 0.6909 | 0.6595 | 0.6743 |
| 100.0 | 0.3877 | 0.1503 | 0.2062 | 0.2584 | 0.6309 | 0.5955 | 0.7021 |
| 250.0 | 0.3108 | 0.0966 | 0.2209 | 0.2799 | 0.6030 | 0.5697 | 0.6980 |
| 500.0 | 0.2958 | 0.0875 | 0.2237 | 0.2844 | 0.6027 | 0.5714 | 0.6959 |
| 1000.0 | 0.2734 | 0.0748 | 0.2276 | 0.2905 | 0.5954 | 0.5647 | 0.6875 |
| 1500.0 | 0.2720 | 0.0740 | 0.2282 | 0.2914 | 0.5949 | 0.5649 | 0.6827 |
| 2000.0 | 0.2703 | 0.0731 | 0.2285 | 0.2922 | 0.5962 | 0.5671 | 0.6791 |
| 3000.0 | 0.2698 | 0.0728 | 0.2291 | 0.2932 | 0.5986 | 0.5701 | 0.6737 |
| 4000.0 | 0.2689 | 0.0723 | 0.2299 | 0.2944 | 0.5992 | 0.5724 | 0.6685 |
| 5000.0 | 0.2690 | 0.0724 | 0.2305 | 0.2953 | 0.6008 | 0.5743 | 0.6668 |
| 6000.0 | 0.2684 | 0.0720 | 0.2310 | 0.2963 | 0.6007 | 0.5749 | 0.6631 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 25.0 (r = 0.5616)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **25.0**
  - Achieves best Pearson correlation: 0.5616
  - R²: 0.3154

- **Alternative option**: Stimulus strength **20.0**
  - Pearson correlation: 0.5567


### For Specific Applications:

- **Binary classification tasks**: Strength 25.0 (accuracy = 0.6962)
- **Minimizing absolute error**: Strength 25.0 (MAE = 0.1728)
- **High precision required**: Strength 25.0 (precision = 0.6824)
- **High recall required**: Strength 100.0 (recall = 0.7021)


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
