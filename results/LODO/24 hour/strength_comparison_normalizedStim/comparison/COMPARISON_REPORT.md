
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
| Pearson Correlation | 25.0 | 0.5376 |
| R-squared | 25.0 | 0.2891 |
| MAE | 25.0 | 0.1765 |
| RMSE | 25.0 | 0.2246 |
| Binary Accuracy | 25.0 | 0.6887 |
| Binary Precision | 25.0 | 0.6739 |
| Binary Recall | 100.0 | 0.6791 |


**Strength 25.0** wins the most criteria (6 out of 7)

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
| 5.0 | 0.4691 | 0.2200 | 0.1889 | 0.2343 | 0.6619 | 0.6655 | 0.5000 |
| 10.0 | 0.5194 | 0.2698 | 0.1808 | 0.2267 | 0.6801 | 0.6715 | 0.5681 |
| 20.0 | 0.5357 | 0.2870 | 0.1771 | 0.2248 | 0.6862 | 0.6703 | 0.5990 |
| 25.0 | 0.5376 | 0.2891 | 0.1765 | 0.2246 | 0.6887 | 0.6739 | 0.6007 |
| 50.0 | 0.5242 | 0.2747 | 0.1794 | 0.2280 | 0.6797 | 0.6458 | 0.6516 |
| 100.0 | 0.3644 | 0.1328 | 0.2091 | 0.2619 | 0.6221 | 0.5820 | 0.6791 |
| 250.0 | 0.2977 | 0.0886 | 0.2217 | 0.2804 | 0.5976 | 0.5592 | 0.6771 |
| 500.0 | 0.2735 | 0.0748 | 0.2255 | 0.2863 | 0.5941 | 0.5584 | 0.6706 |
| 1000.0 | 0.2514 | 0.0632 | 0.2291 | 0.2920 | 0.5863 | 0.5502 | 0.6622 |
| 1500.0 | 0.2516 | 0.0633 | 0.2296 | 0.2926 | 0.5868 | 0.5520 | 0.6569 |
| 2000.0 | 0.2542 | 0.0646 | 0.2292 | 0.2921 | 0.5892 | 0.5553 | 0.6539 |
| 3000.0 | 0.2547 | 0.0649 | 0.2297 | 0.2927 | 0.5915 | 0.5583 | 0.6480 |
| 4000.0 | 0.2530 | 0.0640 | 0.2308 | 0.2941 | 0.5918 | 0.5610 | 0.6411 |
| 5000.0 | 0.2538 | 0.0644 | 0.2312 | 0.2949 | 0.5941 | 0.5636 | 0.6395 |
| 6000.0 | 0.3316 | 0.1099 | 0.2175 | 0.2821 | 0.6205 | 0.5855 | 0.6625 |


---

## Performance Trends

### Key Observations:

1. **Pearson correlation** shows a decreasing trend as stimulus strength increases
2. **Peak performance** at strength 25.0 (r = 0.5376)
3. **MAE** is increasing with higher stimulus strengths


---

## Recommendations

### For General Use:
- **Primary recommendation**: Use stimulus strength **25.0**
  - Achieves best Pearson correlation: 0.5376
  - R²: 0.2891

- **Alternative option**: Stimulus strength **20.0**
  - Pearson correlation: 0.5357


### For Specific Applications:

- **Binary classification tasks**: Strength 25.0 (accuracy = 0.6887)
- **Minimizing absolute error**: Strength 25.0 (MAE = 0.1765)
- **High precision required**: Strength 25.0 (precision = 0.6739)
- **High recall required**: Strength 100.0 (recall = 0.6791)


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
