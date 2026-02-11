
# Leave-One-Drug-Out (LODO) Validation Summary Report

## Overview
- **Total predictions**: 11,752
- **Number of drugs validated**: 113
- **Number of TFs**: 104

## Overall Performance
- **Overall Pearson correlation**: 0.4310

## Per-TF Performance
- **Mean TF correlation**: 0.2953
- **Median TF correlation**: 0.3017
- **Best TF**: P19544 (r = 0.6671)
- **Worst TF**: P08651 (r = -0.8966)

## Per-Drug Performance
- **Mean drug correlation**: 0.4161
- **Median drug correlation**: 0.4267

## Top 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| P19544 |    0.667062 |       113 |
| P05412 |    0.623716 |       113 |
| P15407 |    0.623137 |       113 |
| P37275 |    0.606058 |       113 |
| P14921 |    0.599987 |       113 |
| Q16236 |    0.599506 |       113 |
| P15923 |    0.597804 |       113 |
| P19838 |    0.594531 |       113 |
| Q01543 |    0.586149 |       113 |
| Q04206 |    0.584731 |       113 |

## Bottom 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| P48436 |   0.0108635 |       113 |
| P48431 |  -0.0234452 |       113 |
| Q00653 |  -0.0280858 |       113 |
| P56693 |  -0.033718  |       113 |
| P49715 |  -0.0365406 |       113 |
| P23771 |  -0.0781766 |       113 |
| Q14765 |  -0.123358  |       113 |
| Q13761 |  -0.150319  |       113 |
| P18850 |  -0.170814  |       113 |
| P08651 |  -0.896589  |       113 |

## Files Generated
- `lodo_combined_predictions.csv`: All predictions combined
- `lodo_optimal_predictions.csv`: Optimal dose selected per drug-TF pair
- `lodo_tf_correlations.csv`: Per-TF correlation results
- `lodo_drug_correlations.csv`: Per-drug correlation results
- `lodo_overall_scatter.png`: Overall scatter plot
- `lodo_scatter_best_tfs.png`: Scatter plots for best TFs
- `lodo_scatter_worst_tfs.png`: Scatter plots for worst TFs
- `lodo_correlation_distribution.png`: Distribution of correlations
