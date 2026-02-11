
# Leave-One-Drug-Out (LODO) Validation Summary Report

## Overview
- **Total predictions**: 11,752
- **Number of drugs validated**: 113
- **Number of TFs**: 104

## Overall Performance
- **Overall Pearson correlation**: 0.3724

## Per-TF Performance
- **Mean TF correlation**: 0.2888
- **Median TF correlation**: 0.3204
- **Best TF**: P19544 (r = 0.6188)
- **Worst TF**: P08651 (r = -0.8395)

## Per-Drug Performance
- **Mean drug correlation**: 0.3767
- **Median drug correlation**: 0.4182

## Top 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| P19544 |    0.618805 |       113 |
| P14921 |    0.53431  |       113 |
| Q01094 |    0.531077 |       113 |
| Q14186 |    0.510069 |       113 |
| Q9Y261 |    0.51001  |       113 |
| P15923 |    0.496877 |       113 |
| Q16236 |    0.494047 |       113 |
| P15407 |    0.492271 |       113 |
| P10275 |    0.488515 |       113 |
| P01106 |    0.486343 |       113 |

## Bottom 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| P06401 |   0.0830718 |       113 |
| P20823 |   0.0730077 |       113 |
| P51692 |   0.0558263 |       113 |
| Q00653 |   0.0543645 |       113 |
| P10070 |   0.0508994 |       113 |
| O00327 |   0.0462107 |       113 |
| P23511 |   0.0149227 |       113 |
| P56693 |  -0.0393215 |       113 |
| P18850 |  -0.12761   |       113 |
| P08651 |  -0.839543  |       113 |

## Files Generated
- `lodo_combined_predictions.csv`: All predictions combined
- `lodo_optimal_predictions.csv`: Optimal dose selected per drug-TF pair
- `lodo_tf_correlations.csv`: Per-TF correlation results
- `lodo_drug_correlations.csv`: Per-drug correlation results
- `lodo_overall_scatter.png`: Overall scatter plot
- `lodo_scatter_best_tfs.png`: Scatter plots for best TFs
- `lodo_scatter_worst_tfs.png`: Scatter plots for worst TFs
- `lodo_correlation_distribution.png`: Distribution of correlations
