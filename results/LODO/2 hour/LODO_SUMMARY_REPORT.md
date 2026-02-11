
# Leave-One-Drug-Out (LODO) Validation Summary Report

## Overview
- **Total predictions**: 4,784
- **Number of drugs validated**: 46
- **Number of TFs**: 104

## Overall Performance
- **Overall Pearson correlation**: 0.8335

## Per-TF Performance
- **Mean TF correlation**: 0.1021
- **Median TF correlation**: 0.1234
- **Best TF**: O00716 (r = 0.3793)
- **Worst TF**: P08651 (r = -0.8825)

## Per-Drug Performance
- **Mean drug correlation**: 0.8460
- **Median drug correlation**: 0.8485

## Top 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| O00716 |    0.379319 |        46 |
| P19838 |    0.369239 |        46 |
| P08047 |    0.364137 |        46 |
| Q16665 |    0.354864 |        46 |
| Q13127 |    0.347179 |        46 |
| P48436 |    0.34664  |        46 |
| Q99814 |    0.333088 |        46 |
| P16220 |    0.331503 |        46 |
| Q02447 |    0.330952 |        46 |
| Q01196 |    0.291808 |        46 |

## Bottom 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| Q00653 |   -0.113285 |        46 |
| Q14765 |   -0.11949  |        46 |
| P06401 |   -0.133483 |        46 |
| P23511 |   -0.168368 |        46 |
| P10914 |   -0.185277 |        46 |
| P56693 |   -0.196269 |        46 |
| P17542 |   -0.198869 |        46 |
| Q12772 |   -0.214894 |        46 |
| P36956 |   -0.275916 |        46 |
| P08651 |   -0.882489 |        46 |

## Files Generated
- `lodo_combined_predictions.csv`: All predictions combined
- `lodo_optimal_predictions.csv`: Optimal dose selected per drug-TF pair
- `lodo_tf_correlations.csv`: Per-TF correlation results
- `lodo_drug_correlations.csv`: Per-drug correlation results
- `lodo_overall_scatter.png`: Overall scatter plot
- `lodo_scatter_best_tfs.png`: Scatter plots for best TFs
- `lodo_scatter_worst_tfs.png`: Scatter plots for worst TFs
- `lodo_correlation_distribution.png`: Distribution of correlations
