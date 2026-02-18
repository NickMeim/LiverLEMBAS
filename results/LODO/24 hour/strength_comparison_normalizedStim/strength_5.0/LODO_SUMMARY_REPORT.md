
# LODO Validation Summary - Stimulus Strength 5.0

## Overview
- **Total predictions**: 7,800
- **Number of drugs**: 75
- **Number of TFs**: 104

## Overall Performance
- **Pearson correlation**: 0.4691
- **R-squared**: 0.2200
- **MAE**: 0.1889
- **RMSE**: 0.2343

## Binary Classification (threshold=0.5)
- **Precision**: 0.6655
- **Recall**: 0.5000
- **Accuracy**: 0.6619

## Per-TF Performance
- **Mean TF correlation**: 0.2040
- **Median TF correlation**: 0.2171
- **Best TF**: P37275 (r = 0.4509)
- **Worst TF**: P08651 (r = -0.8562)

## Per-Drug Performance
- **Mean drug correlation**: 0.4622
- **Median drug correlation**: 0.4996

## Top 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| P37275 |    0.450894 |        75 |
| P10275 |    0.436038 |        75 |
| P23769 |    0.408734 |        75 |
| P04150 |    0.397977 |        75 |
| P14921 |    0.395071 |        75 |
| P18848 |    0.394044 |        75 |
| P19544 |    0.382574 |        75 |
| Q9Y261 |    0.372387 |        75 |
| Q92731 |    0.368596 |        75 |
| Q04206 |    0.366992 |        75 |

## Bottom 10 TFs by Correlation
| TF     |   pearson_r |   n_drugs |
|:-------|------------:|----------:|
| P08047 |   0.0218974 |        75 |
| O00327 |   0.015274  |        75 |
| P18850 |  -0.017558  |        75 |
| P48431 |  -0.0227181 |        75 |
| Q14186 |  -0.0291861 |        75 |
| P56693 |  -0.0297982 |        75 |
| P37231 |  -0.0344376 |        75 |
| Q02447 |  -0.0358687 |        75 |
| P17947 |  -0.124349  |        75 |
| P08651 |  -0.856189  |        75 |
