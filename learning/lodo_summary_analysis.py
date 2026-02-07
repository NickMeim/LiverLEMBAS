#!/usr/bin/env python3
"""
Leave-One-Drug-Out (LODO) Summary Analysis Script

This script aggregates LODO validation results across all drugs and:
1. Loads all individual drug prediction CSVs
2. Selects optimal dose per drug-TF pair (closest match)
3. Calculates Pearson correlation per TF across drugs
4. Generates scatter plots for best and worst performing TFs
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='LODO Summary Analysis')

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing LODO prediction CSVs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results (default: results_dir)')
    parser.add_argument('--file_pattern', type=str, default='drug*_lodo_prediction.csv',
                        help='Pattern for prediction files')
    parser.add_argument('--n_best', type=int, default=5,
                        help='Number of best TFs to show')
    parser.add_argument('--n_worst', type=int, default=5,
                        help='Number of worst TFs to show')

    return parser.parse_args()


def load_all_predictions(results_dir, file_pattern):
    """Load all individual drug prediction files."""
    print2log(f'Loading predictions from {results_dir}...')

    file_paths = glob.glob(os.path.join(results_dir, file_pattern))
    print2log(f'Found {len(file_paths)} prediction files')

    all_data = []
    for fp in sorted(file_paths):
        try:
            df = pd.read_csv(fp)
            all_data.append(df)
            print2log(f'  Loaded: {os.path.basename(fp)} ({len(df)} rows)')
        except Exception as e:
            print2log(f'  Error loading {fp}: {e}')

    if not all_data:
        raise ValueError("No prediction files found!")

    combined_df = pd.concat(all_data, ignore_index=True)
    print2log(f'Total predictions: {len(combined_df)}')

    return combined_df


def select_optimal_dose(df):
    """
    For each drug-TF pair, select the dose that minimizes the distance
    between predicted and true activity.
    """
    print2log('Selecting optimal dose per drug-TF pair...')

    # Calculate absolute error
    df['abs_error'] = np.abs(df['true_value'] - df['predicted_value'])

    # Group by drug and TF, select dose with minimum error
    optimal_idx = df.groupby(['drug', 'TF'])['abs_error'].idxmin()
    optimal_df = df.loc[optimal_idx].copy()

    print2log(f'Selected {len(optimal_df)} optimal drug-TF pairs')

    return optimal_df


def calculate_tf_correlations(df):
    """Calculate Pearson correlation for each TF across all drugs."""
    print2log('Calculating per-TF correlations...')

    correlations = []

    for tf in df['TF'].unique():
        tf_data = df[df['TF'] == tf]

        if len(tf_data) < 3:
            continue

        true_vals = tf_data['true_value'].values
        pred_vals = tf_data['predicted_value'].values

        # Handle constant values
        if np.std(true_vals) < 1e-10 or np.std(pred_vals) < 1e-10:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(true_vals, pred_vals)

        correlations.append({
            'TF': tf,
            'pearson_r': r,
            'p_value': p,
            'n_drugs': len(tf_data),
            'mean_true': np.mean(true_vals),
            'std_true': np.std(true_vals),
            'mean_pred': np.mean(pred_vals),
            'std_pred': np.std(pred_vals)
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('pearson_r', ascending=False).reset_index(drop=True)

    print2log(f'Calculated correlations for {len(corr_df)} TFs')
    print2log(f'  Mean Pearson r: {corr_df["pearson_r"].mean():.3f}')
    print2log(f'  Median Pearson r: {corr_df["pearson_r"].median():.3f}')

    return corr_df


def calculate_drug_correlations(df):
    """Calculate Pearson correlation for each drug across all TFs."""
    print2log('Calculating per-drug correlations...')

    correlations = []

    for drug in df['drug'].unique():
        drug_data = df[df['drug'] == drug]

        if len(drug_data) < 3:
            continue

        true_vals = drug_data['true_value'].values
        pred_vals = drug_data['predicted_value'].values

        if np.std(true_vals) < 1e-10 or np.std(pred_vals) < 1e-10:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(true_vals, pred_vals)

        correlations.append({
            'drug': drug,
            'drug_short': drug[:30] + '...' if len(drug) > 30 else drug,
            'pearson_r': r,
            'p_value': p,
            'n_tfs': len(drug_data)
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('pearson_r', ascending=False).reset_index(drop=True)

    print2log(f'Calculated correlations for {len(corr_df)} drugs')

    return corr_df


def plot_scatter_best_worst(df, corr_df, n_best, n_worst, output_dir):
    """Generate scatter plots for best and worst performing TFs."""
    print2log('Generating scatter plots...')

    # Get best and worst TFs
    best_tfs = corr_df.head(n_best)['TF'].values
    worst_tfs = corr_df.tail(n_worst)['TF'].values

    # Setup figure for best TFs
    fig, axes = plt.subplots(1, n_best, figsize=(4*n_best, 4))
    if n_best == 1:
        axes = [axes]

    for i, tf in enumerate(best_tfs):
        tf_data = df[df['TF'] == tf]
        ax = axes[i]

        ax.scatter(tf_data['true_value'], tf_data['predicted_value'],
                   alpha=0.6, c='#1E90FF', s=50)

        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)

        # Calculate correlation for this TF
        r = corr_df[corr_df['TF'] == tf]['pearson_r'].values[0]

        ax.set_xlabel('True Activity', fontsize=12)
        ax.set_ylabel('Predicted Activity', fontsize=12)
        ax.set_title(f'{tf}\nr = {r:.3f}', fontsize=11)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Best Performing TFs (Highest Correlation)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_scatter_best_tfs.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Setup figure for worst TFs
    fig, axes = plt.subplots(1, n_worst, figsize=(4*n_worst, 4))
    if n_worst == 1:
        axes = [axes]

    for i, tf in enumerate(worst_tfs):
        tf_data = df[df['TF'] == tf]
        ax = axes[i]

        ax.scatter(tf_data['true_value'], tf_data['predicted_value'],
                   alpha=0.6, c='#FF6347', s=50)

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)

        r = corr_df[corr_df['TF'] == tf]['pearson_r'].values[0]

        ax.set_xlabel('True Activity', fontsize=12)
        ax.set_ylabel('Predicted Activity', fontsize=12)
        ax.set_title(f'{tf}\nr = {r:.3f}', fontsize=11)
        ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Worst Performing TFs (Lowest Correlation)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_scatter_worst_tfs.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print2log(f'Saved scatter plots to {output_dir}')


def plot_correlation_distribution(corr_df, output_dir):
    """Plot distribution of correlations across TFs."""
    print2log('Generating correlation distribution plot...')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax = axes[0]
    ax.hist(corr_df['pearson_r'], bins=30, color='#4169E1', edgecolor='white', alpha=0.8)
    ax.axvline(corr_df['pearson_r'].mean(), color='red', linestyle='--',
               label=f'Mean: {corr_df["pearson_r"].mean():.3f}')
    ax.axvline(corr_df['pearson_r'].median(), color='orange', linestyle='--',
               label=f'Median: {corr_df["pearson_r"].median():.3f}')
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of TF Correlations', fontsize=14)
    ax.legend()

    # Bar plot of top/bottom TFs
    ax = axes[1]
    top_n = 10
    top_tfs = corr_df.head(top_n)
    bottom_tfs = corr_df.tail(top_n)

    combined = pd.concat([top_tfs, bottom_tfs])
    colors = ['#1E90FF'] * top_n + ['#FF6347'] * top_n

    y_pos = np.arange(len(combined))
    ax.barh(y_pos, combined['pearson_r'], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined['TF'], fontsize=9)
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_title(f'Top {top_n} and Bottom {top_n} TFs', fontsize=14)
    ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_correlation_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_overall_scatter(df, output_dir):
    """Plot overall scatter of all predictions."""
    print2log('Generating overall scatter plot...')

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(df['true_value'], df['predicted_value'],
               alpha=0.3, c='#4169E1', s=20)

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)

    # Overall correlation
    r, p = pearsonr(df['true_value'], df['predicted_value'])

    ax.set_xlabel('True Activity', fontsize=14)
    ax.set_ylabel('Predicted Activity', fontsize=14)
    ax.set_title(f'LODO Validation: All Predictions\n'
                 f'Pearson r = {r:.3f} (n = {len(df)})', fontsize=14)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_overall_scatter.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(df, tf_corr_df, drug_corr_df, output_dir):
    """Generate a summary report of LODO validation results."""
    print2log('Generating summary report...')

    overall_r, _ = pearsonr(df['true_value'], df['predicted_value'])

    report = f"""
# Leave-One-Drug-Out (LODO) Validation Summary Report

Generated using K-Dense Web ([k-dense.ai](https://k-dense.ai))

## Overview
- **Total predictions**: {len(df):,}
- **Number of drugs validated**: {df['drug'].nunique()}
- **Number of TFs**: {df['TF'].nunique()}

## Overall Performance
- **Overall Pearson correlation**: {overall_r:.4f}

## Per-TF Performance
- **Mean TF correlation**: {tf_corr_df['pearson_r'].mean():.4f}
- **Median TF correlation**: {tf_corr_df['pearson_r'].median():.4f}
- **Best TF**: {tf_corr_df.iloc[0]['TF']} (r = {tf_corr_df.iloc[0]['pearson_r']:.4f})
- **Worst TF**: {tf_corr_df.iloc[-1]['TF']} (r = {tf_corr_df.iloc[-1]['pearson_r']:.4f})

## Per-Drug Performance
- **Mean drug correlation**: {drug_corr_df['pearson_r'].mean():.4f}
- **Median drug correlation**: {drug_corr_df['pearson_r'].median():.4f}

## Top 10 TFs by Correlation
{tf_corr_df.head(10)[['TF', 'pearson_r', 'n_drugs']].to_markdown(index=False)}

## Bottom 10 TFs by Correlation
{tf_corr_df.tail(10)[['TF', 'pearson_r', 'n_drugs']].to_markdown(index=False)}

## Files Generated
- `lodo_combined_predictions.csv`: All predictions combined
- `lodo_optimal_predictions.csv`: Optimal dose selected per drug-TF pair
- `lodo_tf_correlations.csv`: Per-TF correlation results
- `lodo_drug_correlations.csv`: Per-drug correlation results
- `lodo_overall_scatter.png`: Overall scatter plot
- `lodo_scatter_best_tfs.png`: Scatter plots for best TFs
- `lodo_scatter_worst_tfs.png`: Scatter plots for worst TFs
- `lodo_correlation_distribution.png`: Distribution of correlations

---
Generated using K-Dense Web (k-dense.ai)
"""

    with open(os.path.join(output_dir, 'LODO_SUMMARY_REPORT.md'), 'w') as f:
        f.write(report)

    print2log(f'Summary report saved to {output_dir}/LODO_SUMMARY_REPORT.md')


def main():
    args = parse_arguments()

    output_dir = args.output_dir or args.results_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load all predictions
    combined_df = load_all_predictions(args.results_dir, args.file_pattern)

    # Save combined data
    combined_df.to_csv(os.path.join(output_dir, 'lodo_combined_predictions.csv'), index=False)

    # Select optimal dose
    optimal_df = select_optimal_dose(combined_df)
    optimal_df.to_csv(os.path.join(output_dir, 'lodo_optimal_predictions.csv'), index=False)

    # Calculate correlations
    tf_corr_df = calculate_tf_correlations(optimal_df)
    tf_corr_df.to_csv(os.path.join(output_dir, 'lodo_tf_correlations.csv'), index=False)

    drug_corr_df = calculate_drug_correlations(optimal_df)
    drug_corr_df.to_csv(os.path.join(output_dir, 'lodo_drug_correlations.csv'), index=False)

    # Generate visualizations
    plot_overall_scatter(optimal_df, output_dir)
    plot_scatter_best_worst(optimal_df, tf_corr_df, args.n_best, args.n_worst, output_dir)
    plot_correlation_distribution(tf_corr_df, output_dir)

    # Generate summary report
    generate_summary_report(optimal_df, tf_corr_df, drug_corr_df, output_dir)

    print2log('LODO summary analysis completed!')


if __name__ == '__main__':
    main()
