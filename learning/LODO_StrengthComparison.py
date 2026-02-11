"""
Multi-Strength LODO Summary Analysis Script

This script aggregates LODO validation results across different stimulus strengths:
1. Processes each strength folder separately
2. Generates validation plots and reports for each strength
3. Compares performance across strengths to identify the optimal stimulus
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
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='Multi-Strength LODO Summary Analysis')

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Parent directory containing strength_X.X folders')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results (default: results_dir/comparison)')
    parser.add_argument('--file_pattern', type=str, default='drug*_lodo_prediction_strength*.csv',
                        help='Pattern for prediction files')
    parser.add_argument('--n_best', type=int, default=5,
                        help='Number of best TFs to show')
    parser.add_argument('--n_worst', type=int, default=5,
                        help='Number of worst TFs to show')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification')

    return parser.parse_args()


def find_strength_folders(results_dir):
    """Find all strength folders in the results directory."""
    strength_folders = []
    pattern = re.compile(r'strength_([\d.]+)')
    
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                strength_value = float(match.group(1))
                strength_folders.append({
                    'path': item_path,
                    'name': item,
                    'strength': strength_value
                })
    
    # Sort by strength value
    strength_folders = sorted(strength_folders, key=lambda x: x['strength'])
    
    print2log(f'Found {len(strength_folders)} strength folders:')
    for sf in strength_folders:
        print2log(f"  - {sf['name']} (strength={sf['strength']})")
    
    return strength_folders


def extract_strength_from_filename(filename):
    """Extract strength value from filename."""
    match = re.search(r'strength([\d.]+)', filename)
    if match:
        return float(match.group(1))
    return None


def load_all_predictions(strength_folder_path, file_pattern):
    """Load all individual drug prediction files for a specific strength."""
    file_paths = glob.glob(os.path.join(strength_folder_path, file_pattern))
    print2log(f'  Found {len(file_paths)} prediction files')

    all_data = []
    for fp in sorted(file_paths):
        try:
            df = pd.read_csv(fp)
            all_data.append(df)
        except Exception as e:
            print2log(f'  Error loading {fp}: {e}')

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Clamp negative predicted TF activity to 0
    combined_df['predicted_value'] = pd.to_numeric(combined_df['predicted_value'], errors='coerce')
    n_neg = (combined_df['predicted_value'] < 0).sum()
    if n_neg > 0:
        combined_df.loc[combined_df['predicted_value'] < 0, 'predicted_value'] = 0.0

    return combined_df


def infer_dose_column(df):
    """Infer the dose column name from common candidates."""
    candidates = [
        'dose', 'Dose', 'DOSE',
        'dose_uM', 'dose_um', 'dose_µM',
        'concentration', 'Concentration',
        'concentration_uM', 'concentration_um',
        'conc', 'Conc',
        'dose_level', 'dose_idx', 'dose_index'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find a dose column. Expected one of: "
        + ", ".join(candidates)
    )


def select_optimal_dose_per_drug_by_tf_correlation(df, dose_col=None, min_points=3):
    """Select ONE dose per drug that maximizes Pearson correlation across TFs."""
    if dose_col is None:
        dose_col = infer_dose_column(df)

    work = df.copy()
    work[dose_col] = pd.to_numeric(work[dose_col], errors='coerce')

    per_drug_choices = []

    for drug, g in work.groupby('drug', sort=False):
        dose_rows = []
        for dose, gd in g.groupby(dose_col, sort=True):
            gd = gd.dropna(subset=['true_value', 'predicted_value'])

            n = len(gd)
            if n == 0:
                continue

            true_vals = gd['true_value'].values
            pred_vals = gd['predicted_value'].values
            mae = float(np.mean(np.abs(true_vals - pred_vals)))

            if n >= min_points and np.std(true_vals) > 1e-10 and np.std(pred_vals) > 1e-10:
                r, p = pearsonr(true_vals, pred_vals)
            else:
                r, p = np.nan, np.nan

            dose_rows.append({
                'drug': drug,
                'dose': dose,
                'pearson_r_across_TFs': r,
                'p_value': p,
                'n_tfs': n,
                'mae_across_TFs': mae
            })

        if not dose_rows:
            continue

        ddf = pd.DataFrame(dose_rows)

        has_r = ddf['pearson_r_across_TFs'].notna().any()
        if has_r:
            ddf = ddf.sort_values(
                by=['pearson_r_across_TFs', 'mae_across_TFs', 'dose'],
                ascending=[False, True, False]
            )
        else:
            ddf = ddf.sort_values(
                by=['mae_across_TFs', 'dose'],
                ascending=[True, False]
            )

        best = ddf.iloc[0].to_dict()
        per_drug_choices.append(best)

    choice_df = pd.DataFrame(per_drug_choices)

    selected_df = work.merge(
        choice_df[['drug', 'dose']],
        left_on=['drug', dose_col],
        right_on=['drug', 'dose'],
        how='inner'
    ).drop(columns=['dose'])

    return selected_df, choice_df


def calculate_tf_correlations(df):
    """Calculate Pearson correlation for each TF across all drugs."""
    correlations = []

    for tf in df['TF'].unique():
        tf_data = df[df['TF'] == tf]

        if len(tf_data) < 3:
            continue

        true_vals = tf_data['true_value'].values
        pred_vals = tf_data['predicted_value'].values

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

    return corr_df


def calculate_drug_correlations(df):
    """Calculate Pearson correlation for each drug across all TFs."""
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

    return corr_df


def _binary_labels_from_values(values, threshold=0.5):
    """Convert continuous activities to binary labels: 1=active, 0=inactive."""
    v = pd.to_numeric(values, errors='coerce')
    return (v >= threshold).astype(int)


def _confusion_and_metrics(y_true, y_pred):
    """Calculate confusion matrix and metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = int(len(y_true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    accuracy = (tp + tn) / n if n > 0 else np.nan

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'n': n,
            'precision': precision, 'recall': recall, 'accuracy': accuracy}


def calculate_overall_metrics(df, threshold=0.5):
    """Calculate overall performance metrics for a dataset."""
    df_clean = df.dropna(subset=['true_value', 'predicted_value'])
    
    if len(df_clean) == 0:
        return {}
    
    true_vals = df_clean['true_value'].values
    pred_vals = df_clean['predicted_value'].values
    
    # Continuous metrics
    mae = float(np.mean(np.abs(true_vals - pred_vals)))
    rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
    
    if np.std(true_vals) > 1e-10 and np.std(pred_vals) > 1e-10:
        r, p = pearsonr(true_vals, pred_vals)
        r_squared = r ** 2
    else:
        r, p, r_squared = np.nan, np.nan, np.nan
    
    # Binary metrics
    y_true = _binary_labels_from_values(true_vals, threshold=threshold)
    y_pred = _binary_labels_from_values(pred_vals, threshold=threshold)
    bin_metrics = _confusion_and_metrics(y_true, y_pred)
    
    return {
        'n_predictions': len(df_clean),
        'n_drugs': df_clean['drug'].nunique(),
        'n_tfs': df_clean['TF'].nunique(),
        'mae': mae,
        'rmse': rmse,
        'pearson_r': r,
        'p_value': p,
        'r_squared': r_squared,
        'binary_precision': bin_metrics['precision'],
        'binary_recall': bin_metrics['recall'],
        'binary_accuracy': bin_metrics['accuracy']
    }


def plot_overall_scatter(df, output_dir, strength_label):
    """Plot overall scatter of all predictions."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(df['true_value'], df['predicted_value'],
               alpha=0.3, c='#4169E1', s=20)

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)

    r, p = pearsonr(df['true_value'], df['predicted_value'])

    ax.set_xlabel('True Activity', fontsize=14)
    ax.set_ylabel('Predicted Activity', fontsize=14)
    ax.set_title(f'LODO Validation: Stimulus Strength {strength_label}\n'
                 f'Pearson r = {r:.3f} (n = {len(df)})', fontsize=14)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_overall_scatter.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_best_worst(df, corr_df, n_best, n_worst, output_dir, strength_label):
    """Generate scatter plots for best and worst performing TFs."""
    best_tfs = corr_df.head(n_best)['TF'].values
    worst_tfs = corr_df.tail(n_worst)['TF'].values

    # Best TFs
    fig, axes = plt.subplots(1, n_best, figsize=(4*n_best, 4))
    if n_best == 1:
        axes = [axes]

    for i, tf in enumerate(best_tfs):
        tf_data = df[df['TF'] == tf]
        ax = axes[i]

        ax.scatter(tf_data['true_value'], tf_data['predicted_value'],
                   alpha=0.6, c='#1E90FF', s=50)

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

    plt.suptitle(f'Best TFs - Strength {strength_label}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_scatter_best_tfs.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Worst TFs
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

    plt.suptitle(f'Worst TFs - Strength {strength_label}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_scatter_worst_tfs.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_distribution(corr_df, output_dir, strength_label):
    """Plot distribution of correlations across TFs."""
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
    ax.set_title(f'TF Correlations - Strength {strength_label}', fontsize=14)
    ax.legend()

    # Bar plot
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


def plot_binary_confusion_matrix(df, output_dir, threshold, strength_label):
    """Generate confusion matrix plot."""
    y_true = _binary_labels_from_values(df['true_value'], threshold=threshold)
    y_pred = _binary_labels_from_values(df['predicted_value'], threshold=threshold)

    m = _confusion_and_metrics(y_true, y_pred)

    cm = np.array([[m['tn'], m['fp']],
                   [m['fn'], m['tp']]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred 0\n(inactive)', 'Pred 1\n(active)'])
    ax.set_yticklabels(['True 0\n(inactive)', 'True 1\n(active)'])
    ax.set_title(
        f'Confusion Matrix - Strength {strength_label}\n'
        f'Precision={m["precision"]:.3f}  Recall={m["recall"]:.3f}  Acc={m["accuracy"]:.3f}'
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_binary_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(df, tf_corr_df, drug_corr_df, overall_metrics, output_dir, strength_label):
    """Generate a summary report."""
    report = f"""
# LODO Validation Summary - Stimulus Strength {strength_label}

## Overview
- **Total predictions**: {overall_metrics['n_predictions']:,}
- **Number of drugs**: {overall_metrics['n_drugs']}
- **Number of TFs**: {overall_metrics['n_tfs']}

## Overall Performance
- **Pearson correlation**: {overall_metrics['pearson_r']:.4f}
- **R-squared**: {overall_metrics['r_squared']:.4f}
- **MAE**: {overall_metrics['mae']:.4f}
- **RMSE**: {overall_metrics['rmse']:.4f}

## Binary Classification (threshold=0.5)
- **Precision**: {overall_metrics['binary_precision']:.4f}
- **Recall**: {overall_metrics['binary_recall']:.4f}
- **Accuracy**: {overall_metrics['binary_accuracy']:.4f}

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
"""

    with open(os.path.join(output_dir, 'LODO_SUMMARY_REPORT.md'), 'w') as f:
        f.write(report)


def plot_selected_dose_level_bargraph_per_drug(choice_df, combined_df, output_dir, dose_col=None, strength_label=None):
    """
    Bar graph counts DRUGS (excluding drugs with only 1 available dose):
      - If drug has 1 dose: excluded
      - If drug has 2 doses: only low/high
      - If drug has >=3 doses: low/medium/high
    """
    print2log('  Generating per-drug selected-dose level bar graph (low/medium/high)...')

    if dose_col is None:
        dose_col = infer_dose_column(combined_df)
    print2log(f'  Using dose column: {dose_col}')

    tmp = combined_df.copy()
    tmp[dose_col] = pd.to_numeric(tmp[dose_col], errors='coerce')

    # Per-drug dose stats
    drug_stats = (
        tmp.groupby('drug')[dose_col]
        .agg(dose_min='min', dose_max='max', n_unique='nunique')
        .reset_index()
    )

    merged = choice_df.merge(drug_stats, on='drug', how='left')

    # Exclude drugs with only 1 dose (or missing stats)
    before = len(merged)
    merged = merged[(merged['n_unique'] >= 2) & merged['dose'].notna() & merged['dose_min'].notna() & merged['dose_max'].notna()].copy()
    excluded = before - len(merged)
    if excluded > 0:
        print2log(f'  Excluded {excluded} drugs from bar graph (missing stats or only 1 available dose).')

    def classify(row):
        # n_unique == 2 -> only low/high
        if int(row['n_unique']) == 2:
            if np.isclose(row['dose'], row['dose_min']):
                return 'low'
            if np.isclose(row['dose'], row['dose_max']):
                return 'high'
            return 'unknown'

        # n_unique >= 3 -> low/medium/high
        if np.isclose(row['dose'], row['dose_min']):
            return 'low'
        if np.isclose(row['dose'], row['dose_max']):
            return 'high'
        return 'medium'

    merged['dose_level_selected'] = merged.apply(classify, axis=1)

    counts = merged['dose_level_selected'].value_counts()
    n_unknown = int(counts.get('unknown', 0))
    if n_unknown > 0:
        print2log(f'  Warning: {n_unknown} drugs had unknown classification (unexpected dose value).')

    order = ['low', 'medium', 'high']
    plot_counts = [int(counts.get(k, 0)) for k in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(order, plot_counts, color=['#1E90FF', '#FFA500', '#FF6347'])

    ax.set_xlabel('Chosen dose relative to available doses', fontsize=12)
    ax.set_ylabel('Number of drugs', fontsize=12)
    
    # Add strength label to title if provided
    title = 'Per-drug chosen dose: low / medium / high\n(excluding drugs with 1 dose)'
    if strength_label:
        title = f'Strength {strength_label}: ' + title
    ax.set_title(title, fontsize=13)

    for b, v in zip(bars, plot_counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), str(v),
                ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'lodo_selected_dose_level_bargraph_per_drug.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()

    # Save tables
    merged.to_csv(os.path.join(output_dir, 'lodo_selected_drug_doses_for_bargraph.csv'), index=False)
    pd.DataFrame({'dose_level': order, 'count': plot_counts}).to_csv(
        os.path.join(output_dir, 'lodo_selected_dose_level_counts_per_drug.csv'),
        index=False
    )

    print2log(f'  Saved per-drug dose-level bar graph to {outpath}')

def process_single_strength(strength_info, args, output_base_dir):
    """Process a single stimulus strength."""
    strength_value = strength_info['strength']
    strength_folder = strength_info['path']
    strength_label = f"{strength_value}"
    
    print2log(f'\n{"="*80}')
    print2log(f'Processing Stimulus Strength: {strength_label}')
    print2log(f'{"="*80}')
    
    # Create output directory for this strength
    strength_output_dir = os.path.join(output_base_dir, f'strength_{strength_value}')
    Path(strength_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load predictions - THIS INCLUDES ALL DOSES
    combined_df = load_all_predictions(strength_folder, args.file_pattern)
    
    if combined_df is None or len(combined_df) == 0:
        print2log(f'  WARNING: No data found for strength {strength_label}, skipping...')
        return None
    
    print2log(f'  Total predictions (all doses): {len(combined_df)}')
    
    # Save combined data (all doses)
    combined_df.to_csv(os.path.join(strength_output_dir, 'lodo_combined_predictions.csv'), index=False)
    
    # Select optimal dose per drug - THIS FILTERS TO ONE DOSE PER DRUG
    optimal_df, drug_dose_choice_df = select_optimal_dose_per_drug_by_tf_correlation(combined_df)
    print2log(f'  Predictions after selecting optimal dose: {len(optimal_df)}')
    
    # Save optimal dose data
    optimal_df.to_csv(os.path.join(strength_output_dir, 'lodo_optimal_predictions.csv'), index=False)
    drug_dose_choice_df.to_csv(os.path.join(strength_output_dir, 'lodo_selected_drug_doses.csv'), index=False)
    
    # ALL SUBSEQUENT ANALYSIS USES optimal_df (ONLY OPTIMAL DOSE PER DRUG)
    
    # Calculate correlations - USING OPTIMAL DOSE ONLY
    tf_corr_df = calculate_tf_correlations(optimal_df)
    tf_corr_df.to_csv(os.path.join(strength_output_dir, 'lodo_tf_correlations.csv'), index=False)
    print2log(f'  TF correlations: mean={tf_corr_df["pearson_r"].mean():.3f}, median={tf_corr_df["pearson_r"].median():.3f}')
    
    drug_corr_df = calculate_drug_correlations(optimal_df)
    drug_corr_df.to_csv(os.path.join(strength_output_dir, 'lodo_drug_correlations.csv'), index=False)
    
    # Calculate overall metrics - USING OPTIMAL DOSE ONLY
    overall_metrics = calculate_overall_metrics(optimal_df, threshold=args.threshold)
    
    # Generate visualizations - ALL USING OPTIMAL DOSE ONLY
    plot_overall_scatter(optimal_df, strength_output_dir, strength_label)
    plot_scatter_best_worst(optimal_df, tf_corr_df, args.n_best, args.n_worst, 
                           strength_output_dir, strength_label)
    plot_correlation_distribution(tf_corr_df, strength_output_dir, strength_label)
    plot_binary_confusion_matrix(optimal_df, strength_output_dir, args.threshold, strength_label)
    
    plot_selected_dose_level_bargraph_per_drug(
        drug_dose_choice_df,
        combined_df,
        strength_output_dir,
        dose_col=None,
        strength_label=strength_label  # Pass the strength label
    )
    
    # Generate summary report
    generate_summary_report(optimal_df, tf_corr_df, drug_corr_df, overall_metrics, 
                           strength_output_dir, strength_label)
    
    print2log(f'  ✓ Completed processing for strength {strength_label}')
    
    # Return metrics for comparison - BASED ON OPTIMAL DOSE ONLY
    return {
        'strength': strength_value,
        'strength_label': strength_label,
        **overall_metrics
    }

def compare_strengths(all_metrics, comparison_output_dir):
    """Generate comparison analysis across all strengths."""
    print2log(f'\n{"="*80}')
    print2log('Generating Cross-Strength Comparison')
    print2log(f'{"="*80}')
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values('strength')
    
    # Save comprehensive metrics table
    metrics_df.to_csv(os.path.join(comparison_output_dir, 'strength_comparison_metrics.csv'), index=False)
    
    # Determine best strength by different criteria
    best_by_criteria = []
    
    criteria = {
        'pearson_r': ('Pearson Correlation', 'max'),
        'r_squared': ('R-squared', 'max'),
        'mae': ('MAE', 'min'),
        'rmse': ('RMSE', 'min'),
        'binary_accuracy': ('Binary Accuracy', 'max'),
        'binary_precision': ('Binary Precision', 'max'),
        'binary_recall': ('Binary Recall', 'max')
    }
    
    for metric, (label, direction) in criteria.items():
        if direction == 'max':
            best_idx = metrics_df[metric].idxmax()
        else:
            best_idx = metrics_df[metric].idxmin()
        
        best_strength = metrics_df.loc[best_idx, 'strength']
        best_value = metrics_df.loc[best_idx, metric]
        
        best_by_criteria.append({
            'criterion': label,
            'best_strength': best_strength,
            'value': best_value
        })
        
        print2log(f'  Best strength by {label}: {best_strength} (value={best_value:.4f})')
    
    best_df = pd.DataFrame(best_by_criteria)
    best_df.to_csv(os.path.join(comparison_output_dir, 'best_strength_by_criterion.csv'), index=False)
    
    # Overall recommendation (based on Pearson correlation)
    overall_best = metrics_df.loc[metrics_df['pearson_r'].idxmax(), 'strength']
    print2log(f'\n  ⭐ OVERALL RECOMMENDED STRENGTH: {overall_best}')
    
    # Generate comparison plots
    plot_metrics_comparison(metrics_df, comparison_output_dir)
    plot_strength_ranking(metrics_df, comparison_output_dir)
    
    # Generate comparison report
    generate_comparison_report(metrics_df, best_df, overall_best, comparison_output_dir)
    
    return metrics_df, overall_best


def plot_metrics_comparison(metrics_df, output_dir):
    """Plot comparison of key metrics across strengths."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics_to_plot = [
        ('pearson_r', 'Pearson Correlation', 'higher is better'),
        ('r_squared', 'R²', 'higher is better'),
        ('mae', 'MAE', 'lower is better'),
        ('rmse', 'RMSE', 'lower is better'),
        ('binary_accuracy', 'Binary Accuracy', 'higher is better'),
        ('binary_precision', 'Binary Precision', 'higher is better')
    ]
    
    for idx, (metric, title, direction) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        x = metrics_df['strength'].values
        y = metrics_df[metric].values
        
        # Plot line and markers
        ax.plot(x, y, marker='o', linewidth=2, markersize=8, color='#4169E1')
        
        # Highlight best value
        if 'higher' in direction:
            best_idx = np.argmax(y)
        else:
            best_idx = np.argmin(y)
        
        ax.scatter([x[best_idx]], [y[best_idx]], s=200, color='red', 
                  marker='*', zorder=5, label=f'Best: {x[best_idx]}')
        
        ax.set_xlabel('Stimulus Strength', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}\n({direction})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Format x-axis
        ax.set_xticks(x)
        ax.set_xticklabels([f'{val:.1f}' for val in x], rotation=45)
    
    plt.suptitle('Performance Metrics Across Stimulus Strengths', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print2log('  ✓ Generated metrics comparison plot')


def plot_strength_ranking(metrics_df, output_dir):
    """Plot ranking of strengths by different criteria."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Normalize metrics for radar/comparison plot
    metrics_to_rank = ['pearson_r', 'r_squared', 'binary_accuracy', 'binary_precision', 'binary_recall']
    
    # For MAE and RMSE, lower is better, so invert them
    normalized_df = metrics_df.copy()
    
    for metric in metrics_to_rank:
        if metric in ['mae', 'rmse']:
            # Invert so higher is better
            max_val = normalized_df[metric].max()
            normalized_df[f'{metric}_norm'] = 1 - (normalized_df[metric] / max_val)
        else:
            # Normalize to 0-1
            min_val = normalized_df[metric].min()
            max_val = normalized_df[metric].max()
            if max_val > min_val:
                normalized_df[f'{metric}_norm'] = (normalized_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[f'{metric}_norm'] = 0.5
    
    # Calculate composite score
    norm_cols = [f'{m}_norm' for m in metrics_to_rank]
    normalized_df['composite_score'] = normalized_df[norm_cols].mean(axis=1)
    
    # Plot 1: Composite scores
    ax = axes[0]
    strengths = normalized_df['strength'].values
    scores = normalized_df['composite_score'].values
    
    bars = ax.bar(range(len(strengths)), scores, color='#4169E1', alpha=0.7)
    
    # Highlight best
    best_idx = np.argmax(scores)
    bars[best_idx].set_color('#FF6347')
    bars[best_idx].set_alpha(1.0)
    
    ax.set_xlabel('Stimulus Strength', fontsize=12)
    ax.set_ylabel('Composite Score', fontsize=12)
    ax.set_title('Overall Performance Score\n(normalized average across all metrics)', fontsize=13)
    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([f'{s:.1f}' for s in strengths], rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Heatmap of normalized metrics
    ax = axes[1]
    
    metric_labels = ['Pearson r', 'R²', 'Accuracy', 'Precision', 'Recall']
    heatmap_data = normalized_df[norm_cols].T.values
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([f'{s:.1f}' for s in strengths], rotation=45)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel('Stimulus Strength', fontsize=12)
    ax.set_title('Normalized Performance Heatmap\n(1.0 = best, 0.0 = worst)', fontsize=13)
    
    # Add text annotations
    for i in range(len(metric_labels)):
        for j in range(len(strengths)):
            text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strength_ranking.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save composite scores
    normalized_df[['strength', 'composite_score'] + metrics_to_rank].to_csv(
        os.path.join(output_dir, 'strength_composite_scores.csv'), index=False
    )
    
    print2log('  ✓ Generated strength ranking plots')


def plot_per_drug_strength_comparison(strength_folders, output_base_dir, comparison_output_dir):
    """Compare per-drug performance across strengths."""
    print2log('  Generating per-drug strength comparison...')
    
    # Load drug correlations for each strength
    all_drug_corrs = []
    
    for sf in strength_folders:
        strength_val = sf['strength']
        corr_file = os.path.join(output_base_dir, f'strength_{strength_val}', 
                                'lodo_drug_correlations.csv')
        
        if os.path.exists(corr_file):
            df = pd.read_csv(corr_file)
            df['strength'] = strength_val
            all_drug_corrs.append(df)
    
    if not all_drug_corrs:
        print2log('    WARNING: No drug correlation files found')
        return
    
    combined_drug_corrs = pd.concat(all_drug_corrs, ignore_index=True)
    
    # Pivot to get drugs x strengths
    pivot_df = combined_drug_corrs.pivot_table(
        index='drug', 
        columns='strength', 
        values='pearson_r'
    )
    
    # Find best strength for each drug
    best_strength_per_drug = pivot_df.idxmax(axis=1)
    best_strength_counts = best_strength_per_drug.value_counts().sort_index()
    
    # Save
    best_strength_per_drug.to_csv(
        os.path.join(comparison_output_dir, 'best_strength_per_drug.csv'),
        header=['best_strength']
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strengths = best_strength_counts.index.values
    counts = best_strength_counts.values
    
    bars = ax.bar(range(len(strengths)), counts, color='#4169E1', alpha=0.7)
    
    # Highlight the most common
    max_idx = np.argmax(counts)
    bars[max_idx].set_color('#FF6347')
    
    ax.set_xlabel('Stimulus Strength', fontsize=12)
    ax.set_ylabel('Number of Drugs', fontsize=12)
    ax.set_title('Best Performing Strength per Drug\n(by Pearson correlation)', fontsize=14)
    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([f'{s:.1f}' for s in strengths], rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_output_dir, 'best_strength_per_drug.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print2log(f'    Most drugs ({int(counts.max())}) perform best at strength {strengths[max_idx]:.1f}')


def plot_per_tf_strength_comparison(strength_folders, output_base_dir, comparison_output_dir):
    """Compare per-TF performance across strengths."""
    print2log('  Generating per-TF strength comparison...')
    
    # Load TF correlations for each strength
    all_tf_corrs = []
    
    for sf in strength_folders:
        strength_val = sf['strength']
        corr_file = os.path.join(output_base_dir, f'strength_{strength_val}', 
                                'lodo_tf_correlations.csv')
        
        if os.path.exists(corr_file):
            df = pd.read_csv(corr_file)
            df['strength'] = strength_val
            all_tf_corrs.append(df)
    
    if not all_tf_corrs:
        print2log('    WARNING: No TF correlation files found')
        return
    
    combined_tf_corrs = pd.concat(all_tf_corrs, ignore_index=True)
    
    # Pivot to get TFs x strengths
    pivot_df = combined_tf_corrs.pivot_table(
        index='TF', 
        columns='strength', 
        values='pearson_r'
    )
    
    # Find best strength for each TF
    best_strength_per_tf = pivot_df.idxmax(axis=1)
    best_strength_counts = best_strength_per_tf.value_counts().sort_index()
    
    # Save
    best_strength_per_tf.to_csv(
        os.path.join(comparison_output_dir, 'best_strength_per_tf.csv'),
        header=['best_strength']
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strengths = best_strength_counts.index.values
    counts = best_strength_counts.values
    
    bars = ax.bar(range(len(strengths)), counts, color='#4169E1', alpha=0.7)
    
    # Highlight the most common
    max_idx = np.argmax(counts)
    bars[max_idx].set_color('#FF6347')
    
    ax.set_xlabel('Stimulus Strength', fontsize=12)
    ax.set_ylabel('Number of TFs', fontsize=12)
    ax.set_title('Best Performing Strength per TF\n(by Pearson correlation)', fontsize=14)
    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([f'{s:.1f}' for s in strengths], rotation=45)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_output_dir, 'best_strength_per_tf.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print2log(f'    Most TFs ({int(counts.max())}) perform best at strength {strengths[max_idx]:.1f}')


def generate_comparison_report(metrics_df, best_df, overall_best, output_dir):
    """Generate comprehensive comparison report."""
    
    # Find which strength wins most criteria
    strength_wins = best_df['best_strength'].value_counts()
    most_wins_strength = strength_wins.idxmax()
    most_wins_count = strength_wins.max()
    
    report = f"""
# Multi-Strength LODO Validation Comparison Report

## Executive Summary

**RECOMMENDED STIMULUS STRENGTH: {overall_best}**

This strength achieved the highest Pearson correlation and demonstrates the best overall predictive performance.

---

## Comparison Overview

Analyzed **{len(metrics_df)}** different stimulus strengths:
{', '.join([f'{s:.1f}' for s in metrics_df['strength'].values])}

---

## Best Strength by Criterion

| Criterion | Best Strength | Value |
|-----------|--------------|-------|
"""
    
    for _, row in best_df.iterrows():
        report += f"| {row['criterion']} | {row['best_strength']:.1f} | {row['value']:.4f} |\n"
    
    report += f"""

**Strength {most_wins_strength:.1f}** wins the most criteria ({int(most_wins_count)} out of {len(best_df)})

---

## Detailed Metrics by Strength

| Strength | Pearson r | R² | MAE | RMSE | Binary Acc | Binary Prec | Binary Recall |
|----------|-----------|-----|-----|------|-----------|-------------|---------------|
"""
    
    for _, row in metrics_df.iterrows():
        report += (f"| {row['strength']:.1f} | {row['pearson_r']:.4f} | "
                  f"{row['r_squared']:.4f} | {row['mae']:.4f} | {row['rmse']:.4f} | "
                  f"{row['binary_accuracy']:.4f} | {row['binary_precision']:.4f} | "
                  f"{row['binary_recall']:.4f} |\n")
    
    report += """

---

## Performance Trends

### Key Observations:

"""
    
    # Add observations about trends
    pearson_trend = 'increasing' if metrics_df['pearson_r'].iloc[-1] > metrics_df['pearson_r'].iloc[0] else 'decreasing'
    report += f"1. **Pearson correlation** shows a {pearson_trend} trend as stimulus strength increases\n"
    
    best_pearson_strength = metrics_df.loc[metrics_df['pearson_r'].idxmax(), 'strength']
    best_pearson_value = metrics_df['pearson_r'].max()
    report += f"2. **Peak performance** at strength {best_pearson_strength:.1f} (r = {best_pearson_value:.4f})\n"
    
    mae_trend = 'increasing' if metrics_df['mae'].iloc[-1] > metrics_df['mae'].iloc[0] else 'decreasing'
    report += f"3. **MAE** is {mae_trend} with higher stimulus strengths\n"
    
    report += """

---

## Recommendations

### For General Use:
"""
    
    report += f"- **Primary recommendation**: Use stimulus strength **{overall_best:.1f}**\n"
    report += f"  - Achieves best Pearson correlation: {metrics_df.loc[metrics_df['strength']==overall_best, 'pearson_r'].values[0]:.4f}\n"
    report += f"  - R²: {metrics_df.loc[metrics_df['strength']==overall_best, 'r_squared'].values[0]:.4f}\n"
    
    # Alternative recommendations
    sorted_by_pearson = metrics_df.sort_values('pearson_r', ascending=False)
    if len(sorted_by_pearson) > 1:
        second_best = sorted_by_pearson.iloc[1]
        report += f"\n- **Alternative option**: Stimulus strength **{second_best['strength']:.1f}**\n"
        report += f"  - Pearson correlation: {second_best['pearson_r']:.4f}\n"
    
    report += """

### For Specific Applications:

"""
    
    # Recommendations by use case
    best_binary_acc = metrics_df.loc[metrics_df['binary_accuracy'].idxmax()]
    report += f"- **Binary classification tasks**: Strength {best_binary_acc['strength']:.1f} (accuracy = {best_binary_acc['binary_accuracy']:.4f})\n"
    
    best_mae = metrics_df.loc[metrics_df['mae'].idxmin()]
    report += f"- **Minimizing absolute error**: Strength {best_mae['strength']:.1f} (MAE = {best_mae['mae']:.4f})\n"
    
    best_precision = metrics_df.loc[metrics_df['binary_precision'].idxmax()]
    report += f"- **High precision required**: Strength {best_precision['strength']:.1f} (precision = {best_precision['binary_precision']:.4f})\n"
    
    best_recall = metrics_df.loc[metrics_df['binary_recall'].idxmax()]
    report += f"- **High recall required**: Strength {best_recall['strength']:.1f} (recall = {best_recall['binary_recall']:.4f})\n"
    
    report += """

---

## Files Generated

### Individual Strength Results:
"""
    
    for strength_val in metrics_df['strength'].values:
        report += f"\n#### Strength {strength_val:.1f}\n"
        report += f"- Location: `strength_{strength_val:.1f}/`\n"
        report += "- Files: predictions, correlations, plots, and summary report\n"
    
    report += """

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
"""
    
    with open(os.path.join(output_dir, 'COMPARISON_REPORT.md'), 'w') as f:
        f.write(report)
    
    print2log('  ✓ Generated comparison report')

def plot_dose_selection_comparison_across_strengths(strength_folders, output_base_dir, comparison_output_dir):
    """Create a panel comparing dose selection (low/medium/high) across all strengths."""
    print2log('  Generating dose selection comparison across strengths...')
    
    # Load dose level counts for each strength
    all_dose_data = []
    
    for sf in strength_folders:
        strength_val = sf['strength']
        counts_file = os.path.join(output_base_dir, f'strength_{strength_val}', 
                                   'lodo_selected_dose_level_counts_per_drug.csv')
        
        if os.path.exists(counts_file):
            df = pd.read_csv(counts_file)
            df['strength'] = strength_val
            all_dose_data.append(df)
        else:
            print2log(f'    WARNING: Dose counts file not found for strength {strength_val}')
    
    if not all_dose_data:
        print2log('    WARNING: No dose selection data found')
        return
    
    combined_dose_data = pd.concat(all_dose_data, ignore_index=True)
    
    # Create pivot table: dose_level x strength
    pivot_df = combined_dose_data.pivot_table(
        index='dose_level', 
        columns='strength', 
        values='count',
        fill_value=0
    )
    
    # Ensure order: low, medium, high
    dose_order = ['low', 'medium', 'high']
    pivot_df = pivot_df.reindex(dose_order, fill_value=0)
    
    n_strengths = len(strength_folders)
    
    # Determine layout: try to make roughly square
    if n_strengths <= 3:
        n_cols = n_strengths
        n_rows = 1
    elif n_strengths <= 6:
        n_cols = 3
        n_rows = 2
    elif n_strengths <= 9:
        n_cols = 3
        n_rows = 3
    else:
        n_cols = 4
        n_rows = int(np.ceil(n_strengths / 4))
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Flatten axes array for easier indexing
    if n_strengths == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    colors = ['#1E90FF', '#FFA500', '#FF6347']  # Blue, Orange, Red for low, medium, high
    
    # Plot each strength in a subplot
    for idx, sf in enumerate(strength_folders):
        strength_val = sf['strength']
        ax = axes[idx]
        
        if strength_val in pivot_df.columns:
            counts = pivot_df[strength_val].values
        else:
            counts = [0, 0, 0]
        
        bars = ax.bar(dose_order, counts, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(val)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Selected Dose Level', fontsize=11)
        ax.set_ylabel('Number of Drugs', fontsize=11)
        ax.set_title(f'Strength {strength_val:.1f}', fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)
    
    # Hide extra subplots if any
    for idx in range(n_strengths, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Dose Selection Distribution Across Stimulus Strengths\n(low / medium / high relative to available doses per drug)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    outpath = os.path.join(comparison_output_dir, 'dose_selection_comparison_across_strengths.png')
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print2log(f'    Saved dose selection comparison panel to {outpath}')
    
    # Also create a stacked bar chart for direct comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strengths = sorted(pivot_df.columns)
    x = np.arange(len(strengths))
    width = 0.6
    
    bottom = np.zeros(len(strengths))
    
    for idx, dose_level in enumerate(dose_order):
        values = [pivot_df.loc[dose_level, s] if s in pivot_df.columns else 0 for s in strengths]
        ax.bar(x, values, width, label=dose_level.capitalize(), 
               bottom=bottom, color=colors[idx], alpha=0.8)
        
        # Add value labels
        for i, val in enumerate(values):
            if val > 0:
                ax.text(i, bottom[i] + val/2, f'{int(val)}',
                       ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        bottom += values
    
    ax.set_xlabel('Stimulus Strength', fontsize=13)
    ax.set_ylabel('Number of Drugs', fontsize=13)
    ax.set_title('Dose Selection Distribution: Stacked Comparison\nAcross All Stimulus Strengths', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s:.1f}' for s in strengths], rotation=45)
    ax.legend(title='Dose Level', fontsize=11, title_fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    stacked_outpath = os.path.join(comparison_output_dir, 'dose_selection_stacked_comparison.png')
    plt.savefig(stacked_outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print2log(f'    Saved stacked dose selection comparison to {stacked_outpath}')
    
    # Save summary table
    pivot_df.to_csv(os.path.join(comparison_output_dir, 'dose_selection_summary_by_strength.csv'))
    
    # Calculate and save percentages
    percent_df = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 100
    percent_df.to_csv(os.path.join(comparison_output_dir, 'dose_selection_percentages_by_strength.csv'))
    
    print2log('    Saved dose selection summary tables')

def main():
    args = parse_arguments()
    
    # Find all strength folders
    strength_folders = find_strength_folders(args.results_dir)
    
    if not strength_folders:
        print2log('ERROR: No strength folders found!')
        return
    
    # Setup output directories
    output_base_dir = args.output_dir or os.path.join(args.results_dir, 'analysis_results')
    comparison_output_dir = os.path.join(output_base_dir, 'comparison')
    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    Path(comparison_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each strength
    all_metrics = []
    
    for strength_info in strength_folders:
        metrics = process_single_strength(strength_info, args, output_base_dir)
        if metrics is not None:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print2log('ERROR: No valid strength data processed!')
        return
    
    # Compare strengths
    metrics_df, overall_best = compare_strengths(all_metrics, comparison_output_dir)
    
    # Additional comparison analyses
    plot_per_drug_strength_comparison(strength_folders, output_base_dir, comparison_output_dir)
    plot_per_tf_strength_comparison(strength_folders, output_base_dir, comparison_output_dir)
    plot_dose_selection_comparison_across_strengths(strength_folders, output_base_dir, comparison_output_dir)
    
    print2log(f'\n{"="*80}')
    print2log('✅ Multi-Strength LODO Analysis Complete!')
    print2log(f'{"="*80}')
    print2log(f'\nResults saved to: {output_base_dir}')
    print2log(f'Comparison results: {comparison_output_dir}')
    print2log(f'\n⭐ RECOMMENDED STIMULUS STRENGTH: {overall_best}')
    print2log(f'\nSee COMPARISON_REPORT.md for detailed analysis and recommendations.')


if __name__ == '__main__':
    main()