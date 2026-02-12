#!/usr/bin/env python
"""
Batch Validation Script - Test Multiple Stimulation Strengths

This script runs validation for a single drug across multiple stimulation strengths
and compares the results.

Usage:
    python lodo_batch_validate_strengths.py \
        --drug_index 0 \
        --strengths 10 25 50 75 100 150 200 \
        --model_dir ../results/LODO/8_hour/
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
# Setup LEMBAS path - adjust as needed for your environment
gpu_lembas_path = '/home/meimetis/LEMBAS-main'
sys.path.insert(1, os.path.join(gpu_lembas_path))
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Batch validate LODO model with multiple stimulation strengths'
    )
    
    parser.add_argument('--drug_index', type=int, required=True,
                       help='Index of drug to validate')
    parser.add_argument('--strengths', type=float, nargs='+', required=True,
                       help='List of stimulation strengths to test (e.g., 10 25 50 100 200)')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base output directory (default: model_dir/strength_comparison)')
    parser.add_argument('--timepoint', type=str, default='8 hour')
    
    # Pass-through arguments for validation script
    parser.add_argument('--DrugsIn', type=str,
                       default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
    parser.add_argument('--TargetsIn', type=str,
                       default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
    parser.add_argument('--TFsOut', type=str,
                       default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
    parser.add_argument('--ChemicalSims', type=str,
                       default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
    parser.add_argument('--PKN', type=str,
                       default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
    parser.add_argument('--PknAnnotation', type=str,
                       default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
    parser.add_argument('--infoPath', type=str,
                       default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
    parser.add_argument('--safe_drugs_file', type=str,
                       default='../preprocessing/preprocessed_data/LeaveOneDrugOut/safe_drugs_to_remove_LODO.csv')
    parser.add_argument('--signs_file', type=str,
                       default='../preprocessing/preprocessed_data/LeaveOneDrugOut/consensus_signs_majority_vote.csv')
    
    return parser.parse_args()


def run_validation_for_strength(args, strength):
    """Run validation for a single stimulation strength."""
    print2log(f"\n{'='*70}")
    print2log(f"Testing stimulation strength: {strength}")
    print2log('='*70)
    
    output_dir = os.path.join(args.output_dir, f'strength_{strength}')
    
    # Build command
    cmd = [
        'python', 'lodo_validate_different_strength.py',
        '--drug_index', str(args.drug_index),
        '--stimulation_strength', str(strength),
        '--model_dir', args.model_dir,
        '--output_dir', output_dir,
        '--timepoint', args.timepoint,
        '--DrugsIn', args.DrugsIn,
        '--TargetsIn', args.TargetsIn,
        '--TFsOut', args.TFsOut,
        '--ChemicalSims', args.ChemicalSims,
        '--PKN', args.PKN,
        '--PknAnnotation', args.PknAnnotation,
        '--infoPath', args.infoPath,
        '--safe_drugs_file', args.safe_drugs_file,
        '--signs_file', args.signs_file
    ]
    
    # Run validation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print2log(f"Error running validation for strength {strength}:")
        print2log(result.stderr)
        return None
    
    # Load summary
    summary_file = os.path.join(output_dir, f'drug{args.drug_index}_summary_strength{strength}.csv')
    if os.path.exists(summary_file):
        return pd.read_csv(summary_file)
    else:
        print2log(f"Warning: Summary file not found for strength {strength}")
        return None


def create_comparison_plots(summaries_df, output_dir, drug_index):
    """Create plots comparing different stimulation strengths."""
    sns.set_style("whitegrid")
    
    # 1. Correlation vs Stimulation Strength
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Overall correlation
    axes[0].plot(summaries_df['stimulation_strength'], 
                summaries_df['overall_correlation'], 
                'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Stimulation Strength', fontsize=12)
    axes[0].set_ylabel('Overall Pearson Correlation', fontsize=12)
    axes[0].set_title('Overall Correlation vs Stimulation Strength', fontsize=14)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Mean per-TF correlation
    axes[1].plot(summaries_df['stimulation_strength'], 
                summaries_df['mean_per_tf_correlation'], 
                'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Stimulation Strength', fontsize=12)
    axes[1].set_ylabel('Mean Per-TF Correlation', fontsize=12)
    axes[1].set_title('Mean Per-TF Correlation vs Stimulation Strength', fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Median per-TF correlation
    axes[2].plot(summaries_df['stimulation_strength'], 
                summaries_df['median_per_tf_correlation'], 
                'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Stimulation Strength', fontsize=12)
    axes[2].set_ylabel('Median Per-TF Correlation', fontsize=12)
    axes[2].set_title('Median Per-TF Correlation vs Stimulation Strength', fontsize=14)
    axes[2].grid(alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'drug{drug_index}_strength_comparison.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Find optimal strength
    optimal_idx = summaries_df['overall_correlation'].idxmax()
    optimal_strength = summaries_df.loc[optimal_idx, 'stimulation_strength']
    optimal_corr = summaries_df.loc[optimal_idx, 'overall_correlation']
    
    print2log(f"\n{'='*70}")
    print2log(f"OPTIMAL STIMULATION STRENGTH ANALYSIS")
    print2log('='*70)
    print2log(f"Best overall correlation: {optimal_corr:.4f} at strength {optimal_strength}")
    print2log(f"Mean per-TF correlation at optimal: {summaries_df.loc[optimal_idx, 'mean_per_tf_correlation']:.4f}")
    print2log(f"Median per-TF correlation at optimal: {summaries_df.loc[optimal_idx, 'median_per_tf_correlation']:.4f}")
    
    # 3. Create table
    fig, ax = plt.subplots(figsize=(12, max(4, len(summaries_df) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = summaries_df[['stimulation_strength', 'overall_correlation', 
                               'mean_per_tf_correlation', 'median_per_tf_correlation']].copy()
    table_data.columns = ['Strength', 'Overall r', 'Mean r', 'Median r']
    table_data = table_data.round(4)
    
    # Highlight optimal row
    colors = [['lightgreen' if i == optimal_idx else 'white' for _ in range(len(table_data.columns))] 
              for i in range(len(table_data))]
    
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center',
                    cellColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title(f'Stimulation Strength Comparison - Drug {drug_index}\n'
             f'(Optimal strength highlighted)', fontsize=14, pad=20)
    
    plt.savefig(os.path.join(output_dir, f'drug{drug_index}_strength_table.png'),
               dpi=300, bbox_inches='tight')
    plt.close()


def load_all_predictions(args, strengths):
    """Load all predictions for detailed analysis."""
    all_predictions = []
    
    for strength in strengths:
        pred_file = os.path.join(
            args.output_dir, 
            f'strength_{strength}',
            f'drug{args.drug_index}_lodo_prediction_strength{strength}.csv'
        )
        
        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file)
            all_predictions.append(df)
    
    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    return None


def create_heatmap_per_tf(all_predictions, output_dir, drug_index):
    """Create heatmap showing per-TF correlation across strengths."""
    from scipy.stats import pearsonr
    
    # Compute correlation for each TF at each strength
    strengths = sorted(all_predictions['stimulation_strength'].unique())
    tfs = sorted(all_predictions['TF'].unique())
    
    corr_matrix = np.zeros((len(tfs), len(strengths)))
    
    for i, tf in enumerate(tfs):
        for j, strength in enumerate(strengths):
            data = all_predictions[
                (all_predictions['TF'] == tf) & 
                (all_predictions['stimulation_strength'] == strength)
            ]
            if len(data) > 1:
                r, _ = pearsonr(data['true_value'], data['predicted_value'])
                corr_matrix[i, j] = r
            else:
                corr_matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(strengths)), max(8, len(tfs) * 0.3)))
    
    im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([f'{s:.0f}' for s in strengths], fontsize=10)
    ax.set_yticks(range(len(tfs)))
    ax.set_yticklabels(tfs, fontsize=8)
    
    ax.set_xlabel('Stimulation Strength', fontsize=12)
    ax.set_ylabel('Transcription Factor', fontsize=12)
    ax.set_title(f'Per-TF Correlation Across Stimulation Strengths\nDrug {drug_index}', 
                fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', fontsize=12)
    
    # Add values in cells
    for i in range(len(tfs)):
        for j in range(len(strengths)):
            if not np.isnan(corr_matrix[i, j]):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'drug{drug_index}_per_tf_heatmap.png'),
               dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_arguments()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'strength_comparison')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print2log(f"Batch validation for drug {args.drug_index}")
    print2log(f"Testing {len(args.strengths)} stimulation strengths: {args.strengths}")
    print2log(f"Output directory: {args.output_dir}")
    
    # Run validation for each strength
    summaries = []
    for strength in args.strengths:
        summary = run_validation_for_strength(args, strength)
        if summary is not None:
            summaries.append(summary)
    
    if not summaries:
        print2log("No successful validations - exiting")
        return
    
    # Combine summaries
    summaries_df = pd.concat(summaries, ignore_index=True)
    summaries_df = summaries_df.sort_values('stimulation_strength')
    
    # Save combined summary
    combined_file = os.path.join(args.output_dir, f'drug{args.drug_index}_all_strengths_summary.csv')
    summaries_df.to_csv(combined_file, index=False)
    print2log(f"\nCombined summary saved to: {combined_file}")
    
    # Create comparison plots
    create_comparison_plots(summaries_df, args.output_dir, args.drug_index)
    
    # Load all predictions for heatmap
    all_predictions = load_all_predictions(args, args.strengths)
    if all_predictions is not None:
        create_heatmap_per_tf(all_predictions, args.output_dir, args.drug_index)
        print2log(f"Per-TF heatmap saved")
    
    print2log(f"\n{'='*70}")
    print2log("BATCH VALIDATION COMPLETE")
    print2log('='*70)
    print2log(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
