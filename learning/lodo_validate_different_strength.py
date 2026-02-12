"""
Validate LODO Models with Different Stimulation Strengths

This script loads already-trained LODO models and re-runs validation
with different stimulation strengths without retraining.

Usage:
    python lodo_validate_different_strength.py \
        --drug_index 0 \
        --stimulation_strength 50 \
        --model_dir ../results/LODO/8_hour/
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import logging
from pathlib import Path
# Setup LEMBAS path - adjust as needed for your environment
gpu_lembas_path = '/home/meimetis/LEMBAS-main'
sys.path.insert(1, os.path.join(gpu_lembas_path))
from LEMBAS.model.bionetwork import format_network, SignalingModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info


class DrugstoSignal(nn.Module):
    """Drug-to-Signal neural network layer."""

    def __init__(self, mask, size1, size2, drugSim, momentum=0.6, drop=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.size_in = size1  # drugs number
        self.size_out = size2  # targets/signal size
        self.mask = mask.double().to(device)
        self.drugSim = drugSim.to(device)
        self.leak = 0.01

        # Initialize weights
        weights = torch.Tensor(size2, size1).double()
        nn.init.uniform_(weights, -0.3, 0.01)
        weights = torch.mul(weights.to(device), self.mask)

        ecfp4_weights = torch.Tensor(size1, size1).double()
        nn.init.uniform_(ecfp4_weights, 0., 1.)

        self.Wsim = nn.Parameter(ecfp4_weights, requires_grad=True)
        self.bn = nn.BatchNorm1d(num_features=size1, momentum=momentum, dtype=torch.double)
        self.dropout = nn.Dropout(drop)
        self.A = nn.Parameter(weights, requires_grad=True)

    def forward(self, x):
        W = torch.mul(self.drugSim, self.Wsim)
        A = torch.mul(self.A, self.mask)
        xhat = torch.mm(x, W)
        xhat = self.bn(xhat)
        xhat = self.dropout(xhat)
        xhat = xhat.transpose(0, 1)
        xhat = torch.mm(A, xhat)
        output = xhat.transpose(0, 1)
        return output


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog='LODO Validation with Different Stimulation Strengths',
        description='Re-validate trained LODO models with different stimulation strengths'
    )

    # Data paths (same as training)
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

    # LODO-specific files
    parser.add_argument('--safe_drugs_file', type=str,
                        default='../preprocessing/preprocessed_data/LeaveOneDrugOut/safe_drugs_to_remove_LODO.csv')
    parser.add_argument('--signs_file', type=str,
                        default='../preprocessing/preprocessed_data/LeaveOneDrugOut/consensus_signs_majority_vote.csv')

    # Model paths
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained models (e.g., ../results/LODO/8_hour/)')
    parser.add_argument('--outPattern', type=str, default='LODO_hepatocytes')
    parser.add_argument('--drug_index', type=int, required=True,
                        help='Index of drug to validate (0-based, must match trained model)')

    # Validation parameters
    parser.add_argument('--stimulation_strength', type=float, required=True,
                        help='New stimulation strength to test')
    parser.add_argument('--timepoint', type=str, default='8 hour')

    # Model architecture parameters (must match training)
    parser.add_argument('--inputAmplitude', type=float, default=3)
    parser.add_argument('--projectionAmplitude', type=float, default=1.2)
    parser.add_argument('--max_LEMBAS_iterations', type=int, default=90)
    parser.add_argument('--LEMBAS_leak', type=float, default=0.01)
    parser.add_argument('--LEMBAS_tolerance', type=float, default=1e-5)
    parser.add_argument('--drug_bn', type=float, default=0.2)
    parser.add_argument('--drug_drop', type=float, default=0.1)

    # Output directory
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: model_dir + strength_{value})')

    return parser.parse_args()

def remove_drug_from_data(data, drug_to_remove, device):
    """
    Correct removal logic (matches your training script):
    - Keep the held-out drug column in drugInput but set it to 0
    - Remove samples that become drug-less after removal
    - Remove the held-out drug from drugTargets rows and drugSim rows/cols
    - Return mask_dt + drugSim tensor + drugs_to_keep
    """
    drugInput = data['drugInput'].copy()
    drugTargets = data['drugTargets'].copy()
    drugSim = data['drugSim'].copy()

    all_drugs = drugInput.columns.tolist()
    if drug_to_remove not in all_drugs:
        raise ValueError(f"Drug {drug_to_remove} not found in drugInput columns")

    # Drugs to keep (exclude held-out)
    drugs_to_keep = [d for d in all_drugs if d != drug_to_remove]

    # Filter drugTargets (rows = drugs)
    drugTargets_filtered = drugTargets.loc[drugs_to_keep, :]

    # Filter drug similarity matrix
    drugSim_filtered = drugSim.loc[drugs_to_keep, drugs_to_keep]

    # Zero out the held-out drug in drugInput (keep columns for consistency)
    drugInput_filtered = drugInput.copy()
    drugInput_filtered[drug_to_remove] = 0

    # Keep samples that have at least one non-zero drug after removal
    samples_with_other_drugs = (drugInput_filtered.drop(columns=[drug_to_remove]).sum(axis=1) > 0)

    # Filter TFOutput and drugInput accordingly
    TFOutput_filtered = data['TFOutput'].loc[samples_with_other_drugs, :]
    drugInput_filtered = drugInput_filtered.loc[samples_with_other_drugs, :]

    # Create new mask without the removed drug
    mask_dt = torch.tensor(np.array(drugTargets_filtered), dtype=torch.double, device=device).transpose(0, 1)
    drugSim_tensor = torch.tensor(drugSim_filtered.values.copy(), dtype=torch.double, device=device)

    return {
        'drugInput': drugInput_filtered,
        'TFOutput': TFOutput_filtered,
        'drugTargets': drugTargets_filtered,
        'mask_dt': mask_dt,
        'drugSim': drugSim_tensor,
        'drugs_to_keep': drugs_to_keep
    }


def load_data(args):
    """Load and preprocess all required data (same as training)."""
    print2log('Loading input data...')

    # Prior knowledge signaling network
    net = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[:, ['source', 'target', 'stimulation', 'inhibition']]

    # Drug inputs and TF outputs
    drugInput = pd.read_csv(args.DrugsIn, sep='\t', index_col=0)
    TFOutput = pd.read_csv(args.TFsOut, sep='\t', low_memory=False, index_col=0)
    TFOutput = TFOutput.loc[drugInput.index, :]

    # Drug targets
    drugTargets = pd.read_csv(args.TargetsIn, sep='\t', low_memory=False, index_col=0)

    # Filter by timepoint
    inputInfo = pd.read_csv(args.infoPath, sep='\t', low_memory=False, index_col=0)
    inputInfo = inputInfo[inputInfo['time'] == args.timepoint]
    TFOutput = TFOutput.loc[inputInfo.index, :]
    drugInput = drugInput.loc[inputInfo.index, :]

    # Format network
    net = format_network(net, weight_label='mode_of_action',
                        stimulation_label='stimulation',
                        inhibition_label='inhibition')
    nodeNames = np.union1d(net.source.values, net.target.values)

    # Subset to intersecting nodes
    inName = np.intersect1d(nodeNames, drugTargets.columns.values)
    outName = np.intersect1d(nodeNames, TFOutput.columns.values)
    TFOutput = TFOutput.loc[:, outName]
    drugTargets = drugTargets.loc[:, inName]
    drugTargets = drugTargets.loc[drugInput.columns.values, :]

    # Drug similarity
    drugSim = pd.read_csv(args.ChemicalSims, index_col=0)
    drugSim = drugSim.loc[drugInput.columns.values, drugInput.columns.values]

    # Load safe drugs
    safe_drugs_df = pd.read_csv(args.safe_drugs_file)
    safe_drugs = safe_drugs_df.loc[
        safe_drugs_df["Status"].astype(str).str.strip().str.lower().eq("safe"),
        "Drug"
    ].astype(str).values

    # Load signs
    signs_df = pd.read_csv(args.signs_file, index_col=0)

    return {
        'net': net,
        'drugInput': drugInput,
        'TFOutput': TFOutput,
        'drugTargets': drugTargets,
        'drugSim': drugSim,
        'safe_drugs': safe_drugs,
        'signs_df': signs_df,
        'inputInfo': inputInfo
    }


def load_trained_model(args, data, drug_held_out, device):
    """Load the trained model for this drug."""
    print2log(f'Loading trained model for drug {args.drug_index}...')
    filtered_data = remove_drug_from_data(data, drug_held_out, device)
    
    # Reconstruct filtered data (drug removed)
    drugInput_filtered = filtered_data['drugInput']
    drugTargets_filtered = filtered_data['drugTargets']
    # Note: remove_drug_from_data already converts the drugSim and mask_dt to torch tensors.
    # We simply re-use these tensors here. If drugSim_filtered is already a tensor, no need to copy values.
    drugSim_filtered = filtered_data['drugSim']
    drugSim_tensor = drugSim_filtered  # Already a torch tensor
    mask_dt = filtered_data['mask_dt']  # Already a torch tensor

    # Model parameters
    bionet_params = {
        'target_steps': 100,
        'max_steps': args.max_LEMBAS_iterations,
        'exp_factor': 50,
        'tolerance': args.LEMBAS_tolerance,
        'leak': args.LEMBAS_leak
    }

    # Reconstruct SignalingModel
    mod = SignalingModel(
        net=data['net'],
        X_in=drugTargets_filtered,
        y_out=data['TFOutput'],
        projection_amplitude_in=args.inputAmplitude,
        projection_amplitude_out=args.projectionAmplitude,
        weight_label='mode_of_action',
        source_label='source',
        target_label='target',
        bionet_params=bionet_params,
        dtype=torch.double,
        device=device,
        # Always provide a valid integer seed to avoid TypeError in torch.manual_seed.
        # If a seed isn't provided via args, use a default like 0.
        seed=0
    )

    # Reconstruct DrugstoSignal
    drugLayer = DrugstoSignal(
        mask=mask_dt,
        size1=mask_dt.shape[1],
        size2=mask_dt.shape[0],
        drugSim=drugSim_tensor,
        momentum=args.drug_bn,
        drop=args.drug_drop,
        device=device
    )

    # Load saved weights
    model_path = os.path.join(args.model_dir, f'{args.outPattern}_drug{args.drug_index}')
    signal_path = model_path + '_signal_state_dict.pth'
    drug_path = model_path + '_drugLayer_state_dict.pth'

    if not os.path.exists(signal_path) or not os.path.exists(drug_path):
        raise FileNotFoundError(f"Model files not found at {model_path}")

    mod.load_state_dict(torch.load(signal_path, map_location=device))
    drugLayer.load_state_dict(torch.load(drug_path, map_location=device))

    mod.eval()
    drugLayer.eval()
    mod = mod.to(device)
    drugLayer = drugLayer.to(device)

    print2log('Model loaded successfully')
    return mod, drugLayer


def validate_with_artificial_stimulation(args, data, mod, drugLayer, drug_held_out, device):
    """Validate by artificially stimulating the held-out drug's targets."""
    print2log(f'Validating with stimulation strength: {args.stimulation_strength}')

    signs_df = data['signs_df']
    drugTargets = data['drugTargets']
    TFOutput = data['TFOutput']
    drugInput = data['drugInput']

    # Get targets for the held-out drug
    if drug_held_out in drugTargets.index:
        drug_target_mask = drugTargets.loc[drug_held_out]
        drug_targets = drug_target_mask[drug_target_mask > 0].index.tolist()
    else:
        print2log(f"Warning: Drug not found in drugTargets")
        drug_targets = []

    # Get signs for this drug
    if drug_held_out in signs_df.index:
        drug_signs = signs_df.loc[drug_held_out]
    else:
        print2log(f"Warning: Drug not found in signs file")
        drug_signs = pd.Series()

    # Get node indices from model
    all_nodes = np.array(list(mod.node_idx_map.keys()))
    num_nodes = len(all_nodes)

    # Create stimulation vector
    stimulation = torch.zeros(1, num_nodes, dtype=torch.double).to(device)
    targets_stimulated = 0

    for target in drug_targets:
        if target in mod.node_idx_map:
            target_idx = mod.node_idx_map[target]

            # Get sign for this target
            if target in drug_signs.index:
                sign = drug_signs[target]
                if sign == 0:
                    continue
                stimulation[0, target_idx] = args.stimulation_strength * sign
                targets_stimulated += 1

    print2log(f"Stimulating {targets_stimulated}/{len(drug_targets)} targets")

    # Get samples containing the held-out drug
    samples_with_drug = drugInput[drug_held_out] > 0
    print2log(f"Validation samples with held-out drug: {samples_with_drug.sum()}")

    results = []

    with torch.no_grad():
        for idx in drugInput[samples_with_drug].index:
            # Get the dose for this sample
            dose = drugInput.loc[idx, drug_held_out]

            # True TF activities
            true_tf = TFOutput.loc[idx, :].values

            # Predict with artificial stimulation
            Baseline = drugLayer(torch.zeros(1, drugLayer.size_in, dtype=torch.double, device=device))
            Baseline_full = mod.input_layer(Baseline)
            YhatFull = mod.signaling_network(stimulation + Baseline_full)
            Yhat = mod.output_layer(YhatFull)
            predicted_tf = Yhat.detach().cpu().numpy().flatten()

            for i, tf in enumerate(TFOutput.columns):
                results.append({
                    'timepoint': args.timepoint,
                    'drug': drug_held_out,
                    'dose': dose,
                    'TF': tf,
                    'true_value': true_tf[i],
                    'predicted_value': predicted_tf[i],
                    'stimulation_strength': args.stimulation_strength
                })

    return pd.DataFrame(results)


def compute_correlations(results_df):
    """Compute per-TF and overall correlations."""
    from scipy.stats import pearsonr

    # Overall correlation
    overall_r, overall_p = pearsonr(results_df['true_value'], results_df['predicted_value'])

    # Per-TF correlations
    tf_correlations = []
    for tf in results_df['TF'].unique():
        tf_data = results_df[results_df['TF'] == tf]
        if len(tf_data) > 1:
            try:
                r, p = pearsonr(tf_data['true_value'], tf_data['predicted_value'])
                # Only add if correlation is valid (not NaN)
                if not np.isnan(r):
                    tf_correlations.append({
                        'TF': tf,
                        'correlation': r,
                        'p_value': p,
                        'n_samples': len(tf_data)
                    })
            except:
                # Skip TFs where correlation cannot be computed
                pass

    tf_corr_df = pd.DataFrame(tf_correlations)

    print2log(f"\nOverall Pearson correlation: {overall_r:.4f} (p={overall_p:.2e})")
    
    if len(tf_corr_df) > 0:
        print2log(f"Mean per-TF correlation: {tf_corr_df['correlation'].mean():.4f}")
        print2log(f"Median per-TF correlation: {tf_corr_df['correlation'].median():.4f}")
        print2log(f"Valid per-TF correlations: {len(tf_corr_df)}/{len(results_df['TF'].unique())}")
    else:
        print2log(f"Warning: No valid per-TF correlations computed (likely due to constant values or insufficient samples)")

    return overall_r, tf_corr_df


def create_plots(results_df, tf_corr_df, output_dir, drug_index, strength):
    """Create validation plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set style
    sns.set_style("whitegrid")

    # 1. Scatter plot: True vs Predicted (all TFs)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(results_df['true_value'], results_df['predicted_value'], 
              alpha=0.5, s=20)
    
    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Compute overall correlation
    from scipy.stats import pearsonr
    r, p = pearsonr(results_df['true_value'], results_df['predicted_value'])
    
    ax.set_xlabel('True TF Activity', fontsize=12)
    ax.set_ylabel('Predicted TF Activity', fontsize=12)
    ax.set_title(f'LODO Validation - Drug {drug_index}\n'
                f'Stimulation Strength = {strength}\n'
                f'r = {r:.4f}, p = {p:.2e}', fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'drug{drug_index}_scatter_strength{strength}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

    # Check if we have valid correlations before plotting per-TF plots
    if len(tf_corr_df) == 0 or 'correlation' not in tf_corr_df.columns:
        print2log(f'Warning: No valid per-TF correlations to plot. Skipping remaining plots.')
        return

    valid_corrs = tf_corr_df['correlation'].dropna()
    if len(valid_corrs) == 0:
        print2log(f'Warning: No valid per-TF correlations to plot. Skipping remaining plots.')
        return

    # 2. Per-TF correlation barplot
    tf_corr_sorted = tf_corr_df.dropna(subset=['correlation']).sort_values('correlation', ascending=False)
    
    if len(tf_corr_sorted) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(tf_corr_sorted) * 0.2)))
        colors = ['green' if c > 0 else 'red' for c in tf_corr_sorted['correlation']]
        ax.barh(range(len(tf_corr_sorted)), tf_corr_sorted['correlation'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(tf_corr_sorted)))
        ax.set_yticklabels(tf_corr_sorted['TF'], fontsize=8)
        ax.set_xlabel('Pearson Correlation', fontsize=12)
        ax.set_title(f'Per-TF Correlations - Drug {drug_index}\n'
                    f'Stimulation Strength = {strength}', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'drug{drug_index}_per_tf_corr_strength{strength}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Distribution of correlations (only if we have multiple valid values)
    if len(valid_corrs) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(valid_corrs, bins=min(30, len(valid_corrs)), edgecolor='black', alpha=0.7)
        ax.axvline(x=valid_corrs.mean(), color='red', 
                  linestyle='--', linewidth=2, label=f'Mean = {valid_corrs.mean():.3f}')
        ax.axvline(x=valid_corrs.median(), color='blue',
                  linestyle='--', linewidth=2, label=f'Median = {valid_corrs.median():.3f}')
        ax.set_xlabel('Correlation', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Distribution of Per-TF Correlations\n'
                    f'Drug {drug_index}, Strength = {strength}', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'drug{drug_index}_corr_dist_strength{strength}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    print2log(f'Plots saved to {output_dir}')


def main():
    args = parse_arguments()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print2log(f'Running on {device}')

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, f'strength_{args.stimulation_strength}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(args)

    # Get drug to validate
    safe_drugs = data['safe_drugs']
    if args.drug_index >= len(safe_drugs):
        raise ValueError(f"Drug index {args.drug_index} out of range. Only {len(safe_drugs)} safe drugs available.")

    drug_held_out = safe_drugs[args.drug_index]
    print2log(f'Validating drug {args.drug_index}: {drug_held_out[:50]}...')

    # Check if drug present at timepoint
    if drug_held_out not in data["drugInput"].columns or (data["drugInput"][drug_held_out].fillna(0) > 0).sum() == 0:
        print2log(f"Skipping drug {args.drug_index} at timepoint {args.timepoint}: no samples with dose > 0")
        return

    # Load trained model
    mod, drugLayer = load_trained_model(args, data, drug_held_out, device)

    # Validate with new stimulation strength
    results_df = validate_with_artificial_stimulation(args, data, mod, drugLayer, drug_held_out, device)

    # Compute correlations
    overall_r, tf_corr_df = compute_correlations(results_df)

    # Save results
    pred_file = os.path.join(args.output_dir, f'drug{args.drug_index}_lodo_prediction_strength{args.stimulation_strength}.csv')
    results_df.to_csv(pred_file, index=False)
    print2log(f'Predictions saved to {pred_file}')

    corr_file = os.path.join(args.output_dir, f'drug{args.drug_index}_per_tf_correlations_strength{args.stimulation_strength}.csv')
    tf_corr_df.to_csv(corr_file, index=False)
    print2log(f'Per-TF correlations saved to {corr_file}')

    # Create plots
    create_plots(results_df, tf_corr_df, args.output_dir, args.drug_index, args.stimulation_strength)

    # Save summary
    summary = {
        'drug_index': args.drug_index,
        'drug_smiles': drug_held_out,
        'timepoint': args.timepoint,
        'stimulation_strength': args.stimulation_strength,
        'overall_correlation': overall_r,
        'mean_per_tf_correlation': tf_corr_df['correlation'].mean() if len(tf_corr_df) > 0 else np.nan,
        'median_per_tf_correlation': tf_corr_df['correlation'].median() if len(tf_corr_df) > 0 else np.nan,
        'n_samples': len(results_df) // max(1, len(tf_corr_df)),
        'n_tfs': len(tf_corr_df),
        'n_tfs_total': len(results_df['TF'].unique())
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(args.output_dir, f'drug{args.drug_index}_summary_strength{args.stimulation_strength}.csv')
    summary_df.to_csv(summary_file, index=False)
    print2log(f'Summary saved to {summary_file}')

    print2log('\nValidation with new stimulation strength completed!')


if __name__ == '__main__':
    main()
