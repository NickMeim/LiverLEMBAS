"""
Generate Consensus Target Ranks Across All Timepoints

This script extracts target ranks from multiple timepoints and creates a
consensus rank matrix (similar to consensus signs).

Usage:
    python generate_consensus_target_ranks.py \
        --timepoints "2 hour" "8 hour" "24 hour" \
        --res_dir ../results/ \
        --output ../preprocessing/preprocessed_data/LeaveOneDrugOut/consensus_target_ranks.csv \
        --rank_method percentile \
        --rank_scope on_target
"""

import torch
import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional
# Setup LEMBAS path
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
# gpu_lembas_path = '/home/meimetis/LEMBAS-main'
sys.path.insert(1, os.path.join(gpu_lembas_path))
from LEMBAS.model.bionetwork import format_network
from utility import DrugstoSignal


def extract_interaction_scores_from_layer(
    drug_layer: DrugstoSignal,
    mask: torch.Tensor,
    device: str = 'cpu'
) -> torch.Tensor:
    """Extract raw weight-based interaction scores from drug layer.

    NOTE: This returns a dense drug x target score matrix due to propagation
    through the drug-drug similarity matrix.

    If you want ranks that ignore similarity-propagated ("off-target") targets,
    use --rank_scope on_target (handled downstream in scores_to_ranks).
    """
    W_drug = drug_layer.Wsim.data
    A = drug_layer.A.data
    drug_similarity = drug_layer.drugSim
    w_bn = drug_layer.bn.weight.data
    var_bn = drug_layer.bn.running_var.data
    epsilon = drug_layer.bn.eps
    mask = mask.to(device)

    # Apply DTI mask to the drug->target matrix (A is targets x drugs)
    A = A * mask

    # Drug-drug similarity mixing
    W = drug_similarity * W_drug
    K = w_bn / torch.sqrt(var_bn + epsilon)
    KW = K * W

    # Propagate to produce dense drug x target scores
    scores_T = torch.mm(A, KW.T)   # (targets x drugs)
    scores = scores_T.T            # (drugs x targets)

    return scores


def extract_scores_for_timepoint(
    model_dir: str,
    model_pattern: str,
    num_models: int,
    drug_targets_df: pd.DataFrame,
    drug_sim_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    timepoint: str,
    drug_bn: float = 0.2,
    drug_drop: float = 0.1,
    device: str = 'cpu'
) -> pd.DataFrame:
    """Extract median scores for one timepoint."""

    print(f"\n[{timepoint}] Extracting scores from {num_models} models in {model_dir}")

    if not os.path.exists(model_dir):
        print(f"[{timepoint}] Warning: Model directory not found, skipping")
        return pd.DataFrame()

    drug_smiles = drug_targets_df.index.tolist()
    targets = drug_targets_df.columns.tolist()

    drugLayer = DrugstoSignal(
        mask=mask_tensor,
        size1=mask_tensor.shape[1],
        size2=mask_tensor.shape[0],
        drugSim=drug_sim_tensor,
        momentum=drug_bn,
        drop=drug_drop,
        device=device
    ).to(device)

    all_scores = []
    found_models = 0

    for i in range(num_models):
        model_idx = i + 1  # IMPORTANT: your checkpoints start at 1
        filename = f"{model_pattern}{model_idx}drugLayer_state_dict_{model_idx}.pth"
        model_path = os.path.join(model_dir, filename)

        if not os.path.exists(model_path):
            continue

        try:
            drugLayer.load_state_dict(torch.load(model_path, map_location=device))
            drugLayer.eval()

            with torch.no_grad():
                scores = extract_interaction_scores_from_layer(drugLayer, mask_tensor, device)
                all_scores.append(scores.cpu().detach().numpy())

            found_models += 1

        except Exception as e:
            print(f"[{timepoint}] Error loading {filename}: {e}")
            continue

    if found_models == 0:
        print(f"[{timepoint}] Warning: No valid models found")
        return pd.DataFrame()

    print(f"[{timepoint}] Successfully loaded {found_models} models")

    median_scores = np.median(np.stack(all_scores, axis=0), axis=0)
    median_scores_df = pd.DataFrame(median_scores, index=drug_smiles, columns=targets)
    return median_scores_df



def _get_on_target_columns_for_drug(
    dti_mask_df: pd.DataFrame,
    drug: str
) -> pd.Index:
    """Return the subset of target columns that are on-target (DTI mask > 0) for a given drug."""
    if drug not in dti_mask_df.index:
        return pd.Index([])
    row = dti_mask_df.loc[drug]
    return row.index[row.values > 0]


def scores_to_ranks(
    scores_df: pd.DataFrame,
    method: str = 'percentile',
    rank_scope: str = 'all',
    dti_mask_df: Optional[pd.DataFrame] = None,
    eps: float = 1e-9
) -> pd.DataFrame:
    """Convert scores to ranks (0-1 scale).

    Args:
        scores_df: DataFrame with drugs as rows, targets as columns
        method: Ranking method
            - 'percentile': Rank by percentile within each drug (0-1)
            - 'normalize': Divide by max absolute score per drug
            - 'softmax': Softmax normalization (exponential weighting)
        rank_scope:
            - 'all': rank across ALL non-zero scores for the drug (default)
            - 'on_target': rank only across targets where DTI mask > 0 for that drug
        dti_mask_df: required if rank_scope == 'on_target' (drugs x targets mask)
        eps: threshold for treating values as non-zero

    Returns:
        ranks_df: DataFrame with same shape, values in [0, 1]
    """
    if rank_scope not in {'all', 'on_target'}:
        raise ValueError("rank_scope must be one of: {'all','on_target'}")
    if rank_scope == 'on_target' and dti_mask_df is None:
        raise ValueError("dti_mask_df must be provided when rank_scope == 'on_target'")

    ranks_df = pd.DataFrame(0.0, index=scores_df.index, columns=scores_df.columns)

    for drug in scores_df.index:
        drug_scores = scores_df.loc[drug]

        if rank_scope == 'on_target':
            on_targets = _get_on_target_columns_for_drug(dti_mask_df, drug)
            if len(on_targets) > 0:
                # Normal behavior: only on-target columns define the distribution
                abs_scores = drug_scores.loc[on_targets].abs()
            else:
                # Fallback: no known targets -> rank using all (off-target) scores
                abs_scores = drug_scores.abs()
        else:
            abs_scores = drug_scores.abs()

        non_zero = abs_scores[abs_scores > eps]
        if len(non_zero) == 0:
            continue

        if method == 'percentile':
            # Vectorized percentile rank: strongest = 1.0
            # Equivalent to: count(values <= v)/N (ties -> max rank)
            ranks_series = non_zero.rank(method='max') / float(len(non_zero))
            ranks_df.loc[drug, ranks_series.index] = ranks_series

        elif method == 'normalize':
            max_score = non_zero.max()
            if max_score <= eps:
                continue
            ranks_df.loc[drug, non_zero.index] = non_zero / max_score

        elif method == 'softmax':
            temperature = non_zero.std() + eps
            exp_scores = np.exp(non_zero / temperature)
            softmax_ranks = exp_scores / exp_scores.sum()
            softmax_ranks = softmax_ranks / softmax_ranks.max()
            ranks_df.loc[drug, non_zero.index] = softmax_ranks

        else:
            raise ValueError(f"Unknown method: {method}")

        # Everything else stays 0.0 in ranks_df.

    return ranks_df


def create_consensus_ranks(
    timepoint_ranks: Dict[str, pd.DataFrame],
    method: str = 'median'
) -> pd.DataFrame:
    """Create consensus ranks across timepoints."""
    print(f"\nCreating consensus ranks using method: {method}")

    # Get all drugs and targets from all timepoints
    all_drugs = set()
    all_targets = set()
    for ranks_df in timepoint_ranks.values():
        all_drugs.update(ranks_df.index)
        all_targets.update(ranks_df.columns)

    all_drugs = sorted(all_drugs)
    all_targets = sorted(all_targets)

    # Initialize consensus matrix
    consensus_ranks = pd.DataFrame(0.0, index=all_drugs, columns=all_targets)

    # Collect ranks from all timepoints
    rank_stack = np.zeros((len(timepoint_ranks), len(all_drugs), len(all_targets)))
    mask_stack = np.zeros((len(timepoint_ranks), len(all_drugs), len(all_targets)), dtype=bool)

    for tp_idx, (_tp_name, ranks_df) in enumerate(timepoint_ranks.items()):
        # align by shared index/columns
        common_drugs = ranks_df.index.intersection(all_drugs)
        common_targets = ranks_df.columns.intersection(all_targets)
        for drug in common_drugs:
            drug_idx = all_drugs.index(drug)
            row = ranks_df.loc[drug, common_targets]
            nz = row[row > 0]
            if nz.empty:
                continue
            for target, rank in nz.items():
                target_idx = all_targets.index(target)
                rank_stack[tp_idx, drug_idx, target_idx] = rank
                mask_stack[tp_idx, drug_idx, target_idx] = True

    # Compute consensus for each drug-target pair
    for drug_idx, drug in enumerate(all_drugs):
        for target_idx, target in enumerate(all_targets):
            ranks = rank_stack[:, drug_idx, target_idx]
            m = mask_stack[:, drug_idx, target_idx]

            if not m.any():
                consensus_ranks.loc[drug, target] = 0.0
                continue

            valid_ranks = ranks[m]

            if method == 'median':
                consensus_ranks.loc[drug, target] = np.median(valid_ranks)
            elif method == 'mean':
                consensus_ranks.loc[drug, target] = np.mean(valid_ranks)
            elif method == 'max':
                consensus_ranks.loc[drug, target] = np.max(valid_ranks)
            elif method == 'weighted_mean':
                weight = len(valid_ranks) / len(timepoint_ranks)
                consensus_ranks.loc[drug, target] = np.mean(valid_ranks) * weight
            else:
                raise ValueError(f"Unknown method: {method}")

    return consensus_ranks


def print_consensus_statistics(
    consensus_ranks: pd.DataFrame,
    timepoint_ranks: Dict[str, pd.DataFrame]
):
    """Print statistics about consensus ranks."""
    print("\n" + "=" * 70)
    print("CONSENSUS RANK STATISTICS")
    print("=" * 70)

    non_zero = consensus_ranks.values.flatten()
    non_zero = non_zero[non_zero > 0]

    print(f"\nTotal drug-target pairs: {consensus_ranks.size}")
    print(f"Non-zero ranks: {len(non_zero)} ({len(non_zero) / consensus_ranks.size * 100:.1f}%)")

    if len(non_zero) > 0:
        print(f"\nConsensus rank distribution (non-zero only):")
        print(f"  Min:    {non_zero.min():.4f}")
        print(f"  25%:    {np.percentile(non_zero, 25):.4f}")
        print(f"  Median: {np.median(non_zero):.4f}")
        print(f"  75%:    {np.percentile(non_zero, 75):.4f}")
        print(f"  Max:    {non_zero.max():.4f}")
        print(f"  Mean:   {non_zero.mean():.4f}")
        print(f"  Std:    {non_zero.std():.4f}")

    print(f"\nPer-timepoint statistics:")
    for tp_name, ranks_df in timepoint_ranks.items():
        tp_non_zero = ranks_df.values.flatten()
        tp_non_zero = tp_non_zero[tp_non_zero > 0]
        mean_val = float(tp_non_zero.mean()) if len(tp_non_zero) else float('nan')
        print(
            f"  {tp_name:12s}: {len(tp_non_zero):5d} non-zero "
            f"({len(tp_non_zero) / ranks_df.size * 100:5.1f}%), mean={mean_val:.3f}"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate consensus target ranks across timepoints'
    )

    parser.add_argument(
        '--timepoints',
        type=str,
        nargs='+',
        default=["2 hour", "8 hour", "24 hour"],
        help='List of timepoints to process (e.g., "2 hour" "8 hour" "24 hour")'
    )
    parser.add_argument('--res_dir', type=str, default='../results/')

    parser.add_argument('--model_pattern', type=str, default='hepatocytes_dtlembas',
                        help='Model filename pattern')
    parser.add_argument('--num_models', type=int, default=50,
                        help='Number of models in each ensemble')

    parser.add_argument('--DrugsIn', type=str,
                        default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
    parser.add_argument('--TargetsIn', type=str,
                        default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
    parser.add_argument('--PKN', type=str,
                        default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv',
                        help='PKN network used in training (to match node set / target subset)')
    parser.add_argument('--ChemicalSims', type=str,
                        default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
    parser.add_argument('--infoPath', type=str,
                        default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')

    parser.add_argument(
        '--output',
        type=str,
        default='../preprocessing/preprocessed_data/LeaveOneDrugOut/consensus_DTI_ranks.csv',
        help='Output CSV file for consensus ranks'
    )
    parser.add_argument('--output_per_timepoint', action='store_true',
                        help='Also save individual timepoint ranks')

    parser.add_argument('--rank_method', type=str, default='percentile',
                        choices=['percentile', 'normalize', 'softmax'],
                        help='Method for converting scores to ranks within each timepoint')

    parser.add_argument(
        '--rank_scope',
        type=str,
        default='on_target',
        choices=['all', 'on_target'],
        help=(
            "Scope used when converting scores to ranks within each drug. "
            "'all' includes similarity-propagated off-target scores in the distribution; "
            "'on_target' computes ranks using only known DTI targets for that drug."
        )
    )

    parser.add_argument('--consensus_method', type=str, default='median',
                        choices=['median', 'mean', 'max', 'weighted_mean'],
                        help='Method for computing consensus across timepoints')
    parser.add_argument('--drug_bn', type=float, default=0.2)
    parser.add_argument('--drug_drop', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'])

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 70)
    print("GENERATING CONSENSUS TARGET RANKS ACROSS TIMEPOINTS")
    print("=" * 70)
    print(f"Timepoints: {', '.join(args.timepoints)}")
    print(f"Rank method (within timepoints): {args.rank_method}")
    print(f"Rank scope (within timepoints): {args.rank_scope}")
    print(f"Consensus method (across timepoints): {args.consensus_method}")

    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    net_raw = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[:, ['source','target','stimulation','inhibition']]
    net_formatted = format_network(
        net_raw,
        weight_label='mode_of_action',
        stimulation_label='stimulation',
        inhibition_label='inhibition'
    )
    nodeNames = np.union1d(net_formatted.source.values, net_formatted.target.values)

    print("\nLoading common data...")
    # Load full matrices once
    drugInput_full  = pd.read_csv(args.DrugsIn,   sep='\t', index_col=0)
    drugTargets_full = pd.read_csv(args.TargetsIn, sep='\t', index_col=0)
    inputInfo_full  = pd.read_csv(args.infoPath,  sep='\t', index_col=0)
    drugSim_full    = pd.read_csv(args.ChemicalSims, index_col=0)

    timepoint_scores: Dict[str, pd.DataFrame] = {}
    timepoint_ranks: Dict[str, pd.DataFrame] = {}

    for tp in args.timepoints:
        if tp == '8 hour':
            model_dir = os.path.join(args.res_dir, 'allmodels/')
        else:
            model_dir = os.path.join(args.res_dir, tp, 'allmodels/')

        print(f"\n{'=' * 70}")
        print(f"Processing timepoint: {tp}")
        print('=' * 70)

        # --- SUBSET EXACTLY LIKE SIGN EXTRACTION ---
        inputInfo = inputInfo_full[inputInfo_full['time'] == tp]
        drugInput_tmp = drugInput_full.loc[inputInfo.index, :]

        # drugs are columns of drugInput_tmp (same as sign extraction)
        druginName = drugInput_tmp.columns.values

        # targets: intersect with PKN nodeNames (this is the key to get 83)
        inName = drugTargets_full.columns.values
        inName = np.intersect1d(nodeNames, inName)

        drugTargets_tp = drugTargets_full.loc[druginName, inName]
        drugSim_tp = drugSim_full.loc[druginName, druginName]

        drugSim_tensor = torch.tensor(drugSim_tp.values.copy(), dtype=torch.double).to(device)
        mask_tensor = torch.tensor(drugTargets_tp.values.T, dtype=torch.double).to(device)

        scores_df = extract_scores_for_timepoint(
            model_dir=model_dir,
            model_pattern=args.model_pattern,
            num_models=args.num_models,
            drug_targets_df=drugTargets_tp,
            drug_sim_tensor=drugSim_tensor,
            mask_tensor=mask_tensor,
            timepoint=tp,
            drug_bn=args.drug_bn,
            drug_drop=args.drug_drop,
            device=device
        )

        if scores_df.empty:
            print(f"[{tp}] Skipping - no scores extracted")
            continue

        ranks_df = scores_to_ranks(
            scores_df,
            method=args.rank_method,
            rank_scope=args.rank_scope,
            dti_mask_df=drugTargets_tp
        )

        timepoint_scores[tp] = scores_df
        timepoint_ranks[tp] = ranks_df

        if args.output_per_timepoint:
            base, ext = os.path.splitext(args.output)
            tp_output = f"{base}_{tp.replace(' ', '_')}{ext or '.csv'}"
            ranks_df.to_csv(tp_output)
            print(f"[{tp}] Saved to: {tp_output}")

    if not timepoint_ranks:
        raise ValueError("No valid timepoint ranks extracted!")

    print(f"\n{'=' * 70}")
    print("CREATING CONSENSUS")
    print('=' * 70)

    consensus_ranks = create_consensus_ranks(
        timepoint_ranks=timepoint_ranks,
        method=args.consensus_method
    )

    print_consensus_statistics(consensus_ranks, timepoint_ranks)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    consensus_ranks.to_csv(args.output)

    print(f"\n{'=' * 70}")
    print(f"CONSENSUS RANKS SAVED TO: {args.output}")
    print('=' * 70)


if __name__ == '__main__':
    main()
