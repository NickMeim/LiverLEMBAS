"""
Script to extract the median sign of drug-target interactions and the induced activity
at target nodes from an ensemble of trained DT-LEMBAS models.
Includes visualization of induced activity distributions.
"""
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
from typing import Tuple, Dict
import sys
from scipy.stats import fisher_exact
# --- Imports and Path Setup ---
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
sys.path.insert(1, os.path.join(gpu_lembas_path))
from LEMBAS.model.bionetwork import format_network, SignalingModel
from utility import DrugstoSignal

def extract_interaction_scores_from_layer(drug_layer: DrugstoSignal, 
                                          mask: torch.Tensor,
                                          device: str = 'cpu') -> torch.Tensor:
    """Extract raw weight-based interaction scores."""
    W_drug = drug_layer.Wsim.data
    A = drug_layer.A.data
    drug_similarity = drug_layer.drugSim 
    w_bn = drug_layer.bn.weight.data
    var_bn = drug_layer.bn.running_var.data
    epsilon = drug_layer.bn.eps
    mask = mask.to(device)
    
    # --- RESTORED MASK APPLICATION ---
    A = A * mask 
    
    W = drug_similarity * W_drug
    K = w_bn / torch.sqrt(var_bn + epsilon)
    KW = K * W
    
    # Propagate via Similarity
    scores_T = torch.mm(A, KW.T)
    scores = scores_T.T
    
    return scores

def extract_weight_based_scores(
    model_dir: str,
    model_pattern_base: str,
    num_models: int,
    drug_targets_df: pd.DataFrame,
    drug_sim_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    drug_bn: float,
    drug_drop: float,
    device: str = 'cpu',
    timepoint: str = 'unknown'
) -> Tuple[pd.DataFrame, np.ndarray]:
    
    if not os.path.exists(model_dir):
        print(f"[{timepoint}] Warning: Model directory not found: {model_dir}")
        return pd.DataFrame(), np.array([])

    drug_smiles = drug_targets_df.index.tolist()
    targets = drug_targets_df.columns.tolist()
    
    drugLayer = DrugstoSignal(
        mask=mask_tensor, size1=mask_tensor.shape[1], size2=mask_tensor.shape[0], 
        drugSim=drug_sim_tensor, momentum=drug_bn, drop=drug_drop, device=device
    ).to(device)
    
    all_scores = np.zeros((num_models, len(drug_smiles), len(targets)))
    
    print(f"[{timepoint}] Extracting scores from {num_models} models...")
    found_models = 0
    
    total_elements = len(drug_smiles) * len(targets)
    
    for i in range(num_models):
        model_idx = i + 1
        filename = f"{model_pattern_base}{model_idx}drugLayer_state_dict_{model_idx}.pth"
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path): continue
        try:
            drugLayer.load_state_dict(torch.load(model_path, map_location=device))
            drugLayer.eval()
            scores = extract_interaction_scores_from_layer(drugLayer, mask_tensor, device)
            
            # Count zeros for this specific model
            scores_np = scores.cpu().detach().numpy()
            zero_count = np.sum(np.abs(scores_np) < 1e-9)
            
            if zero_count > 0:
                print(f"    > Model {model_idx}: {zero_count}/{total_elements} scores are strict Zeros ({zero_count/total_elements:.1%})")
            
            all_scores[i] = scores_np
            found_models += 1
        except Exception as e: 
            print(f"Error model {i}: {e}")
            continue
    
    if found_models == 0:
        print(f"[{timepoint}] No valid models found!")
        return pd.DataFrame(), np.array([])

    median_scores = np.median(all_scores, axis=0)
    median_scores_df = pd.DataFrame(median_scores, index=drug_smiles, columns=targets)
    
    median_zeros = np.sum(np.abs(median_scores) < 1e-9)
    print(f"  >> Median Weight Matrix has {median_zeros} zeros.")
    
    return median_scores_df, all_scores

def compute_delta_activity(
    model_dir: str,
    model_pattern_base: str,
    num_models: int,
    drug_input_df: pd.DataFrame,
    drug_targets_df: pd.DataFrame,
    tf_output_df: pd.DataFrame,
    drug_sim_tensor: torch.Tensor,
    mask_tensor: torch.Tensor,
    net_df: pd.DataFrame,
    projection_amplitude_in: float,
    projection_amplitude_out: float,
    bionet_params: dict,
    drug_bn: float, 
    drug_drop: float, 
    device: str = 'cpu',
    timepoint: str = 'unknown'
) -> Tuple[np.ndarray, dict, list]:
    
    if not os.path.exists(model_dir):
        return np.array([]), {}, []

    print(f"[{timepoint}] Simulating Delta Activity...")
    
    drug_names = drug_targets_df.index.tolist()
    target_names = drug_targets_df.columns.tolist()
    
    max_doses_series = drug_input_df.reindex(columns=drug_names).fillna(0.0).max(axis=0)
    input_treated = torch.tensor(np.diag(max_doses_series.values), dtype=torch.double).to(device)
    input_control = torch.zeros_like(input_treated).to(device)

    mod = SignalingModel(
        net=net_df, X_in=drug_targets_df, y_out=tf_output_df, 
        projection_amplitude_in=projection_amplitude_in, projection_amplitude_out=projection_amplitude_out,
        weight_label='mode_of_action', source_label='source', target_label='target',
        bionet_params=bionet_params, dtype=torch.double, device=device
    )
    
    drugLayer = DrugstoSignal(
        mask=mask_tensor, size1=mask_tensor.shape[1], size2=mask_tensor.shape[0], 
        drugSim=drug_sim_tensor, momentum=drug_bn, drop=drug_drop, device=device
    ).to(device)

    target_indices = [mod.node_idx_map[t] for t in target_names]
    all_deltas = np.zeros((num_models, len(drug_names), len(target_names)))
    
    for i in range(num_models):
        model_idx = i + 1
        signal_path = os.path.join(model_dir, f"{model_pattern_base}{model_idx}signal_state_dict_{model_idx}.pth")
        drug_path = os.path.join(model_dir, f"{model_pattern_base}{model_idx}drugLayer_state_dict_{model_idx}.pth")
        
        if not os.path.exists(signal_path) or not os.path.exists(drug_path): continue
        
        try:
            mod.load_state_dict(torch.load(signal_path, map_location=device))
            drugLayer.load_state_dict(torch.load(drug_path, map_location=device))
            mod.eval()
            drugLayer.eval()
            
            with torch.no_grad():
                Xin_treated = drugLayer(input_treated)
                X_full_treated = mod.input_layer(Xin_treated)
                Xin_control = drugLayer(input_control)
                X_full_control = mod.input_layer(Xin_control)
                
                delta = X_full_treated[:, target_indices] - X_full_control[:, target_indices]
                all_deltas[i] = delta.cpu().detach().numpy()
        except: continue
    
    return all_deltas, mod.node_idx_map, target_names

def determine_hybrid_sign(weight, delta, threshold=0.0):
    if abs(weight) > threshold:
        return 1 if weight > 0 else -1
    else:
        if delta > 1e-9: return 1
        elif delta < -1e-9: return -1
        else: return 0

def analyze_and_plot_single_tp(
    median_weight_scores: pd.DataFrame, 
    all_deltas: np.ndarray,
    output_dir: str,
    threshold: float = 0.0,
    timepoint: str = 'unknown'
) -> pd.DataFrame:
    
    drugs = median_weight_scores.index
    target_names = median_weight_scores.columns.tolist()
    tp_dir = os.path.join(output_dir, timepoint)
    Path(tp_dir).mkdir(parents=True, exist_ok=True)
    
    median_deltas = np.median(all_deltas, axis=0) 
    
    hybrid_signs = np.zeros_like(median_weight_scores.values, dtype=int)
    
    zeros_count = 0
    for r in range(hybrid_signs.shape[0]):
        for c in range(hybrid_signs.shape[1]):
            w_val = median_weight_scores.values[r, c]
            d_val = median_deltas[r, c]
            sign = determine_hybrid_sign(w_val, d_val, threshold)
            hybrid_signs[r, c] = sign
            if sign == 0: zeros_count += 1

    print(f"  > {timepoint} Hybrid Result: {zeros_count} interactions remain 0.")

    hybrid_signs_df = pd.DataFrame(hybrid_signs, index=drugs, columns=target_names)
    hybrid_signs_df.to_csv(os.path.join(tp_dir, f'signs_HYBRID_{timepoint}.csv'))

    # Identify non-control drugs for reports
    valid_drugs = [d for d in drugs if d != 'Control']
    
    if len(valid_drugs) < len(drugs):
        print(f"  > Excluded 'Control' from Mismatch Reports and Plots.")

    report_data = []
    for drug in valid_drugs:
        d_idx = median_weight_scores.index.get_loc(drug)
        for t_idx, target in enumerate(target_names):
            w_val = median_weight_scores.values[d_idx, t_idx]
            d_val = median_deltas[d_idx, t_idx]
            final_sign = hybrid_signs[d_idx, t_idx]
            
            w_sign_raw = 1 if w_val > threshold else (-1 if w_val < -threshold else 0)
            d_sign_raw = 1 if d_val > 0 else (-1 if d_val < 0 else 0)
            
            if w_sign_raw == 0 and d_sign_raw == 0: continue
            
            report_data.append({
                'Drug': drug,
                'Target': target,
                'Weight_Value': w_val,
                'Delta_Value': d_val,
                'Weight_Sign_Raw': w_sign_raw,
                'Delta_Sign_Raw': d_sign_raw,
                'Final_Hybrid_Sign': final_sign,
                'Mismatch': (w_sign_raw != d_sign_raw) and (w_sign_raw != 0)
            })
            
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(os.path.join(tp_dir, f'mismatch_report_{timepoint}.csv'), index=False)
    
    if not report_df.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=report_df, x='Weight_Value', y='Delta_Value', hue='Mismatch', palette={True: 'red', False: 'blue'}, alpha=0.6)
        plt.axhline(0, color='grey', linestyle='--'); plt.axvline(0, color='grey', linestyle='--')
        plt.title(f"Weight vs Delta ({timepoint})")
        plt.tight_layout()
        plt.savefig(os.path.join(tp_dir, f'scatter_weight_vs_delta_{timepoint}.png'))
        plt.close()

    return hybrid_signs_df

def plot_fisher_comparison(df1, df2, name1, name2, output_dir):
    """
    Compares signs between timepoints.
    EXPLICITLY DROPS 'Control' before comparison.
    """
    common_idx = df1.index.intersection(df2.index)
    common_col = df1.columns.intersection(df2.columns)
    
    if 'Control' in common_idx:
        common_idx = common_idx.drop('Control')
    
    v1 = df1.loc[common_idx, common_col].values.flatten()
    v2 = df2.loc[common_idx, common_col].values.flatten()
    
    labels = [-1, 0, 1]
    cm = pd.crosstab(v1, v2, rownames=[name1], colnames=[name2])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Comparison: {name1} vs {name2} (No Control)")
    plt.savefig(os.path.join(output_dir, f"contingency_{name1}_vs_{name2}.png"))
    plt.close()
    
    neg_neg = cm.loc[-1, -1]
    neg_pos = cm.loc[-1, 1]
    pos_neg = cm.loc[1, -1]
    pos_pos = cm.loc[1, 1]
    
    fisher_table = [[neg_neg, neg_pos], [pos_neg, pos_pos]]
    
    if (neg_neg + neg_pos + pos_neg + pos_pos) > 0:
        try:
            odds_ratio, p_value = fisher_exact(fisher_table)
            print(f"  > Fisher's Exact Test ({name1} vs {name2} - Active Only): P={p_value:.4e}")
            with open(os.path.join(output_dir, "fisher_stats.txt"), "a") as f:
                f.write(f"\n--- {name1} vs {name2} ---\nTable (Active):\n{fisher_table}\nP-value: {p_value:.4e}\n")
        except: pass

def analyze_cross_timepoint_consistency(
    sign_dataframes: Dict[str, pd.DataFrame], 
    output_dir: str
):
    print("\n" + "="*70)
    print("CROSS-TIMEPOINT CONSISTENCY (Hybrid Logic)")
    print("="*70)
    
    comparisons = [('2 hour', '8 hour'), ('2 hour', '24 hour'), ('8 hour', '24 hour')]
    for tp1, tp2 in comparisons:
        if tp1 in sign_dataframes and tp2 in sign_dataframes:
            plot_fisher_comparison(sign_dataframes[tp1], sign_dataframes[tp2], tp1, tp2, output_dir)

    timepoints = list(sign_dataframes.keys())
    common_drugs = sign_dataframes[timepoints[0]].index
    common_targets = sign_dataframes[timepoints[0]].columns
    
    for tp in timepoints[1:]:
        common_drugs = common_drugs.intersection(sign_dataframes[tp].index)
        common_targets = common_targets.intersection(sign_dataframes[tp].columns)
        
    shape = (len(timepoints), len(common_drugs), len(common_targets))
    all_signs_tensor = np.zeros(shape)
    
    for i, tp in enumerate(timepoints):
        df = sign_dataframes[tp].loc[common_drugs, common_targets]
        all_signs_tensor[i, :, :] = df.values

    final_signs = np.zeros((len(common_drugs), len(common_targets)), dtype=int)
    consistency_counts = []
    
    stats = {'Consensus_Active': 0, 'No_Signal_All_Zeros': 0, 'Ambiguous_Split_Vote': 0}
    
    print("Calculating majority vote...")
    
    for d in range(len(common_drugs)):
        drug_name = common_drugs[d]
        
        if drug_name == 'Control':
            final_signs[d, :] = 0
            continue 
            
        for t in range(len(common_targets)):
            signs = all_signs_tensor[:, d, t]
            pos_count = np.sum(signs == 1)
            neg_count = np.sum(signs == -1)
            zero_count = np.sum(signs == 0)
            
            if pos_count >= 2:
                final_signs[d, t] = 1
                consistency_counts.append(pos_count)
                stats['Consensus_Active'] += 1
            elif neg_count >= 2:
                final_signs[d, t] = -1
                consistency_counts.append(neg_count)
                stats['Consensus_Active'] += 1
            else:
                if zero_count == 3:
                    stats['No_Signal_All_Zeros'] += 1
                else:
                    stats['Ambiguous_Split_Vote'] += 1
                    consistency_counts.append(max(pos_count, neg_count))

    print("\nConsensus Breakdown (Excluding 'Control'):")
    print(f"  - Active Consensus Found (>=2/3 votes): {stats['Consensus_Active']}")
    print(f"  - No Signal (0/3 votes, All Zeros): {stats['No_Signal_All_Zeros']}")
    print(f"  - Ambiguous/Split (No Majority): {stats['Ambiguous_Split_Vote']}")
    
    consensus_df = pd.DataFrame(final_signs, index=common_drugs, columns=common_targets)
    consensus_df.to_csv(os.path.join(output_dir, 'consensus_signs_majority_vote.csv'))
    
    if len(consistency_counts) > 0:
        plt.figure(figsize=(8, 6))
        sns.histplot(consistency_counts, discrete=True, shrink=0.8)
        plt.title(f"Consistency Across Timepoints (Excluding Control)")
        plt.xlabel("Number of Agreeing Timepoints")
        plt.ylabel("Count of Interactions")
        plt.xticks([1, 2, 3])
        plt.savefig(os.path.join(output_dir, 'consistency_histogram.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', type=str, default='../results/')
    parser.add_argument('--model_pattern', type=str, default='hepatocytes_dtlembas')
    
    # Inputs
    parser.add_argument('--DrugsIn', type=str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
    parser.add_argument('--TargetsIn', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
    parser.add_argument('--TFsOut', type=str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
    parser.add_argument('--ChemicalSims', type=str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
    parser.add_argument('--PKN', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
    parser.add_argument('--infoPath', type=str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
    
    parser.add_argument('--num_models', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='../preprocessing/preprocessed_data/LeaveOneDrugOut/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--projection_amplitude', type=float, default=1.2)
    parser.add_argument('--ConvertToEmpProb', type=bool, default=False)
    parser.add_argument('--drug_bn', type=float, default=0.2)
    parser.add_argument('--drug_drop', type=float, default=0.1)
    parser.add_argument('--leakage', type=float, default=0.01)
    parser.add_argument('--max_iterations', type=int, default=150)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Loading base data files...")
    drugInput_full = pd.read_csv(args.DrugsIn, sep='\t', index_col=0)
    TFOutput_full = pd.read_csv(args.TFsOut, sep='\t', low_memory=False, index_col=0)
    inputInfo_full = pd.read_csv(args.infoPath, sep='\t', low_memory=False, index_col=0)
    drugTargets_full = pd.read_csv(args.TargetsIn, sep='\t', low_memory=False, index_col=0)
    drugSim_full = pd.read_csv(args.ChemicalSims, index_col=0)
    net_raw = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[:,['source','target','stimulation','inhibition']]
    
    net_formatted = format_network(net_raw, weight_label='mode_of_action', 
                                   stimulation_label='stimulation', inhibition_label='inhibition')
    nodeNames = np.union1d(net_formatted.source.values, net_formatted.target.values)
    
    timepoints = ['2 hour', '8 hour', '24 hour']
    results_by_timepoint = {}
    
    for timepoint in timepoints:
        print(f"\n" + "-"*50)
        print(f"Processing Timepoint: {timepoint}")
        print("-" * 50)
        
        # Subsetting
        inputInfo = inputInfo_full[inputInfo_full['time'] == timepoint]
        TFOutput_tmp = TFOutput_full.loc[inputInfo.index, :]
        drugInput_tmp = drugInput_full.loc[inputInfo.index, :]
        
        druginName = drugInput_tmp.columns.values
        inName = drugTargets_full.columns.values
        outconds = drugInput_tmp.index.values
        outName = TFOutput_tmp.columns.values
        
        inName = np.intersect1d(nodeNames, inName)
        outName = np.intersect1d(nodeNames, outName)
        
        TFOutput_tmp = TFOutput_tmp.loc[outconds, outName]
        drugTargets = drugTargets_full.loc[:, inName]
        
        if args.ConvertToEmpProb:
            TFOutput_tmp = 1/(1+np.exp(-TFOutput_tmp))
            
        drugTargets = drugTargets.loc[drugInput_tmp.columns.values, :]
        drugSim = drugSim_full.loc[drugInput_tmp.columns.values, drugInput_tmp.columns.values]
        
        # --- NEW DIAGNOSTIC BLOCK ---
        print("\n[DIAGNOSTICS] Input Interaction Matrix (Prior Knowledge):")
        # 1. Row Sums (Drugs)
        row_sums = drugTargets.sum(axis=1)
        zero_drugs = row_sums[row_sums == 0].index.tolist()
        print(f"  > Drugs with 0 known interactions: {len(zero_drugs)}")
        if len(zero_drugs) > 0:
            print(f"    Names: {zero_drugs}")
            
        # 2. Col Sums (Targets)
        col_sums = drugTargets.sum(axis=0)
        zero_targets = col_sums[col_sums == 0].index.tolist()
        print(f"  > Targets with 0 known interactions: {len(zero_targets)}")
        if len(zero_targets) > 0:
            print(f"    Names: {zero_targets}")
        print("-" * 30 + "\n")
        # ----------------------------

        drugSim_tensor = torch.tensor(drugSim.values.copy(), dtype=torch.double).to(args.device)
        mask_tensor = torch.tensor(drugTargets.values.T, dtype=torch.double).to(args.device)
        
        if timepoint == '8 hour':
            data_path = os.path.join(args.res_dir, 'allmodels/')
        else:
            data_path = os.path.join(args.res_dir, timepoint, 'allmodels/')
            
        median_weight_scores_df, _ = extract_weight_based_scores(
            model_dir=data_path,
            model_pattern_base=args.model_pattern,
            num_models=args.num_models,
            drug_targets_df=drugTargets,
            drug_sim_tensor=drugSim_tensor,
            mask_tensor=mask_tensor,
            drug_bn=args.drug_bn,
            drug_drop=args.drug_drop,
            device=args.device,
            timepoint=timepoint
        )
        
        if median_weight_scores_df.empty: continue

        bionet_params = {'target_steps': 100, 'max_steps': args.max_iterations, 'exp_factor': 50, 'tolerance': 1e-5, 'leak': args.leakage}
        
        all_deltas, _, target_names = compute_delta_activity(
            model_dir=data_path,
            model_pattern_base=args.model_pattern,
            num_models=args.num_models,
            drug_input_df=drugInput_tmp,
            drug_targets_df=drugTargets,
            tf_output_df=TFOutput_tmp,
            drug_sim_tensor=drugSim_tensor,
            mask_tensor=mask_tensor,
            net_df=net_formatted,
            projection_amplitude_in=args.projection_amplitude,
            projection_amplitude_out=args.projection_amplitude,
            bionet_params=bionet_params,
            drug_bn=args.drug_bn,
            drug_drop=args.drug_drop,
            device=args.device,
            timepoint=timepoint
        )

        hybrid_signs_df = analyze_and_plot_single_tp(
            median_weight_scores=median_weight_scores_df,
            all_deltas=all_deltas, 
            output_dir=args.output_dir,
            threshold=args.threshold,
            timepoint=timepoint
        )
        
        results_by_timepoint[timepoint] = hybrid_signs_df

    # --- Cross-Timepoint ---
    if results_by_timepoint:
        analyze_cross_timepoint_consistency(results_by_timepoint, args.output_dir)

if __name__ == '__main__':
    main()