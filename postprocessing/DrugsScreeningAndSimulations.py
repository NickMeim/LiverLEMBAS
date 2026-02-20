import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from matplotlib.backends.backend_pdf import PdfPages

# ── LEMBAS path ────────────────────────────────────────────────────────────────
# Ensure this matches your local setup
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
# gpu_lembas_path = '/home/meimetis/meimetis/LEMBAS-main'
sys.path.insert(1, os.path.join(gpu_lembas_path))

from utility import * 

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

# ==============================================================================
# ARGUMENTS & CONFIGURATION
# ==============================================================================
parser = argparse.ArgumentParser(prog='Drug Screening and In Silico Experiments')
parser.add_argument('--DrugsIn', type=str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', type=str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', type=str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--infoPath', type=str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--res_dir', type=str, default='../results/allmodels/')
parser.add_argument('--out_dir', type=str, default='../results/drug_screening_insilio/')
parser.add_argument('--number_DTLEMBAS_models', type=int, default=50)
parser.add_argument('--modelPattern', type=str, default='hepatocytes_dtlembas')
parser.add_argument('--inputAmplitude', type=float, default=3)
parser.add_argument('--projectionAmplitude', type=float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', type=int, default=90)
parser.add_argument('--LEMBAS_leak', type=float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', default=1e-5)
parser.add_argument('--drug_bn', default=0.2)
parser.add_argument('--drug_drop', default=0.1)
parser.add_argument('--timepoint', type=str, default='8 hour')
parser.add_argument('--stimulation_strength', type=float, default=3000.0)
args = parser.parse_args()

# FIX: Corrected torch available check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S = args.stimulation_strength

# Node/TF Identifiers
NODE_IFNAR1 = 'P17181'
NODE_TGFBR1 = 'P36897'
TF_STAT2    = 'P52630'
TF_SMAD3    = 'P84022'
TF_TP63     = 'Q9H3D4'
TF_STAT3    = 'P40763'
TFS_INTEREST = [TF_STAT2, TF_SMAD3, TF_TP63, TF_STAT3]
TF_NAMES     = {TF_STAT2: 'STAT2', TF_SMAD3: 'SMAD3', TF_TP63: 'TP63', TF_STAT3: 'STAT3'}

# ==============================================================================
# DATA LOADING & ALIGNMENT
# ==============================================================================
os.makedirs(args.out_dir, exist_ok=True)
print2log("Loading data and model files...")

pkn_ann = pd.read_csv(args.PknAnnotation, sep='\t', header=None, names=['uniprot_id', 'symbol', 'is_tf', 'is_target'])
net = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[:, ['source', 'target', 'stimulation', 'inhibition']]
net = format_network(net, weight_label='mode_of_action', stimulation_label='stimulation', inhibition_label='inhibition')
nodeNames = np.union1d(net.source.values, net.target.values)

drugInput = pd.read_csv(args.DrugsIn, sep='\t', index_col=0)
TFOutput  = pd.read_csv(args.TFsOut, sep='\t', low_memory=False, index_col=0)
inputInfo = pd.read_csv(args.infoPath, sep='\t', low_memory=False, index_col=0)
inputInfo = inputInfo[inputInfo['time'] == args.timepoint]

drugInput = drugInput.loc[inputInfo.index, :]
TFOutput  = TFOutput.loc[inputInfo.index, :]

mask_dt_df = pd.read_csv(args.TargetsIn, sep='\t', index_col=0)
mask_dt_df = mask_dt_df.loc[drugInput.columns, :] 
drugSim_df = pd.read_csv(args.ChemicalSims, index_col=0)
drugSim_df = drugSim_df.loc[drugInput.columns, drugInput.columns]

inName  = np.intersect1d(nodeNames, mask_dt_df.columns.values)
outName = np.intersect1d(nodeNames, TFOutput.columns.values)
TFOutput    = TFOutput.loc[:, outName]
mask_dt_df  = mask_dt_df.loc[:, inName]

mask_dt_tensor = torch.tensor(mask_dt_df.values, dtype=torch.double).transpose(0, 1) 
drugSim_tensor = torch.tensor(drugSim_df.values, dtype=torch.double)

n_nodes = len(nodeNames)
node_to_idx = {node: i for i, node in enumerate(nodeNames)}
tf_to_idx = {tf: i for i, tf in enumerate(TFOutput.columns)}
drug_to_idx = {d: i for i, d in enumerate(drugInput.columns)}

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 'exp_factor': 50, 'tolerance': args.LEMBAS_tolerance, 'leak': args.LEMBAS_leak}
mod = SignalingModel(net=net, X_in=mask_dt_df, y_out=TFOutput, projection_amplitude_in=args.inputAmplitude, projection_amplitude_out=args.projectionAmplitude, weight_label='mode_of_action', source_label='source', target_label='target', bionet_params=bionet_params, dtype=torch.double, device=device)
drugLayer = DrugstoSignal(mask_dt_tensor, mask_dt_tensor.shape[1], mask_dt_tensor.shape[0], drugSim_tensor, momentum=args.drug_bn, drop=args.drug_drop, device=device).to(device).to(torch.double)

# ==============================================================================
# GROUND TRUTH CANDIDATE SELECTION
# ==============================================================================
gt_data = TFOutput.loc[inputInfo.index]

def get_gt_candidates():
    s2, s3, t6 = gt_data[TF_STAT2], gt_data[TF_SMAD3], gt_data[TF_TP63]
    mask_stat2 = (s2 > s2.quantile(0.95)) & (s3 >= 0.4) & (s3 <= 0.6) & (t6 >= 0.4) & (t6 <= 0.6)
    c_stat2 = gt_data[mask_stat2].index.tolist()
    mask_tp63 = (t6 < t6.quantile(0.05)) & (s3 >= 0.4) & (s3 <= 0.6) & (s2 >= 0.4) & (s2 <= 0.6)
    c_tp63 = gt_data[mask_tp63].index.tolist()
    mask_smad3 = (s3 < s3.quantile(0.05)) & (t6 >= 0.4) & (t6 <= 0.6) & (s2 >= 0.4) & (s2 <= 0.6)
    c_smad3 = gt_data[mask_smad3].index.tolist()
    mask_dual = (s3 < 0.4) & (t6 < 0.4) & (s2 >= 0.4) & (s2 <= 0.6)
    c_dual = gt_data[mask_dual].index.tolist()
    return {'STAT2_act': c_stat2, 'TP63_inh': c_tp63, 'SMAD3_inh': c_smad3, 'Dual_inh': c_dual}

gt_candidates_dict = get_gt_candidates()
all_gt_candidates = [c for sub in gt_candidates_dict.values() for c in sub]

# ==============================================================================
# PERTURBATION VECTORS
# ==============================================================================
def get_stim_vec(perturbations):
    vec = torch.zeros((1, n_nodes), dtype=torch.double, device=device)
    for node, val in perturbations.items():
        if node in node_to_idx: vec[0, node_to_idx[node]] = val
    return vec

stim_tgf_inh = get_stim_vec({NODE_TGFBR1: -S})
stim_if1_act = get_stim_vec({NODE_IFNAR1: S})
insilico_conds = {
    'TARGET_IFNAR1act_TGFBR1inh': get_stim_vec({NODE_IFNAR1: S, NODE_TGFBR1: -S}),
    'BG_IFNAR1act_TGFBR1act':     get_stim_vec({NODE_IFNAR1: S, NODE_TGFBR1: S}),
    'TRIPLE_SMAD3ko_TP63ko_STAT2act_BG': get_stim_vec({NODE_IFNAR1: S, NODE_TGFBR1: S, TF_STAT2: S, TF_SMAD3: -S, TF_TP63: -S}),
    'TP63ko_STAT2act_only':       get_stim_vec({TF_STAT2: S, TF_TP63: -S}),
    'STAT2_act_only':             get_stim_vec({TF_STAT2: S}),
    'IFNAR1_act_only':            get_stim_vec({NODE_IFNAR1: S}),
    'SMAD3inh_TP63inh_only':      get_stim_vec({TF_SMAD3: -S, TF_TP63: -S}),
    'TP63inh_only':               get_stim_vec({TF_TP63: -S}),
    'TGFBR1inh_only':             get_stim_vec({NODE_TGFBR1: -S})
}

# ==============================================================================
# MAIN ENSEMBLE EXECUTION
# ==============================================================================
results_insilico_tfs = []
results_insilico_nodes = []
results_candidate_validation = defaultdict(list)
results_opt_raw = [] 

dose_grid = np.linspace(0, 5, 50)
X_grid_template = torch.zeros((len(dose_grid), len(drugInput.columns)), dtype=torch.double, device=device)

print2log(f'Starting Ensemble Execution ({args.number_DTLEMBAS_models} models)...')
for i in range(args.number_DTLEMBAS_models):
    print2log(f' Model {i+1}/{args.number_DTLEMBAS_models}')
    mod.load_state_dict(torch.load(os.path.join(args.res_dir, f'{args.modelPattern}{i+1}signal_state_dict_{i+1}.pth'), map_location=device, weights_only=False))
    drugLayer.load_state_dict(torch.load(os.path.join(args.res_dir, f'{args.modelPattern}{i+1}drugLayer_state_dict_{i+1}.pth'), map_location=device, weights_only=False))
    mod.eval(); drugLayer.eval()
    
    with torch.no_grad():
        target_tf_prof = mod.output_layer(mod.signaling_network(insilico_conds['TARGET_IFNAR1act_TGFBR1inh'])).detach().cpu().numpy()[0]
        
        # 1. Standard Perturbations
        for name, vec in insilico_conds.items():
            node_state = mod.signaling_network(vec)
            tf_p = mod.output_layer(node_state).detach().cpu().numpy()[0]
            results_insilico_tfs.append({'condition': name, 'model_no': i+1, **dict(zip(TFOutput.columns, tf_p))})
            results_insilico_nodes.append({'condition': name, 'model_no': i+1, **dict(zip(nodeNames, node_state.detach().cpu().numpy()[0]))})
            
        # 2. Candidate Validation
        if all_gt_candidates:
            X_cand = torch.tensor(drugInput.loc[all_gt_candidates].values, dtype=torch.double, device=device)
            cand_preds = mod.output_layer(mod.signaling_network(mod.input_layer(drugLayer(X_cand)))).detach().cpu().numpy()
            for idx, cond in enumerate(all_gt_candidates):
                results_candidate_validation[cond].append(cand_preds[idx])

# ==============================================================================
# ENSEMBLE QC FILTER & OPTIMIZATION SELECTION
# ==============================================================================
print2log("Applying Strict Ensemble Filter...")
validated_candidates_dict = defaultdict(list)
unique_validated_drugs = set()

for cat, conds in gt_candidates_dict.items():
    for cond in conds:
        mean_p = np.array(results_candidate_validation[cond]).mean(axis=0)
        s2, s3, t6 = mean_p[tf_to_idx[TF_STAT2]], mean_p[tf_to_idx[TF_SMAD3]], mean_p[tf_to_idx[TF_TP63]]
        
        # Strict Criteria: Predicted effect confirms category and neutrality
        qc = False
        if cat == 'STAT2_act' and s2 > 0.6 and 0.4 <= s3 <= 0.6 and 0.4 <= t6 <= 0.6: qc = True
        elif cat == 'TP63_inh' and t6 < 0.4 and 0.4 <= s3 <= 0.6 and 0.4 <= s2 <= 0.6: qc = True
        elif cat == 'SMAD3_inh' and s3 < 0.4 and 0.4 <= t6 <= 0.6 and 0.4 <= s2 <= 0.6: qc = True
        elif cat == 'Dual_inh' and s3 < 0.4 and t6 < 0.4 and 0.4 <= s2 <= 0.6: qc = True
        
        if qc:
            validated_candidates_dict[cat].append(cond)
            unique_validated_drugs.add(drugInput.loc[cond][drugInput.loc[cond]>0].index[0])

# ==============================================================================
# DOSE OPTIMIZATION (ONLY VALIDATED)
# ==============================================================================
print2log(f"Optimizing doses for {len(unique_validated_drugs)} validated drugs...")
for i in range(args.number_DTLEMBAS_models):
    print2log(f' Optimizing using Model {i+1}/{args.number_DTLEMBAS_models}...')
    mod.load_state_dict(torch.load(os.path.join(args.res_dir, f'{args.modelPattern}{i+1}signal_state_dict_{i+1}.pth'), map_location=device, weights_only=False))
    drugLayer.load_state_dict(torch.load(os.path.join(args.res_dir, f'{args.modelPattern}{i+1}drugLayer_state_dict_{i+1}.pth'), map_location=device, weights_only=False))
    with torch.no_grad():
        target_tf_prof = mod.output_layer(mod.signaling_network(insilico_conds['TARGET_IFNAR1act_TGFBR1inh'])).detach().cpu().numpy()[0]
        for drug_smiles in unique_validated_drugs:
            X_batch = X_grid_template.clone()
            X_batch[:, drug_to_idx[drug_smiles]] = torch.tensor(dose_grid, dtype=torch.double, device=device)
            x_in = mod.input_layer(drugLayer(X_batch))
            out_A = mod.output_layer(mod.signaling_network(x_in + stim_tgf_inh)).detach().cpu().numpy()
            out_B = mod.output_layer(mod.signaling_network(x_in + stim_if1_act)).detach().cpu().numpy()
            for d_i, dose in enumerate(dose_grid):
                results_opt_raw.append({'drug': drug_smiles, 'log10_dose': dose, 'model_no': i+1, 
                                        'A_MSE': np.mean((target_tf_prof - out_A[d_i])**2),
                                        'B_MSE': np.mean((target_tf_prof - out_B[d_i])**2)})

# ==============================================================================
# POST-ANALYSIS: METRICS & HEATMAPS
# ==============================================================================
df_insilico_tfs = pd.DataFrame(results_insilico_tfs)
insilico_tf_means = df_insilico_tfs.groupby('condition')[TFOutput.columns].mean()
df_insilico_nodes = pd.DataFrame(results_insilico_nodes)
insilico_node_means = df_insilico_nodes.groupby('condition')[nodeNames].mean()

# 1. Substitution Analysis (Raw & Clipped)
for suffix, clip_vals in [('', False), ('_clipped', True)]:
    subst_rows = []
    for cand, ref, lbl in [('STAT2_act_only', 'IFNAR1_act_only', 'STAT2 as IFNAR1 Alt'),
                            ('SMAD3inh_TP63inh_only', 'TGFBR1inh_only', 'SMAD3+TP63 as TGFBR1 Alt'),
                            ('TP63inh_only', 'TGFBR1inh_only', 'TP63 as TGFBR1 Alt')]:
        v_c, v_r = insilico_tf_means.loc[cand].values, insilico_tf_means.loc[ref].values
        if clip_vals:
            v_c, v_r = np.clip(v_c, 0, 1), np.clip(v_r, 0, 1)
        subst_rows.append({'Comparison': lbl, 'MSE': np.mean((v_r - v_c)**2), 'R2': 1 - (np.sum((v_r - v_c)**2)/np.sum((v_r - np.mean(v_r))**2)),
                            'Pearson': pearsonr(v_r, v_c)[0], 'Spearman': spearmanr(v_r, v_c)[0]})
    pd.DataFrame(subst_rows).to_csv(os.path.join(args.out_dir, f'substitution_analysis_extended{suffix}.csv'), index=False)

# 2. Clustermaps (Force value clipping 0-1)
for df, name, dims in [(insilico_tf_means[TFS_INTEREST], 'Key_TFs', (8, 6)), 
                       (insilico_tf_means, 'All_TFs', (12, 10)), 
                       (insilico_node_means, 'All_Signaling', (15, 12))]:
    df_clipped = np.clip(df, 0, 1)
    g = sns.clustermap(df_clipped.T, cmap='coolwarm', vmin=0, vmax=1, annot=(name=='Key_TFs'), figsize=dims)
    plt.savefig(os.path.join(args.out_dir, f'clustermap_{name}.png'))
    plt.close()

# 3. PDF Validation Report (Only Validated)
pdf_path = os.path.join(args.out_dir, 'validated_drug_report.pdf')
with PdfPages(pdf_path) as pdf:
    for cat, conds in validated_candidates_dict.items():
        for cond in conds:
            preds = np.array(results_candidate_validation[cond])
            plt.figure(figsize=(8, 5))
            x = np.arange(len(TFS_INTEREST))
            plt.bar(x-0.2, [gt_data.loc[cond, tf] for tf in TFS_INTEREST], 0.4, label='Measured', color='skyblue')
            plt.bar(x+0.2, preds.mean(axis=0)[[tf_to_idx[tf] for tf in TFS_INTEREST]], 0.4, yerr=preds.std(axis=0)[[tf_to_idx[tf] for tf in TFS_INTEREST]], label='Predicted', color='salmon', capsize=4)
            plt.xticks(x, [TF_NAMES[tf] for tf in TFS_INTEREST]); plt.title(f'Validated: {cond} ({cat})'); plt.ylim(0, 1); pdf.savefig(); plt.close()

# 4. Optimization Curves (Only Validated)
df_opt = pd.DataFrame(results_opt_raw)
df_opt_grouped = df_opt.groupby(['drug', 'log10_dose']).mean().reset_index()
plt.figure(figsize=(10, 6))
for drug in unique_validated_drugs:
    data = df_opt_grouped[df_opt_grouped['drug'] == drug]
    plt.plot(data['log10_dose'], data['A_MSE'], label=f'Drug: {drug[:15]}...')
plt.xlabel('Log10 Dose'); plt.ylabel('MSE vs Target'); plt.title('Optimization Curves (Validated Drugs)'); plt.legend(); plt.savefig(os.path.join(args.out_dir, 'plot_optimization_curves.png'))

print2log(f"Success! Results saved in {args.out_dir}. Validated {len(unique_validated_drugs)} drugs.")