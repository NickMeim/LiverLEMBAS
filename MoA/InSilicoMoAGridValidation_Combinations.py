import os
import sys
import torch
import logging
import argparse
import numpy as np
import pandas as pd

# ── LEMBAS path ────────────────────────────────────────────────────────────────
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
parser = argparse.ArgumentParser(prog='InSilicoMoAGridValidation_MultiKO')
parser.add_argument('--TargetsIn', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', type=str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', type=str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--res_dir', type=str, default='../results/24 hour/allmodels/')
parser.add_argument('--out_dir', type=str, default='../results/24 hour/insilico_grid_combinations_ko/')
parser.add_argument('--number_DTLEMBAS_models', type=int, default=50)
parser.add_argument('--modelPattern', type=str, default='hepatocytes_dtlembas')
parser.add_argument('--inputAmplitude', type=float, default=3.0)
parser.add_argument('--projectionAmplitude', type=float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', type=int, default=90)
parser.add_argument('--LEMBAS_leak', type=float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', default=1e-5)
parser.add_argument('--grid_resolution', type=int, default=21, help='Points per axis for stimulus grid')
parser.add_argument('--stimulation_strength', type=float, default=3000.0) # For KO
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S = args.stimulation_strength

# Node/TF Identifiers
NODE_IFNAR1 = 'P17181'
NODE_TGFBR1 = 'P36897'
TF_STAT2    = 'P52630'
TF_SMAD3    = 'P84022'
TF_TP63     = 'Q9H3D4'
TF_STAT3    = 'P40763'

# ==============================================================================
# DATA LOADING
# ==============================================================================
os.makedirs(args.out_dir, exist_ok=True)
print2log("Loading data and model structures...")

pkn_ann = pd.read_csv(args.PknAnnotation, sep='\t', header=None, names=['uniprot_id', 'symbol', 'is_tf', 'is_target'])
net = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[:, ['source', 'target', 'stimulation', 'inhibition']]
net = format_network(net, weight_label='mode_of_action', stimulation_label='stimulation', inhibition_label='inhibition')
nodeNames = np.union1d(net.source.values, net.target.values)

TFOutput = pd.read_csv(args.TFsOut, sep='\t', low_memory=False, index_col=0)
outName  = np.intersect1d(nodeNames, TFOutput.columns.values)
TFOutput = TFOutput.loc[:, outName]

mask_dt_df = pd.read_csv(args.TargetsIn, sep='\t', index_col=0)
inName = np.intersect1d(nodeNames, mask_dt_df.columns.values)
mask_dt_df = mask_dt_df.loc[:, inName]

n_nodes = len(nodeNames)
node_to_idx = {node: i for i, node in enumerate(nodeNames)}
tf_to_idx = {tf: i for i, tf in enumerate(TFOutput.columns)}

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================
bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 'exp_factor': 50, 'tolerance': args.LEMBAS_tolerance, 'leak': args.LEMBAS_leak}
mod = SignalingModel(net=net, X_in=mask_dt_df, y_out=TFOutput, projection_amplitude_in=args.inputAmplitude, projection_amplitude_out=args.projectionAmplitude, weight_label='mode_of_action', source_label='source', target_label='target', bionet_params=bionet_params, dtype=torch.double, device=device)

# ==============================================================================
# EXPERIMENT SETUP
# ==============================================================================
perturbation_scenarios = {
    'TP63_KO': {TF_TP63: -S},
    'SMAD3_KO': {TF_SMAD3: -S},
    'TP63_SMAD3_KO': {TF_TP63: -S, TF_SMAD3: -S},
    'STAT2_KO': {TF_STAT2: -S},
    'STAT3_KO': {TF_STAT3: -S},
    'STAT2_STAT3_KO': {TF_STAT2: -S, TF_STAT3: -S},
    'STAT2_TP63_SMAD3_KO': {TF_STAT2: -S, TF_TP63: -S, TF_SMAD3: -S},
    'STAT3_STAT2_TP63_SMAD3_KO': {TF_STAT3: -S, TF_STAT2: -S, TF_TP63: -S, TF_SMAD3: -S}
}

grid_vals = np.linspace(-S, S, args.grid_resolution)
grid_X, grid_Y = np.meshgrid(grid_vals, grid_vals, indexing='ij')
grid_points = np.vstack([grid_X.ravel(), grid_Y.ravel()]).T

# ==============================================================================
# EXECUTION LOOP
# ==============================================================================
all_results = []
print2log(f'Starting Ensemble Execution ({args.number_DTLEMBAS_models} models)...')

for i in range(args.number_DTLEMBAS_models):
    print2log(f'  Model {i+1}/{args.number_DTLEMBAS_models}')
    mod.load_state_dict(torch.load(os.path.join(args.res_dir, f'{args.modelPattern}{i+1}signal_state_dict_{i+1}.pth'), map_location=device, weights_only=False))
    mod.eval()
    
    with torch.no_grad():
        for scenario_name, ko_dict in perturbation_scenarios.items():
            base_ko_vec = torch.zeros((1, n_nodes), dtype=torch.double, device=device)
            for node, val in ko_dict.items():
                if node in node_to_idx: base_ko_vec[0, node_to_idx[node]] = val
            
            batch_stim = base_ko_vec.repeat(len(grid_points), 1)
            if NODE_IFNAR1 in node_to_idx and NODE_TGFBR1 in node_to_idx:
                batch_stim[:, node_to_idx[NODE_IFNAR1]] += torch.tensor(grid_points[:, 0], device=device)
                batch_stim[:, node_to_idx[NODE_TGFBR1]] += torch.tensor(grid_points[:, 1], device=device)
            
            tf_preds = mod.output_layer(mod.signaling_network(batch_stim)).detach().cpu().numpy()
            
            for j in range(len(grid_points)):
                res_row = {'Model_No': i+1, 'Scenario': scenario_name, 'IFNAR1_Stim': grid_points[j, 0], 'TGFBR1_Stim': grid_points[j, 1]}
                for tf_idx, tf_name in enumerate(TFOutput.columns): res_row[tf_name] = tf_preds[j, tf_idx]
                all_results.append(res_row)

df_results = pd.DataFrame(all_results)
out_file = os.path.join(args.out_dir, 'insilico_grid_multi_ko_results.csv')
df_results.to_csv(out_file, index=False)
print2log(f"Done. File saved to: {out_file}")