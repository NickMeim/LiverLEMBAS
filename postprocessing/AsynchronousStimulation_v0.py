import os
import numpy as np
import pandas as pd
import torch
import sys
import itertools
import time
from pathlib import Path

# --- Setup: Ensure LEMBAS utilities can be found ---
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
assert os.path.isdir(gpu_lembas_path), f"LEMBAS library path not found. Please check the path: {gpu_lembas_path}"
sys.path.insert(1, os.path.join(gpu_lembas_path))
from utility import SignalingModel, format_network, utils

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Run a series of chained, asynchronous node stimulation experiments.')
# Model/Data Paths
parser.add_argument('--PKN', action='store', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--TargetsIn', action='store', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store', type=str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--res_dir', action='store', type=str, default='../results/')
# Model Hyperparameters
parser.add_argument('--modelPattern', action='store', type=str, default='hepatocytes_dtlembas')
parser.add_argument('--inputAmplitude', action='store', type=float, default=1.0)
parser.add_argument('--projectionAmplitude', action='store', type=float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', action='store', type=int, default=90)
parser.add_argument('--LEMBAS_leak', action='store', type=float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', action='store', type=float, default=1e-5)
parser.add_argument('--number_DTLEMBAS_models', action='store', type=int, default=50)
# Simulation Parameters
parser.add_argument('--stimulation_strength', action='store', type=float, default=3000.0)
parser.add_argument('--grid_points', type=int, default=201, help='Number of points in the stimulation grid for each cytokine.')
parser.add_argument('--selected_stimulation_nodes', metavar='N', type=str, nargs='*', default=['P36897', 'P17181']) # TGFb, IFNa

args = parser.parse_args()

# --- Basic Setup ---
TIME_ORDER = ["2 hour", "8 hour", "24 hour"]
TIME_MAP = {"0h": 0, "2h": 2, "8h": 8, "24h": 24} # For numeric time values
res_dir = args.res_dir
TIME_DIRS = {
    "2 hour":  os.path.join(res_dir, "2 hour",  "allmodels"),
    "8 hour":  os.path.join(res_dir, "allmodels"),
    "24 hour": os.path.join(res_dir, "24 hour", "allmodels"),
}
utils.set_cores(12)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log(f'Running on {device}')

# --- Data and Model Loading ---
print2log("Loading data and initializing models...")
net = pd.read_csv(args.PKN, sep='\t', usecols=['source', 'target', 'stimulation', 'inhibition'])
drugTargets = pd.read_csv(args.TargetsIn, sep='\t', index_col=0)
TFOutput = pd.read_csv(args.TFsOut, sep='\t', index_col=0)

# Align dataframes
weight_label, source_label, target_label = 'weight', 'source', 'target'
net = format_network(net, weight_label=weight_label, stimulation_label='stimulation', inhibition_label='inhibition')
nodeNames = np.array(list(set(net[source_label]).union(set(net[target_label]))))
inName = np.intersect1d(nodeNames, drugTargets.columns.values)
outName = np.intersect1d(nodeNames, TFOutput.columns.values)
TFOutput = TFOutput.loc[:, outName]
drugTargets = drugTargets.loc[:, inName]

# Initialize model
bionet_params = {'max_steps': args.max_LEMBAS_iterations, 'leak': args.LEMBAS_leak, 'tolerance': args.LEMBAS_tolerance}
mod = SignalingModel(net=net, X_in=drugTargets, y_out=TFOutput, projection_amplitude_in=args.inputAmplitude,
                     projection_amplitude_out=args.projectionAmplitude, bionet_params=bionet_params,
                     weight_label=weight_label, source_label=source_label, target_label=target_label,
                     dtype=torch.double, device=device).to(device)

all_tfs = mod.y_out.columns.values
num_nodes = len(mod.node_idx_map)
tgfb_name, ifna_name = args.selected_stimulation_nodes[:2]
idx_tgfb = int(mod.node_idx_map[tgfb_name])
idx_ifna = int(mod.node_idx_map[ifna_name])
e_T = torch.eye(num_nodes, dtype=torch.double, device=device)[idx_tgfb, :].unsqueeze(0)
e_I = torch.eye(num_nodes, dtype=torch.double, device=device)[idx_ifna, :].unsqueeze(0)

# --- Core Simulation Functions ---
def load_timepoint_weights(tp: str, i_model: int):
    sig_path = os.path.join(TIME_DIRS[tp], f"{args.modelPattern}{i_model}signal_state_dict_{i_model}.pth")
    mod.load_state_dict(torch.load(sig_path, map_location=device, weights_only=True))
    mod.eval()

@torch.no_grad()
def ensemble_step_mean(tp: str, X_in_nodes: torch.Tensor):
    """Averages the ensemble predictions for a given input batch at a specific timepoint."""
    sum_full_state = torch.zeros_like(X_in_nodes)
    sum_tf = torch.zeros(X_in_nodes.size(0), len(all_tfs), device=device, dtype=torch.double)
    for i in range(1, args.number_DTLEMBAS_models + 1):
        load_timepoint_weights(tp, i)
        Y_full_state = mod.signaling_network(X_in_nodes)
        Y_tf = mod.output_layer(Y_full_state)
        sum_full_state += Y_full_state
        sum_tf += Y_tf
    return sum_full_state / args.number_DTLEMBAS_models, sum_tf / args.number_DTLEMBAS_models

def define_experiments():
    """Defines the 9 experimental protocols in a structured way for easy lookup."""
    return [
        ("Exp1", {"TGFb": "0h", "IFNa": "0h"}),
        ("Exp2", {"TGFb": "2h", "IFNa": "2h"}),
        ("Exp3", {"TGFb": "8h", "IFNa": "8h"}),
        ("Exp4", {"TGFb": "0h", "IFNa": "2h"}),
        ("Exp5", {"TGFb": "0h", "IFNa": "8h"}),
        ("Exp6", {"TGFb": "2h", "IFNa": "8h"}),
        ("Exp7", {"TGFb": "2h", "IFNa": "0h"}), # Reversed order from Exp4
        ("Exp8", {"TGFb": "8h", "IFNa": "0h"}), # Reversed order from Exp5
        ("Exp9", {"TGFb": "8h", "IFNa": "2h"}), # Reversed order from Exp6
    ]

# --- Main Execution Logic ---
def main():
    print2log("Starting asynchronous stimulation experiments...")
    start_time = time.time()
    
    out_dir = os.path.join(res_dir, "async_experiments_tidy")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    experiments = define_experiments()
    tgfb_grid = torch.linspace(-args.stimulation_strength, args.stimulation_strength, args.grid_points, device=device, dtype=torch.double)
    ifna_grid = torch.linspace(-args.stimulation_strength, args.stimulation_strength, args.grid_points, device=device, dtype=torch.double)
    
    all_results_list = []

    for exp_name, stim_protocol in experiments:
        print2log(f"--- Running Experiment: {exp_name} ---")
        
        # 1. Create all combinations of stimulus strengths
        path_combinations = list(itertools.product(tgfb_grid, ifna_grid))
        num_paths = len(path_combinations)
        
        # 2. Initialize master state for this experiment's simulations
        master_states = torch.zeros(num_paths, num_nodes, device=device, dtype=torch.double)

        # 3. Iterate through the time course (0h -> 2h -> 8h -> 24h)
        previous_time_h = 0
        for measurement_tp_str in TIME_ORDER:
            # --- Construct the dictionary key correctly (e.g., "2h") ---
            measurement_time_h = TIME_MAP[f"{measurement_tp_str.split(' ')[0]}h"]
            
            # Apply any stimulations scheduled for the *start* of this time interval
            stim_tensor = torch.zeros(num_paths, num_nodes, device=device, dtype=torch.double)
            for path_idx, (t_val, i_val) in enumerate(path_combinations):
                if stim_protocol.get("TGFb") == f"{previous_time_h}h":
                    # Use .squeeze(0) to match tensor dimensions for the in-place addition
                    stim_tensor[path_idx, :] += e_T.squeeze(0) * t_val
                if stim_protocol.get("IFNa") == f"{previous_time_h}h":
                    stim_tensor[path_idx, :] += e_I.squeeze(0) * i_val
            
            X_in = master_states + stim_tensor
            
            # Run simulation using the model for the current measurement timepoint
            mean_full_state, mean_tf = ensemble_step_mean(measurement_tp_str, X_in)

            # Store results in the requested tidy format
            df = pd.DataFrame(mean_tf.cpu().numpy(), columns=all_tfs)
            df['experiment'] = exp_name
            df['TGFb_strength'] = [c[0].item() for c in path_combinations]
            df['IFNa_strength'] = [c[1].item() for c in path_combinations]
            df['TGFb_applied_at'] = stim_protocol.get("TGFb", "None")
            df['IFNa_applied_at'] = stim_protocol.get("IFNa", "None")
            df['measured_at'] = f"{measurement_time_h}h"
            all_results_list.append(df)

            # Update state for the next step ("carry-on")
            master_states = mean_full_state.detach()
            previous_time_h = measurement_time_h

    # 4. Finalize and save all data
    final_df = pd.concat(all_results_list, ignore_index=True)
    out_csv = os.path.join(out_dir, 'async_stimulation_results_tidy.csv')
    final_df.to_csv(out_csv, index=False)
    
    elapsed = time.time() - start_time
    print2log(f"\nSaved all results to {out_csv}")
    print2log(f"All experiments finished in {elapsed/60:.1f} min.")

if __name__ == "__main__":
    main()