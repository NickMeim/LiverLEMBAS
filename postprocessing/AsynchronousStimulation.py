import os
import numpy as np
import pandas as pd
import torch
import sys 
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
import argparse
import logging
from pathlib import Path
from utility import *
import time
import itertools
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(description='Chained asynchronous node stimulation across timepoints')
parser.add_argument('--TargetsIn', action='store', type=str,
                    default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--DrugsIn', action='store', type=str,
                    default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TFsOut', action='store', type=str,
                    default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', action='store', type=str,
                    default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PknAnnotation', action='store', type=str,
                    default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--PKN', action='store', type=str,
                    default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--res_dir', action='store', type=str, default='../results/')
parser.add_argument('--infoPath', action='store', type=str,
                    default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--modelPattern', action='store', type=str, default='hepatocytes_dtlembas')
parser.add_argument('--inputAmplitude', action='store', type=float, default=1.0)
parser.add_argument('--projectionAmplitude', action='store', type=float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', action='store', type=int, default=90)
parser.add_argument('--LEMBAS_leak', action='store', type=float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', action='store', default=1e-5)
parser.add_argument('--drug_bn', action='store', default=0.2)
parser.add_argument('--drug_drop', action='store', default=0.1)
parser.add_argument('--annotation_file_for_visualization', action='store', type=str,
                    default='../data/annotation.csv')
parser.add_argument('--stimulation_strength', action='store', type=float, default=3000.0)
parser.add_argument('--grid_points', type=int, default=11, # it was 201 in the original code
                    help='Number of interpolation points per cytokine in [-stimulation_strength, +stimulation_strength]')
parser.add_argument('--num_steps', action='store', type=float, default=201)  # kept, unused here
parser.add_argument('--selected_stimulation_nodes', metavar='N', type=str, nargs='*', action='store',
                    help='Selected signaling nodes to stimulate (two nodes: e.g., [TGFb, IFNa])',
                    default=['P36897', 'P17181'])  # [TGFb, IFNa] by your convention
parser.add_argument('--ConvertToEmpProb', action='store_true', default=False)
parser.add_argument('--number_DTLEMBAS_models', action='store', type=int, default=50)
parser.add_argument('--TIME_ORDER', metavar='N', type=str, nargs='*', action='store',
                    help='Selected time points and order of time points',
                    default=["2 hour", "8 hour", "24 hour"])
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store',
                    help='Signaling nodes to optionally save / inspect',
                    default=['P42226','P42229','P51692','P40763','P52630','P42224','Q14765',
                             'Q15797','Q15796','P84022','Q13485','Q99717','O43541',
                             'P17181','P48551','P15260','P38484',
                             'P10914','Q00978','Q14653',
                             'P05231','Q13651'])
parser.add_argument(
    '--ensemble_mode',
    choices=['avg_then_chain', 'chain_then_avg'],
    default='avg_then_chain',
    help=('avg_then_chain: average the ensemble at each timepoint and carry the mean; '
          'chain_then_avg: propagate each model’s state across time and only average for output')
)
args = parser.parse_args()

# --- timepoints & where to find the weights for each time ---
TIME_ORDER = args.TIME_ORDER
res_dir = args.res_dir
# Example folder mapping; adjust to your layout
TIME_DIRS = {
    "2 hour":  os.path.join(res_dir, "2 hour",  "allmodels"),
    "8 hour":  os.path.join(res_dir,"allmodels"), # adjust if needed
    "24 hour": os.path.join(res_dir, "24 hour", "allmodels"),
}

### Initialize input variables
TargetsIn = args.TargetsIn
DrugsIn = args.DrugsIn
TFsOut = args.TFsOut
ChemicalSims = args.ChemicalSims
PknAnnotation = args.PknAnnotation
PKN = args.PKN
infoPath = args.infoPath
modelPattern = args.modelPattern
projection_amplitude_in = args.inputAmplitude
projection_amplitude_out = args.projectionAmplitude
ConvertToEmpProb = args.ConvertToEmpProb
number_DTLEMBAS_models = args.number_DTLEMBAS_models
stimulation_strength = args.stimulation_strength
selected_signaling_nodes = np.array(args.selected_signaling_nodes)
selected_stimulation_nodes = np.array(args.selected_stimulation_nodes)
grid_points = args.grid_points

# Initialize cores and device
n_cores = 12
utils.set_cores(n_cores)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log('Running on {}'.format(device))

# ensure result dirs
Path(res_dir).mkdir(parents=True, exist_ok=True)
data_path = res_dir  # we keep this name since the loader uses it

# ─────────────────────────────────────────────────────────────
# Load PKN, inputs/outputs, annotations — unchanged
# ─────────────────────────────────────────────────────────────
net = pd.read_csv(PKN, sep='\t', index_col=False).loc[:, ['source', 'target', 'stimulation', 'inhibition']]
drugInput = pd.read_csv(DrugsIn, sep='\t', index_col=0)
TFOutput = pd.read_csv(TFsOut, sep='\t', low_memory=False, index_col=0)
annot = pd.read_csv(args.annotation_file_for_visualization)
drugTargets = pd.read_csv(TargetsIn, sep='\t', low_memory=False, index_col=0)
inputInfo = pd.read_csv(infoPath, sep='\t', low_memory=False, index_col=0)

# labels expected by utility.format_network / model
weight_label = 'weight'
source_label = 'source'
target_label = 'target'
stimulation_label = 'stimulation'
inhibition_label = 'inhibition'

# format PKN
net = format_network(
    net,
    weight_label=weight_label,
    stimulation_label=stimulation_label,
    inhibition_label=inhibition_label
)

# Align tables to shared nodes
nodeNames = np.array(list(set(net[source_label]).union(set(net[target_label]))))
inName = drugTargets.columns.values
outconds = TFOutput.index.values
outName = TFOutput.columns.values
inName = np.intersect1d(nodeNames, inName)
outName = np.intersect1d(nodeNames, outName)
TFOutput = TFOutput.loc[outconds, outName]
drugTargets = drugTargets.loc[:, inName]
if ConvertToEmpProb:
    print2log('Convereted TF outputs to empirical probabilities')
    TFOutput = 1.0 / (1.0 + np.exp(-TFOutput))
drugInput = drugInput.loc[:, drugTargets.index] if 'Control' in drugInput.columns else drugInput
drugSim = pd.read_csv(ChemicalSims, index_col=0)
drugSim = drugSim.loc[drugInput.columns.values, drugInput.columns.values]
drugSim = torch.tensor(drugSim.values.copy(), dtype=torch.double)

# masks / model init
mask_dt = torch.tensor(np.array(drugTargets)).transpose(0, 1)
bionet_params = {
    'target_steps': 100,
    'max_steps': args.max_LEMBAS_iterations,
    'exp_factor': 50,
    'tolerance': args.LEMBAS_tolerance,
    'leak': args.LEMBAS_leak
}

mod = SignalingModel(
    net=net,
    X_in=drugTargets,
    y_out=TFOutput,
    projection_amplitude_in=projection_amplitude_in,
    projection_amplitude_out=projection_amplitude_out,
    weight_label=weight_label,
    source_label=source_label,
    target_label=target_label,
    bionet_params=bionet_params,
    dtype = torch.double,
    device=device
).to(device)

# drugLayer = DrugstoSignal(mask_dt,
#                           mask_dt.shape[1], mask_dt.shape[0],
#                           drugSim, momentum=args.drug_bn, drop=args.drug_drop,
#                           device=device).to(device)

print2log('Initialize node stimulation parameters…')
all_nodes = np.array(list(mod.node_idx_map.keys()))
all_tfs = mod.y_out.columns.values
num_nodes = len(all_nodes)
num_tfs = mod.y_out.shape[1]

# # Select specific nodes (keep your pattern): e.g., first=TGFb, second=IFNa
# subset_nodes = {k: mod.node_idx_map[k] for k in selected_stimulation_nodes if k in mod.node_idx_map}
# subset_nodes_inds = np.array(list(subset_nodes.values()))
# subset_nodes_names = np.array(list(subset_nodes.keys()))
# assert len(subset_nodes_inds) == 2, "Please pass exactly two stimulation nodes (e.g., [TGFb, IFNa])."

# Select some signaling nodes to save their activity state for later
subset_signaling_nodes = {k: mod.node_idx_map[k] for k in selected_signaling_nodes if k in mod.node_idx_map}
subset_signaling_nodes_inds = np.array(list(subset_signaling_nodes.values()))
subset_signaling_nodes_names = np.array(list(subset_signaling_nodes.keys()))

# One-hot basis for node stim
Xin_eye = torch.eye(num_nodes, dtype=torch.double, device=device)

# ---- Explicit, ordered mapping of stimulation nodes (TGFb first, IFNa second) ----
tgfb_name, ifna_name = args.selected_stimulation_nodes[:2]
if tgfb_name not in mod.node_idx_map or ifna_name not in mod.node_idx_map:
    raise KeyError(f"Selected nodes not in network: "
                   f"{tgfb_name in mod.node_idx_map=}, {ifna_name in mod.node_idx_map=}")

idx_tgfb = int(mod.node_idx_map[tgfb_name])
idx_ifna = int(mod.node_idx_map[ifna_name])
# Row basis for each stimulation (shape: 1 × n_nodes), double dtype on the correct device
e_T = Xin_eye[idx_tgfb, :].unsqueeze(0).to(device=device, dtype=torch.double)
e_I = Xin_eye[idx_ifna, :].unsqueeze(0).to(device=device, dtype=torch.double)

print2log(f"Stim nodes: TGFb='{tgfb_name}' @ {idx_tgfb}, IFNa='{ifna_name}' @ {idx_ifna}")

# ─────────────────────────────────────────────────────────────
# Timepoint handling & ensemble averaging (unchanged pattern)
# ─────────────────────────────────────────────────────────────
if not all(os.path.isdir(TIME_DIRS[tp]) for tp in TIME_ORDER):
    TIME_DIRS = {tp: res_dir for tp in TIME_ORDER}
    print2log("Timepoint model folders not found under res_dir; using res_dir for all timepoints.")

def load_timepoint_weights(tp: str, i_model: int):
    sig_path = os.path.join(TIME_DIRS[tp], f"{modelPattern}{i_model}signal_state_dict_{i_model}.pth")
    try:
        sd_sig = torch.load(sig_path, map_location=device, weights_only=True)
    except TypeError:
        sd_sig = torch.load(sig_path, map_location=device)
    mod.load_state_dict(sd_sig)
    mod.eval()
    mod.signaling_network.weights.data.masked_fill_(mask = mod.signaling_network.mask, value = 0.0)

@torch.no_grad()
def custom_ensemble_step_mean(tp: str, X_in_nodes: torch.Tensor, number_DTLEMBAS_models: int):
    B = X_in_nodes.size(0)
    sum_full = torch.zeros_like(X_in_nodes)
    sum_tf = torch.zeros(B, num_tfs, device=device, dtype=torch.double)
    for i in range(1, number_DTLEMBAS_models + 1):
        load_timepoint_weights(tp, i)
        # Instead of running Yfull = mod.signaling_network(X_in_nodes), customely run Bionet's forward pass
        X_bias = mod.signaling_network.bias # this is the bias with the projection_amplitude included
        # As X_new we are going to use initially X_in_nodes
        X_new = X_in_nodes.T
        for t in range(mod.signaling_network.training_params['max_steps']): # like an RNN, updating from previous time step
            X_old = X_new
            X_new = torch.mm(mod.signaling_network.weights, X_new) # scale matrix by edge weights
            X_new = X_new + X_bias  # add original values and bias       
            X_new = mod.signaling_network.activation(X_new, mod.signaling_network.training_params['leak'])
            if (t % 10 == 0) and (t > 20):
                diff = torch.max(torch.abs(X_new - X_old))    
                if diff.lt(mod.signaling_network.training_params['tolerance']):
                    break
        Yfull = X_new.T
        TF = mod.output_layer(Yfull)
        sum_full += Yfull
        sum_tf += TF
    return sum_full / number_DTLEMBAS_models, sum_tf / number_DTLEMBAS_models

# ─────────────────────────────────────────────────────────────
# Function to pre-calculate all simulation paths
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def create_stimulation_plan(steps, tgfb_grid, ifna_grid):
    """
    Pre-calculates all stimulus combinations for an experiment.
    Returns a dictionary of stimulation tensors, one for each timepoint,
    and a DataFrame for tracking the parameters of each path.
    """
    stim_events = {}
    for tp, t_amp, i_amp, _ in steps:
        if t_amp != 0.0:
            stim_events.setdefault(tp, []).append('T')
        if i_amp != 0.0:
            stim_events.setdefault(tp, []).append('I')

    # Determine which grids to use for the product
    active_grids = []
    # Use a dictionary to map stimulus type to its grid and timepoint for bookkeeping
    stim_map = {}
    for tp, stim_types in stim_events.items():
        if 'T' in stim_types and 'I' in stim_types: # T and I applied together
            active_grids.append(list(zip(tgfb_grid.cpu().numpy(), ifna_grid.cpu().numpy())))
            stim_map[len(active_grids)-1] = {'T': tp, 'I': tp}
        else:
            if 'T' in stim_types:
                active_grids.append(tgfb_grid)
                stim_map[len(active_grids)-1] = {'T': tp}
            if 'I' in stim_types:
                active_grids.append(ifna_grid)
                stim_map[len(active_grids)-1] = {'I': tp}

    if not active_grids: # No stimulation in this experiment
        path_combinations = [(0.0,)] # Dummy combination
        num_paths = 1
    else:
        path_combinations = list(itertools.product(*active_grids))
        num_paths = len(path_combinations)

    # Create bookkeeping dataframe and stimulation plan tensors
    path_info = pd.DataFrame(index=range(num_paths), columns=['T_at_0h', 'I_at_0h', 'T_at_2h', 'I_at_2h', 'T_at_8h', 'I_at_8h']).fillna(0.0)
    stim_plan = {tp: torch.zeros(num_paths, num_nodes, device=device, dtype=torch.double) for tp in TIME_ORDER}

    for path_idx, combo in enumerate(path_combinations):
        for grid_idx, value in enumerate(combo):
            event_info = stim_map.get(grid_idx, {})
            if 'T' in event_info and 'I' in event_info: # Combined T+I grid
                t_val, i_val = value
                tp = event_info['T']
                # FIX: Squeeze the 2D result to 1D before adding
                stim_plan[tp][path_idx, :] += (e_T * t_val + e_I * i_val).squeeze(0)
                path_info.loc[path_idx, f'T_at_{tp.split(" ")[0]}h'] = t_val
                path_info.loc[path_idx, f'I_at_{tp.split(" ")[0]}h'] = i_val
            elif 'T' in event_info:
                tp = event_info['T']
                # FIX: Squeeze the 2D result to 1D before adding
                stim_plan[tp][path_idx, :] += (e_T * value.item()).squeeze(0)
                path_info.loc[path_idx, f'T_at_{tp.split(" ")[0]}h'] = value.item()
            elif 'I' in event_info:
                tp = event_info['I']
                # FIX: Squeeze the 2D result to 1D before adding
                stim_plan[tp][path_idx, :] += (e_I * value.item()).squeeze(0)
                path_info.loc[path_idx, f'I_at_{tp.split(" ")[0]}h'] = value.item()
    
    # Fill in the "added_now" columns for clarity in final output
    for tp in TIME_ORDER:
        key = ""
        if tp.startswith("2"): key = "0h"
        elif tp.startswith("8"): key = "2h"
        elif tp.startswith("24"): key = "8h"
        
        if key:
            # Check if the column exists before trying to access it
            if f'T_at_{key}' in path_info.columns:
                 path_info[f'TGFb_added_at_{key}'] = path_info[f'T_at_{key}']
            if f'I_at_{key}' in path_info.columns:
                 path_info[f'IFNa_added_at_{key}'] = path_info[f'I_at_{key}']

    return stim_plan, path_info, num_paths

# ─────────────────────────────────────────────────────────────
# Define the experiments as desired
# ─────────────────────────────────────────────────────────────
def exp_definitions(amp):
    """
    Returns list of experiments; each experiment is a list of steps.
    Step = (timepoint, tgfb_to_add_now, ifna_to_add_now, description_str)
    """
    T, I = amp, amp
    return [
        # 1) TGFb+IFNa at 0h → run 2h, then 8h, then 24h
        ("Exp1", [
            ("2 hour",  T, I,  "apply TGFb & IFNa at 0h (first step)"),
            ("8 hour",  0, 0,  "carry only"),
            ("24 hour", 0, 0,  "carry only"),
        ]),
        # 2) empty → 2h; then add TGFb+IFNa → run 8h, then 24h
        ("Exp2", [
            ("2 hour",  0, 0,  "empty through 2h"),
            ("8 hour",  T, I,  "add TGFb & IFNa after 2h"),
            ("24 hour", 0, 0,  "carry only"),
        ]),
        # 3) empty → 8h; then add TGFb+IFNa → run 24h
        ("Exp3", [
            # skip 2h
            ("8 hour",  0, 0,  "empty through 8h"),
            ("24 hour", T, I,  "add TGFb & IFNa after 8h"),
        ]),
        # 4) TGFb at 0h → run 2h; then add IFNa → run 8h, then 24h
        ("Exp4", [
            ("2 hour",  T, 0,  "apply TGFb at 0h"),
            ("8 hour",  0, I,  "add IFNa after 2h"),
            ("24 hour", 0, 0,  "carry only"),
        ]),
        # 5) TGFb at 0h → run 8h; then add IFNa → run 24h
        ("Exp5", [
            # skip 2h
            ("8 hour",  T, 0,  "apply TGFb at 0h"),
            ("24 hour", 0, I,  "add IFNa after 8h"),
        ]),
        # 6) empty → 2h; add TGFb → run 8h; add IFNa → run 24h
        ("Exp6", [
            ("2 hour",  0, 0,  "empty through 2h"),
            ("8 hour",  T, 0,  "add TGFb after 2h"),
            ("24 hour", 0, I,  "add IFNa after 8h"),
        ]),
        # 7) IFNa at 0h → run 2h; then add TGFb → run 8h, then 24h
        ("Exp7", [
            ("2 hour",  0, I,  "apply IFNa at 0h"),
            ("8 hour",  T, 0,  "add TGFb after 2h"),
            ("24 hour", 0, 0,  "carry only"),
        ]),
        # 8) IFNa at 0h → run 8h; then add TGFb → run 24h
        ("Exp8", [
            ("8 hour",  0, I,  "apply IFNa at 0h"),
            ("24 hour", T, 0,  "add TGFb after 8h"),
        ]),
        # 9) empty → 2h; add IFNa → run 8h; add TGFb → run 24h
        ("Exp9", [
            ("2 hour",  0, 0,  "empty through 2h"),
            ("8 hour",  0, I,  "add IFNa after 2h"),
            ("24 hour", T, 0,  "add TGFb after 8h"),
        ]),
    ]



# ─────────────────────────────────────────────────────────────
# Execute experiments
# ─────────────────────────────────────────────────────────────
print2log(f"Begin asynchronous experiments ({args.ensemble_mode})…")
out_dir = os.path.join(res_dir, "chained_async_static_batch")
Path(out_dir).mkdir(parents=True, exist_ok=True)

all_results = []
start_time = time.time()
experiments = exp_definitions(stimulation_strength)

# Define stimulation grids once
tgfb_grid = torch.linspace(-stimulation_strength, stimulation_strength, grid_points, device=device, dtype=torch.double)
ifna_grid = torch.linspace(-stimulation_strength, stimulation_strength, grid_points, device=device, dtype=torch.double)

for exp_name, steps in experiments:
    print2log(f'Begun Experiment: {exp_name}')

    # NEW: Pre-calculate the stimulation plan for all parallel simulation paths
    active_timepoints = [s[0] for s in steps]
    stim_plan, path_info, num_paths = create_stimulation_plan(steps, tgfb_grid, ifna_grid)

    # --- Run simulation based on ensemble mode ---
    if args.ensemble_mode == 'avg_then_chain':
        master_states = torch.zeros(num_paths, num_nodes, device=device, dtype=torch.double)
        for tp in active_timepoints:
            X_in = master_states + stim_plan[tp]
            mean_full, mean_tf = custom_ensemble_step_mean(tp, X_in, number_DTLEMBAS_models)
            
            df = pd.DataFrame(mean_tf.detach().cpu().numpy(), columns=all_tfs)
            df['timepoint'] = tp
            all_results.append(pd.concat([path_info.reset_index(drop=True), df], axis=1))

            master_states = mean_full.detach()

    elif args.ensemble_mode == 'chain_then_avg':
        states_per_model = [torch.zeros(num_paths, num_nodes, device=device, dtype=torch.double) for _ in range(number_DTLEMBAS_models)]
        for tp in active_timepoints:
            sum_tf = torch.zeros(num_paths, num_tfs, device=device, dtype=torch.double)
            next_states_per_model = []
            
            for i in range(number_DTLEMBAS_models):
                load_timepoint_weights(tp, i + 1)
                X_in = states_per_model[i] + stim_plan[tp]
                Yfull = mod.signaling_network(X_in)
                TF = mod.output_layer(Yfull)
                sum_tf += TF
                next_states_per_model.append(Yfull.detach())
            
            mean_tf = sum_tf / number_DTLEMBAS_models
            df = pd.DataFrame(mean_tf.detach().cpu().numpy(), columns=all_tfs)
            df['timepoint'] = tp
            all_results.append(pd.concat([path_info.reset_index(drop=True), df], axis=1))
            
            states_per_model = next_states_per_model

    # Add experiment name to the results collected in this loop
    for df in all_results[-len(active_timepoints):]:
        df['experiment'] = exp_name
        df['description'] = next((s[3] for s in steps if s[0] == df['timepoint'].iloc[0]), "")


# --- Finalize and Save ---
final_df = pd.concat(all_results, ignore_index=True)
out_csv = os.path.join(out_dir, f'TFs_chained_async_static_{args.ensemble_mode}.csv')
final_df.to_csv(out_csv, index=False)

elapsed = time.time() - start_time
print2log(f"Saved results → {out_csv}")
print2log(f"Total time: {elapsed/60:.1f} min")
print2log("Finished!")