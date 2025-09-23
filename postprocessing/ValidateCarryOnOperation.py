import os
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
import time
import argparse
import logging

# --- Setup: Ensure LEMBAS utilities can be found ---
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
assert os.path.isdir(gpu_lembas_path), f"LEMBAS library path not found. Please check the path: {gpu_lembas_path}"
sys.path.insert(1, os.path.join(gpu_lembas_path))
from utility import SignalingModel, format_network, utils

# --- Plotting Imports ---
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
## PLSR imports
import pyreadr
from sklearn.cross_decomposition import PLSRegression

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

# --- Argument Parsing (from your old, correct script) ---
parser = argparse.ArgumentParser(description='Validation test for the "carry-on" mechanism in chained simulations.')
parser.add_argument('--PKN', action='store', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--TargetsIn', action='store', type=str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store', type=str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--res_dir', action='store', type=str, default='../results/')
parser.add_argument('--modelPattern', action='store', type=str, default='hepatocytes_dtlembas')
parser.add_argument('--inputAmplitude', action='store', type=float, default=1.0)
parser.add_argument('--projectionAmplitude', action='store', type=float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', action='store', type=int, default=90)
parser.add_argument('--LEMBAS_leak', action='store', type=float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', action='store', type=float, default=1e-5)
parser.add_argument('--stimulation_strength', action='store', type=float, default=3000.0)
parser.add_argument('--grid_points', type=int, default=11)
parser.add_argument('--selected_stimulation_nodes', metavar='N', type=str, nargs='*', default=['P36897', 'P17181'])
parser.add_argument('--number_DTLEMBAS_models', action='store', type=int, default=50)
parser.add_argument('--X_invivo_path', action='store', help='Path to in vivo TF activities (.rds)', default='../data/Xh_tfs_govaere.rds')
parser.add_argument('--Y_invivo_path', action='store', help='Path to in vivo phenotypes (.rds)', default='../data/Yh_tfs_govaere.rds')
parser.add_argument('--num_LVs', action='store', type=int, help='Number of latent variables for PLSR', default=8)
args = parser.parse_args()

# --- Basic Setup ---
TIME_ORDER = ["2 hour", "8 hour", "24 hour"]
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
net = pd.read_csv(args.PKN, sep='\t', usecols=['source', 'target', 'stimulation', 'inhibition'])
drugTargets = pd.read_csv(args.TargetsIn, sep='\t', index_col=0)
TFOutput = pd.read_csv(args.TFsOut, sep='\t', index_col=0)
weight_label, source_label, target_label = 'weight', 'source', 'target'
net = format_network(net, weight_label=weight_label, stimulation_label='stimulation', inhibition_label='inhibition')
nodeNames = np.array(list(set(net[source_label]).union(set(net[target_label]))))
inName = np.intersect1d(nodeNames, drugTargets.columns.values)
outName = np.intersect1d(nodeNames, TFOutput.columns.values)
TFOutput = TFOutput.loc[:, outName]
drugTargets = drugTargets.loc[:, inName]
bionet_params = {'max_steps': args.max_LEMBAS_iterations, 'leak': args.LEMBAS_leak, 'tolerance': args.LEMBAS_tolerance}
mod = SignalingModel(net=net, X_in=drugTargets, y_out=TFOutput, projection_amplitude_in=args.inputAmplitude,
                     projection_amplitude_out=args.projectionAmplitude, bionet_params=bionet_params,
                     weight_label=weight_label, source_label=source_label, target_label=target_label,
                     dtype=torch.double, device=device).to(device)

all_tfs = mod.y_out.columns.values
all_nodes = [k for k, v in sorted(mod.node_idx_map.items(), key=lambda item: item[1])]
num_nodes, num_tfs = len(mod.node_idx_map), mod.y_out.shape[1]
tgfb_name, ifna_name = args.selected_stimulation_nodes[:2]
idx_tgfb = int(mod.node_idx_map[tgfb_name])
idx_ifna = int(mod.node_idx_map[ifna_name])
e_T = torch.eye(num_nodes, dtype=torch.double, device=device)[idx_tgfb, :].unsqueeze(0)
e_I = torch.eye(num_nodes, dtype=torch.double, device=device)[idx_ifna, :].unsqueeze(0)

# --- Core Simulation Functions (restored from your working script) ---
def load_timepoint_weights(tp: str, i_model: int):
    sig_path = os.path.join(TIME_DIRS[tp], f"{args.modelPattern}{i_model}signal_state_dict_{i_model}.pth")
    mod.load_state_dict(torch.load(sig_path, map_location=device, weights_only=True))
    mod.eval()

@torch.no_grad()
def custom_ensemble_step_mean(tp: str, X_in_nodes: torch.Tensor):
    """The 'Chained' method using a manual forward pass."""
    B, sum_full = X_in_nodes.size(0), torch.zeros_like(X_in_nodes)
    sum_tf = torch.zeros(B, num_tfs, device=device, dtype=torch.double)
    for i in range(1, args.number_DTLEMBAS_models + 1):
        load_timepoint_weights(tp, i)
        X_bias = mod.signaling_network.bias
        X_new = X_in_nodes.T
        for t in range(mod.signaling_network.training_params['max_steps']):
            X_old = X_new
            X_new = torch.mm(mod.signaling_network.weights, X_new) + X_bias
            X_new = mod.signaling_network.activation(X_new, mod.signaling_network.training_params['leak'])
            if (t % 10 == 0) and (t > 20) and torch.max(torch.abs(X_new - X_old)).lt(mod.signaling_network.training_params['tolerance']):
                break
        Yfull = X_new.T
        sum_full += Yfull
        sum_tf += mod.output_layer(Yfull)
    return sum_full / args.number_DTLEMBAS_models, sum_tf / args.number_DTLEMBAS_models

@torch.no_grad()
def ensemble_step_mean(tp: str, X_in_nodes: torch.Tensor):
    """The 'Direct' method using the high-level API call."""
    B, sum_full = X_in_nodes.size(0), torch.zeros_like(X_in_nodes)
    sum_tf = torch.zeros(B, num_tfs, device=device, dtype=torch.double)
    for i in range(1, args.number_DTLEMBAS_models + 1):
        load_timepoint_weights(tp, i)
        Yfull = mod.signaling_network(X_in_nodes)
        sum_full += Yfull
        sum_tf += mod.output_layer(Yfull)
    return sum_full / args.number_DTLEMBAS_models, sum_tf / args.number_DTLEMBAS_models

# --- Main Validation Logic ---
def main():
    print2log("Starting carry-on validation test...")
    start_time = time.time()

    # Load data and train PLSR model at the beginning ---
    print2log('Loading in vivo data for PLSR model...')
    Xh_df = pyreadr.read_r(args.X_invivo_path)[None]
    Yh_df = pyreadr.read_r(args.Y_invivo_path)[None]
    print2log('Training PLSR model...')
    plsr_model = PLSRegression(n_components=args.num_LVs, scale=False)
    plsr_model.fit(Xh_df, Yh_df)
    # Get the required TF order for the PLSR model ---
    plsr_tf_order = Xh_df.columns.tolist()
    # Create an index mapping from our model's TF order to the PLSR's TF order
    tf_reorder_indices = [list(all_tfs).index(tf) for tf in plsr_tf_order]

    tgfb_grid = torch.linspace(-args.stimulation_strength, args.stimulation_strength, args.grid_points, device=device, dtype=torch.double)
    ifna_grid = torch.linspace(-args.stimulation_strength, args.stimulation_strength, args.grid_points, device=device, dtype=torch.double)

    # Define the three test cases as per your old script
    test_cases = {
        "TGFb_plus_IFNa": (e_T.unsqueeze(0) * tgfb_grid.view(-1, 1, 1)) + (e_I.unsqueeze(0) * ifna_grid.view(-1, 1, 1)),
        "TGFb_only": e_T * tgfb_grid.view(-1, 1),
        "IFNa_only": e_I * ifna_grid.view(-1, 1)
    }

    all_validation_results = []
    all_plsr_results = []
    for test_name, initial_stim_batch in test_cases.items():
        print2log(f"--- Running Test Case: {test_name} ---")
        if initial_stim_batch.dim() == 3: initial_stim_batch = initial_stim_batch.squeeze(1)
        
        # --- A) Run the "Chained" simulation ---
        state_after_2h, _ = custom_ensemble_step_mean("2 hour", initial_stim_batch)
        state_after_8h, chained_tf_8h = custom_ensemble_step_mean("8 hour", state_after_2h.detach())
        chained_state_24h, chained_tf_24h = custom_ensemble_step_mean("24 hour", state_after_8h.detach())

        # --- B) Run the "Direct" simulation ---
        direct_state_8h, direct_tf_8h = ensemble_step_mean("8 hour", initial_stim_batch)
        direct_state_24h, direct_tf_24h = ensemble_step_mean("24 hour", initial_stim_batch)

        # --- C) Assemble results ---
        results_this_case = []
        for tp, chained_s, direct_s, chained_t, direct_t in [
            ('8 hour', state_after_8h, direct_state_8h, chained_tf_8h, direct_tf_8h),
            ('24 hour', chained_state_24h, direct_state_24h, chained_tf_24h, direct_tf_24h)
        ]:
            # Assemble Signaling States
            df_s_c = pd.DataFrame(chained_s.cpu().numpy(), columns=all_nodes).melt(var_name="variable_id", value_name="chained_activity")
            df_s_d = pd.DataFrame(direct_s.cpu().numpy(), columns=all_nodes).melt(var_name="variable_id", value_name="direct_activity")
            res_s = pd.concat([df_s_c, df_s_d['direct_activity']], axis=1); res_s['level'] = 'Signaling_State'
            # Assemble TF Activities
            df_t_c = pd.DataFrame(chained_t.cpu().numpy(), columns=all_tfs).melt(var_name="variable_id", value_name="chained_activity")
            df_t_d = pd.DataFrame(direct_t.cpu().numpy(), columns=all_tfs).melt(var_name="variable_id", value_name="direct_activity")
            res_t = pd.concat([df_t_c, df_t_d['direct_activity']], axis=1); res_t['level'] = 'TF_Activity'
            
            final_res = pd.concat([res_s, res_t], ignore_index=True); final_res['timepoint'] = tp
            results_this_case.append(final_res)
        
        case_df = pd.concat(results_this_case, ignore_index=True)
        case_df['test_case'] = test_name
        all_validation_results.append(case_df)

        ## Score the phenotype with PLSR
        chained_tf_8h_reordered = chained_tf_8h[:, tf_reorder_indices].cpu().numpy()
        chained_tf_8h_reordered[chained_tf_8h_reordered < 0] = 0
        chained_tf_8h_reordered[chained_tf_8h_reordered > 1] = 1
        chained_tf_8h_reordered = np.log((chained_tf_8h_reordered+1e-6)/(1 - chained_tf_8h_reordered + 1e-6))
        direct_tf_8h_reordered = direct_tf_8h[:, tf_reorder_indices].cpu().numpy()
        direct_tf_8h_reordered[direct_tf_8h_reordered < 0] = 0
        direct_tf_8h_reordered[direct_tf_8h_reordered > 1] = 1
        direct_tf_8h_reordered = np.log((direct_tf_8h_reordered+1e-6)/(1 - direct_tf_8h_reordered + 1e-6))
        chained_tf_24h_reordered = chained_tf_24h[:, tf_reorder_indices].cpu().numpy()
        chained_tf_24h_reordered[chained_tf_24h_reordered < 0] = 0
        chained_tf_24h_reordered[chained_tf_24h_reordered > 1] = 1
        chained_tf_24h_reordered = np.log((chained_tf_24h_reordered+1e-6)/(1 - chained_tf_24h_reordered + 1e-6))
        direct_tf_24h_reordered = direct_tf_24h[:, tf_reorder_indices].cpu().numpy()
        direct_tf_24h_reordered[direct_tf_24h_reordered < 0] = 0
        direct_tf_24h_reordered[direct_tf_24h_reordered > 1] = 1
        direct_tf_24h_reordered = np.log((direct_tf_24h_reordered+1e-6)/(1 - direct_tf_24h_reordered + 1e-6))
        chained_scores_8h = plsr_model.predict(chained_tf_8h_reordered)
        direct_scores_8h = plsr_model.predict(direct_tf_8h_reordered)
        chained_scores_24h = plsr_model.predict(chained_tf_24h_reordered)
        direct_scores_24h = plsr_model.predict(direct_tf_24h_reordered)
        plsr_res_8h = pd.DataFrame({'chained_MAS': chained_scores_8h[:, 0], 
                                    'direct_MAS': direct_scores_8h[:, 0], 
                                    'chained_Fibrosis': chained_scores_8h[:, 1], 
                                    'direct_Fibrosis': direct_scores_8h[:, 1], 
                                    'timepoint': '8 hour'})
        plsr_res_24h = pd.DataFrame({'chained_MAS': chained_scores_24h[:, 0], 
                                     'direct_MAS': direct_scores_24h[:, 0], 
                                     'chained_Fibrosis': chained_scores_24h[:, 1], 
                                     'direct_Fibrosis': direct_scores_24h[:, 1], 
                                     'timepoint': '24 hour'})
        plsr_res = pd.concat([plsr_res_8h, plsr_res_24h], ignore_index=True)
        plsr_res['test_case'] = test_name
        all_plsr_results.append(plsr_res)

    # --- D) Finalize and Save ---
    final_df = pd.concat(all_validation_results, ignore_index=True)
    final_df['delta'] = (final_df['chained_activity'] - final_df['direct_activity']).abs()

    # --- E) Combine PLSR results ---
    plsr_df = pd.concat(all_plsr_results, ignore_index=True)
    
    out_dir = os.path.join(res_dir, "validation_results")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = os.path.join(out_dir, 'carry_on_validation.csv')
    final_df.to_csv(out_csv, index=False)
    print2log(f"Saved detailed results to {out_csv}")
    
    create_plots(final_df, out_dir=out_dir)
    create_plsr_plot(plsr_df, out_dir=out_dir)
    elapsed = time.time() - start_time
    print2log(f"\nValidation finished in {elapsed/60:.1f} min.")

def create_plots(df: pd.DataFrame, out_dir: str):
    print2log("Generating plots...")
    for level in ['TF_Activity', 'Signaling_State']:
        level_df = df[df['level'] == level]
        if level == 'Signaling_State':
            stim_nodes = args.selected_stimulation_nodes
            level_df = level_df[~level_df['variable_id'].isin(stim_nodes)]
            plot_title_level = "Signaling State (Excl. Stimulated)"
        else:
            plot_title_level = "TF Activity"

        level_df_clean = level_df.dropna(subset=['direct_activity', 'chained_activity'])
        if len(level_df_clean) < 2: # Cannot compute correlation with less than 2 points
            print2log(f"  > Not enough data to plot for {level}. Skipping.")
            continue
        r, _ = pearsonr(level_df_clean['direct_activity'], level_df_clean['chained_activity'])
        mean_delta = level_df['delta'].mean(); max_delta = level_df['delta'].max()
        
        plt.figure(figsize=(8, 7))
        ax = sns.scatterplot(data=level_df, x="direct_activity", y="chained_activity", hue="timepoint", style="test_case", alpha=0.5, s=15, edgecolor=None)
        lims = [np.nanmin([ax.get_xlim(), ax.get_ylim()]), np.nanmax([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, linestyle="--", color="black", lw=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Direct Simulation Activity"); ax.set_ylabel("Chained Simulation Activity")
        ax.set_title(f"Chained vs. Direct: {plot_title_level}\nr={r:.4f}, mean|Δ|={mean_delta:.2e}, max|Δ|={max_delta:.2e}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"carry_on_scatter_{level}.png"), dpi=300)
        plt.close()

# --- Helper function for plotting PLSR results ---
def create_plsr_plot(df: pd.DataFrame, out_dir: str):
    """Generates a 2-panel scatter plot for PLSR predicted scores."""
    print2log("Generating plots for PLSR Phenotypes...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    phenotypes = ['MAS', 'Fibrosis']

    for i, pheno in enumerate(phenotypes):
        ax = axes[i]
        direct_col, chained_col = f'direct_{pheno}', f'chained_{pheno}'
        df_clean = df.dropna(subset=[direct_col, chained_col])
        r, p = pearsonr(df_clean[direct_col], df_clean[chained_col])
        
        sns.scatterplot(data=df, x=direct_col, y=chained_col, hue="timepoint", style="test_case", ax=ax, alpha=0.6, s=20)
        
        # --- MODIFIED: Set fixed axis limits based on phenotype ---
        if pheno == 'MAS':
            lims = [0, 8]
        else: # Fibrosis
            lims = [0, 4]

        ax.plot(lims, lims, linestyle="--", color="black", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"Direct Sim. - Predicted {pheno}")
        ax.set_ylabel(f"Chained Sim. - Predicted {pheno}")
        ax.set_title(f"Predicted {pheno} Score\nr={r:.4f}")
        
        if ax.get_legend() is not None:
           ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.75), loc='upper left')
    fig.suptitle("Chained vs. Direct Simulation: PLSR-Predicted Phenotypes")
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    plot_path = os.path.join(out_dir, "carry_on_scatter_phenotypes.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print2log(f"  > Saved PLSR plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()