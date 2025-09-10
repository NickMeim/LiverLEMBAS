import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
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
import pyreadr
from sklearn.cross_decomposition import PLSRegression

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

class PLSRTransform(torch.nn.Module):
    def __init__(self, X_coefs, X_mean, dtype=torch.double,device="cuda"):
        super(PLSRTransform, self).__init__()
        # Convert numpy arrays to torch tensors
        self.X_coefs = torch.tensor(X_coefs, dtype=dtype).to(device)
        self.X_mean = torch.tensor(X_mean, dtype=dtype).to(device)
    def forward(self, X):
        # Center the input X
        X_centered = X - self.X_mean
        # Project X onto the latent space (scores)
        Y_pred = torch.matmul(X_centered, self.X_coefs)
        return Y_pred

# --- Use necessary arguments from the main script ---
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
parser.add_argument('--LEMBAS_tolerance', action='store', default=1e-5)
parser.add_argument('--stimulation_strength', action='store', type=float, default=3000.0)
parser.add_argument('--grid_points', type=int, default=11)
parser.add_argument('--selected_stimulation_nodes', metavar='N', type=str, nargs='*', default=['P36897', 'P17181'])
parser.add_argument('--number_DTLEMBAS_models', action='store', type=int, default=50)
## load the human data too
parser.add_argument('--X_invivo_path', action='store', help='file to load pre-processed in vivo molecular features (rds)', default='../data/Xh_tfs_govaere.rds')
parser.add_argument('--Y_invivo_path', action='store', help='file to load in vivo phenotypes (rds)', default='../data/Yh_tfs_govaere.rds')
parser.add_argument('--num_LVs', action='store', type=int, help='number of latent variables in the PLSR model', default=8)
args = parser.parse_args()

# --- Basic Setup  ---
TIME_ORDER = ["2 hour", "8 hour", "24 hour"]
res_dir = args.res_dir
TIME_DIRS = {
    "2 hour":  os.path.join(res_dir, "2 hour",  "allmodels"),
    "8 hour":  os.path.join(res_dir,"allmodels"),
    "24 hour": os.path.join(res_dir, "24 hour", "allmodels"),
}
utils.set_cores(12)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log(f'Running on {device}')

# --- Data and Model Loading  ---
net = pd.read_csv(args.PKN, sep='\t', usecols=['source', 'target', 'stimulation', 'inhibition'])
drugTargets = pd.read_csv(args.TargetsIn, sep='\t', index_col=0)
TFOutput = pd.read_csv(args.TFsOut, sep='\t', index_col=0)
net = format_network(net, weight_label='weight', stimulation_label='stimulation', inhibition_label='inhibition')
nodeNames = np.array(list(set(net['source']).union(set(net['target']))))
inName = np.intersect1d(nodeNames, drugTargets.columns.values)
outName = np.intersect1d(nodeNames, TFOutput.columns.values)
TFOutput = TFOutput.loc[:, outName]
drugTargets = drugTargets.loc[:, inName]
bionet_params = {'max_steps': args.max_LEMBAS_iterations, 'leak': args.LEMBAS_leak, 'tolerance': args.LEMBAS_tolerance}
# labels used throughout
weight_label = 'weight'
source_label = 'source'
target_label = 'target'
mod = SignalingModel(
    net=net,
    X_in=drugTargets,
    y_out=TFOutput,
    projection_amplitude_in=args.inputAmplitude,
    projection_amplitude_out=args.projectionAmplitude,
    bionet_params=bionet_params,
    weight_label=weight_label,
    source_label=source_label,
    target_label=target_label,
    dtype=torch.double,
    device=device,
).to(device)

all_tfs = mod.y_out.columns.values
all_nodes = [k for k, v in sorted(mod.node_idx_map.items(), key=lambda item: item[1])]
num_nodes, num_tfs = len(mod.node_idx_map), mod.y_out.shape[1]
tgfb_name, ifna_name = args.selected_stimulation_nodes[:2]
idx_tgfb = int(mod.node_idx_map[tgfb_name])
idx_ifna = int(mod.node_idx_map[ifna_name])
Xin_eye = torch.eye(num_nodes, dtype=torch.double, device=device)
e_T = Xin_eye[idx_tgfb, :].unsqueeze(0)
e_I = Xin_eye[idx_ifna, :].unsqueeze(0)

# --- Ensemble Averaging Function  ---
def load_timepoint_weights(tp: str, i_model: int):
    sig_path = os.path.join(TIME_DIRS[tp], f"{args.modelPattern}{i_model}signal_state_dict_{i_model}.pth")
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

@torch.no_grad()
def ensemble_step_mean(tp: str, X_in_nodes: torch.Tensor,number_DTLEMBAS_models: int):
    B = X_in_nodes.size(0)
    sum_full = torch.zeros_like(X_in_nodes)
    sum_tf = torch.zeros(B, num_tfs, device=device, dtype=torch.double)
    for i in range(1, number_DTLEMBAS_models + 1):
        load_timepoint_weights(tp, i)
        Yfull = mod.signaling_network(X_in_nodes)
        TF = mod.output_layer(Yfull)
        sum_full += Yfull
        sum_tf += TF
    return sum_full / number_DTLEMBAS_models, sum_tf / number_DTLEMBAS_models

# -------------------------------------------------------------------
# Main Validation Logic
# -------------------------------------------------------------------
def main():
    print2log("Starting carry-on validation test...")
    start_time = time.time()

    # 0. Load data and train PLSR model ---
    print2log('Loading in vivo data for PLSR model...')
    Xh_data = pyreadr.read_r(args.X_invivo_path)
    Xh_df = Xh_data[None]
    Xh = Xh_df.to_numpy()
    Yh = pyreadr.read_r(args.Y_invivo_path)
    Yh_df = Yh[None]
    Yh = Yh_df.to_numpy()
    print2log('Training PLSR model...')
    plsr_model = PLSRegression(n_components=args.num_LVs, scale=False)
    plsr_model.fit(Xh, Yh)
    plsr_transform = PLSRTransform(plsr_model.coef_.T, plsr_model._x_mean, device=device).to(device)
    plsr_transform.eval()
    # Get the required TF order for the PLSR model ---
    plsr_tf_order = Xh_df.columns.tolist()
    # Create an index mapping from our model's TF order to the PLSR's TF order
    tf_reorder_indices = [list(all_tfs).index(tf) for tf in plsr_tf_order]

    # 1. Define stimulation grids
    tgfb_grid = torch.linspace(-args.stimulation_strength, args.stimulation_strength, args.grid_points, device=device, dtype=torch.double)
    ifna_grid = torch.linspace(-args.stimulation_strength, args.stimulation_strength, args.grid_points, device=device, dtype=torch.double)

    # 2. Define the three test cases
    test_cases = {
        "TGFb_plus_IFNa": (e_T.unsqueeze(0) * tgfb_grid.view(-1, 1, 1)) + (e_I.unsqueeze(0) * ifna_grid.view(-1, 1, 1)),
        "TGFb_only": e_T * tgfb_grid.view(-1, 1),
        "IFNa_only": e_I * ifna_grid.view(-1, 1)
    }

    all_validation_results = []
    all_plsr_results = []
    for test_name, initial_stim_batch in test_cases.items():
        print2log(f"--- Running Test Case: {test_name} ---")
        if initial_stim_batch.dim() == 3:
            initial_stim_batch = initial_stim_batch.squeeze(1)
        
        # --- A) Run the "Chained" simulation ---
        state_after_2h, _ = custom_ensemble_step_mean("2 hour", initial_stim_batch, args.number_DTLEMBAS_models)
        state_after_8h, chained_tf_8h = custom_ensemble_step_mean("8 hour", state_after_2h.detach(), args.number_DTLEMBAS_models)
        chained_state_24h, chained_tf_24h = custom_ensemble_step_mean("24 hour", state_after_8h.detach(), args.number_DTLEMBAS_models)

        # --- B) Run the "Direct" simulation ---
        direct_state_8h, direct_tf_8h = ensemble_step_mean("8 hour", initial_stim_batch, args.number_DTLEMBAS_models)
        direct_state_24h, direct_tf_24h = ensemble_step_mean("24 hour", initial_stim_batch, args.number_DTLEMBAS_models)

        # --- C) Assemble results into a DataFrame for TF Activities---
        df_chained_8h = pd.DataFrame(chained_tf_8h.cpu().numpy(), columns=all_tfs).melt(var_name="variable_id", value_name="chained_activity")
        df_direct_8h = pd.DataFrame(direct_tf_8h.cpu().numpy(), columns=all_tfs).melt(var_name="variable_id", value_name="direct_activity")
        df_chained_24h = pd.DataFrame(chained_tf_24h.cpu().numpy(), columns=all_tfs).melt(var_name="variable_id", value_name="chained_activity")
        df_direct_24h = pd.DataFrame(direct_tf_24h.cpu().numpy(), columns=all_tfs).melt(var_name="variable_id", value_name="direct_activity")

        if test_name == "TGFb_plus_IFNa":
            params = pd.DataFrame({'tgfb_strength': tgfb_grid.cpu().numpy(), 'ifna_strength': ifna_grid.cpu().numpy()})
        elif test_name == "TGFb_only":
            params = pd.DataFrame({'tgfb_strength': tgfb_grid.cpu().numpy(), 'ifna_strength': 0.0})
        else:
            params = pd.DataFrame({'tgfb_strength': 0.0, 'ifna_strength': ifna_grid.cpu().numpy()})
        
        params_long_tf = pd.concat([params] * num_tfs, ignore_index=True)
        res_8h_tf = pd.concat([params_long_tf, df_chained_8h, df_direct_8h['direct_activity']], axis=1)
        res_8h_tf['timepoint'] = '8 hour'
        res_24h_tf = pd.concat([params_long_tf, df_chained_24h, df_direct_24h['direct_activity']], axis=1)
        res_24h_tf['timepoint'] = '24 hour'
        final_res_tf = pd.concat([res_8h_tf, res_24h_tf], ignore_index=True)
        final_res_tf['level'] = 'TF_Activity'

        # --- D) Assemble results for Signaling States ---
        df_chained_state_8h = pd.DataFrame(state_after_8h.cpu().numpy(), columns=all_nodes).melt(var_name="variable_id", value_name="chained_activity")
        df_direct_state_8h = pd.DataFrame(direct_state_8h.cpu().numpy(), columns=all_nodes).melt(var_name="variable_id", value_name="direct_activity")
        df_chained_state_24h = pd.DataFrame(chained_state_24h.cpu().numpy(), columns=all_nodes).melt(var_name="variable_id", value_name="chained_activity")
        df_direct_state_24h = pd.DataFrame(direct_state_24h.cpu().numpy(), columns=all_nodes).melt(var_name="variable_id", value_name="direct_activity")

        params_long_nodes = pd.concat([params] * num_nodes, ignore_index=True)
        res_8h_nodes = pd.concat([params_long_nodes, df_chained_state_8h, df_direct_state_8h['direct_activity']], axis=1)
        res_8h_nodes['timepoint'] = '8 hour'
        res_24h_nodes = pd.concat([params_long_nodes, df_chained_state_24h, df_direct_state_24h['direct_activity']], axis=1)
        res_24h_nodes['timepoint'] = '24 hour'
        final_res_nodes = pd.concat([res_8h_nodes, res_24h_nodes], ignore_index=True)
        final_res_nodes['level'] = 'Signaling_State'
        
        # Combine both levels of results
        final_res = pd.concat([final_res_tf, final_res_nodes], ignore_index=True)
        final_res['test_case'] = test_name
        final_res['delta'] = (final_res['chained_activity'] - final_res['direct_activity']).abs()
        all_validation_results.append(final_res)

        # --- E) Run PLSR model ---
        # Predict phenotypes with PLSR model ---
        chained_tf_8h_np = chained_tf_8h[:, tf_reorder_indices].cpu().numpy()
        direct_tf_8h_np = direct_tf_8h[:, tf_reorder_indices].cpu().numpy()
        chained_tf_24h_np = chained_tf_24h[:, tf_reorder_indices].cpu().numpy()
        direct_tf_24h_np = direct_tf_24h[:, tf_reorder_indices].cpu().numpy()
        chained_scores_8h = plsr_model.predict(chained_tf_8h_np)
        direct_scores_8h = plsr_model.predict(direct_tf_8h_np)
        chained_scores_24h = plsr_model.predict(chained_tf_24h_np)
        direct_scores_24h = plsr_model.predict(direct_tf_24h_np)
        plsr_res_8h = pd.DataFrame({
            'chained_MAS': chained_scores_8h[:, 0], 'direct_MAS': direct_scores_8h[:, 0],
            'chained_Fibrosis': chained_scores_8h[:, 1], 'direct_Fibrosis': direct_scores_8h[:, 1],
            'timepoint': '8 hour'
        })
        plsr_res_24h = pd.DataFrame({
            'chained_MAS': chained_scores_24h[:, 0], 'direct_MAS': direct_scores_24h[:, 0],
            'chained_Fibrosis': chained_scores_24h[:, 1], 'direct_Fibrosis': direct_scores_24h[:, 1],
            'timepoint': '24 hour'
        })
        # This part has a small bug, needs to combine params with each timepoint df separately
        params_8h = pd.concat([params.reset_index(drop=True), plsr_res_8h.reset_index(drop=True)], axis=1)
        params_24h = pd.concat([params.reset_index(drop=True), plsr_res_24h.reset_index(drop=True)], axis=1)
        plsr_res = pd.concat([params_8h, params_24h], ignore_index=True)
        plsr_res['test_case'] = test_name
        all_plsr_results.append(plsr_res)

        # Print summary for this test case
        tf_results = final_res[final_res.level == 'TF_Activity']
        node_results = final_res[final_res.level == 'Signaling_State']

        mean_delta_tf_8h = tf_results[tf_results.timepoint == '8 hour']['delta'].mean()
        max_delta_tf_8h = tf_results[tf_results.timepoint == '8 hour']['delta'].max()
        mean_delta_tf_24h = tf_results[tf_results.timepoint == '24 hour']['delta'].mean()
        max_delta_tf_24h = tf_results[tf_results.timepoint == '24 hour']['delta'].max()
        
        mean_delta_node_8h = node_results[node_results.timepoint == '8 hour']['delta'].mean()
        max_delta_node_8h = node_results[node_results.timepoint == '8 hour']['delta'].max()
        mean_delta_node_24h = node_results[node_results.timepoint == '24 hour']['delta'].mean()
        max_delta_node_24h = node_results[node_results.timepoint == '24 hour']['delta'].max()

        print(f"  > 8h TFs:      Mean |Δ| = {mean_delta_tf_8h:.4e}, Max |Δ| = {max_delta_tf_8h:.4e}")
        print(f"  > 8h States:   Mean |Δ| = {mean_delta_node_8h:.4e}, Max |Δ| = {max_delta_node_8h:.4e}")
        print(f"  > 24h TFs:     Mean |Δ| = {mean_delta_tf_24h:.4e}, Max |Δ| = {max_delta_tf_24h:.4e}")
        print(f"  > 24h States:  Mean |Δ| = {mean_delta_node_24h:.4e}, Max |Δ| = {max_delta_node_24h:.4e}")


    # 3. Finalize and save all results
    final_df = pd.concat(all_validation_results, ignore_index=True)
    plsr_df = pd.concat(all_plsr_results, ignore_index=True)
    out_dir = os.path.join(res_dir, "chained_async")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = os.path.join(out_dir, 'carry_on_validation.csv')
    final_df.to_csv(out_csv, index=False)

    # --- Generate plots ---
    # Plot 1: All Signaling States
    create_plots(df=final_df[final_df.level == 'Signaling_State'], level_name="Signaling_States_All", out_dir=out_dir)
    # Plot 2: All TF Activities
    create_plots(df=final_df[final_df.level == 'TF_Activity'], level_name="TF_Activities_All", out_dir=out_dir)

    # --- Generate the 3 additional plots ---
    # Plot 3: Signaling states excluding stimulated nodes
    stim_nodes = args.selected_stimulation_nodes
    df_states_filtered = final_df[(final_df.level == 'Signaling_State') & (~final_df.variable_id.isin(stim_nodes))]
    create_plots(df=df_states_filtered, level_name="Signaling_States_Excl_Stim", out_dir=out_dir)

    # Plot 4: Subset of TFs
    tf_subset = ['P10914','Q00978','P52630','P42224','P84022','Q13485']
    df_tf_filtered = final_df[(final_df.level == 'TF_Activity') & (final_df.variable_id.isin(tf_subset))]
    create_plots(df=df_tf_filtered, level_name="TF_Activities_Subset", out_dir=out_dir)

    # Plot 5: PLSR-predicted Phenotypes
    create_plsr_plot(df=plsr_df, out_dir=out_dir)

    elapsed = time.time() - start_time
    print2log(f"\nSaved detailed results to {out_csv}")
    print2log(f"Validation finished in {elapsed/60:.1f} min.")

# --- Helper function for plotting ---
def create_plots(df: pd.DataFrame, level_name: str, out_dir: str):
    print2log(f"Generating plots for {level_name}...")
    df_clean = df.dropna(subset=['direct_activity', 'chained_activity'])
    r, p = pearsonr(df_clean['direct_activity'], df_clean['chained_activity'])
    mean_abs_delta = df['delta'].mean()
    max_abs_delta = df['delta'].max()
    plt.figure(figsize=(8, 7)); ax = sns.scatterplot(data=df, x="direct_activity", y="chained_activity", hue="timepoint", style="test_case", alpha=0.5, s=15, edgecolor=None)
    lims = [np.nanmin([ax.get_xlim(), ax.get_ylim()]), np.nanmax([ax.get_xlim(), ax.get_ylim()])]; ax.plot(lims, lims, linestyle="--", color="black", lw=1)
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_xlabel("Direct Simulation Activity"); ax.set_ylabel("Chained Simulation Activity")
    ax.set_title(f"Chained vs. Direct Simulation: {level_name}\nr={r:.4f}, mean|Δ|={mean_abs_delta:.2e}, max|Δ|={max_abs_delta:.2e}")
    plt.tight_layout(); scatter_path = os.path.join(out_dir, f"carry_on_scatter_{level_name.lower()}.png"); plt.savefig(scatter_path, dpi=300)
    print2log(f"  > Saved scatter plot to {scatter_path}"); plt.close()
    absd = df['delta'].dropna().to_numpy(); absd_pos = absd + 1e-18
    plt.figure(figsize=(7, 5)); sns.histplot(absd_pos, bins=60, kde=True, log_scale=(True, False))
    stats = {"mean": absd.mean(), "median": np.median(absd), "95th": np.percentile(absd, 95), "99th": np.percentile(absd, 99), "max": absd.max()}
    for name, val in stats.items():
        plt.axvline(val, ls="--", lw=1, label=f"{name}={val:.2e}")
    plt.xlabel("|Δ| (Absolute Difference)"); plt.ylabel("Count"); plt.title(f"Distribution of |Δ| for {level_name}"); plt.legend(); plt.tight_layout()
    hist_path = os.path.join(out_dir, f"carry_on_histogram_{level_name.lower()}.png"); plt.savefig(hist_path, dpi=300)
    print2log(f"  > Saved histogram to {hist_path}"); plt.close()


# --- Helper function for plotting PLSR results ---
def create_plsr_plot(df: pd.DataFrame, out_dir: str):
    """Generates a 2-panel scatter plot for PLSR predicted scores."""
    print2log("Generating plots for PLSR Phenotypes...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    phenotypes = ['MAS', 'Fibrosis']

    for i, pheno in enumerate(phenotypes):
        ax = axes[i]
        direct_col, chained_col = f'direct_{pheno}', f'chained_{pheno}'
        df_clean = df.dropna(subset=[direct_col, chained_col])
        r, p = pearsonr(df_clean[direct_col], df_clean[chained_col])
        
        sns.scatterplot(data=df, x=direct_col, y=chained_col, hue="timepoint", style="test_case", ax=ax, alpha=0.6, s=20)
        
        # --- FIX: Set fixed axis limits based on phenotype ---
        if pheno == 'MAS':
            lims = [0, 8]
        else:  # Fibrosis
            lims = [0, 4]

        ax.plot(lims, lims, linestyle="--", color="black", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"Direct Sim. - Predicted {pheno}")
        ax.set_ylabel(f"Chained Sim. - Predicted {pheno}")
        ax.set_title(f"Predicted {pheno} Score\nr={r:.4f}")
        
        # Hide the legend on the individual subplot to prepare for a single figure legend
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # --- FIX: Ensure the figure legend is created and visible ---
    handles, labels = axes[0].get_legend_handles_labels()
    # Place legend outside the plot area
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.8), loc='upper left', borderaxespad=0.)
    
    fig.suptitle("Chained vs. Direct Simulation: PLSR-Predicted Phenotypes")
    
    # Adjust layout to prevent overlap and make space for the legend
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    plot_path = os.path.join(out_dir, "carry_on_scatter_phenotypes.png")
    # Use bbox_inches='tight' to ensure the legend is included in the saved file
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print2log(f"  > Saved PLSR plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()