import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
# Note: The 3D plot requires scipy. You likely have it, but if not:
# conda install -c conda-forge scipy
from scipy.interpolate import griddata
## PLSR imports
import pyreadr
from sklearn.cross_decomposition import PLSRegression

def plot_tf_visualizations(df: pd.DataFrame, tf: str, out_dir: str):
    """
    Creates a faceted plot for a single TF, choosing between a 2D heatmap,
    a 1D line plot, or text depending on the data for each experimental phase.
    """
    print(f"Generating dose-response visualizations for {tf}...")
    
    tf_df = df[df["variable_id"] == tf]
    if tf_df.empty:
        print(f"  > Warning: No data found for TF {tf}. Skipping.")
        return

    g = sns.FacetGrid(
        tf_df, col="experiment", row="measured_at",
        col_order=[f"Exp{i}" for i in range(1, 10)], row_order=['2h', '8h', '24h'],
        height=3, aspect=1.1, margin_titles=True
    )

    def smart_draw(*args, **kwargs):
        data_subset = kwargs.pop('data')
        ax = plt.gca()
        tgfb_is_varying = data_subset['TGFb_strength'].nunique() > 1
        ifna_is_varying = data_subset['IFNa_strength'].nunique() > 1

        if tgfb_is_varying and ifna_is_varying:
            try:
                heatmap_data = data_subset.pivot_table(index='IFNa_strength', columns='TGFb_strength', values='activity')
                heatmap_data = heatmap_data.sort_index(ascending=True).sort_index(axis=1, ascending=True)
                sns.heatmap(heatmap_data, ax=ax, cmap="viridis", **kwargs)
                ax.set_xlabel("TGFb Strength"); ax.set_ylabel("IFNa Strength")
            except Exception:
                ax.text(0.5, 0.5, 'Plot Error', ha='center', va='center', color='red')
        elif tgfb_is_varying or ifna_is_varying:
            varying_ax = 'TGFb_strength' if tgfb_is_varying else 'IFNa_strength'
            stim_type = "TGFb" if tgfb_is_varying else "IFNa"
            sorted_data = data_subset.sort_values(varying_ax)
            sns.lineplot(data=sorted_data, x=varying_ax, y='activity', ax=ax, color='navy', marker='o', markersize=4)
            ax.set_xlabel(f"{stim_type} Strength"); ax.set_ylabel("Activity")
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            activity_val = data_subset['activity'].mean()
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, f'Carry-Only\n(Avg: {activity_val:.3f})',
                    ha='center', va='center', fontsize=9, color='black', wrap=True)
            ax.set_xticks([]); ax.set_yticks([])

    g.map_dataframe(smart_draw)
    g.fig.suptitle(f"Dose-Response Activity of TF: {tf}", y=1.03, fontsize=16)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("", "")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, f"summary_plot_dose_response_{tf}.png")
    plt.savefig(out_path, dpi=300)
    print(f"  > Saved dose-response plot to {out_path}")
    plt.close()

def plot_temporal_3d_surface(df: pd.DataFrame, tf: str, stim_strength: float, out_dir: str):
    """
    Creates a faceted 3D surface plot for a single TF based on the TIMING of stimulation
    at the four extreme (+/-) stimulus strength combinations.
    """
    print(f"Generating 3D temporal-response visualizations for {tf}...")
    
    extrema = [-stim_strength, stim_strength]
    surface_df = df[(df["variable_id"] == tf) & (df["TGFb_strength"].isin(extrema)) & (df["IFNa_strength"].isin(extrema))].copy()
    
    if surface_df.empty:
        print(f"  > Warning: No data for 3D plot for TF {tf}. Skipping."); return

    time_map = {'0h': 0, '2h': 2, '8h': 8, 'None': np.nan} 
    surface_df['TGFb_Time_Num'] = surface_df['TGFb_applied_at'].map(time_map)
    surface_df['IFNa_Time_Num'] = surface_df['IFNa_applied_at'].map(time_map)
    
    def get_combo_label(row):
        t_sign = '+' if row['TGFb_strength'] > 0 else '-'
        i_sign = '+' if row['IFNa_strength'] > 0 else '-'
        return f"T({t_sign}), I({i_sign})"
    surface_df['Stim_Combo'] = surface_df.apply(get_combo_label, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), subplot_kw=dict(projection='3d'))
    
    measurement_times = ['2h', '8h', '24h']
    z_min, z_max = surface_df['activity'].min(), surface_df['activity'].max()
    
    combo_order = sorted(surface_df['Stim_Combo'].unique())
    color_palette = sns.color_palette("tab10", n_colors=len(combo_order))
    color_map = {combo: color for combo, color in zip(combo_order, color_palette)}
    legend_patches = [mpatches.Patch(color=color, label=combo) for combo, color in color_map.items()]
    
    for i, tp in enumerate(measurement_times):
        ax = axes[i]
        ax.set_title(f"Measured at {tp}", y=1.0, pad=-15)
        
        for combo, color in color_map.items():
            plot_data = surface_df[(surface_df['measured_at'] == tp) & (surface_df['Stim_Combo'] == combo)].dropna(subset=['TGFb_Time_Num', 'IFNa_Time_Num'])
            if len(plot_data) < 4: continue
            
            try:
                X, Y = np.meshgrid(np.linspace(plot_data.TGFb_Time_Num.min(), plot_data.TGFb_Time_Num.max(), 10), np.linspace(plot_data.IFNa_Time_Num.min(), plot_data.IFNa_Time_Num.max(), 10))
                Z = griddata((plot_data.TGFb_Time_Num, plot_data.IFNa_Time_Num), plot_data.activity, (X, Y), method='cubic')
                ax.plot_surface(X, Y, Z, color=color, alpha=0.6, rstride=1, cstride=1, antialiased=True, shade=True)
                ax.scatter(plot_data.TGFb_Time_Num, plot_data.IFNa_Time_Num, plot_data.activity, color=color, s=25, edgecolor='black', depthshade=True)
            except Exception as e:
                print(f"  > Could not generate 3D surface for {tp}, combo {combo}: {e}")

        ax.set_xlabel("\nTGFb App. Time (h)", linespacing=2.5); ax.set_ylabel("\nIFNa App. Time (h)", linespacing=2.5)
        ax.set_zlabel("TF Activity", labelpad=10); ax.set_zlim(z_min, z_max)
        ax.view_init(elev=20, azim=240)

    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, title=f"TF: {tf} - Stimulation Combination")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = os.path.join(out_dir, f"summary_plot_3D_surface_{tf}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved 3D surface plot to {out_path}")
    plt.close()

# --- Function to create the 3D surface plots for phenotypes ---
def plot_phenotype_3d_surface(df: pd.DataFrame, phenotype: str, stim_strength: float, out_dir: str):
    """
    Creates a faceted 3D surface plot for a single PHENOTYPE, showing its score based on the TIMING of stimulation
    at the four extreme (+/-) stimulus strength combinations.
    """
    print(f"Generating 3D temporal-response visualizations for {phenotype}...")
    
    # 1. Filter data for the four corner stimulus strengths
    extrema = [-stim_strength, stim_strength]
    surface_df = df[
        (df["TGFb_strength"].isin(extrema)) &
        (df["IFNa_strength"].isin(extrema))
    ].copy()
    
    if surface_df.empty:
        print(f"  > Warning: No data for 3D plot for {phenotype}. Skipping."); return

    # 2. Prepare data for plotting
    time_map = {'0h': 0, '2h': 2, '8h': 8, 'None': np.nan} 
    surface_df['TGFb_Time_Num'] = surface_df['TGFb_applied_at'].map(time_map)
    surface_df['IFNa_Time_Num'] = surface_df['IFNa_applied_at'].map(time_map)
    
    def get_combo_label(row):
        t_sign = '+' if row['TGFb_strength'] > 0 else '-'
        i_sign = '+' if row['IFNa_strength'] > 0 else '-'
        return f"T({t_sign}), I({i_sign})"
    surface_df['Stim_Combo'] = surface_df.apply(get_combo_label, axis=1)

    # 3. Manually create the faceted 3D plot
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), subplot_kw=dict(projection='3d'))
    
    measurement_times = ['2h', '8h', '24h']
    z_min, z_max = surface_df[phenotype].min(), surface_df[phenotype].max()
    
    combo_order = sorted(surface_df['Stim_Combo'].unique())
    color_palette = sns.color_palette("tab10", n_colors=len(combo_order))
    color_map = {combo: color for combo, color in zip(combo_order, color_palette)}
    legend_patches = [mpatches.Patch(color=color, label=combo) for combo, color in color_map.items()]
    
    for i, tp in enumerate(measurement_times):
        ax = axes[i]
        ax.set_title(f"Measured at {tp}", y=1.0, pad=-15)
        
        for combo, color in color_map.items():
            plot_data = surface_df[(surface_df['measured_at'] == tp) & (surface_df['Stim_Combo'] == combo)].dropna(subset=['TGFb_Time_Num', 'IFNa_Time_Num'])
            if len(plot_data) < 4: continue
            
            try:
                X, Y = np.meshgrid(np.linspace(plot_data.TGFb_Time_Num.min(), plot_data.TGFb_Time_Num.max(), 10), np.linspace(plot_data.IFNa_Time_Num.min(), plot_data.IFNa_Time_Num.max(), 10))
                Z = griddata((plot_data.TGFb_Time_Num, plot_data.IFNa_Time_Num), plot_data[phenotype], (X, Y), method='cubic')
                ax.plot_surface(X, Y, Z, color=color, alpha=0.6, rstride=1, cstride=1, antialiased=True, shade=True)
                ax.scatter(plot_data.TGFb_Time_Num, plot_data.IFNa_Time_Num, plot_data[phenotype], color=color, s=25, edgecolor='black', depthshade=True)
            except Exception as e:
                print(f"  > Could not generate 3D surface for {tp}, combo {combo}: {e}")

        ax.set_xlabel("\nTGFb App. Time (h)", linespacing=2.5); ax.set_ylabel("\nIFNa App. Time (h)", linespacing=2.5)
        ax.set_zlabel(f"Predicted {phenotype} Score", labelpad=10); ax.set_zlim(z_min, z_max)
        ax.view_init(elev=20, azim=240)

    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, title=f"{phenotype} Score - Stimulation Combination")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    out_path = os.path.join(out_dir, f"summary_plot_3D_surface_{phenotype}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"  > Saved 3D surface plot to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize results from asynchronous stimulation simulations.")
    parser.add_argument(
        '--results_csv',
        type=str,
        default='../results/async_experiments_tidy/async_stimulation_results_no_carry_on.csv',
        help="Path to the tidy CSV file from the new AsynchronousStimulation.py."
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='../results/async_experiments_tidy/plots/',
        help="Directory to save the output plots."
    )
    parser.add_argument('--stimulation_strength', action='store', type=float, default=3000.0)
    parser.add_argument('--X_invivo_path', action='store', help='Path to in vivo TF activities (.rds)', default='../data/Xh_tfs_govaere.rds')
    parser.add_argument('--Y_invivo_path', action='store', help='Path to in vivo phenotypes (.rds)', default='../data/Yh_tfs_govaere.rds')
    parser.add_argument('--num_LVs', action='store', type=int, help='Number of latent variables for PLSR', default=8)
    args = parser.parse_args()

    # --- Data Loading and Preparation ---
    print(f"Loading data from {args.results_csv}...")
    try:
        df = pd.read_csv(args.results_csv)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results_csv}")
        return

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("--- PLSR Phenotype Prediction ---")
    print('Loading in vivo data for PLSR model...')
    Xh_df = pyreadr.read_r(args.X_invivo_path)[None]
    Yh_df = pyreadr.read_r(args.Y_invivo_path)[None]
    print('Training PLSR model...')
    plsr_model = PLSRegression(n_components=args.num_LVs, scale=False)
    plsr_model.fit(Xh_df, Yh_df)
    
    id_vars = ['experiment', 'TGFb_strength', 'IFNa_strength', 'TGFb_applied_at', 'IFNa_applied_at', 'measured_at']
    tf_activities_wide = df.set_index(id_vars)[[c for c in df.columns if c not in id_vars]]

    plsr_tf_order = Xh_df.columns.tolist()
    for tf in plsr_tf_order:
        if tf not in tf_activities_wide.columns:
            tf_activities_wide[tf] = 0.0
    tf_activities_for_pred = tf_activities_wide[plsr_tf_order]
    
    print("Transforming all simulated TF activities...")
    X_sim = tf_activities_for_pred.copy()
    X_sim[X_sim > 1] = 1; X_sim[X_sim < 0] = 0
    epsilon = 1e-6
    X_sim_transformed = np.log((X_sim + epsilon) / (1 - X_sim + epsilon))
    
    print("Predicting phenotypes (MAS and Fibrosis) for all experiments...")
    predicted_scores = plsr_model.predict(X_sim_transformed)
    phenotype_df = pd.DataFrame(predicted_scores, index=X_sim_transformed.index, columns=['MAS', 'Fibrosis'])
    
    # Merge phenotype scores back into the main dataframe
    full_results_df = df.merge(phenotype_df.reset_index(), on=id_vars)

    # --- TF Visualizations (run on the full dataset) ---
    tf_vars_original = [c for c in df.columns if c not in id_vars]
    long_df = df.melt(id_vars=id_vars, value_vars=tf_vars_original, var_name="variable_id", value_name="activity")

    tfs_to_plot = ['P10914', 'Q00978', 'P52630', 'P42224', 'P84022', 'Q13485']
    for tf in tfs_to_plot:
        plot_tf_visualizations(long_df, tf, args.out_dir)
        plot_temporal_3d_surface(long_df, tf, args.stimulation_strength, args.out_dir)
    
    # --- Call the 3D phenotype plotting function for the full dataset ---
    plot_phenotype_3d_surface(full_results_df, 'MAS', args.stimulation_strength, args.out_dir)
    plot_phenotype_3d_surface(full_results_df, 'Fibrosis', args.stimulation_strength, args.out_dir)

    print("\nVisualization complete.")

if __name__ == "__main__":
    main()

