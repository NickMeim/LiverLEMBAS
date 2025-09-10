import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_tf_heatmaps(df: pd.DataFrame, tf: str, out_dir: str):
    """
    Creates a faceted heatmap for a single TF, showing its activity across all experiments and timepoints.
    """
    print(f"Plotting heatmaps for {tf}...")
    
    tf_df = df[df["variable_id"] == tf]
    if tf_df.empty:
        print(f"  > Warning: No data found for TF {tf}. Skipping.")
        return

    # More robustly find the varying stimulation columns for each experiment
    stim_cols = ['T_at_0h', 'I_at_0h', 'T_at_2h', 'I_at_2h', 'T_at_8h', 'I_at_8h']
    exp_stim_map = {}
    for exp in tf_df['experiment'].unique():
        exp_df = tf_df[tf_df['experiment'] == exp]
        # An active column is one that has more than one unique value (i.e., it was varied across the grid)
        active_stim_cols = [col for col in stim_cols if exp_df[col].nunique() > 1]
        
        active_t_cols = [c for c in active_stim_cols if c.startswith('T_')]
        active_i_cols = [c for c in active_stim_cols if c.startswith('I_')]

        # We can only make a 2D heatmap if there is at least one varying TGFb and one varying IFNa stim across the whole experiment
        if active_t_cols and active_i_cols:
            exp_stim_map[exp] = (active_t_cols[0], active_i_cols[0])

    # Create the grid of subplots
    g = sns.FacetGrid(
        tf_df,
        col="experiment",
        row="timepoint",
        col_order=sorted(tf_df['experiment'].unique()),
        row_order=['2 hour', '8 hour', '24 hour'],
        height=3,
        aspect=1.1,
        margin_titles=True
    )

    # This function is applied to each subplot in the grid
    def draw_heatmap(*args, **kwargs):
        data_subset = kwargs.pop('data')
        exp = data_subset['experiment'].iloc[0]
        tp = data_subset['timepoint'].iloc[0]
        ax = plt.gca()

        # Determine the experiment's overall varying axes
        tgfb_ax, ifna_ax = exp_stim_map.get(exp, (None, None))

        # Check for variance within this specific facet's data
        tgfb_is_varying_now = tgfb_ax and data_subset[tgfb_ax].nunique() > 1
        ifna_is_varying_now = ifna_ax and data_subset[ifna_ax].nunique() > 1

        if tgfb_is_varying_now and ifna_is_varying_now:
            # --- CASE 1: Both stimuli are varying -> 2D Heatmap ---
            try:
                heatmap_data = data_subset.pivot_table(index=ifna_ax, columns=tgfb_ax, values='activity')
                heatmap_data = heatmap_data.sort_index(ascending=True).sort_index(axis=1, ascending=True)
                sns.heatmap(heatmap_data, ax=ax, cmap="viridis", **kwargs)
                ax.set_xlabel(f"TGFb ({tgfb_ax.split('_')[1]})")
                ax.set_ylabel(f"IFNa ({ifna_ax.split('_')[1]})")
            except Exception as e:
                ax.text(0.5, 0.5, 'Plotting Error', ha='center', va='center', color='red')
                print(f"  > Could not generate heatmap for {exp} at {tp}: {e}")

        elif tgfb_is_varying_now or ifna_is_varying_now:
            # --- CASE 2: Only one stimulus is varying -> 1D Line Plot ---
            varying_ax = tgfb_ax if tgfb_is_varying_now else ifna_ax
            stim_type = "TGFb" if tgfb_is_varying_now else "IFNa"
            
            sorted_data = data_subset.sort_values(varying_ax)
            sns.lineplot(data=sorted_data, x=varying_ax, y='activity', ax=ax, color='navy', marker='o', markersize=4)
            ax.set_xlabel(f"{stim_type} Stim. ({varying_ax.split('_')[1]})")
            ax.set_ylabel("Activity")
            ax.grid(True, linestyle='--', alpha=0.6)

        else:
            # --- CASE 3: Neither stimulus is varying -> "Carry Only" step ---
            activity_val = data_subset['activity'].mean()
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.5, f'Carry Only\n(Avg Activity: {activity_val:.3f})',
                    ha='center', va='center', fontsize=9, color='black')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')

    # Map the drawing function to the grid
    g.map_dataframe(draw_heatmap)
    
    # Finalize plot details
    g.fig.suptitle(f"Activity of TF: {tf}", y=1.03)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    # Remove the generic axis labels as they are now set per-subplot
    g.set_axis_labels("", "")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    out_path = os.path.join(out_dir, f"heatmap_summary_{tf}.png")
    plt.savefig(out_path, dpi=300)
    print(f"  > Saved plot to {out_path}")
    plt.close()

# --- Function to create the temporal heatmap ---
def plot_temporal_heatmaps(df: pd.DataFrame, tf: str, stim_strength: float, out_dir: str):
    """
    Creates a faceted heatmap for a single TF, showing its activity based on the TIMING of stimulation.
    This plot only uses the min (-3000) and max (+3000) stimulation strengths.
    """
    print(f"Plotting TEMPORAL heatmaps for {tf}...")
    
    # 1. Filter for the specific TF and the extreme stimulation strengths
    stim_cols = ['T_at_0h', 'I_at_0h', 'T_at_2h', 'I_at_2h', 'T_at_8h', 'I_at_8h']
    extrema = [-stim_strength, stim_strength]
    
    # Keep rows where at least one stimulation column is an extremum
    tf_df_full = df[df["variable_id"] == tf]
    filtered_df = tf_df_full[tf_df_full[stim_cols].isin([0] + extrema).all(axis=1)]
    
    if filtered_df.empty:
        print(f"  > Warning: No data found for TF {tf} at extreme stimulations. Skipping.")
        return

    # 2. Engineer features for WHEN each stimulus was applied
    def get_stim_time(row, stim_char):
        for t, col_name in zip(['0h', '2h', '8h'], [f'{stim_char}_at_0h', f'{stim_char}_at_2h', f'{stim_char}_at_8h']):
            if col_name in row and row[col_name] != 0:
                return t
        return 'None'
        
    temporal_df = filtered_df.copy()
    temporal_df['TGFb_Time'] = temporal_df.apply(lambda row: get_stim_time(row, 'T'), axis=1)
    temporal_df['IFNa_Time'] = temporal_df.apply(lambda row: get_stim_time(row, 'I'), axis=1)
    
    # Engineer features for the SIGN of each stimulus
    def get_stim_sign(row, stim_char):
        for col_name in [f'{stim_char}_at_0h', f'{stim_char}_at_2h', f'{stim_char}_at_8h']:
             if col_name in row and row[col_name] > 0: return '+'
             if col_name in row and row[col_name] < 0: return '-'
        return '0'

    temporal_df['TGFb_Sign'] = temporal_df.apply(lambda row: get_stim_sign(row, 'T'), axis=1)
    temporal_df['IFNa_Sign'] = temporal_df.apply(lambda row: get_stim_sign(row, 'I'), axis=1)
    temporal_df['Stim_Combo'] = "TGFb(" + temporal_df['TGFb_Sign'] + "), IFNa(" + temporal_df['IFNa_Sign'] + ")"
    
    # 3. Create the faceted plot
    time_order = ['None', '0h', '2h', '8h']
    combo_order = sorted(temporal_df['Stim_Combo'].unique())

    g = sns.FacetGrid(
        temporal_df,
        col="timepoint",
        row="Stim_Combo",
        row_order=combo_order,
        col_order=['2 hour', '8 hour', '24 hour'],
        height=2.5,
        aspect=1.1,
        margin_titles=True
    )

    # Drawing function for each facet
    def draw_temporal_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        ax = plt.gca()
        try:
            heatmap_data = data.pivot_table(index='IFNa_Time', columns='TGFb_Time', values='activity')
            # Reindex to ensure all timepoints are present and in order
            heatmap_data = heatmap_data.reindex(index=time_order, columns=time_order)
            sns.heatmap(heatmap_data, ax=ax, cmap="magma", annot=True, fmt=".2f", **kwargs)
        except Exception as e:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
    
    g.map_dataframe(draw_temporal_heatmap)
    g.fig.suptitle(f"Timing-Response Activity of TF: {tf}", y=1.03)
    g.set_titles(col_template="Measured at {col_name}", row_template="{row_name}")
    g.set_axis_labels("Time of TGFb Application", "Time of IFNa Application")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    out_path = os.path.join(out_dir, f"heatmap_temporal_summary_{tf}.png")
    plt.savefig(out_path, dpi=300)
    print(f"  > Saved temporal plot to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize results from asynchronous stimulation simulations.")
    parser.add_argument(
        '--results_csv',
        type=str,
        default='../results/chained_async/TFs_chained_async_avg_then_chain.csv',
        help="Path to the input CSV file containing simulation results."
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='../results/chained_async/plots/',
        help="Directory to save the output plots."
    )
    # --- NEW: Add stim strength to parser for filtering ---
    parser.add_argument('--stimulation_strength', action='store', type=float, default=3000.0)

    args = parser.parse_args()

    # --- Data Loading and Preparation ---
    print(f"Loading data from {args.results_csv}...")
    try:
        df = pd.read_csv(args.results_csv)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results_csv}")
        return

    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Melt the dataframe from wide to long format
    id_vars = [c for c in df.columns if not (c.startswith('P') or c.startswith('Q') or c.startswith('O'))]
    tf_vars = [c for c in df.columns if c not in id_vars]
    long_df = df.melt(id_vars=id_vars, value_vars=tf_vars, var_name="variable_id", value_name="activity")

    # --- Plotting ---
    tfs_to_plot = [
        'P10914', 'Q00978', 'P52630',
        'P42224', 'P84022', 'Q13485'
    ]

    for tf in tfs_to_plot:
        # Generate the original heatmap
        plot_tf_heatmaps(long_df, tf, args.out_dir)
        # Generate the new temporal heatmap
        plot_temporal_heatmaps(long_df, tf, args.stimulation_strength, args.out_dir)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()