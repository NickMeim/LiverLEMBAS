import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt

def compare_target_vs_reference(df,target_name, reference_name, feature_col='TF',
                                features2show = ['P10914','Q00978','P42226','P42229','P51692','P40763','P52630','P42224','Q14765','P84022','P01106','P05412',
                                                 'O00716','Q14209','Q01094','Q16254','P04637'], 
                                model_col='set', r_col='r', folds_col='fold',alpha = 0.4):
    """
    Compares Pearson correlations across folds for shuffled and actual models, adjusts p-values,
    and plots a scatterplot.
    Parameters:
        df (pd.DataFrame): DataFrame containing columns for gene, set (e.g. shuffled/train/test), r values, and folds.
        features2show (np.ndarray): 1D array of feature names to highlight on the scatterplot.
        feature_col (str): Column name for the feature.
        model_col (str): Column name for the model set.
        r_col (str): Column name for Pearson correlation values.
        folds_col (str): Column name for folds.
        target_name (str): Name of the target set (e.g. shuffled).
        reference_name (str): Name of the reference set (e.g. test).
    Returns:
        results_df (pd.DataFrame): DataFrame with gene, average r, p-value, and adjusted p-value.
        fig (plt.Figure): Scatterplot figure with avg r vs -log10(p-adjusted).
    """
    
    # Split the dataframe into shuffled and actual data
    df_shuffled = df[df[model_col] == target_name]#.dropna(subset=['r'])
    df_shuffled['r'] = df_shuffled['r'].fillna(0)
    df_actual = df[df[model_col] == reference_name]#.dropna(subset=['r'])
    df_actual['r'] = df_actual['r'].fillna(0)

    features = df[feature_col].unique()
    results = []

    # Iterate through genes and perform t-tests across folds
    for feat in features:
        r_shuffled = df_shuffled[df_shuffled[feature_col] == feat][r_col]
        r_actual = df_actual[df_actual[feature_col] == feat][r_col]
        
        # Perform t-test comparing shuffled and actual
        _, p_value = mannwhitneyu(r_actual, r_shuffled,alternative='greater')
        
        # Compute average r across folds for actual model
        avg_r_actual = np.nanmean(r_actual)

        results.append({'feature': feat, 'avg_r': avg_r_actual, 'p_value': p_value})

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Adjust p-values using Benjamini-Hochberg correction
    results_df['p_adjusted'] = multipletests(results_df['p_value'], method='fdr_bh')[1]

    # Get the first two colors from the default Matplotlib color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_default = default_colors[0]  # First default color
    color_highlight = default_colors[1]  # Second default color
    # Set colors based on whether the gene will be highlighted cause it is important (in genes2show)
    colors = [color_highlight if feat in features2show else color_default for feat in results_df['feature']]
    
    # Plot avg_r vs -log10(p_adjusted)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(results_df['avg_r'], -np.log10(results_df['p_adjusted']), color=colors, alpha=alpha,s=25)

    # # Add text labels for highlighted features
    # for _, row in results_df.iterrows():
    #     if row['feature'] in features2show:
    #         ax.text(row['avg_r'], -np.log10(row['p_adjusted']), row['feature'], 
    #                 fontsize=8, color=color_highlight, ha='left', va='bottom')
    
    ax.set_xlabel('Average r across folds', fontsize=16)
    ax.set_ylabel('-log10(p-adjusted)', fontsize=16)
    ax.set_title('Per feature performance', fontsize=16)
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', linewidth=1.5)
    plt.tight_layout()

    return results_df, fig

### Load data
res_df = pd.read_csv('../results/CV_2hr_with_random_long.csv')
res_df=res_df.iloc[:,:-1]
results, scatterplot_fig = compare_target_vs_reference(res_df,'shuffled Y','test',alpha = 0.6)
scatterplot_fig.savefig('../figures/2 hour/dtlembas_2hr_per_TF_performance_scatterplot.pdf')
scatterplot_fig.savefig('../figures/2 hour/dtlembas_2hr_per_TF_performance_scatterplot.png',dpi=600)