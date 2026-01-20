import os
import numpy as np
import pandas as pd
import torch
import sys 
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
# gpu_lembas_path = '/home/meimetis/meimetis/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
import argparse
import logging
from pathlib import Path
from utility import *
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='Hepatocyte signaling node activity analysis')
parser.add_argument('--DrugsIn', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store',type = str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', action='store',type = str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--infoPath', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--res_dir', action='store',type = str, default='../results/allmodels/')
parser.add_argument('--number_DTLEMBAS_models', action='store',type = int, default=50)
parser.add_argument('--modelPattern', action='store',type = str, default='hepatocytes_dtlembas')
parser.add_argument('--ConvertToEmpProb', action='store',type = bool, default=False)
parser.add_argument('--inputAmplitude', action='store',type = float, default=3)
parser.add_argument('--projectionAmplitude', action='store',type = float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', action='store',type = int, default=90)
parser.add_argument('--LEMBAS_leak', action='store',type = float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', action='store', default=1e-5)
parser.add_argument('--drug_bn', action='store', default=0.2)
parser.add_argument('--drug_drop', action='store', default=0.1)
parser.add_argument('--timepoint', action='store',type=str, default='8 hour')
parser.add_argument('--annotation_file_for_visualization', action='store',type=str, default='../data/annotation.csv')
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected signaling nodes (KO nodes + TGFBR1 + IFNAR1)',
                    default=['Q9H3D4', 'P52630', 'P40763', 'P84022', 'P04637', 'P36897', 'P17181'])

args = parser.parse_args()
DrugsIn = args.DrugsIn
TargetsIn = args.TargetsIn
TFsOut = args.TFsOut
ChemicalSims = args.ChemicalSims
PKN = args.PKN
res_dir = args.res_dir
infoPath = args.infoPath
modelPattern = args.modelPattern
timepoint = args.timepoint
projection_amplitude_in = args.inputAmplitude
projection_amplitude_out = args.projectionAmplitude
ConvertToEmpProb = args.ConvertToEmpProb
number_DTLEMBAS_models = args.number_DTLEMBAS_models
selected_signaling_nodes = args.selected_signaling_nodes
annotation_file_for_visualization = args.annotation_file_for_visualization

n_cores = 12
utils.set_cores(n_cores)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log('Running on {}'.format(device))
data_path = res_dir
if not os.path.isdir(data_path):
    os.mkdir(data_path)

print2log('Loading input data...')
# prior knowledge signaling network
net = pd.read_csv(PKN, sep = '\t', index_col = False).loc[:,['source','target','stimulation','inhibition']]
# ligand input and TF output
drugInput = pd.read_csv(DrugsIn, sep = '\t', index_col = 0)
TFOutput = pd.read_csv(TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput = TFOutput.loc[drugInput.index,:]

# Load drugs information and annotation
annot = pd.read_csv(annotation_file_for_visualization)
drugTargets = pd.read_csv(TargetsIn, sep='\t', low_memory=False, index_col=0)
inputInfo = pd.read_csv(infoPath, sep='\t', low_memory=False, index_col=0)
inputInfo = inputInfo[inputInfo['time']==timepoint]
TFOutput = TFOutput.loc[inputInfo.index,:]
drugInput = drugInput.loc[inputInfo.index,:]

stimulation_label = 'stimulation'
inhibition_label = 'inhibition'
weight_label = 'mode_of_action'
source_label = 'source'
target_label = 'target'
net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)
nodeNames = np.union1d(net.source.values,net.target.values)

#Subset input and output to intersecting nodes
inName = drugTargets.columns.values
outconds = drugInput.index.values
outName = TFOutput.columns.values
inName = np.intersect1d(nodeNames, inName)
outName = np.intersect1d(nodeNames, outName)
TFOutput = TFOutput.loc[outconds,outName]
drugTargets = drugTargets.loc[:,inName]

if ConvertToEmpProb==True:
    print2log('Converted to Empirical probabilities')
    TFOutput = 1/(1+np.exp(-TFOutput))

drugTargets = drugTargets.loc[drugInput.columns.values,:]
drugSim = pd.read_csv(ChemicalSims,index_col=0)
drugSim = drugSim.loc[drugInput.columns.values,drugInput.columns.values]
drugSim = torch.tensor(drugSim.values.copy(), dtype=torch.double)

### Make tensors and initialize model
Xdrugs = drugInput[drugInput['Control'] != 0].to_numpy()
Xdrugs = np.median(Xdrugs, axis=0, keepdims=True)
Xdrugs = torch.tensor(Xdrugs).double().to(device)
Y = torch.tensor(TFOutput.values, dtype=torch.double).to(device)

mask_dt = torch.tensor(np.array(drugTargets)).transpose(0,1)
bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 
                 'exp_factor':50, 'tolerance': args.LEMBAS_tolerance, 
                 'leak':args.LEMBAS_leak}

mod = SignalingModel(net = net,
                     X_in = drugTargets,
                     y_out = TFOutput, 
                     projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                     bionet_params = bionet_params, 
                     dtype = torch.double, device = device)

drugLayer = DrugstoSignal(mask_dt, 
                          mask_dt.shape[1],mask_dt.shape[0],
                          drugSim,momentum=args.drug_bn,drop=args.drug_drop,
                          device = device)
drugLayer = drugLayer.to(device)

print2log('Initialize node parameters...')
all_nodes = np.array(list(mod.node_idx_map.keys()))
all_tfs = mod.y_out.columns.values

# Create mapping from UniProt ID to gene symbol
print2log('Creating UniProt to Symbol mapping...')
uniprot_to_symbol = {}
if 'uniprot' in annot.columns and 'symbol' in annot.columns:
    for _, row in annot.iterrows():
        if pd.notna(row['uniprot']) and pd.notna(row['symbol']):
            uniprot_to_symbol[row['uniprot']] = row['symbol']
    print2log(f'Loaded {len(uniprot_to_symbol)} UniProt to Symbol mappings')
else:
    print2log('WARNING: uniprot or symbol column not found in annotation file')

# Get indices and symbols for selected signaling nodes
subset_nodes = {key: mod.node_idx_map[key] for key in selected_signaling_nodes if key in mod.node_idx_map}
subset_nodes_inds = np.array(list(subset_nodes.values()))
subset_nodes_names = np.array(list(subset_nodes.keys()))
subset_nodes_symbols = [uniprot_to_symbol.get(node, node) for node in subset_nodes_names]

print2log(f'Selected signaling nodes ({len(subset_nodes)}):')
for uniprot, symbol in zip(subset_nodes_names, subset_nodes_symbols):
    print2log(f'  {uniprot} ({symbol})')

# Create output directory
output_dir = 'baseline_signaling_node_activity_analysis/' + timepoint + '/'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Initialize dataframes to store results
results_tf_untreated = []
results_tf_control = []
results_signaling_untreated = []
results_signaling_control = []
results_bias = []

print2log('\nRunning simulations...')
start_time = time.time()

for i in range(number_DTLEMBAS_models):
    print2log(f'Processing model {i+1}/{number_DTLEMBAS_models}')
    
    # Load model
    mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(i+1)+'signal_state_dict_%s.pth'%(i+1))))
    drugLayer.load_state_dict(torch.load(os.path.join(data_path,modelPattern + str(i+1) + 'drugLayer_state_dict_%s.pth'%(i+1))))
    mod.eval()
    drugLayer.eval()
    
    with torch.no_grad():
        # 1. Untreated condition (zero input)
        TF_untreated, NetFull_untreated = mod(torch.zeros(1, drugTargets.shape[1], dtype=torch.double).to(device))
        
        # Store TF predictions for untreated
        df_tf_untreated = pd.DataFrame(TF_untreated.detach().cpu().numpy(), columns=all_tfs)
        df_tf_untreated['model_no'] = i
        df_tf_untreated['condition'] = 'untreated'
        results_tf_untreated.append(df_tf_untreated)
        
        # Store signaling node activities for untreated
        signaling_activities_untreated = NetFull_untreated[:, subset_nodes_inds].detach().cpu().numpy()
        df_sig_untreated = pd.DataFrame(signaling_activities_untreated, columns=subset_nodes_symbols)
        df_sig_untreated['model_no'] = i
        df_sig_untreated['condition'] = 'untreated'
        results_signaling_untreated.append(df_sig_untreated)
        
        # 2. Control condition (median control input through drug layer and model)
        Xin_control = drugLayer(Xdrugs)
        TF_control, NetFull_control = mod(Xin_control)
        
        # Store TF predictions for control
        df_tf_control = pd.DataFrame(TF_control.detach().cpu().numpy(), columns=all_tfs)
        df_tf_control['model_no'] = i
        df_tf_control['condition'] = 'control'
        results_tf_control.append(df_tf_control)
        
        # Store signaling node activities for control
        signaling_activities_control = NetFull_control[:, subset_nodes_inds].detach().cpu().numpy()
        df_sig_control = pd.DataFrame(signaling_activities_control, columns=subset_nodes_symbols)
        df_sig_control['model_no'] = i
        df_sig_control['condition'] = 'control'
        results_signaling_control.append(df_sig_control)
        
        # 3. Extract bias for selected nodes
        bias = mod.signaling_network.bias.detach().cpu().numpy()
        # Flatten bias if needed and extract subset
        bias = bias.flatten() if bias.ndim > 1 else bias
        bias_subset = bias[subset_nodes_inds]
        df_bias = pd.DataFrame([bias_subset], columns=subset_nodes_symbols)
        df_bias['model_no'] = i
        results_bias.append(df_bias)

mins, secs = divmod(time.time() - start_time, 60)
print2log(f"Simulation completed in: {mins:.0f} min {secs:.2f} sec")

# Combine all results
print2log('\nCombining results...')
df_tf_untreated_all = pd.concat(results_tf_untreated, ignore_index=True)
df_tf_control_all = pd.concat(results_tf_control, ignore_index=True)
df_sig_untreated_all = pd.concat(results_signaling_untreated, ignore_index=True)
df_sig_control_all = pd.concat(results_signaling_control, ignore_index=True)
df_bias_all = pd.concat(results_bias, ignore_index=True)

# Save dataframes
print2log('Saving results...')
df_tf_untreated_all.to_csv(output_dir + 'TF_predictions_untreated.csv', index=False)
df_tf_control_all.to_csv(output_dir + 'TF_predictions_control.csv', index=False)
df_sig_untreated_all.to_csv(output_dir + 'signaling_node_activities_untreated.csv', index=False)
df_sig_control_all.to_csv(output_dir + 'signaling_node_activities_control.csv', index=False)
df_bias_all.to_csv(output_dir + 'signaling_node_bias.csv', index=False)

# Identify which KO nodes are TFs
print2log('\nIdentifying KO nodes that are TFs...')
ko_nodes_uniprot = ['Q9H3D4', 'P52630', 'P40763', 'P84022', 'P04637']  # First 5 are KO nodes
ko_tf_nodes = []
ko_tf_symbols = []

for ko_node in ko_nodes_uniprot:
    if ko_node in all_tfs:
        ko_tf_nodes.append(ko_node)
        ko_tf_symbols.append(uniprot_to_symbol.get(ko_node, ko_node))

print2log(f'Found {len(ko_tf_nodes)} KO nodes that are TFs:')
for uniprot, symbol in zip(ko_tf_nodes, ko_tf_symbols):
    print2log(f'  {uniprot} ({symbol})')

# Create boxplots
print2log('\nGenerating boxplot figures...')

# Prepare data for plotting - add 'ALL' category
def prepare_plot_data(df, condition_name):
    """Prepare data for boxplot with ALL category containing all points from all nodes"""
    plot_data = []
    
    # Individual nodes
    for col in subset_nodes_symbols:
        for idx, row in df.iterrows():
            plot_data.append({
                'Node': col,
                'Activity': row[col],
                'model_no': row['model_no']
            })
    
    # ALL nodes (all values from all nodes and all models)
    for col in subset_nodes_symbols:
        for idx, row in df.iterrows():
            plot_data.append({
                'Node': 'ALL',
                'Activity': row[col],
                'model_no': row['model_no']
            })
    
    return pd.DataFrame(plot_data)

# Prepare plot data
plot_data_untreated = prepare_plot_data(df_sig_untreated_all, 'untreated')
plot_data_control = prepare_plot_data(df_sig_control_all, 'control')

# Prepare bias data
plot_data_bias = []
for col in subset_nodes_symbols:
    for idx, row in df_bias_all.iterrows():
        plot_data_bias.append({
            'Node': col,
            'Bias': row[col],
            'model_no': row['model_no']
        })
# ALL bias (all values from all nodes and all models)
for col in subset_nodes_symbols:
    for idx, row in df_bias_all.iterrows():
        plot_data_bias.append({
            'Node': 'ALL',
            'Bias': row[col],
            'model_no': row['model_no']
        })
plot_data_bias = pd.DataFrame(plot_data_bias)

# Calculate y-axis limits for activity plots
activity_min = min(plot_data_untreated['Activity'].min(), plot_data_control['Activity'].min(), 0)
activity_max = 1

# Create PDF with all plots
pdf_path = output_dir + 'signaling_node_activity_boxplots.pdf'
with PdfPages(pdf_path) as pdf:
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Untreated
    fig, ax = plt.subplots(figsize=(12, 6))
    node_order = subset_nodes_symbols + ['ALL']
    sns.boxplot(data=plot_data_untreated, x='Node', y='Activity', order=node_order, ax=ax)
    sns.stripplot(data=plot_data_untreated, x='Node', y='Activity', order=node_order, 
                  color='black', alpha=0.3, size=2, ax=ax, jitter=True)
    ax.set_ylim(activity_min, activity_max)
    ax.set_title('Signaling Node Activity - Untreated Condition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node', fontsize=12)
    ax.set_ylabel('Activity', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # Figure 2: Control
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_data_control, x='Node', y='Activity', order=node_order, ax=ax)
    sns.stripplot(data=plot_data_control, x='Node', y='Activity', order=node_order, 
                  color='black', alpha=0.3, size=2, ax=ax, jitter=True)
    ax.set_ylim(activity_min, activity_max)
    ax.set_title('Signaling Node Activity - Control Condition', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node', fontsize=12)
    ax.set_ylabel('Activity', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    
    # Figure 3: Bias
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_data_bias, x='Node', y='Bias', order=node_order, ax=ax)
    sns.stripplot(data=plot_data_bias, x='Node', y='Bias', order=node_order, 
                  color='black', alpha=0.3, size=2, ax=ax, jitter=True)
    ax.set_title('Signaling Network Bias - Selected Nodes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Node', fontsize=12)
    ax.set_ylabel('Bias', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print2log(f'\nBoxplots saved to: {pdf_path}')

# Create second PDF for TF activities of KO nodes
if len(ko_tf_nodes) > 0:
    print2log('\nGenerating TF activity boxplots for KO nodes...')
    
    # Prepare TF activity data for KO nodes
    def prepare_tf_plot_data(df, condition_name):
        """Prepare TF data for boxplot with ALL category containing all points from all nodes"""
        plot_data = []
        
        # Individual TF nodes
        for col, symbol in zip(ko_tf_nodes, ko_tf_symbols):
            for idx, row in df.iterrows():
                plot_data.append({
                    'Node': symbol,
                    'TF_Activity': row[col],
                    'model_no': row['model_no']
                })
        
        # ALL TFs (all values from all KO TF nodes and all models)
        for col, symbol in zip(ko_tf_nodes, ko_tf_symbols):
            for idx, row in df.iterrows():
                plot_data.append({
                    'Node': 'ALL',
                    'TF_Activity': row[col],
                    'model_no': row['model_no']
                })
        
        return pd.DataFrame(plot_data)
    
    # Prepare plot data for TFs
    plot_data_tf_untreated = prepare_tf_plot_data(df_tf_untreated_all, 'untreated')
    plot_data_tf_control = prepare_tf_plot_data(df_tf_control_all, 'control')
    
    # Calculate y-axis limits for TF activity plots
    tf_activity_min = min(plot_data_tf_untreated['TF_Activity'].min(), plot_data_tf_control['TF_Activity'].min(), 0)
    tf_activity_max = 1
    
    # Create PDF with TF activity plots
    pdf_tf_path = output_dir + 'KO_TF_activity_boxplots.pdf'
    with PdfPages(pdf_tf_path) as pdf:
        sns.set_style("whitegrid")
        
        # Figure 1: Untreated TF activities
        fig, ax = plt.subplots(figsize=(10, 6))
        node_order = ko_tf_symbols + ['ALL']
        sns.boxplot(data=plot_data_tf_untreated, x='Node', y='TF_Activity', order=node_order, ax=ax)
        sns.stripplot(data=plot_data_tf_untreated, x='Node', y='TF_Activity', order=node_order, 
                      color='black', alpha=0.3, size=2, ax=ax, jitter=True)
        ax.set_ylim(tf_activity_min, tf_activity_max)
        ax.set_title('TF Activity for KO Nodes - Untreated Condition', fontsize=14, fontweight='bold')
        ax.set_xlabel('TF Node', fontsize=12)
        ax.set_ylabel('TF Activity', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Figure 2: Control TF activities
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=plot_data_tf_control, x='Node', y='TF_Activity', order=node_order, ax=ax)
        sns.stripplot(data=plot_data_tf_control, x='Node', y='TF_Activity', order=node_order, 
                      color='black', alpha=0.3, size=2, ax=ax, jitter=True)
        ax.set_ylim(tf_activity_min, tf_activity_max)
        ax.set_title('TF Activity for KO Nodes - Control Condition', fontsize=14, fontweight='bold')
        ax.set_xlabel('TF Node', fontsize=12)
        ax.set_ylabel('TF Activity', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    print2log(f'TF activity boxplots saved to: {pdf_tf_path}')
else:
    print2log('\nNo KO nodes found that are TFs - skipping TF activity plots')

# Create third PDF with histograms of all TF activities, highlighting selected TFs
print2log('\nGenerating TF activity distribution histograms...')

# Get all TF activities from the loaded data (TFOutput)
all_tf_values = TFOutput.values.flatten()

# Create PDF with histograms for each selected TF
pdf_hist_path = output_dir + 'TF_activity_distributions.pdf'
with PdfPages(pdf_hist_path) as pdf:
    sns.set_style("whitegrid")
    
    for ko_node, ko_symbol in zip(ko_tf_nodes, ko_tf_symbols):
        # Get the activity values for this specific TF from the data
        tf_values = TFOutput[ko_node].values
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of ALL TF activities
        ax.hist(all_tf_values, bins=50, alpha=0.5, color='gray', 
                label='All TF Activities', edgecolor='black')
        
        # Calculate statistics for the selected TF
        tf_mean = tf_values.mean()
        tf_std = tf_values.std()
        tf_min = tf_values.min()
        tf_max = tf_values.max()
        
        # Add shaded region for the range of this TF's activities
        ax.axvspan(tf_min, tf_max, alpha=0.2, color='red', 
                   label=f'{ko_symbol} range')
        
        # Add dashed line for mean of this TF
        ax.axvline(tf_mean, color='red', linestyle='--', linewidth=2,
                   label=f'{ko_symbol} mean')
        
        # Add dashed lines for mean ± std
        ax.axvline(tf_mean - tf_std, color='orange', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'{ko_symbol} mean ± std')
        ax.axvline(tf_mean + tf_std, color='orange', linestyle='--', linewidth=1.5,
                   alpha=0.7)
        
        ax.set_xlabel('TF Activity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of All TF Activities\nHighlighting {ko_symbol} ({ko_node})', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'{ko_symbol} Statistics:\n'
        stats_text += f'Mean: {tf_mean:.3f}\n'
        stats_text += f'Std: {tf_std:.3f}\n'
        stats_text += f'Min: {tf_min:.3f}\n'
        stats_text += f'Max: {tf_max:.3f}\n'
        stats_text += f'N samples: {len(tf_values)}'
        
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9, family='monospace')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

print2log(f'TF activity distribution histograms saved to: {pdf_hist_path}')
print2log('\nSummary statistics:')
print2log(f'Number of models processed: {number_DTLEMBAS_models}')
print2log(f'Number of selected nodes: {len(subset_nodes)}')
print2log(f'TF predictions saved for untreated and control conditions')
print2log(f'Signaling node activities saved for untreated and control conditions')
print2log(f'Bias values saved for all models')

print2log('\nFinished!')