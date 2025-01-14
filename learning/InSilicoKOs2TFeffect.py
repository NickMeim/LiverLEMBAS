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
from matplotlib import pyplot as plt
import plotnine as p9
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='Hepatocyte simulation')
parser.add_argument('--DrugsIn', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store',type = str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', action='store',type = str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--infoPath', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--res_dir', action='store',type = str, default='../results/allmodels/') #'../results/2 hour/allmodels/'
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
parser.add_argument('--stimulation_strength', action='store',type=float, default=3000)
parser.add_argument('--stimulation_sign', action='store',type = float, default=-1)
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected signaling nodes to predict and save their activity',
                    default=['P23458','O60674','P52333','P29597','P36897','P37173','P01137',
                             'Q15797','Q15796','Q99717','O43541',
                             'P17181','P48551','P15260','P38484',
                             'Q14653',
                             'P05231','Q13651'])

# stats = ['P42226','P42229','P51692','P40763','P52630','P42224','Q14765']
# smads = ['Q15797','Q15796','P84022','Q13485','Q99717','O43541']
# ifns = ['P17181','P48551','P15260','P38484']
# irfs = ['P10914','Q00978','Q14653']
# ils = ['P05231','Q13651']

args = parser.parse_args()
DrugsIn = args.DrugsIn
TargetsIn = args.TargetsIn
TFsOut = args.TFsOut
ChemicalSims = args.ChemicalSims
PknAnnotation = args.PknAnnotation
PKN = args.PKN
res_dir = args.res_dir
infoPath = args.infoPath
modelPattern = args.modelPattern
timepoint = args.timepoint
projection_amplitude_in = args.inputAmplitude
projection_amplitude_out = args.projectionAmplitude
ConvertToEmpProb = args.ConvertToEmpProb
number_DTLEMBAS_models = args.number_DTLEMBAS_models
stimulation_strength=args.stimulation_strength
stimulation_sign=args.stimulation_sign
selected_signaling_nodes = np.array(args.selected_signaling_nodes)
annotation_file_for_visualization = args.annotation_file_for_visualization

n_cores = 12
utils.set_cores(n_cores)
# seed = int(args.no) * (int(args.maxIter) + 1)
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
drugSmiles = drugInput.columns.values
TFOutput = pd.read_csv(TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput = TFOutput.loc[drugInput.index,:]

# Load drugs information
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
druginName = drugInput.columns.values
inName = drugTargets.columns.values
outconds = drugInput.index.values
outName = TFOutput.columns.values
inName = np.intersect1d(nodeNames, inName)
outName = np.intersect1d(nodeNames, outName)
TFOutput = TFOutput.loc[outconds,outName]
drugTargets = drugTargets.loc[:,inName]
if ConvertToEmpProb==True:
    print2log('Convereted to Emprirical probabilities')
    TFOutput = 1/(1+np.exp(-TFOutput))
drugTargets = drugTargets.loc[drugInput.columns.values,:]
drugSim = pd.read_csv(ChemicalSims,index_col=0)
drugSim = drugSim.loc[drugInput.columns.values,drugInput.columns.values]
drugSim = torch.tensor(drugSim.values.copy(), dtype=torch.double)
all_drugs = drugInput.columns
### Make tensors and initialize model
# X = torch.tensor(drugInput.values.copy(), dtype=torch.double).to(device)
Xdrugs = drugInput[drugInput['Control'] != 0].to_numpy()
Xdrugs = np.median(Xdrugs, axis=0, keepdims=True)
Xdrugs = torch.tensor(Xdrugs).double().to(device)
Y = torch.tensor(TFOutput.values, dtype=torch.double).to(device)

mask_dt =torch.tensor(np.array(drugTargets)).transpose(0,1)
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

print2log('Initialize node stimulation parameters...')
nodeNames = np.array(list(mod.node_idx_map.keys()))
num_tfs = mod.y_out.shape[1]
all_tfs = mod.y_out.columns.values
all_nodes = np.array(list(mod.node_idx_map.keys()))

TF_all = torch.zeros(number_DTLEMBAS_models,len(all_nodes),num_tfs, dtype=torch.double)
Xin_ko = torch.eye(len(all_nodes),dtype=torch.double).to(device)
Xin_ko = Xin_ko.fill_diagonal_(stimulation_sign * stimulation_strength)

TF_all_control = torch.zeros(number_DTLEMBAS_models,Xdrugs.shape[0],num_tfs, dtype=torch.double)
TF_all_empty = torch.zeros(number_DTLEMBAS_models,1,num_tfs, dtype=torch.double)
TF_all_control_ko = torch.zeros(number_DTLEMBAS_models,len(all_nodes),num_tfs, dtype=torch.double)

## Select some specific nodes for measuring their activity
subset_nodes = {key: mod.node_idx_map[key] for key in selected_signaling_nodes if key in mod.node_idx_map}
subset_nodes_inds = np.array(list(subset_nodes.values()))
subset_nodes = np.array(list(subset_nodes.keys()))

### Initialize big tensors for signaling nodes
Nodes_subset = torch.zeros(number_DTLEMBAS_models,len(all_nodes),len(subset_nodes), dtype=torch.double)
Nodes_subset_control = torch.zeros(number_DTLEMBAS_models,Xdrugs.shape[0],len(subset_nodes), dtype=torch.double)
Nodes_subset_empty = torch.zeros(number_DTLEMBAS_models,1,len(subset_nodes), dtype=torch.double)
Nodes_subset_control_ko = torch.zeros(number_DTLEMBAS_models,len(all_nodes),len(subset_nodes), dtype=torch.double)

print2log('Begin KO or Activation simulation...')
if (stimulation_sign <0):
    name_stimulation = 'InSilicoKOs'
else:
    name_stimulation = 'InSilicoActivations'
Path(res_dir+name_stimulation).mkdir(parents=True, exist_ok=True)

results_tf = pd.DataFrame({})
results_singaling_nodes = pd.DataFrame({})
start_time = time.time()
for i in range(number_DTLEMBAS_models):
    print2log('Run DT-LEMBAS model '+str(i+1)+' to simulate KOs')
    mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(i+1)+'signal_state_dict_%s.pth'%(i+1))))
    drugLayer.load_state_dict(torch.load(os.path.join(data_path,modelPattern + str(i+1) + 'drugLayer_state_dict_%s.pth'%(i+1))))
    mod.eval()
    drugLayer.eval()
    # print2log('Node stimuli simulation...')
    YhatFull_ko = mod.signaling_network(Xin_ko)
    Yhat_ko = mod.output_layer(YhatFull_ko)
    TF_all[i,:,:] = Yhat_ko.detach().cpu()
    Nodes_subset[i,:,:] = YhatFull_ko[:,subset_nodes_inds].detach().cpu()
    df_ko = pd.DataFrame(Yhat_ko.detach().cpu().numpy())
    df_ko.columns = all_tfs
    df_ko['condition'] = all_nodes
    df_ko['type'] = 'KO'
    df_ko['model_no'] = i
    results_tf = pd.concat([results_tf, df_ko], ignore_index=True).reset_index(drop=True)
    # make dataframe for singaling nodes now
    df_singaling_nodes = pd.DataFrame(YhatFull_ko[:,subset_nodes_inds].detach().cpu().numpy())
    df_singaling_nodes.columns = subset_nodes
    df_singaling_nodes['condition'] = all_nodes
    df_singaling_nodes['type'] = 'KO'
    df_singaling_nodes['model_no'] = i
    results_singaling_nodes = pd.concat([results_singaling_nodes, df_singaling_nodes], ignore_index=True).reset_index(drop=True)

    ## repeat by passing only control to use as differential analysis then
    # print2log('control simulation...')
    Xin = drugLayer(Xdrugs)
    TF, NetFull = mod(Xin)
    TF_all_control[i,:,:] = TF.detach().cpu()
    Nodes_subset_control[i,:,:] = NetFull[:,subset_nodes_inds].detach().cpu()
    df_control = pd.DataFrame(TF.detach().cpu().numpy())
    df_control.columns = all_tfs
    df_control['condition'] = 'control'
    df_control['type'] = 'control'
    df_control['model_no'] = i
    results_tf = pd.concat([results_tf, df_control], ignore_index=True).reset_index(drop=True)
    # make dataframe for singaling nodes now
    df_singaling_nodes = pd.DataFrame(NetFull[:,subset_nodes_inds].detach().cpu().numpy())
    df_singaling_nodes.columns = subset_nodes
    df_singaling_nodes['condition'] = 'control'
    df_singaling_nodes['type'] = 'control'
    df_singaling_nodes['model_no'] = i
    results_singaling_nodes = pd.concat([results_singaling_nodes, df_singaling_nodes], ignore_index=True).reset_index(drop=True)

    ## repeat by passing control + KO to use as differential analysis then
    # print2log('control + node stimuli simulation...')
    Xin = mod.input_layer(drugLayer(Xdrugs)) + Xin_ko
    YhatFull_ko_control = mod.signaling_network(Xin)
    Yhat_ko_control = mod.output_layer(YhatFull_ko_control)
    TF_all_control_ko[i,:,:] = Yhat_ko_control.detach().cpu()
    Nodes_subset_control_ko[i,:,:] = YhatFull_ko_control[:,subset_nodes_inds].detach().cpu()
    df_ko_control = pd.DataFrame(Yhat_ko_control.detach().cpu().numpy())
    df_ko_control.columns = all_tfs
    df_ko_control['condition'] =  [node+'+control' for node in all_nodes]
    df_ko_control['type'] = 'KO+control'
    df_ko_control['model_no'] = i
    results_tf = pd.concat([results_tf, df_ko_control], ignore_index=True).reset_index(drop=True)
    # make dataframe for singaling nodes now
    df_singaling_nodes = pd.DataFrame(YhatFull_ko_control[:,subset_nodes_inds].detach().cpu().numpy())
    df_singaling_nodes.columns = subset_nodes
    df_singaling_nodes['condition'] =  [node+'+control' for node in all_nodes]
    df_singaling_nodes['type'] = 'KO+control'
    df_singaling_nodes['model_no'] = i
    results_singaling_nodes = pd.concat([results_singaling_nodes, df_singaling_nodes], ignore_index=True).reset_index(drop=True)

    ## repeat by passing zero argument
    # print2log('Untreated simulation...')
    TF, NetFull = mod(torch.zeros(1,drugTargets.shape[1],dtype=torch.double).to(device))
    TF_all_empty[i,:,:] = TF.detach().cpu()
    Nodes_subset_empty[i,:,:] = NetFull[:,subset_nodes_inds].detach().cpu()
    df_empty = pd.DataFrame(TF.detach().cpu().numpy())
    df_empty.columns = all_tfs
    df_empty['condition'] = 'untreated'
    df_empty['type'] = 'untreated'
    df_empty['model_no'] = i
    results_tf = pd.concat([results_tf, df_empty], ignore_index=True).reset_index(drop=True)
    # make dataframe for singaling nodes now
    df_singaling_nodes = pd.DataFrame(NetFull[:,subset_nodes_inds].detach().cpu().numpy())
    df_singaling_nodes.columns = subset_nodes
    df_singaling_nodes['condition'] = 'untreated'
    df_singaling_nodes['type'] = 'untreated'
    df_singaling_nodes['model_no'] = i
    results_singaling_nodes = pd.concat([results_singaling_nodes, df_singaling_nodes], ignore_index=True).reset_index(drop=True)

mins, secs = divmod(time.time() - start_time, 60)
print2log("Simulation ran in: {:.0f} min {:.2f} sec".format(mins, secs))
print2log('Saving results...')
results_tf.to_csv(res_dir+name_stimulation+'/TFs_all_conditions.csv')
results_tf = results_tf.melt(id_vars=['condition','type','model_no'],var_name='TF',value_name='activity')
results_singaling_nodes.to_csv(res_dir+name_stimulation+'/Nodes_subset_conditions.csv')
print2log('Saving TF and selected nodesactivities...')
torch.save(TF_all, res_dir+name_stimulation+'/TFs_all.pt')
torch.save(TF_all_control_ko, res_dir+name_stimulation+'/TFs_all_control_ko.pt')
torch.save(TF_all_control, res_dir+name_stimulation+'/TFs_all_control.pt')
torch.save(TF_all_empty, res_dir+name_stimulation+'/TFs_all_empty.pt')
# save also singaling nodes activities
torch.save(Nodes_subset, res_dir+name_stimulation+'/Nodes_subset.pt')
torch.save(Nodes_subset_control, res_dir+name_stimulation+'/Nodes_subset_control.pt')
torch.save(Nodes_subset_control_ko, res_dir+name_stimulation+'/Nodes_subset_control_ko.pt')
torch.save(Nodes_subset_empty, res_dir+name_stimulation+'/Nodes_subset_empty.pt')
print2log('Saving averaged results...')
TF = torch.mean(TF_all, dim=0)
TF = pd.DataFrame(TF.detach().cpu().numpy())
TF.columns = all_tfs
TF.index = all_nodes
TF.to_csv(res_dir+name_stimulation+'/TF_avg.csv')
## Repeat saving for control
print2log('Repeat saving for control!!!')
TF = torch.mean(TF_all_control, dim=0)
TF = pd.DataFrame(TF.detach().cpu().numpy())
TF.columns = all_tfs
TF.index = ['control']
TF.to_csv(res_dir+name_stimulation+'/TF_avg_control.csv')
## Repeat saving for control + KO
print2log('Repeat saving for control + KOs!!!')
TF = torch.mean(TF_all_control_ko, dim=0)
TF = pd.DataFrame(TF.detach().cpu().numpy())
TF.columns = all_tfs
TF.index = [node+'+control' for node in all_nodes]
TF.to_csv(res_dir+name_stimulation+'/TF_avg_control_ko.csv')
## Repeat saving for empty
print2log('Repeat saving for no input!!!')
TF = torch.mean(TF_all_empty, dim=0)
TF = pd.DataFrame(TF.detach().cpu().numpy())
TF.columns = all_tfs
TF.index = ['control']
TF.to_csv(res_dir+name_stimulation+'/TF_avg_empty.csv')

################################ Visualization ################################
print2log('Visualizating top 10 effects on some TFs...')
tf_list = ["P42224","P52630","Q00978","P84022","P10914"]
n_tfs = len(tf_list)
fig, axes = plt.subplots(n_tfs, 1, figsize=(12, 5 * n_tfs), sharex=False)
# Iterate over each TF in the list
plot_list = []
for i, tf_name in enumerate(tf_list):
    # Filter for transcription factor
    # tf_name = "P42224"
    filtered_df = results_tf[results_tf['TF'] == tf_name]
    filtered_df = filtered_df[filtered_df['condition']!=tf_name]
    ## Get rid of NODE + control conditions
    filtered_df = filtered_df[~filtered_df['type'].isin(['KO+control'])]
    untreated_mean = filtered_df[filtered_df['type']=='untreated']['activity'].mean()
    # Exclude 'control' and 'untreated'
    filtered_df = filtered_df[~filtered_df['type'].isin(['control', 'untreated'])]
    # Calculate deviation from 0.5
    filtered_df['deviation'] = (filtered_df['activity'] - untreated_mean).abs()
    # Group by 'type' and 'condition' to get the mean deviation
    mean_deviation = filtered_df.groupby(['type', 'condition'])['deviation'].mean().reset_index()
    # Get the top 6 conditions for each type based on deviation
    top_conditions_per_type = mean_deviation.groupby('type').apply(
        lambda x: x.nlargest(10, 'deviation')
    ).reset_index(drop=True)
    # Merge with the original data to get the top conditions' activities
    top_conditions_df = filtered_df.merge(
        top_conditions_per_type[['type', 'condition']],
        on=['type', 'condition'],
        how='inner'
    )
    # Add back 'control' and 'untreated' for reference
    final_df = pd.concat([
        results_tf[(results_tf['type'].isin(['control', 'untreated'])) & (results_tf['TF'] == tf_name)],
        top_conditions_df
    ])
    ## Re-annotate dataframe to have gene symbols
    # First, merge to replace the 'TF' column
    final_df_merged = pd.merge(final_df, annot, how='left', left_on='TF', right_on='uniprot')
    # Rename the 'symbol' column to 'TF_symbol' (or just 'TF' if you prefer to replace the old one directly)
    final_df_merged.rename(columns={'symbol': 'TF_symbol'}, inplace=True)
    # Drop the 'TF' and 'uniprot' columns, keeping the new mapped column 'TF_symbol'
    final_df_merged.drop(columns=['TF', 'uniprot'], inplace=True)
    # Now, merge again to replace the 'condition' column
    final_df_merged = pd.merge(final_df_merged, annot, how='left', left_on='condition', right_on='uniprot')
    # Rename the 'symbol' column to 'condition_symbol' (or 'condition')
    final_df_merged.rename(columns={'symbol': 'condition_symbol'}, inplace=True)
    # Drop the 'condition' and 'uniprot' columns, keeping the new mapped column 'condition_symbol'
    final_df_merged.drop(columns=['condition', 'uniprot'], inplace=True)
    final_df_merged.rename(columns={'TF_symbol': 'TF', 'condition_symbol': 'condition'}, inplace=True)
    tf_symbol = annot[annot['uniprot'] == tf_name]['symbol'].values[0]
    # Plot the boxplot
    # Create the plot with plotnine
    p = (p9.ggplot(final_df_merged, p9.aes(x='condition', y='activity'))
         + p9.geom_boxplot(p9.aes(group='condition'), outlier_shape='', show_legend=False)  # Exclude outliers
         + p9.geom_point(position=p9.position_jitter(width=0.2), color='black', alpha=0.6, size=2)
         + p9.geom_hline(yintercept=0.5, linetype='dashed', color='black')  # Add a dashed line at 0.5
         + p9.labs(title=f"TF Activity for {tf_symbol}", x='Condition', y='Activity')
         + p9.theme_bw()
         + p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1), figure_size=(12, 6))
    )
    # Add plot to list
    plot_list.append(p)

# Combine all plots into one PDF
from matplotlib.backends.backend_pdf import PdfPages
pdf_path = res_dir+name_stimulation+"/TFs_activity_boxplot.pdf"
with PdfPages(pdf_path) as pdf:
    for plot in plot_list:
        fig = plot.draw()  # Draw the plotnine plot and convert it to matplotlib
        pdf.savefig(fig)   # Save each plot as a page in the PDF
        plt.close(fig)     # Close the plot to free memory
print2log("Finished!")
