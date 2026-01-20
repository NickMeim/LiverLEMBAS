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
parser.add_argument('--num_steps', action='store',type = float, default=101)
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected signaling nodes to predict and save their activity',
                    default=['P36897','P17181'])
parser.add_argument('--knockout_node', action='store', type=str, default='Q9H3D4',
                    help='Node to knockout during simulation (default: Q9H3D4)')
parser.add_argument('--knockout_strength', action='store', type=float, default=-3000,
                    help='Strength of knockout (large negative value, default: -3000)')

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
selected_signaling_nodes = np.array(args.selected_signaling_nodes)
annotation_file_for_visualization = args.annotation_file_for_visualization
num_steps = args.num_steps
knockout_node = args.knockout_node
knockout_strength = args.knockout_strength
#res_dir = res_dir+ timepoint
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

## Select some specific nodes for measuring their activity
subset_nodes = {key: mod.node_idx_map[key] for key in selected_signaling_nodes if key in mod.node_idx_map}
subset_nodes_inds = np.array(list(subset_nodes.values()))
subset_nodes = np.array(list(subset_nodes.keys()))

# Setup knockout node
print2log(f'Setting up knockout for node: {knockout_node}')
if knockout_node not in mod.node_idx_map:
    print2log(f'WARNING: Knockout node {knockout_node} not found in network! Available nodes: {list(mod.node_idx_map.keys())[:10]}...')
    raise ValueError(f'Knockout node {knockout_node} not in network')
    
knockout_node_idx = mod.node_idx_map[knockout_node]
print2log(f'Knockout node {knockout_node} has index {knockout_node_idx}')

Xin = torch.eye(len(all_nodes),dtype=torch.double).to(device)
Xin_1 = Xin[subset_nodes_inds[0],:].unsqueeze(0)
Xin_2 = Xin[subset_nodes_inds[1],:].unsqueeze(0)
Xin_ko = Xin[knockout_node_idx,:].unsqueeze(0)  # Knockout node input

# Define the number of interpolation steps you want
# Create linearly spaced values between -3000 and 3000
stim_range = torch.linspace(-stimulation_strength, stimulation_strength, num_steps)
# Create a grid of all combinations of stimulations
grid_Xin1, grid_Xin2 = torch.meshgrid(stim_range, stim_range, indexing='ij')
# Flatten the grid for batch processing
grid_Xin1 = grid_Xin1.flatten().view(-1, 1)
grid_Xin2 = grid_Xin2.flatten().view(-1, 1)
# Expand the grids to match input dimensions
grid_Xin1 = grid_Xin1.repeat(1, Xin_1.shape[1])
grid_Xin2 = grid_Xin2.repeat(1, Xin_2.shape[1])
# Multiply inputs with the grid values
Xin_1_grid = Xin_1 * grid_Xin1.to(device).double()
Xin_2_grid = Xin_2 * grid_Xin2.to(device).double()
# Apply knockout (strong negative stimulation)
Xin_ko_grid = Xin_ko * knockout_strength
Xin_ko_grid = Xin_ko_grid.repeat(grid_Xin1.shape[0], 1).to(device).double()
# Combine the two inputs with knockout for the model
Xin_combined = Xin_1_grid + Xin_2_grid + Xin_ko_grid



print2log(f'Begin combination stimuli simulation with {knockout_node} knockout...')
name_stimulation = f'combo_simulation_expanded_KO_{knockout_node}'
Path(res_dir+name_stimulation).mkdir(parents=True, exist_ok=True)

results_tf = pd.DataFrame({})
results_singaling_nodes = pd.DataFrame({})
start_time = time.time()
for i in range(number_DTLEMBAS_models):
    print2log('Run DT-LEMBAS model '+str(i+1)+' to simulate with KO')
    mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(i+1)+'signal_state_dict_%s.pth'%(i+1))))
    drugLayer.load_state_dict(torch.load(os.path.join(data_path,modelPattern + str(i+1) + 'drugLayer_state_dict_%s.pth'%(i+1))))
    mod.eval()
    drugLayer.eval()
    # print2log('Node stimuli simulation...')
    with torch.no_grad():
        YhatFull = mod.signaling_network(Xin_combined)
        Yhat = mod.output_layer(YhatFull)
    # TF_all[i,:,:] = Yhat.detach().cpu()
    df = pd.DataFrame(Yhat.detach().cpu().numpy())
    df.columns = all_tfs
    df['TGFb'] = grid_Xin1[:,0].cpu().numpy() # TGFb stimuli quanity
    df['IFNa'] = grid_Xin2[:,0].cpu().numpy() # IFNa stimuli quanity
    df['knockout_node'] = knockout_node
    df['knockout_strength'] = knockout_strength
    df['type'] = 'stimuli_with_KO'
    df['model_no'] = i
    results_tf = pd.concat([results_tf, df], ignore_index=True).reset_index(drop=True)
    
    ## repeat by passing zero argument (untreated + knockout)
    # print2log('Untreated with KO simulation...')
    with torch.no_grad():
        Xin_untreated_ko = Xin_ko * knockout_strength
        YhatFull_ko = mod.signaling_network(Xin_untreated_ko)
        TF_ko = mod.output_layer(YhatFull_ko)
    df_ko = pd.DataFrame(TF_ko.detach().cpu().numpy())
    df_ko.columns = all_tfs
    df_ko['IFNa'] = 'untreated'
    df_ko['TGFb'] = 'untreated'
    df_ko['knockout_node'] = knockout_node
    df_ko['knockout_strength'] = knockout_strength
    df_ko['type'] = 'untreated_with_KO'
    df_ko['model_no'] = i
    results_tf = pd.concat([results_tf, df_ko], ignore_index=True).reset_index(drop=True)
    
    ## Also save completely untreated (no KO) for comparison
    TF, NetFull = mod(torch.zeros(1,drugTargets.shape[1],dtype=torch.double).to(device))
    df_empty = pd.DataFrame(TF.detach().cpu().numpy())
    df_empty.columns = all_tfs
    df_empty['IFNa'] = 'untreated'
    df_empty['TGFb'] = 'untreated'
    df_empty['knockout_node'] = 'none'
    df_empty['knockout_strength'] = 0
    df_empty['type'] = 'untreated_no_KO'
    df_empty['model_no'] = i
    results_tf = pd.concat([results_tf, df_empty], ignore_index=True).reset_index(drop=True)

mins, secs = divmod(time.time() - start_time, 60)
print2log("Simulation ran in: {:.0f} min {:.2f} sec".format(mins, secs))
print2log('Saving results...')
results_tf.to_csv(res_dir+name_stimulation+f'/TFs_all_conditions_expanded_KO_{knockout_node}.csv')

print2log("Finished!")