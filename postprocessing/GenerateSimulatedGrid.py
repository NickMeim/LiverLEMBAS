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
parser.add_argument('--num_steps', action='store',type = float, default=201)
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected signaling nodes to predict and save their activity',
                    default=['P36897','P17181'])

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

Xin = torch.eye(len(all_nodes),dtype=torch.double).to(device)
Xin_1 = Xin[subset_nodes_inds[0],:].unsqueeze(0)
Xin_2 = Xin[subset_nodes_inds[1],:].unsqueeze(0)

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
# Combine the two inputs for the model
Xin_combined = Xin_1_grid + Xin_2_grid


print2log('Begin combination stimuli simulation...')
name_stimulation = 'combo_simulation_expanded'
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
    ## In the indices of TFs put the values of Yhat
    tf_inds = np.array([mod.node_idx_map[tf] for tf in all_tfs])
    # print2log('Node stimuli simulation...')
    with torch.no_grad():
        YhatFull = mod.signaling_network(Xin_combined)
        Yhat = mod.output_layer(YhatFull)
        YhatFull[:,tf_inds] = Yhat
    YhatFull = YhatFull.detach().cpu().numpy()
    YhatFull = pd.DataFrame(YhatFull)
    YhatFull.columns = nodeNames
    YhatFull.to_csv(res_dir+name_stimulation+'/generatedSignalingSimulation'+str(i+1)+'.csv',index=False)

mins, secs = divmod(time.time() - start_time, 60)
print2log("Simulation ran in: {:.0f} min {:.2f} sec".format(mins, secs))

## Save also Xin_combined
Xin_combined = pd.DataFrame(Xin_combined.detach().cpu().numpy())
Xin_combined.columns = nodeNames
Xin_combined = Xin_combined.loc[:,subset_nodes]
Xin_combined.to_csv(res_dir+name_stimulation+'/StimulationGrid.csv',index=False)


print2log("Finished!")
