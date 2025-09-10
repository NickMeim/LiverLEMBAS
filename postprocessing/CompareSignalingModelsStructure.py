import os
import numpy as np
import pandas as pd
import torch
import sys 
# gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
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
parser.add_argument('--res_dir', action='store',type = str, default='../results/') #'../results/2 hour/allmodels/'
parser.add_argument('--number_DTLEMBAS_models', action='store',type = int, default=50)
parser.add_argument('--modelPattern', action='store',type = str, default='hepatocytes_dtlembas')
parser.add_argument('--ConvertToEmpProb', action='store',type = bool, default=False)
parser.add_argument('--max_LEMBAS_iterations', action='store',type = int, default=90)
parser.add_argument('--LEMBAS_leak', action='store',type = float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', action='store', default=1e-5)
parser.add_argument('--inputAmplitude', action='store',type = float, default=3)
parser.add_argument('--projectionAmplitude', action='store',type = float, default=1.2)
parser.add_argument('--MoAFactor', action='store',type = float, default=0.1)
parser.add_argument('--spectralFactor', action='store', default=1e-3)
parser.add_argument('--drug_bn', action='store', default=0.2)
parser.add_argument('--drug_drop', action='store', default=0.1)
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected signaling nodes to predict and save their activity',
                    default=['P42226','P42229','P51692','P40763','P52630','P42224','Q14765',
                             'Q15797','Q15796','P84022','Q13485','Q99717','O43541',
                             'P17181','P48551','P15260','P38484',
                             'P10914','Q00978','Q14653',
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
projection_amplitude_in = args.inputAmplitude
projection_amplitude_out = args.projectionAmplitude
ConvertToEmpProb = args.ConvertToEmpProb
number_DTLEMBAS_models = args.number_DTLEMBAS_models
selected_signaling_nodes = np.array(args.selected_signaling_nodes)

n_cores = 12
utils.set_cores(n_cores)
# seed = int(args.no) * (int(args.maxIter) + 1)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log('Running on {}'.format(device))

print2log('Loading input data...')
# prior knowledge signaling network
net = pd.read_csv(PKN, sep = '\t', index_col = False).loc[:,['source','target','stimulation','inhibition']]
# ligand input and TF output
drugInput = pd.read_csv(DrugsIn, sep = '\t', index_col = 0)
drugSmiles = drugInput.columns.values
TFOutput = pd.read_csv(TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput = TFOutput.loc[drugInput.index,:]
stimulation_label = 'stimulation'
inhibition_label = 'inhibition'
weight_label = 'mode_of_action'
source_label = 'source'
target_label = 'target'
net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)
nodeNames = np.union1d(net.source.values,net.target.values)

all_drugTargets_list = []
nodeNames_list = []
all_nodes_list = []
all_tfs_list = []
for timepoint in ['2 hour','8 hour','24 hour']:
    print2log('Processing timepoint {}'.format(timepoint))
    # Load drugs information
    drugTargets = pd.read_csv(TargetsIn, sep='\t', low_memory=False, index_col=0)
    inputInfo = pd.read_csv(infoPath, sep='\t', low_memory=False, index_col=0)
    inputInfo = inputInfo[inputInfo['time']==timepoint]
    TFOutput_tmp = TFOutput.loc[inputInfo.index,:]
    drugInput_tmp = drugInput.loc[inputInfo.index,:]
    #Subset input and output to intersecting nodes
    druginName = drugInput_tmp.columns.values
    inName = drugTargets.columns.values
    outconds = drugInput_tmp.index.values
    outName = TFOutput_tmp.columns.values
    inName = np.intersect1d(nodeNames, inName)
    outName = np.intersect1d(nodeNames, outName)
    TFOutput_tmp = TFOutput_tmp.loc[outconds,outName]
    drugTargets = drugTargets.loc[:,inName]
    if ConvertToEmpProb==True:
        print2log('Convereted to Emprirical probabilities')
        TFOutput_tmp = 1/(1+np.exp(-TFOutput_tmp))
    drugTargets = drugTargets.loc[drugInput_tmp.columns.values,:]
    drugSim = pd.read_csv(ChemicalSims,index_col=0)
    drugSim = drugSim.loc[drugInput_tmp.columns.values,drugInput_tmp.columns.values]
    drugSim = torch.tensor(drugSim.values.copy(), dtype=torch.double)
    all_drugs = drugInput.columns
    ### Make tensors and initialize model
    X = torch.tensor(drugInput_tmp.values.copy(), dtype=torch.double).to(device)
    Y = torch.tensor(TFOutput_tmp.values, dtype=torch.double).to(device)

    mask_dt =torch.tensor(np.array(drugTargets)).transpose(0,1)
    bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 
                    'exp_factor':50, 'tolerance': args.LEMBAS_tolerance, 
                    'leak':args.LEMBAS_leak}
    mod = SignalingModel(net = net,
                        X_in = drugTargets,
                        y_out = TFOutput_tmp, 
                        projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                        weight_label = weight_label, source_label = source_label, target_label = target_label,
                        bionet_params = bionet_params, 
                        dtype = torch.double, device = device)
    drugLayer = DrugstoSignal(mask_dt, 
                            mask_dt.shape[1],mask_dt.shape[0],
                            drugSim,momentum=args.drug_bn,drop=args.drug_drop,
                            device = device)
    drugLayer = drugLayer.to(device)

    nodeNames = np.array(list(mod.node_idx_map.keys()))
    num_tfs = mod.y_out.shape[1]
    all_tfs = mod.y_out.columns.values
    all_nodes = np.array(list(mod.node_idx_map.values()))

    # append lists for comparison
    all_drugTargets_list.append(drugTargets)
    all_nodes_list.append(all_nodes)
    all_tfs_list.append(all_tfs)
    nodeNames_list.append(nodeNames)

print2log('Make all comparisons for targets')
for target_df_1 in all_drugTargets_list:
    for target_df_2 in all_drugTargets_list:
        print(target_df_1.shape, target_df_2.shape, target_df_1.equals(target_df_2))

print2log('Make all comparisons for nodes')
for node1 in all_nodes_list:
    for node2 in all_nodes_list:
        print(node1.shape, node2.shape, np.all(node1 == node2))

print2log('Make all comparisons for TFs')
for tf1 in all_tfs_list:
    for tf2 in all_tfs_list:
        print(tf1.shape, tf2.shape, np.all(tf1 == tf2))

print2log('Make all comparisons for nodeNames')
for nodeName1 in nodeNames_list:
    for nodeName2 in nodeNames_list:
        print(nodeName1.shape, nodeName2.shape, np.all(nodeName1 == nodeName2))
