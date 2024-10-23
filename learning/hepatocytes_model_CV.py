import os

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
import torch
import pickle


# In[45]:




# In[46]:


import sys 
gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.model.train import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io


# In[47]:


import time
import argparse
import logging
from pathlib import Path
from utility import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='Cell-line simulation')
parser.add_argument('--DrugsIn', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store',type = str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', action='store',type = str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--infoPath', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--res_dir', action='store',type = str, default='../results/')
parser.add_argument('--model_type', action='store',type = int, default=4)
parser.add_argument('--outPattern', action='store',type = str, default='hepatocytes_dtlembas')
parser.add_argument('--no', action='store',type = int, default=0)
parser.add_argument('--batchSize', action='store',type = int, default=35)
parser.add_argument('--MoAFactor', action='store',type = float, default=0.1)
parser.add_argument('--spectralFactor', action='store', default=1e-3)
parser.add_argument('--maxIter', action='store',type = int, default=1000)
parser.add_argument('--noiseLevel', action='store',type = float, default=10)
parser.add_argument('--L2', action='store', default=1e-6)
parser.add_argument('--lamda', action='store',type = float, default=1e-7)
parser.add_argument('--ConvertToEmpProb', action='store',type = bool, default=False)
parser.add_argument('--inputAmplitude', action='store',type = float, default=3)
parser.add_argument('--projectionAmplitude', action='store',type = float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', action='store',type = int, default=90)
parser.add_argument('--LEMBAS_leak', action='store',type = float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', action='store', default=1e-5)
parser.add_argument('--drug_bn', action='store', default=0.2)
parser.add_argument('--drug_drop', action='store', default=0.1)
parser.add_argument('--timepoint', action='store',type=str, default='8 hour')
parser.add_argument('--num_folds', action='store',type=int, default=10)
args = parser.parse_args()
DrugsIn = args.DrugsIn
TargetsIn = args.TargetsIn
TFsOut = args.TFsOut
ChemicalSims = args.ChemicalSims
PknAnnotation = args.PknAnnotation
PKN = args.PKN
res_dir = args.res_dir
infoPath = args.infoPath
outPattern = args.outPattern
outPattern = outPattern + str(args.no)
timepoint = args.timepoint
num_folds = args.num_folds
res_dir = res_dir + 'CV/'


# In[51]:


n_cores = 12
utils.set_cores(n_cores)

seed = int(args.no) * (int(args.maxIter) + 1)
if seed:
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    utils.set_seeds(seed = seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = res_dir
if not os.path.isdir(data_path):
    os.mkdir(data_path)

projection_amplitude_in = args.inputAmplitude
projection_amplitude_out = args.projectionAmplitude
ConvertToEmpProb = args.ConvertToEmpProb

#Setup optimizer
batchSize = args.batchSize
MoAFactor = args.MoAFactor
spectralFactor = args.spectralFactor
maxIter = args.maxIter
L2 = args.L2
lamda = args.lamda

Path(res_dir).mkdir(parents=True, exist_ok=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'



# prior knowledge signaling network
net = pd.read_csv(PKN, sep = '\t', index_col = False).loc[:,['source','target','stimulation','inhibition']]
# ligand input and TF output
drugInput = pd.read_csv(DrugsIn, sep = '\t', index_col = 0)
drugSmiles = drugInput.columns.values
TFOutput = pd.read_csv(TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput = TFOutput.loc[drugInput.index,:]
# Load drugs information
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
mask_dt =torch.tensor(np.array(drugTargets)).transpose(0,1)
# other parameters
bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 
                 'exp_factor':50, 'tolerance': args.LEMBAS_tolerance, 
                 'leak':args.LEMBAS_leak} # fed directly to model

# training parameters
lr_params = {'max_iter': args.maxIter, 
             'learning_rate': 2e-3}
other_params = {'batch_size': args.batchSize, 'noise_level': args.noiseLevel, 'gradient_noise_level': 1e-9}
regularization_params = {'param_lambda_L2': args.L2, 'moa_lambda_L1': args.MoAFactor, 
                         'ligand_lambda_L2': 1e-5, 'uniform_lambda_L2': 1e-5, 
                   'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': args.spectralFactor}
spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 50, 'subset_n_spectral': 10}
target_spectral_radius = 0.8
hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params} # fed into training function

### Split train and test set
print2log('Split train and vailidaiton sets...')
X_all = drugInput.values.copy()
Y_all = TFOutput.values
# Assuming 'drugInput' is your dataframe
drug_names = drugInput.columns
# For each row, figure out which drug is used (non-zero concentration)
drug_used = drugInput.idxmax(axis=1)

if os.path.exists(res_dir+'valid_splits.pkl'):
    print2log('Loading valid splits...')
    with open(res_dir+'valid_splits.pkl', 'rb') as f:
        valid_splits = pickle.load(f)
else:
    print2log('Creating and saving valid splits...')
    # Get the valid splits
    valid_splits = custom_kfold(drug_used, n_splits=num_folds)
    # Save the list to a file
    with open(res_dir+'valid_splits.pkl', 'wb') as f:
        pickle.dump(valid_splits, f)

## Train model
print2log('Begin training...')
perform_cross_validation(valid_splits, X_all, Y_all ,net,drugTargets,TFOutput,
                             projection_amplitude_in, projection_amplitude_out, weight_label, source_label, target_label, drugSim, bionet_params,
                            hyper_params, device, data_path, outPattern, target_spectral_radius,
                            mask_dt, args, res_dir, batchSize,
                            seed,
                            shuffling=False)

### Now train shuffled model
print2log('Begin Shuffled training...')
perform_cross_validation(valid_splits, X_all, Y_all ,net,drugTargets,TFOutput,
                             projection_amplitude_in, projection_amplitude_out, weight_label, source_label, target_label, drugSim, bionet_params,
                            hyper_params, device, data_path, 'shuffledX_'+outPattern, target_spectral_radius,
                            mask_dt, args, res_dir, batchSize,
                            seed,
                            shuffling=True)

