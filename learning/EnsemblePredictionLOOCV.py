import os

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
import torch
import torch.nn as nn

import plotnine as p9


# In[45]:




# In[46]:


import sys 
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
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
parser.add_argument('--outPattern', action='store',type = str, default='hepatocytes_dtlembas')
parser.add_argument('--number_of_models', action='store',type = int, default=10)
parser.add_argument('--batchSize', action='store',type = int, default=35)
parser.add_argument('--MoAFactor', action='store',type = float, default=0.1)
parser.add_argument('--spectralFactor', action='store', default=1e-3)
parser.add_argument('--maxIter', action='store',type = int, default=1000)
parser.add_argument('--noiseLevel', action='store',type = float, default=10)
parser.add_argument('--L2', action='store', default=1e-6)
parser.add_argument('--lamda', action='store',type = float, default=1e-9)
parser.add_argument('--ConvertToEmpProb', action='store',type = bool, default=False)
parser.add_argument('--inputAmplitude', action='store',type = float, default=3)
parser.add_argument('--projectionAmplitude', action='store',type = float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', action='store',type = int, default=90)
parser.add_argument('--LEMBAS_leak', action='store',type = float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', action='store', default=1e-5)
parser.add_argument('--drug_bn', action='store', default=0.2)
parser.add_argument('--drug_drop', action='store', default=0.1)
parser.add_argument('--timepoint', action='store',type=str, default='8 hour')
parser.add_argument('--test_dose', action='store',type=str, default='Middle')
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
number_of_models = args.number_of_models
timepoint = args.timepoint
test_dose = args.test_dose
res_dir = res_dir + test_dose


# In[51]:


n_cores = 12
utils.set_cores(n_cores)

seed = int(number_of_models) * (int(args.maxIter) + 1)
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


def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym,dim=0)
    x_square_sum = torch.sum(xm * xm,dim=0)
    y_square_sum = torch.sum(ym * ym,dim=0)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return r #torch.mean(r)

def numpy_pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    # Compute means for each column
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    # Subtract means (centering)
    xm = x - mx
    ym = y - my
    # Numerator: sum of element-wise products of centered vectors
    r_num = np.sum(xm * ym, axis=0)
    # Denominator: square root of the product of sum of squares of centered vectors
    r_den = np.sqrt(np.sum(xm * xm, axis=0) * np.sum(ym * ym, axis=0))
    # Pearson correlation for each column
    r = r_num / r_den
    return r


class DrugstoSignal(nn.Module):
    def __init__(self,mask, size1,size2,drugSim,momentum=0.6,drop=0.1,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.size_in = size1 ### drugs number
        self.size_out = size2 ### targets/signal size
        self.mask = mask.double().to(device)
        self.drugSim = drugSim.to(device)
        self.leak = 0.01

        # initialize weights and biases
        weights = torch.Tensor(size2,size1).double()
        nn.init.uniform_(weights, -0.3, 0.01) # weight init
        weights = torch.mul(weights.to(device),self.mask)
        ecfp4_weights = torch.Tensor(size1,size1).double()
        nn.init.uniform_(ecfp4_weights, 0., 1.)
        self.Wsim = nn.Parameter(ecfp4_weights, requires_grad=True)
        self.bn = nn.BatchNorm1d(num_features=size1, momentum=momentum, dtype=torch.double)
        self.dropout = nn.Dropout(drop)

        self.A = nn.Parameter(weights, requires_grad = True)

    def forward(self, x):
        W = torch.mul(self.drugSim, self.Wsim)
        A = torch.mul(self.A,self.mask)
        xhat = torch.mm(x,W)
        xhat = self.bn(xhat)
        xhat = self.dropout(xhat)
        xhat=xhat.transpose(0, 1)
        xhat = torch.mm(A, xhat)
        output = xhat.transpose(0, 1)
        return output


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


# Let's format it to fit with the necessary inputs to the SignalingModel:

# In[62]:


net = format_network(net, weight_label = weight_label, stimulation_label = stimulation_label, inhibition_label = inhibition_label)


# In[63]:


nodeNames = np.union1d(net.source.values,net.target.values)


# In[64]:


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


# In[65]:


### Split train and test set
print2log('Split train and test set...')
print2log('Kept ' + test_dose + ' as a test set!')
train_ind= inputInfo[~inputInfo.index.str.contains(test_dose)].index.tolist()
test_ind = np.setdiff1d(inputInfo.index.values, train_ind)
drugInput_train = drugInput.loc[train_ind,:]
drugInput_test = drugInput.loc[test_ind,:]
TFOutput_train = TFOutput.loc[train_ind,:]
TFOutput_test = TFOutput.loc[test_ind,:]
# X = torch.tensor(drugInput.values.copy(), dtype=torch.double).to(device)
# Y = torch.tensor(TFOutput.values, dtype=torch.double).to(device)
X = torch.tensor(drugInput_train.values.copy(), dtype=torch.double).to(device)
Y = torch.tensor(TFOutput_train.values, dtype=torch.double).to(device)
X_test = torch.tensor(drugInput_test.values.copy(), dtype=torch.double).to(device)
Y_test = torch.tensor(TFOutput_test.values, dtype=torch.double).to(device)

#### Random shuffle X
#X = X[torch.randperm(X.size()[0]),:]


# In[66]:


mask_dt =torch.tensor(np.array(drugTargets)).transpose(0,1)


# ### Define Parameters to Build and Train Model

# In[67]:


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
                   'uniform_max': 1/projection_amplitude_out, 'spectral_loss_factor': args.spectralFactor,'lamda':lamda}
spectral_radius_params = {'n_probes_spectral': 5, 'power_steps_spectral': 5, 'subset_n_spectral': 10}
target_spectral_radius = 0.8
hyper_params = {**lr_params, **other_params, **regularization_params, **spectral_radius_params} # fed into training function


# ### Initialize Model
# Next, let's initialize the model and format the inputs/outputs for running the model:

# In[68]:


mod = SignalingModel(net = net,
                     X_in = drugTargets,
                     y_out = TFOutput_train, 
                     projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                     bionet_params = bionet_params, 
                     dtype = torch.double, device = device, seed = seed)


# In[72]:


drugLayer = DrugstoSignal(mask_dt, 
                          mask_dt.shape[1],mask_dt.shape[0],
                          drugSim,momentum=args.drug_bn,drop=args.drug_drop,
                          device = device)
drugLayer = drugLayer.to(device)

TF_test_all = torch.zeros(number_of_models,Y_test.shape[0],Y_test.shape[1], dtype=torch.double)
TF_train_all = torch.zeros(number_of_models,Y.shape[0],Y.shape[1], dtype=torch.double)
for i in range(number_of_models):
    print2log('Load DT-LEMBAS model '+str(i+1)+' to make predictions')
    mod.load_state_dict(torch.load(os.path.join(data_path, outPattern +  str(i+1)+'signal_state_dict_%s.pth'%(i+1))))
    drugLayer.load_state_dict(torch.load(os.path.join(data_path,outPattern + str(i+1) + 'drugLayer_state_dict_%s.pth'%(i+1))))
    mod.eval()
    drugLayer.eval()
    TF_test, _ = mod(drugLayer(X_test))
    # TF_train, _ = mod(drugLayer(X))
    # store model prediction
    TF_test_all[i,:,:] = TF_test.detach().cpu()
    # TF_train_all[i,:,:] = TF_train.detach().cpu()

# Ensemble prediction
print2log('Ensemble prediction...')
Y_hat_test = torch.mean(TF_test_all,0)
# Y_hat = torch.mean(TF_train_all,0)

# Evaluate ensemble performance
# r_train = pearson_r(Y, Y_hat).detach().cpu().numpy()
r_test = pearson_r(Y_test, Y_hat_test.to(device)).detach().cpu().numpy()
# df_res = pd.DataFrame({'TF':TFOutput.columns,'train': r_train, 'test': r_test})
df_res = pd.DataFrame({'TF':TFOutput.columns,'r': r_test})
df_res['set'] = 'test'
df_res['fold'] = test_dose
df_res['time'] = timepoint
df_res.to_csv(res_dir+'/ensemble_performance_'+test_dose+'_'+timepoint+'.csv')