import os
import numpy as np
import pandas as pd
import torch
import sys 
gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
from sklearn.cross_decomposition import PLSRegression
import argparse
import logging
from pathlib import Path
from utility import *
import pyreadr
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

class PLSRTransform(nn.Module):
    def __init__(self, X_coefs, X_mean, dtype=torch.double,device="cuda"):
        super(PLSRTransform, self).__init__()
        # Convert numpy arrays to torch tensors
        self.X_coefs = torch.tensor(X_coefs, dtype=dtype).to(device)
        self.X_mean = torch.tensor(X_mean, dtype=dtype).to(device)
    def forward(self, X):
        # Center the input X
        X_centered = X - self.X_mean
        # Project X onto the latent space (scores)
        Y_pred = torch.matmul(X_centered, self.X_coefs)
        return Y_pred

### Initialize the parsed arguments
parser = argparse.ArgumentParser(prog='Cell-line simulation')
parser.add_argument('--DrugsIn', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store',type = str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', action='store',type = str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
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
# parser.add_argument('--X_invitro_path', action='store',help='file to load pre-processed (replicate merged) in vitro molecular features',default = 'Xm_hepg2.csv')
parser.add_argument('--X_invivo_path', action='store', help='file to load pre-processed in vivo molecular features (rds)',default = '../data/Xh_tfs_transformed.rds')
parser.add_argument('--Y_invivo_path', action='store', help='file to load in vivo phenotypes (csv)',default = '../data/Yh.csv')
parser.add_argument('--num_LVs', action='store', type=int,help='number of latent variables in the PLSR model',default=8)
parser.add_argument('--Wextra_path', action='store', type=str,
                    help='The path to load the extra latent variables (file in rds format)',
                    default='../data/Wtfs_uniprot_kostrzewski_extra.rds')
# 'C:/Users/nmeim/Documents/LiverModeling/FattyLiverModeling/results/Wtfs_uniprot_kostrzewski_extra.rds'
parser.add_argument('--keep_lv', action='store', type=int,help='The extra latent variable of interest to keep [zero-indexed]',default=0)
parser.add_argument('--learning_rate', action='store',type=float, default=300.)
parser.add_argument('--optimization_iterations', action='store', default=2000) # 500 for each phenotype separately
args = parser.parse_args()
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
num_LVs = int(args.num_LVs)
# X_invitro_path = args.X_invitro_path
X_invivo_path = args.X_invivo_path
Y_invivo_path = args.Y_invivo_path
Wextra_path = args.Wextra_path
keep_lv = int(args.keep_lv)
optimization_iterations = int(args.optimization_iterations)
learning_rate = args.learning_rate
number_DTLEMBAS_models = int(args.number_DTLEMBAS_models)

n_cores = 12
utils.set_cores(n_cores)
# seed = int(args.no) * (int(args.maxIter) + 1)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log('Running on {}'.format(device))
data_path = res_dir
if not os.path.isdir(data_path):
    os.mkdir(data_path)

print2log('Loading input data...')
Xh = pyreadr.read_r(X_invivo_path)
Xh = Xh[None]
Xh = Xh.to_numpy()
Yh = pd.read_csv(Y_invivo_path,index_col=0)
Yh = Yh.to_numpy()
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
all_drugs = drugInput.columns

## Load extra LVs
W = pyreadr.read_r(Wextra_path)
W = W[None]
W = W.loc[TFOutput.columns,:]
## conver to numpy
W = W.to_numpy()
extraLV = W[:,keep_lv]
extraLV = torch.tensor(extraLV, dtype=torch.double).to(device)
extraLV = extraLV.unsqueeze(0)
extraLV = torch.nn.Parameter(extraLV, requires_grad=True)

print2log('Initialize DT-LEMBAS and optimization parameters, and PLSR model...')
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

nodeNames = np.array(list(mod.node_idx_map.keys()))
num_tfs = mod.y_out.shape[1]
all_tfs = mod.y_out.columns.values
all_nodes = np.array(list(mod.node_idx_map.keys()))

## train PLSR model
plsr_model = PLSRegression(n_components=num_LVs,scale=False)
plsr_model.fit(Xh, Yh)
X_coefs = plsr_model.coef_
X_mean = plsr_model._x_mean # The mean of X (used for centering)
plsr_transform = PLSRTransform(X_coefs, X_mean,device=device).to(device)
plsr_transform.eval()


print2log('Begin optimization...')
Path(res_dir+'optimal_perturbations/').mkdir(parents=True, exist_ok=True)
# loss_fn = torch.nn.MSELoss()
# loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05).to(device)
# all_losses = []
# all_r = []
for i in range(number_DTLEMBAS_models):
    print2log('Run DT-LEMBAS model '+str(i+1)+' to simulate KOs')
    Xdrugs = drugInput[drugInput['Control'] != 0].to_numpy()
    Xdrugs = np.median(Xdrugs, axis=0, keepdims=True)
    Xdrugs = torch.tensor(Xdrugs).double().to(device)
    mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(i+1)+'_alldata_signal_state_dict_%s.pth'%(i+1))))
    drugLayer.load_state_dict(torch.load(os.path.join(data_path,modelPattern + str(i+1) + '_alldata_drugLayer_state_dict_%s.pth'%(i+1))))
    mod.eval()
    drugLayer.eval()

    ## Calculate control TF
    Xin_ctrl = drugLayer(Xdrugs)
    TF_ctrl, _ = mod(Xin_ctrl)
    TF_ctrl = torch.nn.Parameter(TF_ctrl, requires_grad=True)

    # Initialize optimizable X
    X= torch.nn.Parameter(Xdrugs, requires_grad=True)
    optimizer = torch.optim.Adam([X],lr= learning_rate,weight_decay=0)
    #cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # Wasserstein  = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    for j in range(optimization_iterations):
        optimizer.zero_grad()
        Xin = drugLayer(X)
        TF, _ = mod(Xin)
        Yhat = plsr_transform(TF)
        fitLoss = torch.mean(torch.square(Yhat - torch.tensor([[8,4]],device='cuda')))
        DeltaTF = TF-TF_ctrl
        w1 = extraLV.norm(p=2, dim=1, keepdim=True)
        w2 = DeltaTF.norm(p=2, dim=1, keepdim=True)
        cosineLoss = torch.mm(extraLV, DeltaTF.t()) / (w1 * w2.t()).clamp(min=1e-6)
        # cosineLoss = cosine(extraLV,DeltaTF)
        signLoss = torch.sigmoid(-500 * X).sum() # to control the negative values
        L1Loss =  1e-3 * torch.sum(torch.abs(X))
        loss = fitLoss + signLoss + L1Loss
        # torch.mean(torch.sum(torch.square(DeltaTF-extraLV),dim=1)) 

        loss.backward()
        optimizer.step()

        # all_losses.append(loss.item())
        # all_r.append(np.nanmean(r.detach().cpu().numpy()))
        if (j + 1) % 200 == 0 or j == 0 or (j + 1) == optimization_iterations:
            print2log('DT-LEMBAS model {:.0f}, iteration={:.0f} out of {:.0f}, loss={:.4f}, cosine similarity={:.4f}, MAS={:.2f}, fibrosis={:.2f}'.format(i, j+1,optimization_iterations,loss.item(),cosineLoss.item(),Yhat[:,0].item(),Yhat[:,1].item()))
        #scheduler.step()
    
    print2log('Saving optimal perturbations for model '+str(i+1)+'...')
    optimal_drugs = pd.DataFrame(X.detach().cpu().numpy())
    optimal_drugs.columns = all_drugs
    optimal_drugs.to_csv(res_dir+'optimal_perturbations/optimal_drugs_'+str(i+1)+'.csv')

    Xin = drugLayer(X)
    TF, NetFull = mod(Xin)

    print2log('Saving TF activity for model '+str(i+1)+'...')
    TF = TF.detach().cpu().numpy()
    TF = pd.DataFrame(TF)
    TF.columns = TFOutput.columns
    print2log('Saving node activity for model '+str(i+1)+'...')
    NetFull = NetFull.detach().cpu().numpy()
    NetFull = pd.DataFrame(NetFull)
    NetFull.columns = nodeNames
    TF.to_csv(res_dir+'optimal_perturbations/TF_'+str(i+1)+'.csv')
    NetFull.to_csv(res_dir+'optimal_perturbations/NetFull_'+str(i+1)+'.csv')




