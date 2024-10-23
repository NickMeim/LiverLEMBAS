import torch
import numpy as np
import matplotlib.pyplot as plt
import bionetworkWithDrugs as bionetwork
from pathlib import Path
import plotting
import pandas as pd
from utility import *
import argparse
import logging

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
parser.add_argument('--batchSize', action='store',type = int, default=25)
parser.add_argument('--MoAFactor', action='store',type = float, default=0.1)
parser.add_argument('--spectralFactor', action='store', default=1e-3)
parser.add_argument('--maxIter', action='store',type = int, default=2000)
parser.add_argument('--noiseLevel', action='store',type = float, default=10)
parser.add_argument('--L2', action='store', default=1e-6)
parser.add_argument('--lamda', action='store',type = float, default=0.005)
parser.add_argument('--ConvertToEmpProb', action='store',type = bool, default=False)
parser.add_argument('--inputAmplitude', action='store',type = float, default=3)
parser.add_argument('--projectionAmplitude', action='store',type = float, default=1.2)
parser.add_argument('--LEMBAS_iterations', action='store',type = int, default=120)
parser.add_argument('--LEMBAS_leak', action='store',type = float, default=0.01)
parser.add_argument('--LEMBAS_targetPrecision', action='store', default=1e-6)
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
model_type = int(args.model_type)
outPattern = args.outPattern
outPattern = outPattern + str(args.no)

timepoint = args.timepoint
test_dose = args.test_dose
res_dir = res_dir + test_dose

inputAmplitude = args.inputAmplitude
projectionAmplitude = args.projectionAmplitude
ConvertToEmpProb = args.ConvertToEmpProb

#Setup optimizer
batchSize = args.batchSize
MoAFactor = args.MoAFactor
spectralFactor = args.spectralFactor
maxIter = args.maxIter
noiseLevel = args.noiseLevel
L2 = args.L2
lamda = args.lamda

Path(res_dir).mkdir(parents=True, exist_ok=True)

#Load network
print2log('Loading network...')
networkList, nodeNames, modeOfAction = bionetwork.loadNetwork(PKN)
annotation = pd.read_csv(PknAnnotation, sep='\t')
uniprot2gene = dict(zip(annotation['code'], annotation['name']))
bionetParams = bionetwork.trainingParameters(iterations = args.LEMBAS_iterations, clipping=1, targetPrecision=args.LEMBAS_targetPrecision, leak=args.LEMBAS_leak)
spectralCapacity = np.exp(np.log(1e-2)/bionetParams['iterations'])

# Load data
print2log('Loading data...')
drugInput = pd.read_csv(DrugsIn, sep='\t', low_memory=False, index_col=0)
drugSmiles = drugInput.columns.values
drugTargets = pd.read_csv(TargetsIn, sep='\t', low_memory=False, index_col=0)
TFOutput = pd.read_csv(TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput = TFOutput.loc[drugInput.index,:]
inputInfo = pd.read_csv(infoPath, sep='\t', low_memory=False, index_col=0)
inputInfo = inputInfo[inputInfo['time']==timepoint]
TFOutput = TFOutput.loc[inputInfo.index,:]
drugInput = drugInput.loc[inputInfo.index,:]

#Subset input and output to intersecting nodes
druginName = drugInput.columns.values
inName = drugTargets.columns.values
outconds = drugInput.index.values
outName = TFOutput.columns.values
inName = np.intersect1d(nodeNames, inName)
outName = np.intersect1d(nodeNames, outName)
inNameGene = [uniprot2gene[x] for x in inName]
outNameGene = [uniprot2gene[x] for x in outName]
TFOutput = TFOutput.loc[outconds,outName]
drugTargets = drugTargets.loc[:,inName]
if ConvertToEmpProb==True:
    print2log('Convereted to Emprirical probabilities')
    TFOutput = 1/(1+np.exp(-TFOutput))

drugTargets = drugTargets.loc[drugInput.columns.values,:]
drugSim = pd.read_csv(ChemicalSims,index_col=0)
drugSim = drugSim.loc[drugInput.columns.values,drugInput.columns.values]
drugSim = torch.tensor(drugSim.values.copy(), dtype=torch.double)

### Split train and test set
print2log('Split train and test set...')
print2log('Kept ' + test_dose + ' as a test set!')
train_ind= inputInfo[~inputInfo.index.str.contains(test_dose)].index.tolist()
test_ind = np.setdiff1d(inputInfo.index.values, train_ind)
drugInput_train = drugInput.loc[train_ind,:]
drugInput_test = drugInput.loc[test_ind,:]
TFOutput_train = TFOutput.loc[train_ind,:]
TFOutput_test = TFOutput.loc[test_ind,:]
X = torch.tensor(drugInput_train.values.copy(), dtype=torch.double)
Y = torch.tensor(TFOutput_train.values, dtype=torch.double)
X_test = torch.tensor(drugInput_test.values.copy(), dtype=torch.double)
Y_test = torch.tensor(TFOutput_test.values, dtype=torch.double)

#### Random shuffle X
#X = X[torch.randperm(X.size()[0]),:]

sampleName = drugInput_train.index.values
print2log('Initializing model...')
mask_dt =torch.tensor(np.array(drugTargets)).transpose(0,1)
if model_type==1:
    model = bionetwork.model(networkList, nodeNames, inName, modeOfAction, inputAmplitude, projectionAmplitude, mask_dt,druginName, inName, outName, bionetParams, torch.double,
                             drugSim=None, useMask=True, useDrugs=False)
elif model_type==2:
    model = bionetwork.model(networkList, nodeNames, inName, modeOfAction, inputAmplitude, projectionAmplitude, mask_dt,druginName, inName, outName, bionetParams, torch.double,
                             drugSim=None, useMask=False, useDrugs=False)
elif model_type==3:
    model = bionetwork.model(networkList, nodeNames, inName, modeOfAction, inputAmplitude, projectionAmplitude, mask_dt,druginName, inName, outName, bionetParams, torch.double,
                             drugSim=None, useMask=True, useDrugs=True)
elif model_type==5:
    model = bionetwork.model(networkList, nodeNames, inName, modeOfAction, inputAmplitude, projectionAmplitude, mask_dt,druginName, inName, outName, bionetParams, torch.double,
                             drugSim=drugSim, useMask=True, useDrugs=True,onlyECFP4Similarity=True)
else:
    model = bionetwork.model(networkList, nodeNames, inName, modeOfAction, inputAmplitude, projectionAmplitude, mask_dt,druginName, inName, outName, bionetParams, torch.double,
                             drugSim=drugSim,useMask=True,useDrugs=True)

model.inputLayer.weights.requires_grad = False
model.network.preScaleWeights()



criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1,weight_decay=0) ###Changed to using RMSprop
resetState = optimizer.state.copy()

print2log('Begin training...')
mLoss = criterion(torch.mean(Y, dim=0)*torch.ones(Y.shape), Y)
print2log(mLoss)


stats = plotting.initProgressObject(maxIter)
N = X.shape[0]
curState = torch.rand((X.shape[0], model.network.bias.shape[0]), dtype=torch.double, requires_grad=False) # model.network.bias.shape[0]

e = 0
for e in range(e, maxIter):
    curLr = bionetwork.oneCycle(e, maxIter, maxHeight = 1e-3, minHeight = 1e-8, peak = 500)
    optimizer.param_groups[0]['lr'] = curLr

    curLoss = []
    curEig = []
    trainloader = bionetwork.getSamples(N, batchSize)
    for dataIndex in trainloader:
        model.train()
        model.network.weights.data = model.network.weights.data + 1e-8 * torch.randn(model.network.weights.shape) #breaks potential symmetries
        optimizer.zero_grad()

        dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
        dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])

        Yin = model.drugLayer(dataIn)
        Yin = model.inputLayer(Yin)
        Yin = Yin + noiseLevel * curLr * torch.randn(Yin.shape)
        YhatFull = model.network(Yin)
        Yhat = model.projectionLayer(YhatFull)

        curState[dataIndex, :] = YhatFull.detach()

        fitLoss = criterion(dataOut, Yhat)

        signConstraint = MoAFactor * torch.sum(torch.abs(model.network.weights[model.network.getViolations(model.network.weights)]))
        stateLoss = 1e-5 * bionetwork.uniformLoss(curState, dataIndex, YhatFull, maxConstraintFactor = 50)
        biasLoss = L2 * torch.sum(torch.square(model.network.bias))
        weightLoss = L2 * (torch.sum(torch.square(model.network.weights)) + torch.sum(1/(torch.square(model.network.weights) + 0.5)))
        if (model_type in [3,4]):
            weightLoss = weightLoss + lamda * torch.sum(torch.square(model.drugLayer.Wsim-torch.eye(model.drugLayer.Wsim.shape[0]))) # added to regularize ecfp4
        weightLoss = weightLoss + L2 * (torch.sum(torch.square(model.drugLayer.A)) + torch.sum(1 / (torch.square(model.drugLayer.A) + 0.5))) 
        projectionLoss = 1e-6 * torch.sum(torch.square(model.projectionLayer.weights - projectionAmplitude))
        spectralRadiusLoss, spectralRadius = bionetwork.spectralLoss(model, YhatFull, model.network.weights, expFactor = 10)

        loss = fitLoss + signConstraint + biasLoss + weightLoss + spectralFactor * spectralRadiusLoss + stateLoss + projectionLoss
        loss.backward()
        optimizer.step()

        curEig.append(spectralRadius.item())
        curLoss.append(fitLoss.item())

        stats = plotting.storeProgress(stats, e, curLoss, curEig, curLr, violations=torch.sum(model.network.getViolations(model.network.weights)).item())

    if e % 50 == 0:
        plotting.printStats(e, stats)

    if np.logical_and(e % 100 == 0, e>0):
        optimizer.state = resetState.copy()



stats = plotting.finishProgress(stats)
model.eval()
Yhat, YhatFull = model(X)
torch.save(model, res_dir+'/'+outPattern+'.pt')
print2log('Saved model!')

#%%
plt.rcParams["figure.figsize"] = (9,6)
plt.figure()


T = np.array(range(stats['loss'].shape[0]))
plotting.shadePlot(T, plotting.movingaverage(stats['loss'], 5), plotting.movingaverage(stats['lossSTD'], 10))
plt.plot([0, len(T)], np.array([1, 1])*mLoss.item(), 'black', linestyle='--')
plt.xlim([0, len(T)])
plt.ylim(bottom=1e-3)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.savefig(res_dir+'/fig1_'+outPattern+'.png',dpi=600)

plt.figure()
plt.scatter(Yhat.detach().numpy(), Y.detach().numpy(), alpha=0.2)
plotting.lineOfIdentity()
plotting.addCorrelation(Yhat, Y)
plt.xlabel('Fit')
plt.ylabel('Data')
plt.gca().axis('equal')
plt.gca().set_xticks([0, 0.5, 1])
plt.gca().set_yticks([0, 0.5, 1])
plt.savefig(res_dir+'/fig2_'+outPattern+'.png',dpi=600)

plt.figure()
srModel = bionetwork.getAllSpectralRadius(model, YhatFull)

plt.hist(srModel)
plt.ylabel('SR model')
plt.savefig(res_dir+'/fig3_'+outPattern+'.png')
plt.rcParams["figure.figsize"] = (12,10)
plt.figure()
rank = plotting.compareAllTFs(Yhat, Y, outNameGene)
plt.savefig(res_dir+'/fig4_'+outPattern+'.png',dpi=600)


### Calculate per TF correlation for training set
r_train = pearson_r(Y, Yhat).detach().numpy()

### Make predictions for the test set
model.eval()
Yhat_test, YhatFull_test = model(X_test)
r_test = pearson_r(Y, Yhat).detach().numpy()

df_res = pd.DataFrame({'TF':TFOutput.columns,'train': r_train, 'test': r_test})
df_res['fold'] = test_dose
df_res['time'] = timepoint
df_res['model_no'] = args.no
df_res.to_csv(res_dir+'/performance_'+outPattern +'.csv')
