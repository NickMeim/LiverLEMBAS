import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

parser = argparse.ArgumentParser(prog='Linear interpolation/extrapolation simulation')
parser.add_argument('--DrugsIn', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store',type = str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', action='store',type = str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--infoPath', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--res_dir', action='store',type = str, default='../results/')
parser.add_argument('--outPattern', action='store',type = str, default='hepatocytes_linear_baseline')
parser.add_argument('--ConvertToEmpProb', action='store',type = bool, default=False)
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
timepoint = args.timepoint
test_dose = args.test_dose
res_dir = res_dir + test_dose
outPattern = args.outPattern

data_path = res_dir
if not os.path.isdir(data_path):
    os.mkdir(data_path)

ConvertToEmpProb = args.ConvertToEmpProb
Path(res_dir).mkdir(parents=True, exist_ok=True)

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


# prior knowledge signaling network
net = pd.read_csv(PKN, sep = '\t', index_col = False).loc[:,['source','target','stimulation','inhibition']]
# ligand input and TF output
drugInput = pd.read_csv(DrugsIn, sep = '\t', index_col = 0)
drugSmiles = drugInput.columns.values
TFOutput = pd.read_csv(TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput = TFOutput.loc[drugInput.index,:]
# Load drugs information
inputInfo = pd.read_csv(infoPath, sep='\t', low_memory=False, index_col=0)
inputInfo = inputInfo[inputInfo['time']==timepoint]
TFOutput = TFOutput.loc[inputInfo.index,:]
drugInput = drugInput.loc[inputInfo.index,:]
nodeNames = np.union1d(net.source.values,net.target.values)

#Subset input and output to intersecting nodes
druginName = drugInput.columns.values
outconds = drugInput.index.values
outName = TFOutput.columns.values
outName = np.intersect1d(nodeNames, outName)
TFOutput = TFOutput.loc[outconds,outName]
if ConvertToEmpProb==True:
    print2log('Convereted to Emprirical probabilities')
    TFOutput = 1/(1+np.exp(-TFOutput))

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

### Begin linear interpolation/extrapolation for every TF for every drug

# --- 1) identify drugs with ≥2 doses in train and ≥1 in test
train_counts = drugInput_train.apply(lambda c: c.dropna().nunique())
test_counts  = drugInput_test .apply(lambda c: c.dropna().nunique())
valid_drugs  = train_counts[train_counts > 2].index.intersection(
                 test_counts[test_counts > 1].index)

TFs = TFOutput_train.columns
pred_matrix = pd.DataFrame(index=valid_drugs, columns=TFs, dtype=float)
true_matrix = pd.DataFrame(index=valid_drugs, columns=TFs, dtype=float)

# --- 2) per‐drug / per‐dose interpolation with tqdm progress bar
for drug in tqdm(valid_drugs,
                 desc="Training & predicting per-drug models",
                 unit="drug"):
    # rows with a dose value for this drug
    tr_mask = drugInput_train[drug]!=0
    te_mask = drugInput_test[drug]!=0
    x_tr = drugInput_train.loc[tr_mask, drug].values.reshape(-1, 1)
    y_tr = TFOutput_train.loc[tr_mask, :].values
    x_te = drugInput_test.loc[te_mask, drug].values.reshape(-1, 1)
    y_te = TFOutput_test.loc[te_mask, :].values
    lr = LinearRegression().fit(x_tr, y_tr)
    y_pred = lr.predict(x_te)
    pred_matrix.loc[drug, :] = y_pred
    true_matrix.loc[drug, :] = y_te

# --- 3) per-TF Pearson correlations on the test set
results = []
for tf in TFs:
    r, p = pearsonr(
        true_matrix.loc[:, tf].values,
        pred_matrix.loc[:, tf].values
    )
    results.append({'TF': tf, 'pearson_r': r, 'p_value': p})

corr_df = pd.DataFrame(results)
corr_df['dose'] = test_dose
if test_dose == 'Middle':
    corr_df['method'] = 'interpolation'
else:    
    corr_df['method'] = 'extrapolation'
corr_df.to_csv(os.path.join(res_dir, 'tf_pearson_correlations.csv'), index=False)
print2log(corr_df.pearson_r.mean())
print2log(f"Saved per-TF Pearson correlations to {res_dir}/tf_pearson_correlations.csv")