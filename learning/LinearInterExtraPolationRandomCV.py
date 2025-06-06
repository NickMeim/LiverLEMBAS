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
from sklearn.model_selection import KFold
import pickle

def custom_kfold(drug_used, n_splits=10):
    # Initializing list to store the splits
    valid_splits = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    # Iterating over the splits
    for train_idx, val_idx in kf.split(drug_used):
        train_drugs = drug_used.iloc[train_idx].unique()
        val_drugs = drug_used.iloc[val_idx].unique()
        # Ensure every drug appears at least once in the training set
        if set(val_drugs) - set(train_drugs):
            # If some drug(s) is missing in training, skip this fold
            continue
        else:
            valid_splits.append((train_idx, val_idx))
    # Ensure we have enough valid splits
    while len(valid_splits) < n_splits:
        for train_idx, val_idx in kf.split(drug_used):
            train_drugs = drug_used.iloc[train_idx].unique()
            val_drugs = drug_used.iloc[val_idx].unique()
            
            if set(val_drugs) - set(train_drugs):
                continue
            else:
                valid_splits.append((train_idx, val_idx))
                
            if len(valid_splits) == n_splits:
                break
    return valid_splits

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
timepoint = args.timepoint
if timepoint != '8 hour':
    res_dir = res_dir + timepoint + '/'
res_dir = res_dir + 'CV/'
outPattern = args.outPattern
num_folds = args.num_folds

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
print2log('Get/Prepare CV splits...')
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



### Begin linear interpolation/extrapolation for every TF for every drug in every fold
TFs = TFOutput.columns
fold_results = pd.DataFrame({})
for fold_idx, (train_idx, test_idx) in enumerate(valid_splits):
    print(f"Starting fold {fold_idx + 1}/{len(valid_splits)}...")
    # Split the data for this fold
    drugInput_train = drugInput.iloc[train_idx,:]
    drugInput_test  = drugInput.iloc[test_idx,:]
    TFOutput_train = TFOutput.iloc[train_idx,:]
    TFOutput_test = TFOutput.iloc[test_idx,:]
    # --- 1) identify drugs with ≥2 doses in train and ≥1 in test
    train_counts = drugInput_train.apply(lambda c: c.dropna().nunique())
    test_counts  = drugInput_test .apply(lambda c: c.dropna().nunique())
    valid_drugs  = train_counts[train_counts > 2].index.intersection(
                    test_counts[test_counts > 1].index)
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
    print2log("Calculating per-TF Pearson correlations...")
    results = []
    for tf in TFs:
        r, p = pearsonr(
            true_matrix.loc[:, tf].values,
            pred_matrix.loc[:, tf].values
        )
        results.append({'TF': tf, 'pearson_r': r, 'p_value': p})
    corr_df = pd.DataFrame(results)
    corr_df['timepoint'] = timepoint
    corr_df['fold'] = fold_idx
    fold_results = pd.concat([fold_results, corr_df], ignore_index=True)

fold_results.to_csv(os.path.join(res_dir, 'tf_pearson_correlations_linear_CV.csv'), index=False)
print2log(corr_df.pearson_r.mean())
print2log(f"Saved per-TF Pearson correlations to {res_dir}/tf_pearson_correlations_linear_CV.csv")