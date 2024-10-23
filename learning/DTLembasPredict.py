import os
import numpy as np
import pandas as pd
import torch
import sys 
gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
import argparse
import logging
from pathlib import Path
from utility import *
import time
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def generate_doses(df):
    canonical_smile = df['canonical_smiles'].iloc[0]
    # Original doses for the current canonical_smiles, sorted in ascending order
    original_doses = sorted(df['dose'].values)
    # Minimum and maximum doses
    min_dose = min(original_doses)
    max_dose = max(original_doses)
    # Log-space interpolation between existing doses
    log_interpolated_doses = np.logspace(np.log10(min_dose), np.log10(max_dose), num=10)
    # Extrapolate: Down one order of magnitude smaller, up one order of magnitude larger
    extrapolated_doses = np.logspace(np.log10(min_dose / 10), np.log10(max_dose * 10), num=15)
    # Combine original doses and the generated doses
    all_doses = sorted(set(np.concatenate([original_doses, log_interpolated_doses, extrapolated_doses])))
    # Create a DataFrame with the generated doses
    return pd.DataFrame({'canonical_smiles': [canonical_smile] * len(all_doses), 'dose': all_doses})

parser = argparse.ArgumentParser(prog='Hepatocyte predictions')
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
parser.add_argument('--interpolate_extrapolate', action='store',type=bool, default=False)
parser.add_argument('--SelectedDrugsList', action='store',type=str, default=None)

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
interpolate_extrapolate = args.interpolate_extrapolate

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
all_drugs = drugInput.columns.values
if (args.SelectedDrugsList is not None):
    selected_drugs = pd.read_csv(args.SelectedDrugsList)
    selected_drugs = selected_drugs['canonical_smiles'].values
else:
    selected_drugs = np.copy(all_drugs)

### Make tensors and initialize model
if interpolate_extrapolate==True:
    # drugs = all_drugs[all_drugs!='Control']
    drugs = np.copy(all_drugs)
    drop_drugs = np.setdiff1d(all_drugs,selected_drugs)
    doseInfo = inputInfo[inputInfo['canonical_smiles'].isin(drugs)].loc[:,['canonical_smiles','dose']].reset_index(drop=True)
    expanded_doseInfo = doseInfo.groupby('canonical_smiles').apply(generate_doses).reset_index(drop=True)
    expanded_doseInfo = expanded_doseInfo.drop_duplicates()
    expanded_doseInfo['samples_id'] = ['sample_'+str(id) for id in range(len(expanded_doseInfo))]
    pivoted_doseInfo = expanded_doseInfo.pivot_table(
    index='samples_id',         # Each unique dose will be a row
    columns='canonical_smiles',  # Each canonical_smiles will be a column
    values='dose',        # Fill values with the dose
    aggfunc='first'       # In case of duplicate entries, take the first
    )
    pivoted_doseInfo.fillna(0,inplace=True)
    if ('Control' not in drugs):
        pivoted_doseInfo['Control'] = 0.
    pivoted_doseInfo = pivoted_doseInfo.drop_duplicates()
    pivoted_doseInfo.loc[:,drop_drugs] = 0.
    pivoted_doseInfo = pivoted_doseInfo[(pivoted_doseInfo != 0).any(axis=1)]
    pivoted_doseInfo = pivoted_doseInfo.applymap(lambda x: np.log10(x + 1))
    pivoted_doseInfo = pivoted_doseInfo.loc[:,drugInput.columns.values]
    X = torch.tensor(pivoted_doseInfo.values.copy(), dtype=torch.double).to(device)
else:
    X = torch.tensor(drugInput.values.copy(), dtype=torch.double).to(device)
# Y = torch.tensor(TFOutput.values, dtype=torch.double).to(device)

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

print2log('Begin predictions...')
Path(res_dir+'predictions/').mkdir(parents=True, exist_ok=True)
pivoted_doseInfo.to_csv(res_dir+'predictions/expanded_doseInfo.csv')

results_tf = pd.DataFrame({})
results_singaling_nodes = pd.DataFrame({})
start_time = time.time()
for i in range(number_DTLEMBAS_models):
    print2log('Run DT-LEMBAS model '+str(i+1)+' to simulate KOs')
    mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(i+1)+'_alldata_signal_state_dict_%s.pth'%(i+1))))
    drugLayer.load_state_dict(torch.load(os.path.join(data_path,modelPattern + str(i+1) + '_alldata_drugLayer_state_dict_%s.pth'%(i+1))))
    mod.eval()
    drugLayer.eval()

    Xin = drugLayer(X)
    TF, NetFull = mod(Xin)
    df_tf = pd.DataFrame(TF.detach().cpu().numpy())
    df_tf.columns = all_tfs
    if interpolate_extrapolate==True:
        df_tf['condition'] = pivoted_doseInfo.index.values
    else:
        df_tf['condition'] = drugInput.index.values
    df_tf['model_no'] = i
    results_tf = pd.concat([results_tf, df_tf], ignore_index=True).reset_index(drop=True)
    # make dataframe for singaling nodes now
    df_singaling_nodes = pd.DataFrame(NetFull.detach().cpu().numpy())
    df_singaling_nodes.columns = all_nodes
    if interpolate_extrapolate==True:
        df_singaling_nodes['condition'] = pivoted_doseInfo.index.values
    else:
        df_singaling_nodes['condition'] = drugInput.index.values
    df_singaling_nodes['model_no'] = i
    results_singaling_nodes = pd.concat([results_singaling_nodes, df_singaling_nodes], ignore_index=True).reset_index(drop=True)

mins, secs = divmod(time.time() - start_time, 60)
print2log("Simulation ran in: {:.0f} min {:.2f} sec".format(mins, secs))
print2log('Saving TF predictions...')
results_tf = results_tf.melt(id_vars=['condition','model_no'],var_name='TF',value_name='activity')
results_tf.to_csv(res_dir+'predictions/TFs_all_conditions.csv')
print2log('Saving singaling nodes predictions...')
results_singaling_nodes = results_singaling_nodes.melt(id_vars=['condition','model_no'],var_name='node',value_name='activity')
results_singaling_nodes.to_csv(res_dir+'predictions/SignalingNodes_all_conditions.csv')
print2log('Finished!')
