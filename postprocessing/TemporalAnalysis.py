import os
import numpy as np
import pandas as pd
import torch
import sys 
gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
# gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
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

# df_signaling_states = pd.DataFrame({})
# df_tf_activities = pd.DataFrame({})
for timepoint in ['24 hour']:# ['2 hour','8 hour','24 hour']:
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

    for i in range(number_DTLEMBAS_models):
        print2log('Load DT-LEMBAS model '+str(i+1)+' to simulate get intermediate results')
        if timepoint=='8 hour':
            data_path = res_dir + 'allmodels/'
        else:
            data_path = res_dir + timepoint + '/allmodels/'
        mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(i+1)+'signal_state_dict_%s.pth'%(i+1))))
        drugLayer.load_state_dict(torch.load(os.path.join(data_path,modelPattern + str(i+1) + 'drugLayer_state_dict_%s.pth'%(i+1))))
        mod.eval()
        drugLayer.eval()
        with torch.no_grad():
            Xin = drugLayer(X)
            if timepoint == '2 hour':
                Yhat, YhatFull = mod(Xin)
                tmp = pd.DataFrame(YhatFull.detach().cpu().numpy())
                tmp.columns = nodeNames
                tmp = tmp.loc[:,selected_signaling_nodes]
                tmp['condition'] = drugInput_tmp.index
                tmp['step'] = mod.signaling_network.training_params['max_steps']
                tmp['time'] = timepoint
                tmp['model_no'] =i+1
                tmp = tmp.melt(id_vars=['time', 'model_no', 'condition', 'step'], var_name='node', value_name='signal').reset_index(drop=True)
                df_signaling_states = pd.concat([df_signaling_states,tmp]).reset_index(drop=True)
                tmp = pd.DataFrame(Yhat.detach().cpu().numpy())
                tmp.columns = TFOutput_tmp.columns
                tmp['condition'] = drugInput_tmp.index
                tmp['step'] = mod.signaling_network.training_params['max_steps']
                tmp['time'] = timepoint
                tmp['model_no'] =i+1
                tmp = tmp.melt(id_vars=['time', 'model_no', 'condition', 'step'], var_name='TF', value_name='activity').reset_index(drop=True)
                df_tf_activities = pd.concat([df_tf_activities,tmp]).reset_index(drop=True)
                # Yhat_2h = torch.clone(Yhat)
                # YhatFull_2h = torch.clone(YhatFull)
            else:
                Yin = mod.input_layer(Xin)
                ## Run mannually of the steps to get intermediate results
                mod.signaling_network.weights.data.masked_fill_(mask = mod.signaling_network.mask, value = 0.0)
                X_bias = Yin.T + mod.signaling_network.bias
                X_new = torch.zeros_like(X_bias)
                max_corr=0
                max_corr_8h=0
                max_corr_2h=0
                df_signaling_states = pd.DataFrame({})
                df_tf_activities = pd.DataFrame({})
                for t in range(mod.signaling_network.training_params['max_steps']):
                    X_old = X_new
                    X_new = torch.mm(mod.signaling_network.weights, X_new)
                    X_new = X_new + X_bias
                    X_new = mod.signaling_network.activation(X_new, mod.signaling_network.training_params['leak'])
                    Yhat = mod.output_layer(X_new.T)
                    tmp = pd.DataFrame(X_new.T.detach().cpu().numpy())
                    tmp.columns = nodeNames
                    tmp = tmp.loc[:,selected_signaling_nodes]
                    tmp['condition'] = drugInput_tmp.index
                    tmp['step'] = t
                    tmp['time'] = timepoint
                    tmp['model_no'] =i+1
                    tmp = tmp.melt(id_vars=['time', 'model_no', 'condition', 'step'], var_name='node', value_name='signal').reset_index(drop=True)
                    df_signaling_states = pd.concat([df_signaling_states,tmp]).reset_index(drop=True)
                    tmp = pd.DataFrame(Yhat.detach().cpu().numpy())
                    tmp.columns = TFOutput_tmp.columns
                    tmp['condition'] = drugInput_tmp.index
                    tmp['step'] = t
                    tmp['time'] = timepoint
                    tmp['model_no'] =i+1
                    tmp = tmp.melt(id_vars=['time', 'model_no', 'condition', 'step'], var_name='TF', value_name='activity').reset_index(drop=True)
                    df_tf_activities = pd.concat([df_tf_activities,tmp]).reset_index(drop=True)
                    # if timepoint == '8 hour':
                    #     corr = pearson_r(Yhat,Yhat_2h).mean().item()
                    #     if corr > max_corr:
                    #         max_corr = corr
                    #         Yhat_all_signals = torch.clone(X_new.T)
                    #         closest_step = t
                    # else:
                    #     corr = pearson_r(Yhat,Yhat_8h).mean().item()
                    #     if corr > max_corr:
                    #         max_corr_8h = corr
                    #         Yhat_all_signals_8h = torch.clone(X_new.T)
                    #         closest_step_8h = t
                    #     corr = pearson_r(Yhat,Yhat_2h).mean().item()
                    #     if corr > max_corr_2h:
                    #         max_corr_2h = corr
                    #         Yhat_all_signals_2h = torch.clone(X_new.T)
                    #         closest_step_2h = t
                    if (t % 10 == 0) and (t > 20):
                        diff = torch.max(torch.abs(X_new - X_old))
                        if diff.lt(mod.signaling_network.training_params['tolerance']):
                            break
                Y_full = X_new.T
                # if timepoint == '8 hour':
                #     Yhat_8h = torch.clone(Yhat)
                #     YhatFull_8h = torch.clone(YhatFull)
                #     tmp = pd.DataFrame(Yhat_all_signals.detach().cpu().numpy())
                #     tmp.columns = nodeNames
                #     tmp['condition'] = drugInput_tmp.index
                #     tmp['step'] = closest_step
                #     tmp['time'] = timepoint
                #     tmp['model_no'] =i+1
                #     tmp = tmp.melt(id_vars=['time', 'model_no', 'condition', 'step'], var_name='node', value_name='signal').reset_index(drop=True)
                #     tmp.to_csv(res_dir + 'All_SignalingStates_8h_closestTo2h.csv')
                # else:
                #     tmp = pd.DataFrame(Yhat_all_signals_2h.detach().cpu().numpy())
                #     tmp.columns = nodeNames
                #     tmp['condition'] = drugInput_tmp.index
                #     tmp['step'] = closest_step_2h
                #     tmp['time'] = timepoint
                #     tmp['model_no'] =i+1
                #     tmp = tmp.melt(id_vars=['time', 'model_no', 'condition', 'step'], var_name='node', value_name='signal').reset_index(drop=True)
                #     tmp.to_csv(res_dir + 'All_SignalingStates_24h_closestTo2h.csv')
                #     # now for 8 hours
                #     tmp = pd.DataFrame(Yhat_all_signals_8h.detach().cpu().numpy())
                #     tmp.columns = nodeNames
                #     tmp['condition'] = drugInput_tmp.index
                #     tmp['step'] = closest_step_8h
                #     tmp['time'] = timepoint
                #     tmp['model_no'] =i+1
                #     tmp = tmp.melt(id_vars=['time', 'model_no', 'condition', 'step'], var_name='node', value_name='signal').reset_index(drop=True)
                #     tmp.to_csv(res_dir + 'All_SignalingStates_24h_closestTo8h.csv')
        df_signaling_states.to_csv(res_dir + 'All_intermediate_SignalingStates_'+timepoint+'_model'+str(i+1)+'.csv')
        df_tf_activities.to_csv(res_dir + 'All_intermediate_TFActivities_'+timepoint+'_model'+str(i+1)+'.csv')
print2log('Done')