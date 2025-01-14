import os
import numpy as np
import pandas as pd
import torch
import sys 
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
import argparse
import logging
from pathlib import Path
from utility import *
import networkx as nx
import time
from matplotlib import pyplot as plt
import plotnine as p9
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def extract_first_neighbors(network_df, desired_node):
    # Build the graph from the dataframe
    """
    Build a directed graph from the dataframe, find the first neighbors of the desired node, extract the subnetwork and save it to a CSV file.
    Parameters
    ----------
    network_df : pd.DataFrame
        A dataframe containing the adjacency list of a signaling network with the following columns:
            - 'source' : source node column name
            - 'target' : target node column name
            - 'mode_of_action' : whether the interaction is stimulating (1) or inhibiting (-1) or unknown (0)
    desired_node : str
        the node of interest
    Returns
    -------
    input_nodes : list
        a list of the input nodes of the desired node
    output_nodes : list
        a list of the output nodes of the desired node
    """
    G = nx.DiGraph()
    for _, row in network_df.iterrows():
        G.add_edge(row['source'], row['target'], interaction=row['mode_of_action'])
    # Get 1st neighbors of the desired node
    first_neighbors = set(nx.all_neighbors(G, desired_node))
    sub_nodes = first_neighbors.union({desired_node})
    sub_G = G.subgraph(sub_nodes).copy()
    # Save the subnetwork to CSV
    subnetwork_df = pd.DataFrame(
        [(u, v, d["interaction"]) for u, v, d in sub_G.edges(data=True)],
        columns=["source", "target", "mode_of_action"]
    )
    ### get target nodes of the desired node
    output_nodes = subnetwork_df[subnetwork_df['source'] == desired_node]['target'].unique().tolist() 
    ### get source nodes of the desired node
    input_nodes = subnetwork_df[subnetwork_df['target'] == desired_node]['source'].unique().tolist()
    return output_nodes,input_nodes

parser = argparse.ArgumentParser(prog='Hepatocyte simulation')
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
parser.add_argument('--annotation_file_for_visualization', action='store',type=str, default='../data/annotation.csv')
parser.add_argument('--stimulation_range_neg', action='store',type = float, default=8000)
parser.add_argument('--stimulation_range_pos', action='store',type = float, default=3000)
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected signaling nodes to predict and save their activity',
                    default=['P36897','P37173',
                             'P17181','P48551',
                             'P15260','P38484'])
# tgfbr = ['P36897','P37173']
# ifnRs = ['P17181','P48551','P15260','P38484']
parser.add_argument('--selected_conditions', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected conditions to combine with simulations',
                    default=['erythromycin ethylsuccinate+High+','erythromycin ethylsuccinate+High+', # allyl alcohol+High+ for 2 hour but it does not work for 8 and 24 hour
                             'phenylbutazone+High+','phenylbutazone+High+',
                             'benzbromarone+High+','benzbromarone+High+'])

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
stimulation_range_pos=args.stimulation_range_pos
stimulation_range_neg=args.stimulation_range_neg
selected_signaling_nodes = np.array(args.selected_signaling_nodes)
annotation_file_for_visualization = args.annotation_file_for_visualization
selected_conditions = args.selected_conditions
selected_conditions = [x+timepoint for x in selected_conditions]

n_cores = 14
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
# Xdrugs = torch.tensor(drugInput.values.copy(), dtype=torch.double).to(device)
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

nodeNames = np.array(list(mod.node_idx_map.keys()))
num_tfs = mod.y_out.shape[1]
all_tfs = mod.y_out.columns.values
all_nodes = np.array(list(mod.node_idx_map.keys()))

print2log('Investigate the range of training signals seeing by INF and TFGB receptors')
## Select some specific nodes for measuring their activity
subset_nodes = {key: mod.node_idx_map[key] for key in selected_signaling_nodes if key in mod.node_idx_map}
subset_nodes_inds = np.array(list(subset_nodes.values()))
subset_nodes = np.array(list(subset_nodes.keys()))

### Get first neighbors
print2log('Get first neighbors of nodes of interest...')
outs_inds = []
outs_names = []
ins_inds = []
ins_names = []
for node in selected_signaling_nodes:
    print2log('Extracting first neighbors for node '+node)
    output_nodes,input_nodes = extract_first_neighbors(net, node)
    output_nodes = {key: mod.node_idx_map[key] for key in output_nodes if key in mod.node_idx_map}
    output_nodes_inds = np.array(list(output_nodes.values()))
    output_nodes = np.array(list(output_nodes.keys()))
    input_nodes = {key: mod.node_idx_map[key] for key in input_nodes if key in mod.node_idx_map}
    input_nodes_inds = np.array(list(input_nodes.values()))
    input_nodes = np.array(list(input_nodes.keys()))
    outs_inds.append(output_nodes_inds)
    outs_names.append(output_nodes)
    ins_inds.append(input_nodes_inds)
    ins_names.append(input_nodes)

Path(res_dir+'signal_exploration').mkdir(parents=True, exist_ok=True)
results_tf = pd.DataFrame({})
results_out = pd.DataFrame({})
results_in = pd.DataFrame({})
results_with_drugs = pd.DataFrame({})
start_time = time.time()
# ## Example for one node
# tt = mod.signaling_network.weights.T[:,subset_nodes_inds[0]]
# aa = tt[input_nodes_inds] * NetFull[:,input_nodes_inds] 
# bb = torch.sum(aa,1)
# cc = bb + mod.signaling_network.bias[subset_nodes_inds[0]]
interpolated_values_neg = torch.linspace(-stimulation_range_neg, 0, 101).view(-1, 1)
interpolated_values_pos = torch.linspace(0, stimulation_range_pos, 201).view(-1, 1)
interpolated_values = torch.cat((interpolated_values_neg, interpolated_values_pos),dim=0)
# interpolated_values = torch.linspace(-stimulation_range, stimulation_range, 10001).view(-1, 1)
Xin_stim = torch.eye(len(all_nodes),dtype=torch.double).to(device)
Xin_stim = Xin_stim[subset_nodes_inds,:]
interpolated_values = interpolated_values.repeat(1, Xin_stim.shape[1])
Xin_stim = Xin_stim.unsqueeze(1) * torch.tensor(interpolated_values).double().to(device)
print2log('Begin simulations....')

for i in range(number_DTLEMBAS_models):
    print2log('Run DT-LEMBAS model '+str(i+1))
    mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(i+1)+'signal_state_dict_%s.pth'%(i+1))))
    drugLayer.load_state_dict(torch.load(os.path.join(data_path,modelPattern + str(i+1) + 'drugLayer_state_dict_%s.pth'%(i+1))))
    mod.eval()
    drugLayer.eval()
    # Yhat, NetFull = mod(Xin_drugs)
    # print2log('Simulate convergence of output nodes` activities....')
    ### Perform stimulation
    with torch.no_grad():
        for j in range(Xin_stim.shape[0]):
            YhatFull = mod.signaling_network(Xin_stim[j,:,:])
            Yhat = mod.output_layer(YhatFull)
            df = pd.DataFrame(Yhat.detach().cpu().numpy())
            df.columns = all_tfs
            df['condition'] = interpolated_values[:,0].numpy() 
            df['node'] = subset_nodes[j]
            df['model_no'] = i
            results_tf = pd.concat([results_tf, df], ignore_index=True).reset_index(drop=True)
            
            ### Retrieve output nodes` activites
            output_nodes_inds = np.concat([np.array(subset_nodes_inds[j]).reshape(1),outs_inds[j]])
            output_nodes = np.concat([np.array(subset_nodes[j]).reshape(1),outs_names[j]])
            outActivities = YhatFull[:,output_nodes_inds]
            df = pd.DataFrame(outActivities.detach().cpu().numpy())
            df.columns = output_nodes
            df['condition'] = interpolated_values[:,0].numpy() 
            df['node'] = subset_nodes[j]
            df['model_no'] = i
            df = df.melt(id_vars=['condition', 'node', 'model_no'], var_name='outlet', value_name='activity')
            results_out = pd.concat([results_out, df], ignore_index=True).reset_index(drop=True)

            ### Retrieve input nodes` activites
            input_nodes_inds = np.concat([np.array(subset_nodes_inds[j]).reshape(1),ins_inds[j]])
            input_nodes = np.concat([np.array(subset_nodes[j]).reshape(1),ins_names[j]])
            inActivities = YhatFull[:,input_nodes_inds]
            df = pd.DataFrame(inActivities.detach().cpu().numpy())
            df.columns = input_nodes
            df['condition'] = interpolated_values[:,0].numpy() 
            df['node'] = subset_nodes[j]
            df['model_no'] = i
            df = df.melt(id_vars=['condition', 'node', 'model_no'], var_name='inlet', value_name='activity')
            results_in = pd.concat([results_in, df], ignore_index=True).reset_index(drop=True)

            ### Retrieve output nodes` activites togther with drug stimulation
            Xdrugs = drugInput.loc[selected_conditions[j],:].to_numpy()
            Xdrugs = np.repeat(np.expand_dims(Xdrugs, axis=0), repeats=interpolated_values.shape[0], axis=0)
            Xdrugs = torch.tensor(Xdrugs).double().to(device)
            Xin_drugs = drugLayer(Xdrugs)
            Xin_drugs = mod.input_layer(Xin_drugs)
            YhatFull = mod.signaling_network(Xin_stim[j,:,:]+Xin_drugs)
            Yhat = mod.output_layer(YhatFull)
            output_nodes_inds = np.concat([np.array(subset_nodes_inds[j]).reshape(1),outs_inds[j]])
            output_nodes = np.concat([np.array(subset_nodes[j]).reshape(1),outs_names[j]])
            outActivities = YhatFull[:,output_nodes_inds]
            df = pd.DataFrame(outActivities.detach().cpu().numpy())
            df.columns = output_nodes
            df['condition'] = interpolated_values[:,0].numpy() 
            df['stimuli'] = subset_nodes[j] + '+'+selected_conditions[j]
            df['model_no'] = i
            df = df.melt(id_vars=['condition', 'stimuli', 'model_no'], var_name='outlet', value_name='activity')
            results_with_drugs = pd.concat([results_with_drugs, df], ignore_index=True).reset_index(drop=True)
        
        # print2log('Calculating individual input signals...')
        # ### The Xin after passing from the input layer
        # ### this is the Xin multiplied by the weights that go into a desired node
        # Xin_full = mod.input_layer(Xin)
        # X_bias = Xin_full.T + mod.signaling_network.bias # this is the bias with the projection_amplitude included
        # X_new = torch.zeros_like(X_bias)
        # X_new = torch.mm(mod.signaling_network.weights, X_new) # scale matrix by edge weights
        # X_new = X_new + X_bias  # add original values and bias       
        # X_new = mod.signaling_network.activation(X_new, mod.signaling_network.training_params['leak'])
        # ### Get the neighboring nodes signal (first neighbors found above with networkx)

        # print2log('Calculating total input signal...')
        # ### Calculate the sum of these input signals

        # ### Calculate the wighted sum of these input signals
        # ### First get the weights of the adgecency matrix for these nodes

        # ### The calculate the weighted sum

mins, secs = divmod(time.time() - start_time, 60)
print2log("Simulation ran in: {:.0f} min {:.2f} sec".format(mins, secs))

print2log('Saving results...')
results_tf.to_csv(res_dir+'signal_exploration/TFs_all_conditions.csv')
results_out.to_csv(res_dir+'signal_exploration/Nodes_output_conditions.csv')
results_in.to_csv(res_dir+'signal_exploration/Nodes_input_conditions.csv')
results_with_drugs.to_csv(res_dir+'signal_exploration/Nodes_output_drugs_conditions.csv')

# print2log('Visualize results...')
print2log('Finished!')