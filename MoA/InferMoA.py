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
import networkx as nx
import time
from matplotlib import pyplot as plt
import pcst_fast
import pyreadr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,RepeatedKFold

class FrozenLinear(nn.Module):
    def __init__(self, W_np: np.ndarray, b_np: np.ndarray | float, dtype=torch.double, device="gpu"):
        """
        W_np: (out_features, in_features)
        b_np: (out_features,) or scalar
        """
        super().__init__()
        out_features, in_features = W_np.shape
        self.lin = nn.Linear(in_features, out_features, bias=True).to(device=device, dtype=dtype)
        # copy weights/bias
        with torch.no_grad():
            self.lin.weight.copy_(torch.as_tensor(W_np, dtype=dtype, device=device))
            self.lin.bias.copy_(torch.as_tensor(np.atleast_1d(b_np), dtype=dtype, device=device))
    def forward(self, x):
        return self.lin(x)



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
parser.add_argument('--X_invivo_path', action='store', help='Path to in vivo TF activities (.rds)', default='../data/Xh_tfs_govaere.rds')
parser.add_argument('--Y_invivo_path', action='store', help='Path to in vivo phenotypes (.rds)', default='../data/Yh_tfs_govaere.rds')
parser.add_argument('--stimulation_strength', action='store',type=float, default=3000)
parser.add_argument('--num_steps', action='store',type = float, default=21)
# parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
#                     help='Selected signaling nodes to predict and save their activity',
#                     default=['P42226','P42229','P51692','P40763','P52630','P42224','Q14765',
#                              'Q15797','Q15796','P84022','Q13485','Q99717','O43541',
#                              'P17181','P48551','P15260','P38484',
#                              'P10914','Q00978','Q14653',
#                              'P05231','Q13651'])
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
stimulation_strength=args.stimulation_strength
num_steps = args.num_steps

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
for timepoint in ['8 hour']: #['2 hour','8 hour','24 hour']:
    print2log('Processing timepoint {}'.format(timepoint))
    if timepoint == '8 hour':
        data_path = res_dir + 'allmodels/'
    else:
        data_path = res_dir+timepoint+'/allmodels/'
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
    # X = torch.tensor(drugInput_tmp.values.copy(), dtype=torch.double).to(device)
    # Y = torch.tensor(TFOutput_tmp.values, dtype=torch.double).to(device)
    print2log("Prepare and load the SignalingModel and the Graph...")
    tfs = list(outName)
    mask_dt =torch.tensor(np.array(drugTargets)).transpose(0,1)
    bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 
                    'exp_factor':50, 'tolerance': args.LEMBAS_tolerance, 
                    'leak':args.LEMBAS_leak}
    for model_no in range(number_DTLEMBAS_models):
        mod = SignalingModel(net = net,
                            X_in = drugTargets,
                            y_out = TFOutput_tmp, 
                            projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                            weight_label = weight_label, source_label = source_label, target_label = target_label,
                            bionet_params = bionet_params, 
                            dtype = torch.double, device = device)
        mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(model_no+1)+'signal_state_dict_%s.pth'%(model_no+1))))
        mod.eval()
        mod.signaling_network.weights.data.masked_fill_(mask = mod.signaling_network.mask, value = 0.0)
        weights = mod.signaling_network.weights.detach().cpu().numpy()
        A = pd.DataFrame(weights.T, index = nodeNames, columns = nodeNames)
        sources = ['P36897','P17181']
        G = nx.from_pandas_adjacency(A, create_using=nx.DiGraph())
        # edges_for_pcst = np.array(list(G.edges()))

        print2log("Train TFs ---> Phenotype model...")
        Xh = pyreadr.read_r(args.X_invivo_path)[None].to_numpy()
        Yh = pyreadr.read_r(args.Y_invivo_path)[None].to_numpy()
        params = {'alpha': [0.001,0.01,0.1,0.2,0.5,1,2,5,10,20,50,100,500,1000]}
        ridge = GridSearchCV(estimator=Ridge(),param_grid = params, cv=RepeatedKFold(n_splits=10, n_repeats=10), n_jobs=-1)
        ridge.fit(Xh, Yh)
        coefs = ridge.best_estimator_.coef_
        intercept = ridge.best_estimator_.intercept_
        pheno = FrozenLinear(coefs, intercept, dtype=torch.double, device=device)
        pheno.eval()
        mod.eval()
        print2log("Score nodes and edges of the graph...")
        model = torch.nn.Sequential(
            mod.signaling_network,
            mod.output_layer,
            pheno
        )
        model.eval()
        ## Create grid of stimulations
        nodeNames = np.array(list(mod.node_idx_map.keys()))
        num_tfs = mod.y_out.shape[1]
        all_tfs = mod.y_out.columns.values
        all_nodes = np.array(list(mod.node_idx_map.keys()))
        subset_nodes = {key: mod.node_idx_map[key] for key in ['P36897','P17181'] if key in mod.node_idx_map}
        subset_nodes_inds = np.array(list(subset_nodes.values()))
        subset_nodes = np.array(list(subset_nodes.keys()))

        Xin = torch.eye(len(all_nodes),dtype=torch.double).to(device)
        Xin_1 = Xin[subset_nodes_inds[0],:].unsqueeze(0)
        Xin_2 = Xin[subset_nodes_inds[1],:].unsqueeze(0)
        # Define the number of interpolation steps you want
        # Create linearly spaced values between -3000 and 3000
        stim_range = torch.linspace(-stimulation_strength, stimulation_strength, num_steps)
        # Create a grid of all combinations of stimulations
        grid_Xin1, grid_Xin2 = torch.meshgrid(stim_range, stim_range, indexing='ij')
        # Flatten the grid for batch processing
        grid_Xin1 = grid_Xin1.flatten().view(-1, 1)
        grid_Xin2 = grid_Xin2.flatten().view(-1, 1)
        # Expand the grids to match input dimensions
        grid_Xin1 = grid_Xin1.repeat(1, Xin_1.shape[1])
        grid_Xin2 = grid_Xin2.repeat(1, Xin_2.shape[1])
        # Multiply inputs with the grid values
        Xin_1_grid = Xin_1 * grid_Xin1.to(device).double()
        Xin_2_grid = Xin_2 * grid_Xin2.to(device).double()
        # Combine the two inputs for the model
        Xin_combined = Xin_1_grid + Xin_2_grid
        Xin_combined = Xin_combined.detach()
        ##Run the pass to score nodes and weights
        Y = model(Xin_combined)
        L_objective = torch.sum(Y)
        L_objective.backward()
        db = model[0].bias.grad.T.squeeze().detach()
        dW = model[0].weights.grad.detach()
        # plt.hist(dW.flatten().detach().cpu().numpy()[dW.flatten().detach().cpu().numpy()!=0])
        # plt.show()
        # plt.hist(mod.signaling_network.weights.detach().cpu().numpy()[mod.signaling_network.weights.detach().cpu().numpy()!=0])
        # plt.show()
        activities = model[0](Xin_combined).detach()
        Range = torch.abs(torch.max(activities,0)[0] - torch.min(activities,0)[0])
        nodeScores = torch.abs(db) * Range
        prizes = nodeScores.detach().cpu().numpy()
        inf_prize = nodeScores.max() * 100
        #### PUT IN PRIZES  FOR SOME NODES inf_prize ####
        prizes[subset_nodes_inds[0]] = inf_prize
        prizes[subset_nodes_inds[1]] = inf_prize
        name_to_int_map = {name: i for i, name in enumerate(A.columns)}
        # Convert your list of string edges to a list of integer edges
        integer_edge_list = []
        for u_name, v_name in list(G.edges()):
            integer_edge_list.append([name_to_int_map[u_name], name_to_int_map[v_name]])
        # Now create the final NumPy array with the correct data type
        edges_for_pcst = np.array(integer_edge_list, dtype=np.int64)
        root_int = name_to_int_map['P36897']
        ## now make edge cost
        edgeScores = torch.abs(dW) * torch.abs(mod.signaling_network.weights)
        edgeScores = edgeScores.T.detach().cpu().numpy()
        costs = []
        for u_i, v_i in integer_edge_list:
            # Get the weight from your original matrix
            w = edgeScores[u_i, v_i]
            # Calculate the cost (ensure it's a float)
            costs.append(1.0 / (w + 0.001))
        costs = np.array(costs, dtype=np.float64)


        print2log("Running PCST algorithm...")
        vertex_indices, edge_indices = pcst_fast.pcst_fast(
            edges_for_pcst,
            prizes,
            costs,
            root_int,
            1,
            'gw',
            0)

        print2log("Extracting subgraph...")
        result_nodes = list(vertex_indices)
        result_edges_list = [list(G.edges())[i] for i in edge_indices]
        result_graph = nx.DiGraph(result_edges_list)
        print2log(f"Nodes in subgraph: {result_nodes}")
        print2log(f"Subgraph number of edges: {len(result_edges_list)}")

        print("\nSaving subgraph edgelist to CSV...")
        # Create a map from an edge tuple to its original weight for easy lookup
        # Build a list of dictionaries for the DataFrame by looking up weights directly
        subgraph_data = []
        for u, v in result_edges_list:
            # Directly access the weight from your adjacency matrix DataFrame
            # This assumes the matrix is symmetric, so adj_matrix_df.loc[u, v] works
            weight = A.loc[u, v]
            subgraph_data.append({
                'source': u,
                'target': v,
                'weight': weight
            })
        # Create and save the DataFrame
        df = pd.DataFrame(subgraph_data)
        # Using your variable `model_no` for the filename
        df.to_csv(f'subgraph_edgelist_{model_no}.csv', index=False)
        print(f"Subgraph saved to subgraph_edgelist_{model_no}.csv")

        print2log('\nVisualuze the subgraph...')
        plt.figure(figsize=(12, 12))
        # Use a layout designed for the subgraph for better spacing
        pos = nx.spring_layout(result_graph, seed=42)
        # Draw the subgraph
        nx.draw_networkx_nodes(result_graph, pos, node_color='skyblue', node_size=600)
        nx.draw_networkx_edges(result_graph, pos, edge_color='dodgerblue', width=2.0)
        nx.draw_networkx_labels(result_graph, pos, font_color='black', font_size=10)
        # Highlight input nodes and high-prize TFs that are in the result
        final_input_nodes = [n for n in sources if n in result_graph]
        high_prize_tfs = [i for i, p in enumerate(prizes) if p > 0 and i in result_graph]
        nx.draw_networkx_nodes(result_graph, pos, nodelist=final_input_nodes, node_color='red', node_size=700)
        nx.draw_networkx_nodes(result_graph, pos, nodelist=high_prize_tfs, node_color='gold', node_size=700)
        plt.title("Focused Visualization of PCST Subgraph")
        plt.savefig("pcst_subgraph_result_%s.png"%model_no)
        print("Visualization saved to pcst_subgraph_result_%s.png"%model_no)
    print2log(f"Finished timepoint: {timepoint}")



# reach_from_src = set(sources)
# for s in sources:
#     reach_from_src |= nx.descendants(G, s)
# Gr = G.reverse(copy=False)
# reach_to_tf = set(tfs)
# for t in tfs:
#     reach_to_tf |= nx.descendants(Gr, t)
# keep = reach_from_src & reach_to_tf
# subG = G.subgraph(keep).copy()