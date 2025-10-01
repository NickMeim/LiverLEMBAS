import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys 
# gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
import argparse
import logging
from pathlib import Path
from utility import *
import networkx as nx
from matplotlib import pyplot as plt
import pcst_fast
import pyreadr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,RepeatedKFold
from collections import defaultdict

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
parser.add_argument('--AnnotationCSV', action='store', type=str,
                    default='../data/annotation.csv',
                    help='CSV with columns: uniprot,symbol for node renaming')
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
parser.add_argument('--EDGE_FREQ_MIN', action='store',type = float, default=0.5)
parser.add_argument('--NODE_FREQ_MIN', action='store',type = float, default=0.5)

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

Path('results/individual_networks').mkdir(parents=True, exist_ok=True)

n_cores = 12
utils.set_cores(n_cores)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log('Running on {}'.format(device))

print2log('Loading input data...')
# Load annotation and build mapping dict
ann = pd.read_csv(args.AnnotationCSV)
if not {'uniprot', 'symbol'}.issubset(ann.columns):
    raise ValueError("AnnotationCSV must have columns: 'uniprot' and 'symbol'")
id2sym = dict(zip(ann['uniprot'].astype(str), ann['symbol'].astype(str)))
def to_symbol(x: str) -> str:
    # return symbol if known, otherwise keep original Uniprot
    return id2sym.get(str(x), str(x))
# Aggregators across all models (appearance per model)
edge_models = defaultdict(set)   # key=(src_sym, tgt_sym) -> set of model indices
node_models = defaultdict(set)   # key=node_sym -> set of model indices
edge_weight_sum = defaultdict(float)        # (src_sym, tgt_sym) -> sum of raw weights
edge_weight_abs_sum = defaultdict(float)    # (src_sym, tgt_sym) -> sum of |weights|

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
    
    print2log("Prepare and load the SignalingModel and the Graph...")
    tfs = list(outName)
    mask_dt =torch.tensor(np.array(drugTargets)).transpose(0,1)
    bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 
                    'exp_factor':50, 'tolerance': args.LEMBAS_tolerance, 
                    'leak':args.LEMBAS_leak}
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
    coef_vec = np.mean(np.abs(coefs),0)
    top10_idx = np.argsort(coef_vec)[-10:][::-1]
    top10_tfs = [to_symbol(TFOutput_tmp.columns[i]) for i in top10_idx]
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
        
        mod.eval()
        print2log(f"Model {model_no+1}: Score nodes and edges...")
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
        
        stim_range = torch.linspace(-stimulation_strength, stimulation_strength, int(num_steps))
        
        grid_Xin1, grid_Xin2 = torch.meshgrid(stim_range, stim_range, indexing='ij')
        
        grid_Xin1 = grid_Xin1.flatten().view(-1, 1)
        grid_Xin2 = grid_Xin2.flatten().view(-1, 1)
        
        grid_Xin1 = grid_Xin1.repeat(1, Xin_1.shape[1])
        grid_Xin2 = grid_Xin2.repeat(1, Xin_2.shape[1])
        
        Xin_1_grid = Xin_1 * grid_Xin1.to(device).double()
        Xin_2_grid = Xin_2 * grid_Xin2.to(device).double()
        
        Xin_combined = Xin_1_grid + Xin_2_grid
        Xin_combined = Xin_combined.detach()
        
        Y = model(Xin_combined)
        L_objective = torch.sum(Y)
        L_objective.backward()
        db = model[0].bias.grad.T.squeeze().detach()
        dW = model[0].weights.grad.detach()
        
        activities = model[0](Xin_combined).detach()
        Range = torch.abs(torch.max(activities,0)[0] - torch.min(activities,0)[0])
        nodeScores = torch.abs(db) * Range
        prizes = nodeScores.detach().cpu().numpy()
        inf_prize = nodeScores.max() * 100
        
        prizes[subset_nodes_inds[0]] = inf_prize
        prizes[subset_nodes_inds[1]] = inf_prize
        name_to_int_map = {name: i for i, name in enumerate(A.columns)}
        
        integer_edge_list = []
        for u_name, v_name in list(G.edges()):
            integer_edge_list.append([name_to_int_map[u_name], name_to_int_map[v_name]])
        
        edges_for_pcst = np.array(integer_edge_list, dtype=np.int64)
        root_int = name_to_int_map['P36897']
        
        edgeScores = torch.abs(dW) * torch.abs(mod.signaling_network.weights)
        edgeScores = edgeScores.T.detach().cpu().numpy()
        costs = []
        for u_i, v_i in integer_edge_list:
            w = edgeScores[u_i, v_i]
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
        result_nodes_uniprot = [A.columns[i] for i in vertex_indices]
        result_edges_list = [(A.columns[u], A.columns[v]) for u,v in edges_for_pcst[edge_indices]]
        result_graph = nx.DiGraph(result_edges_list)
        print2log(f"Nodes in subgraph: {len(result_nodes_uniprot)}")
        print2log(f"Subgraph number of edges: {len(result_edges_list)}")

        print("\nSaving subgraph edgelist to CSV...")
        
        subgraph_data = []
        for u, v in result_edges_list:
            weight = A.loc[u, v] 
            subgraph_data.append({
                'source_uniprot': u,
                'target_uniprot': v,
                'source_symbol': to_symbol(u),
                'target_symbol': to_symbol(v),
                'weight': weight
            })
        
        df = pd.DataFrame(subgraph_data)
        
        df.to_csv(f'results/individual_networks/subgraph_edgelist_{model_no}.csv', index=False)
        print(f"Subgraph saved to subgraph_edgelist_{model_no}.csv")

        # Update global aggregators (use symbols so identical proteins collapse across isoforms)
        for _, r in df.iterrows():
            src_sym = r['source_symbol']
            tgt_sym = r['target_symbol']
            key = (src_sym, tgt_sym)
            edge_models[key].add(model_no)
            node_models[src_sym].add(model_no)
            node_models[tgt_sym].add(model_no)
            w = float(r['weight'])
            edge_weight_sum[key] += w
            edge_weight_abs_sum[key] += abs(w)

        print2log('\nVisualuze the subgraph...')
        plt.figure(figsize=(12, 12))
        
        pos = nx.spring_layout(result_graph, seed=42)
        
        nx.draw_networkx_nodes(result_graph, pos, node_color='skyblue', node_size=600)
        nx.draw_networkx_edges(result_graph, pos, edge_color='dodgerblue', width=2.0)
        nx.draw_networkx_labels(result_graph, pos, font_color='black', font_size=10)
        
        final_input_nodes = [n for n in sources if n in result_graph]
        high_prize_tfs = [i for i, p in enumerate(prizes) if p > 0 and i in result_graph]
        nx.draw_networkx_nodes(result_graph, pos, nodelist=final_input_nodes, node_color='red', node_size=700)
        nx.draw_networkx_nodes(result_graph, pos, nodelist=high_prize_tfs, node_color='gold', node_size=700)
        plt.title("Focused Visualization of PCST Subgraph")
        plt.savefig("results/individual_networks/pcst_subgraph_result_%s.png"%model_no)
        print2log("Visualization saved to pcst_subgraph_result_%s.png"%model_no)

    print2log(f"Finished timepoint: {timepoint}")

print2log("\nAggregating edges/nodes across models...")
# Aggregate edges
edge_rows = []
for (src_sym, tgt_sym), model_set in edge_models.items():
    count = len(model_set)
    freq = count / float(number_DTLEMBAS_models)
    mean_w = edge_weight_sum[(src_sym, tgt_sym)] / max(count, 1)
    mean_abs_w = edge_weight_abs_sum[(src_sym, tgt_sym)] / max(count, 1)
    edge_rows.append({
        'source_symbol': src_sym,
        'target_symbol': tgt_sym,
        'models_count': count,
        'models_freq': freq,
        'mean_weight': mean_w,
        'mean_abs_weight': mean_abs_w,
    })
edges_agg_df = pd.DataFrame(edge_rows).sort_values(
    ['models_count', 'source_symbol', 'target_symbol'],
    ascending=[False, True, True]
).reset_index(drop=True)
edges_agg_df.to_csv('results/subgraph_edges_aggregate.csv', index=False)
print2log("Saved aggregated edge appearances to results/subgraph_edges_aggregate.csv")
# Aggregate nodes
node_rows = []
for node_sym, model_set in node_models.items():
    count = len(model_set)
    freq = count / float(number_DTLEMBAS_models)
    node_rows.append({'node_symbol': node_sym, 'models_count': count, 'models_freq': freq})
nodes_agg_df = pd.DataFrame(node_rows).sort_values(
    ['models_count', 'node_symbol'],
    ascending=[False, True]
).reset_index(drop=True)
nodes_agg_df.to_csv('results/subgraph_nodes_aggregate.csv', index=False)
print2log("Saved aggregated node appearances to results/subgraph_nodes_aggregate.csv")

# ----------------------------------------------------------------
# Build filtered aggregate network (Best Paths Approach)
# ----------------------------------------------------------------
print2log("\nBuilding final network using the 'Best Paths' approach...")
SOURCES = ['TGFBR1', 'IFNAR1']

# --- 1. Create the full aggregated graph with weights for pathfinding ---
G_full_agg = nx.DiGraph()
for _, r in edges_agg_df.iterrows():
    # The weight for Dijkstra's should be a cost, so we use 1 - frequency
    G_full_agg.add_edge(r['source_symbol'], r['target_symbol'], weight=(1.0 - r['models_freq']), **r.to_dict())

if not G_full_agg.nodes() or any(s not in G_full_agg for s in SOURCES):
    print2log("Full aggregated graph is empty or missing a source node. Exiting.")
    sys.exit()

# --- 2. Find the best paths from sources to TFs and between sources ---
print2log("--- Finding best paths based on cumulative frequency ---")
nodes_on_best_paths = set()
sources_in_graph = [s for s in SOURCES if s in G_full_agg]
tfs_in_graph = [t for t in top10_tfs if t in G_full_agg]

if not sources_in_graph or not tfs_in_graph:
    print2log("Warning: Sources or TFs are not present in the aggregate graph. Final network may be empty.")

# Find best path for each Source -> TF pair
for source in sources_in_graph:
    for tf in tfs_in_graph:
        try:
            if nx.has_path(G_full_agg, source, tf):
                # 'weight' was set to 1-frequency, so shortest_path finds the path maximizing frequency sum
                path = nx.shortest_path(G_full_agg, source=source, target=tf, weight='weight')
                nodes_on_best_paths.update(path)
        except nx.NetworkXNoPath:
            continue

# Find best path between the two sources to ensure they are connected
if len(sources_in_graph) == 2:
    s1, s2 = sources_in_graph
    try:
        # Search for this path in an undirected version of the graph to find any connection
        if nx.has_path(G_full_agg.to_undirected(), s1, s2):
            path_between_sources = nx.shortest_path(G_full_agg.to_undirected(), source=s1, target=s2, weight='weight')
            nodes_on_best_paths.update(path_between_sources)
    except nx.NetworkXNoPath:
        print2log(f"Warning: Could not find a connecting path between sources {s1} and {s2}.")

# --- 3. Construct the final network ---
if not nodes_on_best_paths:
    print2log("Could not find any paths from Sources to TFs. The final graph will be empty.")
    G_filt = nx.DiGraph()
else:
    print2log(f"Found {len(nodes_on_best_paths)} nodes on best paths.")
    G_filt = G_full_agg.subgraph(nodes_on_best_paths).copy()
    
    # As a final safeguard, ensure the graph is connected. Take the largest component containing a source.
    if not nx.is_weakly_connected(G_filt):
        print2log("Best paths formed disconnected components. Selecting the largest component containing a source.")
        largest_comp_nodes = set()
        for component in nx.weakly_connected_components(G_filt):
            # Check if any source is in this component and if it's the largest found so far
            if any(s in component for s in sources_in_graph) and len(component) > len(largest_comp_nodes):
                largest_comp_nodes = component
        
        if not largest_comp_nodes: # Fallback if no component had a source
            largest_comp_nodes = max(nx.weakly_connected_components(G_filt), key=len)

        G_filt = G_filt.subgraph(largest_comp_nodes).copy()


# --- Post-processing and saving the final filtered graph ---
covered_tfs = [t for t in top10_tfs if t in G_filt]
print2log(f"\nFinal filtered graph nodes: {G_filt.number_of_nodes()}, edges: {G_filt.number_of_edges()}")
print2log(f"Top TFs present in final graph: {covered_tfs}")

# --- Create and save node attributes file ---
print2log("Creating node attributes file...")
all_tf_uniprots = TFOutput.columns
all_tf_symbols = {to_symbol(uniprot) for uniprot in all_tf_uniprots}

node_attributes = []
if G_filt.number_of_nodes() > 0:
    for node in G_filt.nodes():
        node_type = 'Intermediate'
        if node in SOURCES:
            node_type = 'Source'
        elif node in all_tf_symbols:
            node_type = 'TF'
        node_attributes.append({'node_symbol': node, 'node_type': node_type})

nodes_df = pd.DataFrame(node_attributes).sort_values('node_symbol').reset_index(drop=True) if node_attributes else pd.DataFrame(columns=['node_symbol', 'node_type'])
nodes_df.to_csv('results/filtered_nodes_attributes.csv', index=False)
print2log("Saved node attributes to results/filtered_nodes_attributes.csv")

# --- Create and save final edge file ---
filtered_rows = []
for u, v, d in G_filt.edges(data=True):
    mean_weight = d.get('mean_weight', 0.0)
    interaction = '+' if mean_weight > 0 else '-' if mean_weight < 0 else 'neutral'
    filtered_rows.append({
        'source_symbol': u,
        'target_symbol': v,
        'models_count': d.get('models_count'),
        'models_freq': d.get('models_freq'),
        'mean_weight': mean_weight,
        'mean_abs_weight': d.get('mean_abs_weight'),
        'interaction': interaction
    })

if filtered_rows:
    filtered_df = pd.DataFrame(filtered_rows).sort_values(
        ['models_freq', 'models_count', 'source_symbol', 'target_symbol'],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)
else:
    filtered_df = pd.DataFrame(columns=[
        'source_symbol', 'target_symbol', 'models_count', 'models_freq',
        'mean_weight', 'mean_abs_weight', 'interaction'
    ])

filtered_df.to_csv('results/subgraph_edges_aggregate_filtered.csv', index=False)
print2log("Saved filtered aggregate network to results/subgraph_edges_aggregate_filtered.csv")

print2log("\nDone")
