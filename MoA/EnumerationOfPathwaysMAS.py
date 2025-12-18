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
from utility import *
import networkx as nx
import pyreadr
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from itertools import islice
from tqdm import tqdm
import pickle
import time

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

def analyze_paths_with_normalization(grid_path_scores_vector):
    """
    Compute length-normalized path scores (geometric mean per edge).
    Returns normalized_scores dictionary.
    """
    normalized_scores = {}
    for path_edges, scores in grid_path_scores_vector.items():
        path_length = len(path_edges) or 1

        normalized_scores[path_edges] = np.array([
            np.sign(s) * (abs(s) ** (1.0 / path_length)) if s != 0 else 0.0
            for s in scores
        ], dtype=float)

    if not grid_path_scores_vector:
        return normalized_scores

    return normalized_scores

def compute_edge_gradients_grid(model, node_idx_map, sources, pheno_coefs, outName, 
                                stimulation_strength=3000, num_steps=21, 
                                aggregation='range', print_func=print):
    """
    Compute edge gradients over a grid of stimulations for robustness.
    """
    
    print_func(f"\n🔬 Computing edge gradients over {num_steps}x{num_steps} stimulation grid...")
    
    model.eval()
    device = next(model.parameters()).device
    
    if len(sources) != 2:
        raise ValueError("Currently only supports 2 sources for grid")
    
    source_idx_1 = node_idx_map[sources[0]]
    source_idx_2 = node_idx_map[sources[1]]
    num_nodes = len(node_idx_map)
    
    stim_range = torch.linspace(-stimulation_strength, stimulation_strength, int(num_steps))
    grid_stim1, grid_stim2 = torch.meshgrid(stim_range, stim_range, indexing='ij')
    grid_stim1 = grid_stim1.flatten().view(-1, 1)
    grid_stim2 = grid_stim2.flatten().view(-1, 1)
    
    Xin_grid = torch.zeros((num_steps**2, num_nodes), dtype=torch.double, device=device)
    Xin_grid[:, source_idx_1] = grid_stim1.squeeze()
    Xin_grid[:, source_idx_2] = grid_stim2.squeeze()
    
    model.zero_grad()
    Y = model(Xin_grid)
    
    if aggregation == 'sum':
        L_objective = torch.sum(Y)
    elif aggregation == 'range':
        L_objective = torch.max(Y) - torch.min(Y)
    elif aggregation == 'mean':
        L_objective = torch.mean(Y)
    elif aggregation == 'max':
        L_objective = torch.max(torch.abs(Y))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    L_objective.backward()
    
    dW = model[0].weights.grad.detach().cpu().numpy()
    dW_abs = np.abs(dW)
    
    node_names = list(node_idx_map.keys())
    target_indices, source_indices = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes), indexing='ij')
    
    edge_gradients = {
        (node_names[src_idx], node_names[tgt_idx]): float(dW_abs[tgt_idx, src_idx])
        for src_idx, tgt_idx in zip(source_indices.ravel(), target_indices.ravel())
    }
    
    for tf_name in outName:
        if isinstance(outName, np.ndarray):
            tf_idx = int(np.where(outName == tf_name)[0][0])
        else:
            tf_idx = outName.index(tf_name)

        if isinstance(pheno_coefs, torch.Tensor):
            grad_val = abs(pheno_coefs[tf_idx].item())
        else:
            grad_val = abs(float(pheno_coefs[tf_idx]))
        
        edge_gradients[(tf_name, 'MAS')] = grad_val
    
    model.zero_grad()
    
    return edge_gradients

def compute_single_step_jacobian_efficient(bionet_layer, x_equilibrium, x_input):
    weights = bionet_layer.weights.data
    bias = bionet_layer.bias.data
    leak = bionet_layer.training_params['leak']
    X_bias = x_input.T + bias
    z = torch.mm(weights, x_equilibrium.T) + X_bias
    derivative = bionet_layer.delta(z.T, leak)
    jacobian = derivative.unsqueeze(-1) * weights
    return jacobian

def compute_tfs_jacobian_batch(model, x_equilibrium_batch, outName):
    num_nodes = x_equilibrium_batch.shape[1]
    num_tfs = len(outName)
    batch_size = x_equilibrium_batch.shape[0]
    
    x_eq_grad = x_equilibrium_batch.clone().requires_grad_(True)
    tf_acts = model[1](x_eq_grad)
    grad_tfs_wrt_nodes = torch.zeros((batch_size, num_tfs, num_nodes), dtype=torch.double, device=x_eq_grad.device)
    
    for tf_idx in range(num_tfs):
        tf_loss = tf_acts[:, tf_idx].sum()
        grad = torch.autograd.grad(outputs=tf_loss, inputs=x_eq_grad, retain_graph=True)[0]
        grad_tfs_wrt_nodes[:, tf_idx, :] = grad.clone()

    return grad_tfs_wrt_nodes

def compute_path_gradient_from_jacobian(path_edges, node_idx_map, jacobian_batch,
                                        grad_tfs_wrt_nodes, pheno_coefs, outName):
    batch_size = jacobian_batch.shape[0]
    log_abs_gradient = torch.zeros(batch_size, dtype=torch.double, device=jacobian_batch.device)
    sign_accumulator = torch.ones(batch_size, dtype=torch.double, device=jacobian_batch.device)
    current_node_idx = node_idx_map[path_edges[0][0]]
    
    for (u, v) in path_edges:
        if v == 'MAS':
            tf_name = u
            if isinstance(outName, np.ndarray):
                tf_idx = int(np.where(outName == tf_name)[0][0])
            else:
                tf_idx = outName.index(tf_name)
            
            grad_MAS_wrt_tf = pheno_coefs[tf_idx]
            grad_tf_wrt_current_batch = grad_tfs_wrt_nodes[:, tf_idx, current_node_idx]
            derivative = grad_MAS_wrt_tf * grad_tf_wrt_current_batch
        else:
            v_idx = node_idx_map[v]
            derivative = jacobian_batch[:, v_idx, current_node_idx]
            current_node_idx = v_idx
        
        log_abs_gradient += torch.log(torch.abs(derivative) + 1e-100)
        sign_accumulator *= torch.sign(derivative)
    
    path_gradient_batch = sign_accumulator * torch.exp(log_abs_gradient)
    return path_gradient_batch

def score_paths_vectorized_jacobian(model, all_paths, node_idx_map,
                                    x_baseline_batch, x_stim_batch, pheno_coefs, outName,
                                    num_steps_ig=50, show_progress=True, print_func=print):
    model.eval()
    batch_size = x_stim_batch.shape[0]
    path_list = list(all_paths)
    num_paths = len(path_list)
    
    path_score_matrix = torch.zeros((num_paths, batch_size), dtype=torch.double, device=x_stim_batch.device)
    
    iterator = range(1, num_steps_ig + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Vectorized IG Steps", 
                       initial=1, total=num_steps_ig,
                       leave=False, ncols=100)
    
    for step in iterator:
        alpha = step / num_steps_ig
        x_alpha_batch = x_baseline_batch + alpha * (x_stim_batch - x_baseline_batch)
        
        with torch.no_grad():
            x_equilibrium_batch = model[0](x_alpha_batch)
        
        jacobian_batch = compute_single_step_jacobian_efficient(model[0], x_equilibrium_batch, x_alpha_batch)
        grad_tfs_wrt_nodes = compute_tfs_jacobian_batch(model, x_equilibrium_batch, outName)
        
        for i, path_edges in enumerate(path_list):
            path_gradient_batch = compute_path_gradient_from_jacobian(
                path_edges, node_idx_map, jacobian_batch,
                grad_tfs_wrt_nodes, pheno_coefs, outName
            )
            path_score_matrix[i, :] += path_gradient_batch
            
    final_path_scores_grid = {}

    for i, path_edges in enumerate(path_list):
        source_node = path_edges[0][0]
        source_idx = node_idx_map[source_node]
        delta_source_batch = (x_stim_batch - x_baseline_batch)[:, source_idx]
        final_score_vector = torch.zeros_like(delta_source_batch)
        non_zero_mask = delta_source_batch.abs() > 1e-10
        
        if non_zero_mask.any():
            final_score_vector[non_zero_mask] = (
                (path_score_matrix[i, non_zero_mask] / num_steps_ig) * delta_source_batch[non_zero_mask]
            )
        
        final_path_scores_grid[path_edges] = final_score_vector.cpu().numpy()
    
    return final_path_scores_grid

def setup_dual_logging(out_dir='.', log_filename='pathway_enumeration.log', use_timestamp=True):
    import logging
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    if use_timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(log_filename)
        log_filename = f"{base}_{timestamp}{ext}"
    
    logger = logging.getLogger('PathwayEnumeration')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    
    log_path = os.path.join(out_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger.info, log_path

parser = argparse.ArgumentParser(prog='Hepatocyte signaling, pathways enumeration')
parser.add_argument('--log_filename', action='store', type=str, default='pathway_enumeration.log')
parser.add_argument('--enumerated_pathways_path', action='store', type=str, required=True,
                    help='Base path for pickle file to save enumerated pathways.')
parser.add_argument('--AnnotationCSV', action='store', type=str, default='../data/annotation.csv')
parser.add_argument('--DrugsIn', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut', action='store',type = str, default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', action='store',type = str, default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', action='store',type = str, default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--infoPath', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--res_dir', action='store',type = str, default='../results/')
parser.add_argument('--out_dir', action='store',type = str, default='results/')
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
parser.add_argument('--max_path_length', action='store',type = int, default=8)
parser.add_argument('--stimulation_strength', action='store',type=float, default=3000)
parser.add_argument('--num_steps_ig', action='store',type = float, default=100)
parser.add_argument('--GRADIENT_THRESHOLD', action='store',type = float, default=8e-3)

args = parser.parse_args()

# Configuration constants
MAX_PATHS_PER_BATCH = 1200
TOP_K_KEEP = 250
AI_SCORE_INDEX = 3  # Index of combined stimulation score in the vector

print2log, log_file_path = setup_dual_logging(log_filename=args.log_filename, use_timestamp=True)
utils.set_cores(12)
device = "cuda" if torch.cuda.is_available() else "cpu"
print2log(f'Running on {device}')

# Data Loading (Annotation, PKN, Drugs, etc.)
ann = pd.read_csv(args.AnnotationCSV)
id2sym = dict(zip(ann['uniprot'].astype(str), ann['symbol'].astype(str)))

net = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[:,['source','target','stimulation','inhibition']]
drugInput = pd.read_csv(args.DrugsIn, sep='\t', index_col=0)
TFOutput = pd.read_csv(args.TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput = TFOutput.loc[drugInput.index,:]
net = format_network(net, weight_label='mode_of_action', stimulation_label='stimulation', inhibition_label='inhibition')
nodeNames = np.union1d(net.source.values, net.target.values)

for timepoint in ['8 hour']:
    print2log(f'Processing timepoint {timepoint}')
    enumerated_pathways = {}
    enumerated_pathways_path_timepoint = args.enumerated_pathways_path + '_' + timepoint + '.pkl'
    
    data_path = args.res_dir + ('allmodels/' if timepoint == '8 hour' else f'{timepoint}/allmodels/')
    
    inputInfo = pd.read_csv(args.infoPath, sep='\t', low_memory=False, index_col=0)
    inputInfo = inputInfo[inputInfo['time']==timepoint]
    
    TFOutput_tmp = TFOutput.loc[inputInfo.index,:]
    drugInput_tmp = drugInput.loc[inputInfo.index,:]
    drugTargets = pd.read_csv(args.TargetsIn, sep='\t', low_memory=False, index_col=0)
    
    inName = np.intersect1d(nodeNames, drugTargets.columns.values)
    outName = np.intersect1d(nodeNames, TFOutput_tmp.columns.values)
    
    TFOutput_tmp = TFOutput_tmp.loc[:, outName]
    drugTargets = drugTargets.loc[drugInput_tmp.columns.values, inName]
    
    if args.ConvertToEmpProb:
        TFOutput_tmp = 1/(1+np.exp(-TFOutput_tmp))
    
    # Pre-train Phenotype Model (Ridge) - UPDATED TO MATCH PathwayScoring SCRIPT EXACTLY
    Xh = pyreadr.read_r(args.X_invivo_path)[None]
    Xh = Xh.loc[:, outName].to_numpy()
    Yh = pyreadr.read_r(args.Y_invivo_path)[None].to_numpy()[:,0]
    
    # Updated Grid Search parameters to match PathwayScoring script
    # pls = PLSRegression(n_components=8, scale=False)
    # pls.fit(Xh, Yh.reshape(-1, 1))
    # coefs = pls.coef_
    # intercept = pls.intercept_
    # pheno = FrozenLinear(coefs, intercept, dtype=torch.double, device=device)
    params = {'alpha': [0.001,0.01,0.1,0.2,0.5,1,2,5,10,20,50,100,500,1000]}
    ridge = GridSearchCV(estimator=Ridge(), param_grid=params, cv=RepeatedKFold(n_splits=10, n_repeats=10), n_jobs=-1)
    ridge.fit(Xh, Yh)
    coefs = ridge.best_estimator_.coef_
    pheno = FrozenLinear(coefs[np.newaxis, :], ridge.best_estimator_.intercept_, dtype=torch.double, device=device)
    pheno.eval()
    
    for model_no in range(args.number_DTLEMBAS_models):
        print2log(f"MODEL {model_no+1}/{args.number_DTLEMBAS_models}")
        
        # Load Model
        mod = SignalingModel(net=net, X_in=drugTargets, y_out=TFOutput_tmp, 
                             projection_amplitude_in=args.inputAmplitude, 
                             projection_amplitude_out=args.projectionAmplitude,
                             weight_label='mode_of_action', source_label='source', target_label='target',
                             bionet_params={'target_steps': 10, 'max_steps': 10, 'exp_factor': 10, 'tolerance': 1e-1, 'leak': 0.01}, 
                             dtype=torch.double, device=device)
        
        mod.load_state_dict(torch.load(os.path.join(data_path, args.modelPattern + str(model_no+1)+'signal_state_dict_%s.pth'%(model_no+1))))
        mod.eval()
        mod.signaling_network.weights.data.masked_fill_(mask=mod.signaling_network.mask, value=0.0)
        
        # Build Graph
        weights = mod.signaling_network.weights.detach().cpu().numpy()
        A = pd.DataFrame(weights.T, index=nodeNames, columns=nodeNames)
        A = pd.concat([A, pd.DataFrame(np.zeros((len(nodeNames),1)), index=nodeNames, columns=['MAS'])], axis=1)
        A = pd.concat([A, pd.DataFrame(np.zeros((1,len(nodeNames)+1)), index=['MAS'], columns=list(nodeNames)+['MAS'])], axis=0)
        A.loc[outName, 'MAS'] = coefs[:]
        G = nx.from_pandas_adjacency(A, create_using=nx.DiGraph())
        
        model = torch.nn.Sequential(mod.signaling_network, mod.output_layer, pheno)
        model.eval()
        
        # Setup Inputs for Scoring
        tgfbr1_idx = mod.node_idx_map['P36897']
        ifnar1_idx = mod.node_idx_map['P17181']
        
        x_base = torch.zeros((1, len(mod.node_idx_map)), dtype=torch.double).to(device)
        # 4 Critical Inputs: Baseline, TGFBR1 only, IFNAR1 only, Combined
        x_inputs = [x_base.clone() for _ in range(4)]
        x_inputs[1][0, tgfbr1_idx] = args.stimulation_strength
        x_inputs[2][0, ifnar1_idx] = args.stimulation_strength
        x_inputs[3][0, tgfbr1_idx] = args.stimulation_strength
        x_inputs[3][0, ifnar1_idx] = args.stimulation_strength
        
        Xin_critical = torch.cat(x_inputs, dim=0).detach()
        x_baseline_batch = torch.zeros_like(Xin_critical).to(device)

        # Compute Gradients to Filter Graph
        edge_gradients = compute_edge_gradients_grid(
            model=model, node_idx_map=mod.node_idx_map, sources=['P36897', 'P17181'],
            pheno_coefs=torch.tensor(coefs, dtype=torch.double, device=device),
            outName=outName, stimulation_strength=args.stimulation_strength,
            print_func=print2log
        )
        
        G_filtered = nx.DiGraph([
            (u, v, {'weight': grad}) for (u, v), grad in edge_gradients.items()
            if grad >= args.GRADIENT_THRESHOLD and (v != 'MAS' and G.has_edge(u, v) or v == 'MAS')
        ])
        
        final_paths_for_model = []
        
        for source in ['P36897', 'P17181']:
            path_gen = nx.all_simple_paths(G_filtered, source, 'MAS', cutoff=args.max_path_length)
            
            # Fetch first batch
            current_pool_nodes = list(islice(path_gen, MAX_PATHS_PER_BATCH))
            
            # Check if we maxed out
            if len(current_pool_nodes) < MAX_PATHS_PER_BATCH:
                # Case: Did not max out. Keep all. Do NOT score.
                print2log(f"   {source}: Found {len(current_pool_nodes)} paths (Did not max out). Keeping all.")
                final_paths_for_model.extend([tuple(nx.utils.pairwise(p)) for p in current_pool_nodes])
                continue
                
            print2log(f"   {source}: Maxed out initial batch ({MAX_PATHS_PER_BATCH}). Starting batch-wise pruning.")
            
            # Process the first batch (already fetched)
            current_batch_edges = [tuple(nx.utils.pairwise(p)) for p in current_pool_nodes]
            
            # Score
            scores_raw = score_paths_vectorized_jacobian(
                model=model, all_paths=current_batch_edges, node_idx_map=mod.node_idx_map,
                x_baseline_batch=x_baseline_batch, x_stim_batch=Xin_critical,
                pheno_coefs=torch.tensor(coefs, dtype=torch.double, device=device),
                outName=outName, num_steps_ig=int(args.num_steps_ig), show_progress=True
            )
            scores_norm = analyze_paths_with_normalization(scores_raw)
            # Sort & Keep Top K
            current_batch_edges.sort(key=lambda p: abs(scores_norm[p][AI_SCORE_INDEX]), reverse=True)
            kept_paths = current_batch_edges[:TOP_K_KEEP]
            final_paths_for_model.extend(kept_paths)
            print2log(f"     ...Batch 1: Kept {len(kept_paths)} paths.")

            # Process subsequent batches
            batch_idx = 1
            while batch_idx<10: # don't keep iterating forever! If we reach 10 batches (which will be 10 * topk paths) we shall stop
                next_batch_nodes = list(islice(path_gen, MAX_PATHS_PER_BATCH))
                if not next_batch_nodes:
                    break
                
                batch_idx += 1
                next_batch_edges = [tuple(nx.utils.pairwise(p)) for p in next_batch_nodes]
                
                # Score
                scores_raw = score_paths_vectorized_jacobian(
                    model=model, all_paths=next_batch_edges, node_idx_map=mod.node_idx_map,
                    x_baseline_batch=x_baseline_batch, x_stim_batch=Xin_critical,
                    pheno_coefs=torch.tensor(coefs, dtype=torch.double, device=device),
                    outName=outName, num_steps_ig=int(args.num_steps_ig), show_progress=True
                )
                scores_norm = analyze_paths_with_normalization(scores_raw)
                # Sort & Keep Top K
                next_batch_edges.sort(key=lambda p: abs(scores_norm[p][AI_SCORE_INDEX]), reverse=True)
                kept_paths = next_batch_edges[:TOP_K_KEEP]
                final_paths_for_model.extend(kept_paths)
                print2log(f"     ...Batch {batch_idx} ({len(next_batch_nodes)} paths): Kept {len(kept_paths)} paths.")
            
            print2log(f"   {source}: Finished pruning. Kept {len(final_paths_for_model)} paths so far (including previous sources).")
            
        enumerated_pathways[model_no] = final_paths_for_model
        
        # Save final dictionary
        os.makedirs(os.path.dirname(enumerated_pathways_path_timepoint), exist_ok=True)
        with open(enumerated_pathways_path_timepoint, 'wb') as f:
            pickle.dump(enumerated_pathways, f)
    print2log(f"Saved all enumerated pathways to {enumerated_pathways_path_timepoint}")