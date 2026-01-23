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
from pathlib import Path
from utility import *
import networkx as nx
from matplotlib import pyplot as plt
# import pcst_fast
import pyreadr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,RepeatedKFold
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
import pickle
import hashlib
import time
from captum.attr import IntegratedGradients

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
    
def analyze_paths_with_normalization(grid_path_scores_vector, print_func):
    """
    Compute length-normalized path scores (geometric mean per edge)
    and print a brief sanity comparison between raw and normalized
    'synergy' rankings.

    Returns
    -------
    normalized_scores : dict
        {path_edges: np.array([baseline, A_norm, I_norm, AI_norm])}
        with the same keys as grid_path_scores_vector.
    """
    A_idx, I_idx, AI_idx = 1, 2, 3

    # Compute normalized scores
    normalized_scores = {}
    for path_edges, scores in grid_path_scores_vector.items():
        path_length = len(path_edges) or 1

        normalized_scores[path_edges] = np.array([
            np.sign(s) * (abs(s) ** (1.0 / path_length)) if s != 0 else 0.0
            for s in scores
        ], dtype=float)

    if not grid_path_scores_vector:
        print_func("\n⚠️  No paths found for normalization.")
        return normalized_scores

    def synergy(vec: np.ndarray) -> float:
        # AI - A - I
        return float(vec[AI_idx] - vec[A_idx] - vec[I_idx])

    # Top path by RAW synergy
    top_raw_path, top_raw_scores = max(
        grid_path_scores_vector.items(),
        key=lambda x: abs(synergy(x[1]))
    )

    # Top path by NORMALIZED synergy
    top_norm_path, top_norm_scores = max(
        normalized_scores.items(),
        key=lambda x: abs(synergy(x[1]))
    )

    # print_func("\n=== RAW vs LENGTH-NORMALIZED COMPETITIVE PATH (sanity check) ===")
    # print_func(f"  RAW top-synergy path length: {len(top_raw_path)} edges")
    # print_func(f"    RAW synergy: {synergy(top_raw_scores):+.6f}")
    # print_func(f"  NORM top-synergy path length: {len(top_norm_path)} edges")
    # print_func(f"    NORM synergy: {synergy(top_norm_scores):+.6f}")

    # Also show raw synergy of the normalized top path
    raw_for_norm = grid_path_scores_vector[top_norm_path]
    # print_func(f"  RAW synergy of NORM top path: {synergy(raw_for_norm):+.6f}")

    return normalized_scores

def select_top_paths_with_rank_tracking(grid_path_scores_raw,
                                        grid_path_scores_norm,
                                        print_func):
    """
    Select ONE top path per source (TGFBR1, IFNAR1) based on the
    LENGTH-NORMALIZED combined effect (AI_norm), but report both
    RAW and NORMALIZED scores and their solo ranks.

    Parameters
    ----------
    grid_path_scores_raw : dict
        {path_edges: np.array([baseline, A_raw, I_raw, AI_raw])}
    grid_path_scores_norm : dict
        {path_edges: np.array([baseline, A_norm, I_norm, AI_norm])}
    """

    A_idx, I_idx, AI_idx = 1, 2, 3

    print_func('\n' + '='*70)
    print_func('SELECTING TOP PATHS (NORMALIZED RANKING + RAW REPORTING)')
    print_func('='*70)

    def _analyze_source(source_uniprot: str,
                        label: str,
                        solo_idx: int):
        # Subset paths for this receptor
        paths_raw = {k: v for k, v in grid_path_scores_raw.items()
                     if k[0][0] == source_uniprot}
        paths_norm = {k: v for k, v in grid_path_scores_norm.items()
                      if k[0][0] == source_uniprot}

        # Ensure same keys in both dictionaries
        common_keys = set(paths_raw.keys()) & set(paths_norm.keys())
        paths_raw = {k: paths_raw[k] for k in common_keys}
        paths_norm = {k: paths_norm[k] for k in common_keys}

        if not paths_raw:
            print_func(f"\n⚠️  No paths found for {label}.")
            return None

        # Rank all paths by SOLO effect (RAW)
        solo_ranked_raw = sorted(
            paths_raw.items(),
            key=lambda x: abs(x[1][solo_idx]),
            reverse=True
        )

        # Rank all paths by SOLO effect (NORMALIZED)
        solo_ranked_norm = sorted(
            paths_norm.items(),
            key=lambda x: abs(x[1][solo_idx]),
            reverse=True
        )

        # Select top path by NORMALIZED combined effect (AI_norm)
        top_path_edges, top_scores_norm = max(
            paths_norm.items(),
            key=lambda x: abs(x[1][AI_idx])
        )
        top_scores_raw = paths_raw[top_path_edges]

        # Solo ranks of this path
        solo_rank_raw = next(
            i for i, (p, _) in enumerate(solo_ranked_raw, 1)
            if p == top_path_edges
        )
        solo_rank_norm = next(
            i for i, (p, _) in enumerate(solo_ranked_norm, 1)
            if p == top_path_edges
        )

        # Synergy in raw and normalized space
        synergy_raw = (top_scores_raw[AI_idx] -
                       top_scores_raw[A_idx] -
                       top_scores_raw[I_idx])
        synergy_norm = (top_scores_norm[AI_idx] -
                        top_scores_norm[A_idx] -
                        top_scores_norm[I_idx])

        # Pretty path string
        show_len = min(3, len(top_path_edges))
        path_str = " → ".join([e[0] for e in top_path_edges[:show_len]])
        if len(top_path_edges) > show_len:
            path_str += " ... → MAS"
        else:
            path_str += " → MAS"

        # ---- Printing ----
        print_func(f"\n{label} TOP PATH (selected by *normalized* combined effect):")
        print_func(f"   Path length: {len(top_path_edges)} edges")
        print_func(f"   Path: {path_str}")

        # print_func(f"\n   RAW SCORES:")
        # print_func(f"     TGFBR1-only: {top_scores_raw[A_idx]:+.6f}")
        # print_func(f"     IFNAR1-only: {top_scores_raw[I_idx]:+.6f}")
        # print_func(f"     Combined:    {top_scores_raw[AI_idx]:+.6f}")
        # print_func(f"     Synergy:     {synergy_raw:+.6f}")

        # print_func(f"\n   NORMALIZED SCORES (geometric mean per edge):")
        # print_func(f"     TGFBR1-only: {top_scores_norm[A_idx]:+.6f}")
        # print_func(f"     IFNAR1-only: {top_scores_norm[I_idx]:+.6f}")
        # print_func(f"     Combined:    {top_scores_norm[AI_idx]:+.6f} ⭐ (selection criterion)")
        # print_func(f"     Synergy:     {synergy_norm:+.6f}")

        total_paths = len(paths_raw)
        print_func(f"\n   SOLO RANKING:")
        print_func(f"     RAW solo rank:  #{solo_rank_raw} / {total_paths}")
        print_func(f"     NORM solo rank: #{solo_rank_norm} / {total_paths}")

        # if solo_rank_raw > 10:
        #     print_func(f"   ⚠️  RAW: ranks #{solo_rank_raw} when acting alone.")
        #     print_func(f"       → Importance emerges mainly with co-stimulation.")
        # else:
        #     print_func(f"   ✅ RAW: also strong when acting alone (rank #{solo_rank_raw}).")

        return {
            'path': top_path_edges,
            'raw_scores': top_scores_raw,
            'norm_scores': top_scores_norm,
            'synergy_raw': synergy_raw,
            'synergy_norm': synergy_norm,
            'solo_rank_raw': solo_rank_raw,
            'solo_rank_norm': solo_rank_norm,
            'total_paths': total_paths,
        }

    # Run per receptor
    tgfbr1_res = _analyze_source('P36897', '🔴 TGFBR1', solo_idx=A_idx)
    ifnar1_res = _analyze_source('P17181', '🔵 IFNAR1', solo_idx=I_idx)

    return {
        'tgfbr1': tgfbr1_res,
        'ifnar1': ifnar1_res,
    }



def compute_edge_gradients(model, x_stim, node_idx_map, pheno_coefs, outName, print_func=print):
    """
    Compute ∂output/∂edge for every edge in the network.
    This gives us the TRUE importance of each edge.
    """
    
    # Get equilibrium
    with torch.no_grad():
        x_eq = model[0](x_stim)
    
    # Compute full model output
    model.eval()
    x_eq_grad = x_eq.clone().requires_grad_(True)
    output = model[1:](x_eq_grad)  # TF layer + phenotype
    
    # Backprop to get ∂output/∂x_eq
    grad_output_wrt_eq = torch.autograd.grad(
        outputs=output.sum(),
        inputs=x_eq_grad,
        create_graph=False
    )[0]  # [1, Nodes]
    
    # Now compute ∂x_eq/∂edge for each edge
    # Edge (u -> v) affects x_eq[v]
    
    num_nodes = len(node_idx_map)
    edge_gradients = {}
    
    weights = model[0].weights.data  # [Nodes, Nodes]
    
    # For each edge, compute its contribution
    for u_name, u_idx in node_idx_map.items():
        for v_name, v_idx in node_idx_map.items():
            if abs(weights[v_idx, u_idx].item()) < 1e-10:
                continue  # Skip zero edges
            
            # ∂output/∂w_{vu} = (∂output/∂x_eq[v]) * (∂x_eq[v]/∂w_{vu})
            # For recurrent networks, this is complex, but we can approximate:
            # ∂x_eq[v]/∂w_{vu} ≈ x_eq[u] (first-order approximation)
            
            edge_gradient = grad_output_wrt_eq[0, v_idx].item() * x_eq[0, u_idx].item()
            
            edge_gradients[(u_name, v_name)] = abs(edge_gradient)
    
    # Add TF -> phenotype edges
    for tf_name in outName:
        if isinstance(outName, np.ndarray):
            tf_idx = int(np.where(outName == tf_name)[0][0])
        else:
            tf_idx = outName.index(tf_name)
        
        # ∂output/∂TF = phenotype coefficient
        tf_gradient = abs(pheno_coefs[tf_idx].item())
        edge_gradients[(tf_name, 'MAS')] = tf_gradient
    
    # print_func(f"\n📊 Edge gradient statistics:")
    # all_grads = list(edge_gradients.values())
    # print_func(f"  Total edges with gradient: {len(all_grads)}")
    # print_func(f"  Max gradient: {max(all_grads):.4e}")
    # print_func(f"  Mean gradient: {np.mean(all_grads):.4e}")
    # print_func(f"  Min gradient: {min(all_grads):.4e}")
    
    return edge_gradients

def compute_edge_gradients_grid(model, node_idx_map, sources, pheno_coefs, outName, 
                                stimulation_strength=3000, num_steps=21, 
                                aggregation='range', print_func=print):
    """
    Compute edge gradients over a grid of stimulations for robustness.
    
    Args:
        model: Full model (signaling + TF + phenotype)
        node_idx_map: Dict mapping node names to indices
        sources: List of source nodes (e.g., ['P36897', 'P17181'])
        pheno_coefs: Phenotype coefficients
        outName: TF names
        stimulation_strength: Max stimulation amplitude
        num_steps: Grid resolution (num_steps x num_steps grid)
        aggregation: How to aggregate gradients ('sum', 'range', 'mean', 'max')
        print_func: Logging function
    
    Returns:
        edge_gradients: Dict[(u, v) -> gradient magnitude]
    """
    
    print_func(f"\n🔬 Computing edge gradients over {num_steps}x{num_steps} stimulation grid...")
    print_func(f"   Sources: {sources}")
    print_func(f"   Stimulation range: [{-stimulation_strength}, {stimulation_strength}]")
    print_func(f"   Aggregation method: {aggregation}")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get source indices
    if len(sources) != 2:
        raise ValueError("Currently only supports 2 sources for grid")
    
    source_idx_1 = node_idx_map[sources[0]]
    source_idx_2 = node_idx_map[sources[1]]
    num_nodes = len(node_idx_map)
    
    # Create grid of stimulations
    stim_range = torch.linspace(-stimulation_strength, stimulation_strength, int(num_steps))
    grid_stim1, grid_stim2 = torch.meshgrid(stim_range, stim_range, indexing='ij')
    # Flatten grid: (num_steps^2, 1)
    grid_stim1 = grid_stim1.flatten().view(-1, 1)
    grid_stim2 = grid_stim2.flatten().view(-1, 1)
    # Create input batch: (num_steps^2, num_nodes)
    Xin_grid = torch.zeros((num_steps**2, num_nodes), dtype=torch.double, device=device)
    Xin_grid[:, source_idx_1] = grid_stim1.squeeze()
    Xin_grid[:, source_idx_2] = grid_stim2.squeeze()
    
    # Zero gradients
    model.zero_grad()
    # Forward pass
    Y = model(Xin_grid)  # (num_steps^2, 1) or (num_steps^2,)
    
    # Choose objective based on aggregation
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
    
    print_func(f"   Objective value: {L_objective.item():.6f}")
    
    # Backward pass
    L_objective.backward()
    
    # Extract gradients - VECTORIZED VERSION
    dW = model[0].weights.grad.detach().cpu().numpy()  # (num_nodes, num_nodes)
    dW_abs = np.abs(dW)  # Take absolute value once
    
    # Create node name arrays for efficient indexing
    node_names = list(node_idx_map.keys())
    
    # Convert gradient matrix to dictionary using vectorized operations
    # Create meshgrid of indices
    target_indices, source_indices = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes), indexing='ij')
    
    # Flatten and create edge tuples efficiently
    edge_gradients = {
        (node_names[src_idx], node_names[tgt_idx]): float(dW_abs[tgt_idx, src_idx])
        for src_idx, tgt_idx in zip(source_indices.ravel(), target_indices.ravel())
    }
    
    # Add TF -> phenotype edges using pheno_coefs directly
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
    
    # Statistics
    all_grads = list(edge_gradients.values())
    # print_func(f"\n✅ Computed gradients for {len(edge_gradients)} edges")
    # print_func(f"   Max gradient: {max(all_grads):.4e}")
    # print_func(f"   Mean gradient: {np.mean(all_grads):.4e}")
    # print_func(f"   Min gradient: {min(all_grads):.4e}")
    
    # Zero gradients to clean up
    model.zero_grad()
    
    return edge_gradients

def find_top_gradient_weighted_paths(G, edge_gradients, source, target='MAS', 
                                     top_k=20, print_func=print):
    """
    Find top-k shortest paths weighted by INVERSE gradient magnitude.
    High gradient edges have LOW cost, so they're preferred.
    
    This is efficient because it uses Yen's algorithm (nx.shortest_simple_paths).
    """
    
    print_func(f"\n🔍 Finding top-{top_k} gradient-weighted paths from {source}...")
    
    # Create edge weights: cost = 1 / |gradient|
    # High gradient → low cost → preferred in shortest path
    G_weighted = nx.DiGraph()
    
    for (u, v), grad in edge_gradients.items():
        if G.has_edge(u, v):
            # weight = 1.0 / (grad + 1e-6)  # Inverse: high gradient = low cost
            weight = -np.log10(grad + 1e-6)  # Log: high gradient = low cost
            G_weighted.add_edge(u, v, weight=weight)
    
    print_func(f"   Filtered graph: {G_weighted.number_of_nodes()} nodes, {G_weighted.number_of_edges()} edges")
    
    # Check if path exists
    if not nx.has_path(G_weighted, source, target):
        print_func(f"   ⚠️  No path found from {source} to {target}")
        return []
    
    # Find k-shortest paths using Yen's algorithm (efficient!)
    try:
        paths = []
        for path in nx.shortest_simple_paths(G_weighted, source, target, weight='weight'):
            if len(paths) >= top_k:
                break
            
            # Convert to edge tuples
            path_edges = tuple(nx.utils.pairwise(path))
            paths.append(path_edges)
        
        print_func(f"✅ Found {len(paths)} gradient-weighted paths")
        
        # Optional: Show gradient info for top path
        if paths:
            top_path_gradient = 1.0
            for u, v in paths[0]:
                if (u, v) in edge_gradients:
                    top_path_gradient *= edge_gradients[(u, v)]
            print_func(f"   Top path gradient product: {top_path_gradient:.4e}")
        
        return paths
        
    except nx.NetworkXNoPath:
        print_func(f"   ⚠️  No path found from {source} to {target}")
        return []

def compute_single_step_jacobian_efficient(bionet_layer, x_equilibrium, x_input):
    """
    Compute single-step Jacobian at equilibrium.
    
    Args:
        bionet_layer: model[0]
        x_equilibrium: Equilibrium state reached
        x_input: The input that led to this equilibrium (x_alpha)
    """    
    weights = bionet_layer.weights.data
    bias = bionet_layer.bias.data
    leak = bionet_layer.training_params['leak']
    
    # X_bias from the ORIGINAL INPUT (used in forward pass)
    X_bias = x_input.T + bias
    
    # Pre-activation at equilibrium: z = W @ X^* + X_bias
    z = torch.mm(weights, x_equilibrium.T) + X_bias
    
    # Activation derivative
    derivative = bionet_layer.delta(z.T, leak)
    
    # Jacobian = diag(f'(z)) @ W
    # W is [Nodes, Nodes]
    # Jacobian Batch: [Batch, Nodes, Nodes] = [Batch, Nodes, 1] * [Nodes, Nodes]
    jacobian = derivative.unsqueeze(-1) * weights

    # jacobian_batch[k, i, j] = d(node i)_k / d(node j)_k
    
    return jacobian

def compute_tfs_jacobian_batch(model, x_equilibrium_batch, outName):
    """
    Computes the partial derivative of all TFs w.r.t. all recurrent nodes for the batch.
    
    Returns:
        grad_tfs_wrt_nodes: [Batch, Num_TFs, Num_Nodes]
    """
    num_nodes = x_equilibrium_batch.shape[1]
    num_tfs = len(outName)
    batch_size = x_equilibrium_batch.shape[0]
    
    x_eq_grad = x_equilibrium_batch.clone().requires_grad_(True)
    
    # TF activations: [Batch, Num_TFs]
    tf_acts = model[1](x_eq_grad)
    
    # Initialize full gradient tensor: [Batch, Num_TFs, Num_Nodes]
    grad_tfs_wrt_nodes = torch.zeros((batch_size, num_tfs, num_nodes), dtype=torch.double, device=x_eq_grad.device)
    
    # Loop over TFs (must loop because autograd requires scalar loss or single output)
    for tf_idx in range(num_tfs):
        # Sum the TF activity across the entire batch to create a scalar loss
        tf_loss = tf_acts[:, tf_idx].sum()
        
        # Calculate gradient of this summed TF loss w.r.t. the input (x_eq_grad)
        grad = torch.autograd.grad(
            outputs=tf_loss,
            inputs=x_eq_grad,
            retain_graph=True
        )[0] # grad shape is [Batch, Num_Nodes]
        
        # Store for all batches and this specific TF
        grad_tfs_wrt_nodes[:, tf_idx, :] = grad.clone()

    return grad_tfs_wrt_nodes


def compute_path_gradient_from_jacobian(path_edges, node_idx_map, jacobian_batch,
                                        grad_tfs_wrt_nodes, pheno_coefs, outName):
    """Compute path gradient using log-space to avoid underflow."""
    
    batch_size = jacobian_batch.shape[0]
    
    # Work in log space: log(product) = sum(logs)
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
        
        # Accumulate in log space
        log_abs_gradient += torch.log(torch.abs(derivative) + 1e-100)  # Avoid log(0)
        sign_accumulator *= torch.sign(derivative)
    
    # Convert back from log space
    path_gradient_batch = sign_accumulator * torch.exp(log_abs_gradient)
    
    return path_gradient_batch


def score_paths_vectorized_jacobian(model, all_paths, node_idx_map,
                                    x_baseline_batch, x_stim_batch, pheno_coefs, outName,
                                    num_steps_ig=50, show_progress=True,
                                    checkpoint_dir=None, checkpoint_key=None, 
                                    checkpoint_interval=15, print_func=print):
    """
    Scores all paths across the ENTIRE stimulation grid in a single vectorized IG run.
    Now with checkpointing support for long runs.
    
    Args:
        checkpoint_dir: Directory to save/load checkpoints (None = no checkpointing)
        checkpoint_key: Unique identifier for this computation
        checkpoint_interval: Save checkpoint every N steps
        print_func: Function for logging
    """
    
    model.eval()
    batch_size = x_stim_batch.shape[0]
    path_list = list(all_paths)
    num_paths = len(path_list)
    
    # Try to load existing checkpoint
    start_step = 1
    path_score_matrix = None
    
    if checkpoint_dir and checkpoint_key:
        loaded_step, loaded_matrix, loaded_metadata = load_ig_checkpoint(checkpoint_dir, checkpoint_key)
        
        if loaded_matrix is not None:
            # Validate checkpoint compatibility
            if (loaded_metadata['num_paths'] == num_paths and 
                loaded_metadata['batch_size'] == batch_size and
                loaded_metadata['num_steps_ig'] == num_steps_ig):
                
                start_step = loaded_step + 1
                path_score_matrix = loaded_matrix.to(x_stim_batch.device)
                print_func(f"  🔄 Resuming IG computation from step {start_step}/{num_steps_ig}")
            else:
                print_func(f"  ⚠️  Checkpoint found but incompatible (shape mismatch). Starting fresh.")
    
    # Initialize path score matrix if not loaded
    if path_score_matrix is None:
        path_score_matrix = torch.zeros((num_paths, batch_size), 
                                        dtype=torch.double, 
                                        device=x_stim_batch.device)
    
    # Metadata for checkpointing
    metadata = {
        'num_steps_ig': num_steps_ig,
        'num_paths': num_paths,
        'batch_size': batch_size,
        'num_nodes': x_stim_batch.shape[1]
    }
    
    # IG Loop with checkpointing
    iterator = range(start_step, num_steps_ig + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Vectorized IG Steps", 
                       initial=start_step-1, total=num_steps_ig,
                       leave=False, ncols=100)
    
    for step in iterator:
        alpha = step / num_steps_ig
        
        # 1. Vectorized Interpolation
        x_alpha_batch = x_baseline_batch + alpha * (x_stim_batch - x_baseline_batch)
        
        # 2. Get equilibrium state (model[0] handles recurrence internally)
        with torch.no_grad():
            x_equilibrium_batch = model[0](x_alpha_batch)
        
        # 3. Compute single-step Jacobian batch [Batch, Nodes, Nodes]
        jacobian_batch = compute_single_step_jacobian_efficient(model[0], x_equilibrium_batch, x_alpha_batch)
        # Check Jacobian magnitudes
        # print_func(f"  Jacobian stats: max={jacobian_batch.abs().max():.4e}, "
        #         f"mean={jacobian_batch.abs().mean():.4e}, "
        #         f"min={jacobian_batch.abs().min():.4e}")
        # 4. Compute TF gradients batch [Batch, Num_TFs, Num_Nodes]
        grad_tfs_wrt_nodes = compute_tfs_jacobian_batch(model, x_equilibrium_batch, outName)
        
        # 5. Score all paths for the entire batch
        for i, path_edges in enumerate(path_list):
            path_gradient_batch = compute_path_gradient_from_jacobian(
                path_edges, node_idx_map, jacobian_batch,
                grad_tfs_wrt_nodes, pheno_coefs, outName
            )
            # Accumulate the gradient contribution
            path_score_matrix[i, :] += path_gradient_batch
        
        # 6. Save checkpoint at intervals
        if checkpoint_dir and checkpoint_key and step % checkpoint_interval == 0:
            save_ig_checkpoint(checkpoint_dir, checkpoint_key, step, 
                             path_score_matrix, metadata, print_func)
            
            # Clean up old checkpoints (keep last 2)
            cleanup_old_checkpoints(checkpoint_dir, checkpoint_key, keep_last_n=2)
    
    # 7. Finalize IG scores (scale by 1/num_steps_ig and delta_x)
    final_path_scores_grid = {}

    if path_score_matrix.abs().max().item() < 1e-10:
        print_func("⚠️  WARNING: All accumulated gradients are near zero!")
        print_func("   Check that:")
        print_func("   • Model is producing non-zero equilibrium states")
        print_func("   • Jacobians are being computed correctly")
        print_func("   • TF gradients are non-zero")
    else:
        print_func(f"✓ Gradient magnitude: max={path_score_matrix.abs().max().item():.4e}, "
                  f"mean={path_score_matrix.abs().mean().item():.4e}")
    
    # After the IG loop, separate the scoring by source
    A_idx = 1   # TGFBR1 Only
    I_idx = 2   # IFNAR1 Only
    AI_idx = 3  # Combined

    for i, path_edges in enumerate(path_list):
        source_node = path_edges[0][0]
        source_idx = node_idx_map[source_node]
        
        # delta_source_batch is the vector of [Batch] changes in the source node's input
        delta_source_batch = (x_stim_batch - x_baseline_batch)[:, source_idx]
        
        # Final IG score = (Sum of Gradients / num_steps_ig) * delta_source_batch
        # BUT: Only for batch indices where delta is non-zero!
        # Otherwise we multiply valid gradients by zero
        
        final_score_vector = torch.zeros_like(delta_source_batch)
        non_zero_mask = delta_source_batch.abs() > 1e-10
        
        if non_zero_mask.any():
            final_score_vector[non_zero_mask] = (
                (path_score_matrix[i, non_zero_mask] / num_steps_ig) * 
                delta_source_batch[non_zero_mask]
            )
        
        final_path_scores_grid[path_edges] = final_score_vector.cpu().numpy()
    
    # 8. Save final checkpoint
    if checkpoint_dir and checkpoint_key:
        save_ig_checkpoint(checkpoint_dir, checkpoint_key, num_steps_ig, 
                         path_score_matrix, metadata, print_func)
        print_func(f"  ✅ IG computation complete. Final checkpoint saved.")
    
    return final_path_scores_grid


def analyze_critical_paths(grid_path_scores_vector, print_func=print):
    """
    Analyze competition patterns using the four critical stimulation points.
    
    Args:
        grid_path_scores_vector: Dict[path -> numpy array of scores (size 4)]
        ...
        
    Returns:
        competitive_paths: Paths with strong competition/synergy metrics
    """
    
    print_func("\n" + "="*70)
    print_func("ANALYZING COMPETITION PATTERNS USING CRITICAL POINTS")
    print_func("="*70)
    
    # Indices correspond to the order defined in the main loop:
    # 0: BASELINE (Zero stim)
    # 1: TGFBR1 Only (A)
    # 2: IFNAR1 Only (I)
    # 3: Combined (AI)
    
    A_idx = 1
    I_idx = 2
    AI_idx = 3
    
    competitive_paths = []
    
    for path_edges, score_vector in tqdm(grid_path_scores_vector.items(), desc="Analyzing competition", ncols=100):
        
        # 1. Extract Path IG scores for the three key transitions (Baseline -> Point X)
        score_A = score_vector[A_idx]    # Effect of A only
        score_I = score_vector[I_idx]    # Effect of I only
        score_AI = score_vector[AI_idx]  # Effect of A + I combined
        
        # 2. Calculate Expected Additive Effect
        # The true IG score for a path is the change from the true baseline (zero input), 
        # so we sum the individual effects.
        expected_additive = score_A + score_I
        
        # 3. Synergy/Interference Metric
        synergy = score_AI - expected_additive
        
        # 4. Total Dynamic Range (Competition Metric)
        dynamic_range = max(score_A, score_I, score_AI) - min(score_A, score_I, score_AI, 0)
        
        # Determine if the path is truly competitive (opposite signs)
        is_competitive = (np.sign(score_A) != np.sign(score_I))
        
        competitive_paths.append({
            'path': path_edges,
            'tgfbr1_only': score_A,
            'ifnar1_only': score_I,
            'combined': score_AI,
            'expected': expected_additive,
            'synergy': synergy,
            'dynamic_range': dynamic_range,
            'is_competitive': is_competitive
        })
    
    # Sort by competition strength (synergy/interference magnitude)
    competitive_paths.sort(key=lambda x: abs(x['synergy']), reverse=True)
    
    print_func(f"\n✓ Found {len([p for p in competitive_paths if p['is_competitive']])} paths with opposite directional effects (TGFBR1 vs IFNAR1).")
    print_func(f"✓ Sorted paths by synergy/interference magnitude (Max |Synergy|: {competitive_paths[0]['synergy']:+.4f})")
    
    return competitive_paths


def visualize_critical_paths_bar_TOP(top_tgfbr1_path, top_ifnar1_path, 
                                     grid_path_scores_vector, out_dir, 
                                     model_no, print_func=print2log):
    """
    Create bar chart for TOP paths only (TGFBR1 and IFNAR1).
    """
    
    A_idx = 1
    I_idx = 2
    AI_idx = 3
    
    paths_to_plot = []
    
    # Add TGFBR1 top path
    if top_tgfbr1_path[0]:
        scores = top_tgfbr1_path[1] if isinstance(top_tgfbr1_path[1], np.ndarray) else \
                 grid_path_scores_vector[top_tgfbr1_path[0]]
        paths_to_plot.append({
            'name': f"TGFBR1\n{top_tgfbr1_path[0][0][0]} → {top_tgfbr1_path[0][-1][1]}",
            'receptor': 'TGFBR1',
            'tgfbr1_only': scores[A_idx],
            'ifnar1_only': scores[I_idx],
            'combined': scores[AI_idx],
            'synergy': scores[AI_idx] - scores[A_idx] - scores[I_idx]
        })
    
    # Add IFNAR1 top path
    if top_ifnar1_path[0]:
        scores = top_ifnar1_path[1] if isinstance(top_ifnar1_path[1], np.ndarray) else \
                 grid_path_scores_vector[top_ifnar1_path[0]]
        paths_to_plot.append({
            'name': f"IFNAR1\n{top_ifnar1_path[0][0][0]} → {top_ifnar1_path[0][-1][1]}",
            'receptor': 'IFNAR1',
            'tgfbr1_only': scores[A_idx],
            'ifnar1_only': scores[I_idx],
            'combined': scores[AI_idx],
            'synergy': scores[AI_idx] - scores[A_idx] - scores[I_idx]
        })
    
    if not paths_to_plot:
        print_func("No top paths to plot")
        return
    
    # Create bar plot
    n = len(paths_to_plot)
    x = np.arange(n)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scores_array = np.array([[p['tgfbr1_only'], p['ifnar1_only'], p['combined']] 
                             for p in paths_to_plot])
    
    rects1 = ax.bar(x - width, scores_array[:, 0], width, label='TGFBR1 Only', color='tomato')
    rects2 = ax.bar(x, scores_array[:, 1], width, label='IFNAR1 Only', color='skyblue')
    rects3 = ax.bar(x + width, scores_array[:, 2], width, label='Combined (AI)', color='darkgreen')
    
    # Add labels
    ax.set_ylabel('Path Integrated Gradient Score', fontsize=12)
    ax.set_xlabel('Top Path per Receptor', fontsize=12)
    ax.set_title(f"Model {model_no+1}: TOP Path Contributions (Your Method)", fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p['name'] for p in paths_to_plot], fontsize=10)
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add synergy annotations
    for i, p in enumerate(paths_to_plot):
        ax.text(x[i] + width, scores_array[i, 2] * 1.05, 
                f"S:{p['synergy']:+.2f}", 
                ha='center', va='bottom', fontsize=9, color='purple', fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "barplots", f'TOP_paths_custom_method_model{model_no}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_func(f"✓ Saved TOP path visualization (custom method) to: {fig_path}")

def visualize_top_path_captum_validation(top_path_edges, your_score, captum_score, 
                                         receptor_name, out_dir, model_no, print_func):
    """
    Bar plot comparing YOUR method vs CAPTUM on the top path.
    """
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Your Chain-Rule\nMethod', 'Captum Masked\nModel']
    scores = [your_score, captum_score]
    colors = ['dodgerblue', 'orange']
    
    bars = ax.bar(methods, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:+.6f}',
                ha='center', va='bottom' if score > 0 else 'top', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Path IG Score', fontsize=12)
    ax.set_title(f"Model {model_no+1}: TOP {receptor_name} Path\nMethod Comparison", 
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "barplots", f'TOP_{receptor_name}_captum_validation_model{model_no}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_func(f"✓ Saved Captum validation plot for TOP {receptor_name} path: {fig_path}")

def visualize_subgraph(result_graph, sources, to_symbol, out_dir, model_no, print_func):
    """
    Visualize and save the subgraph.
    """
    
    plt.figure(figsize=(12, 12))
    
    # Layout
    pos = nx.spring_layout(result_graph, seed=42, k=2, iterations=50)
    
    # Draw base graph
    nx.draw_networkx_nodes(result_graph, pos, node_color='skyblue', node_size=600)
    nx.draw_networkx_edges(result_graph, pos, edge_color='dodgerblue', width=2.0, alpha=0.6)
    
    # Draw labels with gene symbols
    labels = {node: to_symbol(node) if node not in ['MAS'] else node 
              for node in result_graph.nodes()}
    nx.draw_networkx_labels(result_graph, pos, labels=labels, font_color='black', font_size=9)
    
    # Highlight input nodes (receptors)
    input_nodes = [n for n in sources if n in result_graph]
    if input_nodes:
        nx.draw_networkx_nodes(result_graph, pos, nodelist=input_nodes, 
                              node_color='red', node_size=700, label='Receptors')
    
    # Highlight output node (MAS)
    output_node = 'MAS'
    if output_node in result_graph:
        nx.draw_networkx_nodes(result_graph, pos, nodelist=[output_node], 
                              node_color='gold', node_size=700, label='MAS')
    
    plt.title(f"Model {model_no+1}: Top IFNAR1 & TGFBR1 Paths to MAS", 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(out_dir, 'individual_networks', f"path_ig_subgraph_result_{model_no}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_func(f"✓ Visualization saved to: path_ig_subgraph_result_{model_no}.png")

def print_competition_results(competitive_paths, print_func, top_n=10):
    """
    Pretty print competition analysis results (simplified for critical path analysis).
    """
    if competitive_paths is None:
        return
    
    print_func("\n" + "="*70)
    print_func("TOP COMPETITIVE PATHS (Sorted by Synergy/Interference)")
    print_func("="*70)
    
    # Summary of score magnitudes
    all_scores = []
    for p in competitive_paths:
        all_scores.extend([abs(p['tgfbr1_only']), abs(p['ifnar1_only']), abs(p['combined'])])
    all_scores = np.array([s for s in all_scores if s > 0])  # Filter out true zeros
    
    if len(all_scores) > 0:
        print_func(f"\nScore magnitude summary:")
        print_func(f"  Max:    {all_scores.max():.4e}")
        print_func(f"  Median: {np.median(all_scores):.4e}")
        print_func(f"  Min:    {all_scores.min():.4e}")
    else:
        print_func("\n⚠️  WARNING: All scores are exactly zero!")
        print_func("   This suggests the IG computation failed or gradients vanished.")
    
    for i, path_info in enumerate(competitive_paths[:top_n]):
        path = path_info['path']
        source = path[0][0]
        target = path[-1][1]
        
        print_func(f"\n{i+1}. Path: {source} → ... → {target}")
        print_func(f"   Full path: {' → '.join([e[0] for e in path] + [target])}")
        
        # Use scientific notation if values are very small
        tgfbr1_val = path_info['tgfbr1_only']
        ifnar1_val = path_info['ifnar1_only']
        combined_val = path_info['combined']
        expected_val = path_info['expected']
        synergy_val = path_info['synergy']
        
        # Choose format based on magnitude
        def format_score(val):
            if abs(val) < 1e-4:
                return f"{val:+.4e}"  # Scientific notation
            else:
                return f"{val:+.6f}"  # Fixed point
        
        print_func(f"   TGFBR1 only:  {format_score(tgfbr1_val)}")
        print_func(f"   IFNAR1 only:  {format_score(ifnar1_val)}")
        print_func(f"   Expected sum: {format_score(expected_val)}")
        print_func(f"   Actual combined: {format_score(combined_val)}")
        print_func(f"   Synergy:      {format_score(synergy_val)} ({'Opposite Effect' if path_info['is_competitive'] else 'Same Effect'})")

def extract_and_save_top_path_subgraph(combined_score_vector, A, out_dir, model_no, 
                                       edge_models, node_models, edge_weight_sum, 
                                       edge_weight_abs_sum, to_symbol, print_func, 
                                       top_n=1,
                                       sources=['P36897', 'P17181']):
    """
    Extract top paths from IFNAR1 and TGFBR1 using the combined max stimulation score, 
    save subgraph as CSV and PNG, and update global aggregators.
    """
    
    print_func("\n" + "="*70)
    print_func("STEP 3: EXTRACT TOP PATH SUBGRAPH")
    print_func("="*70)
    
    path_scores = combined_score_vector # Scores from the combined (AI) max point
    
    # Find top paths from each receptor
    ifnar1_paths = {k: v for k, v in path_scores.items() if k[0][0] == 'P17181'} # IFNAR1 is P17181
    tgfbr1_paths = {k: v for k, v in path_scores.items() if k[0][0] == 'P36897'} # TGFBR1 is P36897

    # sort paths by absolute score and take top_n
    ifnar1_ranked = sorted(ifnar1_paths.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    tgfbr1_ranked = sorted(tgfbr1_paths.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    
    # Keep these for backward compatibility (still return “the top-1”)
    top_ifnar1_path = ifnar1_ranked[0] if ifnar1_ranked else (None, 0.0)
    top_tgfbr1_path = tgfbr1_ranked[0] if tgfbr1_ranked else (None, 0.0)
    
    # Print Results 
    print_func(f"\n✓ Top IFNAR1 (P17181) Path Score (Combined Stim): {top_ifnar1_path[1]:+.6f}")
    # if top_ifnar1_path[0]:
    #     print_func(f"  Path: {' → '.join([e[0] for e in top_ifnar1_path[0]] + [top_ifnar1_path[0][-1][1]])}")
    
    print_func(f"\n✓ Top TGFBR1 (P36897) Path Score (Combined Stim): {top_tgfbr1_path[1]:+.6f}")
    # if top_tgfbr1_path[0]:
    #     print_func(f"  Path: {' → '.join([e[0] for e in top_tgfbr1_path[0]] + [top_tgfbr1_path[0][-1][1]])}")
    
    # Collect edges from top paths
    result_edges_set = set()
    
    # if top_ifnar1_path[0]:
    #     for u, v in top_ifnar1_path[0]:
    #         result_edges_set.add((u, v))
    
    # if top_tgfbr1_path[0]:
    #     for u, v in top_tgfbr1_path[0]:
    #         result_edges_set.add((u, v))
    for path_edges, _ in ifnar1_ranked:
        for u, v in path_edges:
            result_edges_set.add((u, v))

    for path_edges, _ in tgfbr1_ranked:
        for u, v in path_edges:
            result_edges_set.add((u, v))
    
    if not result_edges_set:
        print_func("⚠ Warning: No edges to save (no valid paths found)")
        return top_ifnar1_path, top_tgfbr1_path, None, None
    
    # Build result graph and CSV data
    result_graph = nx.DiGraph(list(result_edges_set))
    subgraph_data = []
    
    for u, v in result_edges_set:
        weight = A.loc[u, v]
        subgraph_data.append({
            'source_uniprot': u,
            'target_uniprot': v,
            'source_symbol': to_symbol(u),
            'target_symbol': to_symbol(v),
            'weight': weight,
            'path_ID': 'IFNAR1_TGFBR1_Top_Set'
        })
    
    df = pd.DataFrame(subgraph_data)
    
    # Save CSV
    csv_path = os.path.join(out_dir, 'individual_networks', f'subgraph_edgelist_{model_no}.csv')
    df.to_csv(csv_path, index=False)
    print_func(f"✓ Subgraph saved to: subgraph_edgelist_{model_no}.csv")
    
    # Update global aggregators
    for idx in range(len(df)):
        src_sym = df.loc[idx, 'source_symbol']
        tgt_sym = df.loc[idx, 'target_symbol']
        key = (src_sym, tgt_sym)
        w = df.loc[idx, 'weight']
        
        edge_models[key].add(model_no)
        node_models[src_sym].add(model_no)
        node_models[tgt_sym].add(model_no)
        edge_weight_sum[key] += w
        edge_weight_abs_sum[key] += abs(w)
    
    # Visualize
    visualize_subgraph(result_graph, sources, to_symbol, out_dir, model_no, print_func)
    
    return top_ifnar1_path, top_tgfbr1_path, result_graph, df

def aggregate_subgraph_edges(edge_models, edge_weight_sum, edge_weight_abs_sum, number_models, out_dir, print_func):
    """
    Aggregate subgraph edges across all models.
    """
    
    print_func('\n' + '='*70)
    print_func('AGGREGATING SUBGRAPH EDGES ACROSS ALL MODELS')
    print_func('='*70)
    
    # Create aggregated edge list
    aggregated_edges = []
    
    for (src_sym, tgt_sym), model_set in edge_models.items():
        num_models = len(model_set)
        avg_weight = edge_weight_sum[(src_sym, tgt_sym)] / num_models
        avg_abs_weight = edge_weight_abs_sum[(src_sym, tgt_sym)] / num_models
        
        aggregated_edges.append({
            'source_symbol': src_sym,
            'target_symbol': tgt_sym,
            'num_models': num_models,
            'frequency': num_models / number_models,
            'avg_weight': avg_weight,
            'avg_abs_weight': avg_abs_weight,
            'models': sorted(list(model_set))
        })
    
    # Sort by frequency
    aggregated_edges.sort(key=lambda x: x['frequency'], reverse=True)
    
    # Save aggregated edge list
    df_aggregated = pd.DataFrame(aggregated_edges)
    agg_path = os.path.join(out_dir, 'aggregated_subgraph_edges.csv')
    df_aggregated.to_csv(agg_path, index=False)
    
    print_func(f"\n✓ Saved aggregated edges to: {agg_path}")
    print_func(f"  Total unique edges: {len(aggregated_edges)}")
    
    return aggregated_edges, df_aggregated

def visualize_aggregated_subgraph(aggregated_edges, node_models, number_models, out_dir, print_func, freq_threshold=0.5):
    """
    Visualize aggregated subgraph (edges appearing in >threshold of models).
    """
    
    print_func(f"\nVisualizing aggregated subgraph (edges in >{freq_threshold:.0%} of models)...")
    
    frequent_edges = [(e['source_symbol'], e['target_symbol'], e['avg_weight']) 
                      for e in aggregated_edges if e['frequency'] > freq_threshold]
    
    if not frequent_edges:
        print_func(f"⚠ No edges found in >{freq_threshold:.0%} of models")
        return
    
    G_agg = nx.DiGraph()
    for src, tgt, weight in frequent_edges:
        G_agg.add_edge(src, tgt, weight=weight)
    
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G_agg, k=2, iterations=50, seed=42)
    
    # Node sizes by frequency
    node_sizes = []
    for node in G_agg.nodes():
        freq = len(node_models[node]) / number_models
        node_sizes.append(300 + 700 * freq)
    
    # Edge widths by frequency
    edge_widths = []
    for src, tgt in G_agg.edges():
        edge_data = [e for e in aggregated_edges 
                    if e['source_symbol']==src and e['target_symbol']==tgt][0]
        edge_widths.append(1 + 4 * edge_data['frequency'])
    
    # Draw
    nx.draw_networkx_nodes(G_agg, pos, node_color='lightblue', node_size=node_sizes)
    nx.draw_networkx_edges(G_agg, pos, width=edge_widths, edge_color='gray', 
                          alpha=0.6, arrowsize=15)
    nx.draw_networkx_labels(G_agg, pos, font_size=9, font_weight='bold')
    
    plt.title(f"Aggregated Top Paths Subgraph\n(Edges present in >{freq_threshold:.0%} of {number_models} models)", 
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    agg_fig_path = os.path.join(out_dir, 'aggregated_subgraph_frequent_edges.png')
    plt.savefig(agg_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_func(f"✓ Aggregated visualization saved to: {agg_fig_path}")

def aggregate_path_scores(path_ig_sum, path_ig_count, number_models, print_func):
    """
    Aggregate path scores across all models.
    """
    
    # Compute average path scores
    avg_path_scores = {}
    for path_edges, total_score in path_ig_sum.items():
        count = path_ig_count[path_edges]
        avg_path_scores[path_edges] = total_score / count if count > 0 else 0.0
    
    # Sort by absolute importance
    sorted_paths = sorted(avg_path_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print_func(f"\nTop 20 Most Important Paths (averaged across {number_models} models):")
    print_func("="*70)
    
    # for i, (path_edges, avg_score) in enumerate(sorted_paths[:20]):
    #     source = path_edges[0][0]
    #     target = path_edges[-1][1]
    #     path_length = len(path_edges)
        
    #     print_func(f"\n{i+1}. {source} → ... → {target}")
    #     print_func(f"   Path length: {path_length}")
    #     print_func(f"   Average IG score: {avg_score:+.6f}")
    
    return avg_path_scores, sorted_paths


def save_final_aggregated_results(avg_path_scores, aggregated_edges, 
                                  path_ig_sum, path_ig_count,
                                  edge_models, node_models, 
                                  number_models, out_dir, print_func):
    """
    Save final aggregated results across all models.
    """
    
    final_results = {
        'avg_path_scores': avg_path_scores,
        'aggregated_edges': aggregated_edges,
        'path_ig_sum': dict(path_ig_sum),
        'path_ig_count': dict(path_ig_count),
        'edge_models': {k: list(v) for k, v in edge_models.items()},
        'node_models': {k: list(v) for k, v in node_models.items()},
        'num_models': number_models
    }
    
    final_path = os.path.join(out_dir, 'final_aggregated_results.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(final_results, f)
    
    print_func(f"\n✓ Saved final aggregated results to: {final_path}")
    
    return final_path

def format_score(val):
    """Helper to format scores with appropriate precision."""
    if abs(val) < 1e-4:
        return f"{val:+.4e}"
    else:
        return f"{val:+.6f}"

def validate_biological_sanity(grid_path_scores_vector, print_func, sources=['P36897', 'P17181']):
    """
    Validate that biologically known pro-fibrotic receptors have positive path contributions.
    
    TGFBR1 (P36897) is known to be strongly pro-fibrotic, so its top path should be positive.
    IFNAR1 (P17181) is more context-dependent.
    
    Args:
        grid_path_scores_vector: Dict[path -> numpy array of 4 scores]
        print_func: Logging function
        sources: List of source receptors
    
    Returns:
        validation_results: Dict with validation outcomes
    """
    
    print_func("\n" + "="*70)
    print_func("BIOLOGICAL SANITY CHECK")
    print_func("="*70)
    
    validation_results = {}
    
    # Indices for the 4 critical points
    A_idx = 1   # TGFBR1 Only
    I_idx = 2   # IFNAR1 Only
    AI_idx = 3  # Combined
    
    # Check TGFBR1 (P36897) - Known pro-fibrotic
    tgfbr1_paths = {k: v for k, v in grid_path_scores_vector.items() if k[0][0] == 'P36897'}
    
    if tgfbr1_paths:
        # Find path with largest absolute value when TGFBR1 alone is stimulated
        top_tgfbr1_path = max(tgfbr1_paths.items(), key=lambda item: abs(item[1][A_idx]))
        path_edges, scores = top_tgfbr1_path
        tgfbr1_only_score = scores[A_idx]
        combined_score = scores[AI_idx]
        
        # Check if pro-fibrotic (positive)
        is_profibrotic = tgfbr1_only_score > 0
        
        print_func(f"\n✓ TGFBR1 (P36897) Top Path Validation:")
        print_func(f"  Path: {' → '.join([e[0] for e in path_edges] + [path_edges[-1][1]])}")
        print_func(f"  TGFBR1 Only Score: {format_score(tgfbr1_only_score)}")
        print_func(f"  Combined Score: {format_score(combined_score)}")
        
        if is_profibrotic:
            print_func(f"  ✅ PASS - Top TGFBR1 path is PRO-FIBROTIC (positive), as expected!")
        else:
            print_func(f"  ⚠️  WARNING - Top TGFBR1 path is ANTI-FIBROTIC (negative)!")
            print_func(f"      This contradicts biological knowledge that TGFB is pro-fibrotic.")
            print_func(f"      Possible reasons:")
            print_func(f"        • Model learned compensatory/feedback mechanisms")
            print_func(f"        • Training data has context-dependent effects")
            print_func(f"        • Path scoring may need more IG steps for accuracy")
        
        validation_results['tgfbr1'] = {
            'top_path': path_edges,
            'score_alone': tgfbr1_only_score,
            'score_combined': combined_score,
            'is_profibrotic': is_profibrotic,
            'passes_check': is_profibrotic
        }
    else:
        print_func("  ⚠️  No TGFBR1 paths found!")
        validation_results['tgfbr1'] = {'passes_check': False}
    
    # Check IFNAR1 (P17181) - Context-dependent, just report
    ifnar1_paths = {k: v for k, v in grid_path_scores_vector.items() if k[0][0] == 'P17181'}
    
    if ifnar1_paths:
        top_ifnar1_path = max(ifnar1_paths.items(), key=lambda item: abs(item[1][I_idx]))
        path_edges, scores = top_ifnar1_path
        ifnar1_only_score = scores[I_idx]
        combined_score = scores[AI_idx]
        
        print_func(f"\n✓ IFNAR1 (P17181) Top Path (Informational):")
        print_func(f"  Path: {' → '.join([e[0] for e in path_edges] + [path_edges[-1][1]])}")
        print_func(f"  IFNAR1 Only Score: {ifnar1_only_score:+.6f}")
        print_func(f"  Combined Score: {combined_score:+.6f}")
        
        if ifnar1_only_score > 0:
            print_func(f"  ℹ️  IFNAR1 top path is PRO-FIBROTIC (positive)")
        else:
            print_func(f"  ℹ️  IFNAR1 top path is ANTI-FIBROTIC (negative)")
        
        validation_results['ifnar1'] = {
            'top_path': path_edges,
            'score_alone': ifnar1_only_score,
            'score_combined': combined_score,
            'is_profibrotic': ifnar1_only_score > 0
        }
    else:
        print_func("  ⚠️  No IFNAR1 paths found!")
        validation_results['ifnar1'] = {}
    
    # Check if they're competitive (opposite signs)
    if tgfbr1_paths and ifnar1_paths:
        tgfbr1_sign = np.sign(validation_results['tgfbr1']['score_alone'])
        ifnar1_sign = np.sign(validation_results['ifnar1']['score_alone'])
        
        if tgfbr1_sign != ifnar1_sign and tgfbr1_sign != 0 and ifnar1_sign != 0:
            # print_func(f"\n✓ COMPETITION DETECTED:")
            # print_func(f"  TGFBR1 and IFNAR1 top paths have OPPOSITE effects on MAS!")
            # print_func(f"  This suggests antagonistic signaling between the two receptors.")
            validation_results['competitive'] = True
        else:
            # print_func(f"\n✓ NO COMPETITION:")
            # print_func(f"  TGFBR1 and IFNAR1 top paths have SAME SIGN effects.")
            validation_results['competitive'] = False
    
    return validation_results

def allpaths_biological_sanity_check(grid_path_scores_vector, validation_summary, print_func):
    """
    Sanity check using TOTAL contribution AND extract TOP path info.
    """
    
    print_func("\n" + "="*70)
    print_func("BIOLOGICAL SANITY CHECK (Total Contribution)")
    print_func("="*70)
    
    validation_results = {}
    
    A_idx = 1
    
    # Check TGFBR1 using TOTAL contribution
    if 'tgfbr1' in validation_summary:
        total_contribution = validation_summary['tgfbr1']['total_path_contribution']
        global_ig = validation_summary['tgfbr1']['global_ig']
        
        print_func(f"\n✓ TGFBR1 (P36897) SANITY CHECK - Total Contribution:")
        print_func(f"  Total Path Contribution: {total_contribution:+.6f}")
        print_func(f"  Global IG (Captum):      {global_ig:+.6f}")
        
        if global_ig > 0:
            print_func(f"  ✅ PASS - TGFBR1 total effect is PRO-FIBROTIC, as expected!")
        else:
            print_func(f"  ❌ FAIL - TGFBR1 total effect is ANTI-FIBROTIC (biology issue!)")
        
        # For aggregation/tracking: use total contribution
        validation_results['tgfbr1_total_profibrotic'] = global_ig > 0
        validation_results['tgfbr1_passes'] = global_ig > 0  # Keep for sanity check
        
        # Extract TOP path for plotting
        tgfbr1_paths = {k: v for k, v in grid_path_scores_vector.items() if k[0][0] == 'P36897'}
        if tgfbr1_paths:
            top_path_edges, top_scores = max(tgfbr1_paths.items(), 
                                             key=lambda x: abs(x[1][A_idx]))
            top_path_score = top_scores[A_idx]
            
            print_func(f"\n✓ TGFBR1 TOP PATH (for visualization):")
            print_func(f"  Score: {top_path_score:+.6f} ({'PRO' if top_path_score > 0 else 'ANTI'}-fibrotic)")
            print_func(f"  Path: {' → '.join([e[0] for e in top_path_edges[:3]])} ... → MAS")
            
            # Store top path info for plotting
            validation_results['tgfbr1_top_path'] = top_path_edges
            validation_results['tgfbr1_top_score'] = top_path_score
            validation_results['tgfbr1_top_profibrotic'] = top_path_score > 0
    
    return validation_results

def compute_global_gradient_captum(model, x_baseline, x_stim, source_idx, n_steps=100):
    """
    Compute global gradient ∂output/∂input using Captum's IntegratedGradients.
    This serves as ground truth for validating path-based decomposition.
    
    Args:
        model: Full model (signaling + TF + phenotype)
        x_baseline: Baseline input (zeros)
        x_stim: Stimulation input
        source_idx: Index of source node (e.g., TGFBR1)
        n_steps: Number of IG steps
    
    Returns:
        attribution: ∂output/∂source_input
    """
    
    model.eval()
    
    # Captum's IG requires specific output selection
    def forward_func(x):
        # For phenotype output (single value)
        out = model(x)
        if out.dim() > 1:
            out = out.squeeze()
        return out
    
    # Initialize Captum's IG
    ig = IntegratedGradients(forward_func)
    
    # Compute attributions
    x_baseline_grad = x_baseline.clone().requires_grad_(True)
    x_stim_grad = x_stim.clone().requires_grad_(True)
    
    attributions = ig.attribute(
        x_stim_grad,
        baselines=x_baseline_grad,
        n_steps=n_steps,
        return_convergence_delta=False
    )
    
    # Extract attribution for the source node
    source_attribution = attributions[0, source_idx].item()
    
    return source_attribution

def validate_path_with_captum(model, path_edges, node_idx_map, x_baseline, x_stim, 
                               source_idx, print_func=print):
    """
    Mask out all weights except those on the path, then compute IG with Captum.
    If this gives zero, the path truly has no gradient flow.
    """
    model.eval()
    # Save original weights
    original_weights = model[0].weights.data.clone()
    
    # Create mask: only keep edges that are in the path
    mask = torch.zeros_like(original_weights, dtype=torch.bool)
    
    for (u, v) in path_edges:
        if v != 'MAS':  # Skip the final TF->phenotype edge
            u_idx = node_idx_map[u]
            v_idx = node_idx_map[v]
            mask[v_idx, u_idx] = True  # Adjacency is [target, source]
    
    # Apply mask: zero out all edges NOT in the path
    model[0].weights.data = original_weights * mask.double()
    
    # Compute IG with Captum on this masked model
    def forward_func(x):
        out = model(x)
        if out.dim() > 1:
            out = out.squeeze()
        return out
    
    ig = IntegratedGradients(forward_func)
    
    x_baseline_grad = x_baseline.clone().requires_grad_(True)
    x_stim_grad = x_stim.clone().requires_grad_(True)
    
    attributions = ig.attribute(
        x_stim_grad,
        baselines=x_baseline_grad,
        n_steps=100,
        return_convergence_delta=False
    )
    
    path_score_captum = attributions[0, source_idx].item()
    
    # Restore original weights
    model[0].weights.data = original_weights
    
    # print_func(f"  Path score (Captum, masked model): {path_score_captum:+.6e}")
    
    return path_score_captum

def validate_path_decomposition(grid_path_scores_vector, model, x_baseline_batch, x_stim_batch,
                                 mod, print_func):
    """
    Validate that:
    1. Sum of all path scores = Global IG (path decomposition is correct)
    2. Global TGFBR1 effect is pro-fibrotic (model captures biology)
    """
    
    print_func("\n" + "="*70)
    print_func("PATH DECOMPOSITION VALIDATION")
    print_func("="*70)
    
    validation_summary = {}
    
    # Indices for the 4 critical points
    A_idx = 1   # TGFBR1 Only
    I_idx = 2   # IFNAR1 Only
    AI_idx = 3  # Combined
    
    tgfbr1_idx = mod.node_idx_map['P36897']
    ifnar1_idx = mod.node_idx_map['P17181']
    
    # ========================================================================
    # VALIDATION 1: TGFBR1 Only - Total Path Contribution vs Global IG
    # ========================================================================
    
    print_func("\n--- Validation 1: TGFBR1 Only (A) ---")
    
    # Get all TGFBR1 paths and sum their contributions
    tgfbr1_paths = {k: v for k, v in grid_path_scores_vector.items() if k[0][0] == 'P36897'}
    
    if tgfbr1_paths:
        tgfbr1_path_contributions = np.array([scores[A_idx] for scores in tgfbr1_paths.values()])
        total_tgfbr1_paths = tgfbr1_path_contributions.sum()
        
        print_func(f"\nSum of {len(tgfbr1_paths)} TGFBR1 path contributions: {format_score(total_tgfbr1_paths)}")
        
        # Compute global IG using Captum (ground truth)
        x_baseline_A = x_baseline_batch[A_idx:A_idx+1, :]
        x_stim_A = x_stim_batch[A_idx:A_idx+1, :]
        
        global_ig_tgfbr1 = compute_global_gradient_captum(
            model, x_baseline_A, x_stim_A, 
            source_idx=tgfbr1_idx,
            n_steps=100)
        
        print_func(f"Global IG (Captum) for TGFBR1: {format_score(global_ig_tgfbr1)}")
        
        # Check agreement
        relative_error = abs(total_tgfbr1_paths - global_ig_tgfbr1) / (abs(global_ig_tgfbr1) + 1e-10)
        
        print_func(f"Relative error: {relative_error:.4%}")
        
        if relative_error < 0.10:  # 10% tolerance
            print_func("✅ PASS - Path decomposition matches global gradient!")
        else:
            print_func("⚠️  WARNING - Significant mismatch between paths and global gradient")
            print_func("   Possible reasons:")
            print_func("   • Need more paths (increase max_path_length or paths per source)")
            print_func("   • Need more IG steps for accuracy")
        
        # Check if TGFBR1 is globally pro-fibrotic
        if global_ig_tgfbr1 > 0:
            print_func(f"\n✅ MODEL VALIDATION PASS - TGFBR1 is PRO-FIBROTIC (global IG > 0)")
        else:
            print_func(f"\n❌ MODEL VALIDATION FAIL - TGFBR1 is ANTI-FIBROTIC (global IG < 0)")
            print_func("   This contradicts known biology!")
        
        validation_summary['tgfbr1'] = {
            'total_path_contribution': total_tgfbr1_paths,
            'global_ig': global_ig_tgfbr1,
            'relative_error': relative_error,
            'is_profibrotic': global_ig_tgfbr1 > 0,
            'decomposition_valid': relative_error < 0.10
        }
    
    # ========================================================================
    # VALIDATION 2: IFNAR1 Only - Total Path Contribution vs Global IG
    # ========================================================================
    
    print_func("\n--- Validation 2: IFNAR1 Only (I) ---")
    
    ifnar1_paths = {k: v for k, v in grid_path_scores_vector.items() if k[0][0] == 'P17181'}
    
    if ifnar1_paths:
        ifnar1_path_contributions = np.array([scores[I_idx] for scores in ifnar1_paths.values()])
        total_ifnar1_paths = ifnar1_path_contributions.sum()
        
        print_func(f"\nSum of {len(ifnar1_paths)} IFNAR1 path contributions: {total_ifnar1_paths:+.6f}")
        
        x_baseline_I = x_baseline_batch[I_idx:I_idx+1, :]
        x_stim_I = x_stim_batch[I_idx:I_idx+1, :]
        
        global_ig_ifnar1 = compute_global_gradient_captum(
            model, x_baseline_I, x_stim_I,
            source_idx=ifnar1_idx,
            n_steps=100
        )
        
        print_func(f"Global IG (Captum) for IFNAR1:        {global_ig_ifnar1:+.6f}")
        
        relative_error_ifnar1 = abs(total_ifnar1_paths - global_ig_ifnar1) / (abs(global_ig_ifnar1) + 1e-10)
        print_func(f"Relative error: {relative_error_ifnar1:.4%}")
        
        if relative_error_ifnar1 < 0.10:
            print_func("✅ PASS - Path decomposition matches global gradient!")
        else:
            print_func("⚠️  WARNING - Significant mismatch")
        
        validation_summary['ifnar1'] = {
            'total_path_contribution': total_ifnar1_paths,
            'global_ig': global_ig_ifnar1,
            'relative_error': relative_error_ifnar1,
            'is_profibrotic': global_ig_ifnar1 > 0,
            'decomposition_valid': relative_error_ifnar1 < 0.10
        }
    
    # ========================================================================
    # VALIDATION 3: Check Additivity (A + I ≈ AI for independent effects)
    # ========================================================================
    
    if 'tgfbr1' in validation_summary and 'ifnar1' in validation_summary:        
        expected_sum = validation_summary['tgfbr1']['global_ig'] + validation_summary['ifnar1']['global_ig']
        
        x_baseline_AI = x_baseline_batch[AI_idx:AI_idx+1, :]
        x_stim_AI = x_stim_batch[AI_idx:AI_idx+1, :]
        
        # For combined, we need to compute gradient w.r.t. both inputs
        # For simplicity, just report the total model output
        with torch.no_grad():
            model.eval()
            output_AI = model(x_stim_AI)[0, 0].item() if model(x_stim_AI).dim() > 1 else model(x_stim_AI).item()
            output_baseline = model(x_baseline_AI)[0, 0].item() if model(x_baseline_AI).dim() > 1 else model(x_baseline_AI).item()
        
        actual_combined_effect = output_AI - output_baseline
        synergy = actual_combined_effect - expected_sum
        
        validation_summary['synergy'] = synergy
    
    return validation_summary

## Checkpointing functions
def generate_checkpoint_key(model_no, timepoint, all_paths, x_stim_batch, x_baseline_batch):
    """
    Generate a unique hash key for checkpoint based on inputs.
    This ensures we only reuse checkpoints with identical configurations.
    """
    # Create a deterministic string representation
    paths_str = str(sorted([str(p) for p in all_paths]))
    stim_str = f"{x_stim_batch.cpu().numpy().tobytes()}"
    baseline_str = f"{x_baseline_batch.cpu().numpy().tobytes()}"
    
    key_str = f"{model_no}_{timepoint}_{paths_str}_{stim_str}_{baseline_str}"
    return hashlib.md5(key_str.encode()).hexdigest()

def save_ig_checkpoint(checkpoint_dir, checkpoint_key, step, path_score_matrix, metadata, print_func=print):
    """
    Save intermediate IG computation state.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_{checkpoint_key}_step{step}.pkl')
    
    checkpoint_data = {
        'step': step,
        'path_score_matrix': path_score_matrix.cpu().numpy(),
        'metadata': metadata,
        'timestamp': time.time()
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print_func(f"  💾 Checkpoint saved at step {step}/{metadata['num_steps_ig']}")

def load_ig_checkpoint(checkpoint_dir, checkpoint_key):
    """
    Load the most recent checkpoint for this run.
    
    Returns:
        (step, path_score_matrix, metadata) or (0, None, None) if no checkpoint found
    """
    if not os.path.exists(checkpoint_dir):
        return 0, None, None
    
    # Find all checkpoints for this key
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith(f'ckpt_{checkpoint_key}_step')]
    
    if not checkpoint_files:
        return 0, None, None
    
    # Get the most recent checkpoint
    checkpoint_files.sort(key=lambda f: int(f.split('step')[1].split('.pkl')[0]))
    latest_checkpoint = checkpoint_files[-1]
    
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    step = checkpoint_data['step']
    path_score_matrix = torch.from_numpy(checkpoint_data['path_score_matrix'])
    metadata = checkpoint_data['metadata']
    
    print(f"  ✅ Resuming from checkpoint at step {step}/{metadata['num_steps_ig']}")
    
    return step, path_score_matrix, metadata

def cleanup_old_checkpoints(checkpoint_dir, checkpoint_key, keep_last_n=2):
    """
    Remove old checkpoints to save disk space, keeping only the most recent ones.
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith(f'ckpt_{checkpoint_key}_step')]
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    checkpoint_files.sort(key=lambda f: int(f.split('step')[1].split('.pkl')[0]))
    
    # Remove all but the last keep_last_n
    for old_file in checkpoint_files[:-keep_last_n]:
        os.remove(os.path.join(checkpoint_dir, old_file))

def setup_dual_logging(out_dir='.', log_filename='pathway_scoring.log', use_timestamp=True):
    """
    Setup logging to both console and file with different formats.
    
    Args:
        out_dir: Directory to save log file
        log_filename: Base name of log file
        use_timestamp: If True, add timestamp to filename
    
    Returns:
        print2log: Function to use for all print statements
    """
    import logging
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to filename if requested
    if use_timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(log_filename)
        log_filename = f"{base}_{timestamp}{ext}"
    
    # Create logger
    logger = logging.getLogger('PathwayScoring')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    logger.propagate = False
    
    # Create formatters
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler
    log_path = os.path.join(out_dir, log_filename)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log the setup
    logger.info("="*70)
    logger.info("PATHWAY SCORING ANALYSIS - LOGGING STARTED")
    logger.info("="*70)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Logger directory: {out_dir}")
    logger.info("="*70 + "\n")
    
    return logger.info, log_path

parser = argparse.ArgumentParser(prog='Hepatocyte simulation')
parser.add_argument('--log_filename', action='store', type=str, 
                    default='pathway_scoring.log',
                    help='Name of the log file')
parser.add_argument('--checkpoint_interval', action='store', type=int, default=20,
                    help='Save IG checkpoint every N steps (0 = no checkpointing)')
parser.add_argument('--enumerated_pathways_path', action='store', type=str,
                    help='Pickle file containing enumerated pathways for each model. The dictionary is labeled using model IDs and each element is a list of pathways as lists of nodes.')
parser.add_argument('--resume_from_checkpoint', action='store_true',
                    help='Resume from existing checkpoint if available')
parser.add_argument('--clear_checkpoints', action='store_true',
                    help='Clear all existing checkpoints before starting')
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
parser.add_argument('--res_dir', action='store',type = str, default='../results/') #_'+timepoint
parser.add_argument('--out_dir', action='store',type = str, default='resultsMAS/results_24hr/') # 'results_2hr/' for 2 hour and 'results_24hr/' for 24 hour
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
parser.add_argument('--num_steps_ig', action='store',type = float, default=100) #make it at least 50
parser.add_argument('--GRADIENT_THRESHOLD', action='store',type = float, default=8e-3) # NEW ARGUMENT
parser.add_argument('--EDGE_FREQ_MIN', action='store',type = float, default=0.5)
parser.add_argument('--NODE_FREQ_MIN', action='store',type = float, default=0.5)
parser.add_argument('--top_n_paths', action='store',type = int, default=1)

args = parser.parse_args()
top_n_paths = args.top_n_paths
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
max_path_length = args.max_path_length
num_steps_ig =  args.num_steps_ig
out_dir = args.out_dir
GRADIENT_THRESHOLD = args.GRADIENT_THRESHOLD
enumerated_pathways_path = args.enumerated_pathways_path


print2log, log_file_path = setup_dual_logging(log_filename=args.log_filename, use_timestamp=True)

# At the start of main loop, after argument parsing
if args.clear_checkpoints:
    checkpoint_dir = os.path.join(out_dir, 'ig_checkpoints')
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
        print2log("🗑️  Cleared all existing checkpoints")

Path(os.path.join(out_dir, 'individual_networks')).mkdir(parents=True, exist_ok=True)
Path(os.path.join(out_dir, 'barplots')).mkdir(parents=True, exist_ok=True)

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
    if x in ['fibrosis','MAS','Fibrosis','NAS','mas','nas']:
        return x
    else:
        # return symbol if known, otherwise keep original Uniprot
        return id2sym.get(str(x), str(x))
# Aggregators across all models (appearance per model)
# Aggregators for subgraph edges/nodes across ALL models and ALL timepoints
edge_models = defaultdict(set)        # (src_symbol, tgt_symbol) -> set of model numbers
node_models = defaultdict(set)        # node_symbol -> set of model numbers
edge_weight_sum = defaultdict(float)  # (src_symbol, tgt_symbol) -> sum of weights
edge_weight_abs_sum = defaultdict(float)  # (src_symbol, tgt_symbol) -> sum of |weights|
# Aggregators for path scores across ALL models and ALL timepoints
path_ig_sum = defaultdict(float)      # path_edges -> sum of IG scores
path_ig_count = defaultdict(int)      # path_edges -> count of models

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
for timepoint in ['24 hour']:  # or ['2 hour','8 hour','24 hour']
    print2log(f'Processing timepoint {timepoint}')
    
    if timepoint == '8 hour':
        data_path = res_dir + 'allmodels/'
    else:
        data_path = res_dir + timepoint + '/allmodels/'
    
    # Load drugs information
    drugTargets = pd.read_csv(TargetsIn, sep='\t', low_memory=False, index_col=0)
    inputInfo = pd.read_csv(infoPath, sep='\t', low_memory=False, index_col=0)
    inputInfo = inputInfo[inputInfo['time']==timepoint]
    TFOutput_tmp = TFOutput.loc[inputInfo.index,:]
    drugInput_tmp = drugInput.loc[inputInfo.index,:]
    
    # Subset input and output to intersecting nodes
    druginName = drugInput_tmp.columns.values
    inName = drugTargets.columns.values
    outconds = drugInput_tmp.index.values
    outName = TFOutput_tmp.columns.values
    inName = np.intersect1d(nodeNames, inName)
    outName = np.intersect1d(nodeNames, outName)
    TFOutput_tmp = TFOutput_tmp.loc[outconds,outName]
    drugTargets = drugTargets.loc[:,inName]
    
    if ConvertToEmpProb==True:
        print2log('Converted to Empirical probabilities')
        TFOutput_tmp = 1/(1+np.exp(-TFOutput_tmp))
    
    drugTargets = drugTargets.loc[drugInput_tmp.columns.values,:]
    drugSim = pd.read_csv(ChemicalSims,index_col=0)
    drugSim = drugSim.loc[drugInput_tmp.columns.values,drugInput_tmp.columns.values]
    drugSim = torch.tensor(drugSim.values.copy(), dtype=torch.double)
    all_drugs = drugInput.columns
    
    print2log("Prepare and load the SignalingModel and the Graph...")
    tfs = list(outName)
    mask_dt = torch.tensor(np.array(drugTargets)).transpose(0,1)
    bionet_params = {'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations, 
                    'exp_factor':50, 'tolerance': args.LEMBAS_tolerance, 
                    'leak':args.LEMBAS_leak}
    
    print2log("Train TFs ---> Phenotype model...")
    Xh = pyreadr.read_r(args.X_invivo_path)[None]
    Xh = Xh.loc[:,outName].to_numpy()
    Yh = pyreadr.read_r(args.Y_invivo_path)[None].to_numpy()
    Yh = Yh[:,0]
    params = {'alpha': [0.001,0.01,0.1,0.2,0.5,1,2,5,10,20,50,100,500,1000]}
    ridge = GridSearchCV(estimator=Ridge(),param_grid=params, cv=RepeatedKFold(n_splits=10, n_repeats=10), n_jobs=-1)
    ridge.fit(Xh, Yh)
    coefs = ridge.best_estimator_.coef_
    intercept = ridge.best_estimator_.intercept_
    pheno = FrozenLinear(coefs[np.newaxis, :], intercept, dtype=torch.double, device=device)
    pheno.eval()

    ## check if paths have been already enumated by looking for the pickle file
    ## See if there is already a enumerated_pathways_path file
    enumerated_pathways_path_timepoint = enumerated_pathways_path + '_' + timepoint + '.pkl'
    if os.path.exists(enumerated_pathways_path_timepoint):
        print2log("Enumerated pathways already exist, loading...")
        enumeration_flag = True
        with open(enumerated_pathways_path_timepoint, 'rb') as f:
            enumerated_pathways = pickle.load(f)
    else:
        enumeration_flag = False
    
    for model_no in range(number_DTLEMBAS_models):
        print2log(f"\n{'='*70}")
        print2log(f"MODEL {model_no+1}/{number_DTLEMBAS_models}")
        print2log(f"{'='*70}")
        
        # Load model
        mod = SignalingModel(net=net,
                            X_in=drugTargets,
                            y_out=TFOutput_tmp, 
                            projection_amplitude_in=projection_amplitude_in, 
                            projection_amplitude_out=projection_amplitude_out,
                            weight_label=weight_label, 
                            source_label=source_label, 
                            target_label=target_label,
                            bionet_params=bionet_params, 
                            dtype=torch.double, 
                            device=device)
        
        mod.load_state_dict(torch.load(os.path.join(data_path, modelPattern + str(model_no+1)+'signal_state_dict_%s.pth'%(model_no+1))))
        mod.eval()
        mod.signaling_network.weights.data.masked_fill_(mask=mod.signaling_network.mask, value=0.0)
        
        # Build adjacency matrix
        weights = mod.signaling_network.weights.detach().cpu().numpy()
        A = pd.DataFrame(weights.T, index=nodeNames, columns=nodeNames)
        A = pd.concat([A, pd.DataFrame(np.zeros((len(nodeNames),1)), index=nodeNames, columns=['MAS'])], axis=1)
        A = pd.concat([A, pd.DataFrame(np.zeros((1,len(nodeNames)+1)), index=['MAS'], columns=list(nodeNames)+['MAS'])], axis=0)
        A.loc[outName, 'MAS'] = coefs[:]
        
        sources = ['P36897','P17181']  # TGFBR1, IFNAR1
        G = nx.from_pandas_adjacency(A, create_using=nx.DiGraph())
        
        # Build model
        model = torch.nn.Sequential(
            mod.signaling_network,
            mod.output_layer,
            pheno
        )
        model.eval()
        
        # --- STEP 0: Create CRITICAL input batch ---
        print2log('\n' + '='*70)
        print2log('STEP 0: CREATING CRITICAL INPUT BATCH (4 POINTS)')
        print2log('='*70)
        
        nodeNames_array = np.array(list(mod.node_idx_map.keys()))
        all_nodes = np.array(list(mod.node_idx_map.keys()))
        
        # TGFBR1: P36897, IFNAR1: P17181
        tgfbr1_idx_R = mod.node_idx_map['P36897']
        ifnar1_idx_R = mod.node_idx_map['P17181']
        
        # Create a zero vector for the base input
        x_base = torch.zeros((1, len(nodeNames_array)), dtype=torch.double).to(device)
        
        # Define the 4 critical inputs (Batch size = 4)
        x_baseline_B = x_base.clone()  # B: Zero, Zero (Index 0)
        x_tgfbr1_A = x_base.clone()
        x_tgfbr1_A[0, tgfbr1_idx_R] = stimulation_strength # A: Max TGFBR1, Zero IFNAR1 (Index 1)
        x_ifnar1_I = x_base.clone()
        x_ifnar1_I[0, ifnar1_idx_R] = stimulation_strength # I: Zero TGFBR1, Max IFNAR1 (Index 2)
        x_combined_AI = x_base.clone()
        x_combined_AI[0, tgfbr1_idx_R] = stimulation_strength
        x_combined_AI[0, ifnar1_idx_R] = stimulation_strength # AI: Max TGFBR1, Max IFNAR1 (Index 3)
        
        # Combine into a single batch for vectorized IG computation
        Xin_critical = torch.cat([x_baseline_B, x_tgfbr1_A, x_ifnar1_I, x_combined_AI], dim=0).detach()
        x_baseline_batch = torch.zeros_like(Xin_critical).to(device) # Baseline for IG is always zero vector
        
        print2log(f"Batch created with {Xin_critical.shape[0]} critical points.")
        print2log(f"TGFBR1 ID (P36897) index: {tgfbr1_idx_R}, IFNAR1 ID (P17181) index: {ifnar1_idx_R}")
        
        # --- STEP 1: Define target paths ---
        print2log('\n' + '='*70)
        print2log('STEP 1: DEFINING TARGET PATHS')
        print2log('='*70)
        if not enumeration_flag:
            # First, compute edge gradients to identify active edges
            print2log("\n Computing edge gradients for all edges in the network...")
            # Use a representative stimulation (e.g., combined stimulation)
            AI_idx = 3  # Combined stimulation
            x_stim_representative = Xin_critical[AI_idx:AI_idx+1, :]

            edge_gradients = compute_edge_gradients_grid(
                model=model,
                node_idx_map=mod.node_idx_map,
                sources=sources,
                pheno_coefs=torch.tensor(coefs, dtype=torch.double, device=device),
                outName=outName,
                stimulation_strength=stimulation_strength,
                num_steps=21,  # 10x10 grid = 100 conditions
                aggregation='range',  # max(Y) - min(Y)
                print_func=print2log
            )
            all_grads = list(edge_gradients.values())
            print2log(f"   Edges with |gradient| > {GRADIENT_THRESHOLD:.1e}: {sum(1 for g in all_grads if g > GRADIENT_THRESHOLD)}")

            # Filter the graph G to keep only edges with high gradient magnitude
            # Collect active signaling network edges (u -> v, excluding TF->Phenotype)
            active_signaling_edges = [
                (u, v, {'weight': grad})
                for (u, v), grad in edge_gradients.items()
                if v != 'MAS' and G.has_edge(u, v) and grad >= GRADIENT_THRESHOLD
            ]
            # Collect active TF -> Phenotype edges
            active_tf_edges = [
                (tf_name, 'MAS', {'weight': edge_gradients[(tf_name, 'MAS')]})
                for tf_name in outName
                if edge_gradients.get((tf_name, 'MAS'), 0.0) >= GRADIENT_THRESHOLD
            ]
            # Construct the filtered graph in a single step
            G_filtered = nx.DiGraph(active_signaling_edges + active_tf_edges)
            
            # # Find gradient-weighted paths
            # all_paths = []
            # paths_per_source = 20
            # for source_node in sources:
            #     print2log(f"Finding paths from {source_node}...")
            #     source_paths = find_top_gradient_weighted_paths(
            #         G=G,
            #         edge_gradients=edge_gradients,
            #         source=source_node,
            #         target='MAS',
            #         top_k=paths_per_source,
            #         print_func=print2log
            #     )
            #     for path_edges in source_paths:
            #         if path_edges not in all_paths:
            #             all_paths.append(path_edges)
            # print2log(f"\n✅ Found {len(all_paths)} unique gradient-weighted paths to score")
            all_paths = []
            for source_node in sources:
                print2log(f"Finding paths from {source_node}...")
                # Use all_simple_paths with islice for a controlled path search
                paths_gen = nx.all_simple_paths(G_filtered, source=source_node, target='MAS', cutoff=args.max_path_length)
                top_k_paths = list(islice(paths_gen, 1200))
                print2log(f"   Found {len(top_k_paths)} paths from {source_node}")
                for path in top_k_paths:
                    path_edges = tuple(nx.utils.pairwise(path))
                    if path_edges not in all_paths:
                        all_paths.append(path_edges)
            print2log(f"\n✅ Enumerated {len(all_paths)} unique paths to score")
        else:
            all_paths = enumerated_pathways[model_no]
                
        # --- STEP 2: Score paths across CRITICAL batch ---
        print2log('\n' + '='*70)
        print2log('STEP 2: SCORING PATHS ACROSS CRITICAL BATCH (VECTORIZED)')
        print2log('='*70)
        
        checkpoint_dir = os.path.join('.', 'ig_checkpoints') if args.checkpoint_interval > 0 else None
        checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval > 0 else None
        # Generate unique checkpoint key for this model
        checkpoint_key = generate_checkpoint_key(
            model_no=model_no, 
            timepoint=timepoint,
            all_paths=all_paths,
            x_stim_batch=Xin_critical,
            x_baseline_batch=x_baseline_batch
        )
        
        # Use the vectorized scoring function
        grid_path_scores_vector = score_paths_vectorized_jacobian(
            model=model,
            all_paths=all_paths,
            node_idx_map=mod.node_idx_map,
            x_baseline_batch=x_baseline_batch,
            x_stim_batch=Xin_critical,
            pheno_coefs=torch.tensor(coefs, dtype=torch.double, device=device),
            outName=outName,
            num_steps_ig=int(num_steps_ig),
            show_progress=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_key=checkpoint_key,
            checkpoint_interval=checkpoint_interval,
            print_func=print2log
        )

        # print2log("\n" + "="*70)
        # print2log("🔬 VALIDATING PATH SCORES WITH CAPTUM (MASKED MODEL TEST)")
        # print2log("="*70)
        # print2log("\nThis test will:")
        # print2log("  1. Take the top TGFBR1 path")
        # print2log("  2. Mask the model to ONLY use edges in that path")
        # print2log("  3. Compute IG with Captum on the masked model")
        # print2log("  4. Compare to your chain-rule method")
        # print2log("\nIf Captum also gives ~zero → Path has no gradient flow (path selection issue)")
        # print2log("If Captum gives non-zero → Your chain-rule has a bug\n")

        # Get the top TGFBR1 path
        A_idx = 1  # TGFBR1 only stimulation
        tgfbr1_paths = {k: v for k, v in grid_path_scores_vector.items() if k[0][0] == 'P36897'}

        if tgfbr1_paths:
            # Find path with largest absolute value
            top_tgfbr1_path_edges, top_tgfbr1_scores = max(
                tgfbr1_paths.items(), 
                key=lambda x: abs(x[1][A_idx])
            )
            
            your_method_score = top_tgfbr1_scores[A_idx]
            
            # print2log(f"Testing TOP TGFBR1 path:")
            # print2log(f"  Path length: {len(top_tgfbr1_path_edges)} edges")
            # print2log(f"  Path: {' → '.join([e[0] for e in top_tgfbr1_path_edges[:3]])} ... → MAS")
            # print2log(f"  Your method score: {your_method_score:+.6e}")
            
            # Extract the correct batch inputs for TGFBR1 only
            x_baseline_A = x_baseline_batch[A_idx:A_idx+1, :]
            x_stim_A = Xin_critical[A_idx:A_idx+1, :]
            
            # Run Captum validation
            captum_score = validate_path_with_captum(
                model=model,
                path_edges=top_tgfbr1_path_edges,
                node_idx_map=mod.node_idx_map,
                x_baseline=x_baseline_A,
                x_stim=x_stim_A,
                source_idx=tgfbr1_idx_R,
                print_func=print2log
            )
            
            # print2log("\n" + "-"*70)
            # print2log("DIAGNOSIS:")
            # print2log("-"*70)
            
            # if abs(captum_score) < 1e-10 and abs(your_method_score) < 1e-10:
            #     print2log("❌ BOTH methods give ~zero")
            #     print2log("   → This path has NO GRADIENT FLOW in the model")
            #     print2log("   → Your path finding method is selecting inactive paths")
            #     print2log("   → SOLUTION: Use gradient-weighted path finding")
                
            # elif abs(captum_score) < 1e-10 and abs(your_method_score) > 1e-10:
            #     print2log("⚠️  Captum gives zero, but your method gives non-zero")
            #     print2log("   → Something is wrong with your implementation")
            #     print2log("   → (Captum should be ground truth)")
                
            # elif abs(captum_score) > 1e-10:
            #     print2log(f"✅ Path HAS gradient flow! Captum score: {captum_score:+.6e}")
                
            #     if abs(your_method_score) < 1e-10:
            #         print2log("❌ But your method gives ~zero")
            #         print2log("   → BUG in your chain-rule implementation")
            #         print2log("   → Check log-space conversion and accumulation")
            #     else:
            #         ratio = your_method_score / captum_score
            #         error = abs(ratio - 1.0)
                    
            #         print2log(f"\n   Ratio (your / Captum): {ratio:.4e}")
            #         print2log(f"   Relative error: {error:.2%}")
                    
            #         if error < 0.1:
            #             print2log("   ✅ EXCELLENT - Your method is correct!")
            #         elif error < 0.5:
            #             print2log("   ⚠️  Close but not exact - minor numerical issue")
            #         else:
            #             print2log("   ❌ Large discrepancy - check implementation")
            
            # print2log("="*70 + "\n")
            
        else:
            print2log("❌ No TGFBR1 paths found to test!\n")
        
        # --- STEP 2B: Analyze competition using the critical points ---
        print2log('\n' + '='*70)
        print2log('STEP 2B: ANALYZING COMPETITION PATTERNS')
        print2log('='*70)
        
        # grid_path_scores_vector is a Dict[path -> numpy array of 4 scores]
        competitive_paths = analyze_critical_paths(grid_path_scores_vector=grid_path_scores_vector,print_func=print2log)
        
        # Print results
        # print_competition_results(competitive_paths, print2log, top_n=10)

        # # Biological sanity check
        # validation_results = validate_biological_sanity(
        #     grid_path_scores_vector=grid_path_scores_vector,
        #     print_func=print2log,
        #     sources=sources)
        
        # # Validate path decomposition (sum of paths = global gradient)
        # validation_summary = validate_path_decomposition(
        #     grid_path_scores_vector=grid_path_scores_vector,
        #     model=model,
        #     x_baseline_batch=x_baseline_batch,
        #     x_stim_batch=Xin_critical,
        #     mod=mod,
        #     print_func=print2log
        # )
        
        # allpaths_validation_results = allpaths_biological_sanity_check(
        #     grid_path_scores_vector=grid_path_scores_vector,
        #     validation_summary=validation_summary,
        #     print_func=print2log
        # )

        grid_path_scores_norm = analyze_paths_with_normalization(
            grid_path_scores_vector=grid_path_scores_vector,
            print_func=print2log
        )

        path_selection_results = select_top_paths_with_rank_tracking(
            grid_path_scores_raw=grid_path_scores_vector,
            grid_path_scores_norm=grid_path_scores_norm,
            print_func=print2log
        )

        # Choose which paths to visualize:
        #   - Use normalized-top paths, but plot RAW scores for those paths
        tgf_res = path_selection_results.get('tgfbr1')
        if tgf_res is not None:
            top_tgfbr1_path_for_plot = (tgf_res['path'], tgf_res['raw_scores'])
        else:
            # Fallback: use the Captum test path (raw TGFBR1-only top)
            top_tgfbr1_path_for_plot = (top_tgfbr1_path_edges, top_tgfbr1_scores)

        ifn_res = path_selection_results.get('ifnar1')
        if ifn_res is not None:
            top_ifnar1_path_for_plot = (ifn_res['path'], ifn_res['raw_scores'])
        else:
            top_ifnar1_path_for_plot = (None, None)

        # ------------------------------------------------------------------
        # Visualize using the NORMALIZED-selected top paths (RAW values)
        # ------------------------------------------------------------------
        visualize_critical_paths_bar_TOP(
            top_tgfbr1_path=top_tgfbr1_path_for_plot,
            top_ifnar1_path=top_ifnar1_path_for_plot,
            grid_path_scores_vector=grid_path_scores_vector,
            out_dir=out_dir,
            model_no=model_no,
            print_func=print2log
        )

        # Captum validation still uses the raw TGFBR1-only top path
        visualize_top_path_captum_validation(
            top_path_edges=top_tgfbr1_path_edges,
            your_score=your_method_score,
            captum_score=captum_score,
            receptor_name='TGFBR1',
            out_dir=out_dir,
            model_no=model_no,
            print_func=print2log
        )
        
        # --- Update aggregation (Using the Combined (AI) score for ensemble voting) ---
        combined_score_vector = {path: scores[3] for path, scores in grid_path_scores_vector.items()}
        
        for path_edges in all_paths:
            score = combined_score_vector[path_edges]
            path_ig_sum[path_edges] += score
            path_ig_count[path_edges] += 1

        # # Track biological validation results across models
        # if 'validation_tracker' not in locals():
        #     validation_tracker = {
        #         'tgfbr1_total_profibrotic_count': 0,
        #         'tgfbr1_top_profibrotic_count': 0,
        #         'captum_top_profibrotic_count': 0,
        #         'decomposition_valid_count': 0,
        #         'total_models': 0,
        #         'failed_models_total': [],
        #         'failed_models_top': [],
        #         'decomposition_failed': [],
        #         'top_path_sign_agreement_count': 0,
        #         'top_path_sign_disagreement_count': 0
        #     }

        # validation_tracker['total_models'] += 1

        # # Check TOTAL contribution (for sanity)
        # if allpaths_validation_results.get('tgfbr1_passes', False):
        #     validation_tracker['tgfbr1_total_profibrotic_count'] += 1
        # else:
        #     validation_tracker['failed_models_total'].append(model_no)

        # # Check TOP path (your method)
        # if allpaths_validation_results.get('tgfbr1_top_profibrotic', False):
        #     validation_tracker['tgfbr1_top_profibrotic_count'] += 1
        # else:
        #     validation_tracker['failed_models_top'].append(model_no)

        # # Check if Captum also finds pro-fibrotic TOP path
        # if captum_score > 0:
        #     validation_tracker['captum_top_profibrotic_count'] += 1
        
        # --- STEP 3: Extract top path subgraph (Uses the Combined (AI) score) ---
        top_ifnar1_path, top_tgfbr1_path, result_graph, subgraph_df = extract_and_save_top_path_subgraph(
            combined_score_vector=combined_score_vector,
            A=A,
            out_dir=out_dir,
            model_no=model_no,
            edge_models=edge_models,
            node_models=node_models,
            edge_weight_sum=edge_weight_sum,
            edge_weight_abs_sum=edge_weight_abs_sum,
            to_symbol=to_symbol,
            print_func=print2log,
            top_n=top_n_paths,
            sources=sources
        )

        # if 'path_selection_results' in locals():
        #     A_idx, I_idx, AI_idx = 1, 2, 3
        #     for receptor_key in ('tgfbr1', 'ifnar1'):
        #         res = path_selection_results.get(receptor_key)
        #         if res is None:
        #             continue
        #         raw_scores = res['raw_scores']
        #         norm_scores = res['norm_scores']
        #         raw_val = raw_scores[AI_idx]
        #         norm_val = norm_scores[AI_idx]
        #         if raw_val == 0 or norm_val == 0:
        #             continue
        #         if np.sign(raw_val) == np.sign(norm_val):
        #             validation_tracker['top_path_sign_agreement_count'] += 1
        #         else:
        #             validation_tracker['top_path_sign_disagreement_count'] += 1
        
        print2log(f"\n{'='*70}")
        print2log(f"✓ COMPLETED MODEL {model_no+1}/{number_DTLEMBAS_models}")
        print2log(f"{'='*70}\n")
    
    # AGGREGATE ACROSS ALL MODELS
    aggregated_edges, df_aggregated = aggregate_subgraph_edges(
        edge_models=edge_models,
        edge_weight_sum=edge_weight_sum,
        edge_weight_abs_sum=edge_weight_abs_sum,
        number_models=number_DTLEMBAS_models,
        out_dir=out_dir,
        print_func=print2log
    )

    # Visualize aggregated subgraph
    visualize_aggregated_subgraph(
        aggregated_edges=aggregated_edges,
        node_models=node_models,
        number_models=number_DTLEMBAS_models,
        out_dir=out_dir,
        print_func=print2log,
        freq_threshold=args.EDGE_FREQ_MIN
    )

    # Aggregate path scores
    avg_path_scores, sorted_paths = aggregate_path_scores(
        path_ig_sum=path_ig_sum,
        path_ig_count=path_ig_count,
        number_models=number_DTLEMBAS_models,
        print_func=print2log
    )

    # Save final results
    save_final_aggregated_results(
        avg_path_scores=avg_path_scores,
        aggregated_edges=aggregated_edges,
        path_ig_sum=path_ig_sum,
        path_ig_count=path_ig_count,
        edge_models=edge_models,
        node_models=node_models,
        number_models=number_DTLEMBAS_models,
        out_dir=out_dir,
        print_func=print2log
    )

    # print2log('\n' + '='*70)
    # print2log('FINAL VALIDATION SUMMARY')
    # print2log('='*70)

    # # Biological validation
    # profibrotic_total_percent = (validation_tracker['tgfbr1_total_profibrotic_count'] / 
    #                          validation_tracker['total_models'] * 100)
    # profibrotic_top_percent = (validation_tracker['tgfbr1_top_profibrotic_count'] / 
    #                         validation_tracker['total_models'] * 100)
    # captum_top_percent = (validation_tracker['captum_top_profibrotic_count'] / 
    #                     validation_tracker['total_models'] * 100)

    # print2log(f"\n1. SANITY CHECK - TGFBR1 Total Contribution (All Paths):")
    # print2log(f"   {validation_tracker['tgfbr1_total_profibrotic_count']}/{validation_tracker['total_models']} models ({profibrotic_total_percent:.1f}%) have pro-fibrotic total effect")

    # if profibrotic_total_percent >= 90:
    #     print2log(f"   ✅ EXCELLENT - High consistency with biological knowledge!")
    # elif profibrotic_total_percent >= 70:
    #     print2log(f"   ⚠️  MODERATE - Some models show unexpected total effects")
    # else:
    #     print2log(f"   ❌ POOR - Many models contradict known TGFB biology")

    # print2log(f"\n2. TOP PATH COMPARISON (The King!):")
    # print2log(f"   My method (top path):      {validation_tracker['tgfbr1_top_profibrotic_count']}/{validation_tracker['total_models']} models ({profibrotic_top_percent:.1f}%) have pro-fibrotic TGFBR1 top path")
    # print2log(f"   Captum (top path, masked): {validation_tracker['captum_top_profibrotic_count']}/{validation_tracker['total_models']} models ({captum_top_percent:.1f}%) have pro-fibrotic TGFBR1 top path")

    # # Compare top path consistency
    # if abs(profibrotic_top_percent - captum_top_percent) < 10:
    #     print2log(f"\n   ✅ EXCELLENT - Your top path and Captum agree ({abs(profibrotic_top_percent - captum_top_percent):.1f}% difference)")
    # elif abs(profibrotic_top_percent - captum_top_percent) < 30:
    #     print2log(f"\n   ⚠️  MODERATE - Some discrepancy between methods ({abs(profibrotic_top_percent - captum_top_percent):.1f}% difference)")
    # else:
    #     print2log(f"\n   ❌ POOR - Large discrepancy between methods ({abs(profibrotic_top_percent - captum_top_percent):.1f}% difference)")

    # if validation_tracker['failed_models_top']:
    #     print2log(f"\n   Models with anti-fibrotic TOP path: {validation_tracker['failed_models_top'][:10]}")

    # # RAW vs NORMALIZED TOP PATH SIGN CHECK
    # agree = validation_tracker.get('top_path_sign_agreement_count', 0)
    # disagree = validation_tracker.get('top_path_sign_disagreement_count', 0)
    # total_sign = agree + disagree
    # print2log(f"\n RAW vs NORMALIZED TOP PATH SIGN CHECK:")
    # if total_sign == 0:
    #     print2log("   (No receptor–model pairs with both raw and normalized top-path signs available.)")
    # else:
    #     frac = 100.0 * agree / total_sign
    #     print2log(
    #         f"   Agreement in combined-effect sign for selected top paths: "
    #         f"{agree}/{total_sign} receptor–model pairs ({frac:.1f}%)"
    #     )
    #     if frac >= 90:
    #         print2log("   ✅ EXCELLENT - Normalization preserves the biological direction of the top paths.")
    #     elif frac >= 70:
    #         print2log("   ⚠️  MODERATE - Some top paths flip sign after normalization; interpret with care.")
    #     else:
    #         print2log("   ❌ POOR - Normalization often reverses the top path direction; reconsider the ranking strategy.")
    
    print2log("\n" + "="*70)
    print2log(f"Finished timepoint: {timepoint}")
    print2log("="*70)

print2log("\n" + "="*70)
print2log("ANALYSIS COMPLETE!")
print2log("="*70)