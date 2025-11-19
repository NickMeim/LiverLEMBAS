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
# import pcst_fast
import pyreadr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,RepeatedKFold
from collections import defaultdict
from itertools import islice
from tqdm import tqdm
import pickle

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
    derivative = bionet_layer.delta(z, leak)
    
    # Jacobian = diag(f'(z)) @ W (NO +I because X_bias is constant!)
    jacobian = derivative.squeeze(1).unsqueeze(1) * weights
    
    return jacobian


def test_jacobian_computation(bionet_layer, x_equilibrium, rtol=1e-4):
    """
    Verify that efficient analytical Jacobian matches autograd.
    """
    
    print("\n" + "="*70)
    print("TESTING JACOBIAN COMPUTATION")
    print("="*70)
    
    # Method 1: Efficient analytical (fast)
    import time
    start = time.time()
    jac_analytical = compute_single_step_jacobian_efficient(bionet_layer, x_equilibrium)
    time_analytical = time.time() - start
    
    # Method 2: Loop-based autograd (slow but reliable reference)
    start = time.time()
    jac_autograd = compute_single_step_jacobian_autograd_reference(bionet_layer, x_equilibrium)
    time_autograd = time.time() - start
    
    # Compare
    max_diff = torch.max(torch.abs(jac_autograd - jac_analytical)).item()
    max_val = torch.max(torch.abs(jac_autograd)).item()
    rel_diff = max_diff / (max_val + 1e-8)
    
    print(f"\nEfficient analytical: {time_analytical:.4f}s")
    print(f"Autograd reference: {time_autograd:.4f}s")
    print(f"Speedup: {time_autograd/time_analytical:.1f}x")
    print(f"\nMax absolute difference: {max_diff:.2e}")
    print(f"Max relative difference: {rel_diff:.2e}")
    
    if rel_diff < rtol:
        print("\n✓ PASSED: Jacobians match within tolerance!")
        return True
    else:
        print("\n✗ FAILED: Jacobians differ!")
        # Print some sample values for debugging
        print("\nSample comparison (first 5x5):")
        print("Analytical:")
        print(jac_analytical[:5, :5])
        print("Autograd:")
        print(jac_autograd[:5, :5])
        return False


def compute_single_step_jacobian_autograd_reference(bionet_layer, x_equilibrium):
    """
    Reference implementation using autograd (slow but correct).
    Matches the ACTUAL BioNet.forward() recurrence exactly.
    """
    
    x_eq = x_equilibrium.clone().requires_grad_(True)
    
    weights = bionet_layer.weights
    bias = bionet_layer.bias
    leak = bionet_layer.training_params['leak']
    
    # ONE step of recurrence (matching BioNet.forward EXACTLY)
    X_bias = x_eq.T + bias  # Constant term
    X_new = torch.mm(weights, x_eq.T)  # W @ X
    X_new = X_new + X_bias              # Add constant
    X_new = bionet_layer.activation(X_new, leak)  # Activation
    X_new = X_new.T  # Back to (1, num_nodes)
    
    num_nodes = x_eq.shape[1]
    jacobian = torch.zeros((num_nodes, num_nodes), dtype=x_eq.dtype, device=x_eq.device)
    
    for i in range(num_nodes):
        if x_eq.grad is not None:
            x_eq.grad.zero_()
        
        grad_outputs = torch.zeros_like(X_new)
        grad_outputs[0, i] = 1.0
        X_new.backward(grad_outputs, retain_graph=True)  # ← Fixed: was X_next, now X_new
        
        if x_eq.grad is not None:
            jacobian[i, :] = x_eq.grad[0, :].clone()
    
    return jacobian


# ============================================================================
# PATH SCORING WITH EFFICIENT JACOBIAN
# ============================================================================

def compute_path_gradient_from_jacobian(path_edges, node_idx_map, jacobian,
                                        grad_tfs_wrt_nodes, pheno_coefs, outName):
    """
    Compute path gradient using pre-computed Jacobian.
    
    For path P = [n0 → n1 → n2 → ... → nk → TF → fibrosis]:
        gradient = J[n1,n0] × J[n2,n1] × ... × J[nk,n(k-1)] × ∂TF/∂nk × coef[TF]
    
    Args:
        path_edges: List of (u, v) tuples defining the path
        node_idx_map: Dict mapping node names to indices
        jacobian: Pre-computed Jacobian matrix, shape (num_nodes, num_nodes)
        grad_tfs_wrt_nodes: Gradients of TFs w.r.t. nodes, shape (num_tfs, num_nodes)
        pheno_coefs: Linear coefficients from TFs to phenotype
        outName: List of TF names
    
    Returns:
        gradient: Scalar gradient along this path
    """
    
    source_node = path_edges[0][0]
    source_idx = node_idx_map[source_node]
    
    current_gradient = 1.0
    current_node_idx = source_idx
    
    for (u, v) in path_edges:
        if v == 'fibrosis':
            # Final edge: TF -> fibrosis
            tf_name = u
            tf_idx = outName.index(tf_name)
            
            # ∂fibrosis/∂TF (linear coefficient)
            grad_fibrosis_wrt_tf = pheno_coefs[tf_idx]
            
            # ∂TF/∂current_node
            grad_tf_wrt_current = grad_tfs_wrt_nodes[tf_idx, current_node_idx].item()
            
            current_gradient *= grad_fibrosis_wrt_tf * grad_tf_wrt_current
        else:
            # Edge within signaling network
            v_idx = node_idx_map[v]
            
            # Partial derivative from Jacobian
            partial_derivative = jacobian[v_idx, current_node_idx].item()
            
            current_gradient *= partial_derivative
            current_node_idx = v_idx
    
    return current_gradient


def score_paths_single_step_jacobian(model, all_paths, node_idx_map,
                                     x_baseline, x_stim, pheno_coefs, outName,
                                     num_steps_ig=50,
                                     show_progress=True):
    """
    Score paths using efficient analytical single-step Jacobian.
    
    This is the RECOMMENDED method - closest analogy to GGM partial correlations.
    
    Args:
        model: Sequential model [signaling_network, output_layer, phenotype]
        all_paths: List of paths (each path is tuple of (u,v) edges)
        node_idx_map: Dict mapping node names to indices
        x_baseline: Baseline input
        x_stim: Stimulation input
        pheno_coefs: Coefficients from TFs to phenotype
        outName: List of TF names
        num_steps_ig: Number of IG integration steps
        show_progress: Whether to show progress bar
    
    Returns:
        path_scores: Dict mapping path_edges -> path IG score
    """
    
    model.eval()
    path_scores = {path: 0.0 for path in all_paths}
    
    iterator = range(1, num_steps_ig + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="IG steps", leave=False, ncols=100)
    
    for step in iterator:
        alpha = step / num_steps_ig
        x_alpha = x_baseline + alpha * (x_stim - x_baseline)
        
        # Get equilibrium state (model[0] handles recurrence internally)
        with torch.no_grad():
            x_equilibrium = model[0](x_alpha)
        
        # Compute single-step Jacobian efficiently (NO LOOPS!)
        jacobian = compute_single_step_jacobian_efficient(model[0], x_equilibrium, x_alpha)
        
        # Compute TF gradients
        x_eq_grad = x_equilibrium.clone().requires_grad_(True)
        tf_acts = model[1](x_eq_grad)
        
        num_nodes = x_equilibrium.shape[1]
        grad_tfs_wrt_nodes = torch.zeros((len(outName), num_nodes), dtype=torch.double)
        
        for tf_idx in range(len(outName)):
            grad = torch.autograd.grad(
                outputs=tf_acts[0, tf_idx],
                inputs=x_eq_grad,
                retain_graph=True
            )[0]
            grad_tfs_wrt_nodes[tf_idx, :] = grad[0, :]
        
        # Score all paths (fast - just lookups and multiplications)
        for path_edges in all_paths:
            gradient = compute_path_gradient_from_jacobian(
                path_edges, node_idx_map, jacobian,
                grad_tfs_wrt_nodes, pheno_coefs, outName
            )
            path_scores[path_edges] += gradient
    
    # Finalize IG scores
    for path_edges in all_paths:
        source_idx = node_idx_map[path_edges[0][0]]
        delta_source = (x_stim - x_baseline)[0, source_idx].item()
        path_scores[path_edges] = (path_scores[path_edges] / num_steps_ig) * delta_source
    
    return path_scores


# ============================================================================
# GRID-BASED COMPETITION ANALYSIS
# ============================================================================

def score_paths_across_grid(model, all_paths, node_idx_map,
                            x_baseline, Xin_combined,
                            pheno_coefs, outName,
                            num_steps_ig=50,
                            checkpoint_dir=None,
                            checkpoint_prefix='',
                            resume_from_checkpoint=True):
    """
    Score all paths across the ENTIRE stimulation grid.
    
    Args:
        model: Sequential model [signaling_network, output_layer, phenotype]
        all_paths: List of paths
        node_idx_map: Dict mapping node names to indices
        x_baseline: Baseline input (typically zeros)
        Xin_combined: Grid of stimulations, shape (num_grid_points, num_nodes)
        pheno_coefs: Coefficients from TFs to phenotype
        outName: List of TF names
        num_steps_ig: Number of IG integration steps per grid point
        checkpoint_dir: Directory for checkpoints
        checkpoint_prefix: Prefix for checkpoint files
        resume_from_checkpoint: Whether to resume from checkpoint
    
    Returns:
        grid_path_scores: Dict[grid_idx -> Dict[path -> score]]
        grid_metadata: Dict[grid_idx -> metadata]
    """
    
    model.eval()
    num_grid_points = Xin_combined.shape[0]
    
    # Initialize storage
    grid_path_scores = {}
    grid_metadata = {}
    
    # Setup checkpointing
    checkpoint_path = None
    start_grid_idx = 0
    
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_prefix}_grid_checkpoint.pkl')
        
        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                grid_path_scores = checkpoint_data['grid_path_scores']
                grid_metadata = checkpoint_data['grid_metadata']
                start_grid_idx = checkpoint_data['last_grid_idx'] + 1
                
                print(f"✓ Resumed from checkpoint: grid point {start_grid_idx}/{num_grid_points}")
                
                if start_grid_idx >= num_grid_points:
                    print(f"✓ Already completed! Returning cached results.")
                    return grid_path_scores, grid_metadata
                    
            except Exception as e:
                print(f"⚠ Warning: Could not load checkpoint: {e}")
                start_grid_idx = 0
    
    print(f"\n{'='*70}")
    print(f"SCORING PATHS ACROSS STIMULATION GRID")
    print(f"{'='*70}")
    print(f"Grid size: {num_grid_points} points")
    print(f"Paths to score: {len(all_paths)}")
    print(f"IG steps per point: {num_steps_ig}")
    print(f"Method: Single-step Jacobian (GGM analogy)")
    print(f"Checkpoint: {'Enabled' if checkpoint_path else 'Disabled'}")
    print(f"{'='*70}\n")
    
    # Progress bar for grid points
    pbar_grid = tqdm(range(start_grid_idx, num_grid_points),
                     desc="Grid Points",
                     unit="point",
                     ncols=100)
    
    try:
        for grid_idx in pbar_grid:
            # Get this grid point's stimulation
            x_stim = Xin_combined[grid_idx:grid_idx+1, :]
            
            # Extract stimulation levels for metadata
            tgfbr1_idx = node_idx_map.get('P36897')
            ifnar1_idx = node_idx_map.get('P17181')
            
            if tgfbr1_idx is not None and ifnar1_idx is not None:
                tgfbr1_stim = x_stim[0, tgfbr1_idx].item()
                ifnar1_stim = x_stim[0, ifnar1_idx].item()
            else:
                tgfbr1_stim = 0
                ifnar1_stim = 0
            
            pbar_grid.set_postfix({
                'TGFBR1': f'{tgfbr1_stim:.0f}', 
                'IFNAR1': f'{ifnar1_stim:.0f}'
            })
            
            # Score paths for this grid point
            path_scores = score_paths_single_step_jacobian(
                model=model,
                all_paths=all_paths,
                node_idx_map=node_idx_map,
                x_baseline=x_baseline,
                x_stim=x_stim,
                pheno_coefs=pheno_coefs,
                outName=outName,
                num_steps_ig=num_steps_ig,
                show_progress=False  # Don't show nested progress bar
            )
            
            # Store results
            grid_path_scores[grid_idx] = path_scores
            grid_metadata[grid_idx] = {
                'tgfbr1_stim': tgfbr1_stim,
                'ifnar1_stim': ifnar1_stim,
                'x_stim': x_stim.cpu().numpy()
            }
            
            # Save checkpoint periodically
            if checkpoint_path and (grid_idx % 10 == 0 or grid_idx == num_grid_points - 1):
                checkpoint_data = {
                    'grid_path_scores': grid_path_scores,
                    'grid_metadata': grid_metadata,
                    'last_grid_idx': grid_idx,
                    'num_grid_points': num_grid_points
                }
                
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                pbar_grid.set_postfix({
                    'TGFBR1': f'{tgfbr1_stim:.0f}', 
                    'IFNAR1': f'{ifnar1_stim:.0f}',
                    'saved': '✓'
                })
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Saving checkpoint...")
        if checkpoint_path:
            checkpoint_data = {
                'grid_path_scores': grid_path_scores,
                'grid_metadata': grid_metadata,
                'last_grid_idx': grid_idx - 1,
                'num_grid_points': num_grid_points
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"✓ Checkpoint saved. Resume from grid point {grid_idx}")
        raise
    
    finally:
        pbar_grid.close()
    
    print(f"\n✓ Completed scoring {len(all_paths)} paths across {num_grid_points} grid points")
    
    # Clean up checkpoint on success
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Checkpoint cleaned up")
    
    return grid_path_scores, grid_metadata


def analyze_competition_grid(grid_path_scores, grid_metadata, all_paths,
                             stimulation_strength):
    """
    Analyze competition patterns across the stimulation grid.
    
    Returns:
        competitive_paths: Paths with sign changes (opposite effects)
        cooperative_paths: Paths with synergistic effects
        path_importance_grids: Dict mapping paths to 2D score arrays
    """
    
    print("\n" + "="*70)
    print("ANALYZING COMPETITION PATTERNS ACROSS GRID")
    print("="*70)
    
    num_grid_points = len(grid_path_scores)
    grid_size = int(np.sqrt(num_grid_points))
    
    print(f"Grid size: {grid_size} × {grid_size}")
    print(f"Analyzing {len(all_paths)} paths...")
    
    # Create importance grids for each path
    path_importance_grids = {}
    
    for path_edges in tqdm(all_paths, desc="Creating importance grids", ncols=100):
        scores = []
        for grid_idx in range(num_grid_points):
            score = grid_path_scores[grid_idx].get(path_edges, 0)
            scores.append(score)
        
        score_grid = np.array(scores).reshape((grid_size, grid_size))
        path_importance_grids[path_edges] = score_grid
    
    # Find competitive paths (sign changes across grid)
    competitive_paths = []
    
    for path_edges, score_grid in path_importance_grids.items():
        has_positive = (score_grid > 0.01).any()
        has_negative = (score_grid < -0.01).any()
        
        if has_positive and has_negative:
            max_val = score_grid.max()
            min_val = score_grid.min()
            competition_strength = abs(max_val - min_val)
            
            competitive_paths.append({
                'path': path_edges,
                'score_grid': score_grid,
                'max_score': max_val,
                'min_score': min_val,
                'competition': competition_strength
            })
    
    competitive_paths.sort(key=lambda x: x['competition'], reverse=True)
    
    # Find cooperative/synergistic paths
    cooperative_paths = []
    
    center_idx = grid_size // 2  # (0,0) baseline point
    
    for path_edges, score_grid in path_importance_grids.items():
        # Scores at key grid points
        baseline_score = score_grid[center_idx, center_idx]
        tgfbr1_only_score = score_grid[center_idx, -1]  # Max TGFBR1, zero IFNAR1
        ifnar1_only_score = score_grid[-1, center_idx]  # Zero TGFBR1, max IFNAR1
        both_score = score_grid[-1, -1]  # Both max
        
        # Expected additive effect
        expected = tgfbr1_only_score + ifnar1_only_score - baseline_score
        synergy = both_score - expected
        
        if abs(synergy) > 0.01:
            cooperative_paths.append({
                'path': path_edges,
                'score_grid': score_grid,
                'tgfbr1_only': tgfbr1_only_score,
                'ifnar1_only': ifnar1_only_score,
                'both': both_score,
                'expected': expected,
                'synergy': synergy
            })
    
    cooperative_paths.sort(key=lambda x: abs(x['synergy']), reverse=True)
    
    print(f"\n✓ Found {len(competitive_paths)} competitive paths (sign changes)")
    print(f"✓ Found {len(cooperative_paths)} cooperative paths (synergy/interference)")
    
    return competitive_paths, cooperative_paths, path_importance_grids


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_competition_heatmaps(competitive_paths, stimulation_strength,
                                   out_dir, model_no, top_n=10):
    """
    Create heatmap visualizations of top competitive paths.
    """
    
    n_paths_to_plot = min(top_n, len(competitive_paths))
    if n_paths_to_plot == 0:
        print("No competitive paths to plot")
        return
    
    n_rows = (n_paths_to_plot + 4) // 5  # Ceiling division
    n_cols = min(5, n_paths_to_plot)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(n_rows * n_cols):
        ax = axes[i]
        
        if i < n_paths_to_plot:
            path_info = competitive_paths[i]
            score_grid = path_info['score_grid']
            path_edges = path_info['path']
            
            # Create heatmap
            vmax = max(abs(score_grid.max()), abs(score_grid.min()))
            im = ax.imshow(score_grid, cmap='RdBu_r', origin='lower',
                          extent=[-stimulation_strength, stimulation_strength,
                                 -stimulation_strength, stimulation_strength],
                          vmin=-vmax, vmax=vmax, aspect='auto')
            
            ax.set_xlabel('TGFBR1 stimulation', fontsize=9)
            ax.set_ylabel('IFNAR1 stimulation', fontsize=9)
            
            # Path description
            source = path_edges[0][0]
            target = path_edges[-1][1]
            ax.set_title(f"#{i+1}: {source} → {target}\n"
                        f"Competition: {path_info['competition']:.3f}", 
                        fontsize=9)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Mark key points
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'competition_heatmaps_model{model_no}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved competition heatmaps to: {fig_path}")


def print_competition_results(competitive_paths, cooperative_paths, print_func, top_n=10):
    """
    Pretty print competition analysis results.
    """
    
    print_func("\n" + "="*70)
    print_func("TOP COMPETITIVE PATHS (opposite effects across grid)")
    print_func("="*70)
    
    for i, path_info in enumerate(competitive_paths[:top_n]):
        path = path_info['path']
        source = path[0][0]
        target = path[-1][1]
        
        print_func(f"\n{i+1}. Path: {source} → ... → {target}")
        print_func(f"   Full path: {' → '.join([e[0] for e in path] + [target])}")
        print_func(f"   Max score: {path_info['max_score']:+.6f}")
        print_func(f"   Min score: {path_info['min_score']:+.6f}")
        print_func(f"   Competition strength: {path_info['competition']:.6f}")
    
    print_func("\n" + "="*70)
    print_func("TOP COOPERATIVE PATHS (synergistic effects)")
    print_func("="*70)
    
    for i, path_info in enumerate(cooperative_paths[:top_n]):
        path = path_info['path']
        source = path[0][0]
        target = path[-1][1]
        
        print_func(f"\n{i+1}. Path: {source} → ... → {target}")
        print_func(f"   Full path: {' → '.join([e[0] for e in path] + [target])}")
        print_func(f"   TGFBR1 only:  {path_info['tgfbr1_only']:+.6f}")
        print_func(f"   IFNAR1 only:  {path_info['ifnar1_only']:+.6f}")
        print_func(f"   Expected sum: {path_info['expected']:+.6f}")
        print_func(f"   Actual both:  {path_info['both']:+.6f}")
        print_func(f"   Synergy:      {path_info['synergy']:+.6f}")
        
        if path_info['synergy'] > 0:
            print_func(f"   → Positive synergy: combined > sum")
        else:
            print_func(f"   → Negative synergy: interference/competition")

def extract_and_save_top_path_subgraph(grid_path_scores, grid_metadata, all_paths,
                                       A, nodeNames, outName, out_dir, model_no,
                                       edge_models, node_models,
                                       edge_weight_sum, edge_weight_abs_sum,
                                       to_symbol, print_func,
                                       sources=['P36897', 'P17181']):
    """
    Extract top paths from IFNAR1 and TGFBR1, save subgraph as CSV and PNG.
    Updates global aggregators.
    
    Args:
        grid_path_scores: Dict[grid_idx -> Dict[path -> score]]
        grid_metadata: Dict[grid_idx -> metadata]
        all_paths: List of all paths
        A: Adjacency matrix DataFrame (includes 'fibrosis' node)
        nodeNames: List of node names
        outName: List of TF names
        out_dir: Output directory
        model_no: Model number
        edge_models, node_models: Global aggregators (modified in place)
        edge_weight_sum, edge_weight_abs_sum: Global aggregators (modified in place)
        to_symbol: Function to convert UniProt ID to gene symbol
        print_func: Logging function
        sources: List of receptor node IDs
    
    Returns:
        top_ifnar1_path: Tuple of (path_edges, score)
        top_tgfbr1_path: Tuple of (path_edges, score)
        result_graph: NetworkX DiGraph of subgraph
        df: DataFrame of subgraph edges
    """
    
    print_func("\n" + "="*70)
    print_func("STEP 3: EXTRACT TOP PATH SUBGRAPH")
    print_func("="*70)
    
    # Use the max stimulation grid point (both receptors fully activated)
    num_grid_points = len(grid_path_scores)
    grid_size = int(np.sqrt(num_grid_points))
    max_stim_idx = num_grid_points - 1  # Last point in grid
    
    path_scores = grid_path_scores[max_stim_idx]
    
    print_func(f"Using grid point {max_stim_idx} (both receptors at max stimulation)")
    
    # Find top paths from each receptor
    ifnar1_paths = {k: v for k, v in path_scores.items() if k[0][0] == 'P17181'}
    tgfbr1_paths = {k: v for k, v in path_scores.items() if k[0][0] == 'P36897'}
    
    if not ifnar1_paths:
        print_func("⚠ Warning: No paths found starting from IFNAR1 (P17181)")
        top_ifnar1_path = None
    else:
        top_ifnar1_path = max(ifnar1_paths.items(), key=lambda item: abs(item[1]))
        print_func(f"\n✓ Top IFNAR1 (P17181) Path Score: {top_ifnar1_path[1]:+.6f}")
        print_func(f"  Path: {' → '.join([e[0] for e in top_ifnar1_path[0]] + [top_ifnar1_path[0][-1][1]])}")
    
    if not tgfbr1_paths:
        print_func("⚠ Warning: No paths found starting from TGFBR1 (P36897)")
        top_tgfbr1_path = None
    else:
        top_tgfbr1_path = max(tgfbr1_paths.items(), key=lambda item: abs(item[1]))
        print_func(f"\n✓ Top TGFBR1 (P36897) Path Score: {top_tgfbr1_path[1]:+.6f}")
        print_func(f"  Path: {' → '.join([e[0] for e in top_tgfbr1_path[0]] + [top_tgfbr1_path[0][-1][1]])}")
    
    # Collect edges from top paths
    print_func("\nExtracting and saving Path-IG Subgraph...")
    result_edges_set = set()
    
    if top_ifnar1_path and top_ifnar1_path[0]:
        for u, v in top_ifnar1_path[0]:
            result_edges_set.add((u, v))
        print_func(f"  Added {len(top_ifnar1_path[0])} edges from IFNAR1 path")
    
    if top_tgfbr1_path and top_tgfbr1_path[0]:
        for u, v in top_tgfbr1_path[0]:
            result_edges_set.add((u, v))
        print_func(f"  Added {len(top_tgfbr1_path[0])} edges from TGFBR1 path")
    
    if not result_edges_set:
        print_func("⚠ Warning: No edges to save (no valid paths found)")
        return top_ifnar1_path, top_tgfbr1_path, None, None
    
    # Build result graph
    result_graph = nx.DiGraph(list(result_edges_set))
    result_nodes_uniprot = list(result_graph.nodes())
    
    print_func(f"\n✓ Nodes in Path-IG subgraph: {len(result_nodes_uniprot)}")
    print_func(f"✓ Subgraph number of edges: {len(result_edges_set)}")
    
    # Create CSV data
    print_func("\nSaving subgraph edgelist to CSV...")
    subgraph_data = []
    
    for u, v in result_edges_set:
        try:
            weight = A.loc[u, v]
        except KeyError:
            print_func(f"  ⚠ Warning: Edge ({u}, {v}) not found in A, using 0.0")
            weight = 0.0
        
        try:
            source_symbol = to_symbol(u)
            target_symbol = to_symbol(v)
        except:
            source_symbol = u
            target_symbol = v
        
        subgraph_data.append({
            'source_uniprot': u,
            'target_uniprot': v,
            'source_symbol': source_symbol,
            'target_symbol': target_symbol,
            'weight': weight,
            'path_ID': 'IFNAR1_TGFBR1_Top_Set'
        })
    
    df = pd.DataFrame(subgraph_data)
    
    # Save CSV
    csv_path = os.path.join(out_dir, 'individual_networks', f'subgraph_edgelist_{model_no}.csv')
    df.to_csv(csv_path, index=False)
    print_func(f"✓ Subgraph saved to: subgraph_edgelist_{model_no}.csv")
    
    # Update global aggregators
    print_func("\nUpdating global aggregators...")
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
    
    print_func(f"✓ Updated aggregators: {len(df)} edges processed")
    
    # Visualize and save
    print_func("\nVisualizing the subgraph...")
    visualize_subgraph(result_graph, sources, to_symbol, out_dir, model_no, print_func)
    
    print_func("="*70)
    
    return top_ifnar1_path, top_tgfbr1_path, result_graph, df


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
    labels = {node: to_symbol(node) if node not in ['fibrosis'] else node 
              for node in result_graph.nodes()}
    nx.draw_networkx_labels(result_graph, pos, labels=labels, font_color='black', font_size=9)
    
    # Highlight input nodes (receptors)
    input_nodes = [n for n in sources if n in result_graph]
    if input_nodes:
        nx.draw_networkx_nodes(result_graph, pos, nodelist=input_nodes, 
                              node_color='red', node_size=700, label='Receptors')
    
    # Highlight output node (fibrosis)
    output_node = 'fibrosis'
    if output_node in result_graph:
        nx.draw_networkx_nodes(result_graph, pos, nodelist=[output_node], 
                              node_color='gold', node_size=700, label='Fibrosis')
    
    plt.title(f"Model {model_no+1}: Top IFNAR1 & TGFBR1 Paths to Fibrosis", 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(out_dir, 'individual_networks', f"path_ig_subgraph_result_{model_no}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print_func(f"✓ Visualization saved to: path_ig_subgraph_result_{model_no}.png")


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_path_scores(grid_path_scores, grid_metadata, competitive_paths, 
                    cooperative_paths, path_importance_grids, 
                    out_dir, model_no):
    """
    Save all path scoring results to disk.
    """
    
    results_dir = Path(out_dir) / 'path_ig_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive results
    results = {
        'grid_path_scores': grid_path_scores,
        'grid_metadata': grid_metadata,
        'competitive_paths': competitive_paths,
        'cooperative_paths': cooperative_paths,
        'path_importance_grids': path_importance_grids,
        'model_number': model_no
    }
    
    results_path = results_dir / f'model_{model_no}_path_ig_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Saved results to: {results_path}")
    
    return results_path


# ============================================================================
# AGGREGATION ACROSS MODELS
# ============================================================================

def aggregate_subgraph_edges(edge_models, node_models, edge_weight_sum, 
                            edge_weight_abs_sum, number_models, out_dir, print_func):
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
    print_func(f"  Edges in >50% models: {sum(1 for e in aggregated_edges if e['frequency'] > 0.5)}")
    print_func(f"  Edges in >75% models: {sum(1 for e in aggregated_edges if e['frequency'] > 0.75)}")
    
    # Print top edges
    print_func("\nTop 20 Most Frequent Edges (across models):")
    print_func("="*70)
    for i, edge in enumerate(aggregated_edges[:20]):
        print_func(f"{i+1}. {edge['source_symbol']} → {edge['target_symbol']}")
        print_func(f"   Frequency: {edge['frequency']:.1%} ({edge['num_models']}/{number_models} models)")
        print_func(f"   Avg weight: {edge['avg_weight']:+.4f}")
    
    return aggregated_edges, df_aggregated


def visualize_aggregated_subgraph(aggregated_edges, node_models, number_models, 
                                  out_dir, print_func, freq_threshold=0.5):
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
    
    for i, (path_edges, avg_score) in enumerate(sorted_paths[:20]):
        source = path_edges[0][0]
        target = path_edges[-1][1]
        path_length = len(path_edges)
        
        print_func(f"\n{i+1}. {source} → ... → {target}")
        print_func(f"   Path length: {path_length}")
        print_func(f"   Average IG score: {avg_score:+.6f}")
    
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
parser.add_argument('--max_path_length', action='store',type = int, default=10)
parser.add_argument('--stimulation_strength', action='store',type=float, default=3000)
parser.add_argument('--num_steps', action='store',type = float, default=11) # make it at least 21 or 101
parser.add_argument('--num_steps_ig', action='store',type = float, default=10) #make it at least 50
parser.add_argument('--EDGE_FREQ_MIN', action='store',type = float, default=0.2)
parser.add_argument('--NODE_FREQ_MIN', action='store',type = float, default=0.5)

fibrosis_output_idx = 0
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
max_path_length = args.max_path_length
num_steps_ig =  args.num_steps_ig
out_dir = args.out_dir

Path(os.path.join(out_dir, 'individual_networks')).mkdir(parents=True, exist_ok=True)

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
for timepoint in ['8 hour']:  # or ['2 hour','8 hour','24 hour']
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
    Yh = Yh[:,1]
    params = {'alpha': [0.001,0.01,0.1,0.2,0.5,1,2,5,10,20,50,100,500,1000]}
    ridge = GridSearchCV(estimator=Ridge(),param_grid=params, cv=RepeatedKFold(n_splits=10, n_repeats=10), n_jobs=-1)
    ridge.fit(Xh, Yh)
    coefs = ridge.best_estimator_.coef_
    intercept = ridge.best_estimator_.intercept_
    pheno = FrozenLinear(coefs[np.newaxis, :], intercept, dtype=torch.double, device=device)
    pheno.eval()
    
    # ============================================================================
    # MAIN MODEL LOOP
    # ============================================================================
    
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
        A = pd.concat([A, pd.DataFrame(np.zeros((len(nodeNames),1)), index=nodeNames, columns=['fibrosis'])], axis=1)
        A = pd.concat([A, pd.DataFrame(np.zeros((1,len(nodeNames)+1)), index=['fibrosis'], columns=list(nodeNames)+['fibrosis'])], axis=0)
        A.loc[outName, 'fibrosis'] = coefs[:]
        
        sources = ['P36897','P17181']  # TGFBR1, IFNAR1
        G = nx.from_pandas_adjacency(A, create_using=nx.DiGraph())
        
        # Build model
        model = torch.nn.Sequential(
            mod.signaling_network,
            mod.output_layer,
            pheno
        )
        model.eval()
        
        # OPTIONAL: Test Jacobian computation on first model
        if model_no == 0:
            print2log("\nTesting Jacobian computation on first model...")
            x_test = torch.zeros((1, len(nodeNames)), dtype=torch.double).to(device)
            with torch.no_grad():
                x_eq_test = model[0](x_test)
            test_jacobian_computation(model[0], x_eq_test)
        
        # --- STEP 0: Create grid of stimulations ---
        print2log('\n' + '='*70)
        print2log('STEP 0: CREATING STIMULATION GRID')
        print2log('='*70)
        
        nodeNames_array = np.array(list(mod.node_idx_map.keys()))
        all_nodes = np.array(list(mod.node_idx_map.keys()))
        
        subset_nodes = {key: mod.node_idx_map[key] for key in sources if key in mod.node_idx_map}
        subset_nodes_inds = np.array(list(subset_nodes.values()))
        
        Xin = torch.eye(len(all_nodes), dtype=torch.double).to(device)
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
        
        print2log(f"Grid size: {int(num_steps)} × {int(num_steps)} = {Xin_combined.shape[0]} points")
        print2log(f"Stimulation range: [{-stimulation_strength:.0f}, {stimulation_strength:.0f}]")
        
        # Calculate net effects
        netEffect = np.linalg.inv(np.eye(len(A)) - A.values)
        print2log(f"Net effect of IFNAR1 (P17181) on fibrosis: {netEffect[subset_nodes_inds[1],-1]:.4f}")
        print2log(f"Net effect of TGFBR1 (P36897) on fibrosis: {netEffect[subset_nodes_inds[0],-1]:.4f}")
        
        # --- STEP 1: Define target paths ---
        print2log('\n' + '='*70)
        print2log('STEP 1: DEFINING TARGET PATHS')
        print2log('='*70)
        
        all_paths = []
        for source_node in sources:
            print2log(f"Finding paths from {source_node}...")
            paths_gen = nx.shortest_simple_paths(G, source=source_node, target='fibrosis')
            top_k_paths = list(islice(paths_gen, 5))
            for path in top_k_paths:
                path_edges = tuple(nx.utils.pairwise(path))
                if path_edges not in all_paths:
                    all_paths.append(path_edges)
                    path_str = ' → '.join([e[0] for e in path_edges] + [path_edges[-1][1]])
                    print2log(f"  Path {len(all_paths)}: {path_str}")
        
        print2log(f"\n✓ Found {len(all_paths)} unique paths to score")
        
        # --- STEP 2: Score paths across grid ---
        print2log('\n' + '='*70)
        print2log('STEP 2: SCORING PATHS ACROSS STIMULATION GRID')
        print2log('='*70)
        
        checkpoint_dir = os.path.join(out_dir, 'ig_checkpoints')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(out_dir, 'individual_networks')).mkdir(parents=True, exist_ok=True)
        
        x_baseline = torch.zeros((1, len(nodeNames_array)), dtype=torch.double).to(device)
        
        grid_path_scores, grid_metadata = score_paths_across_grid(
            model=model,
            all_paths=all_paths,
            node_idx_map=mod.node_idx_map,
            x_baseline=x_baseline,
            Xin_combined=Xin_combined,
            pheno_coefs=torch.tensor(coefs, dtype=torch.double),
            outName=outName,
            num_steps_ig=int(num_steps_ig),
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=f'model{model_no}',
            resume_from_checkpoint=True
        )
        
        # --- STEP 2B: Analyze competition ---
        print2log('\n' + '='*70)
        print2log('STEP 2B: ANALYZING COMPETITION PATTERNS')
        print2log('='*70)
        
        competitive_paths, cooperative_paths, path_importance_grids = analyze_competition_grid(
            grid_path_scores=grid_path_scores,
            grid_metadata=grid_metadata,
            all_paths=all_paths,
            stimulation_strength=stimulation_strength
        )
        
        # Print results
        print_competition_results(competitive_paths, cooperative_paths, print2log, top_n=10)
        
        # Visualize
        visualize_competition_heatmaps(
            competitive_paths=competitive_paths,
            stimulation_strength=stimulation_strength,
            out_dir=out_dir,
            model_no=model_no,
            top_n=10
        )
        
        # Save results
        save_path_scores(
            grid_path_scores=grid_path_scores,
            grid_metadata=grid_metadata,
            competitive_paths=competitive_paths,
            cooperative_paths=cooperative_paths,
            path_importance_grids=path_importance_grids,
            out_dir=out_dir,
            model_no=model_no
        )
        
        # --- STEP 3: Extract top path subgraph ---
        top_ifnar1_path, top_tgfbr1_path, result_graph, subgraph_df = extract_and_save_top_path_subgraph(
            grid_path_scores=grid_path_scores,
            grid_metadata=grid_metadata,
            all_paths=all_paths,
            A=A,
            nodeNames=nodeNames,
            outName=outName,
            out_dir=out_dir,
            model_no=model_no,
            edge_models=edge_models,
            node_models=node_models,
            edge_weight_sum=edge_weight_sum,
            edge_weight_abs_sum=edge_weight_abs_sum,
            to_symbol=to_symbol,
            print_func=print2log,
            sources=sources
        )
        
        # Update path aggregators (using max stimulation point)
        max_stim_idx = len(grid_path_scores) - 1
        for path_edges in all_paths:
            score = grid_path_scores[max_stim_idx].get(path_edges, 0)
            path_ig_sum[path_edges] += score
            path_ig_count[path_edges] += 1
        
        print2log(f"\n{'='*70}")
        print2log(f"✓ COMPLETED MODEL {model_no+1}/{number_DTLEMBAS_models}")
        print2log(f"{'='*70}\n")
    
    print2log(f"Finished timepoint: {timepoint}")

# ============================================================================
# FINAL: AGGREGATE ACROSS ALL MODELS
# ============================================================================

print2log('\n' + '='*70)
print2log('FINAL: AGGREGATING RESULTS ACROSS ALL MODELS')
print2log('='*70)

# Aggregate subgraph edges
aggregated_edges, df_aggregated = aggregate_subgraph_edges(
    edge_models=edge_models,
    node_models=node_models,
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

print2log("\n" + "="*70)
print2log("ANALYSIS COMPLETE!")
print2log("="*70)