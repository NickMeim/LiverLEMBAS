"""
Leave-One-Drug-Out (LODO) Training and Validation Script

This script implements the LODO validation strategy for DT-LEMBAS models:
1. Removes one safe drug from training data
2. Trains a model without that drug
3. Validates by artificially stimulating the drug's targets
4. Saves predictions for correlation analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import logging
import time
from pathlib import Path

# Setup LEMBAS path - adjust as needed for your environment
gpu_lembas_path = '/home/meimetis/LEMBAS-main'
sys.path.insert(1, os.path.join(gpu_lembas_path))

from LEMBAS.model.bionetwork import format_network, SignalingModel
import LEMBAS.utilities as utils
from LEMBAS import io

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info


def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    idx = 0
    while idx < N:
        end = idx + batchSize
        batch_indices = order[idx:end]
        # merge a singleton batch with the previous batch
        if len(batch_indices) == 1 and len(outList) > 0:
            outList[-1] = np.concatenate((outList[-1], batch_indices))
            break
        # if the dataset itself has only one sample, duplicate the index
        if len(batch_indices) == 1 and len(outList) == 0:
            outList.append(np.concatenate((batch_indices, batch_indices)))
            break
        outList.append(batch_indices)
        idx = end
    return outList


def pearson_r(y_true, y_pred):
    """Compute Pearson correlation for each column."""
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)
    my = torch.mean(y, dim=0)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym, dim=0)
    x_square_sum = torch.sum(xm * xm, dim=0)
    y_square_sum = torch.sum(ym * ym, dim=0)
    r_den = torch.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return r


class DrugstoSignal(nn.Module):
    """Drug-to-Signal neural network layer."""

    def __init__(self, mask, size1, size2, drugSim, momentum=0.6, drop=0.1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.size_in = size1  # drugs number
        self.size_out = size2  # targets/signal size
        self.mask = mask.double().to(device)
        self.drugSim = drugSim.to(device)
        self.leak = 0.01

        # Initialize weights
        weights = torch.Tensor(size2, size1).double()
        nn.init.uniform_(weights, -0.3, 0.01)
        weights = torch.mul(weights.to(device), self.mask)

        ecfp4_weights = torch.Tensor(size1, size1).double()
        nn.init.uniform_(ecfp4_weights, 0., 1.)

        self.Wsim = nn.Parameter(ecfp4_weights, requires_grad=True)
        self.bn = nn.BatchNorm1d(num_features=size1, momentum=momentum, dtype=torch.double)
        self.dropout = nn.Dropout(drop)
        self.A = nn.Parameter(weights, requires_grad=True)

    def forward(self, x):
        W = torch.mul(self.drugSim, self.Wsim)
        A = torch.mul(self.A, self.mask)
        xhat = torch.mm(x, W)
        xhat = self.bn(xhat)
        xhat = self.dropout(xhat)
        xhat = xhat.transpose(0, 1)
        xhat = torch.mm(A, xhat)
        output = xhat.transpose(0, 1)
        return output


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(prog='LODO Training and Validation')

    # Data paths
    parser.add_argument('--DrugsIn', type=str,
                        default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
    parser.add_argument('--TargetsIn', type=str,
                        default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
    parser.add_argument('--TFsOut', type=str,
                        default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
    parser.add_argument('--ChemicalSims', type=str,
                        default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
    parser.add_argument('--PKN', type=str,
                        default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
    parser.add_argument('--PknAnnotation', type=str,
                        default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
    parser.add_argument('--infoPath', type=str,
                        default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')

    # LODO-specific files
    parser.add_argument('--safe_drugs_file', type=str,
                        default='../preprocessing/preprocessed_data/safe_drugs_to_remove_LODO.csv')
    parser.add_argument('--signs_file', type=str,
                        default='../preprocessing/preprocessed_data/consensus_signs_majority_vote.csv')

    # Output and model settings
    parser.add_argument('--res_dir', type=str, default='../results/LODO/')
    parser.add_argument('--outPattern', type=str, default='lodo_hepatocytes')
    parser.add_argument('--drug_index', type=int, required=True,
                        help='Index of drug to hold out (0-based)')
    parser.add_argument('--timepoint', type=str, default='8 hour')

    # Training parameters
    parser.add_argument('--batchSize', type=int, default=35)
    parser.add_argument('--maxIter', type=int, default=1000)
    parser.add_argument('--MoAFactor', type=float, default=0.1)
    parser.add_argument('--spectralFactor', type=float, default=1e-3)
    parser.add_argument('--noiseLevel', type=float, default=10)
    parser.add_argument('--L2', type=float, default=1e-6)
    parser.add_argument('--lamda', type=float, default=1e-9)

    # Model architecture parameters
    parser.add_argument('--inputAmplitude', type=float, default=3)
    parser.add_argument('--projectionAmplitude', type=float, default=1.2)
    parser.add_argument('--max_LEMBAS_iterations', type=int, default=90)
    parser.add_argument('--LEMBAS_leak', type=float, default=0.01)
    parser.add_argument('--LEMBAS_tolerance', type=float, default=1e-5)
    parser.add_argument('--drug_bn', type=float, default=0.2)
    parser.add_argument('--drug_drop', type=float, default=0.1)

    # Validation parameters
    parser.add_argument('--stimulation_strength', type=float, default=100,
                        help='Strength of artificial stimulation (+/- based on sign)')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def load_data(args):
    """Load and preprocess all required data."""
    print2log('Loading input data...')

    # Prior knowledge signaling network
    net = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[:, ['source', 'target', 'stimulation', 'inhibition']]

    # Drug inputs and TF outputs
    drugInput = pd.read_csv(args.DrugsIn, sep='\t', index_col=0)
    TFOutput = pd.read_csv(args.TFsOut, sep='\t', low_memory=False, index_col=0)
    TFOutput = TFOutput.loc[drugInput.index, :]

    # Drug targets
    drugTargets = pd.read_csv(args.TargetsIn, sep='\t', low_memory=False, index_col=0)

    # Filter by timepoint
    inputInfo = pd.read_csv(args.infoPath, sep='\t', low_memory=False, index_col=0)
    inputInfo = inputInfo[inputInfo['time'] == args.timepoint]
    TFOutput = TFOutput.loc[inputInfo.index, :]
    drugInput = drugInput.loc[inputInfo.index, :]

    # Format network
    net = format_network(net, weight_label='mode_of_action',
                        stimulation_label='stimulation',
                        inhibition_label='inhibition')
    nodeNames = np.union1d(net.source.values, net.target.values)

    # Subset to intersecting nodes
    inName = np.intersect1d(nodeNames, drugTargets.columns.values)
    outName = np.intersect1d(nodeNames, TFOutput.columns.values)
    TFOutput = TFOutput.loc[:, outName]
    drugTargets = drugTargets.loc[:, inName]
    drugTargets = drugTargets.loc[drugInput.columns.values, :]

    # Chemical similarities
    drugSim = pd.read_csv(args.ChemicalSims, index_col=0)
    drugSim = drugSim.loc[drugInput.columns.values, drugInput.columns.values]

    # Load safe drugs and signs
    safe_drugs_df = pd.read_csv(args.safe_drugs_file)
    safe_drugs = safe_drugs_df[safe_drugs_df['Status'] == 'Safe']['Drug'].values

    signs_df = pd.read_csv(args.signs_file, index_col=0)

    return {
        'net': net,
        'drugInput': drugInput,
        'TFOutput': TFOutput,
        'drugTargets': drugTargets,
        'drugSim': drugSim,
        'safe_drugs': safe_drugs,
        'signs_df': signs_df,
        'inputInfo': inputInfo
    }


def remove_drug_from_data(data, drug_to_remove, device):
    """Remove specified drug from training data while preserving network structure."""
    drugInput = data['drugInput'].copy()
    drugTargets = data['drugTargets'].copy()
    drugSim = data['drugSim'].copy()

    # Get list of all drugs
    all_drugs = drugInput.columns.tolist()

    if drug_to_remove not in all_drugs:
        raise ValueError(f"Drug {drug_to_remove} not found in drug list")

    # Remove drug from drug-target matrix and similarity matrix
    drugs_to_keep = [d for d in all_drugs if d != drug_to_remove]

    # Filter drugTargets
    drugTargets_filtered = drugTargets.loc[drugs_to_keep, :]

    # Filter drug similarity matrix
    drugSim_filtered = drugSim.loc[drugs_to_keep, drugs_to_keep]

    # Zero out the held-out drug in drugInput (keep columns for consistency)
    # But we need to remove samples that ONLY contain the held-out drug
    drugInput_filtered = drugInput.copy()
    drugInput_filtered[drug_to_remove] = 0

    # Keep samples that have at least one non-zero drug after removal
    samples_with_other_drugs = (drugInput_filtered.drop(columns=[drug_to_remove]).sum(axis=1) > 0)

    # Filter TFOutput accordingly
    TFOutput_filtered = data['TFOutput'].loc[samples_with_other_drugs, :]
    drugInput_filtered = drugInput_filtered.loc[samples_with_other_drugs, :]

    # Create new mask without the removed drug
    mask_dt = torch.tensor(np.array(drugTargets_filtered)).transpose(0, 1)
    drugSim_tensor = torch.tensor(drugSim_filtered.values.copy(), dtype=torch.double)

    return {
        'drugInput': drugInput_filtered,
        'TFOutput': TFOutput_filtered,
        'drugTargets': drugTargets_filtered,
        'mask_dt': mask_dt,
        'drugSim': drugSim_tensor,
        'drugs_to_keep': drugs_to_keep
    }


def train_model(args, data, filtered_data, device):
    """Train the DT-LEMBAS model without the held-out drug."""
    print2log('Initializing model...')

    # Set seed
    if args.seed:
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        utils.set_seeds(seed=args.seed)

    # Bionetwork parameters
    bionet_params = {
        'target_steps': 100,
        'max_steps': args.max_LEMBAS_iterations,
        'exp_factor': 50,
        'tolerance': args.LEMBAS_tolerance,
        'leak': args.LEMBAS_leak
    }

    # Initialize SignalingModel
    mod = SignalingModel(
        net=data['net'],
        X_in=filtered_data['drugTargets'],
        y_out=filtered_data['TFOutput'],
        projection_amplitude_in=args.inputAmplitude,
        projection_amplitude_out=args.projectionAmplitude,
        weight_label='mode_of_action',
        source_label='source',
        target_label='target',
        bionet_params=bionet_params,
        dtype=torch.double,
        device=device,
        seed=args.seed
    )

    # Initialize DrugLayer
    drugLayer = DrugstoSignal(
        filtered_data['mask_dt'],
        filtered_data['mask_dt'].shape[1],
        filtered_data['mask_dt'].shape[0],
        filtered_data['drugSim'],
        momentum=args.drug_bn,
        drop=args.drug_drop,
        device=device
    )
    drugLayer = drugLayer.to(device)

    # Prepare training data
    X = torch.tensor(filtered_data['drugInput'][filtered_data['drugs_to_keep']].values.copy(),
                     dtype=torch.double).to(device)
    Y = torch.tensor(filtered_data['TFOutput'].values, dtype=torch.double).to(device)

    # Setup optimizer
    mod.input_layer.weights.requires_grad = False
    mod.signaling_network.prescale_weights(target_radius=0.8)

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(
        list(mod.parameters()) + list(drugLayer.parameters()),
        lr=1, weight_decay=0
    )

    # Hyperparameters
    hyper_params = {
        'max_iter': args.maxIter,
        'learning_rate': 2e-3,
        'batch_size': args.batchSize,
        'noise_level': args.noiseLevel,
        'gradient_noise_level': 1e-9,
        'param_lambda_L2': args.L2,
        'moa_lambda_L1': args.MoAFactor,
        'ligand_lambda_L2': 1e-5,
        'uniform_lambda_L2': 1e-5,
        'uniform_max': 1/args.projectionAmplitude,
        'spectral_loss_factor': args.spectralFactor,
        'n_probes_spectral': 5,
        'power_steps_spectral': 5,
        'subset_n_spectral': 10
    }

    stats = utils.initialize_progress(hyper_params['max_iter'])
    mLoss = criterion(torch.mean(Y, dim=0) * torch.ones(Y.shape, device=Y.device), Y)
    N = X.shape[0]

    print2log(f'Begin training for {N} samples...')
    print2log(f'Baseline loss: {mLoss.item():.4f}')

    start_time = time.time()
    reset_epoch = 200
    reset_state = optimizer.state.copy()

    for e in range(hyper_params['max_iter']):
        # Learning rate schedule
        cur_lr = utils.get_lr(e, hyper_params['max_iter'],
                             max_height=hyper_params['learning_rate'],
                             start_height=hyper_params['learning_rate']/10,
                             end_height=1e-4, peak=500)
        optimizer.param_groups[0]['lr'] = cur_lr

        cur_loss = []
        cur_eig = []

        trainloader = getSamples(N, args.batchSize)
        for dataIndex in trainloader:
            mod.train()
            drugLayer.train()
            optimizer.zero_grad()

            dataIn = X[dataIndex, :].view(len(dataIndex), X.shape[1])
            dataOut = Y[dataIndex, :].view(len(dataIndex), Y.shape[1])

            Xin = drugLayer(dataIn)
            X_full = mod.input_layer(Xin)
            network_noise = torch.randn(X_full.shape, device=X_full.device)
            X_full = X_full + hyper_params['noise_level'] * cur_lr * network_noise
            YhatFull = mod.signaling_network(X_full)
            Yhat = mod.output_layer(YhatFull)

            # Losses
            fit_loss = criterion(dataOut, Yhat)
            sign_reg = mod.signaling_network.sign_regularization(lambda_L1=hyper_params['moa_lambda_L1'])
            ligand_reg = mod.ligand_regularization(lambda_L2=hyper_params['ligand_lambda_L2'])
            stability_loss, spectral_radius = mod.signaling_network.get_SS_loss(
                Y_full=YhatFull.detach(),
                spectral_loss_factor=hyper_params['spectral_loss_factor'],
                subset_n=hyper_params['subset_n_spectral'],
                n_probes=hyper_params['n_probes_spectral'],
                power_steps=hyper_params['power_steps_spectral']
            )
            uniform_reg = mod.uniform_regularization(
                lambda_L2=hyper_params['uniform_lambda_L2']*cur_lr,
                Y_full=YhatFull,
                target_min=0,
                target_max=hyper_params['uniform_max']
            )
            param_reg = mod.L2_reg(hyper_params['param_lambda_L2'])
            DTweightLoss = 1e-6 * torch.sum(torch.square(drugLayer.A))

            total_loss = (fit_loss + sign_reg + ligand_reg + param_reg +
                         stability_loss + uniform_reg + DTweightLoss)

            total_loss.backward()
            mod.add_gradient_noise(noise_level=hyper_params['gradient_noise_level'])
            optimizer.step()

            cur_eig.append(spectral_radius)
            cur_loss.append(fit_loss.item())

        stats = utils.update_progress(stats, iter=e, loss=cur_loss, eig=cur_eig,
                                      learning_rate=cur_lr,
                                      n_sign_mismatches=mod.signaling_network.count_sign_mismatch())

        if e % 100 == 0:
            utils.print_stats(stats, iter=e)

        if np.logical_and(e % reset_epoch == 0, e > 0):
            optimizer.state = reset_state.copy()

    mins, secs = divmod(time.time() - start_time, 60)
    print2log(f"Training completed in: {mins:.0f} min {secs:.2f} sec")

    return mod, drugLayer, stats


def validate_with_artificial_stimulation(args, data, mod,drugLayer, drug_held_out, device):
    """Validate by artificially stimulating the held-out drug's targets."""
    print2log(f'Validating with artificial stimulation for drug: {drug_held_out[:50]}...')

    signs_df = data['signs_df']
    drugTargets = data['drugTargets']
    TFOutput = data['TFOutput']
    drugInput = data['drugInput']

    # Get targets for the held-out drug
    if drug_held_out in drugTargets.index:
        drug_target_mask = drugTargets.loc[drug_held_out]
        drug_targets = drug_target_mask[drug_target_mask > 0].index.tolist()
    else:
        print2log(f"Warning: Drug not found in drugTargets")
        drug_targets = []

    # Get signs for this drug
    if drug_held_out in signs_df.index:
        drug_signs = signs_df.loc[drug_held_out]
    else:
        print2log(f"Warning: Drug not found in signs file")
        drug_signs = pd.Series()

    # Get node indices from model
    all_nodes = np.array(list(mod.node_idx_map.keys()))
    num_nodes = len(all_nodes)

    # Create stimulation vector
    stimulation = torch.zeros(1, num_nodes, dtype=torch.double).to(device)

    for target in drug_targets:
        if target in mod.node_idx_map:
            target_idx = mod.node_idx_map[target]

            # Get sign for this target
            if target in drug_signs.index:
                sign = drug_signs[target]
                if sign == 0:
                    # Unknown sign - skip this target
                    continue
                # Apply stimulation based on sign
                # Positive sign = activation, Negative sign = inhibition
                stimulation[0, target_idx] = args.stimulation_strength * sign

    print2log(f"Stimulating {len(drug_targets)} targets with strength {args.stimulation_strength}")

    # Get samples containing the held-out drug
    samples_with_drug = drugInput[drug_held_out] > 0

    results = []
    mod.eval()
    drugLayer.eval()

    with torch.no_grad():
        for idx in drugInput[samples_with_drug].index:
            # Get the dose for this sample
            dose = drugInput.loc[idx, drug_held_out]

            # True TF activities
            true_tf = TFOutput.loc[idx, :].values

            # Predict with artificial stimulation
            Baseline = drugLayer(torch.zeros(1, drugInput.shape[1], device=device))
            Baseline_full = mod.input_layer(Baseline)
            YhatFull = mod.signaling_network(stimulation + Baseline_full)
            Yhat = mod.output_layer(YhatFull)
            predicted_tf = Yhat.detach().cpu().numpy().flatten()

            for i, tf in enumerate(TFOutput.columns):
                results.append({
                    'timepoint': args.timepoint,
                    'drug': drug_held_out,
                    'dose': dose,
                    'TF': tf,
                    'true_value': true_tf[i],
                    'predicted_value': predicted_tf[i]
                })

    return pd.DataFrame(results)


def main():
    args = parse_arguments()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print2log(f'Running on {device}')

    # Create output directory
    Path(args.res_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_data(args)

    # Get drug to hold out
    safe_drugs = data['safe_drugs']
    if args.drug_index >= len(safe_drugs):
        raise ValueError(f"Drug index {args.drug_index} out of range. Only {len(safe_drugs)} safe drugs available.")

    drug_held_out = safe_drugs[args.drug_index]
    print2log(f'Holding out drug {args.drug_index}: {drug_held_out[:50]}...')

    # Remove drug from training data
    filtered_data = remove_drug_from_data(data, drug_held_out, device)
    print2log(f'Training samples: {len(filtered_data["TFOutput"])}')

    # Train model
    mod, drugLayer, stats = train_model(args, data, filtered_data, device)

    # Save model
    model_path = os.path.join(args.res_dir, args.timepoint, f'{args.outPattern}_drug{args.drug_index}')
    torch.save(obj=mod.state_dict(), f=model_path + '_signal_state_dict.pth')
    torch.save(obj=drugLayer.state_dict(), f=model_path + '_drugLayer_state_dict.pth')
    io.write_pickled_object(stats, model_path + '_training_stats.pickle')
    print2log(f'Model saved to {model_path}')

    # Validate
    results_df = validate_with_artificial_stimulation(args, data, mod, filtered_data, drug_held_out, device)

    # Save results
    output_file = os.path.join(args.res_dir,args.timepoint, f'drug{args.drug_index}_lodo_prediction.csv')
    results_df.to_csv(output_file, index=False)
    print2log(f'Results saved to {output_file}')

    print2log('LODO validation completed!')


if __name__ == '__main__':
    main()
