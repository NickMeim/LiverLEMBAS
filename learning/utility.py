import torch
import torch.nn as nn
import numpy as np
import time

import os

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
import torch
import plotnine as p9
from sklearn.model_selection import KFold
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info


# In[45]:




# In[46]:


import sys 
gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
from LEMBAS.model.bionetwork import format_network, SignalingModel
from LEMBAS.model.train import train_signaling_model
import LEMBAS.utilities as utils
from LEMBAS import plotting, io

def getSamples(N, batchSize):
    order = np.random.permutation(N)
    outList = []
    while len(order)>0:
        outList.append(order[0:batchSize])
        order = order[batchSize:]
    return outList


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

def non_zero_check(X_train):
    # Check if all features have non-zero values in at least one sample
    return np.all(np.any(X_train != 0, axis=0))

# Create a custom KFold splitter
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

class DrugstoSignal(nn.Module):
    def __init__(self,mask, size1,size2,drugSim,momentum=0.6,drop=0.1,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):

        super().__init__()

        self.size_in = size1 ### drugs number
        self.size_out = size2 ### targets/signal size
        self.mask = mask.double().to(device)
        self.drugSim = drugSim.to(device)
        self.leak = 0.01

        # initialize weights and biases
        weights = torch.Tensor(size2,size1).double()
        nn.init.uniform_(weights, -0.3, 0.01) # weight init
        weights = torch.mul(weights.to(device),self.mask)
        ecfp4_weights = torch.Tensor(size1,size1).double()
        nn.init.uniform_(ecfp4_weights, 0., 1.)
        self.Wsim = nn.Parameter(ecfp4_weights, requires_grad=True)
        self.bn = nn.BatchNorm1d(num_features=size1, momentum=momentum, dtype=torch.double)
        self.dropout = nn.Dropout(drop)

        self.A = nn.Parameter(weights, requires_grad = True)

    def forward(self, x):
        W = torch.mul(self.drugSim, self.Wsim)
        A = torch.mul(self.A,self.mask)
        xhat = torch.mm(x,W)
        xhat = self.bn(xhat)
        xhat = self.dropout(xhat)
        xhat=xhat.transpose(0, 1)
        xhat = torch.mm(A, xhat)
        output = xhat.transpose(0, 1)
        return output

def perform_cross_validation(valid_splits, X, Y,net,drugTargets,TFOutput,
                             projection_amplitude_in, projection_amplitude_out, weight_label, source_label, target_label, drugSim, bionet_params,
                            hyper_params, device, data_path, outPattern, target_spectral_radius,
                            mask_dt, args, res_dir, batchSize,
                            seed,shuffling=False):
    fold_results = pd.DataFrame({})

    for fold_idx, (train_idx, test_idx) in enumerate(valid_splits):
        print(f"Starting fold {fold_idx + 1}/{len(valid_splits)}...")

        mod = SignalingModel(net = net,
                     X_in = drugTargets,
                     y_out = TFOutput.iloc[train_idx,:], 
                     projection_amplitude_in = projection_amplitude_in, projection_amplitude_out = projection_amplitude_out,
                     weight_label = weight_label, source_label = source_label, target_label = target_label,
                     bionet_params = bionet_params, 
                     dtype = torch.double, device = device, seed = seed)
        drugLayer = DrugstoSignal(mask_dt, 
                          mask_dt.shape[1],mask_dt.shape[0],
                          drugSim,momentum=args.drug_bn,drop=args.drug_drop,
                          device = device)
        drugLayer = drugLayer.to(device)

        # Split the data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Convert data to torch tensors if not already
        X_train = torch.tensor(X_train, dtype=torch.double).to(device)
        Y_train = torch.tensor(Y_train, dtype=torch.double).to(device)
        X_test = torch.tensor(X_test, dtype=torch.double).to(device)
        Y_test = torch.tensor(Y_test, dtype=torch.double).to(device)

        if shuffling:
            X_train = X_train[:,torch.randperm(X_train.shape[1])]

        # Initialize stats
        stats = utils.initialize_progress(hyper_params['max_iter'])
        mod.input_layer.weights.requires_grad = False  # freeze scaling factors
        mod.signaling_network.prescale_weights(target_radius = target_spectral_radius)

        # loss and optimizer
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(list(mod.parameters())+list(drugLayer.parameters()), 
                                    lr=1,weight_decay=0)
        mLoss = criterion(torch.mean(Y_train, dim=0) * torch.ones(Y_train.shape, device = Y_train.device), Y_train)
        N = X_train.shape[0]

        print2log(mLoss.item())
        start_time = time.time()
        reset_epoch = 200
        reset_state = optimizer.state.copy()
        for e in range(hyper_params['max_iter']):
            cur_lr = utils.get_lr(e, hyper_params['max_iter'], max_height=hyper_params['learning_rate'],
                                  start_height=hyper_params['learning_rate'] / 10, end_height=1e-4, peak=500)
            optimizer.param_groups[0]['lr'] = cur_lr

            trainloader = getSamples(N , batchSize)
            cur_loss = []
            cur_eig = []

            for dataIndex in trainloader:
                mod.train()
                drugLayer.train()

                optimizer.zero_grad()

                dataIn = X_train[dataIndex, :].view(len(dataIndex), X_train.shape[1])
                dataOut = Y_train[dataIndex, :].view(len(dataIndex), Y_train.shape[1])

                Xin = drugLayer(dataIn)
                X_full = mod.input_layer(Xin)
                network_noise = torch.randn(X_full.shape, device=X_full.device)
                X_full = X_full + hyper_params['noise_level'] * cur_lr * network_noise
                YhatFull = mod.signaling_network(X_full)
                Yhat = mod.output_layer(YhatFull)

                fit_loss = criterion(dataOut, Yhat)

                sign_reg = mod.signaling_network.sign_regularization(lambda_L1=hyper_params['moa_lambda_L1'])
                ligand_reg = mod.ligand_regularization(lambda_L2=hyper_params['ligand_lambda_L2'])
                stability_loss, spectral_radius = mod.signaling_network.get_SS_loss(Y_full=YhatFull.detach(),
                                                                                   spectral_loss_factor=hyper_params['spectral_loss_factor'],
                                                                                   subset_n=hyper_params['subset_n_spectral'],
                                                                                   n_probes=hyper_params['n_probes_spectral'],
                                                                                   power_steps=hyper_params['power_steps_spectral'])
                uniform_reg = mod.uniform_regularization(lambda_L2=hyper_params['uniform_lambda_L2'] * cur_lr,
                                                         Y_full=YhatFull, target_min=0,
                                                         target_max=hyper_params['uniform_max'])
                param_reg = mod.L2_reg(hyper_params['param_lambda_L2'])
                #DTregularization = hyper_params["lamda"] * torch.sum(torch.square(drugLayer.Wsim-torch.eye(drugLayer.Wsim.shape[0],device=device)))
                DTweightLoss = 1e-6 * torch.sum(torch.square(drugLayer.A)) #+ 1e-8*(torch.sum(1 / (torch.square(drugLayer.A) + 0.5)))

                total_loss = fit_loss + sign_reg + ligand_reg + param_reg + stability_loss + uniform_reg + DTweightLoss # + DTregularization

                total_loss.backward()
                mod.add_gradient_noise(noise_level=hyper_params['gradient_noise_level'])
                optimizer.step()

                cur_eig.append(spectral_radius)
                cur_loss.append(fit_loss.item())

            stats = utils.update_progress(stats, iter=e, loss=cur_loss, eig=cur_eig, learning_rate=cur_lr,
                                          n_sign_mismatches=mod.signaling_network.count_sign_mismatch())

            if e % 100 == 0 or (e+1)==hyper_params['max_iter']:
                utils.print_stats(stats, iter = e)
                # store results
                io.write_pickled_object(stats, os.path.join(data_path,outPattern + '_training_stats_fold%s.pickle'%fold_idx))
                
            if np.logical_and(e % reset_epoch == 0, e>0):
                optimizer.state = reset_state.copy()
        

        io.write_pickled_object(stats, os.path.join(data_path,outPattern + '_training_stats_fold%s.pickle'%fold_idx))
        mins, secs = divmod(time.time() - start_time, 60)
        print2log("Training ran in: {:.0f} min {:.2f} sec".format(mins, secs))

        # Evaluate on train and test set
        mod.eval()
        drugLayer.eval()

        Xin_train = drugLayer(X_train)
        Y_hat_train, _ = mod(Xin_train)
        Xin_test = drugLayer(X_test)
        Y_hat_test, _ = mod(Xin_test)

        # Calculate Pearson correlation
        r_train = pearson_r(Y_train, Y_hat_train).detach().cpu().numpy()
        r_test = pearson_r(Y_test, Y_hat_test).detach().cpu().numpy()

        # Save fold results
        fold_results= pd.concat([fold_results,pd.DataFrame({
            'fold': fold_idx,
            'TF':TFOutput.columns.values,
            'r_train': r_train,
            'r_test': r_test,
        })], axis=0).reset_index(drop=True)
        fold_results.to_csv(res_dir + outPattern + '_fold_results.csv')

        # Save model and results for this fold
        torch.save(obj=mod.state_dict(), f=os.path.join(data_path, outPattern + f'signal_state_dict_fold{fold_idx}.pth'))
        torch.save(obj=drugLayer.state_dict(),
                   f=os.path.join(data_path, outPattern + f'drugLayer_state_dict_fold{fold_idx}.pth'))
        
        loss_smooth = utils.get_moving_average(values = stats['loss_mean'], n_steps = 5)
        loss_sigma_smooth = utils.get_moving_average(values = stats['loss_sigma'], n_steps = 10)
        epochs = np.array(range(stats['loss_mean'].shape[0]))

        p1A = plotting.shade_plot(X = epochs, Y = loss_smooth, sigma = loss_sigma_smooth, x_label = 'Epoch', y_label = 'Loss')
        p1A += p9.scale_y_log10()
        p1A += p9.geom_hline(yintercept = mLoss.item(), linetype="dashed", color="black")
        p1A.save(filename=res_dir+outPattern+"_training_loss_fold%s.png"%fold_idx, dpi=300) 

    return fold_results
