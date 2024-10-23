import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from utility_functions import *
from sklearn.model_selection import KFold
import pyreadr
import pickle
import argparse


### Initialize the parsed arguments
parser = argparse.ArgumentParser(description='Run different ML models')
parser.add_argument('--TF_activity_path', action='store',help='Path of TF activity data in a specific cell line in TSV file',default = '../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--Gene_expression_path', action='store', help='Path of Gene expression data in a specific cell line in TSV file',default = '../preprocessing/preprocessed_data/GeneExpressionHepatocytes.csv')
parser.add_argument('--num_folds', action='store', type=int,help='number of folds for CV',default=10)
parser.add_argument('--res_dir', action='store', help='folder to save results of the analysis',default='../results/')
parser.add_argument('--device', action='store', help='Device to use with Pytorch (cuda or cpu with default=cpu)',default='cuda')
parser.add_argument('--shared_hidden_layers', metavar='N', type=int, nargs='*', action='store', help='Sharred hidden layers of the multi-task model',default=[128])
parser.add_argument('--task_specific_hidden_layers', metavar='N', type=int, nargs='*', action='store', help='Task-specific hidden layers of the multi-task model',default=[256])
parser.add_argument('--batch_normalization_momentum', action='store',type=float, default=0.2)
parser.add_argument('--drop_out_rate', action='store',type=float, default=0.25)
parser.add_argument('--bias_flag', action='store',type=bool, default= True)
parser.add_argument('--input_reg_model', action='store', help='name of regulatory model', default="ANN_Hepatocytes_reg_model_humans.pt")
parser.add_argument('--epochs', action='store',type=int, default=400)
parser.add_argument('--learning_rate', action='store',type=float, default=0.001)
parser.add_argument('--lamda_reg', action='store',type=float, default=1e-1)
parser.add_argument('--batch_size', action='store',type=int, help='bacth size for training',default=15)
parser.add_argument('--genes_annotation',action='store',type=str,help='path to file containing the gens to show in the scatterplot',default='../preprocessing/preprocessed_data/Hepatocytes_human_important_genes.rds')
parser.add_argument('--infoPath', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--pretrained_reg_model', action='store',type = str, default='../data/archs4_regulatory_model.pt')
parser.add_argument('--trained_gene_names', action='store',type = str, default='../data/trained_archs4_genes.npy')
parser.add_argument('--trained_TF_names', action='store',type = str, default='../data/trained_archs4_TFs.npy')
args = parser.parse_args()
infoPath = args.infoPath
num_folds = int(args.num_folds)
Gene_expression_path = args.Gene_expression_path
TF_activity_path = args.TF_activity_path
res_dir = args.res_dir
input_reg_model= args.input_reg_model
epochs = int(args.epochs)
learning_rate = args.learning_rate
lamda_reg= args.lamda_reg
batch_size = int(args.batch_size)
task_specific_hidden_layers = args.task_specific_hidden_layers
shared_hidden_layers = args.shared_hidden_layers
drop_out_rate = args.drop_out_rate
batch_normalization_momentum = args.batch_normalization_momentum
bias_flag = args.bias_flag
genes_annotation = args.genes_annotation
trained_TF_names = args.trained_TF_names
device = args.device

Path(res_dir).mkdir(parents=True, exist_ok=True)

# Convert the parsed arguments into a dictionary
args_dict = vars(args)
# Save the dictionary to a pickle file
with open(res_dir + 'regulatory_model_parameters.pkl', 'wb') as f:
    pickle.dump(args_dict, f)


print('Loading data...')                   
trained_genes = np.load(args.trained_gene_names, allow_pickle=True)
trained_TFs = np.load(args.trained_TF_names, allow_pickle=True)
### Load the data
inputInfo = pd.read_csv(infoPath, sep='\t', low_memory=False, index_col=0)
inputInfo = inputInfo[inputInfo['time']=='8 hour']
# W_total = pd.read_csv(res_dir+'W_total.csv')
Y = pd.read_csv(Gene_expression_path,index_col=0)
Y = Y.loc[inputInfo.index,:]
# Y = Y.iloc[:,1:5]
X = pd.read_csv(TF_activity_path,sep='\t',index_col=0)
transcription_factors = X.columns.values
X = X.loc[Y.index,:]
genes = Y.columns
genes2show = pyreadr.read_r(genes_annotation)
genes2show = genes2show[None].values
genes2show = genes2show.squeeze()

gene_order = np.where(np.isin(genes, trained_genes))[0]
tfs_order = np.where(np.isin(transcription_factors, trained_TFs))[0]

### Start building multitask model
print('Initializing regulatory model and training parameters...')
X = torch.tensor(X.values, dtype=torch.double).to(device)
Y = torch.tensor(Y.values, dtype=torch.double).to(device)

# ### Load large pre-trained model
# pretrained_model = SimpleANN(input_dim = trained_TFs.shape[0], 
#                       hidden_layers = shared_hidden_layers+task_specific_hidden_layers,
#                       output_dim = trained_genes.shape[0],
#                       momentum=batch_normalization_momentum, 
#                       drop_rate=drop_out_rate, 
#                       activation=nn.LeakyReLU(0.01), 
#                       bias=bias_flag)
# pretrained_model.load_state_dict(torch.load(args.pretrained_reg_model, map_location='cpu'))   

### Perform CV
print('Starting {:.0f}-fold cross-validation training...'.format(num_folds))
kf = KFold(n_splits=num_folds, shuffle=True,random_state=42)
all_mu_losses = []
all_mu_r = []
all_std_losses = []
all_std_r = []
all_r = []
all_r_training = []
all_r_shuffled_genes = []
all_r_shuffled_mask = []
cv_res = pd.DataFrame({})

# Loop over each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/{num_folds}")
    
    # Split the data
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Initialize model
    model = SimpleANN(input_dim = X.shape[1], 
                      hidden_layers = shared_hidden_layers+task_specific_hidden_layers,
                      output_dim = Y.shape[1],
                      momentum=batch_normalization_momentum, 
                      drop_rate=drop_out_rate, 
                      activation=nn.LeakyReLU(0.01), 
                      bias=bias_flag).to(device)
    # ## Initialize with pre-trained weights for genes we have
    # model.output_layer.weight.data[gene_order,:] = pretrained_model.output_layer.weight.data.to(device)
    # if model.output_layer.bias is not None:
    #     model.output_layer.bias.data[gene_order] = pretrained_model.output_layer.bias.data.to(device)
    # for ll in range(len(model.layers)):
    #     if isinstance(model.layers[ll], nn.Linear):
    #         if ll==0:
    #             model.layers[ll].weight.data = pretrained_model.layers[ll].weight.data[:,tfs_order].to(device)
    #         else:
    #             model.layers[ll].weight.data = pretrained_model.layers[ll].weight.data.to(device)
    #         if model.layers[ll].bias is not None:
    #             model.layers[ll].bias.data = pretrained_model.layers[ll].bias.data.to(device)
    #     if isinstance(model.layers[ll], nn.BatchNorm1d):
    #         model.layers[ll].weight.data = pretrained_model.layers[ll].weight.data.to(device)
    #         if model.layers[ll].bias is not None:
    #             model.layers[ll].bias.data = pretrained_model.layers[ll].bias.data.to(device)
    # Train the model
    model, mu_losses, mu_r, std_losses, std_r, r_train = train_model(X_train, Y_train,model, learning_rate, epochs, lamda_reg, batch_size)
    # Store results for this fold
    all_mu_losses.append(np.array(mu_losses))
    all_mu_r.append(np.array(mu_r))
    all_std_losses.append(np.array(std_losses))
    all_std_r.append(np.array(std_r))
    # all_r_training.append(np.nanmean(r))
    # Evaluation
    model.eval()
    with torch.no_grad():
        yhat = model(X_test)
        r = pearson_r(Y_test, yhat)
    r_test = r.detach().cpu().numpy()
    # Store r for this fold
    all_r.append(np.nanmean(r_test))

    print(f'Fold {fold+1}/{num_folds} r train = {np.nanmean(r_train):.2f},r test = {np.nanmean(r_test):.2f}')

    ## Train randomized model with shuffled genes
    print('Train randomized model with shuffled genes...')
    random_corrs = []
    
    # Initialize random model
    model_shuffled_genes = SimpleANN(input_dim = X.shape[1], 
                                     hidden_layers = shared_hidden_layers+task_specific_hidden_layers,
                                     output_dim = Y.shape[1],
                                     momentum=batch_normalization_momentum, 
                                     drop_rate=drop_out_rate, 
                                     activation=nn.LeakyReLU(0.01), 
                                     bias=bias_flag).to(device)
    # ## Initialize with pre-trained weights for genes we have
    # model_shuffled_genes.output_layer.weight.data[gene_order,:] = pretrained_model.output_layer.weight.data.to(device)
    # if model_shuffled_genes.output_layer.bias is not None:
    #     model_shuffled_genes.output_layer.bias.data[gene_order] = pretrained_model.output_layer.bias.data.to(device)
    # for ll in range(len(model_shuffled_genes.layers)):
    #     if isinstance(model_shuffled_genes.layers[ll], nn.Linear):
    #         if ll==0:
    #             model_shuffled_genes.layers[ll].weight.data = pretrained_model.layers[ll].weight.data[:,tfs_order].to(device)
    #         else:
    #             model_shuffled_genes.layers[ll].weight.data = pretrained_model.layers[ll].weight.data.to(device)
    #         if model_shuffled_genes.layers[ll].bias is not None:
    #             model_shuffled_genes.layers[ll].bias.data = pretrained_model.layers[ll].bias.data.to(device)
    #     if isinstance(model_shuffled_genes.layers[ll], nn.BatchNorm1d):
    #         model_shuffled_genes.layers[ll].weight.data = pretrained_model.layers[ll].weight.data.to(device)
    #         if model_shuffled_genes.layers[ll].bias is not None:
    #             model_shuffled_genes.layers[ll].bias.data = pretrained_model.layers[ll].bias.data.to(device)
    # Train the model
    Y_train_shuffled = Y_train[:,np.random.permutation(Y_train.shape[1])]
    model_shuffled_genes, mu_losses, mu_r, std_losses, std_r, r = train_model(X_train, Y_train_shuffled,model_shuffled_genes, learning_rate, epochs, lamda_reg, batch_size,printing=False)
    # Evaluation
    model_shuffled_genes.eval()
    with torch.no_grad():
        yhat = model_shuffled_genes(X_test)
        r = pearson_r(Y_test, yhat)
    # r = r.detach().cpu().numpy()
    ##random_corrs.append(np.nanmean(r))
    ## Store r for this fold
    ##all_r_shuffled_genes.append(random_corrs)
    # all_r_shuffled_genes.append(np.nanmean(r))
    r_shuffled = r.detach().cpu().numpy()

    df_res = pd.DataFrame({'gene':genes,'train': r_train, 'test': r_test, 'shuffled': r_shuffled})
    df_res['fold'] = fold
    cv_res = pd.concat([cv_res,df_res.reset_index(drop=True)], axis=0)
    cv_res=cv_res.reset_index(drop=True)
    cv_res.to_csv(res_dir+'ann_human_heps_cv_performance.csv')

### Save results
cv_res.to_csv(res_dir+'ann_human_heps_cv_performance.csv')
# Now create a figure with 10 subplots (one for each fold)
fig, axs = plt.subplots(num_folds, 2, figsize=(12, 5 * num_folds))
for fold in range(num_folds):
    # Line plot for each fold
    ax1 = axs[fold, 0]
    ax1.plot(np.arange(1, epochs + 1), all_mu_losses[fold], 'g-', label='loss')
    ax1.fill_between(np.arange(1, epochs + 1), all_mu_losses[fold] - 2 * all_std_losses[fold], 
                     all_mu_losses[fold] + 2 * all_std_losses[fold], color='g', alpha=0.3)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # Secondary axis for r
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, epochs + 1), all_mu_r[fold], 'b-', label='r')
    ax2.fill_between(np.arange(1, epochs + 1), all_mu_r[fold] - 2 * all_std_r[fold], 
                     all_mu_r[fold] + 2 * all_std_r[fold], color='b', alpha=0.3)
    ax2.set_ylabel('r', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Histogram for each fold
    ax3 = axs[fold, 1]
    ax3.hist(all_r[fold], bins=20, color='blue', edgecolor='black')
    ax3.set_xlabel('r')
    ax3.set_ylabel('counts')
    ax3.set_title(f'Fold {fold+1} - CV per gene pearson correlation')

    # Print the mean in the top left corner of the histogram
    ax3.text(0.05, 0.95, f'Mean = {np.nanmean(all_r[fold]):.2f}', 
             transform=ax3.transAxes,
             fontsize=12, 
             verticalalignment='top', 
             horizontalalignment='left')

fig.tight_layout()
# plt.savefig(res_dir + 'multitask_cross_validation_training_convergence.pdf')
plt.savefig(res_dir + 'ann_human_heps_cross_validation_training_convergence.pdf')

cv_res = cv_res.melt(id_vars=['gene','fold'], value_vars=['train','test','shuffled'],var_name='set',value_name='r')
### Get performance scatterplots and results dataframe
results_df, scatterplot_fig = compare_target_vs_reference(df = cv_res,
                                                          genes2show=genes2show,
                                                          target_name = 'shuffled',
                                                          reference_name = 'test',
                                                          gene_col='gene',
                                                          model_col='set',
                                                          r_col='r',
                                                          folds_col='fold',
                                                          alpha = 0.4)
scatterplot_fig.savefig(res_dir + 'per_gene_ann_human_heps_performance_scatterplot.pdf')
# scatterplot_fig.savefig(res_dir + 'per_gene_multitask_performance_scatterplot.pdf')

### Boxplot of aggregated results
combined_p_value, violinplot_fig = plot_combined_p_values(df=cv_res,
                                                       merged_results_df=results_df,
                                                       target_name='shuffled',
                                                       reference_name='test',
                                                       set_col='set',
                                                       r_col='r')
# violinplot_fig.savefig(res_dir + 'multitask_combined_violinplot.pdf')
violinplot_fig.savefig(res_dir + 'ann_human_heps_combined_violinplot.pdf')
print(f'Combined p-value: {combined_p_value:.2e}')

# Train one full model with all the data
print('Training one full model with all the data...')
# Initialize model
# full_model = MultiTaskMLP(
#     input_dim=X.shape[1], 
#     shared_layers=shared_hidden_layers,
#     task_specific_layers=task_specific_hidden_layers, 
#     num_tasks=Y.shape[1],
#     momentum=batch_normalization_momentum, 
#     drop_rate=drop_out_rate, 
#     activation=nn.LeakyReLU(0.1), 
#     bias=bias_flag
# ).to(device)
full_model = SimpleANN(
    input_dim=X.shape[1], 
    hidden_layers=shared_hidden_layers+task_specific_hidden_layers, 
    output_dim=Y.shape[1],
    momentum=batch_normalization_momentum, 
    drop_rate=drop_out_rate, 
    activation=nn.LeakyReLU(0.1), 
    bias=bias_flag
).to(device)

# Train the model
full_model, mu_losses, mu_r, std_losses, std_r, r = train_model(X, Y, full_model, learning_rate, epochs, lamda_reg, batch_size)

# Evaluate the model
full_model.eval()
with torch.no_grad():
    yhat = full_model(X)
    r = pearson_r(Y, yhat)

# Convert results to numpy arrays
r = r.detach().cpu().numpy()
mu_losses = np.array(mu_losses)
mu_r = np.array(mu_r)
std_losses = np.array(std_losses)
std_r = np.array(std_r)

# Save the trained model
torch.save(full_model.state_dict(), res_dir + input_reg_model)

# Create a figure with 2 rows and 1 column of subplots
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(6, 8))

# Plot loss vs epochs on the left y-axis
ax1.plot(np.arange(1, epochs + 1), mu_losses, 'g-', label='Loss')
ax1.fill_between(np.arange(1, epochs + 1), mu_losses - 2 * std_losses, mu_losses + 2 * std_losses, color='g', alpha=0.3)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Create a second set of axes that shares the same x-axis
ax2 = ax1.twinx()

# Plot r vs epochs on the right y-axis
ax2.plot(np.arange(1, epochs + 1), mu_r, 'b-', label='r')
ax2.fill_between(np.arange(1, epochs + 1), mu_r - 2 * std_r, mu_r + 2 * std_r, color='b', alpha=0.3)
ax2.set_ylabel('r', color='b')
ax2.tick_params(axis='y', labelcolor='b')

# Histogram for Pearson correlation values
ax3.hist(r, bins=20, color='blue', edgecolor='black')
ax3.set_xlabel('r')
ax3.set_ylabel('Counts')
ax3.set_title('Per Gene Pearson Correlation')

# Print the mean in the top right corner of the histogram
ax3.text(0.05, 0.95, f'Mean = {np.nanmean(r):.2f}', 
         transform=ax3.transAxes,
         fontsize=12, 
         verticalalignment='top', 
         horizontalalignment='left')

fig.tight_layout()  # To prevent label overlap

# Save the figure as a PDF
# plt.savefig(res_dir + 'multitask_fit_performance.pdf')
plt.savefig(res_dir + 'ann_human_heps_fit_performance.pdf')