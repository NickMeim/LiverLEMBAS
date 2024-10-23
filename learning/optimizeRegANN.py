import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from utility_functions import *
from sklearn.model_selection import KFold,train_test_split
import pyreadr
import pickle
import argparse
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from functools import partial
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



### Initialize the parsed arguments
parser = argparse.ArgumentParser(description='Run different ML models')
parser.add_argument('--TF_activity_path', action='store',help='Path of TF activity data in a specific cell line in TSV file',default = '../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--Gene_expression_path', action='store', help='Path of Gene expression data in a specific cell line in TSV file',default = '../preprocessing/preprocessed_data/GeneExpressionHepatocytes.csv')
parser.add_argument('--num_folds', action='store', type=int,help='number of folds for CV',default=5)
parser.add_argument('--res_dir', action='store', help='folder to save results of the analysis',default='../results/')
parser.add_argument('--device', action='store', help='Device to use with Pytorch (cuda or cpu with default=cpu)',default='cuda')
parser.add_argument('--shared_hidden_layers', metavar='N', type=int, nargs='*', action='store', help='Sharred hidden layers of the multi-task model',default=[256,512])
parser.add_argument('--task_specific_hidden_layers', metavar='N', type=int, nargs='*', action='store', help='Task-specific hidden layers of the multi-task model',default=[1024,2048])
parser.add_argument('--batch_normalization_momentum', action='store',type=float, default=0.6)
parser.add_argument('--drop_out_rate', action='store',type=float, default=0.25)
parser.add_argument('--bias_flag', action='store',type=bool, default= True)
parser.add_argument('--input_reg_model', action='store', help='name of regulatory model', default="ANN_Hepatocytes_reg_model_humans.pt")
parser.add_argument('--epochs', action='store',type=int, default=300)
parser.add_argument('--learning_rate', action='store',type=float, default=0.01)
parser.add_argument('--lamda_reg', action='store',type=float, default=1e-3)
parser.add_argument('--batch_size', action='store',type=int, help='bacth size for training',default=96)
parser.add_argument('--genes_annotation',action='store',type=str,help='path to file containing the gens to show in the scatterplot',default='../preprocessing/preprocessed_data/Hepatocytes_human_important_genes.rds')
parser.add_argument('--infoPath', action='store',type = str, default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
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
device = args.device

Path(res_dir).mkdir(parents=True, exist_ok=True)

# Convert the parsed arguments into a dictionary
args_dict = vars(args)


print('Loading data...')
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

### Start splting
X_tune, X_test, Y_tune, Y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=42)
np.save(res_dir + 'X_test.npy', X_test)
np.save(res_dir + 'Y_test.npy', Y_test)
X_tune = torch.tensor(X_tune, dtype=torch.double).to(device)
Y_tune = torch.tensor(Y_tune, dtype=torch.double).to(device)
X_test = torch.tensor(X_test, dtype=torch.double).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.double).to(device)

# Define the search space for hyperopt
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1)),
    'shared_hidden_layers': hp.choice('shared_hidden_layers', [[256, 512], [512, 1024], [128, 256]]),
    'task_specific_hidden_layers': hp.choice('task_specific_hidden_layers', [[1024, 2048], [512, 1024], [2048, 4096]]),
    'drop_out_rate': hp.uniform('drop_out_rate', 0.1, 0.5),
    'batch_normalization_momentum': hp.uniform('batch_normalization_momentum', 0.1, 0.6),
    'lamda_reg': hp.loguniform('lamda_reg', np.log(1e-5), np.log(1e-1)),
    'leak': hp.uniform('leak', 5e-3, 5e-1),
    'epochs': hp.choice('epochs', [200, 300, 400, 500, 600,700]),
    'batch_size': hp.choice('batch_size', [25, 35, 64,96, 128, 256])
}

def objective(params):
    learning_rate = params['learning_rate']
    hidden_layers_shared = params['shared_hidden_layers']
    hidden_layers_task_specific = params['task_specific_hidden_layers']
    drop_out_rate = params['drop_out_rate']
    batch_normalization_momentum = params['batch_normalization_momentum']
    lamda_reg = params['lamda_reg']
    leak = params['leak']
    epochs = params['epochs']
    batch_size = params['batch_size']
    # Initialize model
    model = SimpleANN(
        input_dim=X.shape[1], 
        hidden_layers=hidden_layers_shared + hidden_layers_task_specific,
        output_dim=Y.shape[1],
        momentum=batch_normalization_momentum, 
        drop_rate=drop_out_rate, 
        activation=torch.nn.LeakyReLU(leak), 
        bias=bias_flag
    ).to(device)

    # Train model and validate using cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True,random_state=42)
    all_r = []
    for fold, (train_index, test_index) in enumerate(kf.split(X_tune)):
        if fold==0:
            X_train, X_val = X_tune[train_index], X_tune[test_index]
            Y_train, Y_val = Y_tune[train_index], Y_tune[test_index]
            # Train the model
            model, _, _, _, _, _ = train_model(X_train, Y_train, model, learning_rate, epochs, lamda_reg, batch_size)
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                yhat = model(X_val)
                r = pearson_r(Y_val, yhat).cpu().numpy()
                all_r.append(np.nanmean(r))

    # Calculate the mean correlation across folds
    mean_r = np.mean(all_r)
    return {'loss': -mean_r, 'status': STATUS_OK}

# Retrieve best hyperparameters
# Run hyperopt optimization
trials = Trials()
best = fmin(
    fn=objective, 
    space=space, 
    algo=tpe.suggest, 
    max_evals=100, 
    trials=trials
)

print("Best hyperparameters:", best)
# Save the best hyperparameters
with open(res_dir + 'best_hyperparameters.pkl', 'wb') as f:
    pickle.dump(best, f)

# Train a final model with the best hyperparameters
best_model = SimpleANN(
    input_dim=X.shape[1],
    hidden_layers=best['shared_hidden_layers'] + best['task_specific_hidden_layers'],
    output_dim=Y.shape[1],
    momentum=best['batch_normalization_momentum'],
    drop_rate=best['drop_out_rate'],
    activation=torch.nn.LeakyReLU(best),
    bias=bias_flag
).to(device)

# Train final model with all data and best hyperparameters
best_model, _, _, _, _, _ = train_model(X_tune, Y_tune, best_model, best['learning_rate'], best['epochs'], best['lamda_reg'], best['batch_size'])

# Save the final model
torch.save(best_model.state_dict(), res_dir + 'best_model.pt')

# Evaluate the model
best_model.eval()
with torch.no_grad():
    yhat = best_model(X_test)
    r = pearson_r(Y_test, yhat)

# Convert results to numpy arrays
r = r.detach().cpu().numpy()
print('Test mean r = {}'.format(np.nanmean(r)))
plt.figure()
plt.hist(r, bins=20)
plt.xlabel('Pearson r')
plt.ylabel('Count')
plt.savefig(res_dir + 'hist_r_test.pdf')
