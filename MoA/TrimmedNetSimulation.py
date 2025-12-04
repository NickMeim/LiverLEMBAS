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
from sklearn.model_selection import GridSearchCV,RepeatedKFold
from collections import defaultdict
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
parser.add_argument('--out_dir', action='store',type = str, default='results/') # 'results_2hr/' for 2 hour and 'results_24hr/' for 24 hour
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
parser.add_argument('--EDGE_FREQ_MIN', action='store',type = float, default=0.5)
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
max_path_length = args.max_path_length
num_steps_ig =  args.num_steps_ig
out_dir = args.out_dir
GRADIENT_THRESHOLD = args.GRADIENT_THRESHOLD
enumerated_pathways_path = args.enumerated_pathways_path


print2log, log_file_path = setup_dual_logging(log_filename=args.log_filename, use_timestamp=True)

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
sym2id = dict(zip(ann['symbol'].astype(str), ann['uniprot'].astype(str)))
def to_symbol(x: str) -> str:
    if x in ['fibrosis','MAS','Fibrosis','NAS','mas','nas']:
        return x
    else:
        # return symbol if known, otherwise keep original Uniprot
        return id2sym.get(str(x), str(x))
def to_uniprot(x: str) -> str:
    if x in ['fibrosis','MAS','Fibrosis','NAS','mas','nas']:
        return x
    else:
        # return symbol if known, otherwise keep original Uniprot
        return sym2id.get(str(x), str(x))
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

core_net = pd.read_csv(os.path.join(out_dir, 'aggregated_subgraph_edges.csv'),index_col=None)

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
    Yh = Yh[:,1]
    params = {'alpha': [0.001,0.01,0.1,0.2,0.5,1,2,5,10,20,50,100,500,1000]}
    ridge = GridSearchCV(estimator=Ridge(),param_grid=params, cv=RepeatedKFold(n_splits=10, n_repeats=10), n_jobs=-1)
    ridge.fit(Xh, Yh)
    coefs = ridge.best_estimator_.coef_
    intercept = ridge.best_estimator_.intercept_
    pheno = FrozenLinear(coefs[np.newaxis, :], intercept, dtype=torch.double, device=device)
    pheno.eval()
    
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
        node_idx_map: dict[str, int] = mod.node_idx_map
        num_nodes = len(node_idx_map)
        keep_mask = torch.zeros_like(mod.signaling_network.weights, dtype=torch.bool)
        for (u, v) in allowed_edges_uniprot:
            if u in node_idx_map and v in node_idx_map:
                s_idx = node_idx_map[u]
                t_idx = node_idx_map[v]
                # weights[t_idx, s_idx] corresponds to (u -> v)
                keep_mask[t_idx, s_idx] = True
        allowed_nodes_uniprot = {u for u in allowed_nodes_uniprot if u in node_idx_map}
        bias_keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=mod.signaling_network.bias.device)
        for u in allowed_nodes_uniprot:
            if u in node_idx_map:
                idx = node_idx_map[u]
                bias_keep_mask[idx] = True
        mod.signaling_network.weights.data.masked_fill_(mask=~keep_mask, value=0.0)
        mod.signaling_network.bias.masked_fill_(~bias_keep_mask, 0.0)
        
        Yfull = mod.signaling_network(x_stim)
        YTF = mod.output_layer(Yfull)
        YTF = YTF.detach().cpu().numpy()
        YTF = pd.DataFrame(YTF, columns=TFOutput_tmp.columns.values)
        YTF['IFNAR1'] = x_stim[:,mod.node_idx_map['P17181']]
        YTF['TGFBR1'] = x_stim[:,mod.node_idx_map['P36897']]
        YTF.to_csv(os.path.join(out_dir ,'Ytf_'+ str(model_no+1)+'.csv'))
        
    
    print2log("\n" + "="*70)
    print2log(f"Finished timepoint: {timepoint}")
    print2log("="*70)

print2log("\n" + "="*70)
print2log("ANALYSIS COMPLETE!")
print2log("="*70)