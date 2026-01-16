import os
import numpy as np
import pandas as pd
import torch
import sys 
# gpu_lembas_path = '/nobackup/users/meimetis/LEMBAS-main'
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
sys.path.insert(1,os.path.join(gpu_lembas_path))
import argparse
from utility import *
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def make_allowed_edges_uniprot_from_core(core_net: pd.DataFrame, sym2id: dict) -> set:
    PHENOS = {'fibrosis','MAS','Fibrosis','NAS','mas','nas'}
    # 1) drop phenotype edges
    core_f = core_net[
        ~core_net["Source_symbol"].isin(PHENOS) &
        ~core_net["Target_symbol"].isin(PHENOS)
    ].copy()

    # 2) map symbols -> uniprot
    src_u = core_f["Source_symbol"].map(sym2id)
    tgt_u = core_f["Target_symbol"].map(sym2id)

    # 3) sanity check missing mappings
    missing = core_f[src_u.isna() | tgt_u.isna()][["Source_symbol", "Target_symbol"]]
    if len(missing) > 0:
        # If you *know* everything should map, fail loudly
        raise ValueError(
            "Some core symbols could not be mapped to UniProt. Examples:\n"
            + missing.head(20).to_string(index=False)
        )

    allowed_edges_uniprot = set(zip(src_u.astype(str), tgt_u.astype(str)))
    return allowed_edges_uniprot

parser = argparse.ArgumentParser(prog='Hepatocyte simulation')
parser.add_argument('--log_filename', action='store', type=str, 
                    default='trimmed_simulation.log',
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
parser.add_argument('--res_dir', action='store',type = str, default='../results/')
parser.add_argument('--out_dir', action='store',type = str, default='TrimmedSimulationResults/')
parser.add_argument('--saved_PPS_MoA', action='store',type = str, default='PPS_results/aggregated_edges_8hours_freq0.5.csv')
parser.add_argument('--timepoint', type=str, action='store', help='Timepoint to simulate',default='8 hour')
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
parser.add_argument('--stimulation_strength', action='store',type=float, default=3000)
parser.add_argument('--num_steps', action='store',type = float, default=101)
parser.add_argument('--selected_signaling_nodes', metavar='N', type=str, nargs='*', action='store', 
                    help='Selected signaling nodes to predict and save their activity',
                    default=['P36897','P17181'])


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
out_dir = args.out_dir
saved_PPS_MoA = args.saved_PPS_MoA
selected_signaling_nodes = np.array(args.selected_signaling_nodes)
num_steps = args.num_steps
timepoint = args.timepoint
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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

core_net = pd.read_csv(saved_PPS_MoA,index_col=None)
allowed_edges_uniprot = make_allowed_edges_uniprot_from_core(core_net, sym2id)
allowed_nodes_uniprot = set()
for u, v in allowed_edges_uniprot:
    allowed_nodes_uniprot.add(u)
    allowed_nodes_uniprot.add(v)

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
results_all = pd.DataFrame({})  
results_control = pd.DataFrame({})  
for model_no in range(number_DTLEMBAS_models):
    print2log(f"MODEL {model_no+1}/{number_DTLEMBAS_models}")
        
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
    with torch.no_grad():
        mod.signaling_network.weights.data.masked_fill_(mask=~keep_mask, value=0.0)
        bias = mod.signaling_network.bias
        mask = ~bias_keep_mask
        if bias.dim() == 2 and bias.shape[1] == 1:
            mask = mask.view(-1, 1)
        mod.signaling_network.bias.masked_fill_(~mask, 0.0)
    mod.eval()

    # Prepare stimulation input
    all_nodes = np.array(list(mod.node_idx_map.keys()))
    # indices for the two selected nodes
    subset_nodes = {key: mod.node_idx_map[key] for key in selected_signaling_nodes if key in mod.node_idx_map}
    subset_nodes_inds = np.array(list(subset_nodes.values()))
    if len(subset_nodes_inds) != 2:
        raise ValueError("Could not find both P36897 and P17181 in mod.node_idx_map.")

    # one-hot basis vectors for each receptor (same approach as Xin_combined script)
    with torch.no_grad():
        Xin = torch.eye(len(all_nodes), dtype=torch.double).to(device)
        Xin_1 = Xin[subset_nodes_inds[0], :].unsqueeze(0)
        Xin_2 = Xin[subset_nodes_inds[1], :].unsqueeze(0)
        # stimulation grid values
        stim_range = torch.linspace(-stimulation_strength, stimulation_strength, int(num_steps))
        grid_Xin1, grid_Xin2 = torch.meshgrid(stim_range, stim_range, indexing='ij')
        # flatten to (num_steps^2, 1)
        grid_Xin1 = grid_Xin1.flatten().view(-1, 1)
        grid_Xin2 = grid_Xin2.flatten().view(-1, 1)
        # expand to full node dimension
        grid_Xin1 = grid_Xin1.repeat(1, Xin_1.shape[1])
        grid_Xin2 = grid_Xin2.repeat(1, Xin_2.shape[1])
        # scale the one-hot vectors and combine
        Xin_1_grid = Xin_1 * grid_Xin1.to(device).double()
        Xin_2_grid = Xin_2 * grid_Xin2.to(device).double()
        x_stim = Xin_1_grid + Xin_2_grid
        
    # Run simulation
    with torch.no_grad():
        Yfull = mod.signaling_network(x_stim)
        YTF = mod.output_layer(Yfull)
        x0 = torch.zeros(1, drugTargets.shape[1], dtype=torch.double).to(device)
        TF0, _ = mod(x0) 
    YTF = YTF.detach().cpu().numpy()
    df = pd.DataFrame(YTF, columns=TFOutput_tmp.columns.values)
    df['IFNAR1'] = x_stim[:,mod.node_idx_map['P17181']].detach().cpu().numpy()
    df['TGFBR1'] = x_stim[:,mod.node_idx_map['P36897']].detach().cpu().numpy()
    df['type'] = 'stimuli'
    df['model_no'] = model_no
    results_all = pd.concat([results_all, df], axis=0, ignore_index=True)

    df_control = pd.DataFrame(TF0.detach().cpu().numpy(), columns=TFOutput_tmp.columns.values)
    df_control['type'] = 'untreated'
    df_control['model_no'] = model_no
    results_control = pd.concat([results_control, df_control], axis=0, ignore_index=True)

results_all.to_csv(os.path.join(out_dir ,'TrimmedSimulation_PPS_'+ timepoint+'.csv'))
results_control.to_csv(os.path.join(out_dir, 'TrimmedSimulation_control_PPS_' + timepoint + '.csv'))
print2log("\n" + "="*70)
print2log(f"Finished timepoint: {timepoint}")
print2log("="*70)