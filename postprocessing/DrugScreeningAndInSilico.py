"""
DrugScreeningAndInSilico.py
=============================================================
Performs three main tasks:
  1) Screens drugs in actual TF activity data AND model-predicted data for:
       1a) STAT2 activators (SMAD3 & TP63 neutral)
       1b) TP63 inhibitors (STAT2 & SMAD3 neutral; relaxes SMAD3 if needed)
       1c) TP63 + SMAD3 dual inhibitors (STAT2 neutral; falls back to SMAD3-only)
     Saves top-3 per category as PDF panels.

  2) For the top drugs at their best dose, runs in silico signaling experiments:
       2a) Activate TGFBR1 + IFNAR1
       2b) Inhibit TGFBR1 + Activate IFNAR1
       2c) Activate TGFBR1 + STAT2; Inhibit SMAD3, TP63, IFNAR1
       2d) Inhibit IFNAR1 + each top drug from 1b and 1c
       2e) Activate TGFBR1 + each top drug from 1a
     All results (one CSV per sub-task, model_no column).

  3) For each top drug, finds the dose that produces the TF activity profile
     closest to 2a (in the 2c stimulation context) and closest to 2b
     (when combined with IFNAR1 activation).

Usage:
  python DrugScreeningAndInSilico.py [--DrugsIn ...] [--res_dir ...] ...
=============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
import logging
import time
from collections import defaultdict

# ── matplotlib / seaborn ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

# ── LEMBAS path ────────────────────────────────────────────────────────────────
gpu_lembas_path = 'C:/Users/nmeim/Documents/Artificial-Signaling-Network/LEMBAS-main'
# gpu_lembas_path = '/home/meimetis/meimetis/LEMBAS-main'
sys.path.insert(1, os.path.join(gpu_lembas_path))
from utility import *

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger  = logging.getLogger()
print2log = logger.info

# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================
parser = argparse.ArgumentParser(prog='Drug Screening and In Silico Experiments')
parser.add_argument('--DrugsIn',    type=str,
    default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv')
parser.add_argument('--TargetsIn',  type=str,
    default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
parser.add_argument('--TFsOut',     type=str,
    default='../preprocessing/preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv')
parser.add_argument('--ChemicalSims', type=str,
    default='../preprocessing/preprocessed_data/ChemicalSims/microarray_similarities.csv')
parser.add_argument('--PKN',        type=str,
    default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
parser.add_argument('--PknAnnotation', type=str,
    default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv')
parser.add_argument('--infoPath',   type=str,
    default='../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv')
parser.add_argument('--res_dir',    type=str, default='../results/allmodels/')
parser.add_argument('--out_dir',    type=str, default='../results/drug_screening_insilio/')
parser.add_argument('--number_DTLEMBAS_models', type=int, default=50)
parser.add_argument('--modelPattern', type=str, default='hepatocytes_dtlembas')
parser.add_argument('--ConvertToEmpProb', type=bool, default=False)
parser.add_argument('--inputAmplitude',   type=float, default=3)
parser.add_argument('--projectionAmplitude', type=float, default=1.2)
parser.add_argument('--max_LEMBAS_iterations', type=int, default=90)
parser.add_argument('--LEMBAS_leak',      type=float, default=0.01)
parser.add_argument('--LEMBAS_tolerance', default=1e-5)
parser.add_argument('--drug_bn',   default=0.2)
parser.add_argument('--drug_drop', default=0.1)
parser.add_argument('--timepoint', type=str, default='8 hour')
parser.add_argument('--annotation_file_for_visualization', type=str,
    default='../data/annotation.csv')
parser.add_argument('--stimulation_strength', type=float, default=3000.0)
parser.add_argument('--top_n', type=int, default=3,
    help='Number of top drugs to report per category')

args = parser.parse_args()

# ==============================================================================
# BIOLOGICAL IDs
# ==============================================================================
TF_STAT2    = 'P52630'   # STAT2
TF_SMAD3    = 'P84022'   # SMAD3
TF_TP63     = 'Q9H3D4'   # TP63
TF_STAT3    = 'P40763'   # STAT3
NODE_TGFBR1 = 'P36897'   # TGFBR1
NODE_IFNAR1 = 'P17181'   # IFNAR1

HIGH_THRESH = 0.6   # >= activated
LOW_THRESH  = 0.4   # <= inhibited

S      = args.stimulation_strength
TOP_N  = args.top_n
TFS_PANEL = [TF_STAT2, TF_SMAD3, TF_TP63, TF_STAT3]

# ==============================================================================
# SETUP
# ==============================================================================
n_cores = 12
utils.set_cores(n_cores)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print2log(f'Running on {device}')

data_path = args.res_dir
out_dir   = args.out_dir
Path(out_dir).mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOAD DATA
# ==============================================================================
print2log('Loading input data...')
net = pd.read_csv(args.PKN, sep='\t', index_col=False).loc[
    :, ['source', 'target', 'stimulation', 'inhibition']]
drugInput   = pd.read_csv(args.DrugsIn, sep='\t', index_col=0)
TFOutput    = pd.read_csv(args.TFsOut, sep='\t', low_memory=False, index_col=0)
TFOutput    = TFOutput.loc[drugInput.index, :]
annot       = pd.read_csv(args.annotation_file_for_visualization)
drugTargets = pd.read_csv(args.TargetsIn, sep='\t', low_memory=False, index_col=0)
inputInfo   = pd.read_csv(args.infoPath, sep='\t', low_memory=False, index_col=0)
inputInfo   = inputInfo[inputInfo['time'] == args.timepoint]
TFOutput    = TFOutput.loc[inputInfo.index, :]
drugInput   = drugInput.loc[inputInfo.index, :]

net = format_network(net,
    weight_label='mode_of_action',
    stimulation_label='stimulation',
    inhibition_label='inhibition')
nodeNames = np.union1d(net.source.values, net.target.values)

inName  = np.intersect1d(nodeNames, drugTargets.columns.values)
outName = np.intersect1d(nodeNames, TFOutput.columns.values)
outconds = drugInput.index.values
TFOutput    = TFOutput.loc[outconds, outName]
drugTargets = drugTargets.loc[:, inName]

if args.ConvertToEmpProb:
    print2log('Converting to empirical probabilities...')
    TFOutput = 1 / (1 + np.exp(-TFOutput))

drugTargets = drugTargets.loc[drugInput.columns.values, :]
drugSim = pd.read_csv(args.ChemicalSims, index_col=0)
drugSim = drugSim.loc[drugInput.columns.values, drugInput.columns.values]
drugSim = torch.tensor(drugSim.values.copy(), dtype=torch.double)

# ── Annotation helpers ─────────────────────────────────────────────────────────
uniprot_to_symbol: dict[str, str] = {}
if {'uniprot', 'symbol'}.issubset(annot.columns):
    for _, r in annot.iterrows():
        if pd.notna(r['uniprot']) and pd.notna(r['symbol']):
            uniprot_to_symbol[str(r['uniprot'])] = str(r['symbol'])

def uid(u: str) -> str:
    return uniprot_to_symbol.get(u, u)

smiles_to_name: dict[str, str] = {}
for col in ('smiles', 'SMILES'):
    if col in annot.columns and 'name' in annot.columns:
        for _, r in annot.iterrows():
            if pd.notna(r.get(col)) and pd.notna(r.get('name')):
                smiles_to_name[str(r[col])] = str(r['name'])
        break

def drug_label(smiles: str) -> str:
    name = smiles_to_name.get(smiles, smiles)
    return name[:40] if len(name) > 40 else name

# ==============================================================================
# INITIALIZE MODELS
# ==============================================================================
mask_dt = torch.tensor(np.array(drugTargets)).transpose(0, 1)
bionet_params = {
    'target_steps': 100, 'max_steps': args.max_LEMBAS_iterations,
    'exp_factor': 50, 'tolerance': args.LEMBAS_tolerance,
    'leak': args.LEMBAS_leak
}
mod = SignalingModel(
    net=net, X_in=drugTargets, y_out=TFOutput,
    projection_amplitude_in=args.inputAmplitude,
    projection_amplitude_out=args.projectionAmplitude,
    weight_label='mode_of_action', source_label='source', target_label='target',
    bionet_params=bionet_params, dtype=torch.double, device=device)

drugLayer = DrugstoSignal(
    mask_dt, mask_dt.shape[1], mask_dt.shape[0],
    drugSim, momentum=args.drug_bn, drop=args.drug_drop, device=device)
drugLayer = drugLayer.to(device)

all_nodes  = np.array(list(mod.node_idx_map.keys()))
all_tfs    = mod.y_out.columns.values
num_tfs    = len(all_tfs)
num_nodes  = len(all_nodes)

print2log(f'Network: {num_nodes} nodes, {num_tfs} TFs')

# Availability checks
for tf in TFS_PANEL:
    if tf not in all_tfs:
        print2log(f'WARNING: TF {tf} ({uid(tf)}) not in TF output space.')
for node in [NODE_TGFBR1, NODE_IFNAR1, TF_STAT2, TF_SMAD3, TF_TP63]:
    if node not in mod.node_idx_map:
        print2log(f'WARNING: Node {node} ({uid(node)}) not in signaling network.')

# ==============================================================================
# UTILITY: BUILD NODE-SPACE STIMULATION VECTOR
# ==============================================================================
def build_stim(stim_dict: dict) -> torch.Tensor:
    """Return tensor (1, num_nodes) with specified node stimulations."""
    Xin  = torch.zeros(1, num_nodes, dtype=torch.double, device=device)
    Xeye = torch.eye(num_nodes, dtype=torch.double, device=device)
    for node_id, strength in stim_dict.items():
        if node_id in mod.node_idx_map:
            Xin += Xeye[mod.node_idx_map[node_id], :].unsqueeze(0) * strength
        else:
            print2log(f'  WARNING: {node_id} not in network – skipping.')
    return Xin

# ==============================================================================
# UTILITY: GET DRUG NODE-SPACE SIGNAL VIA FORWARD HOOK
# Captures the input to mod.signaling_network (i.e., the drug-target projection
# that has already been projected from drug-target space to full node space)
# ==============================================================================
def get_drug_node_signal(drug_target_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pass drug_target_tensor through mod, capture the node-space input to
    mod.signaling_network via a forward pre-hook.  Shape: (batch, n_nodes).
    """
    captured = []
    def _pre_hook(module, args):
        captured.append(args[0].detach().clone())
    handle = mod.signaling_network.register_forward_pre_hook(_pre_hook)
    with torch.no_grad():
        mod(drug_target_tensor)
    handle.remove()
    if captured:
        return captured[0]
    # Fallback: if hook did not fire, return zeros in node space
    print2log('WARNING: Forward hook did not capture node signal. Using zeros.')
    return torch.zeros(drug_target_tensor.shape[0], num_nodes,
                       dtype=torch.double, device=device)

# ==============================================================================
# UTILITY: IDENTIFY DRUG & DOSE PER CONDITION
# drugInput rows = conditions, columns = drug SMILES + 'Control'
# 'Control' column is non-zero for control/DMSO conditions.
# ==============================================================================
def build_drug_dose_map(drug_input_df: pd.DataFrame) -> dict:
    """Return {condition: (drug_smiles_or_'Control', dose_value)}."""
    result = {}
    for cond in drug_input_df.index:
        row  = drug_input_df.loc[cond]
        ctrl = row.get('Control', 0)
        if ctrl != 0:
            result[cond] = ('Control', ctrl)
            continue
        non_ctrl = row.drop('Control', errors='ignore')
        active   = non_ctrl[non_ctrl != 0]
        if active.empty:
            result[cond] = ('Control', 0)
        else:
            result[cond] = (active.index[0], float(active.iloc[0]))
    return result

drug_dose_map = build_drug_dose_map(drugInput)

# drug → [(condition, dose), ...]
drug_to_conds: dict[str, list] = defaultdict(list)
for cond, (drug, dose) in drug_dose_map.items():
    if drug != 'Control':
        drug_to_conds[drug].append((cond, dose))

print2log(f'Found {len(drug_to_conds)} unique non-control drugs across {len(drugInput)} conditions.')

# ==============================================================================
# STEP 1: PREDICT TF ACTIVITIES FOR ALL DRUG CONDITIONS (all models)
# ==============================================================================
print2log('\nPredicting TF activities for all drug conditions across all models...')
start = time.time()

all_drug_conds = [c for c, (d, _) in drug_dose_map.items() if d != 'Control']
X_all = torch.tensor(
    drugInput.loc[all_drug_conds].values.copy(), dtype=torch.double, device=device)

# Accumulator: shape (n_models, n_drug_conds, n_tfs)
pred_accum = np.zeros((args.number_DTLEMBAS_models, len(all_drug_conds), num_tfs))

for i in range(args.number_DTLEMBAS_models):
    print2log(f'  Model {i+1}/{args.number_DTLEMBAS_models}')
    mod.load_state_dict(torch.load(
        os.path.join(data_path, f'{args.modelPattern}{i+1}signal_state_dict_{i+1}.pth')))
    drugLayer.load_state_dict(torch.load(
        os.path.join(data_path, f'{args.modelPattern}{i+1}drugLayer_state_dict_{i+1}.pth')))
    mod.eval(); drugLayer.eval()
    with torch.no_grad():
        # drugLayer: (n_conds, n_drugs) → (n_conds, n_drug_targets)
        drug_target_act = drugLayer(X_all)
        # mod: (n_conds, n_drug_targets) → TF predictions via forward
        TF_pred, _ = mod(drug_target_act)
    pred_accum[i] = TF_pred.detach().cpu().numpy()

mins, secs = divmod(time.time() - start, 60)
print2log(f'Predictions done in {mins:.0f}m {secs:.1f}s')

pred_mean = pred_accum.mean(axis=0)   # (n_drug_conds, n_tfs)
pred_std  = pred_accum.std(axis=0)
pred_mean_df = pd.DataFrame(pred_mean, index=all_drug_conds, columns=all_tfs)
pred_std_df  = pd.DataFrame(pred_std,  index=all_drug_conds, columns=all_tfs)

# Save per-model predictions – only the 4 TFs of interest
tfs_of_interest = [tf for tf in [TF_STAT2, TF_SMAD3, TF_TP63, TF_STAT3] if tf in all_tfs]
tfs_of_interest_names = [uid(tf) for tf in tfs_of_interest]
pred_long = []
for i in range(args.number_DTLEMBAS_models):
    df_i = pd.DataFrame(pred_accum[i], index=all_drug_conds, columns=all_tfs)
    df_i = df_i[tfs_of_interest].copy()
    df_i.columns = tfs_of_interest_names   # rename to gene symbols for readability
    df_i['model_no']  = i
    df_i['condition'] = all_drug_conds
    pred_long.append(df_i)
pred_long_df = pd.concat(pred_long, ignore_index=True)
pred_long_df.to_csv(os.path.join(out_dir, 'drug_condition_predictions_all_models.csv'), index=False)
print2log(f'Per-model drug predictions saved (columns: {tfs_of_interest_names}).')

# ==============================================================================
# TASK 1 HELPER: CRITERIA CHECK FUNCTIONS
# ==============================================================================
def _get(row, tf_id, default=0.5):
    return float(row[tf_id]) if tf_id in row.index else default

def check_1a(row) -> bool:
    """STAT2 >= 0.6;  0.4 <= SMAD3 <= 0.6;  0.4 <= TP63 <= 0.6"""
    return (
        _get(row, TF_STAT2) >= HIGH_THRESH and
        LOW_THRESH <= _get(row, TF_SMAD3) <= HIGH_THRESH and
        LOW_THRESH <= _get(row, TF_TP63)  <= HIGH_THRESH
    )

def check_1b_full(row) -> bool:
    """TP63 <= 0.4;  0.4 <= STAT2 <= 0.6;  0.4 <= SMAD3 <= 0.6"""
    return (
        _get(row, TF_TP63)  <= LOW_THRESH and
        LOW_THRESH <= _get(row, TF_STAT2) <= HIGH_THRESH and
        LOW_THRESH <= _get(row, TF_SMAD3) <= HIGH_THRESH
    )

def check_1b_relaxed(row) -> bool:
    """TP63 <= 0.4;  0.4 <= STAT2 <= 0.6   (no SMAD3 constraint)"""
    return (
        _get(row, TF_TP63)  <= LOW_THRESH and
        LOW_THRESH <= _get(row, TF_STAT2) <= HIGH_THRESH
    )

def check_1c_dual(row) -> bool:
    """TP63 <= 0.4;  SMAD3 <= 0.4;  0.4 <= STAT2 <= 0.6"""
    return (
        _get(row, TF_TP63)  <= LOW_THRESH and
        _get(row, TF_SMAD3) <= LOW_THRESH and
        LOW_THRESH <= _get(row, TF_STAT2) <= HIGH_THRESH
    )

def check_1c_smad3_only(row) -> bool:
    """SMAD3 <= 0.4;  0.4 <= STAT2 <= 0.6"""
    return (
        _get(row, TF_SMAD3) <= LOW_THRESH and
        LOW_THRESH <= _get(row, TF_STAT2) <= HIGH_THRESH
    )

# Scoring: higher = better candidate
def score_1a(row): return _get(row, TF_STAT2)
def score_1b(row): return -_get(row, TF_TP63)
def score_1c_dual(row): return -(_get(row, TF_TP63) + _get(row, TF_SMAD3))
def score_1c_smad3(row): return -_get(row, TF_SMAD3)

# ==============================================================================
# TASK 1 HELPER: FIND TOP DRUGS
# ==============================================================================
def find_top_drugs(check_fn, score_fn, label: str, top_n: int = TOP_N):
    """
    Search both actual (TFOutput) and predicted (pred_mean_df) data.
    Returns list of dicts (length <= top_n), one per drug.
    """
    drug_best: dict[str, dict] = {}

    for source, df_source in [('actual', TFOutput), ('predicted', pred_mean_df)]:
        for drug, cond_list in drug_to_conds.items():
            best_score = None
            best_cond  = None
            best_dose  = None
            for cond, dose in cond_list:
                if cond not in df_source.index:
                    continue
                row = df_source.loc[cond]
                if check_fn(row):
                    sc = score_fn(row)
                    if best_score is None or sc > best_score:
                        best_score = sc
                        best_cond  = cond
                        best_dose  = dose
            if best_score is not None:
                entry = dict(drug=drug, best_cond=best_cond,
                             best_dose=best_dose, score=best_score,
                             source=source)
                if drug not in drug_best or best_score > drug_best[drug]['score']:
                    drug_best[drug] = entry

    ranked = sorted(drug_best.values(), key=lambda x: x['score'], reverse=True)[:top_n]
    print2log(f'[{label}] Qualifying drugs: {len(drug_best)}, showing top {len(ranked)}')
    for r in ranked:
        print2log(f'  {drug_label(r["drug"])!r:<45}  dose={r["best_dose"]:.3g}'
                  f'  score={r["score"]:.3f}  source={r["source"]}')
    return ranked

# ==============================================================================
# TASK 1a – STAT2 ACTIVATORS
# ==============================================================================
print2log('\n=== TASK 1a: STAT2 activators (SMAD3 & TP63 neutral) ===')
top_1a = find_top_drugs(check_1a, score_1a, '1a')

# ==============================================================================
# TASK 1b – TP63 INHIBITORS (neutral STAT2 and SMAD3)
# ==============================================================================
print2log('\n=== TASK 1b: TP63 inhibitors ===')
top_1b = find_top_drugs(check_1b_full, score_1b, '1b-full')
if not top_1b:
    print2log('[1b] No hits with SMAD3 constraint – relaxing to TP63+STAT2 only.')
    top_1b = find_top_drugs(check_1b_relaxed, score_1b, '1b-relaxed')

# ==============================================================================
# TASK 1c – TP63 + SMAD3 DUAL INHIBITORS (neutral STAT2)
# ==============================================================================
print2log('\n=== TASK 1c: TP63 + SMAD3 dual inhibitors ===')
top_1c = find_top_drugs(check_1c_dual, score_1c_dual, '1c-dual')
if not top_1c:
    print2log('[1c] No dual TP63+SMAD3 hits – focusing on SMAD3 only.')
    top_1c = find_top_drugs(check_1c_smad3_only, score_1c_smad3, '1c-smad3-only')

# ==============================================================================
# TASK 1: PDF FIGURE PANELS
# ==============================================================================
tfs_plot    = [tf for tf in TFS_PANEL if tf in all_tfs]
tfs_names   = [uid(tf) for tf in tfs_plot]
palette     = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

def get_dose_activity_df(drug: str, source: str) -> pd.DataFrame:
    """Returns DataFrame[dose, TF1, TF2, ...] averaged over replicates."""
    df_src = TFOutput if source == 'actual' else pred_mean_df
    rows = []
    for cond, dose in drug_to_conds.get(drug, []):
        if cond not in df_src.index:
            continue
        row = {'dose': dose}
        for tf in tfs_plot:
            row[tf] = float(df_src.loc[cond, tf]) if tf in df_src.columns else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby('dose').mean().reset_index().sort_values('dose')

def make_pdf_panel(top_drugs: list, category_label: str, pdf_path: str):
    if not top_drugs:
        print2log(f'No drugs found for {category_label} – skipping PDF.')
        return
    sns.set_style('whitegrid')
    with PdfPages(pdf_path) as pdf:
        for info in top_drugs:
            drug  = info['drug']
            dlbl  = drug_label(drug)
            bdose = info['best_dose']

            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(
                f'{category_label}\nDrug: {dlbl}  |  Best dose: {bdose:.3g}',
                fontsize=12, fontweight='bold')

            for ax, source in zip(axes, ('actual', 'predicted')):
                df = get_dose_activity_df(drug, source)
                title = source.capitalize() + ' data'
                if df.empty:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(title)
                    continue

                n_doses = len(df)
                x  = np.arange(n_doses)
                bw = 0.18
                for j, (tf, name, col) in enumerate(zip(tfs_plot, tfs_names, palette)):
                    if tf not in df.columns:
                        continue
                    offsets = x + (j - len(tfs_plot) / 2 + 0.5) * bw
                    bars = ax.bar(offsets, df[tf].values, width=bw,
                                  label=name, color=col, alpha=0.82,
                                  edgecolor='white', linewidth=0.5)
                    # Highlight bar at best dose
                    bd_mask = df['dose'].values == bdose
                    for bar, ishigh in zip(bars, bd_mask):
                        if ishigh:
                            bar.set_edgecolor('black')
                            bar.set_linewidth(2)

                ax.axhline(HIGH_THRESH, color='dimgray', linestyle='--',
                           linewidth=1, alpha=0.7, label='0.6 threshold')
                ax.axhline(LOW_THRESH, color='dimgray', linestyle=':',
                           linewidth=1, alpha=0.7, label='0.4 threshold')
                ax.set_xticks(x)
                ax.set_xticklabels([f'{d:.2g}' for d in df['dose'].values],
                                   rotation=45, ha='right', fontsize=8)
                ax.set_xlabel('Dose', fontsize=10)
                ax.set_ylabel('TF Activity', fontsize=10)
                ax.set_ylim(0, 1.05)
                ax.set_title(title, fontsize=11)
                ax.legend(fontsize=7, loc='upper right', ncol=2)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print2log(f'PDF saved → {pdf_path}')

make_pdf_panel(top_1a, 'Task 1a – STAT2 Activators (SMAD3 & TP63 neutral)',
               os.path.join(out_dir, 'task1a_STAT2_activators.pdf'))
make_pdf_panel(top_1b, 'Task 1b – TP63 Inhibitors (STAT2 neutral)',
               os.path.join(out_dir, 'task1b_TP63_inhibitors.pdf'))
make_pdf_panel(top_1c, 'Task 1c – SMAD3/TP63 Inhibitors (STAT2 neutral)',
               os.path.join(out_dir, 'task1c_SMAD3_TP63_inhibitors.pdf'))

# Save summary CSV
summary_rows = []
for cat, drug_list in [('1a_STAT2_activator', top_1a),
                        ('1b_TP63_inhibitor',  top_1b),
                        ('1c_dual_inhibitor',  top_1c)]:
    for rank, info in enumerate(drug_list, 1):
        summary_rows.append(dict(
            category=cat, rank=rank,
            drug_smiles=info['drug'],
            drug_name=drug_label(info['drug']),
            best_dose=info['best_dose'],
            best_condition=info['best_cond'],
            score=info['score'],
            source=info['source']
        ))
pd.DataFrame(summary_rows).to_csv(
    os.path.join(out_dir, 'task1_top_drugs_summary.csv'), index=False)
print2log('Task 1 summary saved.')

# ==============================================================================
# TASK 2: IN SILICO EXPERIMENTS
# ==============================================================================
print2log('\n=== TASK 2: In silico experiments ===')

# Pre-compute all stimulation vectors (pure node-space)
stim_2a = build_stim({NODE_TGFBR1:  S, NODE_IFNAR1:  S})
stim_2b = build_stim({NODE_TGFBR1: -S, NODE_IFNAR1:  S})
stim_2c = build_stim({NODE_TGFBR1:  S, TF_STAT2:    S,
                       TF_SMAD3:   -S, TF_TP63:    -S, NODE_IFNAR1: -S})
stim_ifnar1_neg  = build_stim({NODE_IFNAR1: -S})
stim_tgfbr1_pos  = build_stim({NODE_TGFBR1:  S})
stim_ifnar1_pos  = build_stim({NODE_IFNAR1:  S})  # used in Task 3

# Deduplicate drug lists for 2d and 2e
def dedup_drugs(drug_list: list) -> list:
    seen, out = set(), []
    for d in drug_list:
        if d['drug'] not in seen:
            seen.add(d['drug']); out.append(d)
    return out

drugs_2d = dedup_drugs(top_1c + top_1b)   # TP63/SMAD3 inhibitors → inhibit IFNAR1
drugs_2e = dedup_drugs(top_1a)             # STAT2 activators → activate TGFBR1

# Pre-load drug condition tensors (best dose for each drug)
def get_drug_target_tensor(best_cond: str) -> torch.Tensor:
    arr = drugInput.loc[best_cond].values.copy().astype(float)
    return torch.tensor(arr, dtype=torch.double, device=device).unsqueeze(0)

# Collect results for each sub-task
results_2abc = []     # 2a, 2b, 2c (pure stimulation)
results_2d   = []     # 2d (drug + IFNAR1 inhibition)
results_2e   = []     # 2e (drug + TGFBR1 activation)

print2log('Running Task 2 – loading each model once...')
start = time.time()

for i in range(args.number_DTLEMBAS_models):
    print2log(f'  Model {i+1}/{args.number_DTLEMBAS_models}')
    mod.load_state_dict(torch.load(
        os.path.join(data_path, f'{args.modelPattern}{i+1}signal_state_dict_{i+1}.pth')))
    drugLayer.load_state_dict(torch.load(
        os.path.join(data_path, f'{args.modelPattern}{i+1}drugLayer_state_dict_{i+1}.pth')))
    mod.eval(); drugLayer.eval()

    with torch.no_grad():

        # ── helper: predict TFs from node-space input ───────────────────────
        def predict_tfs(xin: torch.Tensor) -> np.ndarray:
            return mod.output_layer(
                mod.signaling_network(xin)
            ).detach().cpu().numpy()[0]

        # ── 2a: Activate TGFBR1 + IFNAR1 ────────────────────────────────────
        tf_2a = predict_tfs(stim_2a)
        results_2abc.append(dict(zip(all_tfs, tf_2a)) |
            {'experiment': '2a_TGFbR1pos_IFNaR1pos',
             'TGFbR1_stim': S, 'IFNaR1_stim': S, 'model_no': i})

        # ── 2b: Inhibit TGFBR1 + Activate IFNAR1 ───────────────────────────
        tf_2b = predict_tfs(stim_2b)
        results_2abc.append(dict(zip(all_tfs, tf_2b)) |
            {'experiment': '2b_TGFbR1neg_IFNaR1pos',
             'TGFbR1_stim': -S, 'IFNaR1_stim': S, 'model_no': i})

        # ── 2c: TGFbR1+STAT2 ON; SMAD3+TP63+IFNAR1 OFF ─────────────────────
        tf_2c = predict_tfs(stim_2c)
        results_2abc.append(dict(zip(all_tfs, tf_2c)) |
            {'experiment': '2c_TGFbR1pos_STAT2pos_SMAD3neg_TP63neg_IFNaR1neg',
             'TGFbR1_stim': S, 'IFNaR1_stim': -S,
             'STAT2_stim': S, 'SMAD3_stim': -S, 'TP63_stim': -S, 'model_no': i})

        # ── 2d: Inhibit IFNAR1 + each drug from 1b/1c ───────────────────────
        for info in drugs_2d:
            dlbl  = drug_label(info['drug'])
            bcond = info['best_cond']
            if bcond not in drugInput.index:
                print2log(f'  WARNING: Condition {bcond} not found – skipping {dlbl}.')
                continue
            Xdt = drugLayer(get_drug_target_tensor(bcond))  # n_drug_targets
            # Capture projected node signal
            node_sig = get_drug_node_signal(Xdt)            # n_nodes
            combined = node_sig + stim_ifnar1_neg
            tf_2d    = predict_tfs(combined)
            results_2d.append(dict(zip(all_tfs, tf_2d)) |
                {'experiment': f'2d_IFNaR1neg+{dlbl[:25]}',
                 'drug_name': dlbl, 'drug_smiles': info['drug'],
                 'drug_best_dose': info['best_dose'],
                 'IFNaR1_stim': -S, 'drug_category': '1b/1c', 'model_no': i})

        # ── 2e: Activate TGFBR1 + each drug from 1a ─────────────────────────
        for info in drugs_2e:
            dlbl  = drug_label(info['drug'])
            bcond = info['best_cond']
            if bcond not in drugInput.index:
                print2log(f'  WARNING: Condition {bcond} not found – skipping {dlbl}.')
                continue
            Xdt = drugLayer(get_drug_target_tensor(bcond))
            node_sig = get_drug_node_signal(Xdt)
            combined = node_sig + stim_tgfbr1_pos
            tf_2e    = predict_tfs(combined)
            results_2e.append(dict(zip(all_tfs, tf_2e)) |
                {'experiment': f'2e_TGFbR1pos+{dlbl[:25]}',
                 'drug_name': dlbl, 'drug_smiles': info['drug'],
                 'drug_best_dose': info['best_dose'],
                 'TGFbR1_stim': S, 'drug_category': '1a', 'model_no': i})

mins, secs = divmod(time.time() - start, 60)
print2log(f'Task 2 completed in {mins:.0f}m {secs:.1f}s')

# Save Task 2 CSVs
def save_results(rows, path):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print2log(f'Saved → {path}  ({len(df)} rows)')

save_results(results_2abc, os.path.join(out_dir, 'task2abc_pure_stimulations.csv'))
save_results(results_2d,   os.path.join(out_dir, 'task2d_IFNaR1neg_plus_drugs.csv'))
save_results(results_2e,   os.path.join(out_dir, 'task2e_TGFbR1pos_plus_drugs.csv'))

# Combined Task 2 CSV
all_task2 = results_2abc + results_2d + results_2e
save_results(all_task2, os.path.join(out_dir, 'task2_all_insilio_results.csv'))

# ==============================================================================
# TASK 3: DOSE MATCHING
# ==============================================================================
print2log('\n=== TASK 3: Dose matching ===')
print2log('Finding dose per drug that best matches 2a (in 2c context) and 2b '
          '(with IFNAR1 activation).')

# Reference profiles: mean over models from Task 2
df_2abc = pd.DataFrame(results_2abc)
ref_2a  = df_2abc[df_2abc['experiment'] == '2a_TGFbR1pos_IFNaR1pos'][list(all_tfs)].mean(axis=0).values
ref_2b  = df_2abc[df_2abc['experiment'] == '2b_TGFbR1neg_IFNaR1pos'][list(all_tfs)].mean(axis=0).values

def pearson_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(r)

# Collect all unique drugs across all categories
all_top_drugs: dict[str, dict] = {}
for lst in (top_1a, top_1b, top_1c):
    for info in lst:
        if info['drug'] not in all_top_drugs:
            all_top_drugs[info['drug']] = info

# For each drug × dose × model, predict under 2c-context and 2b-context
# We accumulate (drug, cond) → list of predictions per model
from collections import defaultdict
preds_2c_ctx   : dict = defaultdict(list)   # key=(drug,cond)
preds_2b_ifnar : dict = defaultdict(list)

# Pre-build list of (drug, cond, dose) to test
dose_combos = []
for drug, info in all_top_drugs.items():
    for cond, dose in drug_to_conds.get(drug, []):
        if cond in drugInput.index:
            dose_combos.append((drug, cond, dose))

print2log(f'Testing {len(dose_combos)} (drug, dose) combinations × '
          f'{args.number_DTLEMBAS_models} models...')
start = time.time()

stim_2c_ctx    = stim_2c         # (1, n_nodes) – same as Task 2c
stim_2b_ifnar  = stim_2b         # (1, n_nodes) – same as Task 2b

for i in range(args.number_DTLEMBAS_models):
    print2log(f'  Task 3 – Model {i+1}/{args.number_DTLEMBAS_models}')
    mod.load_state_dict(torch.load(
        os.path.join(data_path, f'{args.modelPattern}{i+1}signal_state_dict_{i+1}.pth')))
    drugLayer.load_state_dict(torch.load(
        os.path.join(data_path, f'{args.modelPattern}{i+1}drugLayer_state_dict_{i+1}.pth')))
    mod.eval(); drugLayer.eval()

    with torch.no_grad():
        for drug, cond, dose in dose_combos:
            Xdt      = drugLayer(get_drug_target_tensor(cond))
            node_sig = get_drug_node_signal(Xdt)

            # Context A: drug + 2c stimulation → compare to 2a profile
            yhat_2c = mod.output_layer(
                mod.signaling_network(node_sig + stim_2c_ctx)
            ).detach().cpu().numpy()[0]
            preds_2c_ctx[(drug, cond)].append(yhat_2c)

            # Context B: drug + IFNAR1 activation (2b context) → compare to 2b profile
            yhat_2b_ifnar = mod.output_layer(
                mod.signaling_network(node_sig + stim_2b_ifnar)
            ).detach().cpu().numpy()[0]
            preds_2b_ifnar[(drug, cond)].append(yhat_2b_ifnar)

mins, secs = divmod(time.time() - start, 60)
print2log(f'Task 3 predictions done in {mins:.0f}m {secs:.1f}s')

# Compute similarities and find best dose per drug
dose_match_rows  = []
best_dose_summary = []

for drug, info in all_top_drugs.items():
    dlbl = drug_label(drug)
    best_sim_2a, best_dose_2a = None, None
    best_sim_2b, best_dose_2b = None, None

    for cond, dose in drug_to_conds.get(drug, []):
        key = (drug, cond)
        if key not in preds_2c_ctx:
            continue

        mean_2c_ctx   = np.mean(preds_2c_ctx[key],   axis=0)
        mean_2b_ifnar = np.mean(preds_2b_ifnar[key], axis=0)

        sim_2a = pearson_sim(mean_2c_ctx,   ref_2a)
        sim_2b = pearson_sim(mean_2b_ifnar, ref_2b)

        dose_match_rows.append({
            'drug':       drug,
            'drug_name':  dlbl,
            'dose':       dose,
            'condition':  cond,
            'pearson_sim_to_2a_in_2c_context':  sim_2a,
            'pearson_sim_to_2b_with_IFNAR1_act': sim_2b,
        })

        if best_sim_2a is None or sim_2a > best_sim_2a:
            best_sim_2a, best_dose_2a = sim_2a, dose
        if best_sim_2b is None or sim_2b > best_sim_2b:
            best_sim_2b, best_dose_2b = sim_2b, dose

    if best_dose_2a is not None:
        print2log(f'  {dlbl}:')
        print2log(f'    Closest to 2a (2c context): dose={best_dose_2a:.3g} '
                  f'  Pearson r={best_sim_2a:.3f}')
        print2log(f'    Closest to 2b (IFNAR1 act): dose={best_dose_2b:.3g} '
                  f'  Pearson r={best_sim_2b:.3f}')
        best_dose_summary.append({
            'drug':      drug,
            'drug_name': dlbl,
            'best_dose_matching_2a_in_2c_context':   best_dose_2a,
            'pearson_r_2a': best_sim_2a,
            'best_dose_matching_2b_with_IFNAR1_act': best_dose_2b,
            'pearson_r_2b': best_sim_2b,
        })

# Save Task 3 results
pd.DataFrame(dose_match_rows).to_csv(
    os.path.join(out_dir, 'task3_dose_similarity_all.csv'), index=False)
pd.DataFrame(best_dose_summary).to_csv(
    os.path.join(out_dir, 'task3_best_dose_per_drug.csv'), index=False)
print2log('Task 3 results saved.')

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print2log('\n' + '='*65)
print2log('ALL TASKS COMPLETE')
print2log(f'Output directory: {out_dir}')
print2log('Files generated:')
for f in sorted(os.listdir(out_dir)):
    print2log(f'  {f}')
print2log('='*65)
