import pandas as pd
import numpy as np
import argparse
import networkx as nx

def identify_safe_drugs_robust(pkn_path, dti_mask_path, output_path):
    # 1. Load the PKN as a Directed Graph
    net = pd.read_csv(pkn_path, sep='\t')
    G = nx.DiGraph()
    for _, row in net.iterrows():
        G.add_edge(row['source'], row['target'])
    
    # 2. Load DTI and TF data
    dti = pd.read_csv(dti_mask_path, sep='\t', index_col=0)
    targets = dti.columns.values
    nodeNames = np.union1d(net.source.values, net.target.values)
    targets = np.intersect1d(nodeNames, targets)
    dti = dti.loc[:, targets]
    drugs = dti.index
    
    results = []
    
    for drug in drugs:
        # Targets of the drug we want to remove
        targets_of_drug = set(dti.columns[dti.loc[drug] == 1])
        
        # Targets that remain hit by ALL OTHER drugs
        other_drugs = dti.drop(drug, axis=0)
        remaining_drug_targets = set()
        for d in other_drugs.index:
            remaining_drug_targets.update(other_drugs.columns[other_drugs.loc[d] == 1])
            
        is_safe = True
        reasons = []
        
        for t in targets_of_drug:
            # A drug is safe to remove only if its target 't' is still 'exercised' by others.
            # This means 't' must be reachable from at least one target in remaining_drug_targets.
            
            # Check if t is directly hit by another drug
            hit_by_others = t in remaining_drug_targets
            
            # Check if t is reachable from any other drug target via PKN
            reachable_from_others = False
            if not hit_by_others and t in G:
                # We check if any of the other drug targets can reach 't'
                for s in remaining_drug_targets:
                    if s in G and nx.has_path(G, s, t):
                        reachable_from_others = True
                        break
            
            if not hit_by_others and not reachable_from_others:
                is_safe = False
                reasons.append(f"Target '{t}' becomes isolated (not hit or reachable by other drugs).")
        
        results.append({
            'Drug': drug,
            'Status': 'Safe' if is_safe else 'Unsafe',
            'Reason': "; ".join(reasons) if reasons else "N/A"
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Analysis complete. Safe: {sum(results_df['Status'] == 'Safe')}, Unsafe: {sum(results_df['Status'] == 'Unsafe')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--PKN', default='../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv')
    parser.add_argument('--TargetsIn', default='../preprocessing/preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv')
    parser.add_argument('--output', default='../preprocessing/preprocessed_data/LeaveOneDrugOut/safe_drugs_to_remove_LODO.csv')
    args = parser.parse_args()

    identify_safe_drugs_robust(args.PKN, args.TargetsIn, args.output)
