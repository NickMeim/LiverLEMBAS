"""
AnnotateNetworkNodes.py
-----------------------
Reads one or more network CSV files whose edges are described by
source_symbol / target_symbol columns, collects every unique node,
and annotates each node with a 'type' column:

    source     – TGFBR1 or IFNAR1 (receptor inputs)
    TF         – transcription factor (output layer of LEMBAS models)
    phenotype  – MAS or fibrosis
    mid_node   – everything else

TF identity is resolved automatically from the TF_predictions CSV files
found in baseline_signaling_node_activity_analysis/ by mapping UniProt IDs
→ gene symbols using the individual-network edge lists in results/.

Usage
-----
# Annotate all combined networks (default):
python AnnotateNetworkNodes.py

# Annotate a specific network file:
python AnnotateNetworkNodes.py --networks combined_network_8hr_top3.csv

# Annotate a custom PKN-with-symbols CSV:
python AnnotateNetworkNodes.py --networks my_pkn_symbols.csv \
                               --source_col source_symbol \
                               --target_col target_symbol

Output
------
node_annotations.csv   saved to the same folder as this script
Columns: node, type
"""

import argparse
import glob
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (all relative to the directory that contains this script)
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

# Individual edge-lists used to build the UniProt → symbol mapping
INDIVIDUAL_NET_PATTERNS = [
    os.path.join(BASE, "results", "individual_networks", "*.csv"),
    os.path.join(BASE, "results_*", "individual_networks", "*.csv"),
    os.path.join(BASE, "resultsMAS", "results_*", "individual_networks", "*.csv"),
]

# TF activity files whose column headers carry TF UniProt IDs
TF_PREDICTION_PATTERN = os.path.join(
    BASE, "baseline_signaling_node_activity_analysis", "*", "TF_predictions_control.csv"
)

# Nodes with fixed roles
SOURCE_NODES = {"TGFBR1", "IFNAR1"}
PHENOTYPE_NODES = {"MAS", "fibrosis"}

# Column names used for non-meta entries in TF prediction files
TF_PRED_META_COLS = {"model_no", "condition"}


# ---------------------------------------------------------------------------
# Helper: build UniProt → gene-symbol mapping
# ---------------------------------------------------------------------------
def build_uniprot_symbol_map() -> dict:
    """
    Scan every individual-network edge list for (source_uniprot, source_symbol)
    and (target_uniprot, target_symbol) pairs and return a uniprot→symbol dict.
    """
    mapping: dict[str, str] = {}
    files = []
    for pat in INDIVIDUAL_NET_PATTERNS:
        files.extend(glob.glob(pat))

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
            for u_col, s_col in [
                ("source_uniprot", "source_symbol"),
                ("target_uniprot", "target_symbol"),
            ]:
                if u_col in df.columns and s_col in df.columns:
                    pairs = df[[u_col, s_col]].dropna().drop_duplicates()
                    for _, row in pairs.iterrows():
                        u, s = str(row[u_col]), str(row[s_col])
                        if u not in PHENOTYPE_NODES and u != "nan":
                            mapping[u] = s
        except Exception:
            pass

    return mapping


# ---------------------------------------------------------------------------
# Helper: derive the set of TF gene symbols
# ---------------------------------------------------------------------------
def get_tf_symbols(uniprot2sym: dict) -> set:
    """
    Read column headers from every TF_predictions_control.csv, strip meta
    columns, map remaining UniProt IDs → symbols via uniprot2sym, and return
    the set of resolved gene symbols.
    """
    tf_uniprots: set[str] = set()
    for fpath in glob.glob(TF_PREDICTION_PATTERN):
        try:
            df = pd.read_csv(fpath, nrows=0)
            for col in df.columns:
                if col not in TF_PRED_META_COLS:
                    tf_uniprots.add(col)
        except Exception:
            pass

    tf_symbols: set[str] = set()
    for u in tf_uniprots:
        if u in uniprot2sym:
            tf_symbols.add(uniprot2sym[u])

    return tf_symbols


# ---------------------------------------------------------------------------
# Helper: classify a single node symbol
# ---------------------------------------------------------------------------
def classify_node(symbol: str, tf_symbols: set) -> str:
    if symbol in SOURCE_NODES:
        return "source"
    if symbol in PHENOTYPE_NODES:
        return "phenotype"
    if symbol in tf_symbols:
        return "TF"
    return "mid_node"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Annotate network nodes with a 'type' column."
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        default=None,
        help=(
            "One or more network CSV files to read nodes from. "
            "Defaults to all combined_network_*.csv in the script folder."
        ),
    )
    parser.add_argument(
        "--source_col",
        default="source_symbol",
        help="Column name for source nodes (default: source_symbol).",
    )
    parser.add_argument(
        "--target_col",
        default="target_symbol",
        help="Column name for target nodes (default: target_symbol).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE, "node_annotations.csv"),
        help="Output CSV path (default: node_annotations.csv in script folder).",
    )
    args = parser.parse_args()

    # Resolve input network files
    if args.networks is None:
        net_files = sorted(glob.glob(os.path.join(BASE, "combined_network_*.csv")))
    else:
        net_files = args.networks

    if not net_files:
        raise FileNotFoundError(
            "No network CSV files found. Provide --networks or ensure "
            "combined_network_*.csv files exist in the script folder."
        )

    print(f"Network files to process ({len(net_files)}):")
    for f in net_files:
        print(f"  {os.path.basename(f)}")

    # Build TF set
    print("\nBuilding UniProt → symbol mapping from individual networks...")
    uniprot2sym = build_uniprot_symbol_map()
    print(f"  Mapping entries: {len(uniprot2sym)}")

    print("Deriving TF set from TF_predictions files...")
    tf_symbols = get_tf_symbols(uniprot2sym)
    print(f"  TF symbols resolved: {len(tf_symbols)}")

    # Collect all unique nodes across all input networks
    all_nodes: set[str] = set()
    for fpath in net_files:
        df = pd.read_csv(fpath)
        if args.source_col not in df.columns or args.target_col not in df.columns:
            raise ValueError(
                f"Columns '{args.source_col}' / '{args.target_col}' not found in {fpath}. "
                "Use --source_col / --target_col to specify correct column names."
            )
        all_nodes.update(df[args.source_col].dropna().astype(str))
        all_nodes.update(df[args.target_col].dropna().astype(str))

    print(f"\nTotal unique nodes across all networks: {len(all_nodes)}")

    # Annotate
    records = []
    for node in sorted(all_nodes):
        records.append({"node": node, "type": classify_node(node, tf_symbols)})

    annotations = pd.DataFrame(records)

    # Summary
    type_counts = annotations["type"].value_counts()
    print("\nNode type breakdown:")
    for t, n in type_counts.items():
        print(f"  {t:<12} {n}")

    # Save
    annotations.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
