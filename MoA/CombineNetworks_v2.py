"""
CombineNetworks_v2.py
---------------------
Same purpose as CombineNetworks.py but uses a num_models-weighted average
for avg_weight and avg_abs_weight instead of a simple arithmetic mean.

Why this matters
----------------
In each source file, avg_weight = (sum of per-model weights) / num_models.
So sum_of_weights = avg_weight * num_models.

When combining fibrosis and MAS, the true combined mean across all
individual model-level weights is:

    combined_avg_weight =
        (avg_weight_fib * num_models_fib + avg_weight_mas * num_models_mas)
        / (num_models_fib + num_models_mas)

This is the weighted mean by num_models. It ensures that a strong, consistent
signal (high num_models, large |avg_weight|) in one disease is not diluted or
sign-flipped by a weak, sparse signal (low num_models, small |avg_weight|) in
the other.

The same weighted formula is applied to avg_abs_weight.
For num_models and frequency, a simple mean is kept (each disease network
has the same total of 50 models, so the simple mean is the natural average
across both cohorts).

For edges unique to one disease (the other has NaN), the single value is
used as-is — no averaging needed.

Output columns
--------------
source_symbol, target_symbol, avg_weight, avg_abs_weight,
frequency, num_models, MoA
"""

import os
import glob
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

TIMEPOINT_PAIRS = {
    "2hr": (
        "results_2hr",
        "resultsMAS/results_2hr",
    ),
    "2hr_top3": (
        "results_2hr_top3",
        "resultsMAS/results_2hr_top_3",
    ),
    "8hr_top3": (
        "results_8hr_top3",
        "resultsMAS/results_8hr_top_3",
    ),
    "24hr": (
        "results_24hr",
        "resultsMAS/results_24hr",
    ),
    "24hr_top3": (
        "results_24hr_top3",
        "resultsMAS/results_24hr_top_3",
    ),
}

EDGE_COLS    = ["source_symbol", "target_symbol"]
NUMERIC_COLS = ["avg_weight", "avg_abs_weight", "frequency", "num_models"]


def load_network(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=None)
    keep = EDGE_COLS + [c for c in NUMERIC_COLS if c in df.columns]
    return df[keep].copy()


def weighted_combine(fib: pd.DataFrame, mas: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fibrosis and MAS networks with a num_models-weighted average for
    avg_weight and avg_abs_weight, and a simple mean for frequency / num_models.
    """
    # Outer join so edges unique to one disease are preserved
    merged = fib.merge(
        mas,
        on=EDGE_COLS,
        how="outer",
        suffixes=("_fib", "_mas"),
    )

    rows = []
    for _, row in merged.iterrows():
        fib_w  = row.get("avg_weight_fib")
        mas_w  = row.get("avg_weight_mas")
        fib_aw = row.get("avg_abs_weight_fib")
        mas_aw = row.get("avg_abs_weight_mas")
        fib_n  = row.get("num_models_fib")
        mas_n  = row.get("num_models_mas")
        fib_f  = row.get("frequency_fib")
        mas_f  = row.get("frequency_mas")

        fib_only = pd.isna(mas_w)
        mas_only = pd.isna(fib_w)

        if fib_only:
            # Edge unique to fibrosis — keep as-is
            avg_w  = fib_w
            avg_aw = fib_aw
            avg_n  = fib_n
            avg_f  = fib_f
        elif mas_only:
            # Edge unique to MAS — keep as-is
            avg_w  = mas_w
            avg_aw = mas_aw
            avg_n  = mas_n
            avg_f  = mas_f
        else:
            # Edge present in both: weighted average for weights,
            # simple mean for num_models and frequency
            total_n = fib_n + mas_n
            avg_w   = (fib_w  * fib_n + mas_w  * mas_n) / total_n
            avg_aw  = (fib_aw * fib_n + mas_aw * mas_n) / total_n
            avg_n   = (fib_n  + mas_n) / 2.0
            avg_f   = (fib_f  + mas_f) / 2.0

        rows.append({
            "source_symbol":  row["source_symbol"],
            "target_symbol":  row["target_symbol"],
            "avg_weight":     avg_w,
            "avg_abs_weight": avg_aw,
            "frequency":      avg_f,
            "num_models":     avg_n,
            "MoA":            "stimulation" if avg_w >= 0 else "inhibition",
        })

    result = pd.DataFrame(rows, columns=EDGE_COLS + NUMERIC_COLS + ["MoA"])
    return result.reset_index(drop=True)


def main():
    results_summary = []

    for label, (fib_subdir, mas_subdir) in TIMEPOINT_PAIRS.items():
        fib_csv = os.path.join(BASE, fib_subdir, "aggregated_subgraph_edges.csv")
        mas_csv = os.path.join(BASE, mas_subdir, "aggregated_subgraph_edges.csv")

        missing = [p for p in (fib_csv, mas_csv) if not os.path.isfile(p)]
        if missing:
            print(f"[{label}] SKIP — file(s) not found:")
            for p in missing:
                print(f"        {p}")
            continue

        print(f"[{label}] Combining networks (weighted avg) ...")
        fib = load_network(fib_csv)
        mas = load_network(mas_csv)
        combined = weighted_combine(fib, mas)

        out_path = os.path.join(BASE, f"combined_network_v2_{label}.csv")
        combined.to_csv(out_path, index=False)

        n_edges = len(combined)
        n_stim  = (combined["MoA"] == "stimulation").sum()
        n_inhib = (combined["MoA"] == "inhibition").sum()
        results_summary.append((label, n_edges, n_stim, n_inhib, out_path))
        print(f"         → {n_edges} edges  ({n_stim} stimulation, {n_inhib} inhibition)")
        print(f"         Saved: {out_path}")

    # -------------------------------------------------------------------------
    # Comparison: flag edges where v1 and v2 differ in sign
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Sign changes vs v1 (simple mean) per timepoint")
    print("=" * 70)
    for label, *_ in results_summary:
        v1 = pd.read_csv(os.path.join(BASE, f"combined_network_{label}.csv"))
        v2 = pd.read_csv(os.path.join(BASE, f"combined_network_v2_{label}.csv"))
        diff = v1.merge(v2, on=EDGE_COLS, suffixes=("_v1", "_v2"))
        flipped = diff[
            diff["avg_weight_v1"] * diff["avg_weight_v2"] < 0
        ][["source_symbol", "target_symbol",
           "avg_weight_v1", "num_models_v1",
           "avg_weight_v2", "num_models_v2"]]
        if len(flipped):
            print(f"\n[{label}] {len(flipped)} edge(s) with sign change:")
            print(flipped.to_string(index=False))
        else:
            print(f"[{label}] No sign changes.")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Timepoint':<15} {'Edges':>8} {'Stimulation':>13} {'Inhibition':>12}")
    print("-" * 50)
    for label, ne, ns, ni, _ in results_summary:
        print(f"{label:<15} {ne:>8} {ns:>13} {ni:>12}")
    print("=" * 70)


if __name__ == "__main__":
    main()