#!/usr/bin/env python3
"""
Generate combined heatmap figures where for each metric (accuracy, loss, time)
we plot side-by-side heatmaps for multiple models (e.g. CNN and MLP).

Usage:
    python3 plot_combined_heatmaps.py --csv lr_lrdecay_results_cnn.csv lr_lrdecay_results_mlp.csv \
        --labels cnn mlp --out plots

If --labels is omitted, labels are inferred from CSV filenames.
"""
import os
import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import seaborn as sns
except Exception:
    sns = None


def detect_sep(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head = f.read(1024)
    if ';' in head.splitlines()[0]:
        return ';'
    return ','


def load_df(path):
    sep = ','
    try:
        sep = detect_sep(path)
    except Exception:
        pass
    if pd is None:
        raise RuntimeError('pandas required')
    df = pd.read_csv(path, sep=sep, comment='#')
    if 'accuracy' in df.columns:
        df['accuracy'] = df['accuracy'].astype(str).str.replace('%','',regex=False)
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    for c in ('LR','LR_DECAY','loss','train_time_s'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def pivot_metric(df, metric, param1='LR', param2='LR_DECAY'):
    # pivot on the provided parameter names (defaults to LR x LR_DECAY)
    if param1 not in df.columns or param2 not in df.columns:
        # try case-insensitive match
        cols = {c.upper(): c for c in df.columns}
        p1 = cols.get(param1.upper())
        p2 = cols.get(param2.upper())
        if p1 and p2:
            param1, param2 = p1, p2
        else:
            raise KeyError(f"Missing pivot columns: {param1} or {param2} not in CSV columns {list(df.columns)}")
    pivot = df.pivot_table(index=param1, columns=param2, values=metric, aggfunc='mean')
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def plot_combined(pivots, labels, outpath, metric, cmap='viridis', fmt='.2f', param1_name='LR', param2_name='LR_DECAY'):
    n = len(pivots)
    if n == 0:
        return
    # determine common vmin/vmax for comparable colorbars
    # flatten each pivot's numeric values (they may have different shapes)
    list_vals = [p.values.flatten().astype(float) for p in pivots]
    # concatenate non-nan entries across all pivots
    non_nan_segments = [v[~np.isnan(v)] for v in list_vals if v.size>0]
    all_vals = np.concatenate(non_nan_segments) if len(non_nan_segments) > 0 else np.array([])
    if all_vals.size == 0:
        print('No numeric data for', metric)
        return
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    # plot
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), squeeze=False)
    for i, (pivot, label) in enumerate(zip(pivots, labels)):
        ax = axes[0, i]
        if sns is not None:
            sns.set()
            sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, cbar=i==n-1, vmin=vmin, vmax=vmax, ax=ax, cbar_kws={'label':metric})
        else:
            im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
            if i == n-1:
                fig.colorbar(im, ax=axes[0,:].tolist(), label=metric)
            for (r,c), val in np.ndenumerate(pivot.values):
                if not math.isnan(val):
                    ax.text(c, r, f"{val:{fmt}}", ha='center', va='center', color='white', fontsize=8)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_title(f"{label}")
    # label axes using the provided parameter names (param1 -> y, param2 -> x)
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    plt.suptitle(f'{metric} (mean)')
    plt.tight_layout(rect=[0,0,1,0.96])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print('Saved', outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', nargs='+', required=True, help='one or more per-model CSV files')
    parser.add_argument('--labels', nargs='+', help='labels for the CSVs (same length)')
    parser.add_argument('--out', default='plots', help='output folder')
    parser.add_argument('--prefix', default=None, help='optional prefix for output filenames (eg: lr_vs_lrdecay)')
    parser.add_argument('--param1-name', default='LR', help='name of the first parameter column in the CSV (default: LR)')
    parser.add_argument('--param2-name', default='LR_DECAY', help='name of the second parameter column in the CSV (default: LR_DECAY)')
    args = parser.parse_args()

    csvs = args.csv
    labels = args.labels or []
    if labels and len(labels) != len(csvs):
        print('labels count must match csv count')
        sys.exit(2)
    if not labels:
        # infer from filenames
        labels = [os.path.splitext(os.path.basename(p))[0].replace('lr_lrdecay_results_','') for p in csvs]

    dfs = []
    for p in csvs:
        if not os.path.exists(p):
            print('CSV not found:', p)
            sys.exit(1)
        dfs.append(load_df(p))

    # fmt values: seaborn expects format like '.2f' (not '%.2f')
    metrics = [('accuracy','accuracy_combined.png','.2f'), ('loss','loss_combined.png','.3f'), ('train_time_s','time_combined.png','.0f')]
    prefix = args.prefix
    for metric, fname, fmt in metrics:
        try:
            pivots = []
            for df in dfs:
                try:
                    pivot = pivot_metric(df, metric, param1=args.param1_name, param2=args.param2_name)
                except KeyError as ke:
                    print(f"Skipping dataframe for metric '{metric}': {ke}")
                    pivot = None
                pivots.append(pivot)

            # filter out None pivots (missing columns)
            pivots = [p for p in pivots if p is not None]
            if not pivots:
                print(f"No pivots available for metric '{metric}'; skipping")
                continue

            if prefix:
                fname = f"{prefix}_{metric}_combined.png"
            outpath = os.path.join(args.out, fname)

            try:
                plot_combined(pivots, labels, outpath, metric, fmt=fmt, param1_name=args.param1_name, param2_name=args.param2_name)
            except Exception as e:
                print(f"Failed to plot metric '{metric}': {e}")
                import traceback
                traceback.print_exc()
                continue
        except Exception as e:
            print(f"Unexpected error while processing metric '{metric}': {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()
