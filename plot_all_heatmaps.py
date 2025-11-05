#!/usr/bin/env python3
"""
Generate heatmaps for accuracy, loss and training time from lr_lrdecay_matrix.csv
Saves images to plots/accuracy_heatmap.png, plots/loss_heatmap.png, plots/time_heatmap.png
"""
import os
import math
import sys
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

import argparse


def detect_sep(path):
    # quick detect: if semicolon present in header use ; else ,
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head = f.read(1024)
    if ';' in head.splitlines()[0]:
        return ';'
    return ','


parser = argparse.ArgumentParser(description='Generate heatmaps from a CSV of LR x LR_DECAY results')
parser.add_argument('csv', nargs='?', default=None, help='CSV file (default: auto lr_lrdecay_results.csv or lr_lrdecay_matrix.csv)')
parser.add_argument('--out', default='plots', help='output folder for images')
args = parser.parse_args()

CSV = args.csv or ('lr_lrdecay_results.csv' if os.path.exists('lr_lrdecay_results.csv') else 'lr_lrdecay_matrix.csv')
OUT_DIR = args.out
os.makedirs(OUT_DIR, exist_ok=True)


def load_df():
    # detect separator
    sep = ','
    try:
        sep = detect_sep(CSV)
    except Exception:
        pass
    if pd is None:
        return None
    df = pd.read_csv(CSV, sep=sep, comment='#')
    # sanitize
    if 'accuracy' in df.columns:
        df['accuracy'] = df['accuracy'].astype(str).str.replace('%','',regex=False)
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    for c in ('LR','LR_DECAY','loss','train_time_s'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def pivot_metric(df, metric):
    pivot = df.pivot_table(index='LR', columns='LR_DECAY', values=metric, aggfunc='mean')
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def plot_pivot(pivot, outpath, title, cmap='viridis', fmt='.2f'):
    plt.figure(figsize=(10,6))
    if sns is not None:
        sns.set()
        ax = sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, cbar_kws={'label': title})
    else:
        plt.imshow(pivot.values, aspect='auto', cmap=cmap)
        plt.colorbar(label=title)
        # annotate
        for (i,j), val in np.ndenumerate(pivot.values):
            if not math.isnan(val):
                plt.text(j, i, f"{val:{fmt}}", ha='center', va='center', color='white', fontsize=8)
        plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=45)
        plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.title(title)
    plt.xlabel('LR_DECAY')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print('Saved', outpath)


def main():
    if not os.path.exists(CSV):
        print('CSV not found:', CSV)
        sys.exit(1)
    df = load_df()
    if df is None:
        # fallback manual parsing without pandas
        import csv
        rows = []
        sep = ','
        try:
            sep = detect_sep(CSV)
        except Exception:
            pass
        with open(CSV, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter=sep)
            for r in reader:
                rows.append(r)
        import pandas as _pd
        df = _pd.DataFrame(rows)
        if 'accuracy' in df.columns:
            df['accuracy'] = df['accuracy'].astype(str).str.replace('%','',regex=False)
            df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        for c in ('LR','LR_DECAY','loss','train_time_s'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    # metrics
    for metric, fname, fmt in [('accuracy','accuracy_heatmap.png','%.2f'), ('loss','loss_heatmap.png','%.3f'), ('train_time_s','time_heatmap.png','%.0f')]:
        pivot = pivot_metric(df, metric)
        if pivot.size == 0:
            print('No data for', metric)
            continue
        outpath = os.path.join(OUT_DIR, fname)
        plot_pivot(pivot, outpath, f'{metric} (mean)')


if __name__ == '__main__':
    main()
