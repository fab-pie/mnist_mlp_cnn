#!/usr/bin/env python3
"""
Rebuild a CSV for a given prefix and model by scanning logs/* files produced by the run scripts.

Usage:
    python3 scripts/rebuild_csv_from_logs.py <prefix> <model_label> <param2_token> [--out <out_csv>]

Examples:
    python3 scripts/rebuild_csv_from_logs.py lr_vs_angle cnn ang --out results_lr_vs_angle_cnn.csv
    python3 scripts/rebuild_csv_from_logs.py lr_vs_batch mlp batch --out results_lr_vs_batch_mlp.csv

It extracts LR and the second parameter (param2_token) from filenames like:
   run_cnn_1_lr0p005_ang0_s15.log  or run_mlp_3_lr0p01_batch128_s15.log
and parses loss/accuracy/train_time_s from the log body using the same regex heuristics as append_from_log.py
"""
import re
import sys
from pathlib import Path

if len(sys.argv) < 4:
    print('Usage: rebuild_csv_from_logs.py <prefix> <model_label> <param2_token> [--out <out_csv>]')
    sys.exit(2)

prefix = sys.argv[1]
label = sys.argv[2]
param2_token = sys.argv[3].lower()  # e.g. 'ang' or 'batch' or 'pat'
out_csv = None
if '--out' in sys.argv:
    out_idx = sys.argv.index('--out')
    if out_idx+1 < len(sys.argv):
        out_csv = sys.argv[out_idx+1]
if out_csv is None:
    out_csv = f'results_{prefix}_{label}.csv'

logdir = Path('logs')
if not logdir.exists():
    print('No logs/ directory found; nothing to do')
    sys.exit(1)

# Only select logs that contain the model label and the param2 token (case-insensitive)
files = sorted([p for p in logdir.iterdir() if p.is_file() and f'_{label}_' in p.name and f'_{param2_token}' in p.name.lower()])
rows = []

# helper to parse metrics from log text
def parse_log_text(txt):
    lr = ''
    decay = ''
    # find LR in text
    m = re.search(r'LR[:= ]+([0-9.eE+-]+)', txt)
    if m:
        lr = m.group(1)
    m = re.search(r'LR[_ ]?DECAY[:= ]+([0-9.eE+-]+)', txt)
    if m:
        decay = m.group(1)
    loss_m = re.findall(r'loss[:= ]+([0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)', txt)
    loss = loss_m[-1] if loss_m else ''
    acc_m = re.findall(r'(?:accuracy|acc)[:= ]+([0-9]+\.?[0-9]*)%?', txt, flags=re.IGNORECASE)
    accuracy = acc_m[-1] if acc_m else ''
    time_s = ''
    t_m = re.search(r'train[_ ]?time[:= ]+([0-9]+\.?[0-9]*)s', txt, flags=re.IGNORECASE)
    if t_m:
        time_s = t_m.group(1)
    return dict(LR=lr, LR_DECAY=decay, loss=loss, accuracy=accuracy, train_time_s=time_s)

for p in files:
    name = p.name
    # try to extract lr and the param2 value from filename, e.g. lr0p005_ang0 or lr0p01_batch128
    # support patterns like lr0p005_ang0 or lr0p005_ang0_s15 or run_cnn_1_lr0p005_ang0_s15.log
    pat = rf'lr([0-9p]+).*_{param2_token}([0-9p]+)'
    m = re.search(pat, name, flags=re.IGNORECASE)
    if not m:
        # try alternative ordering: _<param2>...lr...
        pat2 = rf'_{param2_token}([0-9p]+).*lr([0-9p]+)'
        m2 = re.search(pat2, name, flags=re.IGNORECASE)
        if not m2:
            # skip files that don't match expected naming
            continue
        lr_s = m2.group(2).replace('p','.')
        param2_s = m2.group(1).replace('p','.')
    else:
        lr_s = m.group(1).replace('p','.')
        param2_s = m.group(2).replace('p','.')

    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
    except Exception as e:
        print('Failed to read', p, e)
        continue
    parsed = parse_log_text(txt)
    # fallback: use extracted filename values if parsed empty
    if not parsed['LR']:
        parsed['LR'] = lr_s
    # train_time might be missing; leave as is
    row = (parsed['LR'], param2_s, parsed.get('loss',''), parsed.get('accuracy',''), parsed.get('train_time_s',''))
    rows.append(row)

if not rows:
    print('No matching log files found for label', label, 'and token', param2_token)
    sys.exit(1)

# write CSV sorted by LR then param2
try:
    rows_sorted = sorted(rows, key=lambda r: (float(r[0]) if r[0] else 0.0, float(r[1]) if r[1] else 0.0))
except Exception:
    rows_sorted = rows

with open(out_csv, 'w', encoding='utf-8') as f:
    # header uses uppercase parameter names for consistency
    header_param2 = param2_token.upper()
    f.write(f'LR,{header_param2},loss,accuracy,train_time_s\n')
    for r in rows_sorted:
        f.write(','.join(str(x) for x in r) + '\n')
print('Wrote', out_csv)
