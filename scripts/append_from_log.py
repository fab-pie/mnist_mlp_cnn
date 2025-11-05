#!/usr/bin/env python3
"""
Parse a run log and append LR,LR_DECAY,loss,accuracy,train_time_s to a CSV.
Usage: append_from_log.py <out_csv> <log_file>
If log_file is omitted, uses ./run_last.log
"""
import re
import sys
import os
from datetime import timedelta


def parse_log_text(txt):
    # defaults from env if available
    lr = os.environ.get('LR','')
    decay = os.environ.get('LR_DECAY','')

    # explicit occurrences
    m = re.search(r'LR[:= ]+([0-9.eE+-]+)', txt)
    if m:
        lr = m.group(1)
    m = re.search(r'LR[_ ]?DECAY[:= ]+([0-9.eE+-]+)', txt)
    if m:
        decay = m.group(1)

    # last loss
    loss_m = re.findall(r'loss[:= ]+([0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?)', txt)
    loss = loss_m[-1] if loss_m else ''

    # last accuracy
    acc_m = re.findall(r'(?:accuracy|acc)[:= ]+([0-9]+\.?[0-9]*)%?', txt, flags=re.IGNORECASE)
    accuracy = acc_m[-1] if acc_m else ''

    # time extraction
    time_s = ''
    # explicit train time like 'train time: 12.34s'
    t_m = re.search(r'train[_ ]?time[:= ]+([0-9]+\.?[0-9]*)s', txt, flags=re.IGNORECASE)
    if t_m:
        time_s = t_m.group(1)
    else:
        # find last loss position and search after it for MM:SS or HH:MM:SS
        loss_iter = list(re.finditer(r'loss[:= ]+[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?', txt))
        search_region = txt
        if loss_iter:
            last_loss_pos = loss_iter[-1].end()
            search_region = txt[last_loss_pos:]
            m2 = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?)', search_region)
            if m2:
                tstr = m2.group(1)
                parts = tstr.split(':')
                try:
                    if len(parts) == 3:
                        h,m,s = map(int,parts); total = h*3600 + m*60 + s
                    else:
                        m,s = map(int,parts); total = m*60 + s
                    time_s = str(total)
                except Exception:
                    time_s = ''
        # fallback: global bracket search like '[00:21<00:00, ...]'
        if not time_s:
            br = re.findall(r'\[\s*(\d{1,2}:\d{2}(?::\d{2})?)', txt)
            if br:
                tstr = br[-1]
                parts = tstr.split(':')
                try:
                    if len(parts) == 3:
                        h,m,s = map(int,parts); total = h*3600 + m*60 + s
                    else:
                        m,s = map(int,parts); total = m*60 + s
                    time_s = str(total)
                except Exception:
                    time_s = ''

    return dict(LR=lr, LR_DECAY=decay, loss=loss, accuracy=accuracy, train_time_s=time_s)


def append_row(out_csv, row, extra_params=None):
    """
    Append a row to out_csv. If extra_params is provided (comma-separated names), include them as columns
    and try to take their values from environment variables (fall back to empty string).
    """
    base_cols = ['LR', 'LR_DECAY', 'loss', 'accuracy', 'train_time_s']
    extra = []
    if extra_params:
        extra = [p.strip() for p in extra_params.split(',') if p.strip()]

    # If the CSV already exists and has a header, honor that header order to avoid misalignment.
    header_cols = None
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        try:
            with open(out_csv, 'r', encoding='utf-8') as f:
                first = f.readline().strip()
                if first:
                    header_cols = [h.strip() for h in first.split(',') if h.strip()]
        except Exception:
            header_cols = None

    # If no existing header, build one: extras first, then base columns that are not duplicated
    if header_cols is None or len(header_cols) == 0:
        remaining_base = [c for c in base_cols if c not in extra]
        header_cols = extra + remaining_base

    # write row according to header_cols order
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, 'a', encoding='utf-8') as f:
        if write_header:
            f.write(','.join(header_cols) + '\n')
        vals = []
        for col in header_cols:
            # extras are expected to be provided via env variables
            if col in extra:
                vals.append(os.environ.get(col, ''))
            elif col in base_cols:
                vals.append(row.get(col, '') or '')
            else:
                # unknown column: try env then row then empty
                vals.append(os.environ.get(col, '') or row.get(col, '') or '')
        line = ','.join(str(v) for v in vals) + '\n'
        f.write(line)


def main():
    if len(sys.argv) < 2:
        print('Usage: append_from_log.py <out_csv> [log_file] [extra_params_comma_sep]')
        sys.exit(2)
    out_csv = sys.argv[1]
    log_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].count(',') else (os.path.join(os.getcwd(), 'run_last.log') if len(sys.argv) < 3 else os.path.join(os.getcwd(), 'run_last.log'))
    # optional third arg: comma-separated names of extra params to include (read from env)
    extra_params = None
    if len(sys.argv) >= 3:
        # if third arg looks like a comma-separated list (contains comma) assume it's extra params
        if sys.argv[2].count(','):
            extra_params = sys.argv[2]
        elif len(sys.argv) >= 4:
            extra_params = sys.argv[3]
    
    # normalize log_file if provided as second arg and not an extra-param list
    if len(sys.argv) >= 3 and os.path.exists(sys.argv[2]):
        log_file = sys.argv[2]
    if not os.path.exists(log_file):
        print('Log file not found:', log_file)
        sys.exit(1)
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    row = parse_log_text(txt)
    append_row(out_csv, row, extra_params=extra_params)
    print('Appended row to', out_csv)


if __name__ == '__main__':
    main()
