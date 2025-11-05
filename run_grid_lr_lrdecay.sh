#!/usr/bin/env bash
# Run a grid search over LR x LR_DECAY and log results using run_and_log.sh
# Explicitly named script for LR vs LR_DECAY

set -euo pipefail

# default grid (5x5 example) - edit as needed
LRS=(0.005 0.01 0.02 0.05 0.1)
DECAYS=(0.90 0.92 0.95 0.98 1)

# models to run: <script> <label>
# label is used in CSV/plot filenames (e.g. cnn, mlp)
MODELS=("mnist_convnet.py cnn" "mnist_mlp.py mlp")

# parameter names for this run (used in CSV headers and output filenames)
PARAM1_NAME="LR"
PARAM2_NAME="LR_DECAY"
PREFIX="lr_vs_lrdecay"

# runtime settings (can be overridden via env)
STEPS=${STEPS:-70}
BATCH=${BATCH:-512}
JIT=${JIT:-1}
DRY_RUN=${DRY_RUN:-0}

LOGDIR=logs
mkdir -p "$LOGDIR"

echo "Grid search: ${#LRS[@]} x ${#DECAYS[@]} = $(( ${#LRS[@]} * ${#DECAYS[@]} )) runs"
echo "STEPS=$STEPS BATCH=$BATCH JIT=$JIT DRY_RUN=$DRY_RUN"

# For each model, run the full grid and produce a per-model CSV
CSV_LIST=()
for m in "${MODELS[@]}"; do
  # split into script and label
  script=$(echo "$m" | awk '{print $1}')
  label=$(echo "$m" | awk '{print $2}')
  out_csv="results_${PREFIX}_${label}.csv"
  CSV_LIST+=("$out_csv")
  echo "Creating/clearing CSV for model $label: $out_csv"
  # header includes the parametr names (PARAM1_NAME, PARAM2_NAME)
  printf '%s\n' "${PARAM1_NAME},${PARAM2_NAME},loss,accuracy,train_time_s" > "$out_csv"

  run_idx=0
  for lr in "${LRS[@]}"; do
    for dec in "${DECAYS[@]}"; do
      run_idx=$((run_idx+1))
      # format values for filenames (replace dot with p)
      lr_f=$(echo "$lr" | tr '.' 'p')
      dec_f=$(echo "$dec" | tr '.' 'p')
      export LR="$lr"
      export LR_DECAY="$dec"
      export STEPS="$STEPS"
      export BATCH="$BATCH"
      export JIT="$JIT"

      # call the wrapper via bash to avoid execute-permission issues
      cmd="bash ./run_and_log.sh python $script"

      if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY_RUN] would run: model=$label LR=$lr LR_DECAY=$dec STEPS=$STEPS BATCH=$BATCH JIT=$JIT -> $cmd"
      else
        echo "[RUN ${label}_${run_idx}] model=$label LR=$lr LR_DECAY=$dec -> running..."
        # run the training and logging wrapper (it writes run_last.log)
        $cmd

        # move the last log to logs/ with a descriptive name
        if [ -f run_last.log ]; then
          mv run_last.log "$LOGDIR/run_${label}_${run_idx}_lr${lr_f}_dec${dec_f}_s${STEPS}.log"
        fi

        # append parsed metrics to per-model CSV using the helper script
        if [ -f scripts/append_from_log.py ]; then
          lastlog="$LOGDIR/run_${label}_${run_idx}_lr${lr_f}_dec${dec_f}_s${STEPS}.log"
          # pass extra param names so append_from_log.py can include them from env
          python3 scripts/append_from_log.py "$out_csv" "$lastlog" "${PARAM1_NAME},${PARAM2_NAME}"
        else
          echo "Warning: scripts/append_from_log.py not found - skipping CSV append"
        fi

        # small pause to free resources
        sleep 1
      fi
    done
  done
done

if [ "$DRY_RUN" != "1" ]; then
  echo "All runs finished. Regenerating combined heatmaps if available..."
  if command -v python3 >/dev/null 2>&1 && [ -f plot_combined_heatmaps.py ]; then
  # pass explicit parameter names so plotting pivots on the right CSV columns
  python3 plot_combined_heatmaps.py --csv "${CSV_LIST[@]}" --out plots --prefix "$PREFIX" --param1-name "$PARAM1_NAME" --param2-name "$PARAM2_NAME" || echo "plot generation failed (install pandas/seaborn for nicer plots)"
  echo "Combined plots saved in plots/"
  else
    echo "Plot script not available; skip plots. CSVs: ${CSV_LIST[*]}"
  fi
  echo "Logs saved in $LOGDIR and CSVs: ${CSV_LIST[*]}"
fi
