#!/usr/bin/env bash
# Run grid comparing LR vs SHIFT for cnn and mlp
set -euo pipefail

# default grid values
LRS=(0.005 0.01 0.02 0.05 0.1)
SHIFT=(0.1 0.2 0.5 1.0 2.0)

# models
MODELS=("mnist_convnet.py cnn" "mnist_mlp.py mlp")

# runtime settings
STEPS=${STEPS:-70}
BATCH=${BATCH:-512}
JIT=${JIT:-1}
DRY_RUN=${DRY_RUN:-0}

LOGDIR=logs
mkdir -p "$LOGDIR"

PARAM1_NAME="LR"
PARAM2_NAME="SHIFT"
PREFIX="lr_vs_shift"

CSV_LIST=()
for m in "${MODELS[@]}"; do
  script=$(echo "$m" | awk '{print $1}')
  label=$(echo "$m" | awk '{print $2}')
  out_csv="results_${PREFIX}_${label}.csv"
  CSV_LIST+=("$out_csv")
  echo "Creating/clearing CSV for model $label: $out_csv"
  printf '%s\n' "${PARAM1_NAME},${PARAM2_NAME},loss,accuracy,train_time_s" > "$out_csv"

  run_idx=0
  for lr in "${LRS[@]}"; do
    for pat in "${SHIFT[@]}"; do
      run_idx=$((run_idx+1))
      lr_f=$(echo "$lr" | tr '.' 'p')
      pat_f=$(printf "%s" "$pat")
      export LR="$lr"
      export SHIFT="$pat"
      export STEPS="$STEPS"
      export BATCH="$BATCH"
      export JIT="$JIT"

      cmd="bash ./run_and_log.sh python $script"
      if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY_RUN] would run: model=$label LR=$lr SHIFT=$pat -> $cmd"
      else
        echo "[RUN ${label}_${run_idx}] model=$label LR=$lr SHIFT=$pat -> running..."
        $cmd
        if [ -f run_last.log ]; then
          # include the parameter name in the filename so downstream parsing can find it (e.g. _shift5)
          mv run_last.log "$LOGDIR/run_${label}_${run_idx}_lr${lr_f}_${PARAM2_NAME,,}${pat_f}_s${STEPS}.log"
        fi
        if [ -f scripts/append_from_log.py ]; then
          lastlog="$LOGDIR/run_${label}_${run_idx}_lr${lr_f}_${PARAM2_NAME,,}${pat_f}_s${STEPS}.log"
          python3 scripts/append_from_log.py "$out_csv" "$lastlog" "${PARAM1_NAME},${PARAM2_NAME}"
        else
          echo "Warning: scripts/append_from_log.py not found - skipping CSV append"
        fi
        sleep 1
      fi
    done
  done
done

if [ "$DRY_RUN" != "1" ]; then
  if command -v python3 >/dev/null 2>&1 && [ -f plot_combined_heatmaps.py ]; then
    # pass explicit parameter names so the plot script pivots on the correct columns
    python3 plot_combined_heatmaps.py --csv "${CSV_LIST[@]}" --out plots --prefix "$PREFIX" --param1-name "$PARAM1_NAME" --param2-name "$PARAM2_NAME" || echo "plot generation failed"
    echo "Combined plots saved in plots/"
  else
    echo "Plot script not available; skip plots. CSVs: ${CSV_LIST[*]}"
  fi
  echo "Logs saved in $LOGDIR and CSVs: ${CSV_LIST[*]}"
fi
