Hyperparameter experiments — summary and reproducible notes

This document summarises the automated hyperparameter experiments, the scripts used for grid sweeps, selected results for the MLP and CNN models, and practical recommendations for reproducing and extending the sweeps.

Goals
------
- Automate pairwise hyperparameter sweeps (for example: learning rate vs augmentation amount) and visualise the results as heatmaps.
- Focus on learning rate (`LR`) while comparing it to other influential hyperparameters (ANGLE, SCALE, SHIFT, BATCH, PATIENCE, LR_DECAY) to identify stable, high-accuracy regions.

Where data lives
-----------------
- Per-run logs: `logs/` (sweep scripts write `run_last.log` then move it into `logs/` with parameter-encoded filenames).
- Aggregated CSVs: `results_*.csv` at the repository root (e.g. `results_lr_vs_angle_cnn.csv`).
- Plots: `plots/` — generated heatmaps and combined plots.
- Helper scripts: `scripts/rebuild_csv_from_logs.py`, `scripts/append_from_log.py`, and plotting scripts `plot_all_heatmaps.py` / `plot_combined_heatmaps.py`.

How automation works
---------------------
- The bash helpers (e.g. `run_grid_lr_angle.sh`, `run_grid_lr_shift.sh`, ...) iterate over two parameter lists, export the chosen variables, then call the appropriate training script (`mnist_convnet.py` or `mnist_mlp.py`).
- Each run writes textual output to `run_last.log`; the wrapper renames/moves that file into `logs/` with an informative name (model, run index, LR, other param, STEPS).
- After a sweep you can rebuild CSVs from the logs and re-generate heatmaps with the plotting scripts.

Automation note
---------------
Grid runs can be slow. On my machine a full grid with `STEPS=70` typically took ~2 hours; so for exploratory sweeps I used smaller `STEPS` (e.g. `STEPS=50`) to get faster feedback. Use larger `STEPS` (e.g. 200) for final runs once you have a promising hyperparameter band.

## ⚠️ IMPORTANT — previous LR-based results are invalid

IMPORTANT: I discovered that automated sweep runs that used LR as an axis produced incorrect results — some sweep wrappers did not actually pass the `LR` value through to the training process. In short: heatmaps or CSVs where LR is a dimension may not reflect true LR variations.

Good news: the launcher scripts have been fixed for future runs. New sweeps will correctly use the `LR` value passed from the shell.

What to do now
- Treat existing CSVs/heatmaps that depend on LR as unreliable and avoid making decisions from them.
- To verify the fix and produce valid LR-based results, run a normal grid sweep (do not use `STEPS=5`). For example, run a standard exploratory grid with `STEPS=50` and then rebuild CSVs and plots:

```bash
# run a normal exploratory grid (STEPS=50)
STEPS=50 JIT=1 bash run_grid_lr_angle.sh

# after the sweep completes, rebuild CSVs and plots
python3 scripts/rebuild_csv_from_logs.py
python3 plot_all_heatmaps.py
python3 plot_combined_heatmaps.py
```

Quick log check
- After a job finishes, inspect the most recent log to confirm the printed lr value (example):

```bash
# find the most recent log and grep for the lr line
LATEST=$(ls -t logs | head -n1)
grep -i "lr:" logs/$LATEST || tail -n 50 logs/$LATEST
```

Recommended remediation steps
1. Re-run a validation grid (STEPS=50) with LR as an axis and confirm logs show different LR values.
2. If validation is OK, re-run full sweeps and regenerate CSVs/plots.
3. Mark previous LR-dependent results as deprecated in any reports or notes.


Reproducing a grid sweep (example)
----------------------------------
From the repository root:

```bash
# example: exploratory sweep LR vs ANGLE
STEPS=50 JIT=1 bash run_grid_lr_angle.sh

# after the sweep finishes
python3 scripts/rebuild_csv_from_logs.py
python3 plot_all_heatmaps.py
```

Key runs & results (selected runs from your logs)
-------------------------------------------------
Below are the best representative runs you recorded and their important metrics. These are taken from the logs you saved during development.

MLP (selected)
* JIT=1 STEPS=200 ANGLE=5 SCALE=0.2
  - Example best LR: 0.02 → loss 0.17, accuracy 97.19% (200 steps)
* JIT=1 STEPS=200 ANGLE=5 SCALE=0.2 LR=0.005
  - LR 0.005 → loss 0.63, accuracy 88.38% (same setup except lower LR)

Observation: the MLP is sensitive to the learning rate in these settings — too small LR (or excessive decay) reduces final accuracy. Increasing steps (e.g. 200) helps reach higher accuracy when using a correctly chosen LR.

CNN (selected)
* BATCH=512 LR=0.015 LR_DECAY=0.92 PATIENCE=40 JIT=1 STEPS=200 ANGLE=18 SCALE=0.12 SHIFT=3.0
  - Effective LR observed: ~0.0138 → loss 0.09, accuracy 98.76% (200 steps)
* BATCH=512 JIT=1 STEPS=70 ANGLE=3 SCALE=0.12 SHIFT=0.2
  - LR 0.02 → loss 0.10, accuracy 98.24% (70 steps)
* JIT=1 STEPS=200 LR_DECAY=0.88 LR=0.01 ANGLE=5 SCALE=0.2
  - Effective LR after decay: ~0.0088 → loss 0.04, accuracy 99.02% (200 steps)

Observation: the CNN is robust and reaches >98% in several configurations. LR scheduling (decay on plateau) + augmentation (ANGLE/SCALE/SHIFT) produce stable high-accuracy results. Larger batch sizes (512) were used in the high-accuracy runs.

Patterns and practical recommendations
-------------------------------------
- Learning rate (LR):
  - MLP: relatively sensitive. Values ~0.02 performed much better than ~0.005 in your experiments. If you lower LR, consider increasing STEPS substantially or reducing LR_DECAY aggressiveness.
  - CNN: starting LR around 0.01–0.02 with a decay policy (LR_DECAY ~ 0.88–0.95 and PATIENCE ~ 20–40) worked well.
- Augmentation (ANGLE / SCALE / SHIFT):
  - Increasing ANGLE improves robustness but too high values (e.g. >15°) may distort digits. ANGLE around 3–10 produced good trade-offs depending on the run.
  - SCALE and SHIFT also improve robustness; moderate values (SCALE ~ 0.05–0.2, SHIFT small percentage or small pixels) are good starting points.
- Batch size: 512 was used successfully for CNN. Large batches speed up training per epoch but can interact with LR; if you use a very large batch, lowering LR slightly may help.
- STEPS: for best results, run longer (200 steps gave higher final accuracy for both models). The default 70 steps is a quicker compromise for sweeps; for automated exploratory grids we used smaller STEPS to save time (see note above).
- PATIENCE & LR_DECAY: use patience large enough to let the model improve (20–40). When no improvement is seen, multiplying LR by LR_DECAY (0.88–0.95) produced steady improvements in the CNN experiments.

Interpolation & activation experiments
--------------------------------------
- Interpolation (bilinear vs nearest):
  - I tested both `bilinear` and `nearest` resampling modes for geometric transforms / augmentations.
  - In general, `bilinear` produced slightly better final accuracy (smoother transformed images -> more realistic augmentation), but it is noticeably slower at runtime.
  - `nearest` is faster and suitable for quick exploratory sweeps (grid runs).
  - `bilinear` tends to yield slightly better accuracy (smoother images) but has a higher runtime cost; use it for final, higher-quality runs when you prioritise accuracy over speed.

- Activation functions (tanh, sigmoid, etc.):
  - I briefly tried other activations (hyperbolic tangent, sigmoid) instead of the default (SiLU / ReLU-style activations used in the code).
  - Results were inconsistent — sometimes accuracy dropped, sometimes it stayed similar; I didn't observe a clear, repeatable pattern in these small trials.
  - Recommendation: to evaluate activation functions reliably, run controlled ablation studies (fix seeds and repeat each configuration multiple times) to reduce noise and measure statistical significance.

Tips for interpreting heatmaps
----------------------------
- Heatmaps (LR vs other param) show regions where accuracy is high and where it fails. Use them to find a stable LR band rather than a single LR value.
- If a heatmap shows accuracy drops when PATIENCE increases, it can indicate that too-slow decay prevented escaping local minima or that the effective LR became too small during training.

Re-running the best experiments (commands)
-----------------------------------------
To reproduce the best CNN run above:

```bash
BATCH=512 LR=0.015 LR_DECAY=0.92 PATIENCE=40 JIT=1 STEPS=200 ANGLE=3 SCALE=0.12 SHIFT=0.05 python3 mnist_convnet.py
```

To reproduce the best MLP run observed:

```bash
JIT=1 STEPS=200 LR=0.02 ANGLE=5 SCALE=0.2 python3 mnist_mlp.py
```

Regenerating CSVs and plots after running sweeps
------------------------------------------------
1. Collect logs (the sweep scripts rename the logs into `logs/`).
2. Rebuild CSV files from logs:

```bash
python3 scripts/rebuild_csv_from_logs.py
```

3. Recreate heatmaps:

```bash
python3 plot_all_heatmaps.py
python3 plot_combined_heatmaps.py
```