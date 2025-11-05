
# MNIST TinyGrad → WebGPU Demo

A small interactive demo that shows handwritten digit classification (MNIST) in the browser using models trained with tinygrad and exported for WebGPU.

Live demo: (no public demo link provided)  
(Run locally — instructions below.)

<img width="2559" height="1243" alt="image" src="https://github.com/user-attachments/assets/a5632a22-989c-44d6-bccc-0a2fb13a4722" />

## Overview

This repository contains:

- Python training and export scripts using tinygrad (`mnist_mlp.py`, `mnist_convnet.py`).
- Exported model artifacts and a small single-page web app that runs inference with WebGPU (`docs/` served by GitHub Pages).

Two models are provided:
- MLP (multi-layer perceptron) trained on MNIST
- CNN (convolutional neural network) trained on MNIST

Both models are exported to a small JavaScript loader (`*.js`) and weight bundles (`*.safetensors`) so inference runs entirely in the browser using WebGPU.

## Features

- Draw digits on a canvas (pen + clear).
- Switch between CNN and MLP models.
- Real-time inference (inputs are resized to 28x28 and normalized).
- Confidence visualization: a 10-bar probability chart (softmax output).
- Lightweight WebGPU-based inference using exported tinygrad artifacts.

## Model Summary

| Model | Notes |
|---|---|
| MLP | Small fully-connected network exported from `mnist_mlp.py` (good baseline) |
| CNN | Small convnet exported from `mnist_convnet.py` (higher accuracy target) |

See `HYPERPARAMETERS.md` for training runs, hyperparameters explored, and accuracy logs.

## How the models and training work

High-level flow:

- Training scripts (`mnist_mlp.py` and `mnist_convnet.py`) load the MNIST dataset and apply lightweight on-the-fly data augmentation (random rotation, scale and shift) using a small `geometric_transform` helper.
- Inputs are normalized to the range [-1, 1] before being fed to the model.
- The MLP is a simple fully-connected network (784 -> 512 -> 512 -> 10) with SiLU activations. The CNN uses a small convnet (Conv → SiLU → Conv → SiLU → BatchNorm → MaxPool → Conv… → linear) designed as a compact but accurate feature extractor.
- Training uses tinygrad's `Muon` optimizer and a simple learning-rate schedule: if validation/test accuracy does not improve for `PATIENCE` steps, the LR is multiplied by `LR_DECAY` and the best-known weights are reloaded.
- After training, the best model state is exported using the repository's `export_model` helper, producing a `*.js` loader and `*.webgpu.safetensors` weight file which the web app consumes.

Why augmentation matters here:
- The random `ANGLE`, `SCALE` and `SHIFT` augmentations improve robustness to hand-drawn digits that are rotated, resized or slightly displaced. You can control their ranges with environment variables (see next section).

## Key hyperparameters (how to change them)

All training scripts read configuration from environment variables. Typical variables you can set on the command line before running the Python scripts:

- `STEPS` — number of training iterations (default in scripts: 70). Example: `STEPS=100`.
- `BATCH` — batch size used during training (default: 512).
- `LR` — initial learning rate (e.g. `0.005`, `0.01`, `0.02`).
- `LR_DECAY` — multiplicative factor applied to the learning rate when no improvement is seen after `PATIENCE` steps (default: `0.9`).
- `PATIENCE` — number of iterations to wait without improvement before decaying LR (default: `50`).
- `ANGLE` — maximum rotation (degrees) used for random augmentation (e.g. `5`, `15`).
- `SCALE` — relative scale jitter (e.g. `0.1` for ±10%).
- `SHIFT` — pixel shift magnitude used for translation augmentation (relative units used in the scripts).
- `JIT` — enable/disable tinygrad JIT where applicable (used in run scripts).
- `SAMPLING` — sampling mode for geometric transforms (nearest or bilinear).

Example: run a training session with a lower LR and more steps:

```bash
STEPS=100 LR=0.01 LR_DECAY=0.95 PATIENCE=20 BATCH=256 JIT=1 python3 mnist_convnet.py
```

The scripts save the best model weights into `mnist_convnet/mnist_convnet.safetensors` (or the MLP folder equivalent) and then export `mnist_convnet.js` + `mnist_convnet.webgpu.safetensors` for the web app.

## Automated experiments and parameter sweeps

I included several helper bash scripts to automate grid searches across hyperparameters. They set environment variables, run the training script, and move each run log into `logs/` so you can compare runs easily.

Examples of the scripts in the repo:

- `run_grid_lr_angle.sh` — sweeps learning rate vs rotation angle.
- `run_grid_lr_shift.sh` — sweeps learning rate vs translation/shift.
- `run_grid_lr_scale.sh`, `run_grid_lr_batch.sh`, `run_grid_lr_lrdecay.sh`, `run_grid_lr_patience.sh` — similar experiments for other parameters.

Usage example (from repo root):

```bash
# run a grid comparing LR and ANGLE (set STEPS/BATCH/JIT if you want)
STEPS=50 BATCH=512 JIT=1 bash run_grid_lr_angle.sh
```

What the scripts do:

- They iterate over parameter lists (defined in the script) and export the chosen env vars for each run.
- Each run writes a `run_last.log` then the script renames and moves it to `logs/` with a filename that encodes the parameters (e.g. `run_cnn_1_lr0p005_ang5_s50.log`).

Post-processing:
- Use `scripts/append_from_log.py` or `scripts/rebuild_csv_from_logs.py` to rebuild or append results into the `results_*.csv` files present in the repo. These CSVs are used to generate the plots in `plots/`.

## Inspecting results and plots

- CSV results files are stored at the repo root, for example `results_lr_vs_angle_cnn.csv` and `results_lr_vs_angle_mlp.csv`.
- A set of plots has been generated and placed in `plots/` (e.g. `lr_vs_angle_accuracy_combined.png`). If you re-run sweeps, regenerate the CSVs with `scripts/rebuild_csv_from_logs.py` and re-run `plot_all_heatmaps.py` or `plot_combined_heatmaps.py` to update the images.

## Tips for hyperparameter tuning

- Start with a small sweep over `LR` (e.g. 0.005, 0.01, 0.02) and a moderate `STEPS` (50–100). Observe the validation curve in the logs.
- If the accuracy plateaus early, decrease `LR` or increase `PATIENCE` (so LR decay happens less often).
- If training is unstable (loss jumps), lower `LR` and/or reduce batch size.
- Use the augmentation parameters (`ANGLE`, `SCALE`, `SHIFT`) to test robustness; larger augmentation ranges increase robustness but can make convergence slower.

## Where logs and artifacts are saved

- `logs/` — per-run textual logs. Filenames encode model, LR and other parameters.
- `mnist_convnet/`, `mnist_mlp/` — model artifacts, exported JS and safetensors.
- `results_*.csv` and `plots/` — aggregated results and visualizations from sweep runs.

## Quick start — Run locally

1. Clone the repository:

```bash
git clone https://github.com/fab-pie/mnist_mlp_cnn.git
cd mnist_mlp_cnn
```

2. Serve the project locally (WebGPU requires HTTP):

```bash
python3 -m http.server 8000
# then open http://localhost:8000/docs/ in a WebGPU-capable browser
```

Note: Modern Chrome/Chromium-based browsers have experimental WebGPU support. You may need to enable the WebGPU flag or run an up-to-date browser with WebGPU enabled.

## Train & export (if you want to re-create models)

These scripts use tinygrad and the project's export utilities. Example commands used during development:

```bash
# Train and export the MLP (example, 100 steps)
STEPS=100 JIT=1 python3 mnist_mlp.py

# Train and export the CNN
STEPS=100 JIT=1 python3 mnist_convnet.py
```

Each command creates a folder (`mnist_mlp/` or `mnist_convnet/`) with the exported JavaScript loader and `.safetensors` weight files.

## Serving the web app locally

The repository contains a single-page web app and exported model artifacts under `docs/` (and `Webapp/` in some branches). To run the app locally do:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/docs/ in a WebGPU-capable browser
```

Note: WebGPU support varies by browser. Use an up-to-date Chromium build with WebGPU enabled.

## Notes on large files and versioning

Model weight files (`*.safetensors`) are large. Options:

- Keep in repo (simple, but increases repo size).
- Use Git LFS for large binaries.
- Host weights externally (e.g., cloud storage or GitHub Releases) and fetch them at runtime.

If you want, I can add a minimal `.gitignore` and a Git LFS setup to keep the repository lightweight.

## Troubleshooting

- If the web app shows "Loading models..." for a long time, open the DevTools Console to see error messages (failed fetches, CORS, or WebGPU adapter errors).
- If modules fail to load, verify the relative paths used by `docs/main.js` and that the corresponding `.js` and `.safetensors` files are present next to it.

## Files of interest

- `mnist_mlp.py`, `mnist_convnet.py` — training & export scripts (tinygrad)
- `docs/index.html`, `docs/main.js` — web application and loader
- `HYPERPARAMETERS.md` — hyperparameter experiments and accuracy logs

