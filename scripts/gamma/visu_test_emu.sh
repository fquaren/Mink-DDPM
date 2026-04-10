#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
SCRIPT_PATH="${PROJECT_ROOT}/plotting/plot_emulators_tests.py"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/emulators_test_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
# Explicitly set the GPU (Good practice even with 1 GPU)
export CUDA_VISIBLE_DEVICES=0

# Force Python to flush stdout/stderr immediately so you can tail the log in real-time
export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH


BASELINE_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/GammaEmulator_Baseline_SingleRun_2026-04-02_10-39-14/best_model_checkpoint.pth"
LIPSCHITZ_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/GammaEmulator_Lipschitz_SingleRun_2026-04-02_10-39-14/best_model_checkpoint.pth"
CONSTRAINED_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/GammaEmulator_Constrained_SingleRun_2026-04-02_10-39-14/best_model_checkpoint.pth"


# --- EXECUTION ---
echo "Starting training on RTX 6000..."
echo "Logs will be saved to: $LOG_FILE"

# We redirect both stdout (1) and stderr (2) to the log file
source /home/fquareng/.bashrc
micromamba run -n dl-stable python "$SCRIPT_PATH" \
   --ckpt_baseline "$BASELINE_CKPT" \
   --ckpt_lipschitz "$LIPSCHITZ_CKPT" \
   --ckpt_constrained "$CONSTRAINED_CKPT" \
   > "$LOG_FILE" 2>&1

echo "Training finished."