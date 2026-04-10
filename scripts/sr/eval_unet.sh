#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
SCRIPT_PATH="${PROJECT_ROOT}/eval/SR/eval_unet.py"

# Specify the exact run directory here
TARGET_RUN_DIR="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/UNet_SR_Minkowski_20260403_184335/unet_best.pth"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/unet_eval_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH

# --- EXECUTION ---
echo "Starting evaluation for $TARGET_RUN_DIR on RTX 6000..."
echo "Logs will be saved to: $LOG_FILE"

source /home/fquareng/.bashrc

micromamba run -n dl-stable python "$SCRIPT_PATH" --checkpoint "$TARGET_RUN_DIR" > "$LOG_FILE" 2>&1

echo "Evaluation finished."
