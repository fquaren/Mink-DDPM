#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
SCRIPT_PATH="${PROJECT_ROOT}/eval/SR/eval_bicubic.py"

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

micromamba run -n dl-stable python "$SCRIPT_PATH" > "$LOG_FILE" 2>&1

echo "Evaluation finished."