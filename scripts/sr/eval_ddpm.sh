#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
SCRIPT_PATH="${PROJECT_ROOT}/eval/eval_ddpm.py"

# Specify the exact run directory here
TARGET_RUN_DIR="/home/fquareng/work/ch2/sr_experiment_runs/DDPM_SR_Geometric_20260303_105033"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/ddpm_eval_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# --- EXECUTION ---
echo "Starting evaluation for $TARGET_RUN_DIR on RTX 6000..."
echo "Logs will be saved to: $LOG_FILE"

source /home/fquareng/.bashrc

if [ ! -d "$TARGET_RUN_DIR" ]; then
    echo "Error: Directory $TARGET_RUN_DIR missing. Exiting."
    exit 1
fi

micromamba run -n dl python "$SCRIPT_PATH" --run_dir "$TARGET_RUN_DIR" > "$LOG_FILE" 2>&1

echo "Evaluation finished."