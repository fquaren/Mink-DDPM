#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
SCRIPT_PATH="${PROJECT_ROOT}/eval/emulator_statistical_baselines.py"
CONFIG_PATH="${PROJECT_ROOT}/config.yaml"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/emulator_baselines_run_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
# Explicitly set the GPU (Good practice even with 1 GPU)
export CUDA_VISIBLE_DEVICES=0

# Force Python to flush stdout/stderr immediately so you can tail the log in real-time
export PYTHONUNBUFFERED=1

# --- EXECUTION ---
echo "Starting training on RTX 6000..."
echo "Logs will be saved to: $LOG_FILE"

# We redirect both stdout (1) and stderr (2) to the log file
source /home/fquareng/.bashrc
micromamba run -n dl python "$SCRIPT_PATH" "$CONFIG_PATH" > "$LOG_FILE" 2>&1

echo "Training finished."