#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
SCRIPT_PATH="${PROJECT_ROOT}/src/train_unet_ana.py"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/unet_run_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
# Explicitly set the GPU (Good practice even with 1 GPU)
export CUDA_VISIBLE_DEVICES=0

# Force Python to flush stdout/stderr immediately so you can tail the log in real-time
export PYTHONUNBUFFERED=1

# Set the library path for the conda environment
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH

# --- EXECUTION ---
# echo "Starting training on RTX 6000..."
echo "Logs will be saved to: $LOG_FILE"

# We redirect both stdout (1) and stderr (2) to the log file
source /home/fquareng/.bashrc
micromamba run -n dl-stable python "$SCRIPT_PATH" \
    --data_percentage 25.0 \
    --weight_geom 0.001 \
    --params_path "/home/fquareng/work/ch2/Mink-DDPM/unet_ana_params.yaml" >> "$LOG_FILE" 2>&1

echo "Training finished."