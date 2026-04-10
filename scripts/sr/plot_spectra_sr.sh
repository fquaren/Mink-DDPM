#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
# Ensure this matches the name you saved the new python script as:
SCRIPT_PATH="${PROJECT_ROOT}/plotting/plot_spectra_sr.py"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/spectra_comparison_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
# Explicitly set the GPU
export CUDA_VISIBLE_DEVICES=0

# Force Python to flush stdout/stderr immediately so you can tail the log in real-time
export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH

# --- CHECKPOINTS ---
UNET_BASELINE_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/UNet_SR_Baseline_20260330_210715/unet_best.pth"
UNET_ANALYTICAL_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/UNet_AnalyticalBaseline_20260330_205954/unet_best.pth"
UNET_LIPCNN_CKPT="/home/fquareng/work/ch2/sr_experiment_runs/UNet_SR_Minkowski_20260403_133415/unet_best.pth"
DDPM_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/DDPM_SR_Minkowski_20260330_205641/ddpm_best.pth"

# --- EXECUTION ---
echo "Starting full-dataset spectral analysis on RTX 6000..."
echo "Logs will be saved to: $LOG_FILE"

source /home/fquareng/.bashrc

# Run the spectra plotting script
# Note: Uncomment '--max_batches 10' to run a quick test before processing the entire dataset
micromamba run -n dl-stable python "$SCRIPT_PATH" \
      --unet_baseline "$UNET_BASELINE_CKPT" \
      --unet_analytical "$UNET_ANALYTICAL_CKPT" \
      --unet_lipcnn "$UNET_LIPCNN_CKPT" \
      --ddpm_ckpt "$DDPM_CKPT" \
      --ddim_steps 50 \
      --max_batches 10 \
      >> "$LOG_FILE" 2>&1
      

echo "Spectral analysis and plotting finished. Check $LOG_FILE for details."