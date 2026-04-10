#!/bin/bash

# --- CONFIGURATION ---
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM" 
SCRIPT_PATH="${PROJECT_ROOT}/plotting/plot_sr_comparison.py"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/sr_comparison_${TIMESTAMP}.log"

# --- HARDWARE SETTINGS ---
# Explicitly set the GPU (Good practice even with 1 GPU)
export CUDA_VISIBLE_DEVICES=0

# Force Python to flush stdout/stderr immediately so you can tail the log in real-time
export PYTHONUNBUFFERED=1

export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH

UNET_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/UNet_SR_Baseline_20260330_210715/unet_best.pth"
UNET_Mink_Ana_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/UNet_AnalyticalBaseline_20260403_184412/unet_best.pth"
UNET_Mink_Emu_CNN_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/UNet_SR_Baseline_20260330_210715/unet_best.pth"
UNET_Mink_Emu_CLIP_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/UNet_SR_Minkowski_20260403_184335/unet_best.pth"
DDPM_CKPT="/home/fquareng/work/ch2/ci26_revision_runs/sr_experiment_runs/DDPM_SR_Minkowski_20260402_094108/ddpm_best.pth"

# --- EXECUTION ---
echo "Starting training on RTX 6000..."
echo "Logs will be saved to: $LOG_FILE"

# We redirect both stdout (1) and stderr (2) to the log file
source /home/fquareng/.bashrc

# for I in {10..20}; do
#    echo "Processing index $I..."

#    # Use >> to append logs so previous iterations aren't deleted
#    micromamba run -n dl-stable python "$SCRIPT_PATH" \
#       --unet_ckpts "$UNET_CKPT" "$UNET_Mink_Ana_CKPT" "$UNET_Mink_Emu_CNN_CKPT" "$UNET_Mink_Emu_CLIP_CKPT" \
#       --ddpm_ckpt "$DDPM_CKPT" \
#       --index "$I" \
#       >> "$LOG_FILE" 2>&1

# done

micromamba run -n dl-stable python "$SCRIPT_PATH" \
      --unet_ckpts "$UNET_CKPT" "$UNET_Mink_Ana_CKPT" "$UNET_Mink_Emu_CNN_CKPT" "$UNET_Mink_Emu_CLIP_CKPT" \
      --ddpm_ckpt "$DDPM_CKPT" \
      --index 16 \
      >> "$LOG_FILE" 2>&1

echo "Plotting finished."