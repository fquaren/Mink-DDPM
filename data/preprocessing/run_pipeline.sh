#!/bin/bash

# --- CONFIGURATION ---
# Adjust these paths to match your local workstation directory structure
PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM"
SCRIPT_DIR="${PROJECT_ROOT}/data/preprocessing"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

# Logging setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/preprocessing_pipeline_${TIMESTAMP}.log"

# --- HARDWARE & RESOURCE SETTINGS ---
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# CRITICAL: Prevent deadlocks in parallel workers on local machines
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Local Numba cache to prevent permission/concurrent write errors
export NUMBA_CACHE_DIR="${PROJECT_ROOT}/numba_cache_${TIMESTAMP}"
mkdir -p $NUMBA_CACHE_DIR

# Define pipeline scripts
METADATA_SCRIPT="${SCRIPT_DIR}/generate_metadata.py"
CONSOLIDATE_SCRIPT="${SCRIPT_DIR}/consolidate_and_split_shuf.py"
PREPROCESS_SCRIPT="${SCRIPT_DIR}/preprocess_data.py"
GAMMA_SCRIPT="${SCRIPT_DIR}/compute_gamma_targets.py"

# --- EXECUTION ---
echo "Starting local preprocessing pipeline..."
echo "Hardware: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Logs will be saved to: $LOG_FILE"

# Source the micromamba initialization
source /home/fquareng/.bashrc

# Wrap the execution in a block to redirect all output to the log file
{
    echo "Configuration file: ${CONFIG_FILE}"
    echo "Job starting on $(hostname) at $(date)"

    echo ""
    echo "--- STAGE 1: Generating Patch Metadata and Timestamp Map ---"
    micromamba run -n dl python "${METADATA_SCRIPT}" "${CONFIG_FILE}"
    echo "--- STAGE 1 COMPLETE ---"

    echo ""
    echo "--- STAGE 2: Consolidating and Shuffling Metadata ---"
    micromamba run -n dl python "${CONSOLIDATE_SCRIPT}" "${CONFIG_FILE}"
    echo "--- STAGE 2 COMPLETE ---"

    echo ""
    echo "--- STAGE 3: Preprocessing Data into Final Zarr Store ---"
    micromamba run -n dl python "${PREPROCESS_SCRIPT}" "${CONFIG_FILE}"
    echo "--- STAGE 3 COMPLETE ---"

    echo ""
    echo "--- STAGE 4: Extracting Topological Gamma Targets ---"
    micromamba run -n dl python "${GAMMA_SCRIPT}" "${CONFIG_FILE}"
    echo "--- STAGE 4 COMPLETE ---"

    echo ""
    echo "Cleaning up local Numba cache directory..."
    rm -rf $NUMBA_CACHE_DIR

    echo "--- PIPELINE FINISHED SUCCESSFULLY at $(date) ---"
} > "$LOG_FILE" 2>&1

echo "Pipeline initiated in the background or completed. Check $LOG_FILE for details."