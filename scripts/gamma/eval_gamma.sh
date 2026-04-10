#!/bin/bash

# ==============================================================================
# SCRIPT: evaluate_all_parallel.sh
# DESCRIPTION: Evaluates emulators in parallel batches, followed by baselines.
# ==============================================================================

PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM"
EVAL_EMU_SCRIPT="${PROJECT_ROOT}/eval/gamma/eval_emu.py"
RUNS_DIR="/home/fquareng/work/ch2/ci26_revision_runs"
CONFIG_PATH="${PROJECT_ROOT}/config.yaml"
LOG_DIR="${PROJECT_ROOT}/logs/Eval_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl-stable"
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH

mkdir -p "$LOG_DIR"

source ~/.bashrc
micromamba activate "$ENV_NAME"

ARCHS=("Constrained" "Lipschitz" "Baseline") 

echo "======================================================================"
echo "STARTING PARALLEL EVALUATION"
echo "Logs saved to: $LOG_DIR"
echo "======================================================================"

# 1. Evaluate Neural Network Emulators (Parallel per Architecture)
for arch in "${ARCHS[@]}"; do
    echo "--- Launching batch for Architecture: ${arch} ---"
    
    for run_dir in "${RUNS_DIR}/GammaEmulator_${arch}"*; do
        if [ -d "$run_dir" ]; then
            run_name=$(basename "$run_dir")
            echo "Found run: $run_name"
            log_file="${LOG_DIR}/eval_${run_name}.log"
            echo "Logging to: $log_file"
            
            echo "  -> Dispatching: $run_name"
            python "$EVAL_EMU_SCRIPT" \
                --run_dir "$run_dir" \
                --arch "$arch" \
                > "$log_file" 2>&1 &
        fi
    done
    
    echo "Waiting for ${arch} evaluations to complete..."
    wait
done

echo "======================================================================"
echo "EVALUATION PIPELINE COMPLETE"
echo "======================================================================"