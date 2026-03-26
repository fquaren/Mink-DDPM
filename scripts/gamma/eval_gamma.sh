#!/bin/bash

# ==============================================================================
# SCRIPT: evaluate_all_parallel.sh
# DESCRIPTION: Evaluates emulators in parallel batches, followed by baselines.
# ==============================================================================

PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM"
EVAL_EMU_SCRIPT="${PROJECT_ROOT}/src/eval.py"
EVAL_BASE_SCRIPT="${PROJECT_ROOT}/src/evaluate_baselines.py"
RUNS_DIR="${PROJECT_ROOT}/final_experiment_runs"
CONFIG_PATH="${PROJECT_ROOT}/config.yaml"
LOG_DIR="${PROJECT_ROOT}/logs/Eval_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl-stable"

mkdir -p "$LOG_DIR"

eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

ARCHS=("Baseline" "Lipschitz" "Constrained")

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
            log_file="${LOG_DIR}/eval_${run_name}.log"
            
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

# 2. Evaluate Baselines
echo "--- Evaluating Baselines (PCR & Analytical) ---"
log_file="${LOG_DIR}/eval_baselines.log"
python "$EVAL_BASE_SCRIPT" "$CONFIG_PATH" > "$log_file" 2>&1

echo "======================================================================"
echo "EVALUATION PIPELINE COMPLETE"
echo "======================================================================"