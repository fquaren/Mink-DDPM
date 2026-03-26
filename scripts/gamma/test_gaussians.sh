#!/bin/bash

# ==============================================================================
# SCRIPT: run_synthetic_test.sh
# DESCRIPTION: Evaluates trained emulators on a synthetic multi-peak Gaussian field.
# ==============================================================================

PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM"
# Ensure this matches the name you gave to the Python script generated previously
TEST_SCRIPT="${PROJECT_ROOT}/src/test_synthetic_gaussian.py" 
RUNS_DIR="${PROJECT_ROOT}/final_experiment_runs"
SAVE_DIR="${PROJECT_ROOT}/eval_results/synthetic_gaussian"
LOG_DIR="${PROJECT_ROOT}/logs/GaussianTest_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl-stable"

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$ENV_NAME"

echo "======================================================================"
echo "STARTING SYNTHETIC GAUSSIAN EVALUATION"
echo "Logs saved to: $LOG_DIR"
echo "Plots will be saved to: $SAVE_DIR"
echo "======================================================================"

# Find the most recent run directory for each architecture
# BASE_DIR=$(ls -td "${RUNS_DIR}/GammaEmulator_Baseline_"* 2>/dev/null | head -n 1)
# LIP_DIR=$(ls -td "${RUNS_DIR}/GammaEmulator_Lipschitz_"* 2>/dev/null | head -n 1)
# CONS_DIR=$(ls -td "${RUNS_DIR}/GammaEmulator_Constrained_"* 2>/dev/null | head -n 1)

BASE_DIR=""
LIP_DIR=""
CONS_DIR=""

echo "Detected Runs:"
echo "  -> Baseline:    ${BASE_DIR:-[NOT FOUND]}"
echo "  -> Lipschitz:   ${LIP_DIR:-[NOT FOUND]}"
echo "  -> Constrained: ${CONS_DIR:-[NOT FOUND]}"
echo "----------------------------------------------------------------------"

LOG_FILE="${LOG_DIR}/eval_synthetic_gaussian.log"

# Construct the Python command dynamically based on available runs
CMD="python \"$TEST_SCRIPT\" --save_dir \"$SAVE_DIR\""

if [ -n "$BASE_DIR" ]; then
    CMD="$CMD --baseline_dir \"$BASE_DIR\""
fi

if [ -n "$LIP_DIR" ]; then
    CMD="$CMD --lipschitz_dir \"$LIP_DIR\""
fi

if [ -n "$CONS_DIR" ]; then
    CMD="$CMD --constrained_dir \"$CONS_DIR\""
fi

echo "Executing test script..."
echo "Command: $CMD"

# Run the command and pipe output to the log file
eval "$CMD" > "$LOG_FILE" 2>&1

echo "======================================================================"
echo "EVALUATION COMPLETE"
echo "Check the log file for details: $LOG_FILE"
echo "======================================================================"