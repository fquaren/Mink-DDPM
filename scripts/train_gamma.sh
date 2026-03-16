#!/bin/bash

# ==============================================================================
# SCRIPT: run_emulator_ablation.sh
# DESCRIPTION: Runs 5 sequential batches. Each batch runs 3 architectures in parallel.
# ==============================================================================

PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM"
TRAIN_SCRIPT_PATH="${PROJECT_ROOT}/src/train_gamma.py"
LOG_DIR="${PROJECT_ROOT}/logs/GammaEmulators_Ablation_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl"

mkdir -p "$LOG_DIR"

# --- ENVIRONMENT SETUP ---
eval "$(micromamba shell hook --shell bash)"
source ~/.bashrc
micromamba activate "$ENV_NAME"

ARCHS=("Baseline" "Lipschitz" "Constrained") 
SEEDS=(42)

# Counter for visual progress
BATCH_NUM=1
TOTAL_BATCHES=${#SEEDS[@]}

echo "======================================================================"
echo "STARTING BATCHED ABLATION STUDY"
echo "Strategy: ${TOTAL_BATCHES} Sequential Batches x 3 Parallel Jobs"
echo "======================================================================"

# --- OUTER LOOP: SEQUENTIAL (Iterate through Seeds) ---
for seed in "${SEEDS[@]}"; do
    
    echo "----------------------------------------------------------------------"
    echo "Starting Batch ${BATCH_NUM}/${TOTAL_BATCHES} | Seed: ${seed}"
    echo "----------------------------------------------------------------------"

    # --- INNER LOOP: PARALLEL (Iterate through Architectures) ---
    for arch in "${ARCHS[@]}"; do
        
        # Log name must include seed to prevent overwriting
        LOG_FILE="${LOG_DIR}/${arch}_seed${seed}.log"
        echo "Logging to: $LOG_FILE"
        
        echo "   [Batch ${BATCH_NUM}] Launching: $arch (Seed $seed)"
        
        python "$TRAIN_SCRIPT_PATH" \
            --arch "$arch" \
            --optimize \
            --data_fraction 1 \
            --seed "$seed" \
            > "$LOG_FILE" 2>&1 &
            
        # Optional: Save PID if you need to kill specific jobs later
        # PIDS+=($!) 
    done

    echo "   >>> Waiting for all 3 architectures to finish for Seed ${seed}..."
    wait
    
    echo "   >>> Batch ${BATCH_NUM} Complete."
    ((BATCH_NUM++))

done

echo "======================================================================"
echo "Full Ablation Study (15 Experiments) Complete."
echo "======================================================================"