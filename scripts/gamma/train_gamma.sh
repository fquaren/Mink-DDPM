#!/bin/bash

# ==============================================================================
# SCRIPT: run_emulator_ablation.sh
# DESCRIPTION: Runs 5 sequential batches. Each batch runs 3 architectures in parallel.
# ==============================================================================

export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib:$LD_LIBRARY_PATH

PROJECT_ROOT="/home/fquareng/work/ch2/Mink-DDPM"
TRAIN_SCRIPT_PATH="${PROJECT_ROOT}/src/train_gamma.py"
LOG_DIR="${PROJECT_ROOT}/logs/GammaEmulators_Ablation_$(date +%Y%m%d_%H%M%S)"
ENV_NAME="dl-stable"

mkdir -p "$LOG_DIR"

# --- ENVIRONMENT SETUP ---
eval "$(micromamba shell hook --shell bash)"
source ~/.bashrc
micromamba activate "$ENV_NAME"

ARCHS=("Constrained" "Baseline" "Lipschitz")
SEEDS=(42)  # You can add more seeds for a more robust study, e.g., (42 123 2024)

GPUS=(1 2 3)  # Assuming you have at least 3 GPUs. Adjust if you have fewer or more.

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
    for i in "${!ARCHS[@]}"; do
        arch="${ARCHS[$i]}"
        gpu_id=${GPUS[$((i % ${#GPUS[@]}))]}
        
        LOG_FILE="${LOG_DIR}/${arch}_seed${seed}.log"
        echo "   [Batch ${BATCH_NUM}] Launching: $arch on GPU $gpu_id (Seed $seed)"
        echo "   Logs will be saved to: $LOG_FILE"
        
        # Inline assignment guarantees the Python process inherits the variable
        CUDA_VISIBLE_DEVICES=$gpu_id python "$TRAIN_SCRIPT_PATH" \
            --arch "$arch" \
            --load_params "${PROJECT_ROOT}/training_params/${arch}_emulator_hp.yaml" \
            --data_fraction 0.1 \
            --seed "$seed" \
            > "$LOG_FILE" 2>&1 &
    done

    echo "   >>> Waiting for all 3 architectures to finish for Seed ${seed}..."
    wait
    
    echo "   >>> Batch ${BATCH_NUM} Complete."
    ((BATCH_NUM++))

done

echo "======================================================================"
echo "Full Ablation Study (15 Experiments) Complete."
echo "======================================================================"