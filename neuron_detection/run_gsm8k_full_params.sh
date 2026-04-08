#!/bin/bash

# Full-Params: GSM8K Downstream Finetuning (No Safety Neuron Freeze)
#
# Experimental setup:
#   - Full parameter finetuning on GSM8K
#   - LR sweep: 1e-5, 3e-5, 5e-5
#
# Matched hyperparameters to WaRP Phase 3 / freeze_sn script:
#   - epochs: 3
#   - batch_size x grad_accum = 4 x 4 = 16 (effective)
#   - weight_decay: 0.01
#   - warmup_ratio: 0.1
#   - lr_scheduler: linear
#   - max_length: 1024
#   - num_train_samples: 7473 (all GSM8K train)

set -e

# Run on physical GPU index 1 only
export CUDA_VISIBLE_DEVICES=0

echo "========================================================================"
echo "Full-Params: GSM8K Downstream Finetuning (No Freeze)"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

# SN/RSN tuned model path (or HF repo id)
MODEL_PATH="meta-llama/Llama-3.2-3B"

# LR sweep
LR_LIST=("1e-5" "3e-5" "5e-5")

# Matched hyperparameters
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4               # effective batch = 4 x 4 = 16
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
LR_SCHEDULER="cosine"
OPTIMIZER="adamw_torch"
MAX_GRAD_NORM=1.0
MAX_LENGTH=1024
NUM_TRAIN_SAMPLES=7473     # all GSM8K train set

BASE_OUTPUT_DIR="./llama3.2_3b_base_gsm8k_ft_full_params"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/gsm8k_full_params"
mkdir -p "$LOG_DIR"

if [ ! -d "$MODEL_PATH" ] && [[ "$MODEL_PATH" != kmseong/* ]] && [[ "$MODEL_PATH" != meta-llama/* ]]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

echo "Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (physical GPU 0)"
echo "  Model:               $MODEL_PATH"
echo "  LR sweep:            ${LR_LIST[*]}"
echo "  Epochs:              $EPOCHS"
echo "  Batch x AccumSteps:  $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM)) (effective)"
echo "  Weight decay:        $WEIGHT_DECAY"
echo "  Warmup ratio:        $WARMUP_RATIO"
echo "  LR scheduler:        $LR_SCHEDULER"
echo "  Optimizer:           $OPTIMIZER"
echo "  Max grad norm:       $MAX_GRAD_NORM"
echo "  Max length:          $MAX_LENGTH"
echo "  Num train samples:   $NUM_TRAIN_SAMPLES"
echo ""

OUTPUT_DIRS=()

for LEARNING_RATE in "${LR_LIST[@]}"; do
    LR_SAFE=$(echo "$LEARNING_RATE" | sed 's/[^a-zA-Z0-9_-]/_/g')
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_lr${LR_SAFE}_${TIMESTAMP}"
    LOG_FILE="${LOG_DIR}/gsm8k_full_params_lr${LR_SAFE}_${TIMESTAMP}.log"

    echo "----------------------------------------------------------------------"
    echo "  LR = $LEARNING_RATE"
    echo "----------------------------------------------------------------------"

    python finetune_gsm8k_full_params.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --learning_rate "$LEARNING_RATE" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --grad_accum $GRAD_ACCUM \
        --weight_decay $WEIGHT_DECAY \
        --warmup_ratio $WARMUP_RATIO \
        --lr_scheduler_type $LR_SCHEDULER \
        --max_grad_norm $MAX_GRAD_NORM \
        --max_length $MAX_LENGTH \
        --num_train_samples $NUM_TRAIN_SAMPLES \
        --num_eval_samples 0 \
        --bf16 \
        --seed 42 \
        2>&1 | tee "$LOG_FILE"

    if [ -d "$OUTPUT_DIR" ]; then
        echo "OK: LR=$LEARNING_RATE completed: $OUTPUT_DIR"
        OUTPUT_DIRS+=("$OUTPUT_DIR")
    else
        echo "WARNING: Output directory not found for LR=$LEARNING_RATE"
    fi

    echo ""
done

# ========================================================================
# Summary
# ========================================================================
echo "========================================================================"
echo "Full-Params GSM8K Finetuning Complete"
echo "========================================================================"
echo ""
echo "Output Models:"
for dir in "${OUTPUT_DIRS[@]}"; do
    echo "  - $dir"
done
echo ""
echo "Log Files: $LOG_DIR/gsm8k_full_params_lr*_${TIMESTAMP}.log"
echo ""
echo "Training Configuration Summary:"
echo "  - Model:           $MODEL_PATH"
echo "  - Learning Rates:  ${LR_LIST[*]}"
echo "  - Epochs:          $EPOCHS"
echo "  - Effective Batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - Weight decay:    $WEIGHT_DECAY"
echo "  - Warmup ratio:    $WARMUP_RATIO"
echo "  - LR scheduler:    $LR_SCHEDULER"
echo "  - Optimizer:       $OPTIMIZER"
echo "========================================================================"
