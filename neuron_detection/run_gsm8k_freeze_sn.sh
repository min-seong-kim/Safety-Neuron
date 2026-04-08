#!/bin/bash

# RSN-Tune: GSM8K Downstream Finetuning (Safety Neurons Frozen)
#
# 실험 구조:
#   - Safety neuron (RSN)은 freeze
#   - 나머지 파라미터로 GSM8K downstream finetuning
#   - LR: 1e-5, 3e-5, 5e-5 sweep
#
# WaRP Phase 3와 공정한 비교를 위해 동일한 하이퍼파라미터 사용:
#   - epochs: 3
#   - batch_size x grad_accum = 4 x 4 = 16 (effective)
#   - weight_decay: 0.01
#   - warmup_ratio: 0.1
#   - lr_scheduler: linear
#   - optimizer: adamw_torch
#   - max_length: 1024
#   - num_train_samples: 7473 (all GSM8K train)

set -e

echo "========================================================================"
echo "RSN-Tune: GSM8K Downstream Finetuning (Safety Neurons Frozen)"
echo "========================================================================"
echo ""

# ========================================================================
# Configuration
# ========================================================================

# RSN-Tuned 모델 경로 (safety training이 완료된 모델)
MODEL_PATH="./only_rsn_tuned_model_lr3e-5_lr3e-5_20260408_213540"

# RSN safety neuron 파일 (Foundation Neuron과의 차집합)
SAFETY_NEURONS_FILE="./output_neurons/critical_safety_neuron_20260408_212408.txt"

# LR sweep
LR_LIST=("1e-5" "3e-5" "5e-5")

# ── WaRP Phase 3와 동일하게 맞춘 하이퍼파라미터 ──────────────────────────
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4               # effective batch = 4 x 4 = 16
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
LR_SCHEDULER="cosine"
OPTIMIZER="adamw_torch"
MAX_GRAD_NORM=1.0
MAX_LENGTH=1024
NUM_TRAIN_SAMPLES=7473     # 전체 GSM8K train set
# ─────────────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR="./sn_tune_gsm8k_ft_freeze_sn"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/sn_tune_gsm8k_freeze_sn"
mkdir -p "$LOG_DIR"

# 파일 존재 확인
if [ ! -f "$SAFETY_NEURONS_FILE" ]; then
    echo "❌ ERROR: Safety neurons file not found: $SAFETY_NEURONS_FILE"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ] && [[ "$MODEL_PATH" != kmseong/* ]]; then
    echo "❌ ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

echo "Configuration:"
echo "  Model:               $MODEL_PATH"
echo "  Safety neurons file: $SAFETY_NEURONS_FILE"
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
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_lr${LR_SAFE}"
    LOG_FILE="${LOG_DIR}/gsm8k_freeze_sn_lr${LR_SAFE}_${TIMESTAMP}.log"

    echo "──────────────────────────────────────────────────────────────────────"
    echo "  LR = $LEARNING_RATE"
    echo "──────────────────────────────────────────────────────────────────────"

    python finetune_gsm8k_freeze_sn.py \
        --model_path "$MODEL_PATH" \
        --safety_neurons_file "$SAFETY_NEURONS_FILE" \
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

    # 가장 최근에 생성된 output 디렉토리 찾기 (타임스탬프 suffix가 붙어서 저장됨)
    LATEST_OUTPUT=$(find . -maxdepth 1 -name "$(basename $OUTPUT_DIR)_*" -type d \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$LATEST_OUTPUT" ]; then
        echo "⚠️  WARNING: Output directory not found for LR=$LEARNING_RATE"
    else
        echo "✅ LR=$LEARNING_RATE completed: $LATEST_OUTPUT"
        OUTPUT_DIRS+=("$LATEST_OUTPUT")
    fi

    echo ""
done

# ========================================================================
# Summary
# ========================================================================
echo "========================================================================"
echo "🎉 RSN-Tune GSM8K Finetuning Complete!"
echo "========================================================================"
echo ""
echo "📁 Output Models:"
for dir in "${OUTPUT_DIRS[@]}"; do
    echo "    ✅ $dir"
done
echo ""
echo "📝 Log Files: $LOG_DIR/gsm8k_freeze_sn_lr*_${TIMESTAMP}.log"
echo ""
echo "⚙️  Training Configuration Summary (matched to WaRP Phase 3):"
echo "  - Model:           $MODEL_PATH"
echo "  - Learning Rates:  ${LR_LIST[*]}"
echo "  - Epochs:          $EPOCHS"
echo "  - Effective Batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - Weight decay:    $WEIGHT_DECAY"
echo "  - Warmup ratio:    $WARMUP_RATIO"
echo "  - LR scheduler:    $LR_SCHEDULER"
echo "  - Optimizer:       $OPTIMIZER"
echo "========================================================================"
