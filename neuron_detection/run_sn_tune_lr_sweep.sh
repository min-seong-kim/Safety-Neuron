#!/bin/bash
# bash run_sn_tune_lr_sweep.sh ./output_neurons/safety_neuron_threshold_20260406_172646.txt ./corpus_all/circuit_breakers_train.json ./only_sn_tuned_model



set -e

NEURONS_FILE="${1:-./output_neurons/safety_neuron_threshold_20260406_172646.txt}"
DATASET="${2:-./corpus_all/circuit_breakers_train.json}"
OUTPUT_BASE="${3:-./only_sn_tuned_model}"

echo "Safety neurons: $NEURONS_FILE"
echo "Dataset: $DATASET"
echo "Output base: $OUTPUT_BASE"
echo ""

for LR in 1e-5 3e-5 5e-5; do
    echo "=========================================="
    echo "Training with LR=$LR"
    echo "=========================================="
    python sn_tune.py \
        "$NEURONS_FILE" \
        "$DATASET" \
        "$OUTPUT_BASE" \
        "$LR"
    echo "Done: LR=$LR"
    echo ""
done

echo "All 3 models trained."
