#!/usr/bin/env bash
# =============================================================================
# run_sn_pipeline.sh
#
# Phase 1: SN-Tune  (train only safety neurons on circuit breakers)
# Phase 2: GSM8K fine-tuning with safety neuron freezing
#
# NOTE: Safety neurons are detected on the ROTATED model for higher reliability
#       (rotation concentrates safety directions → larger intersection).
#       However, SN-Tune and GSM8K fine-tuning are performed on the ORIGINAL
#       model, because the rotation breaks inter-layer representation consistency
#       (each layer has a different V; the model output becomes ill-defined).
#       Row indices from the rotated model correspond to the same row indices
#       in the original model (rotation is right-multiplication → row index preserved).
#
# Detection pipeline (run separately before this script):
#   1. apply_safety_basis_rotation.py  → rotated model
#   2. safety_neuron_detection_v2.py   → raw neuron txt (rotation space)
#   3. map_rotated_to_original_neurons.py → _original_space.txt  ← NEURON_FILE below
#
# Usage:
#   bash run_sn_pipeline.sh [--hf_token TOKEN]
#
# All major parameters are in the "Configuration" section below.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

# Safety neuron file mapped back to original space
# (output of map_rotated_to_original_neurons.py)
NEURON_FILE="./output_neurons/safety_neuron_accelerated_20260505_193126_original_space.txt"

# Circuit Breakers dataset
DATASET_FILE="./corpus_all/circuit_breakers_train.json"

# ORIGINAL (non-rotated) base model for SN-Tune and GSM8K fine-tuning
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"

# SN-Tune output directory prefix (timestamp + lr suffix appended automatically)
SN_LOCAL_MODEL_DIR="./only_sn_tuned_model_llama2_7b"

# HF upload names (leave empty "" to skip upload)
SN_UPLOAD_NAME="kmseong/Llama-2-7b-chat-hf_only_sn_tuned_lr5e-5_rotation_space"
GSM8K_UPLOAD_NAME="kmseong/Llama-2-7b-chat-hf_gsm8k_ft_freeze_rotation_space_sn_lr5e-5"

# HF token (can also be passed via --hf_token argument)
HF_TOKEN=""

# SN-Tune hyperparameters
SN_LR="5e-5"

# GSM8K fine-tuning hyperparameters
GSM8K_OUTPUT_DIR="./llama2_7b_gsm8k_ft_freeze_sn"
GSM8K_LR="5e-5"
GSM8K_EPOCHS=3

# ── Parse arguments ────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf_token) HF_TOKEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ────────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: Required file not found: $1"
        exit 1
    fi
}

# ── Preflight checks ───────────────────────────────────────────────────────────

log "=== SN Pipeline Start ==="
check_file "$NEURON_FILE"
check_file "$DATASET_FILE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Phase 1: SN-Tune (original model) ────────────────────────────────────────

log ">>> Phase 1: SN-Tune"
log "    model      : $BASE_MODEL  (original, non-rotated)"
log "    neuron_file: $NEURON_FILE  (original space)"
log "    lr         : $SN_LR"

SN_TUNE_ARGS=(
    --neuron_file      "$NEURON_FILE"
    --dataset_file     "$DATASET_FILE"
    --local_model_name "$SN_LOCAL_MODEL_DIR"
    --model_name       "$BASE_MODEL"
    --learning_rate    "$SN_LR"
)
[[ -n "$SN_UPLOAD_NAME" ]] && SN_TUNE_ARGS+=(--upload_name "$SN_UPLOAD_NAME")
[[ -n "$HF_TOKEN"       ]] && SN_TUNE_ARGS+=(--hf_token    "$HF_TOKEN")

python sn_tune.py "${SN_TUNE_ARGS[@]}"

# sn_tune.py saves to {SN_LOCAL_MODEL_DIR}_lr{lr}_{timestamp}
# find the most recently created matching directory
SN_SAVED_DIR=$(ls -dt "${SN_LOCAL_MODEL_DIR}"_lr*_20* 2>/dev/null | head -n 1 || true)

if [[ -z "$SN_SAVED_DIR" ]]; then
    log "WARNING: Could not find local SN-Tune output directory."
    log "         Will fall back to HF upload name for Phase 2."
    PHASE2_MODEL="$SN_UPLOAD_NAME"
else
    log "SN-Tune output directory: $SN_SAVED_DIR"
    PHASE2_MODEL="$SN_SAVED_DIR"
fi

# ── Phase 2: GSM8K fine-tuning (freeze safety neurons) ───────────────────────

log ">>> Phase 2: GSM8K fine-tuning (freeze safety neurons)"
log "    model      : $PHASE2_MODEL"
log "    neuron_file: $NEURON_FILE  (original space)"
log "    output_dir : $GSM8K_OUTPUT_DIR"
log "    lr         : $GSM8K_LR"
log "    epochs     : $GSM8K_EPOCHS"

GSM8K_ARGS=(
    --model_path          "$PHASE2_MODEL"
    --safety_neurons_file "$(realpath "$NEURON_FILE")"
    --output_dir          "$GSM8K_OUTPUT_DIR"
    --learning_rate       "$GSM8K_LR"
    --epochs              "$GSM8K_EPOCHS"
)
[[ -n "$GSM8K_UPLOAD_NAME" ]] && GSM8K_ARGS+=(--upload_name "$GSM8K_UPLOAD_NAME")
[[ -n "$HF_TOKEN"          ]] && GSM8K_ARGS+=(--hf_token    "$HF_TOKEN")

python finetune_gsm8k_freeze_sn.py "${GSM8K_ARGS[@]}"

log "=== SN Pipeline Complete ==="
