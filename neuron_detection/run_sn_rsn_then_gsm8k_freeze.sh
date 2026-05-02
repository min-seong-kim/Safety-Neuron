#!/usr/bin/env bash

# End-to-end pipeline:
# 1) SN-Tune (upload)
# 2) RSN-Tune (upload)
# 3) GSM8K finetune with SN model (safety neurons frozen, upload)
# 4) GSM8K finetune with RSN model (safety neurons frozen, upload)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs/pipeline"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/sn_rsn_gsm8k_pipeline_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "======================================================================"
echo "SN/RSN + GSM8K Freeze Pipeline"
echo "Start time: $(date)"
echo "Working dir: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo "======================================================================"

# Optional: export HF token once for both scripts if needed
# export HUGGINGFACE_HUB_TOKEN="your_hf_token"
# export WANDB_API_KEY="your_wandb_api_key"

# ----------------------------------------------------------------------
# Step 1) SN-Tune
# ----------------------------------------------------------------------
echo "[1/4] SN-Tune start"
python sn_tune.py \
    --neuron_file /home/yonsei_jong/Safety-Neuron/neuron_detection/output_neurons/safety_neuron_accelerated_20260502_013602.txt \
    --dataset_file ./corpus_all/circuit_breakers_train.json \
    --local_model_name ./only_sn_tuned_model_llama2_7b_lr5e-5 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --upload_name kmseong/llama2_7b_chat_only_sn_tuned_lr5e-5_revised

echo "[1/4] SN-Tune done"

# ----------------------------------------------------------------------
# Step 2) RSN-Tune
# ----------------------------------------------------------------------
echo "[2/4] RSN-Tune start"
python sn_tune.py \
    --neuron_file /home/yonsei_jong/Safety-Neuron/neuron_detection/output_neurons/critical_safety_neuron_20260502_022558.txt \
    --dataset_file ./corpus_all/circuit_breakers_train.json \
    --local_model_name ./only_rsn_tuned_model_llama2_7b_chat_lr5e-5 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --upload_name kmseong/llama2_7b_chat_only_rsn_tuned_lr5e-5_revised

echo "[2/4] RSN-Tune done"

# ----------------------------------------------------------------------
# Step 3) GSM8K finetune with SN model (freeze SN)
# ----------------------------------------------------------------------
echo "[3/4] GSM8K freeze-SN (SN model) start"
python finetune_gsm8k_freeze_sn.py \
    --model_path kmseong/llama2_7b_chat_only_sn_tuned_lr5e-5_revised \
    --safety_neurons_file /home/yonsei_jong/Safety-Neuron/neuron_detection/output_neurons/safety_neuron_accelerated_20260502_013602.txt \
    --output_dir ./llama2_7b_base_gsm8k_ft_freeze_sn_lr5e-5 \
    --learning_rate 5e-5 \
    --epochs 3 \
    --upload_name kmseong/llama2_7b_base_gsm8k_ft_freeze_sn_lr5e-5_revised

echo "[3/4] GSM8K freeze-SN (SN model) done"

# ----------------------------------------------------------------------
# Step 4) GSM8K finetune with RSN model (freeze RSN)
# ----------------------------------------------------------------------
echo "[4/4] GSM8K freeze-SN (RSN model) start"
python finetune_gsm8k_freeze_sn.py \
    --model_path kmseong/llama2_7b_chat_only_rsn_tuned_lr5e-5_revised \
    --safety_neurons_file /home/yonsei_jong/Safety-Neuron/neuron_detection/output_neurons/critical_safety_neuron_20260502_022558.txt \
    --output_dir ./llama2_7b_chat_gsm8k_ft_freeze_rsn_lr5e-5_new \
    --learning_rate 5e-5 \
    --epochs 3 \
    --upload_name kmseong/llama2_7b_chat_gsm8k_ft_freeze_rsn_lr5e-5_new_revised

echo "[4/4] GSM8K freeze-SN (RSN model) done"

echo "======================================================================"
echo "Pipeline complete: $(date)"
echo "All stages finished successfully."
echo "======================================================================"
