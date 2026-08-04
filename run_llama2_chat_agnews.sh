#!/usr/bin/env bash
# experiment_neuron_detection.sh 실행 래퍼 (llama2-7b-chat / circuit_breakers / AG News).
#
# - detection 3단계는 2026-07-28 실행 결과를 재사용해 건너뛴다.
# - SN-Tune / RSN-Tune → AG News 8k downstream FT (safety neuron freeze) 순으로 4번 학습.
# - GPU 1번만 사용.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

export ROOT_DIR=/home/edgeai_lab
export CONDA_ROOT=/home/edgeai_lab/miniconda3
export CONDA_ENV=hb_sn          # detection env (패치된 modeling_llama.py)
export TRAIN_CONDA_ENV=hb       # training env (stock transformers)

NEURONS=/home/edgeai_lab/Safety-Neuron/neuron_detection/output_neurons
DATA=/home/edgeai_lab/Safety-WaRP-LLM/data

# 2026-07-28 llama2-7b-chat + circuit_breakers 탐지 결과 재사용
export EXISTING_SAFETY_NEURON_PATH="$NEURONS/safety_neuron_accelerated_20260728_044429.txt"
export EXISTING_CRITICAL_SAFETY_NEURON_PATH="$NEURONS/critical_safety_neuron_20260728_044945.txt"

export SAFETY_DATASET_FILE="$DATA/circuit_breakers_train.json"
export AGNEWS_TRAIN_FILE="$DATA/agnews_train_8k_seed42.json"

# 아래는 전부 호출자가 미리 export 한 값이 우선한다 (예: RUN_SN=0 bash run_....sh).
export RUN_SUFFIX="${RUN_SUFFIX:--cb}"
export RUN_SN="${RUN_SN:-1}"
export RUN_RSN="${RUN_RSN:-1}"

export DOWNSTREAM_SCRIPT="${DOWNSTREAM_SCRIPT:-finetune_agnews_freeze_sn.py}"
export DOWNSTREAM_TASK="${DOWNSTREAM_TASK:-agnews}"
export DOWNSTREAM_LEARNING_RATE="${DOWNSTREAM_LEARNING_RATE:-3e-5}"
export DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-3}"

exec bash /home/edgeai_lab/Safety-Neuron/experiment_neuron_detection.sh
