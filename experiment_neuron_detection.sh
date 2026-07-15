#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0      # 사용할 GPU 지정 (0번만 사용, 둘 다 쓰려면 0,1)

ROOT_DIR="/home/edgeai_lab"
cd "$ROOT_DIR"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-hb_sn}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda.sh 없음: $CONDA_SH (CONDA_ROOT 설정 확인)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] 환경 '$CONDA_ENV'의 python 없음: $PYTHON_BIN" >&2
  exit 1
fi
if ! "$PYTHON_BIN" -c "import transformers" 2>/dev/null; then
  echo "[ERROR] $PYTHON_BIN 에서 transformers import 실패. 해당 env에 설치: pip install transformers" >&2
  exit 1
fi

EXPECTED_MODELING_LLAMA="${EXPECTED_MODELING_LLAMA:-}"

# Optional strict check: only enforced when EXPECTED_MODELING_LLAMA is explicitly provided.
if [[ -n "$EXPECTED_MODELING_LLAMA" ]]; then
    if [[ ! -f "$EXPECTED_MODELING_LLAMA" ]]; then
        echo "[ERROR] 기대한 modeling_llama.py 파일이 없습니다: $EXPECTED_MODELING_LLAMA" >&2
        exit 1
    fi

    if ! EXPECTED_MODELING_LLAMA="$EXPECTED_MODELING_LLAMA" "$PYTHON_BIN" - <<'PY' >/dev/null; then
import os
from transformers.models.llama import modeling_llama

expected = os.path.realpath(os.environ["EXPECTED_MODELING_LLAMA"])
loaded = os.path.realpath(modeling_llama.__file__)
if loaded != expected:
    raise SystemExit(1)
PY
        echo "[ERROR] 원하는 modeling_llama 경로가 로드되지 않았습니다." >&2
        echo "        기대 경로: $EXPECTED_MODELING_LLAMA" >&2
        echo "        현재 env: $CONDA_ENV" >&2
        exit 1
    fi
fi

# foundation 0.2% safety 1%

BASE_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TAG="qwen2_5_7b-instruct"
SAFETY_DATASET_FILE="/home/edgeai_lab/Safety-WaRP-LLM/data/circuit_breakers_train.json"
MIN_RESULTS="$ROOT_DIR/minseong_results"
FOUNDATION_DIR="$MIN_RESULTS/foundation_${TAG}"
SAFETY_DIR="$MIN_RESULTS/safety_${TAG}"
mkdir -p "$FOUNDATION_DIR" "$SAFETY_DIR"
NEURON_OUTPUT_DIR="$ROOT_DIR/Safety-Neuron/neuron_detection/output_neurons"
mkdir -p "$NEURON_OUTPUT_DIR"

find_latest_neuron_file() {
    local pattern="$1"
    local latest
    latest="$(ls -1t "$NEURON_OUTPUT_DIR"/$pattern 2>/dev/null | head -n 1 || true)"
    if [[ -z "$latest" ]]; then
        echo "[ERROR] output_neurons에서 패턴 '$pattern' 파일을 찾지 못했습니다: $NEURON_OUTPUT_DIR" >&2
        exit 1
    fi
    printf '%s\n' "$latest"
}

# compute_critical 입력 (실제 생성된 파일명과 맞출 것)
SAFETY_NEURON_FILE="${SAFETY_NEURON_FILE:-safety_neuron.txt}"
UTILITY_NEURON_FILE="${UTILITY_NEURON_FILE:-utility_neurons.txt}"
SAFETY_NEURON_IDX="$SAFETY_DIR/$SAFETY_NEURON_FILE"
UTILITY_NEURON_IDX="$FOUNDATION_DIR/$UTILITY_NEURON_FILE"
CRITICAL_DIR="$MIN_RESULTS/critical_safety_${TAG}"
mkdir -p "$CRITICAL_DIR"
CRITICAL_SAFETY_NEURON_IDX="$CRITICAL_DIR/critical_safety_neurons.txt"

# Optional fast path: if both existing files are provided, skip detection stages.
# 기존에 검출해 둔 파일로 detection 단계를 건너뛰고 바로 training 으로 진행하려면 아래 두 경로를 채운다.
# (다시 detection 부터 돌리려면 두 변수를 빈 문자열 "" 로 되돌린다.)
EXISTING_SAFETY_NEURON_PATH=""
EXISTING_CRITICAL_SAFETY_NEURON_PATH=""

if [[ -n "$EXISTING_SAFETY_NEURON_PATH" && -n "$EXISTING_CRITICAL_SAFETY_NEURON_PATH" ]]; then
    if [[ ! -f "$EXISTING_SAFETY_NEURON_PATH" ]]; then
        echo "[ERROR] EXISTING_SAFETY_NEURON_PATH 파일이 없습니다: $EXISTING_SAFETY_NEURON_PATH" >&2
        exit 1
    fi
    if [[ ! -f "$EXISTING_CRITICAL_SAFETY_NEURON_PATH" ]]; then
        echo "[ERROR] EXISTING_CRITICAL_SAFETY_NEURON_PATH 파일이 없습니다: $EXISTING_CRITICAL_SAFETY_NEURON_PATH" >&2
        exit 1
    fi

    # 원본과 목적지가 동일 파일이면 cp 가 에러(set -e 로 중단)나므로 다를 때만 복사.
    if [[ "$(realpath "$EXISTING_SAFETY_NEURON_PATH")" != "$(realpath "$SAFETY_NEURON_IDX")" ]]; then
        cp -f "$EXISTING_SAFETY_NEURON_PATH" "$SAFETY_NEURON_IDX"
    fi
    if [[ "$(realpath "$EXISTING_CRITICAL_SAFETY_NEURON_PATH")" != "$(realpath "$CRITICAL_SAFETY_NEURON_IDX")" ]]; then
        cp -f "$EXISTING_CRITICAL_SAFETY_NEURON_PATH" "$CRITICAL_SAFETY_NEURON_IDX"
    fi

    echo "[INFO] Detection 단계 스킵: 기존 safety/critical 파일 사용"
    echo "[INFO] Safety neurons  : $EXISTING_SAFETY_NEURON_PATH -> $EXISTING_SAFETY_NEURON_PATH"
    echo "[INFO] Critical neurons: $EXISTING_CRITICAL_SAFETY_NEURON_PATH -> $EXISTING_CRITICAL_SAFETY_NEURON_PATH"
else

echo "===== Find Foundation Neurons ======"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/safety_neuron_detection_v2_revised.py" 1000 \
    --model_name "$BASE_MODEL_NAME" \
    --top_number_ffn 600 \
    --top_number_attn 100 \
    --utility_neuron

LATEST_UTILITY_FILE="$(find_latest_neuron_file "utility_neurons_*.txt")"
cp -f "$LATEST_UTILITY_FILE" "$UTILITY_NEURON_IDX"
echo "[INFO] Utility neurons: $LATEST_UTILITY_FILE -> $UTILITY_NEURON_IDX"

echo "===== Find Safety Neurons ======"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/safety_neuron_detection_v2_revised.py" 4994 \
    --model_name "$BASE_MODEL_NAME" \
    --top_number_ffn 1200 \
    --top_number_attn 200 \
    --safety_neuron

LATEST_SAFETY_FILE="$(find_latest_neuron_file "safety_neuron_accelerated_*.txt")"
cp -f "$LATEST_SAFETY_FILE" "$SAFETY_NEURON_IDX"
echo "[INFO] Safety neurons: $LATEST_SAFETY_FILE -> $SAFETY_NEURON_IDX"

echo "===== Compute Critical Safety Neurons ======"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/compute_critical_safety_neurons.py" \
    "$SAFETY_NEURON_IDX" \
    "$UTILITY_NEURON_IDX"

LATEST_CRITICAL_FILE="$(find_latest_neuron_file "critical_safety_neuron_*.txt")"
cp -f "$LATEST_CRITICAL_FILE" "$CRITICAL_SAFETY_NEURON_IDX"
echo "[INFO] Critical safety neurons: $LATEST_CRITICAL_FILE -> $CRITICAL_SAFETY_NEURON_IDX"


echo "===== Calculate Safety Neuron Percentage ======"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/calculate_safety_neuron_percentage.py" \
    --neuron_file "$CRITICAL_SAFETY_NEURON_IDX" \
    --model_name "$BASE_MODEL_NAME" \
    > "$MIN_RESULTS/safety_${TAG}/safety_neuron_percentage_report.txt"

fi


echo "===== Switch Conda Env For Training ======"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-hb}"
conda activate "$TRAIN_CONDA_ENV"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] 환경 '$TRAIN_CONDA_ENV'의 python 없음: $PYTHON_BIN" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import transformers" 2>/dev/null; then
    echo "[ERROR] $PYTHON_BIN 에서 transformers import 실패. 해당 env에 설치: pip install transformers" >&2
    exit 1
fi




SN_LEARNING_RATE="5e-5"
# echo "===== Train SN-Tune ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/sn_tune.py" \
#     --neuron_file "$SAFETY_NEURON_IDX" \
#     --dataset_file "$SAFETY_DATASET_FILE" \
#     --learning_rate "$SN_LEARNING_RATE" \
#     --local_model_name "kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE}" \
#     --model_name "$BASE_MODEL_NAME" \
#     --upload_name "kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE}"



echo "===== Train RSN-Tune ======"
RSN_LOCAL_BASE="$MIN_RESULTS/rsn_tuned_${TAG}"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/sn_tune.py" \
    --neuron_file "$CRITICAL_SAFETY_NEURON_IDX" \
    --dataset_file "$SAFETY_DATASET_FILE" \
    --learning_rate "$SN_LEARNING_RATE" \
    --local_model_name "$RSN_LOCAL_BASE" \
    --model_name "$BASE_MODEL_NAME" \
    --upload_name "kmseong/${TAG}-only-rsn-tuned-lr${SN_LEARNING_RATE}"

# sn_tune.py 는 저장 시 디렉토리명에 _lr<lr>_<timestamp> 를 덧붙이므로 실제 경로를 찾아낸다.
# 이 경로를 downstream 에 직접 넘겨서 HF 업로드 성공 여부와 무관하게 체인이 이어지도록 한다.
RSN_MODEL_DIR="$(ls -1td "${RSN_LOCAL_BASE}"_* 2>/dev/null | head -n 1 || true)"
if [[ -z "$RSN_MODEL_DIR" || ! -f "$RSN_MODEL_DIR/config.json" ]]; then
    echo "[ERROR] RSN-Tune 출력 디렉토리를 찾지 못했습니다: ${RSN_LOCAL_BASE}_*" >&2
    exit 1
fi
echo "[INFO] RSN model dir: $RSN_MODEL_DIR"



DOWNSTREAM_LEARNING_RATE="${DOWNSTREAM_LEARNING_RATE:-5e-5}"  # downstream FT 전용 LR (SN_LEARNING_RATE와 독립)

# echo "===== Train GSM8K Finetune: SN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_gsm8k_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$SAFETY_NEURON_IDX" \
#     --output_dir "$MIN_RESULTS/gsm8k_sn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-gsm8k-sn-tuned-lr${DOWNSTREAM_LEARNING_RATE}

echo "===== Train GSM8K Finetune: RSN ======"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_gsm8k_freeze_sn.py" \
    --model_path "$RSN_MODEL_DIR" \
    --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
    --safety_neurons_file "$CRITICAL_SAFETY_NEURON_IDX" \
    --output_dir "$MIN_RESULTS/gsm8k_rsn_finetune_${TAG}" \
    --upload_name kmseong/${TAG}-gsm8k-rsn-tuned-lr${DOWNSTREAM_LEARNING_RATE}

# echo "===== Train MATH Finetune: SN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_hendrycks_math_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$SAFETY_NEURON_IDX" \
#     --output_dir "$MIN_RESULTS/math_sn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-math-sn-tuned-lr${DOWNSTREAM_LEARNING_RATE}

# echo "===== Train MATH Finetune: RSN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_hendrycks_math_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-rsn-tuned-lr${SN_LEARNING_RATE} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$CRITICAL_SAFETY_NEURON_IDX" \
#     --output_dir "$MIN_RESULTS/math_rsn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-math-rsn-tuned-lr${DOWNSTREAM_LEARNING_RATE}



# echo "===== Train MBPP Finetune: SN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_mbpp_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$SAFETY_NEURON_IDX" \
#     --safety_data_path "$SAFETY_DATASET_FILE" \
#     --output_dir "$MIN_RESULTS/mbpp_sn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-mbpp-sn-tuned-lr${DOWNSTREAM_LEARNING_RATE}

# echo "===== Train MBPP Finetune: RSN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_mbpp_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-rsn-tuned-lr${SN_LEARNING_RATE} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$CRITICAL_SAFETY_NEURON_IDX" \
#     --safety_data_path "$SAFETY_DATASET_FILE" \
#     --output_dir "$MIN_RESULTS/mbpp_rsn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-mbpp-rsn-tuned-lr${DOWNSTREAM_LEARNING_RATE}

