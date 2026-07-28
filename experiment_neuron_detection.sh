#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0      # 사용할 GPU 지정 (0번만 사용, 둘 다 쓰려면 0,1)

ROOT_DIR="${ROOT_DIR:-/root}"
cd "$ROOT_DIR"
CONDA_ROOT="${CONDA_ROOT:-/opt/miniforge3}"
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

# 패치 내용 검증: detection 은 modeling_llama.py 가 노출하는 _last_*_score 를 읽는다.
# stock 파일이 로드되면 모든 프롬프트가 조용히 실패하고 빈 뉴런 파일이 나오므로 여기서 끊는다.
if ! "$PYTHON_BIN" - <<'PY'; then
import inspect, sys
from transformers.models.llama import modeling_llama

src = inspect.getsource(modeling_llama)
missing = [m for m in ("_last_ffn_up_score", "_last_q_score", "_last_v_score") if m not in src]
if missing:
    print(f"[patch-check] {modeling_llama.__file__} 에 없는 마커: {missing}", file=sys.stderr)
    sys.exit(1)
print(f"[patch-check] OK: {modeling_llama.__file__}")
PY
    echo "[ERROR] 패치되지 않은 modeling_llama.py 가 로드되었습니다 (env: $CONDA_ENV)." >&2
    echo "        neuron_detection/transformers/models/'modeling_llama (2).py' 를 해당 env 의" >&2
    echo "        site-packages/transformers/models/llama/modeling_llama.py 로 복사하세요." >&2
    exit 1
fi

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

BASE_MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
TAG="llama2_7b-chat"

# 업로드/출력 이름 접미사 — 실행 조건(탐지 코퍼스 등)을 구분하기 위한 태그.
#   -bt      : beavertails 탐지 (2026-07-27 실행)
#   -cb-harm : circuit_breakers 탐지 + downstream 에 harmful 혼합 (현재 설정)
RUN_SUFFIX="${RUN_SUFFIX:--cb-harm}"

# safety neuron 탐지와 SN/RSN-Tune 이 함께 쓰는 safety 코퍼스.
SAFETY_DATASET_FILE="${SAFETY_DATASET_FILE:-/root/Safety-WaRP-LLM/data/circuit_breakers_train.json}"
if [[ ! -f "$SAFETY_DATASET_FILE" ]]; then
  echo "[ERROR] safety 코퍼스가 없습니다: $SAFETY_DATASET_FILE" >&2
  exit 1
fi

# 레이어당 top-k. foundation 을 작게 잡을수록 critical(= safety \ foundation)이 커진다.
FOUNDATION_TOP_FFN="${FOUNDATION_TOP_FFN:-300}"
FOUNDATION_TOP_ATTN="${FOUNDATION_TOP_ATTN:-50}"
SAFETY_TOP_FFN="${SAFETY_TOP_FFN:-1200}"
SAFETY_TOP_ATTN="${SAFETY_TOP_ATTN:-200}"

MIN_RESULTS="$ROOT_DIR/minseong_results"
# 실행별로 디렉토리를 분리해 이전 실행의 뉴런 파일을 덮어쓰지 않는다.
FOUNDATION_DIR="$MIN_RESULTS/foundation_${TAG}${RUN_SUFFIX}"
SAFETY_DIR="$MIN_RESULTS/safety_${TAG}${RUN_SUFFIX}"
mkdir -p "$FOUNDATION_DIR" "$SAFETY_DIR"
NEURON_OUTPUT_DIR="$ROOT_DIR/Safety-Neuron/neuron_detection/output_neurons"
mkdir -p "$NEURON_OUTPUT_DIR"

# output_neurons 에는 과거 실행 파일이 수백 개 쌓여 있어서 ls -1t 만으로는 stale 파일을
# 집을 수 있다. 각 stage 직전에 touch 하는 마커보다 새 파일만 인정한다.
STAGE_MARKER="$(mktemp)"
trap 'rm -f "$STAGE_MARKER"' EXIT

find_latest_neuron_file() {
    local pattern="$1"
    local latest
    latest="$(ls -1t "$NEURON_OUTPUT_DIR"/$pattern 2>/dev/null | head -n 1 || true)"
    if [[ -z "$latest" ]]; then
        echo "[ERROR] output_neurons에서 패턴 '$pattern' 파일을 찾지 못했습니다: $NEURON_OUTPUT_DIR" >&2
        exit 1
    fi
    if [[ ! "$latest" -nt "$STAGE_MARKER" ]]; then
        echo "[ERROR] 이번 실행에서 생성된 파일이 아닙니다 (stale): $latest" >&2
        exit 1
    fi
    # 탐지 스크립트는 프롬프트별 예외를 삼키고 exit 0 으로 끝나므로 빈 파일이 나올 수 있다.
    if ! "$PYTHON_BIN" - "$latest" <<'PY' >&2
import ast, sys
path = sys.argv[1]
names = ["ffn_up", "ffn_down", "q", "k", "v"]
with open(path) as f:
    lines = [ln for ln in f.read().splitlines() if ln.strip()]
if len(lines) != 5:
    print(f"[neuron-check] {path}: 5줄이 아님 ({len(lines)}줄)")
    sys.exit(1)
total = 0
for name, ln in zip(names, lines):
    d = ast.literal_eval(ln)
    n = sum(len(v) for v in d.values())
    print(f"[neuron-check] {name:9s} layers={len(d)} neurons={n}")
    total += n
if total == 0:
    print(f"[neuron-check] {path}: 검출된 뉴런이 0개입니다 — 탐지가 전부 실패했습니다")
    sys.exit(1)
print(f"[neuron-check] OK: {path} (총 {total}개)")
PY
    then
        exit 1
    fi
    printf '%s\n' "$latest"
}

# compute_critical 입력 (실제 생성된 파일명과 맞출 것)
SAFETY_NEURON_FILE="${SAFETY_NEURON_FILE:-safety_neuron.txt}"
UTILITY_NEURON_FILE="${UTILITY_NEURON_FILE:-utility_neurons.txt}"
SAFETY_NEURON_IDX="$SAFETY_DIR/$SAFETY_NEURON_FILE"
UTILITY_NEURON_IDX="$FOUNDATION_DIR/$UTILITY_NEURON_FILE"
CRITICAL_DIR="$MIN_RESULTS/critical_safety_${TAG}${RUN_SUFFIX}"
mkdir -p "$CRITICAL_DIR"
CRITICAL_SAFETY_NEURON_IDX="$CRITICAL_DIR/critical_safety_neurons.txt"

# Optional fast path: if both existing files are provided, skip detection stages.
# 기존에 검출해 둔 파일로 detection 단계를 건너뛰고 바로 training 으로 진행하려면 아래 두 경로를 채운다.
# (다시 detection 부터 돌리려면 두 변수를 빈 문자열 "" 로 되돌린다.)
EXISTING_SAFETY_NEURON_PATH="${EXISTING_SAFETY_NEURON_PATH:-}"
EXISTING_CRITICAL_SAFETY_NEURON_PATH="${EXISTING_CRITICAL_SAFETY_NEURON_PATH:-}"

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
touch "$STAGE_MARKER"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/safety_neuron_detection_v2_revised.py" 1000 \
    --model_name "$BASE_MODEL_NAME" \
    --top_number_ffn "$FOUNDATION_TOP_FFN" \
    --top_number_attn "$FOUNDATION_TOP_ATTN" \
    --utility_neuron

LATEST_UTILITY_FILE="$(find_latest_neuron_file "utility_neurons_*.txt")"
cp -f "$LATEST_UTILITY_FILE" "$UTILITY_NEURON_IDX"
echo "[INFO] Utility neurons: $LATEST_UTILITY_FILE -> $UTILITY_NEURON_IDX"

echo "===== Find Safety Neurons ======"
touch "$STAGE_MARKER"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/safety_neuron_detection_v2_revised.py" 4994 \
    --model_name "$BASE_MODEL_NAME" \
    --top_number_ffn "$SAFETY_TOP_FFN" \
    --top_number_attn "$SAFETY_TOP_ATTN" \
    --dataset_file "$SAFETY_DATASET_FILE" \
    --safety_neuron

LATEST_SAFETY_FILE="$(find_latest_neuron_file "safety_neuron_accelerated_*.txt")"
cp -f "$LATEST_SAFETY_FILE" "$SAFETY_NEURON_IDX"
echo "[INFO] Safety neurons: $LATEST_SAFETY_FILE -> $SAFETY_NEURON_IDX"

echo "===== Compute Critical Safety Neurons ======"
touch "$STAGE_MARKER"
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

# 어떤 계열을 학습할지 선택 (둘 다 1 이면 SN → RSN 순서로 전부 실행).
#   SN  = 전체 safety neuron (N_safe) 학습
#   RSN = critical safety neuron (N_safe \ N_foundation) 학습
RUN_SN="${RUN_SN:-1}"
RUN_RSN="${RUN_RSN:-1}"

# sn_tune.py 는 저장 시 디렉토리명에 _lr<lr>_<timestamp> 를 덧붙이므로 실제 경로를 찾아낸다.
# 이 경로를 downstream 에 직접 넘겨서 HF 업로드 성공 여부와 무관하게 체인이 이어지도록 한다.
resolve_tuned_dir() {
    local base="$1" label="$2" resolved
    resolved="$(ls -1td "${base}"_* 2>/dev/null | head -n 1 || true)"
    if [[ -z "$resolved" || ! -f "$resolved/config.json" ]]; then
        echo "[ERROR] ${label} 출력 디렉토리를 찾지 못했습니다: ${base}_*" >&2
        exit 1
    fi
    printf '%s\n' "$resolved"
}

if [[ "$RUN_SN" == "1" ]]; then
echo "===== Train SN-Tune ======"
SN_LOCAL_BASE="$MIN_RESULTS/sn_tuned_${TAG}${RUN_SUFFIX}"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/sn_tune.py" \
    --neuron_file "$SAFETY_NEURON_IDX" \
    --dataset_file "$SAFETY_DATASET_FILE" \
    --learning_rate "$SN_LEARNING_RATE" \
    --local_model_name "$SN_LOCAL_BASE" \
    --model_name "$BASE_MODEL_NAME" \
    --upload_name "kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE}${RUN_SUFFIX}"

SN_MODEL_DIR="$(resolve_tuned_dir "$SN_LOCAL_BASE" "SN-Tune")"
echo "[INFO] SN model dir: $SN_MODEL_DIR"
fi

if [[ "$RUN_RSN" == "1" ]]; then
echo "===== Train RSN-Tune ======"
RSN_LOCAL_BASE="$MIN_RESULTS/rsn_tuned_${TAG}${RUN_SUFFIX}"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/sn_tune.py" \
    --neuron_file "$CRITICAL_SAFETY_NEURON_IDX" \
    --dataset_file "$SAFETY_DATASET_FILE" \
    --learning_rate "$SN_LEARNING_RATE" \
    --local_model_name "$RSN_LOCAL_BASE" \
    --model_name "$BASE_MODEL_NAME" \
    --upload_name "kmseong/${TAG}-only-rsn-tuned-lr${SN_LEARNING_RATE}${RUN_SUFFIX}"

RSN_MODEL_DIR="$(resolve_tuned_dir "$RSN_LOCAL_BASE" "RSN-Tune")"
echo "[INFO] RSN model dir: $RSN_MODEL_DIR"
fi



DOWNSTREAM_LEARNING_RATE="${DOWNSTREAM_LEARNING_RATE:-5e-5}"  # downstream FT 전용 LR (SN_LEARNING_RATE와 독립)

# downstream 학습 스크립트와 harmful 혼합 설정.
#   finetune_gsm8k_freeze_sn.py          : GSM8K 단독 (harmful 인자 미지원)
#   finetune_gsm8k_harmful_freeze_sn.py  : GSM8K + harmful 혼합
DOWNSTREAM_SCRIPT="${DOWNSTREAM_SCRIPT:-finetune_gsm8k_harmful_freeze_sn.py}"
HARMFUL_DATA_FILE="${HARMFUL_DATA_FILE:-/root/Safety-WaRP-LLM/data/beavertails_harmful_747.json}"
HARMFUL_NUM="${HARMFUL_NUM:-747}"          # GSM8K 7473 의 10%
HARMFUL_ANSWER_FIELD="${HARMFUL_ANSWER_FIELD:-response}"

HARMFUL_ARGS=()
if [[ "$DOWNSTREAM_SCRIPT" == *harmful* ]]; then
    if [[ ! -f "$HARMFUL_DATA_FILE" ]]; then
        echo "[ERROR] harmful 데이터가 없습니다: $HARMFUL_DATA_FILE" >&2
        exit 1
    fi
    HARMFUL_ARGS=(
        --harmful_data_file "$HARMFUL_DATA_FILE"
        --harmful_num "$HARMFUL_NUM"
        --harmful_answer_field "$HARMFUL_ANSWER_FIELD"
    )
fi

if [[ "$RUN_SN" == "1" ]]; then
echo "===== Train GSM8K Finetune: SN ======"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/$DOWNSTREAM_SCRIPT" \
    --model_path "$SN_MODEL_DIR" \
    --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
    --safety_neurons_file "$SAFETY_NEURON_IDX" \
    "${HARMFUL_ARGS[@]}" \
    --output_dir "$MIN_RESULTS/gsm8k_sn_finetune_${TAG}${RUN_SUFFIX}" \
    --upload_name kmseong/${TAG}-gsm8k-sn-tuned-lr${DOWNSTREAM_LEARNING_RATE}${RUN_SUFFIX}
fi

if [[ "$RUN_RSN" == "1" ]]; then
echo "===== Train GSM8K Finetune: RSN ======"
"$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/$DOWNSTREAM_SCRIPT" \
    --model_path "$RSN_MODEL_DIR" \
    --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
    --safety_neurons_file "$CRITICAL_SAFETY_NEURON_IDX" \
    "${HARMFUL_ARGS[@]}" \
    --output_dir "$MIN_RESULTS/gsm8k_rsn_finetune_${TAG}${RUN_SUFFIX}" \
    --upload_name kmseong/${TAG}-gsm8k-rsn-tuned-lr${DOWNSTREAM_LEARNING_RATE}${RUN_SUFFIX}
fi

# echo "===== Train MATH Finetune: SN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_hendrycks_math_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE}${RUN_SUFFIX} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$SAFETY_NEURON_IDX" \
#     --output_dir "$MIN_RESULTS/math_sn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-math-sn-tuned-lr${DOWNSTREAM_LEARNING_RATE}${RUN_SUFFIX}

# echo "===== Train MATH Finetune: RSN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_hendrycks_math_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-rsn-tuned-lr${SN_LEARNING_RATE}${RUN_SUFFIX} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$CRITICAL_SAFETY_NEURON_IDX" \
#     --output_dir "$MIN_RESULTS/math_rsn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-math-rsn-tuned-lr${DOWNSTREAM_LEARNING_RATE}${RUN_SUFFIX}



# echo "===== Train MBPP Finetune: SN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_mbpp_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-sn-tuned-lr${SN_LEARNING_RATE}${RUN_SUFFIX} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$SAFETY_NEURON_IDX" \
#     --safety_data_path "$SAFETY_DATASET_FILE" \
#     --output_dir "$MIN_RESULTS/mbpp_sn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-mbpp-sn-tuned-lr${DOWNSTREAM_LEARNING_RATE}${RUN_SUFFIX}

# echo "===== Train MBPP Finetune: RSN ======"
# "$PYTHON_BIN" "$ROOT_DIR/Safety-Neuron/neuron_detection/finetune_mbpp_freeze_sn.py" \
#     --model_path kmseong/${TAG}-only-rsn-tuned-lr${SN_LEARNING_RATE}${RUN_SUFFIX} \
#     --learning_rate "$DOWNSTREAM_LEARNING_RATE" \
#     --safety_neurons_file "$CRITICAL_SAFETY_NEURON_IDX" \
#     --safety_data_path "$SAFETY_DATASET_FILE" \
#     --output_dir "$MIN_RESULTS/mbpp_rsn_finetune_${TAG}" \
#     --upload_name kmseong/${TAG}-mbpp-rsn-tuned-lr${DOWNSTREAM_LEARNING_RATE}${RUN_SUFFIX}

