#!/usr/bin/env python3
"""
map_rotated_to_original_neurons.py

회전 공간에서 검출된 safety neurons를 원본 공간에서 사용 가능한 JSON 형식으로 변환.

=== 변환이 필요 없는 이유 ===

rotation: W_new = W @ V  (right multiplication = 입력 공간 회전)

  F.linear(h, W_new) = h @ W_new^T = h @ (W @ V)^T = h @ V^T @ W^T

  → 입력 h 에 V^T 를 먼저 곱하는 것과 동일.
  → W 의 '행(row)' 인덱스는 변하지 않는다.

각 score 가 측정하는 차원:
  - ffn_up_score    : up_proj 출력, shape [intermediate=11008]
                      = W_up 의 행 인덱스 → rotation 후에도 동일
  - ffn_down_score  : 동일한 up_score 재사용 → 동일 인덱스
  - q_score/k_score : qk_score, shape [num_heads × head_dim] = [4096]
                      = W_q/W_k 의 행 인덱스 → rotation 후에도 동일
  - v_score         : attn_output 기반, shape [4096]
                      = W_v 의 행 인덱스 → rotation 후에도 동일

따라서 rotated model 에서 검출된 인덱스 i 는
original model 의 동일한 행 위치 i 를 가리킨다.

=== 이 스크립트가 하는 일 ===

1. 5-line text 형식 (safety_neuron_accelerated_*.txt) 을 로드
2. 원본 공간 ready txt 로 저장  (인덱스 값은 그대로, 형식만 변환)
3. fine-tuning 코드에서 바로 읽을 수 있는 구조 제공

Usage:

python map_rotated_to_original_neurons.py \
    --input  output_neurons/safety_neuron_accelerated_20260505_193126.txt \
    --output output_neurons/safety_neuron_accelerated_20260505_193126_original_space.txt \
    --model_name kmseong/llama2_7b_chat-safety-rotation
"""

import argparse
import json
import sys
from pathlib import Path

MODULE_KEYS = ["ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v"]

# fine-tuning 시 각 module key 가 freeze 하는 weight 의 차원 설명 (문서용)
FREEZE_DESCRIPTION = {
    "ffn_up":   "rows of up_proj.weight   [intermediate dim]",
    "ffn_down": "cols of down_proj.weight [intermediate dim]  (ffn_down_score = up_score)",
    "attn_q":   "rows of q_proj.weight    [num_heads × head_dim]",
    "attn_k":   "rows of k_proj.weight    [num_kv_heads × head_dim]",
    "attn_v":   "rows of v_proj.weight    [num_kv_heads × head_dim]",
}


def load_neuron_file(path: str) -> dict:
    """
    5-line text detection 결과 로드.

    Line 0: ffn_up   {"layer_idx": [neuron_idx, ...], ...}
    Line 1: ffn_down
    Line 2: attn_q
    Line 3: attn_k
    Line 4: attn_v
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) != 5:
        raise ValueError(
            f"Expected exactly 5 lines (ffn_up/ffn_down/q/k/v), got {len(lines)}.\n"
            f"File: {path}"
        )

    result = {}
    for key, line in zip(MODULE_KEYS, lines):
        raw = json.loads(line)
        # str → int key 변환, value 는 list[int]
        result[key] = {int(k): [int(x) for x in v] for k, v in raw.items()}
    return result


def print_stats(neurons: dict, header: str = "Neuron Statistics") -> None:
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    total = 0
    for key in MODULE_KEYS:
        layer_dict = neurons.get(key, {})
        nonempty = sum(1 for v in layer_dict.values() if v)
        count = sum(len(v) for v in layer_dict.values())
        total += count
        desc = FREEZE_DESCRIPTION.get(key, "")
        print(f"  {key:10s}: {count:6d} neurons  ({nonempty} layers non-empty)")
        print(f"             → {desc}")
    print(f"  {'TOTAL':10s}: {total:6d} neurons")
    print(f"{'=' * 60}\n")


def verify_indices(neurons: dict, model_name: str = None) -> None:
    """
    모델 config 에서 각 module 의 최대 가능 인덱스를 체크.
    model_name 이 주어지면 실제 config 로드, 없으면 스킵.
    """
    if model_name is None:
        return

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name)

        hidden = cfg.hidden_size                          # 4096
        intermediate = cfg.intermediate_size              # 11008
        num_q_heads = cfg.num_attention_heads             # 32
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
        head_dim = hidden // num_q_heads                  # 128

        max_idx = {
            "ffn_up":   intermediate - 1,
            "ffn_down": intermediate - 1,
            "attn_q":   num_q_heads  * head_dim - 1,
            "attn_k":   num_kv_heads * head_dim - 1,
            "attn_v":   num_kv_heads * head_dim - 1,
        }

        print("[Verification] Checking index bounds against model config...")
        ok = True
        for key, layer_dict in neurons.items():
            mx = max_idx.get(key, None)
            if mx is None:
                continue
            for layer_idx, idx_list in layer_dict.items():
                if idx_list and max(idx_list) > mx:
                    print(f"  [WARN] {key} layer {layer_idx}: "
                          f"max index {max(idx_list)} > max valid {mx}")
                    ok = False
        if ok:
            print("[Verification] All indices within valid range. ✓\n")

    except Exception as e:
        print(f"[Verification] Skipped (could not load config: {e})\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert rotated-space SN detection output to original-space txt (5-line format).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to safety_neuron_accelerated_*.txt (5-line format)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .txt path for original-space neurons (same 5-line format as input)"
    )
    parser.add_argument(
        "--model_name", default=None,
        help="(Optional) HF model name/path for index range verification"
    )
    args = parser.parse_args()

    print("\n[Rotation type] W_new = W @ V  (input-space rotation)")
    print("[Index mapping] Output neuron indices (row indices of W) are PRESERVED.")
    print("[Action]        No index transformation applied; format only.\n")

    # 1. Load
    neurons = load_neuron_file(args.input)
    print_stats(neurons, header=f"Loaded from: {args.input}")

    # 2. Optional verification
    verify_indices(neurons, args.model_name)

    # 3. Save in the same 5-line txt format as input
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for key in MODULE_KEYS:
            layer_dict = neurons[key]
            line = {str(layer_idx): sorted(idx_list)
                    for layer_idx, idx_list in sorted(layer_dict.items())}
            f.write(json.dumps(line) + "\n")

    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
