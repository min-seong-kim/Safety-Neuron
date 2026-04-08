"""
Calculate the percentage of detected safety neurons against model-wide neuron count.

Usage:
python calculate_safety_neuron_percentage.py \
  --neuron_file ./output_neurons/critical-safety-neuron_20260406_201744.txt \
  --model_name meta-llama/Llama-3.2-3B
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate safety neuron percentage")
    parser.add_argument(
        "--neuron_file",
        type=str,
        required=True,
        help="Path to detection output file (5 JSON lines: ffn_up, ffn_down, q, k, v)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="Hugging Face model ID for architecture config",
    )
    return parser.parse_args()


def load_neuron_file(file_path: Path) -> Dict[str, Dict[int, List[int]]]:
    with file_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 5:
        raise ValueError(f"Expected 5 JSON lines, got {len(lines)}")

    keys = ["ffn_up", "ffn_down", "q", "k", "v"]
    result: Dict[str, Dict[int, List[int]]] = {}

    for key, raw in zip(keys, lines[:5]):
        obj = json.loads(raw)
        result[key] = {int(layer): list(indices) for layer, indices in obj.items()}

    return result


def count_selected_neurons(neurons: Dict[str, Dict[int, List[int]]]) -> int:
    return sum(len(indices) for layer_map in neurons.values() for indices in layer_map.values())


def main():
    args = parse_args()

    neuron_file = Path(args.neuron_file)
    if not neuron_file.exists():
        raise FileNotFoundError(f"Neuron file not found: {neuron_file}")

    neurons = load_neuron_file(neuron_file)
    total_selected = count_selected_neurons(neurons)

    cfg = AutoConfig.from_pretrained(args.model_name)
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size

    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    # Model-wide neuron baseline (output channels of transformer linear layers per block):
    # q, k, v, o, gate, up, down
    total_model_neurons = num_layers * (
        hidden_size +      # q
        kv_dim +           # k
        kv_dim +           # v
        hidden_size +      # o
        intermediate_size +  # gate
        intermediate_size +  # up
        hidden_size        # down
    )

    percentage = (total_selected / total_model_neurons * 100.0) if total_model_neurons > 0 else 0.0

    print("=" * 72)
    print("Safety Neuron Percentage Report")
    print("=" * 72)
    print(f"Neuron file: {neuron_file}")
    print(f"Model: {args.model_name}")
    print("-" * 72)
    print(f"Safety neurons found: {total_selected:,}")
    print(f"Model total neurons (q/k/v/o + gate/up/down): {total_model_neurons:,}")
    print(f"Safety neuron percentage: {percentage:.4f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
