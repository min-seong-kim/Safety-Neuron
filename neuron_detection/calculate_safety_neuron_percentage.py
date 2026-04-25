"""
Calculate the percentage of detected safety neurons against model-wide neuron count.

Usage:
python calculate_safety_neuron_percentage.py \
  --neuron_file ./output_neurons/safety_neuron_accelerated_20260420_194552.txt \
  --model_name meta-llama/Llama-3.1-8B-instruct

  
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoConfig

from neuron_percentage_utils import (
    calculate_detected_parameter_count_from_neurons,
    calculate_total_model_neurons_from_config,
    calculate_total_model_parameters_from_config,
)


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
        default="meta-llama/Llama-3.1-8B",
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

    # 1) Existing neuron-level ratio (for backward compatibility)
    total_model_neurons = calculate_total_model_neurons_from_config(cfg)
    neuron_percentage = (total_selected / total_model_neurons * 100.0) if total_model_neurons > 0 else 0.0

    # 2) Parameter-level ratio (column-wise neuron interpretation)
    detected_safety_params = calculate_detected_parameter_count_from_neurons(neurons, cfg)
    total_model_params = calculate_total_model_parameters_from_config(cfg)
    parameter_percentage = (detected_safety_params / total_model_params * 100.0) if total_model_params > 0 else 0.0

    print("=" * 72)
    print("Safety Neuron Percentage Report")
    print("=" * 72)
    print(f"Neuron file: {neuron_file}")
    print(f"Model: {args.model_name}")
    print("-" * 72)
    print(f"Safety neurons found: {total_selected:,}")
    print(f"Model total neurons (q/k/v/o + gate/up/down): {total_model_neurons:,}")
    print(f"Safety neuron percentage: {neuron_percentage:.4f}%")
    print("-" * 72)
    print(f"Detected safety parameters (column-wise): {detected_safety_params:,}")
    print(f"Model total parameters (estimated): {total_model_params:,}")
    print(f"Safety parameter ratio: {parameter_percentage:.4f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
