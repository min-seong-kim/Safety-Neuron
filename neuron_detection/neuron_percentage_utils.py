"""Shared helpers for consistent neuron/parameter percentage calculations.

This module centralizes the model-wide denominator used across scripts:
q/k/v/o + gate/up/down output channels across all transformer layers.
"""

from typing import Any, Dict, List, Optional


LLAMA31_8B_FALLBACK_DIMS = {
    "num_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}


def calculate_total_model_neurons_from_dims(
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: Optional[int] = None,
) -> int:
    """Calculate model-wide neuron denominator from architecture dimensions."""
    num_kv_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
    head_dim = hidden_size // num_attention_heads
    kv_dim = num_kv_heads * head_dim

    return num_layers * (
        hidden_size +        # q
        kv_dim +             # k
        kv_dim +             # v
        hidden_size +        # o
        intermediate_size +  # gate
        intermediate_size +  # up
        hidden_size          # down
    )


def calculate_total_model_neurons_from_config(cfg: Any) -> int:
    """Calculate model-wide neuron denominator from a Hugging Face config."""
    return calculate_total_model_neurons_from_dims(
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
    )


def calculate_detected_parameter_count_from_neurons(
    neurons: Dict[str, Dict[int, List[int]]],
    cfg: Any,
) -> int:
    """Calculate detected safety parameter count assuming indices are column-wise.

    If neuron indices refer to matrix columns, each selected index contributes
    one full column of parameters. Therefore per-neuron parameter counts are:
      - ffn_up: out_features = intermediate_size
      - ffn_down: out_features = hidden_size
      - q: out_features = hidden_size
      - k: out_features = kv_dim
      - v: out_features = kv_dim
    """
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    per_neuron_params = {
        "ffn_up": intermediate_size,
        "ffn_down": hidden_size,
        "q": hidden_size,
        "k": kv_dim,
        "v": kv_dim,
    }

    total = 0
    for module_name, layer_map in neurons.items():
        unit = per_neuron_params.get(module_name)
        if unit is None:
            continue
        total += sum(len(indices) * unit for indices in layer_map.values())
    return total


def calculate_total_model_parameters_from_config(cfg: Any) -> int:
    """Estimate total model parameters from config (Llama-style architecture).

    This includes token embedding, per-layer core blocks, final norm, and
    lm_head when weights are not tied.
    """
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    vocab_size = cfg.vocab_size

    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    per_layer_params = (
        hidden_size * hidden_size +          # q_proj
        kv_dim * hidden_size +               # k_proj
        kv_dim * hidden_size +               # v_proj
        hidden_size * hidden_size +          # o_proj
        intermediate_size * hidden_size +    # gate_proj
        intermediate_size * hidden_size +    # up_proj
        hidden_size * intermediate_size +    # down_proj
        2 * hidden_size                      # input/post-attention layernorm
    )

    embedding_params = vocab_size * hidden_size
    final_norm_params = hidden_size
    lm_head_params = 0 if getattr(cfg, "tie_word_embeddings", True) else (hidden_size * vocab_size)

    return embedding_params + num_layers * per_layer_params + final_norm_params + lm_head_params
