"""
python foundation_neuron_detection.py 1000 \
    --model_name meta-llama/Llama-3.1-8B \
    --ffn_active_fraction 0.01 \
    --attn_active_fraction 0.01

python foundation_neuron_detection.py 1000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --ffn_active_fraction 0.01 \
    --attn_active_fraction 0.01
"""

import os
import sys
import argparse
import torch
import random
import logging
import json
from typing import Dict, Set, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import math
import time

import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from neuron_percentage_utils import calculate_total_model_neurons_from_config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(112)
torch.manual_seed(112)

# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------
def is_instruct_model(name: str) -> bool:
    name = name.lower()
    return ("instruct" in name) or ("chat" in name)

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
model_name = DEFAULT_MODEL_NAME
tokenizer = None
model = None
NUM_LAYERS = 0

# Threshold hyperparameters
DEFAULT_FFN_ACTIVE_FRACTION = 0.05
DEFAULT_ATTN_ACTIVE_FRACTION = 0.05
FFN_ACTIVE_FRACTION = DEFAULT_FFN_ACTIVE_FRACTION
ATTN_ACTIVE_FRACTION = DEFAULT_ATTN_ACTIVE_FRACTION
MIN_NEURONS_FOR_QUANTILE = 10

# Accelerated detection hyperparameters
ATTN_QUERY_WINDOW = None      # None이면 전체 query position 사용
CAPTURE_HIDDEN_TO_CPU = False # hidden input을 GPU에 유지
DETAIL_LOG_PROMPT_LIMIT = 3
NEG_INF = -1e9


def initialize_model_and_tokenizer(selected_model_name: str):
    """Initialize global model/tokenizer after CLI args are parsed."""
    global model_name, model, tokenizer, NUM_LAYERS

    model_name = selected_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},  # Force all layers to cuda:0 (single GPU)
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    NUM_LAYERS = model.config.num_hidden_layers


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Foundation neuron detection with configurable model and thresholds"
    )
    parser.add_argument(
        "num_docs",
        type=int,
        nargs="?",
        default=1000,
        help="Number of Wikipedia documents to process",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--ffn_active_fraction",
        type=float,
        default=DEFAULT_FFN_ACTIVE_FRACTION,
        help="Global top fraction for FFN neurons (0~1)",
    )
    parser.add_argument(
        "--attn_active_fraction",
        "--attn_activ_fraction",
        dest="attn_active_fraction",
        type=float,
        default=DEFAULT_ATTN_ACTIVE_FRACTION,
        help="Global top fraction for attention neurons (0~1)",
    )

    args = parser.parse_args(argv)

    if not (0.0 < args.ffn_active_fraction <= 1.0):
        parser.error("--ffn_active_fraction must be in (0, 1].")
    if not (0.0 < args.attn_active_fraction <= 1.0):
        parser.error("--attn_active_fraction must be in (0, 1].")

    return args


def calculate_model_total_neurons() -> int:
    """
    Same denominator as safety_neuron_detection.py:
    q/k/v/o + gate/up/down output channels across all layers.
    """
    return calculate_total_model_neurons_from_config(model.config)


def should_log_detail(prompt_idx: int) -> bool:
    return prompt_idx < DETAIL_LOG_PROMPT_LIMIT


def log_tensor_stats(name: str, tensor: Optional[torch.Tensor], prompt_idx: int, layer_idx: int):
    if tensor is None:
        logger.debug(f"[Prompt {prompt_idx}][Layer {layer_idx}] {name}: None")
        return

    try:
        t = tensor.detach().float()
        nan_count = torch.isnan(t).sum().item()
        inf_count = torch.isinf(t).sum().item()
        logger.debug(
            f"[Prompt {prompt_idx}][Layer {layer_idx}] {name}: "
            f"shape={tuple(t.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
            f"min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}, "
            f"nan={nan_count}, inf={inf_count}"
        )
    except Exception as e:
        logger.debug(f"[Prompt {prompt_idx}][Layer {layer_idx}] {name}: stats failed: {e}")


def get_attention_metadata(attn_module):
    cfg = getattr(attn_module, "config", None)
    if cfg is None:
        cfg = model.config

    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(cfg, "num_attention_heads", None)
    if num_heads is None:
        raise RuntimeError("Cannot determine num_heads from attention module or config.")

    num_kv_heads = getattr(attn_module, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if num_kv_heads is None:
        k_out = attn_module.k_proj.weight.shape[0]
        q_out = attn_module.q_proj.weight.shape[0]
        inferred_head_dim = q_out // num_heads
        num_kv_heads = k_out // inferred_head_dim

    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        q_out = attn_module.q_proj.weight.shape[0]
        if q_out % num_heads != 0:
            raise RuntimeError(
                f"q_proj out_features ({q_out}) is not divisible by num_heads ({num_heads})."
            )
        head_dim = q_out // num_heads

    if num_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})."
        )

    num_kv_groups = num_heads // num_kv_heads

    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_kv_groups": num_kv_groups,
    }


def repeat_kv_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, seqlen, num_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(bsz, seqlen, num_kv_heads, n_rep, head_dim)
    return x.reshape(bsz, seqlen, num_kv_heads * n_rep, head_dim)


def build_causal_mask_for_query_subset(
    seq_len: int,
    query_start: int,
    device: torch.device,
) -> torch.Tensor:
    q_positions = torch.arange(query_start, seq_len, device=device)
    k_positions = torch.arange(seq_len, device=device)
    return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)


def compute_ffn_importance_from_hffn(
    hffn: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    h_sq = hffn.float().pow(2).sum(dim=(0, 1))
    w_sq = down_proj_weight.float().pow(2).sum(dim=0)
    return h_sq * w_sq


def compute_q_importance_parallel(
    base_scores: torch.Tensor,
    base_probs: torch.Tensor,
    q_used: torch.Tensor,
    k_rep: torch.Tensor,
    head_dim: int,
    prompt_idx: int,
    layer_idx: int,
) -> torch.Tensor:
    device = base_scores.device
    _, num_heads, _, _ = base_scores.shape
    d = head_dim
    q_importance = torch.zeros(num_heads, d, device=device, dtype=torch.float32)

    for h in range(num_heads):
        scores_h = base_scores[:, h, :, :]
        probs_h = base_probs[:, h, :, :]
        q_h = q_used[:, :, h, :].float()
        k_h = k_rep[:, :, h, :].float()
        delta = q_h.unsqueeze(2) * k_h.unsqueeze(1)
        perturbed_scores = scores_h.unsqueeze(-1) - (delta / math.sqrt(head_dim))
        perturbed_probs = torch.softmax(perturbed_scores, dim=2)
        diff = perturbed_probs - probs_h.unsqueeze(-1)
        imp = diff.pow(2).sum(dim=(0, 1, 2))
        q_importance[h] = imp

    if should_log_detail(prompt_idx):
        log_tensor_stats("q_importance_parallel", q_importance, prompt_idx, layer_idx)

    return q_importance.reshape(-1).detach()


def compute_k_importance_parallel(
    base_scores: torch.Tensor,
    base_probs: torch.Tensor,
    q_used: torch.Tensor,
    k_orig: torch.Tensor,
    num_kv_groups: int,
    head_dim: int,
    prompt_idx: int,
    layer_idx: int,
) -> torch.Tensor:
    device = base_scores.device
    _, _, _, _ = base_scores.shape
    _, _, num_kv_heads, d = k_orig.shape
    k_importance = torch.zeros(num_kv_heads, d, device=device, dtype=torch.float32)

    for kvh in range(num_kv_heads):
        h_start = kvh * num_kv_groups
        h_end = (kvh + 1) * num_kv_groups
        k_h = k_orig[:, :, kvh, :].float()
        scores_grp = base_scores[:, h_start:h_end]
        probs_grp = base_probs[:, h_start:h_end]
        q_grp = q_used[:, :, h_start:h_end, :].float()
        imp_total = torch.zeros(d, device=device, dtype=torch.float32)

        for g in range(h_end - h_start):
            scores_h = scores_grp[:, g, :, :]
            probs_h = probs_grp[:, g, :, :]
            q_h = q_grp[:, :, g, :]
            delta = q_h.unsqueeze(2) * k_h.unsqueeze(1)
            perturbed_scores = scores_h.unsqueeze(-1) - (delta / math.sqrt(head_dim))
            perturbed_probs = torch.softmax(perturbed_scores, dim=2)
            diff = perturbed_probs - probs_h.unsqueeze(-1)
            imp = diff.pow(2).sum(dim=(0, 1, 2))
            imp_total += imp

        k_importance[kvh] = imp_total

    if should_log_detail(prompt_idx):
        log_tensor_stats("k_importance_parallel", k_importance, prompt_idx, layer_idx)

    return k_importance.reshape(-1).detach()


def compute_v_importance_linear(
    base_probs: torch.Tensor,
    v_orig: torch.Tensor,
    o_proj_weight: torch.Tensor,
    num_kv_groups: int,
    prompt_idx: int,
    layer_idx: int,
) -> torch.Tensor:
    device = base_probs.device
    _, num_heads, _, _ = base_probs.shape
    _, _, num_kv_heads, head_dim = v_orig.shape
    o_col_norm_sq = o_proj_weight.float().pow(2).sum(dim=0).view(num_heads, head_dim)
    v_importance = torch.zeros(num_kv_heads, head_dim, device=device, dtype=torch.float32)

    for kvh in range(num_kv_heads):
        h_start = kvh * num_kv_groups
        h_end = (kvh + 1) * num_kv_groups
        probs_grp = base_probs[:, h_start:h_end, :, :]
        v_h = v_orig[:, :, kvh, :].float()
        o_norm_grp = o_col_norm_sq[h_start:h_end, :]
        ctx = torch.einsum("bgqt,btd->bgqd", probs_grp, v_h)
        ctx_sq_sum = ctx.pow(2).sum(dim=(0, 2))
        v_importance[kvh] = (ctx_sq_sum * o_norm_grp).sum(dim=0)

    if should_log_detail(prompt_idx):
        log_tensor_stats("v_importance", v_importance, prompt_idx, layer_idx)

    return v_importance.reshape(-1).detach()


def select_by_threshold(importance: torch.Tensor,
                        active_fraction: float) -> Set[int]:
    """
    Given a 1D importance vector [D], select indices whose importance >= epsilon.
    """
    if importance.numel() == 0:
        return set()

    if importance.numel() < MIN_NEURONS_FOR_QUANTILE:
        return set()

    q = max(0.0, min(1.0, 1.0 - active_fraction))
    epsilon = torch.quantile(importance, q)

    active_mask = importance >= epsilon
    indices = torch.nonzero(active_mask, as_tuple=False).view(-1)

    return {idx.item() for idx in indices}


def select_global_by_threshold(
    layer_importance: Dict[int, torch.Tensor],
    active_fraction: float,
    module_name: str,
) -> Dict[int, Set[int]]:
    """
    Select active neurons with one global threshold per module by aggregating
    importance values from all layers.
    """
    result: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    non_empty = {
        layer_idx: imp
        for layer_idx, imp in layer_importance.items()
        if imp is not None and imp.numel() > 0
    }
    if not non_empty:
        logger.debug(f"select_global_by_threshold[{module_name}]: no activations captured")
        return result

    all_importance = torch.cat([imp.view(-1) for imp in non_empty.values()], dim=0)
    if all_importance.numel() < MIN_NEURONS_FOR_QUANTILE:
        logger.debug(
            f"select_global_by_threshold[{module_name}]: too few neurons "
            f"({all_importance.numel()} < {MIN_NEURONS_FOR_QUANTILE})"
        )
        return result

    q = max(0.0, min(1.0, 1.0 - active_fraction))
    epsilon = torch.quantile(all_importance, q)

    selected_total = 0
    for layer_idx, imp in non_empty.items():
        active_mask = imp >= epsilon
        indices = torch.nonzero(active_mask, as_tuple=False).view(-1)
        selected = {idx.item() for idx in indices}
        result[layer_idx] = selected
        selected_total += len(selected)

    logger.debug(
        f"select_global_by_threshold[{module_name}]: total_neurons={all_importance.numel()}, "
        f"selected={selected_total}, active_fraction={active_fraction}, epsilon={epsilon.item():.6f}"
    )
    return result


def detect_safety_neurons_threshold(
    prompt: str,
    prompt_idx: int = 0,
) -> Optional[
    Tuple[
        Dict[int, Set[int]],
        Dict[int, Set[int]],
        Dict[int, Set[int]],
        Dict[int, Set[int]],
        Dict[int, Set[int]],
    ]
]:
    ffn_up_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    ffn_down_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    q_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    k_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    v_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    try:
        if is_instruct_model(model_name):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {"input_ids": input_ids}
        else:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        captured_inputs: Dict[str, torch.Tensor] = {}

        def get_attn_pre_hook(name: str):
            def hook(module, args, kwargs):
                x = None
                if kwargs is not None and "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
                    x = kwargs["hidden_states"]
                elif args is not None and len(args) > 0 and args[0] is not None:
                    x = args[0]
                if x is None:
                    return
                x = x.detach()
                if CAPTURE_HIDDEN_TO_CPU:
                    x = x.to("cpu")
                captured_inputs[name] = x
            return hook

        def get_mlp_pre_hook(name: str):
            def hook(module, module_inputs):
                if not module_inputs:
                    return
                x = module_inputs[0].detach()
                if CAPTURE_HIDDEN_TO_CPU:
                    x = x.to("cpu")
                captured_inputs[name] = x
            return hook

        hooks = []
        for layer_idx in range(NUM_LAYERS):
            layer = model.model.layers[layer_idx]
            hooks.append(
                layer.self_attn.register_forward_pre_hook(
                    get_attn_pre_hook(f"layer_{layer_idx}_attn_in"),
                    with_kwargs=True,
                )
            )
            hooks.append(layer.mlp.register_forward_pre_hook(get_mlp_pre_hook(f"layer_{layer_idx}_mlp_in")))

        try:
            with torch.no_grad():
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=False,
                    return_dict=True,
                )

            ffn_up_importance: Dict[int, torch.Tensor] = {}
            ffn_down_importance: Dict[int, torch.Tensor] = {}
            q_importance: Dict[int, torch.Tensor] = {}
            k_importance: Dict[int, torch.Tensor] = {}
            v_importance: Dict[int, torch.Tensor] = {}

            for layer_idx in range(NUM_LAYERS):
                layer_t0 = time.perf_counter()
                layer = model.model.layers[layer_idx]
                attn_key = f"layer_{layer_idx}_attn_in"
                mlp_key = f"layer_{layer_idx}_mlp_in"

                if attn_key not in captured_inputs or mlp_key not in captured_inputs:
                    raise RuntimeError(f"Missing captured input at layer {layer_idx}")

                attn_in = captured_inputs.pop(attn_key)
                mlp_in = captured_inputs.pop(mlp_key)

                attn_dtype = layer.self_attn.q_proj.weight.dtype
                mlp_dtype = layer.mlp.up_proj.weight.dtype
                layer_device = layer.self_attn.q_proj.weight.device

                attn_x = attn_in.to(device=layer_device, dtype=attn_dtype, non_blocking=True)
                mlp_x = mlp_in.to(device=layer_device, dtype=mlp_dtype, non_blocking=True)

                gate_out = layer.mlp.gate_proj(mlp_x)
                up_out = layer.mlp.up_proj(mlp_x)
                hffn = F.silu(gate_out.float()) * up_out.float()
                ffn_imp = compute_ffn_importance_from_hffn(
                    hffn=hffn,
                    down_proj_weight=layer.mlp.down_proj.weight,
                )
                ffn_up_importance[layer_idx] = ffn_imp
                ffn_down_importance[layer_idx] = ffn_imp
                del gate_out, up_out, hffn, mlp_x, mlp_in

                q_proj = layer.self_attn.q_proj(attn_x).float()
                k_proj = layer.self_attn.k_proj(attn_x).float()
                v_proj = layer.self_attn.v_proj(attn_x).float()

                attn_meta = get_attention_metadata(layer.self_attn)
                num_heads = attn_meta["num_heads"]
                num_kv_heads = attn_meta["num_kv_heads"]
                head_dim = attn_meta["head_dim"]
                num_kv_groups = attn_meta["num_kv_groups"]

                bsz, full_seq_len, _ = q_proj.shape
                q = q_proj.reshape(bsz, full_seq_len, num_heads, head_dim)
                k = k_proj.reshape(bsz, full_seq_len, num_kv_heads, head_dim)
                v = v_proj.reshape(bsz, full_seq_len, num_kv_heads, head_dim)
                k_rep = repeat_kv_heads(k, num_kv_groups)

                query_start = 0 if ATTN_QUERY_WINDOW is None else max(0, full_seq_len - ATTN_QUERY_WINDOW)
                q_used = q[:, query_start:, :, :]

                base_scores = torch.einsum("bqhd,bthd->bhqt", q_used, k_rep) / math.sqrt(head_dim)
                causal_mask = build_causal_mask_for_query_subset(
                    seq_len=full_seq_len,
                    query_start=query_start,
                    device=base_scores.device,
                )
                base_scores = base_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), NEG_INF)

                if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                    key_mask = inputs["attention_mask"].to(base_scores.device).bool()
                    base_scores = base_scores.masked_fill(~key_mask.unsqueeze(1).unsqueeze(1), NEG_INF)

                base_probs = torch.softmax(base_scores, dim=-1).float()

                q_imp = compute_q_importance_parallel(
                    base_scores=base_scores,
                    base_probs=base_probs,
                    q_used=q_used,
                    k_rep=k_rep,
                    head_dim=head_dim,
                    prompt_idx=prompt_idx,
                    layer_idx=layer_idx,
                )
                k_imp = compute_k_importance_parallel(
                    base_scores=base_scores,
                    base_probs=base_probs,
                    q_used=q_used,
                    k_orig=k,
                    num_kv_groups=num_kv_groups,
                    head_dim=head_dim,
                    prompt_idx=prompt_idx,
                    layer_idx=layer_idx,
                )
                v_imp = compute_v_importance_linear(
                    base_probs=base_probs,
                    v_orig=v,
                    o_proj_weight=layer.self_attn.o_proj.weight,
                    num_kv_groups=num_kv_groups,
                    prompt_idx=prompt_idx,
                    layer_idx=layer_idx,
                )

                q_importance[layer_idx] = q_imp
                k_importance[layer_idx] = k_imp
                v_importance[layer_idx] = v_imp

                del (
                    attn_x, attn_in,
                    q_proj, k_proj, v_proj,
                    q, k, v, k_rep, q_used,
                    base_scores, base_probs,
                    q_imp, k_imp, v_imp,
                )

                layer_t1 = time.perf_counter()
                logger.debug(
                    f"[Prompt {prompt_idx}][Layer {layer_idx}] layer importance done in "
                    f"{layer_t1 - layer_t0:.3f}s"
                )

            ffn_up_dict = select_global_by_threshold(
                ffn_up_importance,
                FFN_ACTIVE_FRACTION,
                module_name="ffn_up",
            )
            ffn_down_dict = select_global_by_threshold(
                ffn_down_importance,
                FFN_ACTIVE_FRACTION,
                module_name="ffn_down",
            )
            q_dict = select_global_by_threshold(
                q_importance,
                ATTN_ACTIVE_FRACTION,
                module_name="q",
            )
            k_dict = select_global_by_threshold(
                k_importance,
                ATTN_ACTIVE_FRACTION,
                module_name="k",
            )
            v_dict = select_global_by_threshold(
                v_importance,
                ATTN_ACTIVE_FRACTION,
                module_name="v",
            )

        finally:
            for h in hooks:
                h.remove()
            captured_inputs.clear()

    except Exception as e:
        logger.exception(f"Error in neuron detection (Prompt {prompt_idx}): {e}")
        return None

    return ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict


def compute_intersection(neuron_sets_list: List[Dict[int, Set[int]]], module_name: str = "module") -> Dict[int, Set[int]]:
    """
    Compute intersection of neuron sets across all documents.
    """
    if not neuron_sets_list:
        logger.info(f"[compute_intersection][{module_name}] no neuron sets; reduced=0")
        return {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    intersection_dict: Dict[int, Set[int]] = {}
    all_layers = range(NUM_LAYERS)
    before_union_total = 0
    after_intersection_total = 0

    for layer_idx in all_layers:
        layer_sets = []
        for neuron_dict in neuron_sets_list:
            layer_sets.append(neuron_dict.get(layer_idx, set()))

        if not layer_sets:
            intersection_dict[layer_idx] = set()
            continue

        non_empty_sets = [s for s in layer_sets if s]
        if non_empty_sets:
            union_set = set.union(*non_empty_sets)
            common = set.intersection(*non_empty_sets)
        else:
            union_set = set()
            common = set()

        before_union_total += len(union_set)
        after_intersection_total += len(common)
        intersection_dict[layer_idx] = common

    reduced = before_union_total - after_intersection_total
    logger.info(
        f"[compute_intersection][{module_name}] prompts={len(neuron_sets_list)}, "
        f"before(union)={before_union_total}, after(intersection)={after_intersection_total}, reduced={reduced}"
    )

    return intersection_dict


def load_wikipedia_data(num_samples: int = 1000) -> List[str]:
    """
    Load Wikipedia data from Hugging Face.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        List of text samples from Wikipedia
    """
    logger.info("Loading Wikipedia dataset (subset: 20231101.en)...")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=False,
            cache_dir=os.path.join(SCRIPT_DIR, "wikipedia_cache")
        )
        
        # Extract text and sample
        texts = []
        logger.info(f"Sampling {num_samples} documents from Wikipedia...")
        
        # Get random indices (seed fixed for reproducibility across runs)
        total_size = len(dataset)
        random.seed(112)
        random_indices = random.sample(range(total_size), min(num_samples, total_size))
        
        for idx in tqdm(random_indices, desc="Loading Wikipedia docs"):
            try:
                text = dataset[idx]['text']
                if text.strip():
                    texts.append(text)
            except Exception as e:
                continue
        
        logger.info(f"Successfully loaded {len(texts)} Wikipedia samples")
        return texts
        
    except Exception as e:
        logger.error(f"Error loading Wikipedia dataset: {e}")
        logger.error("Please check your internet connection or HuggingFace access")
        raise


def main(argv):
    """
    Main function to detect foundation neurons from Wikipedia.
    
    Usage:
        python neuron_detection_foundation.py [num_docs] [model_path]
        
    Example:
        python neuron_detection_foundation.py 1000
        python neuron_detection_foundation.py 500 meta-llama/Llama-3.2-3B-Instruct
    """

    # =====================================================================
    # 로깅 설정: 파일 핸들러 추가
    # =====================================================================
    log_dir = os.path.join(SCRIPT_DIR, "logs", "foundation_neuron_detection")
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"foundation_neuron_detection_{log_timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    global FFN_ACTIVE_FRACTION, ATTN_ACTIVE_FRACTION

    args = parse_args(argv)
    FFN_ACTIVE_FRACTION = args.ffn_active_fraction
    ATTN_ACTIVE_FRACTION = args.attn_active_fraction
    initialize_model_and_tokenizer(args.model_name)

    num_docs = args.num_docs

    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log file: {log_file}")
    
    logger.info("="*70)
    logger.info("Foundation Neuron Detection from Wikipedia")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Num Wikipedia docs: {num_docs}")
    logger.info(f"FFN_ACTIVE_FRACTION: {FFN_ACTIVE_FRACTION}")
    logger.info(f"ATTN_ACTIVE_FRACTION: {ATTN_ACTIVE_FRACTION}\n")
    
    # Step 1: Load Wikipedia data
    wikipedia_docs = load_wikipedia_data(num_samples=num_docs)
    
    if not wikipedia_docs:
        logger.error("Failed to load Wikipedia data")
        sys.exit(1)
    
    # Step 2: Detect neurons for each document
    logger.info("\nDetecting foundation neurons for each Wikipedia document...")
    ffn_up_sets: List[Dict[int, Set[int]]] = []
    ffn_down_sets: List[Dict[int, Set[int]]] = []
    q_sets: List[Dict[int, Set[int]]] = []
    k_sets: List[Dict[int, Set[int]]] = []
    v_sets: List[Dict[int, Set[int]]] = []
    
    failed_count = 0
    successful_count = 0

    for idx, doc in enumerate(tqdm(wikipedia_docs, desc="Detecting neurons")):
        result = detect_safety_neurons_threshold(doc, prompt_idx=idx)
        if result is None:
            failed_count += 1
            logger.warning(f"Failed doc idx={idx}")
            continue

        ffn_up, ffn_down, q, k, v = result
        ffn_up_sets.append(ffn_up)
        ffn_down_sets.append(ffn_down)
        q_sets.append(q)
        k_sets.append(k)
        v_sets.append(v)
        successful_count += 1

    logger.info(f"Successfully processed {successful_count}/{len(wikipedia_docs)} documents (failed={failed_count})")
    
    # Step 3: Compute intersection (Foundation Neurons)
    logger.info("\nComputing foundation neuron intersections...")
    ffn_up_foundation = compute_intersection(ffn_up_sets, module_name="ffn_up")
    ffn_down_foundation = compute_intersection(ffn_down_sets, module_name="ffn_down")
    q_foundation = compute_intersection(q_sets, module_name="q")
    k_foundation = compute_intersection(k_sets, module_name="k")
    v_foundation = compute_intersection(v_sets, module_name="v")
    
    # Step 4: Save results
    output_dir = os.path.join(SCRIPT_DIR, "output_neurons")
    os.makedirs(output_dir, exist_ok=True)
    clean_model_name = model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"utility_neurons_{len(ffn_up_sets)}_{timestamp}.txt")
    
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        # safety_neuron_detection.py와 동일한 JSON line 포맷
        f.write(json.dumps({str(k): list(v) for k, v in ffn_up_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in ffn_down_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in q_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in k_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in v_foundation.items()}) + "\n")
    
    # Statistics
    logger.info("\n" + "="*70)
    logger.info("Utility Neuron Detection Results")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Wikipedia documents processed: {len(ffn_up_sets)}/{len(wikipedia_docs)}\n")
    
    total_foundation_neurons = 0
    total_ffn_neurons = 0
    total_attn_neurons = 0
    
    for layer_idx in range(NUM_LAYERS):
        ffn_up_count = len(ffn_up_foundation.get(layer_idx, set()))
        ffn_down_count = len(ffn_down_foundation.get(layer_idx, set()))
        q_count = len(q_foundation.get(layer_idx, set()))
        k_count = len(k_foundation.get(layer_idx, set()))
        v_count = len(v_foundation.get(layer_idx, set()))
        
        ffn_count = ffn_up_count + ffn_down_count
        attn_count = q_count + k_count + v_count
        layer_neurons = ffn_count + attn_count
        
        if layer_neurons > 0:
            logger.info(f"Layer {layer_idx}: {layer_neurons} foundation neurons (FFN: {ffn_count}, Attention: {attn_count})")
            total_foundation_neurons += layer_neurons
            total_ffn_neurons += ffn_count
            total_attn_neurons += attn_count
    
    total_model_neurons = calculate_model_total_neurons()
    foundation_sparsity = total_foundation_neurons / total_model_neurons if total_model_neurons > 0 else 0
    logger.info(f"\nTotal foundation neurons detected: {total_foundation_neurons} (FFN: {total_ffn_neurons}, Attention: {total_attn_neurons})")
    logger.info(f"Total model neurons (q/k/v/o + gate/up/down): {total_model_neurons:,}")
    logger.info(f"Foundation sparsity: {foundation_sparsity*100:.4f}%")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"Log: {log_file}")
    logger.info("="*70)
    
    # Print next steps
    logger.info("\n📋 Next Steps:")
    logger.info(f"1. ✓ Safety Neurons: Already detected")
    logger.info(f"2. ✓ Foundation Neurons: Just detected (saved above)")
    logger.info(f"3. → Run: python neuron_detection_rsn.py")
    logger.info(f"   to compute RSN = Safety - (Safety ∩ Foundation)")


if __name__ == "__main__":
    main(sys.argv[1:])
