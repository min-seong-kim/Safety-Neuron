'''
python safety_neuron_detection_v2.py 4994 \
    --model_name meta-llama/Llama-3.1-8B \
    --ffn_active_fraction 0.05 \
    --attn_active_fraction 0.05

python safety_neuron_detection_v2.py 200 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --ffn_active_fraction 0.4 \
    --attn_active_fraction 0.4
'''
from neuron_percentage_utils import calculate_total_model_neurons_from_config

import os
import argparse
from typing import Dict, Set, List, Tuple, Optional
import sys
import json
from tqdm import tqdm
import logging
import random
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 로거 초기 설정 (나중에 파일 핸들러 추가됨)
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

# ------------------------------------------------------------------
# Threshold hyperparameters
# ------------------------------------------------------------------
DEFAULT_FFN_ACTIVE_FRACTION = 0.15
DEFAULT_ATTN_ACTIVE_FRACTION = 0.15
FFN_ACTIVE_FRACTION = DEFAULT_FFN_ACTIVE_FRACTION
ATTN_ACTIVE_FRACTION = DEFAULT_ATTN_ACTIVE_FRACTION
MIN_NEURONS_FOR_QUANTILE = 10

# ------------------------------------------------------------------
# Accelerated detection hyperparameters
# ------------------------------------------------------------------
ATTN_QUERY_WINDOW = None      # None이면 전체 query position 사용
CAPTURE_HIDDEN_TO_CPU = False # hidden input을 GPU에 유지
DETAIL_LOG_PROMPT_LIMIT = 3
NEG_INF = -1e9

USE_FULL_DIM_PARALLEL_QK = True


def initialize_model_and_tokenizer(selected_model_name: str):
    """Initialize global model/tokenizer after CLI args are parsed."""
    global model_name, model, tokenizer, NUM_LAYERS

    model_name = selected_model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Safety neuron detection with configurable model and thresholds"
    )
    parser.add_argument(
        "num_prompts",
        type=int,
        help="Number of prompts to process from circuit_breakers_train.json",
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
    Same denominator as calculate_safety_neuron_percentage.py:
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
    """
    Robustly extract attention metadata across different HF LlamaAttention implementations.
    """
    cfg = getattr(attn_module, "config", None)
    if cfg is None:
        cfg = model.config

    # num_heads
    num_heads = getattr(attn_module, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(cfg, "num_attention_heads", None)
    if num_heads is None:
        raise RuntimeError("Cannot determine num_heads from attention module or config.")

    # num_kv_heads
    num_kv_heads = getattr(attn_module, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if num_kv_heads is None:
        # fallback: infer from k_proj output shape
        k_out = attn_module.k_proj.weight.shape[0]
        q_out = attn_module.q_proj.weight.shape[0]
        inferred_head_dim = q_out // num_heads
        num_kv_heads = k_out // inferred_head_dim

    # head_dim
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
    """
    x: [B, T, H_kv, D]
    return: [B, T, H_q, D]
    """
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
    """
    Returns bool mask of shape [Q, T]
    where Q = seq_len - query_start
    """
    q_positions = torch.arange(query_start, seq_len, device=device)  # [Q]
    k_positions = torch.arange(seq_len, device=device)               # [T]
    return k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)      # [Q, T]


def compute_ffn_importance_from_hffn(
    hffn: torch.Tensor,             # [B, T, I]
    down_proj_weight: torch.Tensor  # [H, I]
) -> torch.Tensor:
    """
    Eq. (9)-style efficient importance:
    contribution of neuron k is hffn[..., k] * Wdown[:, k]

    importance[k] = || contribution_k ||_2^2
                  = sum_{b,t} hffn[b,t,k]^2 * ||Wdown[:,k]||_2^2
    """
    h_sq = hffn.float().pow(2).sum(dim=(0, 1))                # [I]
    w_sq = down_proj_weight.float().pow(2).sum(dim=0)         # [I]
    return h_sq * w_sq                                        # [I]


def compute_q_importance_parallel(
    base_scores: torch.Tensor,  # [B, Hq, Q, T]
    base_probs: torch.Tensor,   # [B, Hq, Q, T]
    q_used: torch.Tensor,       # [B, Q, Hq, D]
    k_rep: torch.Tensor,        # [B, T, Hq, D]
    head_dim: int,
    prompt_idx: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    More paper-like parallel Q importance.
    Per head, compute all D dimensions simultaneously.
    Returns flattened importance [Hq * D]
    """
    device = base_scores.device
    _, num_heads, _, _ = base_scores.shape
    d = head_dim

    q_importance = torch.zeros(num_heads, d, device=device, dtype=torch.float32)

    for h in range(num_heads):
        scores_h = base_scores[:, h, :, :]   # [B, Q, T]
        probs_h = base_probs[:, h, :, :]     # [B, Q, T]
        q_h = q_used[:, :, h, :].float()     # [B, Q, D]
        k_h = k_rep[:, :, h, :].float()      # [B, T, D]

        # delta: [B, Q, T, D]
        delta = q_h.unsqueeze(2) * k_h.unsqueeze(1)

        # perturbed_scores: [B, Q, T, D]
        perturbed_scores = scores_h.unsqueeze(-1) - (delta / math.sqrt(head_dim))
        perturbed_probs = torch.softmax(perturbed_scores, dim=2)

        diff = perturbed_probs - probs_h.unsqueeze(-1)
        imp = diff.pow(2).sum(dim=(0, 1, 2))   # [D]
        q_importance[h] = imp

        if should_log_detail(prompt_idx):
            logger.debug(
                f"[Prompt {prompt_idx}][Layer {layer_idx}][Q head {h}] "
                f"delta_shape={tuple(delta.shape)}, imp_shape={tuple(imp.shape)}"
            )

    if should_log_detail(prompt_idx):
        log_tensor_stats("q_importance_parallel", q_importance, prompt_idx, layer_idx)

    return q_importance.reshape(-1).detach()


def compute_k_importance_parallel(
    base_scores: torch.Tensor,  # [B, Hq, Q, T]
    base_probs: torch.Tensor,   # [B, Hq, Q, T]
    q_used: torch.Tensor,       # [B, Q, Hq, D]
    k_orig: torch.Tensor,       # [B, T, Hkv, D]
    num_kv_groups: int,
    head_dim: int,
    prompt_idx: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    More paper-like parallel K importance.
    Per KV head, compute all D dimensions simultaneously.
    Returns flattened importance [Hkv * D]
    """
    device = base_scores.device
    _, _, _, _ = base_scores.shape
    _, _, num_kv_heads, d = k_orig.shape

    k_importance = torch.zeros(num_kv_heads, d, device=device, dtype=torch.float32)

    for kvh in range(num_kv_heads):
        h_start = kvh * num_kv_groups
        h_end = (kvh + 1) * num_kv_groups

        k_h = k_orig[:, :, kvh, :].float()              # [B, T, D]
        scores_grp = base_scores[:, h_start:h_end]      # [B, g, Q, T]
        probs_grp = base_probs[:, h_start:h_end]        # [B, g, Q, T]
        q_grp = q_used[:, :, h_start:h_end, :].float()  # [B, Q, g, D]

        imp_total = torch.zeros(d, device=device, dtype=torch.float32)

        for g in range(h_end - h_start):
            scores_h = scores_grp[:, g, :, :]   # [B, Q, T]
            probs_h = probs_grp[:, g, :, :]     # [B, Q, T]
            q_h = q_grp[:, :, g, :]             # [B, Q, D]

            # delta: [B, Q, T, D]
            delta = q_h.unsqueeze(2) * k_h.unsqueeze(1)

            perturbed_scores = scores_h.unsqueeze(-1) - (delta / math.sqrt(head_dim))
            perturbed_probs = torch.softmax(perturbed_scores, dim=2)

            diff = perturbed_probs - probs_h.unsqueeze(-1)
            imp = diff.pow(2).sum(dim=(0, 1, 2))   # [D]
            imp_total += imp

        k_importance[kvh] = imp_total

        if should_log_detail(prompt_idx):
            logger.debug(
                f"[Prompt {prompt_idx}][Layer {layer_idx}][K kv_head {kvh}] "
                f"imp_total_shape={tuple(imp_total.shape)}"
            )

    if should_log_detail(prompt_idx):
        log_tensor_stats("k_importance_parallel", k_importance, prompt_idx, layer_idx)

    return k_importance.reshape(-1).detach()


def compute_v_importance_linear(
    base_probs: torch.Tensor,      # [B, Hq, Q, T]
    v_orig: torch.Tensor,          # [B, T, Hkv, D]
    o_proj_weight: torch.Tensor,   # [hidden, hidden]
    num_kv_groups: int,
    prompt_idx: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    V is in the linear part after softmax, so we use an Eq. (9)-style linear importance.

    Approximation:
    - compute context contribution caused by each original V neuron
    - scale by || corresponding o_proj column ||^2
    - aggregate over all repeated heads that share the same KV head

    Returns flattened importance [Hkv * D]
    """
    device = base_probs.device
    _, num_heads, q_len, seq_len = base_probs.shape
    _, _, num_kv_heads, head_dim = v_orig.shape

    # o_proj columns correspond to concatenated [Hq, D]
    o_col_norm_sq = o_proj_weight.float().pow(2).sum(dim=0).view(num_heads, head_dim)  # [Hq, D]

    v_importance = torch.zeros(num_kv_heads, head_dim, device=device, dtype=torch.float32)

    for kvh in range(num_kv_heads):
        h_start = kvh * num_kv_groups
        h_end = (kvh + 1) * num_kv_groups

        probs_grp = base_probs[:, h_start:h_end, :, :]         # [B, g, Q, T]
        v_h = v_orig[:, :, kvh, :].float()                     # [B, T, D]
        o_norm_grp = o_col_norm_sq[h_start:h_end, :]           # [g, D]

        # ctx[b, g, q, d] = sum_t probs[b,g,q,t] * v[b,t,d]
        ctx = torch.einsum("bgqt,btd->bgqd", probs_grp, v_h)   # [B, g, Q, D]
        ctx_sq_sum = ctx.pow(2).sum(dim=(0, 2))                # [g, D]

        v_importance[kvh] = (ctx_sq_sum * o_norm_grp).sum(dim=0)

    if should_log_detail(prompt_idx):
        log_tensor_stats("v_importance", v_importance, prompt_idx, layer_idx)

    return v_importance.reshape(-1).detach()

def select_by_threshold(importance: torch.Tensor,
                        active_fraction: float) -> Set[int]:
    """
    Given a 1D importance vector [D], select indices whose importance >= epsilon,
    where epsilon is chosen as the (1 - active_fraction) quantile.

    importance: torch.Tensor, shape [D]
    active_fraction: e.g., 0.005 (top 0.5%)

    Returns:
        Set of neuron indices (as integers) above threshold.
        - Importance는 activation의 절댓값(L1)에 기반
        - 각 query x마다 상위 active_fraction%의 뉴런을 선택 (Nx)
    """
    if importance.numel() == 0:
        logger.info("select_by_threshold: Empty importance tensor")
        return set()

    # If too few neurons, fall back to empty set
    if importance.numel() < MIN_NEURONS_FOR_QUANTILE:
        logger.info(f"select_by_threshold: Too few neurons ({importance.numel()} < {MIN_NEURONS_FOR_QUANTILE})")
        return set()

    # Compute epsilon = quantile(importance, 1 - active_fraction)
    # Eq. (2): Nx = { N_i^(l) | Imp(N_i^(l)|x) >= epsilon }
    q = max(0.0, min(1.0, 1.0 - active_fraction))
    epsilon = torch.quantile(importance, q)

    # Select neurons above threshold
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

    layer_importance: layer_idx -> importance tensor [D_layer]
    active_fraction: global top-k fraction to keep across all layers
    module_name: only for debug logging

    Returns:
      layer_idx -> selected neuron index set within that layer
    """
    result: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    non_empty = {
        layer_idx: imp
        for layer_idx, imp in layer_importance.items()
        if imp is not None and imp.numel() > 0
    }
    if not non_empty:
        logger.info(f"select_global_by_threshold[{module_name}]: no activations captured")
        return result

    all_importance = torch.cat([imp.view(-1) for imp in non_empty.values()], dim=0)
    if all_importance.numel() < MIN_NEURONS_FOR_QUANTILE:
        logger.info(
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
        selected = set(indices.tolist())
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
    """
    Accelerated Safety Neuron Detection version.

    Returns:
      ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict
    """
    ffn_up_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    ffn_down_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    q_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    k_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}
    v_dict: Dict[int, Set[int]] = {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    try:
        # ------------------------------------------------------------
        # 1) Tokenize
        # ------------------------------------------------------------
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

        seq_len = inputs["input_ids"].shape[1]
        logger.debug(
            f"[Prompt {prompt_idx}] tokenized: seq_len={seq_len}, "
            f"has_attention_mask={'attention_mask' in inputs}, device={device}"
        )

        # ------------------------------------------------------------
        # 2) Capture only hidden-state inputs of self_attn / mlp
        #    (much safer than storing q/k/v/up/gate outputs for all layers)
        # ------------------------------------------------------------
        captured_inputs: Dict[str, torch.Tensor] = {}

        def get_attn_pre_hook(name: str):
            def hook(module, args, kwargs):
                x = None

                # 1) keyword argument로 들어오는 경우
                if kwargs is not None and "hidden_states" in kwargs and kwargs["hidden_states"] is not None:
                    x = kwargs["hidden_states"]

                # 2) 혹시 positional로 들어오는 경우 fallback
                elif args is not None and len(args) > 0 and args[0] is not None:
                    x = args[0]

                if x is None:
                    logger.debug(f"[Prompt {prompt_idx}] attn pre_hook {name}: hidden_states not found")
                    return

                x = x.detach()
                if CAPTURE_HIDDEN_TO_CPU:
                    x = x.to("cpu")
                    logger.debug(f"[Prompt {prompt_idx}] captured {name} on CPU")
                else:
                    logger.debug(f"[Prompt {prompt_idx}] captured {name} on GPU")

                captured_inputs[name] = x

            return hook


        def get_mlp_pre_hook(name: str):
            def hook(module, module_inputs):
                if not module_inputs:
                    logger.debug(f"[Prompt {prompt_idx}] mlp pre_hook {name}: empty input tuple")
                    return

                x = module_inputs[0].detach()
                if CAPTURE_HIDDEN_TO_CPU:
                    x = x.to("cpu")
                    logger.debug(f"[Prompt {prompt_idx}] captured {name} on CPU")
                else:
                    logger.debug(f"[Prompt {prompt_idx}] captured {name} on GPU")

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

            hooks.append(
                layer.mlp.register_forward_pre_hook(
                    get_mlp_pre_hook(f"layer_{layer_idx}_mlp_in")
                )
            )

        try:
            # --------------------------------------------------------
            # 3) Forward pass once
            # --------------------------------------------------------
            with torch.no_grad():
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=False,
                    return_dict=True,
                )

            expected_keys = []
            for li in range(NUM_LAYERS):
                expected_keys.append(f"layer_{li}_attn_in")
                expected_keys.append(f"layer_{li}_mlp_in")

            missing_keys = [k for k in expected_keys if k not in captured_inputs]

            logger.debug(
                f"[Prompt {prompt_idx}] forward done. captured_keys={len(captured_inputs)} "
                f"(expected={NUM_LAYERS * 2}), missing_keys_count={len(missing_keys)}"
            )

            if should_log_detail(prompt_idx):
                logger.debug(f"[Prompt {prompt_idx}] missing_keys={missing_keys}")

            # --------------------------------------------------------
            # 4) Layer-wise accelerated importance computation
            # --------------------------------------------------------
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

                if attn_key not in captured_inputs:
                    raise RuntimeError(f"Missing captured input: {attn_key}")
                if mlp_key not in captured_inputs:
                    raise RuntimeError(f"Missing captured input: {mlp_key}")

                try:
                    # ------------------------------------------------
                    # Bring hidden inputs back to module device/dtype
                    # ------------------------------------------------
                    attn_in = captured_inputs.pop(attn_key)
                    mlp_in = captured_inputs.pop(mlp_key)

                    attn_dtype = layer.self_attn.q_proj.weight.dtype
                    mlp_dtype = layer.mlp.up_proj.weight.dtype
                    layer_device = layer.self_attn.q_proj.weight.device

                    attn_x = attn_in.to(device=layer_device, dtype=attn_dtype, non_blocking=True)
                    mlp_x = mlp_in.to(device=layer_device, dtype=mlp_dtype, non_blocking=True)

                    if should_log_detail(prompt_idx):
                        logger.debug(
                            f"[Prompt {prompt_idx}][Layer {layer_idx}] "
                            f"attn_in_shape={tuple(attn_x.shape)}, mlp_in_shape={tuple(mlp_x.shape)}, "
                            f"attn_dtype={attn_x.dtype}, mlp_dtype={mlp_x.dtype}, device={attn_x.device}"
                        )

                    # ------------------------------------------------
                    # FFN accelerated importance
                    # hffn = SiLU(Wgate(x)) * Wup(x)
                    # importance via Wdown
                    # ------------------------------------------------
                    gate_out = layer.mlp.gate_proj(mlp_x)          # [B, T, I]
                    up_out = layer.mlp.up_proj(mlp_x)              # [B, T, I]
                    hffn = F.silu(gate_out.float()) * up_out.float()

                    if should_log_detail(prompt_idx):
                        log_tensor_stats("gate_out", gate_out, prompt_idx, layer_idx)
                        log_tensor_stats("up_out", up_out, prompt_idx, layer_idx)
                        log_tensor_stats("hffn", hffn, prompt_idx, layer_idx)

                    ffn_imp = compute_ffn_importance_from_hffn(
                        hffn=hffn,
                        down_proj_weight=layer.mlp.down_proj.weight,
                    )
                    ffn_up_importance[layer_idx] = ffn_imp
                    ffn_down_importance[layer_idx] = ffn_imp  # paper: Wdown importance can be derived the same way

                    # Free FFN temps
                    del gate_out, up_out, hffn, mlp_x, mlp_in

                    # ------------------------------------------------
                    # Attention accelerated importance
                    # ------------------------------------------------
                    q_proj = layer.self_attn.q_proj(attn_x).float()  # [B, T, Hq*D]
                    k_proj = layer.self_attn.k_proj(attn_x).float()  # [B, T, Hkv*D]
                    v_proj = layer.self_attn.v_proj(attn_x).float()  # [B, T, Hkv*D]

                    attn_meta = get_attention_metadata(layer.self_attn)

                    num_heads = attn_meta["num_heads"]
                    num_kv_heads = attn_meta["num_kv_heads"]
                    head_dim = attn_meta["head_dim"]
                    num_kv_groups = attn_meta["num_kv_groups"]

                    if should_log_detail(prompt_idx):
                        logger.debug(
                            f"[Prompt {prompt_idx}][Layer {layer_idx}] attn_meta: "
                            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
                            f"head_dim={head_dim}, num_kv_groups={num_kv_groups}, "
                            f"q_proj_weight_shape={tuple(layer.self_attn.q_proj.weight.shape)}, "
                            f"k_proj_weight_shape={tuple(layer.self_attn.k_proj.weight.shape)}, "
                            f"v_proj_weight_shape={tuple(layer.self_attn.v_proj.weight.shape)}, "
                            f"o_proj_weight_shape={tuple(layer.self_attn.o_proj.weight.shape)}"
                        )

                    bsz, full_seq_len, _ = q_proj.shape

                    q = q_proj.reshape(bsz, full_seq_len, num_heads, head_dim)
                    k = k_proj.reshape(bsz, full_seq_len, num_kv_heads, head_dim)
                    v = v_proj.reshape(bsz, full_seq_len, num_kv_heads, head_dim)

                    k_rep = repeat_kv_heads(k, num_kv_groups)  # [B, T, Hq, D]

                    if ATTN_QUERY_WINDOW is None:
                        query_start = 0
                    else:
                        query_start = max(0, full_seq_len - ATTN_QUERY_WINDOW)

                    q_used = q[:, query_start:, :, :]
                    q_len = q_used.shape[1]

                    if should_log_detail(prompt_idx):
                        logger.debug(
                            f"[Prompt {prompt_idx}][Layer {layer_idx}] "
                            f"full_seq_len={full_seq_len}, query_start={query_start}, q_len_used={q_len}, "
                            f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
                            f"num_kv_groups={num_kv_groups}"
                        )
                        log_tensor_stats("q_proj", q_proj, prompt_idx, layer_idx)
                        log_tensor_stats("k_proj", k_proj, prompt_idx, layer_idx)
                        log_tensor_stats("v_proj", v_proj, prompt_idx, layer_idx)

                    # Base attention scores using only last q_len query positions
                    base_scores = torch.einsum("bqhd,bthd->bhqt", q_used, k_rep) / math.sqrt(head_dim)  # [B,Hq,Q,T]

                    causal_mask = build_causal_mask_for_query_subset(
                        seq_len=full_seq_len,
                        query_start=query_start,
                        device=base_scores.device,
                    )  # [Q, T]

                    base_scores = base_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), NEG_INF)

                    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
                        key_mask = inputs["attention_mask"].to(base_scores.device).bool()  # [B, T]
                        base_scores = base_scores.masked_fill(~key_mask.unsqueeze(1).unsqueeze(1), NEG_INF)

                    base_probs = torch.softmax(base_scores, dim=-1).float()

                    if should_log_detail(prompt_idx):
                        log_tensor_stats("base_scores", base_scores, prompt_idx, layer_idx)
                        log_tensor_stats("base_probs", base_probs, prompt_idx, layer_idx)

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

                    # Free attention temps
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

                except Exception as layer_e:
                    logger.exception(
                        f"[Prompt {prompt_idx}][Layer {layer_idx}] "
                        f"layer-wise accelerated importance failed: {layer_e}"
                    )
                    raise

            # --------------------------------------------------------
            # 5) Global threshold per module across all layers
            # --------------------------------------------------------
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

            if should_log_detail(prompt_idx):
                ffn_up_total = sum(len(v) for v in ffn_up_dict.values())
                ffn_down_total = sum(len(v) for v in ffn_down_dict.values())
                q_total = sum(len(v) for v in q_dict.values())
                k_total = sum(len(v) for v in k_dict.values())
                v_total = sum(len(v) for v in v_dict.values())

                logger.debug(
                    f"[Prompt {prompt_idx}] selected neurons after threshold: "
                    f"ffn_up={ffn_up_total}, ffn_down={ffn_down_total}, "
                    f"q={q_total}, k={k_total}, v={v_total}"
                )

        finally:
            for h in hooks:
                h.remove()
            captured_inputs.clear()

    except Exception as e:
        logger.exception(f"Error in neuron detection (Prompt {prompt_idx}): {e}")
        return None

    return ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict


def compute_intersection(
    neuron_sets_list: List[Dict[int, Set[int]]],
    module_name: str = "module"
) -> Dict[int, Set[int]]:
    """
    Compute exact intersection across all prompts (Eq. 3).

    Eq. (3): N_safe = ⋂_{x in X} Nx
    - A neuron must appear in EVERY prompt-specific set Nx.
    - If any prompt has an empty set at a layer, intersection becomes empty.
    """
    if not neuron_sets_list:
        logger.info(f"[compute_intersection][{module_name}] no neuron sets; reduced=0")
        return {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    intersection_dict: Dict[int, Set[int]] = {}
    before_union_total = 0
    after_intersection_total = 0

    for layer_idx in range(NUM_LAYERS):
        layer_sets = [
            neuron_dict.get(layer_idx, set())
            for neuron_dict in neuron_sets_list
        ]

        # Union is just for logging/diagnostics
        union_set = set().union(*layer_sets) if layer_sets else set()

        # Exact intersection across ALL prompts
        if not layer_sets:
            common = set()
        else:
            common = set(layer_sets[0])
            for s in layer_sets[1:]:
                common &= s

        before_union_total += len(union_set)
        after_intersection_total += len(common)
        intersection_dict[layer_idx] = common

    reduced = before_union_total - after_intersection_total
    logger.info(
        f"[compute_intersection][{module_name}] prompts={len(neuron_sets_list)}, "
        f"before(union)={before_union_total}, after(intersection)={after_intersection_total}, reduced={reduced}"
    )

    return intersection_dict


def main(argv):
    global FFN_ACTIVE_FRACTION, ATTN_ACTIVE_FRACTION

    args = parse_args(argv)

    FFN_ACTIVE_FRACTION = args.ffn_active_fraction
    ATTN_ACTIVE_FRACTION = args.attn_active_fraction

    initialize_model_and_tokenizer(args.model_name)

    # =====================================================================
    # 로깅 설정: 파일 핸들러 추가
    # =====================================================================
    log_dir = os.path.join(SCRIPT_DIR, "logs", "safety_neuron_detection")
    os.makedirs(log_dir, exist_ok=True)
    
    # 파일 이름: 현재 날짜 및 시간
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"safety_neuron_detection_{log_timestamp}.log")
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러도 추가 (기존 출력 유지)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # 로거에 핸들러 추가
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"FFN_ACTIVE_FRACTION: {FFN_ACTIVE_FRACTION}, ATTN_ACTIVE_FRACTION: {ATTN_ACTIVE_FRACTION}")

    num_prompts = args.num_prompts
    logger.info(f"Number of prompts to process: {num_prompts}")
    file_path = os.path.join(SCRIPT_DIR, "corpus_all", "circuit_breakers_train.json")
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        logger.error(f"No valid 'prompt' entries found in: {file_path}")
        sys.exit(1)

    if len(records) > num_prompts:
        records = records[:num_prompts]

    lines = [item.get("prompt", "") for item in records]

    logger.info(f"Processing {len(lines)} prompts from {file_path}")

    # 각 prompt x에 대해 Nx를 수집
    ffn_up_sets: List[Dict[int, Set[int]]] = []
    ffn_down_sets: List[Dict[int, Set[int]]] = []
    q_sets: List[Dict[int, Set[int]]] = []
    k_sets: List[Dict[int, Set[int]]] = []
    v_sets: List[Dict[int, Set[int]]] = []

    failed_count = 0
    successful_count = 0

    for idx, prompt in enumerate(tqdm(lines, desc="Detecting neurons")):
        result = detect_safety_neurons_threshold(prompt, prompt_idx=idx)

        if result is None:
            failed_count += 1
            logger.warning(f"Failed prompt idx={idx}")
            continue

        ffn_up, ffn_down, q, k, v = result
        ffn_up_sets.append(ffn_up)
        ffn_down_sets.append(ffn_down)
        q_sets.append(q)
        k_sets.append(k)
        v_sets.append(v)
        successful_count += 1
    logger.info(f"Detection complete: success={successful_count}, failed={failed_count}")

    # Eq. (3): N_safe = ⋂_x N_x
    ffn_up_common = compute_intersection(ffn_up_sets, module_name="ffn_up")
    ffn_down_common = compute_intersection(ffn_down_sets, module_name="ffn_down")
    q_common = compute_intersection(q_sets, module_name="q")
    k_common = compute_intersection(k_sets, module_name="k")
    v_common = compute_intersection(v_sets, module_name="v")

    # 결과 저장
    output_dir = os.path.join(SCRIPT_DIR, "output_neurons")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"safety_neuron_accelerated_{log_timestamp}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        # Dict[int, Set[int]] -> str으로 저장
        f.write(json.dumps({str(k): list(v) for k, v in ffn_up_common.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in ffn_down_common.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in q_common.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in k_common.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in v_common.items()}) + "\n")

    # 최종 결과 계산
    total_safety_neurons = 0
    for layer_idx in range(NUM_LAYERS):
        ffn_up_count = len(ffn_up_common.get(layer_idx, set()))
        ffn_down_count = len(ffn_down_common.get(layer_idx, set()))
        q_count = len(q_common.get(layer_idx, set()))
        k_count = len(k_common.get(layer_idx, set()))
        v_count = len(v_common.get(layer_idx, set()))
        total_safety_neurons += ffn_up_count + ffn_down_count + q_count + k_count + v_count

    total_model_neurons = calculate_model_total_neurons()
    actual_sparsity = total_safety_neurons / total_model_neurons if total_model_neurons > 0 else 0
    
    logger.info(f"\n{'='*70}")
    logger.info("Safety Neuron Detection Results")
    logger.info(f"{'='*70}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Total safety neurons: {total_safety_neurons:,}")
    logger.info(f"Total model neurons (q/k/v/o + gate/up/down): {total_model_neurons:,}")
    logger.info(f"Detected safety neuron percentage: {actual_sparsity*100:.4f}%")
    logger.info(f"Output: {output_file}")
    logger.info(f"Log: {log_file}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
