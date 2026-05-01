"""
Safety neuron detection in WaRP basis-rotated space.

Instead of detecting original output neurons (output dimension indices of W),
this script detects which WaRP basis directions (columns of U from Phase 1 SVD)
are safety-critical.  These rotated "neurons" correspond to the principal
components of the activation space that carry safety-relevant information.

Motivation
----------
In WaRP, every linear layer weight is reparameterised as W = coeff @ U^T where
- U  : [in_dim, in_dim] – orthonormal matrix of principal input directions (Phase 1)
- coeff : [out_dim, in_dim] – coefficients in the rotated basis

A "neuron" in the rotated space = column j of coeff = how the j-th principal
direction u_j contributes to all output neurons.  For sn-tune applied to WaRP,
we want to freeze the basis-direction columns that are safety-critical, not the
original output-dimension rows.

Importance metric (per basis direction j)
------------------------------------------
For FFN-down  (basis for down_proj input = intermediate activation space):
    z_j       = hffn @ u_j                   ∈ R^{B×T}
    importance(j) = Σ_{b,t} z_j^2 · ‖W_down @ u_j‖²

For FFN-up (basis for up_proj / gate_proj shared input = MLP hidden state):
    z_j       = mlp_x @ u_j                  ∈ R^{B×T}
    importance(j) = Σ_{b,t} z_j^2 · Σ_k ((W_up[k,j]² + W_gate[k,j]²) · ‖W_down[:,k]‖²)

For Attention Q / K (basis for q_proj / k_proj input = attn hidden state):
    z_j       = attn_x @ u_j                 ∈ R^{B×T}
    importance(j) = Σ_{b,t} z_j^2 · ‖W_proj_rot[:, j]‖²

For Attention V (same input basis, weighted by o_proj column norms):
    importance(j) = Σ_{b,t} z_j^2 · Σ_d (W_v_rot[d,j]² · ‖W_o[:,d]‖²)

Usage
-----
python safety_neuron_detection_rotation.py 4994 \
    --model_name meta-llama/Llama-2-7b-hf \
    --basis_dir /home/yonsei_jong/Safety-WaRP-LLM/checkpoints/phase1_20260428_000405/basis \
    --layer_types attn_q,attn_k,attn_v,ffn_down,ffn_up \
    --top_k_basis 4994 \
    --ffn_active_fraction 0.2 \
    --attn_active_fraction 0.2

Output file format (same as safety_neuron_detection_v2.py)
-----------------------------------------------------------
5 JSON lines: ffn_up, ffn_down, q, k, v
Each line: {"layer_idx": [basis_direction_indices, ...], ...}
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(112)
torch.manual_seed(112)

# ------------------------------------------------------------------
# Global state (initialised in main / initialize_*)
# ------------------------------------------------------------------
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
model_name: str = DEFAULT_MODEL_NAME
tokenizer = None
model = None
NUM_LAYERS: int = 0

# WaRP basis: (layer_idx, layer_type) → U [in_dim, k]
BASIS: Dict[Tuple[int, str], torch.Tensor] = {}

# ------------------------------------------------------------------
# Threshold hyperparameters
# ------------------------------------------------------------------
DEFAULT_FFN_ACTIVE_FRACTION = 0.1
DEFAULT_ATTN_ACTIVE_FRACTION = 0.1
FFN_ACTIVE_FRACTION = DEFAULT_FFN_ACTIVE_FRACTION
ATTN_ACTIVE_FRACTION = DEFAULT_ATTN_ACTIVE_FRACTION
MIN_NEURONS_FOR_QUANTILE = 10

DETAIL_LOG_PROMPT_LIMIT = 3
NEG_INF = -1e9


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def is_instruct_model(name: str) -> bool:
    name = name.lower()
    return ("instruct" in name) or ("chat" in name)


def should_log_detail(prompt_idx: int) -> bool:
    return prompt_idx < DETAIL_LOG_PROMPT_LIMIT


# ------------------------------------------------------------------
# Model initialisation
# ------------------------------------------------------------------
def initialize_model_and_tokenizer(selected_model_name: str) -> None:
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


# ------------------------------------------------------------------
# Basis loading
# ------------------------------------------------------------------
def load_basis(
    basis_dir: str,
    layer_types: List[str],
    num_layers: int,
    top_k: Optional[int],
    device: str = "cuda",
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Load WaRP SVD basis matrices from Phase 1 checkpoint directory.

    Each file ``basis/<layer_type>/layer_XX_svd.pt`` stores:
        'U'  : [in_dim, in_dim]  – principal-direction matrix (orthonormal columns,
                                   sorted by decreasing singular value)
        'S'  : [in_dim]          – singular values
        'UT' : [in_dim, in_dim]  – transpose of U (same as U for symmetric Gram)

    We take ``U[:, :top_k]`` as the working basis (all columns if top_k is None).

    Activations are projected as ``z = x @ U_k  ∈  R^{B×T×k}``,
    so ``z[:,:,j] = x @ u_j`` where u_j is the j-th principal direction.

    Returns
    -------
    dict : (layer_idx, layer_type) → U_k  [in_dim, k]  on *device*, dtype float32
    """
    basis: Dict[Tuple[int, str], torch.Tensor] = {}
    for layer_type in layer_types:
        layer_type_dir = os.path.join(basis_dir, layer_type)
        if not os.path.exists(layer_type_dir):
            logger.warning(f"Basis dir not found for {layer_type}: {layer_type_dir}")
            continue

        svd_files = sorted(
            f for f in os.listdir(layer_type_dir) if f.endswith("_svd.pt")
        )
        loaded = 0
        last_shape = None
        for fname in svd_files:
            parts = fname.split("_")  # layer_XX_svd.pt
            if len(parts) < 2:
                continue
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue

            fpath = os.path.join(layer_type_dir, fname)
            data = torch.load(fpath, map_location="cpu")
            U = data["U"]  # [in_dim, in_dim]

            # Columns are principal directions (sorted by descending singular value)
            if top_k is not None and top_k < U.shape[1]:
                U = U[:, :top_k]

            U = U.to(device=device, dtype=torch.float32)
            basis[(layer_idx, layer_type)] = U
            loaded += 1
            last_shape = tuple(U.shape)

        if loaded > 0:
            logger.info(
                f"Loaded basis [{layer_type}]: {loaded} layers, "
                f"U shape per layer = {last_shape}"
            )
        else:
            logger.warning(f"No basis files loaded for {layer_type}")

    return basis


# ------------------------------------------------------------------
# Rotated importance functions
# ------------------------------------------------------------------

def compute_ffn_down_rotated_importance(
    hffn: torch.Tensor,              # [B, T, I]  intermediate activations
    down_proj_weight: torch.Tensor,  # [H, I]
    U_basis: torch.Tensor,           # [I, k]  basis for down_proj input space
) -> torch.Tensor:
    """
    Importance of each WaRP basis direction in the ffn_down input space.

    The intermediate activation hffn = SiLU(gate) * up is the input to down_proj.
    A "neuron" in the rotated space = how much the j-th principal direction of
    hffn activates on safety prompts, weighted by how strongly it projects
    through W_down to the output.

    importance(j) = Σ_{b,t} z_j(b,t)²  ·  ‖W_down @ u_j‖²
    where z_j(b,t) = hffn[b,t] · u_j  (scalar)
    """
    # z : [B, T, k]
    z = hffn.float() @ U_basis          # [B, T, k]
    z_sq = z.pow(2).sum(dim=(0, 1))     # [k]

    # W_down_rot = W_down @ U  → [H, k]
    W_down_rot = down_proj_weight.float() @ U_basis   # [H, k]
    W_down_rot_sq = W_down_rot.pow(2).sum(dim=0)      # [k]

    return (z_sq * W_down_rot_sq).detach()


def compute_ffn_up_rotated_importance(
    mlp_x: torch.Tensor,             # [B, T, H_in]  MLP hidden input
    up_proj_weight: torch.Tensor,    # [I, H_in]
    gate_proj_weight: torch.Tensor,  # [I, H_in]
    down_proj_weight: torch.Tensor,  # [H_out, I]
    U_basis: torch.Tensor,           # [H_in, k]  basis for ffn_up / gate input space
) -> torch.Tensor:
    """
    Importance of each WaRP basis direction in the ffn_up (MLP hidden) input space.

    Gate and up projections share the same input (mlp_x), so both are accounted for.

    importance(j) = Σ_{b,t} z_j(b,t)²
                    · Σ_k [ (W_up_rot[k,j]² + W_gate_rot[k,j]²) · ‖W_down[:,k]‖² ]
    """
    z = mlp_x.float() @ U_basis         # [B, T, k]
    z_sq = z.pow(2).sum(dim=(0, 1))     # [k]

    W_up_rot  = up_proj_weight.float()  @ U_basis  # [I, k]
    W_gate_rot = gate_proj_weight.float() @ U_basis # [I, k]

    # Per-intermediate-neuron weight by down_proj column norms
    w_down_sq = down_proj_weight.float().pow(2).sum(dim=0)  # [I]

    # Combined gate + up contribution weighted by W_down norms → [k]
    W_combined_sq = ((W_up_rot.pow(2) + W_gate_rot.pow(2)) * w_down_sq.unsqueeze(-1)).sum(dim=0)

    return (z_sq * W_combined_sq).detach()


def compute_attn_proj_rotated_importance(
    attn_x: torch.Tensor,       # [B, T, H_in]
    proj_weight: torch.Tensor,  # [out_dim, H_in]
    U_basis: torch.Tensor,      # [H_in, k]
    scale_sq: Optional[torch.Tensor] = None,  # [out_dim]  per-output scaling
) -> torch.Tensor:
    """
    Importance of each WaRP basis direction for an attention projection (Q, K, or V).

    importance(j) = Σ_{b,t} z_j(b,t)²  ·  Σ_d (W_rot[d,j]²  · scale_sq[d])

    For Q and K:  scale_sq = None  (uniform unit scale)
    For V:        scale_sq = per-kv-head-aggregated o_proj column norms
                             to weight the V contribution by its output influence.
    """
    z = attn_x.float() @ U_basis        # [B, T, k]
    z_sq = z.pow(2).sum(dim=(0, 1))     # [k]

    W_rot = proj_weight.float() @ U_basis  # [out_dim, k]

    if scale_sq is not None:
        W_rot_sq = (W_rot.pow(2) * scale_sq.unsqueeze(-1)).sum(dim=0)  # [k]
    else:
        W_rot_sq = W_rot.pow(2).sum(dim=0)  # [k]

    return (z_sq * W_rot_sq).detach()


# ------------------------------------------------------------------
# Selection helpers (identical logic to safety_neuron_detection_v2.py)
# ------------------------------------------------------------------

def select_global_by_threshold(
    layer_importance: Dict[int, torch.Tensor],
    active_fraction: float,
    module_name: str,
) -> Dict[int, Set[int]]:
    """
    Select active basis directions using a single global threshold across all layers.

    Mirrors ``select_global_by_threshold`` from safety_neuron_detection_v2.py.
    """
    result: Dict[int, Set[int]] = {li: set() for li in range(NUM_LAYERS)}

    non_empty = {
        li: imp
        for li, imp in layer_importance.items()
        if imp is not None and imp.numel() > 0
    }
    if not non_empty:
        logger.info(f"select_global_by_threshold[{module_name}]: no importance values")
        return result

    all_imp = torch.cat([imp.view(-1) for imp in non_empty.values()], dim=0)
    if all_imp.numel() < MIN_NEURONS_FOR_QUANTILE:
        logger.info(
            f"select_global_by_threshold[{module_name}]: "
            f"too few basis dirs ({all_imp.numel()})"
        )
        return result

    q = max(0.0, min(1.0, 1.0 - active_fraction))
    epsilon = torch.quantile(all_imp, q)

    selected_total = 0
    for li, imp in non_empty.items():
        active_mask = imp >= epsilon
        result[li] = set(torch.nonzero(active_mask, as_tuple=False).view(-1).tolist())
        selected_total += len(result[li])

    logger.debug(
        f"select_global_by_threshold[{module_name}]: "
        f"total_dirs={all_imp.numel()}, selected={selected_total}, "
        f"active_fraction={active_fraction}, epsilon={epsilon.item():.6f}"
    )
    return result


# ------------------------------------------------------------------
# Core detection
# ------------------------------------------------------------------

def detect_safety_neurons_rotated(
    prompt: str,
    prompt_idx: int = 0,
) -> Optional[
    Tuple[
        Dict[int, Set[int]],  # ffn_up  basis directions
        Dict[int, Set[int]],  # ffn_down basis directions
        Dict[int, Set[int]],  # attn_q  basis directions
        Dict[int, Set[int]],  # attn_k  basis directions
        Dict[int, Set[int]],  # attn_v  basis directions
    ]
]:
    """
    Per-prompt safety neuron detection in the WaRP-rotated basis space.

    Algorithm
    ---------
    1. Tokenise prompt (same as v2).
    2. Register pre-hooks to capture hidden-state inputs of self_attn and mlp.
    3. Single forward pass (no gradient).
    4. For each layer: project captured activations onto the WaRP basis U and
       compute the rotated importance scores.
       - If no basis is loaded for a (layer, type) pair it is silently skipped.
    5. Apply global threshold per module to select safety basis directions.

    Returns
    -------
    5-tuple of dicts mapping layer_idx → set of important basis-direction indices,
    or None on error.
    """
    def _empty_dict() -> Dict[int, Set[int]]:
        return {li: set() for li in range(NUM_LAYERS)}

    ffn_up_dict   = _empty_dict()
    ffn_down_dict = _empty_dict()
    q_dict        = _empty_dict()
    k_dict        = _empty_dict()
    v_dict        = _empty_dict()

    try:
        # ----------------------------------------------------------------
        # 1) Tokenise
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # 2) Register hooks (capture hidden-state inputs)
        # ----------------------------------------------------------------
        captured: Dict[str, torch.Tensor] = {}

        def _attn_pre_hook(name: str):
            def hook(module, args, kwargs):
                x = None
                if kwargs is not None and "hidden_states" in kwargs \
                        and kwargs["hidden_states"] is not None:
                    x = kwargs["hidden_states"]
                elif args is not None and len(args) > 0 and args[0] is not None:
                    x = args[0]
                if x is not None:
                    captured[name] = x.detach()
            return hook

        def _mlp_pre_hook(name: str):
            def hook(module, module_inputs):
                if module_inputs:
                    captured[name] = module_inputs[0].detach()
            return hook

        hooks = []
        for li in range(NUM_LAYERS):
            layer = model.model.layers[li]
            hooks.append(
                layer.self_attn.register_forward_pre_hook(
                    _attn_pre_hook(f"layer_{li}_attn_in"), with_kwargs=True
                )
            )
            hooks.append(
                layer.mlp.register_forward_pre_hook(
                    _mlp_pre_hook(f"layer_{li}_mlp_in")
                )
            )

        try:
            # ----------------------------------------------------------------
            # 3) Single forward pass
            # ----------------------------------------------------------------
            with torch.no_grad():
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=False,
                    return_dict=True,
                )

            # ----------------------------------------------------------------
            # 4) Layer-wise rotated importance
            # ----------------------------------------------------------------
            ffn_up_imp:   Dict[int, torch.Tensor] = {}
            ffn_down_imp: Dict[int, torch.Tensor] = {}
            q_imp:        Dict[int, torch.Tensor] = {}
            k_imp:        Dict[int, torch.Tensor] = {}
            v_imp:        Dict[int, torch.Tensor] = {}

            for li in range(NUM_LAYERS):
                layer = model.model.layers[li]
                attn_key = f"layer_{li}_attn_in"
                mlp_key  = f"layer_{li}_mlp_in"

                if attn_key not in captured or mlp_key not in captured:
                    raise RuntimeError(f"Missing captured input at layer {li}")

                try:
                    attn_in = captured.pop(attn_key)
                    mlp_in  = captured.pop(mlp_key)

                    layer_dev   = layer.self_attn.q_proj.weight.device
                    attn_dtype  = layer.self_attn.q_proj.weight.dtype
                    mlp_dtype   = layer.mlp.up_proj.weight.dtype

                    attn_x = attn_in.to(device=layer_dev, dtype=attn_dtype)
                    mlp_x  = mlp_in.to(device=layer_dev, dtype=mlp_dtype)

                    # --------------------------------------------------------
                    # FFN – compute intermediate activations needed for ffn_down
                    # --------------------------------------------------------
                    gate_out = layer.mlp.gate_proj(mlp_x)                  # [B,T,I]
                    up_out   = layer.mlp.up_proj(mlp_x)                    # [B,T,I]
                    hffn     = F.silu(gate_out.float()) * up_out.float()   # [B,T,I]

                    # ffn_down rotated importance
                    key_down = (li, "ffn_down")
                    if key_down in BASIS:
                        U_down = BASIS[key_down].to(device=layer_dev)
                        ffn_down_imp[li] = compute_ffn_down_rotated_importance(
                            hffn, layer.mlp.down_proj.weight, U_down
                        )

                    # ffn_up rotated importance (MLP hidden input space)
                    key_up = (li, "ffn_up")
                    if key_up in BASIS:
                        U_up = BASIS[key_up].to(device=layer_dev)
                        ffn_up_imp[li] = compute_ffn_up_rotated_importance(
                            mlp_x,
                            layer.mlp.up_proj.weight,
                            layer.mlp.gate_proj.weight,
                            layer.mlp.down_proj.weight,
                            U_up,
                        )

                    del gate_out, up_out, hffn, mlp_x, mlp_in

                    # --------------------------------------------------------
                    # Attention – o_proj column norms for V scaling
                    # --------------------------------------------------------
                    o_col_sq = layer.self_attn.o_proj.weight.float().pow(2).sum(dim=0)  # [Hq*D]

                    # Q
                    key_q = (li, "attn_q")
                    if key_q in BASIS:
                        U_q = BASIS[key_q].to(device=layer_dev)
                        q_imp[li] = compute_attn_proj_rotated_importance(
                            attn_x, layer.self_attn.q_proj.weight, U_q
                        )

                    # K
                    key_k = (li, "attn_k")
                    if key_k in BASIS:
                        U_k = BASIS[key_k].to(device=layer_dev)
                        k_imp[li] = compute_attn_proj_rotated_importance(
                            attn_x, layer.self_attn.k_proj.weight, U_k
                        )

                    # V – weighted by aggregated o_proj column norms
                    key_v = (li, "attn_v")
                    if key_v in BASIS:
                        U_v = BASIS[key_v].to(device=layer_dev)

                        # Aggregate o_proj norms per KV head (GQA-aware)
                        num_heads    = model.config.num_attention_heads
                        head_dim     = layer.self_attn.q_proj.weight.shape[0] // num_heads
                        num_kv_heads = layer.self_attn.k_proj.weight.shape[0] // head_dim
                        num_kv_groups = num_heads // num_kv_heads

                        # o_col_sq: [Hq*D] → [Hq, D] → [Hkv, g, D] → mean over g → [Hkv*D]
                        o_col_per_kv = (
                            o_col_sq
                            .view(num_kv_heads, num_kv_groups, head_dim)
                            .mean(dim=1)           # [Hkv, D]
                            .reshape(-1)           # [Hkv*D]
                        )

                        v_imp[li] = compute_attn_proj_rotated_importance(
                            attn_x,
                            layer.self_attn.v_proj.weight,
                            U_v,
                            scale_sq=o_col_per_kv,
                        )

                    del attn_x, attn_in, o_col_sq

                    if should_log_detail(prompt_idx):
                        logger.debug(
                            f"[Prompt {prompt_idx}][Layer {li}] rotated importance done"
                        )

                except Exception as layer_exc:
                    logger.exception(
                        f"[Prompt {prompt_idx}][Layer {li}] layer importance failed: {layer_exc}"
                    )
                    raise

            # ----------------------------------------------------------------
            # 5) Global threshold selection per module
            # ----------------------------------------------------------------
            ffn_up_dict   = select_global_by_threshold(ffn_up_imp,   FFN_ACTIVE_FRACTION,  "ffn_up_rot")
            ffn_down_dict = select_global_by_threshold(ffn_down_imp, FFN_ACTIVE_FRACTION,  "ffn_down_rot")
            q_dict        = select_global_by_threshold(q_imp,        ATTN_ACTIVE_FRACTION, "q_rot")
            k_dict        = select_global_by_threshold(k_imp,        ATTN_ACTIVE_FRACTION, "k_rot")
            v_dict        = select_global_by_threshold(v_imp,        ATTN_ACTIVE_FRACTION, "v_rot")

        finally:
            for h in hooks:
                h.remove()
            captured.clear()

    except Exception as exc:
        logger.exception(f"Error in rotated detection (Prompt {prompt_idx}): {exc}")
        return None

    return ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict


# ------------------------------------------------------------------
# Intersection (identical logic to v2)
# ------------------------------------------------------------------

def compute_intersection(
    neuron_sets_list: List[Dict[int, Set[int]]],
    module_name: str = "module",
) -> Dict[int, Set[int]]:
    """
    Exact intersection across all prompts:  N_safe = ⋂_{x ∈ X} N_x  (Eq. 3).
    A basis direction must appear in *every* prompt-specific set to be retained.
    """
    if not neuron_sets_list:
        return {li: set() for li in range(NUM_LAYERS)}

    intersection: Dict[int, Set[int]] = {}
    before_total = 0
    after_total  = 0

    for li in range(NUM_LAYERS):
        layer_sets = [d.get(li, set()) for d in neuron_sets_list]
        union      = set().union(*layer_sets) if layer_sets else set()
        common     = set(layer_sets[0])
        for s in layer_sets[1:]:
            common &= s
        before_total += len(union)
        after_total  += len(common)
        intersection[li] = common

    logger.info(
        f"[compute_intersection][{module_name}] "
        f"prompts={len(neuron_sets_list)}, "
        f"union={before_total}, intersection={after_total}, "
        f"reduced={before_total - after_total}"
    )
    return intersection


# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safety neuron detection in WaRP basis-rotated space"
    )
    parser.add_argument(
        "num_prompts",
        type=int,
        help="Number of prompts from circuit_breakers_train.json",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--basis_dir",
        type=str,
        required=True,
        help="Path to Phase 1 basis directory (contains attn_q/, ffn_up/, etc.)",
    )
    parser.add_argument(
        "--layer_types",
        type=str,
        default="attn_q,attn_k,attn_v,ffn_down,ffn_up",
        help="Comma-separated layer types that have a basis to load",
    )
    parser.add_argument(
        "--top_k_basis",
        type=int,
        default=None,
        help=(
            "Use only the top-k principal directions from each basis "
            "(None = use full basis; reducing this greatly speeds up detection)"
        ),
    )
    parser.add_argument(
        "--ffn_active_fraction",
        type=float,
        default=DEFAULT_FFN_ACTIVE_FRACTION,
        help="Global top fraction for FFN basis directions to keep (0, 1]",
    )
    parser.add_argument(
        "--attn_active_fraction",
        "--attn_activ_fraction",
        dest="attn_active_fraction",
        type=float,
        default=DEFAULT_ATTN_ACTIVE_FRACTION,
        help="Global top fraction for attention basis directions to keep (0, 1]",
    )

    args = parser.parse_args(argv)

    if not (0.0 < args.ffn_active_fraction <= 1.0):
        parser.error("--ffn_active_fraction must be in (0, 1].")
    if not (0.0 < args.attn_active_fraction <= 1.0):
        parser.error("--attn_active_fraction must be in (0, 1].")

    return args


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv: List[str]) -> None:
    global FFN_ACTIVE_FRACTION, ATTN_ACTIVE_FRACTION, BASIS

    args = parse_args(argv)
    FFN_ACTIVE_FRACTION  = args.ffn_active_fraction
    ATTN_ACTIVE_FRACTION = args.attn_active_fraction

    initialize_model_and_tokenizer(args.model_name)

    # ----------------------------------------------------------------
    # Logging setup
    # ----------------------------------------------------------------
    log_dir = os.path.join(SCRIPT_DIR, "logs", "safety_neuron_detection_rotation")
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sn_rotation_{log_timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(f"Model          : {model_name}")
    logger.info(f"Basis dir      : {args.basis_dir}")
    logger.info(f"Layer types    : {args.layer_types}")
    logger.info(f"Top-k basis    : {args.top_k_basis or 'full'}")
    logger.info(f"FFN fraction   : {FFN_ACTIVE_FRACTION}")
    logger.info(f"Attn fraction  : {ATTN_ACTIVE_FRACTION}")

    # ----------------------------------------------------------------
    # Load WaRP basis
    # ----------------------------------------------------------------
    layer_types = [lt.strip() for lt in args.layer_types.split(",")]
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    BASIS = load_basis(
        basis_dir=args.basis_dir,
        layer_types=layer_types,
        num_layers=NUM_LAYERS,
        top_k=args.top_k_basis,
        device=compute_device,
    )
    logger.info(f"Total basis matrices loaded: {len(BASIS)}")

    if not BASIS:
        logger.error("No basis matrices loaded – check --basis_dir and --layer_types.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Load safety prompts
    # ----------------------------------------------------------------
    corpus_path = os.path.join(SCRIPT_DIR, "corpus_all", "circuit_breakers_train.json")
    if not os.path.exists(corpus_path):
        logger.error(f"Dataset not found: {corpus_path}")
        sys.exit(1)

    with open(corpus_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    if len(records) > args.num_prompts:
        records = records[: args.num_prompts]

    prompts = [item.get("prompt", "") for item in records]
    logger.info(f"Processing {len(prompts)} prompts")

    # ----------------------------------------------------------------
    # Per-prompt detection
    # ----------------------------------------------------------------
    ffn_up_sets:   List[Dict[int, Set[int]]] = []
    ffn_down_sets: List[Dict[int, Set[int]]] = []
    q_sets:        List[Dict[int, Set[int]]] = []
    k_sets:        List[Dict[int, Set[int]]] = []
    v_sets:        List[Dict[int, Set[int]]] = []

    failed_count     = 0
    successful_count = 0

    for idx, prompt in enumerate(tqdm(prompts, desc="Detecting rotated safety neurons")):
        result = detect_safety_neurons_rotated(prompt, prompt_idx=idx)
        if result is None:
            failed_count += 1
            logger.warning(f"Failed prompt idx={idx}")
            continue

        fu, fd, q, k, v = result
        ffn_up_sets.append(fu)
        ffn_down_sets.append(fd)
        q_sets.append(q)
        k_sets.append(k)
        v_sets.append(v)
        successful_count += 1

    logger.info(f"Detection complete: success={successful_count}, failed={failed_count}")

    # ----------------------------------------------------------------
    # Intersection across all prompts  (Eq. 3: N_safe = ⋂_x N_x)
    # ----------------------------------------------------------------
    ffn_up_common   = compute_intersection(ffn_up_sets,   "ffn_up_rot")
    ffn_down_common = compute_intersection(ffn_down_sets, "ffn_down_rot")
    q_common        = compute_intersection(q_sets,        "q_rot")
    k_common        = compute_intersection(k_sets,        "k_rot")
    v_common        = compute_intersection(v_sets,        "v_rot")

    # ----------------------------------------------------------------
    # Save output  (same format as safety_neuron_detection_v2.py)
    # ----------------------------------------------------------------
    output_dir = os.path.join(SCRIPT_DIR, "output_neurons")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"safety_neuron_rotation_{log_timestamp}.txt")

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({str(ki): list(vi) for ki, vi in ffn_up_common.items()})   + "\n")
        fh.write(json.dumps({str(ki): list(vi) for ki, vi in ffn_down_common.items()}) + "\n")
        fh.write(json.dumps({str(ki): list(vi) for ki, vi in q_common.items()})        + "\n")
        fh.write(json.dumps({str(ki): list(vi) for ki, vi in k_common.items()})        + "\n")
        fh.write(json.dumps({str(ki): list(vi) for ki, vi in v_common.items()})        + "\n")

    # ----------------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------------
    total_basis_dirs = sum(
        len(ffn_up_common.get(li, set()))
        + len(ffn_down_common.get(li, set()))
        + len(q_common.get(li, set()))
        + len(k_common.get(li, set()))
        + len(v_common.get(li, set()))
        for li in range(NUM_LAYERS)
    )

    k_dim = args.top_k_basis or "full"
    logger.info(f"\n{'='*70}")
    logger.info("Safety Neuron Detection Results (WaRP Rotated Basis Space)")
    logger.info(f"{'='*70}")
    logger.info(f"Model                : {model_name}")
    logger.info(f"Basis dir            : {args.basis_dir}")
    logger.info(f"Basis dimensionality : {k_dim} principal directions per layer")
    logger.info(f"Total safety basis directions found: {total_basis_dirs:,}")
    logger.info(f"Output file          : {output_file}")
    logger.info(f"Log file             : {log_file}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
