'''
python safety_neuron_detection.py 4994
'''

import os
from typing import Dict, Set, List, Tuple
import sys
import json
from tqdm import tqdm
import logging
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

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
    return "abcde" in name.lower()

model_name = "meta-llama/Llama-3.2-3B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"":  0},  # Force all layers to cuda:0 (single GPU)
    torch_dtype=torch.bfloat16,
)
model.eval()

# Llama-3.2-3B: 28 layers, 3072 model hidden size
NUM_LAYERS = 28
HIDDEN_DIM = 3072

# 전체 FFN 뉴런 수(논문에서 말하는 "<1%" sparsity 참조용)
TOTAL_NEURONS = NUM_LAYERS * HIDDEN_DIM

# ------------------------------------------------------------------
# Threshold hyperparameters (epsilon 역할)
# ------------------------------------------------------------------
# 각 layer/module에서 "최상위 몇 %의 뉴런을 활성 뉴런으로 볼 것인가?"
# 예: 0.005 -> 상위 0.5% (논문: safety neuron은 전체의 <1% 라는 관찰과 일치)
FFN_ACTIVE_FRACTION = 0.1
ATTN_ACTIVE_FRACTION = 0.1

# quantile 연산 시 최소 샘플 수가 너무 적을 때를 대비한 safeguard
MIN_NEURONS_FOR_QUANTILE = 10


def calculate_model_total_neurons() -> int:
    """
    Same denominator as calculate_safety_neuron_percentage.py:
    q/k/v/o + gate/up/down output channels across all layers.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    intermediate_size = cfg.intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    head_dim = hidden_size // num_heads
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
        logger.debug("select_by_threshold: Empty importance tensor")
        return set()

    # If too few neurons, fall back to empty set
    if importance.numel() < MIN_NEURONS_FOR_QUANTILE:
        logger.debug(f"select_by_threshold: Too few neurons ({importance.numel()} < {MIN_NEURONS_FOR_QUANTILE})")
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
) -> Tuple[Dict[int, Set[int]],
           Dict[int, Set[int]],
           Dict[int, Set[int]],
           Dict[int, Set[int]],
           Dict[int, Set[int]]]:

    # 로그 제거: 진행상황만 tqdm으로 표시

    ffn_up_dict: Dict[int, Set[int]] = {}
    ffn_down_dict: Dict[int, Set[int]] = {}
    q_dict: Dict[int, Set[int]] = {}
    k_dict: Dict[int, Set[int]] = {}
    v_dict: Dict[int, Set[int]] = {}

    try:
        # 1) Tokenize input harmful query x
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

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 2) Set up hooks to capture hi(x) at various submodules
        activations_dict = {}

        def get_activation_hook(name):
            def hook(module, input, output):
                # hi(x)를 output으로 간주 (FFN up_proj, down_proj, Q/K/V proj)
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                # Keep activations on GPU (model is on single cuda:0)
                activations_dict[name] = act.detach()
            return hook

        hooks = []
        for layer_idx in range(NUM_LAYERS):
            layer = model.model.layers[layer_idx]

            # FFN: up_proj, down_proj
            if hasattr(layer.mlp, "up_proj"):
                hooks.append(
                    layer.mlp.up_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_ffn_up")
                    )
                )
            if hasattr(layer.mlp, "down_proj"):
                hooks.append(
                    layer.mlp.down_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_ffn_down")
                    )
                )

            # Attention: q_proj, k_proj, v_proj
            if hasattr(layer.self_attn, "q_proj"):
                hooks.append(
                    layer.self_attn.q_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_attn_q")
                    )
                )
            if hasattr(layer.self_attn, "k_proj"):
                hooks.append(
                    layer.self_attn.k_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_attn_k")
                    )
                )
            if hasattr(layer.self_attn, "v_proj"):
                hooks.append(
                    layer.self_attn.v_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_attn_v")
                    )
                )

        try:
            # 3) Forward pass (no grad)
            with torch.no_grad():
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=False,
                    return_dict=True,
                )

            # 4) Compute per-layer importance first, then apply one global
            # threshold per module across all layers.
            ffn_up_importance: Dict[int, torch.Tensor] = {}
            ffn_down_importance: Dict[int, torch.Tensor] = {}
            q_importance: Dict[int, torch.Tensor] = {}
            k_importance: Dict[int, torch.Tensor] = {}
            v_importance: Dict[int, torch.Tensor] = {}

            for layer_idx in range(NUM_LAYERS):
                # ---------- FFN up_proj ----------
                ffn_up_key = f"layer_{layer_idx}_ffn_up"
                if ffn_up_key in activations_dict:
                    act = activations_dict[ffn_up_key].float()  # [B, T, D_ffn]
                    # Importance ≈ mean(|activation|) over batch & seq
                    ffn_up_importance[layer_idx] = torch.abs(act).mean(dim=(0, 1))  # [D_ffn]

                # ---------- FFN down_proj ----------
                ffn_down_key = f"layer_{layer_idx}_ffn_down"
                if ffn_down_key in activations_dict:
                    act = activations_dict[ffn_down_key].float()
                    ffn_down_importance[layer_idx] = torch.abs(act).mean(dim=(0, 1))

                # ---------- Attention Q ----------
                q_key = f"layer_{layer_idx}_attn_q"
                if q_key in activations_dict:
                    act = activations_dict[q_key].float()
                    q_importance[layer_idx] = torch.abs(act).mean(dim=(0, 1))  # [D_q]

                # ---------- Attention K ----------
                k_key = f"layer_{layer_idx}_attn_k"
                if k_key in activations_dict:
                    act = activations_dict[k_key].float()
                    k_importance[layer_idx] = torch.abs(act).mean(dim=(0, 1))  # [D_k]

                # ---------- Attention V ----------
                v_key = f"layer_{layer_idx}_attn_v"
                if v_key in activations_dict:
                    act = activations_dict[v_key].float()
                    v_importance[layer_idx] = torch.abs(act).mean(dim=(0, 1))  # [D_v]

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
            # 5) Remove hooks
            for h in hooks:
                h.remove()

    except Exception as e:
        logger.error(f"Error in neuron detection (Query {prompt_idx}): {e}")
        # Fallback: empty sets
        for layer_idx in range(NUM_LAYERS):
            ffn_up_dict[layer_idx] = set()
            ffn_down_dict[layer_idx] = set()
            q_dict[layer_idx] = set()
            k_dict[layer_idx] = set()
            v_dict[layer_idx] = set()

    return ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict


def compute_intersection(neuron_sets_list: List[Dict[int, Set[int]]], module_name: str = "module") -> Dict[int, Set[int]]:
    """
    Compute intersection of neuron sets across all prompts (Eq. 3).

    Eq. (3): N_safe = ⋂_{x in X} Nx
    - 모든 harmful query에서 공통으로 나타나는 뉴런만 선택

    Input:
      neuron_sets_list: 각 prompt x에 대해 얻은 Nx (layer -> set of neuron indices) 리스트

    Output:
      intersection_dict: layer -> set of neurons that appear in EVERY prompt
                         => 진짜 Safety Neuron (N_safe)
    """
    if not neuron_sets_list:
        logger.info(f"[compute_intersection][{module_name}] no neuron sets; reduced=0")
        return {layer_idx: set() for layer_idx in range(NUM_LAYERS)}

    intersection_dict: Dict[int, Set[int]] = {}
    before_union_total = 0
    after_intersection_total = 0

    for layer_idx in range(NUM_LAYERS):
        # 각 query에서 이 layer의 활성 뉴런들 수집
        layer_sets = [
            neuron_dict.get(layer_idx, set()) 
            for neuron_dict in neuron_sets_list
        ]

        # 모든 layer_sets가 비어있으면 이 layer는 공집합
        if not layer_sets or all(not s for s in layer_sets):
            intersection_dict[layer_idx] = set()
            continue

        # Eq. (3): 교집합 계산
        # (모든 query에서 나타나는 뉴런만 남김)
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


def main(argv):
    if len(argv) < 1:
        logger.error("Usage: python safety_neuron_detection.py <num_prompts>")
        logger.error("Example: python safety_neuron_detection.py 800")
        sys.exit(1)

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
    console_handler.setLevel(logging.INFO)
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

    num_prompts = int(argv[0])
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
    for idx, prompt in enumerate(tqdm(lines, desc="Detecting neurons")):
        try:
            ffn_up, ffn_down, q, k, v = detect_safety_neurons_threshold(prompt, prompt_idx=idx)
            ffn_up_sets.append(ffn_up)
            ffn_down_sets.append(ffn_down)
            q_sets.append(q)
            k_sets.append(k)
            v_sets.append(v)
        except Exception as e:
            failed_count += 1
            logger.warning(f"Failed prompt idx={idx}: {e}")

    successful_count = len(ffn_up_sets)
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
    output_file = os.path.join(output_dir, f"safety_neuron_threshold_{log_timestamp}.txt")

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
