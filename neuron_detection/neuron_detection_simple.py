"""
Safety Neuron Detection based on Paper Method (Threshold Version)

- For each harmful query x:
    * Run model, record activations for FFN(up/down) and Attention(Q/K/V)
    * Compute a scalar importance per neuron (per dimension)
    * Select neurons whose importance >= epsilon (layer-wise threshold)
      -> This corresponds to Nx in Eq. (2)

- Across a corpus X of harmful queries:
    * Intersect Nx over all x in X
      -> This corresponds to N_safe in Eq. (3)

Threshold choice (epsilon):
    - For each layer/module, we use a quantile-based threshold:
        epsilon = quantile(importance, 1 - ACTIVE_FRACTION)
    - Here ACTIVE_FRACTION = 0.005 (top 0.5% per module per query)
"""

import os
from typing import Dict, Set, List, Tuple
import sys
from tqdm import tqdm
import logging
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(112)
torch.manual_seed(112)

# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
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
FFN_ACTIVE_FRACTION = 0.005
ATTN_ACTIVE_FRACTION = 0.005

# quantile 연산 시 최소 샘플 수가 너무 적을 때를 대비한 safeguard
MIN_NEURONS_FOR_QUANTILE = 10


def select_by_threshold(importance: torch.Tensor,
                        active_fraction: float) -> Set[str]:
    """
    Given a 1D importance vector [D], select indices whose importance >= epsilon,
    where epsilon is chosen as the (1 - active_fraction) quantile.

    importance: torch.Tensor, shape [D]
    active_fraction: e.g., 0.005 (top 0.5%)

    Returns:
        Set of neuron indices (as "neuron_j" string) above threshold.
    """
    if importance.numel() == 0:
        return set()

    # If too few neurons, fall back to "no thresholding"
    if importance.numel() < MIN_NEURONS_FOR_QUANTILE:
        # Degenerate case: keep nothing
        return set()

    # Compute epsilon = quantile(importance, 1 - active_fraction)
    q = max(0.0, min(1.0, 1.0 - active_fraction))
    epsilon = torch.quantile(importance, q)

    # Nx = { N_i^(l) | Imp(N_i^(l)|x) >= epsilon }
    active_mask = importance >= epsilon
    indices = torch.nonzero(active_mask, as_tuple=False).view(-1)

    return {f"neuron_{idx.item()}" for idx in indices}


def detect_safety_neurons_threshold(
    prompt: str,
) -> Tuple[Dict[int, Set[str]],
           Dict[int, Set[str]],
           Dict[int, Set[str]],
           Dict[int, Set[str]],
           Dict[int, Set[str]]]:
    """
    Foundational safety neuron detection (threshold-based, activation approximation).

    논문의 Eq. (1)–(3)와의 대응:
      - hi(x): layer/module의 activation
      - Imp(N_i^(l)|x): |activation|의 평균값을 importance로 사용 (approx.)
      - epsilon: layer/module별 activation 중요도의 (1 - fraction) quantile
      - Nx: epsilon 이상인 뉴런 집합
    """

    ffn_up_dict: Dict[int, Set[str]] = {}
    ffn_down_dict: Dict[int, Set[str]] = {}
    q_dict: Dict[int, Set[str]] = {}
    k_dict: Dict[int, Set[str]] = {}
    v_dict: Dict[int, Set[str]] = {}

    try:
        # 1) Tokenize input harmful query x
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
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
                activations_dict[name] = act.detach().cpu()
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

            # 4) For each layer, compute importance & thresholded Nx
            for layer_idx in range(NUM_LAYERS):
                # ---------- FFN up_proj ----------
                ffn_up_key = f"layer_{layer_idx}_ffn_up"
                if ffn_up_key in activations_dict:
                    act = activations_dict[ffn_up_key].float()  # [B, T, D_ffn]
                    # Importance ≈ mean(|activation|) over batch & seq
                    importance = torch.abs(act).mean(dim=(0, 1))  # [D_ffn]
                    indices_set = select_by_threshold(
                        importance, FFN_ACTIVE_FRACTION
                    )
                    ffn_up_dict[layer_idx] = indices_set
                else:
                    ffn_up_dict[layer_idx] = set()

                # ---------- FFN down_proj ----------
                ffn_down_key = f"layer_{layer_idx}_ffn_down"
                if ffn_down_key in activations_dict:
                    act = activations_dict[ffn_down_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))
                    indices_set = select_by_threshold(
                        importance, FFN_ACTIVE_FRACTION
                    )
                    ffn_down_dict[layer_idx] = indices_set
                else:
                    ffn_down_dict[layer_idx] = set()

                # ---------- Attention Q ----------
                q_key = f"layer_{layer_idx}_attn_q"
                if q_key in activations_dict:
                    act = activations_dict[q_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))  # [D_q]
                    indices_set = select_by_threshold(
                        importance, ATTN_ACTIVE_FRACTION
                    )
                    q_dict[layer_idx] = indices_set
                else:
                    q_dict[layer_idx] = set()

                # ---------- Attention K ----------
                k_key = f"layer_{layer_idx}_attn_k"
                if k_key in activations_dict:
                    act = activations_dict[k_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))  # [D_k]
                    indices_set = select_by_threshold(
                        importance, ATTN_ACTIVE_FRACTION
                    )
                    k_dict[layer_idx] = indices_set
                else:
                    k_dict[layer_idx] = set()

                # ---------- Attention V ----------
                v_key = f"layer_{layer_idx}_attn_v"
                if v_key in activations_dict:
                    act = activations_dict[v_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))  # [D_v]
                    indices_set = select_by_threshold(
                        importance, ATTN_ACTIVE_FRACTION
                    )
                    v_dict[layer_idx] = indices_set
                else:
                    v_dict[layer_idx] = set()

        finally:
            # 5) Remove hooks
            for h in hooks:
                h.remove()

    except Exception as e:
        logger.error(f"Error in neuron detection: {e}")
        # Fallback: empty sets
        for layer_idx in range(NUM_LAYERS):
            ffn_up_dict[layer_idx] = set()
            ffn_down_dict[layer_idx] = set()
            q_dict[layer_idx] = set()
            k_dict[layer_idx] = set()
            v_dict[layer_idx] = set()

    return ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict


def compute_intersection(neuron_sets_list: List[Dict[int, Set[str]]]) -> Dict[int, Set[str]]:
    """
    Compute intersection of neuron sets across all prompts (Eq. 3).

    Input:
      neuron_sets_list: 각 prompt x에 대해 얻은 Nx (layer -> set of neuron names) 리스트

    Output:
      intersection_dict: layer -> set of neurons that appear in EVERY prompt
                         => N_safe for that layer
    """
    if not neuron_sets_list:
        return {}

    intersection_dict: Dict[int, Set[str]] = {}
    all_layers = range(NUM_LAYERS)

    for layer_idx in all_layers:
        layer_sets = []
        for neuron_dict in neuron_sets_list:
            layer_sets.append(neuron_dict.get(layer_idx, set()))

        # 아무 것도 없으면 공집합
        if not layer_sets:
            intersection_dict[layer_idx] = set()
            continue

        # Nx들의 교집합
        common = set.intersection(*layer_sets) if layer_sets else set()
        intersection_dict[layer_idx] = common

    return intersection_dict


def main(argv):
    if len(argv) < 1:
        logger.error("Usage: python neuron_detection_paper_based_v2.py <dataset_name> [num_prompts]")
        logger.error("Example: python neuron_detection_paper_based_v2.py harmful_prompts 50")

        corpus_dir = "./corpus_all"
        if os.path.exists(corpus_dir):
            logger.error("\nAvailable datasets in corpus_all/:")
            datasets = [f[:-4] for f in os.listdir(corpus_dir) if f.endswith(".txt")]
            for ds in sorted(datasets):
                logger.error(f"  - {ds}")
        sys.exit(1)

    dataset_name = argv[0]
    num_prompts = int(argv[1]) if len(argv) > 1 else 50

    file_path = f"./corpus_all/{dataset_name}.txt"
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    lines = random.sample(lines, min(num_prompts, len(lines)))

    logger.info(f"Loaded {len(lines)} prompts from dataset: {dataset_name}")
    logger.info(f"Model: {model_name} ({NUM_LAYERS} layers, {HIDDEN_DIM} hidden_dim)")
    logger.info(f"FFN_ACTIVE_FRACTION = {FFN_ACTIVE_FRACTION}, "
                f"ATTN_ACTIVE_FRACTION = {ATTN_ACTIVE_FRACTION}")

    # 각 prompt x에 대해 Nx를 수집
    ffn_up_sets: List[Dict[int, Set[str]]] = []
    ffn_down_sets: List[Dict[int, Set[str]]] = []
    q_sets: List[Dict[int, Set[str]]] = []
    k_sets: List[Dict[int, Set[str]]] = []
    v_sets: List[Dict[int, Set[str]]] = []

    logger.info("\n=== Starting Threshold-based Safety Neuron Detection ===")
    for idx, prompt in enumerate(tqdm(lines, desc="Detecting neurons")):
        try:
            ffn_up, ffn_down, q, k, v = detect_safety_neurons_threshold(prompt)
            ffn_up_sets.append(ffn_up)
            ffn_down_sets.append(ffn_down)
            q_sets.append(q)
            k_sets.append(k)
            v_sets.append(v)
        except Exception as e:
            logger.warning(f"Failed prompt {idx}: {str(e)[:100]}")

    logger.info(f"Successfully processed {len(ffn_up_sets)}/{len(lines)} prompts")

    # Eq. (3): N_safe = ⋂_x N_x
    logger.info("\nComputing neuron intersections across all prompts...")
    ffn_up_common = compute_intersection(ffn_up_sets)
    ffn_down_common = compute_intersection(ffn_down_sets)
    q_common = compute_intersection(q_sets)
    k_common = compute_intersection(k_sets)
    v_common = compute_intersection(v_sets)

    # 결과 저장
    os.makedirs("./output_neurons", exist_ok=True)
    clean_model_name = model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        f"./output_neurons/"
        f"{clean_model_name}_{dataset_name}_threshold_neurons_{len(ffn_up_sets)}_{timestamp}.txt"
    )

    logger.info(f"\nSaving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(ffn_up_common) + "\n")
        f.write(str(ffn_down_common) + "\n")
        f.write(str(q_common) + "\n")
        f.write(str(k_common) + "\n")
        f.write(str(v_common) + "\n")

    # 통계 출력
    logger.info("\n" + "=" * 70)
    logger.info("Threshold-based Safety Neuron Detection Results")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Prompts processed: {len(ffn_up_sets)}/{len(lines)}")
    logger.info(f"Total neurons in model (FFN only, coarse): {TOTAL_NEURONS:,}")
    logger.info(f"FFN_ACTIVE_FRACTION: {FFN_ACTIVE_FRACTION}")
    logger.info(f"ATTN_ACTIVE_FRACTION: {ATTN_ACTIVE_FRACTION}\n")

    total_safety_neurons = 0
    total_ffn_neurons = 0
    total_attn_neurons = 0
    for layer_idx in range(NUM_LAYERS):
        ffn_up_count = len(ffn_up_common.get(layer_idx, set()))
        ffn_down_count = len(ffn_down_common.get(layer_idx, set()))
        q_count = len(q_common.get(layer_idx, set()))
        k_count = len(k_common.get(layer_idx, set()))
        v_count = len(v_common.get(layer_idx, set()))
        
        ffn_count = ffn_up_count + ffn_down_count
        attn_count = q_count + k_count + v_count
        layer_neurons = ffn_count + attn_count
        
        if layer_neurons > 0:
            logger.info(f"Layer {layer_idx}: {layer_neurons} safety neurons (FFN: {ffn_count}, Attention: {attn_count})")
            total_safety_neurons += layer_neurons
            total_ffn_neurons += ffn_count
            total_attn_neurons += attn_count

    actual_sparsity = total_safety_neurons / TOTAL_NEURONS if TOTAL_NEURONS > 0 else 0
    logger.info(f"\nTotal safety neurons detected: {total_safety_neurons} (FFN: {total_ffn_neurons}, Attention: {total_attn_neurons})")
    logger.info(f"Actual sparsity: {actual_sparsity*100:.4f}%")
    logger.info(f"Output saved to: {output_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1:])
