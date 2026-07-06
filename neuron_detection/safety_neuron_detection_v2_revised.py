'''
Usage (current implementation)
==============================

This script follows the original GitHub detection style:
- use scores computed/stored inside patched transformers `modeling_llama.py`
- select layer-wise fixed top-k indices
    - attention top-k: 200
    - FFN top-k: 1200
- take intersection across prompts to get final neurons

1) Safety neuron detection (from corpus_all/circuit_breakers_train.json)
python safety_neuron_detection_v2.py 4994 \
    --model_name kmseong/llama2_7b_chat-safety-rotation \
    --top_number_ffn 400 \
    --top_number_attn 80 \
    --safety_neuron

2) Utility neuron detection (from Wikipedia)
python safety_neuron_detection_v2.py 1000 \
    --model_name kmseong/llama2_7b_chat-safety-rotation \
    --top_number_ffn 300 \
    --top_number_attn 50 \
    --utility_neuron

Notes
-----
- `num_prompts` is the number of samples used for intersection.
- For instruct/chat models, input is built by `apply_chat_template(...)`.
- `--top_number_attn` and `--top_number_ffn` control per-layer top-k selection.
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
from datasets import load_dataset
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

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
# Accelerated detection hyperparameters
# ------------------------------------------------------------------
DETAIL_LOG_PROMPT_LIMIT = 3
TOP_NUMBER_ATTN = 2000
TOP_NUMBER_FFN = 12000


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
        attn_implementation="eager",
    )
    model.eval()

    NUM_LAYERS = model.config.num_hidden_layers


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Safety neuron detection with configurable model and per-layer top-k"
    )
    parser.add_argument(
        "num_prompts",
        type=int,
        help="Number of samples to process (prompts for --safety_neuron, documents for --utility_neuron)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--top_number_ffn",
        type=int,
        default=TOP_NUMBER_FFN,
        help="Per-layer top-k for FFN neuron selection",
    )
    parser.add_argument(
        "--top_number_attn",
        type=int,
        default=TOP_NUMBER_ATTN,
        help="Per-layer top-k for attention neuron selection",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--safety_neuron",
        action="store_true",
        help="Detect safety neurons from circuit_breakers_train.json",
    )
    mode_group.add_argument(
        "--utility_neuron",
        action="store_true",
        help="Detect utility neurons from Wikipedia dataset",
    )

    args = parser.parse_args(argv)

    if args.top_number_ffn <= 0:
        parser.error("--top_number_ffn must be a positive integer.")
    if args.top_number_attn <= 0:
        parser.error("--top_number_attn must be a positive integer.")

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


def select_topk_indices(score: Optional[torch.Tensor], top_k: int) -> Set[int]:
    """Select top-k indices from a 1D score vector."""
    if score is None:
        return set()

    if isinstance(score, torch.Tensor):
        arr = score.detach().float().view(-1).cpu().numpy()
    else:
        arr = np.asarray(score, dtype=np.float32).reshape(-1)

    if arr.size == 0:
        return set()

    k = min(top_k, arr.size)
    if k <= 0:
        return set()

    indices = np.argsort(arr)[-k:][::-1]
    return {int(idx) for idx in indices.tolist()}


def select_topk_from_ranked_indices(raw_indices, top_k: int) -> Set[int]:
    """Select top-k when input is already ranked neuron indices (highest importance first)."""
    if raw_indices is None:
        return set()
    arr = np.asarray(raw_indices).reshape(-1)
    if arr.size == 0:
        return set()
    k = min(top_k, arr.size)
    if k <= 0:
        return set()
    return {int(v) for v in arr[:k].tolist()}


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
    """Original GitHub style detection: use forward-stored scores + per-layer fixed top-k."""
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

        # --------------------------------------------------------
        # 2) Run forward once
        # --------------------------------------------------------
        with torch.no_grad():
            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=False,
                return_dict=True,
            )

        # --------------------------------------------------------
        # 3) Per-layer fixed top-k selection from forward-stored scores
        # --------------------------------------------------------
        for layer_idx in range(NUM_LAYERS):
            layer = model.model.layers[layer_idx]

            ffn_up_score = getattr(layer.mlp, "_last_ffn_up_score", None)
            ffn_down_score = getattr(layer.mlp, "_last_ffn_down_score", None)
            q_score = getattr(layer.self_attn, "_last_q_score", None)
            k_score = getattr(layer.self_attn, "_last_k_score", None)
            v_score = getattr(layer.self_attn, "_last_v_score", None)

            if any(score is None for score in [ffn_up_score, ffn_down_score, q_score, k_score, v_score]):
                # Compatibility path for patched modeling_llama variants that expose
                # per-layer ranked indices via early_exit_layers outputs.
                with torch.no_grad():
                    fallback_out = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        output_hidden_states=True,
                        return_dict=True,
                        early_exit_layers=list(range(NUM_LAYERS)),
                    )

                if not isinstance(fallback_out, tuple) or len(fallback_out) < 7:
                    raise RuntimeError(
                        f"Missing forward-captured score tensor(s) at layer {layer_idx} and fallback early_exit output "
                        f"is not compatible (type={type(fallback_out)}, len={len(fallback_out) if isinstance(fallback_out, tuple) else 'N/A'})."
                    )

                activate_keys_fwd_up = fallback_out[2]
                activate_keys_fwd_down = fallback_out[3]
                activate_keys_q = fallback_out[4]
                activate_keys_k = fallback_out[5]
                activate_keys_v = fallback_out[6]

                for li in range(NUM_LAYERS):
                    ffn_up_dict[li] = select_topk_from_ranked_indices(activate_keys_fwd_up.get(li, []), TOP_NUMBER_FFN)
                    ffn_down_dict[li] = select_topk_from_ranked_indices(activate_keys_fwd_down.get(li, []), TOP_NUMBER_FFN)
                    q_dict[li] = select_topk_from_ranked_indices(activate_keys_q.get(li, []), TOP_NUMBER_ATTN)
                    k_dict[li] = select_topk_from_ranked_indices(activate_keys_k.get(li, []), TOP_NUMBER_ATTN)
                    v_dict[li] = select_topk_from_ranked_indices(activate_keys_v.get(li, []), TOP_NUMBER_ATTN)

                break

            ffn_up_dict[layer_idx] = select_topk_indices(ffn_up_score, TOP_NUMBER_FFN)
            ffn_down_dict[layer_idx] = select_topk_indices(ffn_down_score, TOP_NUMBER_FFN)
            q_dict[layer_idx] = select_topk_indices(q_score, TOP_NUMBER_ATTN)
            k_dict[layer_idx] = select_topk_indices(k_score, TOP_NUMBER_ATTN)
            v_dict[layer_idx] = select_topk_indices(v_score, TOP_NUMBER_ATTN)

        if should_log_detail(prompt_idx):
            ffn_up_total = sum(len(v) for v in ffn_up_dict.values())
            ffn_down_total = sum(len(v) for v in ffn_down_dict.values())
            q_total = sum(len(v) for v in q_dict.values())
            k_total = sum(len(v) for v in k_dict.values())
            v_total = sum(len(v) for v in v_dict.values())

            logger.debug(
                f"[Prompt {prompt_idx}] selected neurons by top-k: "
                f"ffn_up={ffn_up_total}, ffn_down={ffn_down_total}, "
                f"q={q_total}, k={k_total}, v={v_total}"
            )

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


def load_wikipedia_data(num_samples: int = 1000) -> List[str]:
    """Load Wikipedia data from Hugging Face."""
    logger.info("Loading Wikipedia dataset (subset: 20231101.en)...")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=False,
            cache_dir=os.path.join(SCRIPT_DIR, "wikipedia_cache"),
        )

        total_size = len(dataset)
        random.seed(112)
        random_indices = random.sample(range(total_size), min(num_samples, total_size))

        texts = []
        for idx in tqdm(random_indices, desc="Loading Wikipedia docs"):
            try:
                item = dataset[idx]
                text = item.get("text", "").strip()
                if text:
                    texts.append(text[:2000])
            except Exception as e:
                logger.warning(f"Failed to load Wikipedia doc {idx}: {e}")

        logger.info(f"Successfully loaded {len(texts)} Wikipedia samples")
        return texts

    except Exception as e:
        logger.error(f"Error loading Wikipedia dataset: {e}")
        raise


def main(argv):
    global TOP_NUMBER_FFN, TOP_NUMBER_ATTN

    args = parse_args(argv)

    TOP_NUMBER_FFN = args.top_number_ffn
    TOP_NUMBER_ATTN = args.top_number_attn

    initialize_model_and_tokenizer(args.model_name)

    # =====================================================================
    # 로깅 설정: 파일 핸들러 추가
    # =====================================================================
    log_dir = os.path.join(SCRIPT_DIR, "logs", "neuron_detection")
    os.makedirs(log_dir, exist_ok=True)

    # 파일 이름: 현재 날짜 및 시간
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_prefix = "safety_neuron" if args.safety_neuron else "utility_neuron"
    log_file = os.path.join(log_dir, f"{log_prefix}_{log_timestamp}.log")
    
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
    logger.info(f"TOP_NUMBER_FFN: {TOP_NUMBER_FFN}, TOP_NUMBER_ATTN: {TOP_NUMBER_ATTN}")

    num_samples = args.num_prompts

    if args.safety_neuron:
        logger.info("[Mode] Safety Neuron Detection")
        logger.info(f"Number of prompts to process: {num_samples}")
        file_path = os.path.join(SCRIPT_DIR, "corpus_all", "circuit_breakers_train.json")
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            logger.error(f"No valid 'prompt' entries found in: {file_path}")
            sys.exit(1)

        if len(records) > num_samples:
            records = records[:num_samples]

        lines = [item.get("prompt", "") for item in records]
        logger.info(f"Processing {len(lines)} prompts from {file_path}")

    else:  # args.utility_neuron
        logger.info("[Mode] Utility Neuron Detection (Wikipedia)")
        logger.info(f"Number of Wikipedia documents to process: {num_samples}")
        lines = load_wikipedia_data(num_samples=num_samples)
        if not lines:
            logger.error("Failed to load Wikipedia data")
            sys.exit(1)
        logger.info(f"Processing {len(lines)} Wikipedia documents")

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
    if args.safety_neuron:
        output_file = os.path.join(output_dir, f"safety_neuron_accelerated_{log_timestamp}.txt")
    else:
        output_file = os.path.join(output_dir, f"utility_neurons_{len(ffn_up_sets)}_{log_timestamp}.txt")

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
    
    mode_label = "Safety" if args.safety_neuron else "Utility"
    logger.info(f"\n{'='*70}")
    logger.info(f"{mode_label} Neuron Detection Results")
    logger.info(f"{'='*70}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Total {mode_label.lower()} neurons: {total_safety_neurons:,}")
    logger.info(f"Total model neurons (q/k/v/o + gate/up/down): {total_model_neurons:,}")
    logger.info(f"Detected {mode_label.lower()} neuron percentage: {actual_sparsity*100:.4f}%")
    logger.info(f"Output: {output_file}")
    logger.info(f"Log: {log_file}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
