"""
MBPP Fine-tuning with Safety Neuron Freezing.

Safety neuron은 freeze하고 나머지 파라미터만 학습하여 safety 성능 유지.
(finetune_mbpp_full_params.py의 freeze_sn 버전)

Example Usage:
python finetune_mbpp_freeze_sn.py \
    --model_path kmseong/llama2_7b_chat_only_sn_tuned_lr5e-5_revised \
    --safety_neurons_file ./output_neurons/safety_neuron_accelerated_20260502_013602.txt \
    --output_dir ./mbpp_eval/llama2_7b_chat_mbpp_freeze_sn_lr5e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/llama2_7b-chat-sn-tuned-mbpp-freeze_sn-lr5e-5

python finetune_mbpp_freeze_sn.py \
    --model_path kmseong/llama2_7b_chat_only_rsn_tuned_lr5e-5_revised \
    --safety_neurons_file ./output_neurons/critical_safety_neuron_20260502_022558.txt \
    --output_dir ./mbpp_eval/llama2_7b_chat_mbpp_freeze_sn_lr5e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/llama2_7b-chat-rsn-tuned-mbpp-freeze_sn-lr5e-5





"""

import argparse
import ast
import gc
import json
import logging
import os
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


MBPP_INSTRUCTION = (
    "Write a Python function to solve the following programming problem. "
    "Provide only the complete, runnable Python code without any explanation.\n"
)


# =====================================================================
# Argument Parsing
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="MBPP Fine-tuning with Safety Neuron Freezing"
    )

    # Model
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace model ID or local path (SN-Tuned model)")

    # Safety neurons
    p.add_argument("--safety_neurons_file", type=str, required=True,
                   help="Path to safety neurons txt file (5 lines: ffn_up, ffn_down, q, k, v)")

    # MBPP dataset
    p.add_argument("--mbpp_dataset_name", type=str, default="google-research-datasets/mbpp",
                   help="HuggingFace dataset name for MBPP")
    p.add_argument("--mbpp_subset", type=str, default="full",
                   help="Dataset subset/config (full | sanitized)")
    p.add_argument("--mbpp_train_split", type=str, default="train",
                   help="Split name used for training")
    p.add_argument("--mbpp_eval_split", type=str, default="validation",
                   help="Split name used for evaluation (set to '' to skip)")

    p.add_argument("--num_train_samples", type=int, default=0,
                   help="학습 샘플 수 (0=전체)")
    p.add_argument("--num_eval_samples", type=int, default=0,
                   help="평가 샘플 수 (0=전체, eval 비활성 시 무시)")
    p.add_argument("--seed", type=int, default=42)

    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", type=str, default="adamw_torch")

    # Sequence
    p.add_argument("--max_length", type=int, default=1024)

    # Memory/speed
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # Logging/saving
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default="./cache")

    # HuggingFace upload
    p.add_argument("--upload_name", type=str, default=None,
                   help="HF repo id (예: kmseong/model-name). 설정 시 학습 후 자동 업로드")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HuggingFace API 토큰 (없으면 HF_TOKEN 환경변수 사용)")

    # Safety mixing
    p.add_argument("--safety_data_path", type=str,
                   default="/home/yonsei_jong/Safety-WaRP-LLM/data/circuit_breakers_train.json",
                   help="Safety dataset JSON 경로 (circuit_breakers_train.json)")
    p.add_argument("--safety_mix_ratio", type=float, default=0.0,
                   help="학습 데이터 수 대비 safety 데이터 비율 (예: 0.1=10%, 0=비활성)")

    return p.parse_args()


# =====================================================================
# Logging Setup
# =====================================================================
def setup_logging():
    log_dir = "./logs/mbpp_freeze_sn"
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_mbpp_freeze_sn_{log_timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


# =====================================================================
# Helpers
# =====================================================================
def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower() or "chat" in str(model_ref).lower()


def _as_text(value) -> str:
    return "" if value is None else str(value).strip()


def _select_random_n(ds, n: int, seed: int):
    if n is None or n <= 0 or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _keep_answer_budget(
    prompt_ids: List[int],
    answer_ids: List[int],
    max_length: int,
) -> Tuple[List[int], List[int]]:
    if len(prompt_ids) + len(answer_ids) <= max_length:
        return prompt_ids, answer_ids
    if max_length <= 1:
        raise ValueError(f"max_length must be > 1, got {max_length}")
    # 코드 정답은 길 수 있으므로 최소 128 토큰 보장
    answer_floor = max(128, max_length // 4)
    answer_budget = min(len(answer_ids), answer_floor)
    prompt_budget = max_length - answer_budget
    answer_ids = answer_ids[:answer_budget]
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    return prompt_ids, answer_ids


def render_chat_fallback(prompt: str, response: str, model_ref: str) -> Tuple[str, str]:
    prompt = prompt.strip()
    if "llama-2" in str(model_ref).lower():
        prompt_text = f"<s>[INST] {prompt} [/INST]"
        return prompt_text, f"{prompt_text} {response.strip()} </s>"
    prompt_text = f"User:\n{prompt}\n\nAssistant:\n"
    return prompt_text, f"{prompt_text}{response.strip()}"


# =====================================================================
# MBPP row → (prompt, response)
# =====================================================================
def mbpp_prompt_response(row: Dict, prefer_chat: bool = False) -> Tuple[str, str]:
    """
    MBPP 데이터셋 포맷 (google-research-datasets/mbpp):
      row["text"]      = 문제 설명 (자연어)
      row["code"]      = 정답 파이썬 코드
      row["test_list"] = 테스트 케이스 목록 (list of str)
    """
    problem  = _as_text(row.get("text", ""))
    solution = _as_text(row.get("code", ""))

    test_list = row.get("test_list", []) or []
    if test_list:
        tests_str = "\n".join(f"  {t}" for t in test_list[:3])
        problem_with_tests = f"{problem}\n\nYour code should pass the following tests:\n{tests_str}"
    else:
        problem_with_tests = problem

    if prefer_chat:
        user_content = f"{MBPP_INSTRUCTION}\n{problem_with_tests}"
        return user_content, solution

    # base 모델: alpaca-style
    prompt = (
        f"### Instruction:\n{MBPP_INSTRUCTION}\n\n"
        f"### Input:\n{problem_with_tests}\n\n"
        f"### Response:\n"
    )
    return prompt, solution


# =====================================================================
# Tokenization
# =====================================================================
def tokenize_prompt_response(
    prompt: str,
    response: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    prompt = _as_text(prompt)
    response = _as_text(response)
    if not prompt or not response:
        raise ValueError("prompt and response must be non-empty")

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids):]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
        except Exception:
            prompt_text, full_text = render_chat_fallback(prompt, response, model_ref)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids):]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
    else:
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)

    prompt_ids, answer_ids = _keep_answer_budget(prompt_ids, answer_ids, max_length)

    input_ids = prompt_ids + answer_ids
    labels    = [-100] * len(prompt_ids) + answer_ids
    if not any(label != -100 for label in labels):
        raise ValueError("tokenization produced no supervised response tokens")

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
    }


# =====================================================================
# Data Collator
# =====================================================================
@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id  = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"]      + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0]      * pad_len)
            labels.append(f["labels"]             + [-100]  * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


# =====================================================================
# Dataset loading & tokenization
# =====================================================================
def load_mbpp_rows(args, logger, eval_dataset: bool = False):
    split = args.mbpp_eval_split if eval_dataset else args.mbpp_train_split
    label = "eval" if eval_dataset else "train"

    if eval_dataset and not split:
        logger.info("Eval split not specified, skipping eval dataset")
        return None

    logger.info(f"Loading MBPP {label} split='{split}' from {args.mbpp_dataset_name} ...")
    try:
        ds = load_dataset(
            args.mbpp_dataset_name,
            args.mbpp_subset,
            split=split,
            cache_dir=args.cache_dir,
            trust_remote_code=False,
        )
    except Exception as exc:
        logger.error(f"Failed to load MBPP dataset: {exc}")
        raise

    n = args.num_eval_samples if eval_dataset else args.num_train_samples
    ds = _select_random_n(ds, n, args.seed + (1 if eval_dataset else 0))
    logger.info(f"Loaded MBPP {label} rows: {len(ds):,}")
    return ds


def tokenize_dataset_rows(ds, args, tokenizer, model_path: str, logger, desc: str = "Tokenizing"):
    prefer_chat = is_instruct_model(model_path)

    def preprocess(ex):
        prompt, response = mbpp_prompt_response(dict(ex), prefer_chat=prefer_chat)
        return tokenize_prompt_response(prompt, response, tokenizer, args.max_length, model_path)

    return ds.map(
        preprocess,
        remove_columns=ds.column_names,
        num_proc=max(1, args.num_workers),
        desc=desc,
    )


# =====================================================================
# Safety data mixing
# =====================================================================
def maybe_mix_safety(train_tok, args, tokenizer, model_path: str, logger, num_mbpp: int = 0):
    if args.safety_mix_ratio <= 0:
        return train_tok

    if not os.path.exists(args.safety_data_path):
        raise FileNotFoundError(f"Safety dataset not found: {args.safety_data_path}")

    with open(args.safety_data_path, "r", encoding="utf-8") as f:
        safety_raw = json.load(f)

    num_safety = int(len(train_tok) * args.safety_mix_ratio)
    rng = random.Random(args.seed)
    sampled = rng.sample(safety_raw, min(num_safety, len(safety_raw)))

    def preprocess_safety(ex):
        return tokenize_prompt_response(
            ex["prompt"],
            ex["llama3_output"],
            tokenizer,
            args.max_length,
            model_path,
        )

    safety_hf  = HFDataset.from_list(sampled)
    safety_tok = safety_hf.map(
        preprocess_safety,
        remove_columns=safety_hf.column_names,
        desc="Tokenizing safety data",
    )

    mixed = concatenate_datasets([train_tok, safety_tok]).shuffle(seed=args.seed)
    logger.info(f"Safety data mixed: {len(safety_tok)} samples (ratio={args.safety_mix_ratio})")
    logger.info(
        f"Total training samples: {len(mixed)} "
        f"(MBPP {num_mbpp} + Safety {len(safety_tok)})"
    )
    return mixed


# =====================================================================
# HuggingFace upload
# =====================================================================
def upload_to_hf(output_dir: str, upload_name: str, hf_token: Optional[str], logger):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub 설치 필요: pip install huggingface_hub")
        return

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        logger.error("HF 토큰이 없습니다. --hf_token 또는 HF_TOKEN 환경변수를 설정하세요.")
        return

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=upload_name, repo_type="model", exist_ok=True, private=False)
        logger.info(f"Repo created/found: {upload_name}")
    except Exception as exc:
        logger.warning(f"Repo creation warning: {exc}")

    ignore_patterns = [".wandb/*", "wandb/*", "*.log", "__pycache__/*", "cache/*"]
    logger.info(f"Uploading {output_dir} → {upload_name} ...")
    try:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=upload_name,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            commit_message="MBPP fine-tuned model (freeze_sn)",
        )
        logger.info(f"Upload completed: https://huggingface.co/{upload_name}")
    except Exception as exc:
        logger.error(f"Upload failed: {exc}")


# =====================================================================
# Load Safety Neurons
# =====================================================================
def load_safety_neurons(output_file: str, logger) -> Dict:
    """
    Load safety neurons from detection output file.

    Format (5 lines):
        Line 0: ffn_up   (JSON dict)
        Line 1: ffn_down (JSON dict)
        Line 2: q        (JSON dict)
        Line 3: k        (JSON dict)
        Line 4: v        (JSON dict)
    """
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    try:
        ffn_up_raw   = ast.literal_eval(lines[0].strip())
        ffn_down_raw = ast.literal_eval(lines[1].strip())
        q_raw        = ast.literal_eval(lines[2].strip())
        k_raw        = ast.literal_eval(lines[3].strip())
        v_raw        = ast.literal_eval(lines[4].strip())

        safety_neurons = {
            "ffn_up":   {int(k): v for k, v in ffn_up_raw.items()},
            "ffn_down": {int(k): v for k, v in ffn_down_raw.items()},
            "q":        {int(k): v for k, v in q_raw.items()},
            "k":        {int(k): v for k, v in k_raw.items()},
            "v":        {int(k): v for k, v in v_raw.items()},
        }
    except Exception as e:
        logger.error(f"Error parsing safety neurons file: {e}")
        raise

    logger.info(f"Loaded safety neurons from {output_file}")
    logger.info(f"\n{'='*70}")
    logger.info(f"Safety Neurons Loaded - Detailed Breakdown")
    logger.info(f"{'='*70}")

    total_neurons = 0
    for module_type in ["ffn_up", "ffn_down", "q", "k", "v"]:
        module_total = sum(len(n) for n in safety_neurons[module_type].values())
        logger.info(f"  {module_type:12} : {module_total:4} neurons")
        total_neurons += module_total
        layers_with_neurons = [
            l for l in safety_neurons[module_type] if safety_neurons[module_type][l]
        ]
        if layers_with_neurons:
            logger.info(
                f"    └─ Layers with neurons: {layers_with_neurons[:5]}"
                f"{'...' if len(layers_with_neurons) > 5 else ''}"
            )

    logger.info(f"\nTotal safety neurons: {total_neurons}")
    logger.info(f"{'='*70}\n")
    return safety_neurons


# =====================================================================
# Safety Neuron Freezing (train all except safety neurons)
# =====================================================================
def _sanitize_indices(indices, axis_size, module_type, layer_idx, param_name, logger):
    """Keep only valid integer indices within [0, axis_size)."""
    valid, dropped, seen = [], 0, set()
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            dropped += 1
            continue
        if 0 <= i < axis_size and i not in seen:
            valid.append(i)
            seen.add(i)
        else:
            dropped += 1
    if dropped > 0:
        logger.warning(
            f"[IndexSanitize] {param_name} layer={layer_idx} module={module_type}: "
            f"dropped {dropped} out-of-range indices (axis_size={axis_size})"
        )
    return valid


def _remap_kv_indices(indices, layer_idx, module_type, axis_size, model, logger):
    """
    Remap k/v neuron indices from expanded attention space
    (num_heads * head_dim) to KV space (num_kv_heads * head_dim)
    when the model uses GQA (grouped-query attention).
    """
    if module_type not in ("k", "v") or not indices:
        return [int(i) for i in indices]

    converted = []
    for idx in indices:
        try:
            converted.append(int(idx))
        except Exception:
            continue

    if not converted or max(converted) < axis_size:
        return converted

    attn = model.model.layers[layer_idx].self_attn
    num_heads    = getattr(attn, "num_heads", None) or getattr(model.config, "num_attention_heads", None)
    num_kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(model.config, "num_key_value_heads", None)
    head_dim     = getattr(attn, "head_dim", None)
    if head_dim is None and isinstance(num_heads, int) and num_heads > 0:
        hidden = getattr(model.config, "hidden_size", None)
        if isinstance(hidden, int):
            head_dim = hidden // num_heads

    if not all(isinstance(x, int) and x > 0 for x in (num_heads, num_kv_heads, head_dim)):
        logger.warning(
            f"[IndexRemap] {module_type} layer={layer_idx}: cannot resolve attention geometry; keeping raw indices"
        )
        return converted
    if num_heads % num_kv_heads != 0:
        return converted

    n_rep = num_heads // num_kv_heads
    remapped, remap_count = [], 0
    for i in converted:
        if i < 0:
            continue
        if i < axis_size:
            remapped.append(i)
        elif i < num_heads * head_dim:
            head_idx = i // head_dim
            dim_idx  = i % head_dim
            new_i    = (head_idx // n_rep) * head_dim + dim_idx
            remapped.append(new_i)
            remap_count += 1
    if remap_count > 0:
        logger.info(
            f"[IndexRemap] {module_type} layer={layer_idx}: remapped {remap_count} indices "
            f"from expanded attention space to KV space"
        )
    return remapped


def setup_safety_neuron_freezing(model, safety_neurons: Dict, logger) -> List:
    """
    Freeze only safety neurons; train everything else.

    Reverse of sn_tune.py:
      sn_tune.py    → freeze all, train only safety neurons
      This function → train all, freeze only safety neurons (via gradient hook)

    Returns frozen_param_specs: list of (param, indices, axis) used by
    SafetyNeuronRestoreCallback to undo weight-decay updates on safety neurons.
    """
    total_params = 0
    frozen_neuron_params = 0
    frozen_modules = {"ffn_up": 0, "ffn_down": 0, "q": 0, "k": 0, "v": 0}
    frozen_param_specs = []  # (param, indices, axis) — for weight-decay bypass correction

    for param in model.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        total_params += param.numel()

        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            continue
        try:
            layer_idx = int(parts[2])
        except ValueError:
            continue

        def _make_zero_hook_rows(indices):
            """Zero out gradient rows (used for up_proj, q/k/v_proj)."""
            def hook(grad):
                grad = grad.clone()
                grad[indices, :] = 0.0
                return grad
            return hook

        def _make_zero_hook_cols(indices):
            """Zero out gradient columns (used for down_proj)."""
            def hook(grad):
                grad = grad.clone()
                grad[:, indices] = 0.0
                return grad
            return hook

        if "mlp.up_proj.weight" in name:
            indices = safety_neurons["ffn_up"].get(layer_idx, [])
            indices = _sanitize_indices(indices, param.shape[0], "ffn_up", layer_idx, name, logger)
            if indices:
                frozen_neuron_params += len(indices) * param.shape[1]
                frozen_modules["ffn_up"] += 1
                param.register_hook(_make_zero_hook_rows(indices))
                frozen_param_specs.append((param, list(indices), "rows"))

        elif "mlp.down_proj.weight" in name:
            indices = safety_neurons["ffn_down"].get(layer_idx, [])
            indices = _sanitize_indices(indices, param.shape[1], "ffn_down", layer_idx, name, logger)
            if indices:
                frozen_neuron_params += len(indices) * param.shape[0]
                frozen_modules["ffn_down"] += 1
                param.register_hook(_make_zero_hook_cols(indices))
                frozen_param_specs.append((param, list(indices), "cols"))

        elif "self_attn.q_proj.weight" in name:
            indices = safety_neurons["q"].get(layer_idx, [])
            indices = _sanitize_indices(indices, param.shape[0], "q", layer_idx, name, logger)
            if indices:
                frozen_neuron_params += len(indices) * param.shape[1]
                frozen_modules["q"] += 1
                param.register_hook(_make_zero_hook_rows(indices))
                frozen_param_specs.append((param, list(indices), "rows"))

        elif "self_attn.k_proj.weight" in name:
            indices = safety_neurons["k"].get(layer_idx, [])
            indices = _remap_kv_indices(indices, layer_idx, "k", param.shape[0], model, logger)
            indices = _sanitize_indices(indices, param.shape[0], "k", layer_idx, name, logger)
            if indices:
                frozen_neuron_params += len(indices) * param.shape[1]
                frozen_modules["k"] += 1
                param.register_hook(_make_zero_hook_rows(indices))
                frozen_param_specs.append((param, list(indices), "rows"))

        elif "self_attn.v_proj.weight" in name:
            indices = safety_neurons["v"].get(layer_idx, [])
            indices = _remap_kv_indices(indices, layer_idx, "v", param.shape[0], model, logger)
            indices = _sanitize_indices(indices, param.shape[0], "v", layer_idx, name, logger)
            if indices:
                frozen_neuron_params += len(indices) * param.shape[1]
                frozen_modules["v"] += 1
                param.register_hook(_make_zero_hook_rows(indices))
                frozen_param_specs.append((param, list(indices), "rows"))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n{'='*70}")
    logger.info("Safety Neuron Freezing Setup Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total parameters:                        {total_params:,}")
    logger.info(f"Frozen safety neuron params (effective): {frozen_neuron_params:,}")
    logger.info(f"Trainable parameters:                    {trainable_params:,}")
    logger.info(f"Trainable ratio:            {trainable_params / total_params * 100:.4f}%")
    logger.info(f"Frozen safety neuron ratio: {frozen_neuron_params / total_params * 100:.4f}%")
    logger.info("\nLayers with frozen safety neurons:")
    for module_type, count in frozen_modules.items():
        if count > 0:
            logger.info(f"  {module_type:12} : {count} layers")
    logger.info(f"{'='*70}\n")
    return frozen_param_specs


# =====================================================================
# Safety Neuron Restore Callback
# =====================================================================
class SafetyNeuronRestoreCallback(TrainerCallback):
    """
    Restores safety neuron weights after every optimizer step.

    AdamW의 weight-decay term (λθ)은 gradient hook과 독립적으로 적용되므로
    safety neuron 가중치가 0으로 drift할 수 있다.
    이 콜백은 초기 frozen 값을 저장하고 매 optimizer step 후 복원하여
    진정한 파라미터 freezing을 보장한다.
    """

    def __init__(self, frozen_param_specs: List):
        # frozen_param_specs: list of (param, indices, axis)
        #   axis = "rows"  →  param[indices, :]  (up/q/k/v_proj)
        #   axis = "cols"  →  param[:, indices]  (down_proj)
        self._specs = frozen_param_specs
        self._frozen_vals = []
        for param, indices, axis in frozen_param_specs:
            with torch.no_grad():
                if axis == "rows":
                    self._frozen_vals.append(param.data[indices, :].clone())
                else:
                    self._frozen_vals.append(param.data[:, indices].clone())

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each optimizer step — restore frozen weights."""
        for (param, indices, axis), frozen_val in zip(self._specs, self._frozen_vals):
            with torch.no_grad():
                if axis == "rows":
                    param.data[indices, :] = frozen_val
                else:
                    param.data[:, indices] = frozen_val
        return control


# =====================================================================
# Main
# =====================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    raw_path   = args.model_path
    is_local   = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger, log_file = setup_logging()

    logger.info(f"\n{'='*70}")
    logger.info("  MBPP Fine-tuning with Safety Neuron Freezing")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")
    logger.info("Configuration:")
    logger.info(f"   ├─ Model:               {model_path}")
    logger.info(f"   ├─ Safety neurons file: {args.safety_neurons_file}")
    logger.info(
        f"   ├─ Input formatting:    "
        f"{'chat template' if is_instruct_model(model_path) else 'base plain prompt'}"
    )
    logger.info(f"   ├─ Dataset:             {args.mbpp_dataset_name} / {args.mbpp_subset}")
    logger.info(f"   ├─ Train split:         {args.mbpp_train_split}")
    logger.info(f"   ├─ Safety mix ratio:    {args.safety_mix_ratio}")
    logger.info(f"   ├─ Batch size:          {args.batch_size}")
    logger.info(f"   ├─ Grad accum:          {args.grad_accum}")
    logger.info(f"   ├─ Epochs:              {args.epochs}")
    logger.info(f"   ├─ LR:                  {args.learning_rate}")
    logger.info(f"   ├─ Strategy:            Freeze safety neurons, train others")
    logger.info(f"   └─ Output dir:          {args.output_dir}")

    if not os.path.exists(args.safety_neurons_file):
        logger.error(f"Safety neurons file not found: {args.safety_neurons_file}")
        raise FileNotFoundError(f"Safety neurons file not found: {args.safety_neurons_file}")

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  [1/5] Loading Tokenizer")
    logger.info(f"{'='*70}\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        logger.info("Tokenizer loaded from local files")
    except Exception as e:
        logger.warning(f"local_files_only failed ({e}); trying HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        logger.info("Tokenizer loaded from HuggingFace Hub")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  [2/5] Loading Model (bf16)")
    logger.info(f"{'='*70}\n")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False,
        )
        logger.info("Model loaded from local files")
    except Exception as e:
        logger.warning(f"local_files_only failed ({e}); trying HuggingFace Hub...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
        )
        logger.info("Model loaded from HuggingFace Hub")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params / 1e9:.2f}B parameters, dtype={model.dtype}")

    # ------------------------------------------------------------------
    # 3. Safety Neuron Freezing
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  [3/5] Loading Safety Neurons and Setting up Freezing")
    logger.info(f"{'='*70}\n")

    safety_neurons = load_safety_neurons(args.safety_neurons_file, logger)
    frozen_param_specs = setup_safety_neuron_freezing(model, safety_neurons, logger)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Freezing complete — trainable: {trainable_params / 1e9:.2f}B "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ------------------------------------------------------------------
    # 4. Dataset
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  [4/5] Loading MBPP Dataset")
    logger.info(f"{'='*70}\n")

    train_rows = load_mbpp_rows(args, logger, eval_dataset=False)
    eval_rows  = load_mbpp_rows(args, logger, eval_dataset=True)

    train_tok = tokenize_dataset_rows(train_rows, args, tokenizer, model_path, logger, desc="Tokenizing train")
    eval_tok  = None
    if eval_rows is not None and args.num_eval_samples > 0:
        eval_tok = tokenize_dataset_rows(eval_rows, args, tokenizer, model_path, logger, desc="Tokenizing eval")

    num_mbpp  = len(train_tok)
    train_tok = maybe_mix_safety(train_tok, args, tokenizer, model_path, logger, num_mbpp=num_mbpp)

    logger.info(f"Train samples: {len(train_tok):,}")

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  [5/5] Training")
    logger.info(f"{'='*70}\n")

    do_eval = eval_tok is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy=("steps" if do_eval else "no"),
        eval_steps=(args.eval_steps if do_eval else None),
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim=args.optim,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
        callbacks=[SafetyNeuronRestoreCallback(frozen_param_specs)],
    )

    logger.info("Starting training...")
    trainer.train()

    # ------------------------------------------------------------------
    # Save Model
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  Saving Fine-tuned Model")
    logger.info(f"{'='*70}\n")

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = f"{args.output_dir}_{timestamp}"

        logger.info("Step 1: Preparing model for saving...")
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Step 2: Moving model to CPU for safe serialization...")
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Step 3: Saving model weights...")
        model.save_pretrained(
            final_output_dir,
            safe_serialization=True,
            max_shard_size="4GB",
            push_to_hub=False,
        )
        logger.info("   Model weights saved (safetensors)")

        logger.info("Step 4: Saving tokenizer...")
        tokenizer.save_pretrained(final_output_dir, safe_serialization=True)

        logger.info("Step 5: Saving model config...")
        model.config.save_pretrained(final_output_dir)
        if hasattr(model, "generation_config"):
            model.generation_config.save_pretrained(final_output_dir)

        logger.info("Step 6: Verifying saved model integrity...")
        required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        missing_files = [
            fname for fname in required_files
            if not os.path.exists(os.path.join(final_output_dir, fname))
        ]
        if missing_files:
            raise FileNotFoundError(f"Missing/corrupted files: {missing_files}")

        model_files = [f for f in os.listdir(final_output_dir) if f.endswith(".safetensors")]
        if not model_files:
            raise FileNotFoundError("No safetensors files found after save!")
        logger.info(f"   Found {len(model_files)} model shard file(s)")

        total_size = sum(
            os.path.getsize(os.path.join(final_output_dir, f))
            for f in os.listdir(final_output_dir)
            if os.path.isfile(os.path.join(final_output_dir, f))
        )
        logger.info(f"   Total size: {total_size/1e9:.2f} GB")

        logger.info("Step 7: Final verification - attempting to load saved model...")
        test_tokenizer = AutoTokenizer.from_pretrained(final_output_dir)
        test_model = AutoModelForCausalLM.from_pretrained(
            final_output_dir,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
        )
        del test_tokenizer, test_model
        gc.collect()
        logger.info("   Model verified successfully")

        logger.info(f"\nFine-tuned model saved and verified: {os.path.abspath(final_output_dir)}")

    except Exception as e:
        logger.error(f"CRITICAL ERROR during model saving: {e}")
        logger.error(traceback.format_exc())
        raise

    # Save training config
    config = {
        "base_model":              args.model_path,
        "fine_tuning_type":        "MBPP Fine-tuning with Safety Neuron Freezing",
        "safety_neurons_file":     args.safety_neurons_file,
        "dataset":                 "mbpp",
        "mbpp_dataset_name":       args.mbpp_dataset_name,
        "mbpp_subset":             args.mbpp_subset,
        "mbpp_train_split":        args.mbpp_train_split,
        "mbpp_eval_split":         args.mbpp_eval_split,
        "num_train_samples":       len(train_tok),
        "num_eval_samples":        len(eval_tok) if eval_tok is not None else 0,
        "batch_size":              args.batch_size,
        "grad_accum":              args.grad_accum,
        "learning_rate":           args.learning_rate,
        "weight_decay":            args.weight_decay,
        "warmup_ratio":            args.warmup_ratio,
        "epochs":                  args.epochs,
        "max_length":              args.max_length,
        "max_grad_norm":           args.max_grad_norm,
        "lr_scheduler_type":       args.lr_scheduler_type,
        "optimizer":               args.optim,
        "gradient_checkpointing":  args.gradient_checkpointing,
        "dtype":                   "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
        "safety_mix_ratio":        args.safety_mix_ratio,
        "safety_data_path":        args.safety_data_path if args.safety_mix_ratio > 0 else None,
        "strategy":                "Freeze safety neurons, train others",
    }
    config_path = os.path.join(final_output_dir, "finetune_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Config saved to {config_path}")

    if args.upload_name:
        logger.info(f"Uploading to HuggingFace: {args.upload_name}")
        upload_to_hf(final_output_dir, args.upload_name, args.hf_token, logger)

    logger.info(f"\n{'='*70}")
    logger.info("  Fine-tuning Complete!")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
