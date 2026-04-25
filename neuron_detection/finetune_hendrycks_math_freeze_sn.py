"""
Hendrycks MATH 데이터셋을 사용하여 SN-Tuned 모델 파인튜닝 (Safety Neuron Freeze)

Safety neuron은 freeze하고 나머지 파라미터만 학습하여 safety 성능 유지
(finetune_gsm8k_freeze_sn.py의 MATH 버전)

Instruct 모델 기준:
- tokenizer.apply_chat_template 사용
- user prompt는 labels=-100으로 마스킹
- assistant response만 loss 계산

Example Usage:
python finetune_hendrycks_math_freeze_sn.py \
    --model_path kmseong/llama3.1_8b_instruct_only_rsn_tuned_lr3e-5 \
    --safety_neurons_file ./output_neurons/critical_safety_neuron_20260418_204749.txt \
    --output_dir ./math_ft_8b_instruct_freeze_rsn_lr1e-5 \
    --upload_name kmseong/math_ft_8b_instruct_freeze_rsn_lr1e-5

python finetune_hendrycks_math_freeze_sn.py \
    --model_path kmseong/llama2_7b_chat_only_rsn_tuned_lr3e-5 \
    --safety_neurons_file ./output_neurons/critical_safety_neuron_20260418_204636.txt \
    --output_dir ./math_ft_chat_after_rsn_tune_freeze_rsn_lr3e-5 \
    --upload_name kmseong/llama2_7b_chat_math_ft_freeze_rsn_lr3e-5
    
"""

import argparse
import ast
import gc
import json
import logging
import os
import random
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import torch
import wandb
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =====================================================================
# Argument Parsing
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Hendrycks MATH Fine-tuning with Safety Neuron Freezing"
    )

    # Model
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace model ID or local path (SN-Tuned model)")

    # Safety neurons
    p.add_argument("--safety_neurons_file", type=str, required=True,
                   help="Path to safety neurons txt file (5 JSON lines: ffn_up, ffn_down, q, k, v)")

    # MATH dataset
    p.add_argument("--math_dataset_source", type=str, default="official",
                   choices=["official", "flat_competition_math"])
    p.add_argument("--math_official_dataset_path", type=str, default="EleutherAI/hendrycks_math")
    p.add_argument("--math_flat_dataset_path", type=str, default="qwedsacf/competition_math")
    p.add_argument("--math_subjects", type=str, default="all")
    p.add_argument("--math_levels", type=str, default="all")
    p.add_argument("--num_train_samples", type=int, default=0)
    p.add_argument("--num_eval_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--math_train_on_mixed_formats", action="store_true", default=False)

    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5)
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
    p.add_argument("--output_dir", type=str, default="./math_freeze_sn_finetune")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--upload_name", type=str, default=None,
                   help="Optional Hugging Face repo id (e.g., username/model-name). If set, upload after training")
    p.add_argument("--hf_token", type=str, default=None,
                   help="Optional Hugging Face token for upload")

    return p.parse_args()


# =====================================================================
# Helpers
# =====================================================================
def is_instruct_model(model_ref: str) -> bool:
    model_ref = model_ref.lower()
    return any(tag in model_ref for tag in ('instruct', 'chat'))


def normalize_csv_arg(raw_value: str) -> str:
    value = str(raw_value).strip()
    if len(value) >= 2 and (
        (value[0] == '"' and value[-1] == '"')
        or (value[0] == "'" and value[-1] == "'")
    ):
        value = value[1:-1].strip()
    return value


# =====================================================================
# MATH Answer Extraction / Target Building
# =====================================================================
def last_boxed_only_string(text: str):
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if s is None:
        raise ValueError("remove_boxed received None")
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left) :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s


def extract_final_answer_from_solution(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError(
            f"Could not find final boxed answer in solution: {solution[:300]!r}"
        )
    return remove_boxed(boxed).strip()


def clean_solution_for_reasoning(solution: str, final_answer: str) -> str:
    multi_space_re = re.compile(r"\n{3,}")
    text = solution.strip()
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        text = text.replace(boxed, final_answer)
    text = text.replace("$", "")
    text = text.replace("\\[", "")
    text = text.replace("\\]", "")
    text = text.replace("\\(", "")
    text = text.replace("\\)", "")
    text = text.replace("\\boxed", "")
    text = text.replace("\\fbox", "")
    text = multi_space_re.sub("\n\n", text)
    return text.strip()


def build_target(solution: str, rng: random.Random, train_on_mixed_formats: bool) -> str:
    final_answer = extract_final_answer_from_solution(solution)
    rationale = clean_solution_for_reasoning(solution, final_answer)

    long_target = f"{rationale}\nFinal Answer: ${final_answer}$"
    short_target = f"Final Answer: ${final_answer}$"
    minimal_target = f"${final_answer}$"

    if not train_on_mixed_formats:
        return long_target

    draw = rng.random()
    if draw < 0.70:
        return long_target
    if draw < 0.90:
        return short_target
    return minimal_target


# =====================================================================
# Tokenization
# =====================================================================
def tokenize_math_sft_example(
    problem: str,
    target_text: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    problem = str(problem).strip()
    target_text = str(target_text).strip()

    if is_instruct_model(model_ref):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": target_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    # Base model: plain Question/Answer format
    prompt_text = f"Question: {problem}\nAnswer:"
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remain,
    )["input_ids"]

    if tokenizer.eos_token_id is not None and (
        len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id
    ):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


# =====================================================================
# Data Collator
# =====================================================================
@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# =====================================================================
# Load Safety Neurons
# =====================================================================
def load_safety_neurons(output_file: str, logger) -> Dict:
    """
    Load safety neurons from detection output file.

    Format (5 lines):
        Line 0: ffn_up  (JSON dict)
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
def setup_safety_neuron_freezing(model, safety_neurons: Dict, logger) -> List:
    """
    Freeze only safety neurons; train everything else.

    Reverse of sn_tune.py:
      sn_tune.py         → freeze all, train only safety neurons
      This function      → train all, freeze only safety neurons (via gradient hook)

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
            if indices:
                frozen_neuron_params += len(indices) * param.shape[1]
                frozen_modules["ffn_up"] += 1
                param.register_hook(_make_zero_hook_rows(indices))
                frozen_param_specs.append((param, list(indices), "rows"))

        elif "mlp.down_proj.weight" in name:
            indices = safety_neurons["ffn_down"].get(layer_idx, [])
            if indices:
                # down_proj: [hidden_dim, intermediate_dim] — neurons are columns
                frozen_neuron_params += len(indices) * param.shape[0]
                frozen_modules["ffn_down"] += 1
                param.register_hook(_make_zero_hook_cols(indices))
                frozen_param_specs.append((param, list(indices), "cols"))

        elif "self_attn.q_proj.weight" in name:
            indices = safety_neurons["q"].get(layer_idx, [])
            if indices:
                frozen_neuron_params += len(indices) * param.shape[1]
                frozen_modules["q"] += 1
                param.register_hook(_make_zero_hook_rows(indices))
                frozen_param_specs.append((param, list(indices), "rows"))

        elif "self_attn.k_proj.weight" in name:
            indices = safety_neurons["k"].get(layer_idx, [])
            if indices:
                frozen_neuron_params += len(indices) * param.shape[1]
                frozen_modules["k"] += 1
                param.register_hook(_make_zero_hook_rows(indices))
                frozen_param_specs.append((param, list(indices), "rows"))

        elif "self_attn.v_proj.weight" in name:
            indices = safety_neurons["v"].get(layer_idx, [])
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

    AdamW's weight-decay term (λθ) is applied independently of gradient hooks,
    so safety neuron weights would otherwise drift toward 0 even when the
    gradient hook zeros out the gradient signal.  This callback saves the
    initial (frozen) values at construction time and writes them back after
    every optimizer step, guaranteeing true parameter freezing.
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
# Logging Setup
# =====================================================================
def setup_logging():
    log_dir = "./logs/safety_neuron_math_freeze"
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_math_freeze_sn_{log_timestamp}.log")

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
# Main
# =====================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path

    logger, log_file = setup_logging()

    logger.info(f"\n{'='*70}")
    logger.info("  🚀 Hendrycks MATH Fine-tuning with Safety Neuron Freezing")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")
    logger.info("⚙️  Configuration:")
    logger.info(f"   ├─ Model:               {model_path}")
    logger.info(f"   ├─ Safety neurons file: {args.safety_neurons_file}")
    logger.info(
        f"   ├─ Input formatting:    "
        f"{'chat template' if is_instruct_model(model_path) else 'Question/Answer plain text'}"
    )
    logger.info(f"   ├─ Subjects:            {args.math_subjects}")
    logger.info(f"   ├─ Levels:              {args.math_levels}")
    logger.info(f"   ├─ Train samples:       {args.num_train_samples}")
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
        logger.info("✓ Tokenizer loaded from local files")
    except Exception as e:
        logger.warning(f"local_files_only failed ({e}); trying HuggingFace Hub...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        logger.info("✓ Tokenizer loaded from HuggingFace Hub")

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
        logger.info("✓ Model loaded from local files")
    except Exception as e:
        logger.warning(f"local_files_only failed ({e}); trying HuggingFace Hub...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
        )
        logger.info("✓ Model loaded from HuggingFace Hub")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model loaded: {total_params / 1e9:.2f}B parameters, dtype={model.dtype}")

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
        f"✅ Freezing complete — trainable: {trainable_params / 1e9:.2f}B "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # ------------------------------------------------------------------
    # 4. Dataset
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  [4/5] Loading Hendrycks MATH Dataset")
    logger.info(f"{'='*70}\n")

    subject_to_config = {
        "Algebra": "algebra",
        "Counting & Probability": "counting_and_probability",
        "Geometry": "geometry",
        "Intermediate Algebra": "intermediate_algebra",
        "Number Theory": "number_theory",
        "Prealgebra": "prealgebra",
        "Precalculus": "precalculus",
    }
    valid_levels = {f"Level {i}" for i in range(1, 6)}

    subjects_arg = normalize_csv_arg(args.math_subjects)
    subjects = (
        list(subject_to_config.keys())
        if subjects_arg.lower() == "all"
        else [normalize_csv_arg(s) for s in subjects_arg.split(",") if normalize_csv_arg(s)]
    )

    if args.math_dataset_source == "official":
        datasets_per_subject = []
        for subject in subjects:
            config_name = subject_to_config[subject]
            ds = load_dataset(
                args.math_official_dataset_path,
                config_name,
                split="train",
                cache_dir=args.cache_dir,
            )
            ds = ds.map(lambda ex, sub=subject: {"type": sub})
            datasets_per_subject.append(ds)
        train_ds = concatenate_datasets(datasets_per_subject)
    else:
        train_ds = load_dataset(
            args.math_flat_dataset_path, split="train", cache_dir=args.cache_dir
        )
        subject_set = set(subjects)
        train_ds = train_ds.filter(lambda ex: ex.get("type") in subject_set)

    levels_arg = normalize_csv_arg(args.math_levels)
    if levels_arg.lower() != "all":
        levels = []
        for item in levels_arg.split(","):
            item = normalize_csv_arg(item)
            if not item:
                continue
            lvl = item if item.startswith("Level ") else f"Level {int(item)}"
            if lvl not in valid_levels:
                raise ValueError(f"Invalid math level: {item}")
            levels.append(lvl)
        allowed_levels = set(levels)
        train_ds = train_ds.filter(lambda ex: ex.get("level") in allowed_levels)

    train_ds = train_ds.shuffle(seed=args.seed)
    if args.num_train_samples and args.num_train_samples > 0:
        train_ds = train_ds.select(range(min(args.num_train_samples, len(train_ds))))

    logger.info(f"✅ Train samples after filtering: {len(train_ds)}")

    def preprocess_train(ex, idx: int):
        problem = ex.get("problem", "").strip()
        solution = ex.get("solution", "").strip()
        rng = random.Random(args.seed + idx)
        target_text = build_target(solution, rng, args.math_train_on_mixed_formats)
        return tokenize_math_sft_example(problem, target_text, tokenizer, args.max_length, model_path)

    train_tok = train_ds.map(
        preprocess_train,
        with_indices=True,
        remove_columns=train_ds.column_names,
        num_proc=None,
        desc="Tokenizing Hendrycks MATH train",
    )

    eval_tok = None
    if args.num_eval_samples and args.num_eval_samples > 0:
        eval_tok = train_tok.select(range(min(args.num_eval_samples, len(train_tok))))

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*70}")
    logger.info("  [5/5] Training")
    logger.info(f"{'='*70}\n")

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    wandb.init(
        entity="gokms0509-yonsei-university",
        project="Hendrycks MATH Freeze SN Finetuning",
        name=run_name,
        config={
            "model_path": model_path,
            "safety_neurons_file": os.path.basename(args.safety_neurons_file),
            "strategy": "freeze_safety_neurons",
            "learning_rate": args.learning_rate,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": args.lr_scheduler_type,
            "dataset": "hendrycks_math",
            "math_subjects": args.math_subjects,
            "is_instruct": is_instruct_model(model_path),
        },
    )

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer)
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
        data_collator=data_collator,
        callbacks=[SafetyNeuronRestoreCallback(frozen_param_specs)],
    )

    logger.info("Starting training...")
    trainer.train()

    # ------------------------------------------------------------------
    # Save Model (with full verification, same as finetune_gsm8k_freeze_sn.py)
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
        logger.info(f"   ├─ Using safe_serialization=True (safetensors)")
        logger.info(f"   ├─ Output directory: {os.path.abspath(final_output_dir)}")
        model.save_pretrained(
            final_output_dir,
            safe_serialization=True,
            max_shard_size="4GB",
            push_to_hub=False,
        )
        logger.info("   └─ ✅ Model weights saved successfully")

        logger.info("Step 4: Saving tokenizer...")
        tokenizer.save_pretrained(final_output_dir, safe_serialization=True)
        logger.info("   └─ ✅ Tokenizer saved")

        logger.info("Step 5: Saving model config and generation settings...")
        model.config.save_pretrained(final_output_dir)
        if hasattr(model, "generation_config"):
            model.generation_config.save_pretrained(final_output_dir)
        logger.info("   └─ ✅ Configs saved")

        logger.info("Step 6: Verifying saved model integrity...")
        required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
        missing_files = []
        for fname in required_files:
            fpath = os.path.join(final_output_dir, fname)
            if not os.path.exists(fpath):
                missing_files.append(fname)
            else:
                logger.info(f"   ├─ {fname}: {os.path.getsize(fpath)/1024:.2f} KB ✅")
        if missing_files:
            raise FileNotFoundError(f"Missing/corrupted files: {missing_files}")

        model_files = [f for f in os.listdir(final_output_dir) if f.endswith(".safetensors")]
        if not model_files:
            raise FileNotFoundError("No safetensors files found after save!")
        logger.info(f"   ├─ ✅ Found {len(model_files)} model shard file(s)")

        logger.info("\n📦 Saved files:")
        total_size = 0
        for fname in sorted(os.listdir(final_output_dir)):
            fpath = os.path.join(final_output_dir, fname)
            if os.path.isfile(fpath):
                fsize = os.path.getsize(fpath)
                total_size += fsize
                logger.info(f"   ├─ {fname}: {fsize/1e9:.2f} GB")
        logger.info(f"   └─ Total size: {total_size/1e9:.2f} GB ✅")

        logger.info("\nStep 7: Final verification - attempting to load saved model...")
        try:
            test_tokenizer = AutoTokenizer.from_pretrained(final_output_dir)
            test_model = AutoModelForCausalLM.from_pretrained(
                final_output_dir,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                local_files_only=True,
            )
            del test_tokenizer, test_model
            gc.collect()
            logger.info("   └─ ✅ Model verified successfully!")
        except Exception as load_err:
            logger.error(f"   └─ ❌ Failed to load saved model: {load_err}")
            raise

        logger.info(f"\n✅✅✅ Fine-tuned model saved and verified successfully!")
        logger.info(f"   Output directory: {os.path.abspath(final_output_dir)}")
        logger.info(f"   Total size: {total_size/1e9:.2f} GB")
        logger.info(f"   Status: ✅ READY FOR EVALUATION")

    except Exception as e:
        logger.error(f"\n❌❌❌ CRITICAL ERROR during model saving: {e}")
        logger.error(traceback.format_exc())
        raise

    # Save training config
    config = {
        "base_model": args.model_path,
        "fine_tuning_type": "Hendrycks MATH Fine-tuning with Safety Neuron Freezing",
        "safety_neurons_file": args.safety_neurons_file,
        "dataset": "hendrycks_math",
        "math_dataset_source": args.math_dataset_source,
        "math_subjects": args.math_subjects,
        "math_levels": args.math_levels,
        "num_train_samples": args.num_train_samples,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optimizer": args.optim,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
        "input_formatting": (
            "chat template" if is_instruct_model(model_path) else "Question/Answer plain text"
        ),
        "strategy": "Freeze safety neurons, train others",
    }

    config_path = os.path.join(final_output_dir, "finetune_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"✅ Config saved to {config_path}")

    if args.upload_name:
        logger.info(f"\nStarting upload to Hugging Face: {args.upload_name}")
        try:
            from upload_sn_tuned_model import upload_to_huggingface

            upload_to_huggingface(final_output_dir, args.upload_name, args.hf_token)
            logger.info(f"✅ Upload completed: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.error("Model was saved locally; you can upload manually with upload_sn_tuned_model.py")

    logger.info(f"\n{'='*70}")
    logger.info("  ✅ Fine-tuning Complete!")
    logger.info(f"{'='*70}\n")
    wandb.finish()


if __name__ == "__main__":
    main()
