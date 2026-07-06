"""
ARC-Challenge 데이터셋을 사용하여 SN-Tuned 모델의 ARC-C finetuning (Safety Neuron Freeze)

Safety neuron은 freeze하고 나머지 파라미터만 학습하여 safety 성능 유지

Example Usage:
python finetune_arc-c_freeze_sn.py \
    --model_path kmseong/llama-2-7b-chat-hf-only-sn-tuned-lr5e-5 \
    --safety_neurons_file ./output_neurons/safety_neuron_accelerated_20260503_063506.txt \
    --output_dir ./llama2_7b_chat_arc_ft_freeze_sn_lr5e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/llama-2-7b-chat-hf-arc-sn-tuned-lr5e-5

python finetune_arc-c_freeze_sn.py \
    --model_path kmseong/llama-2-7b-chat-hf-only-rsn-tuned-lr5e-5 \
    --safety_neurons_file ./output_neurons/critical_safety_neuron_20260503_063923.txt \
    --output_dir ./llama2_7b_chat_arc_ft_freeze_rsn_lr5e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/llama-2-7b-chat-hf-arc-rsn-tuned-lr5e-5
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
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _peft_available = True
except ImportError:
    _peft_available = False


# arc_challenge_chat.yaml 의 doc_to_text와 동일한 포맷
ARC_CHAT_PROMPT_TEMPLATE = (
    'Given the following question and four candidate answers (A, B, C and D), '
    'choose the best answer.\n'
    'Question: {question}\n'
    '{choices}\n'
    'Your response should end with "The best answer is [the_answer_letter]" '
    'where the [the_answer_letter] is one of A, B, C or D.'
)
ARC_GEN_PREFIX = "The best answer is"
LETTER_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}


# =====================================================================
# Argument parsing
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description='ARC-Challenge Finetuning with Safety Neuron Freezing')

    # model
    p.add_argument('--model_path', type=str, required=True,
                   help='HuggingFace model ID or local path (SN-Tuned model)')

    # safety neurons
    p.add_argument('--safety_neurons_file', type=str, required=True,
                   help='Path to safety neurons txt file')

    # data
    p.add_argument("--dataset_name",      type=str, default="allenai/ai2_arc")
    p.add_argument("--dataset_subset",    type=str, default="ARC-Challenge")
    p.add_argument("--train_split",       type=str, default="train")
    p.add_argument("--eval_split",        type=str, default="test")
    p.add_argument("--num_train_samples", type=int, default=0,
                   help="학습 샘플 수 (0=전체, ARC-Challenge train=1119)")
    p.add_argument("--num_eval_samples",  type=int, default=0)
    p.add_argument("--seed",              type=int, default=42)

    # safety data mixing
    p.add_argument("--safety_data_path", type=str,
                   default="./corpus_all/circuit_breakers_train.json",
                   help="Safety dataset JSON 경로 (circuit_breakers_train.json 형식)")
    p.add_argument("--safety_mix_ratio", type=float, default=0.0,
                   help="ARC 데이터 수 대비 safety 데이터 비율 (e.g. 0.1 = 10%%, 0=비활성화)")

    # training
    p.add_argument("--batch_size",         type=int,   default=2)
    p.add_argument("--eval_batch_size",    type=int,   default=4)
    p.add_argument("--grad_accum",         type=int,   default=8)
    p.add_argument("--epochs",             type=int,   default=3)
    p.add_argument("--learning_rate",      type=float, default=5e-5)
    p.add_argument("--weight_decay",       type=float, default=0.01)
    p.add_argument("--warmup_ratio",       type=float, default=0.1)
    p.add_argument("--lr_scheduler_type",  type=str,   default="cosine")
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--optim",              type=str,   default="adamw_torch")
    p.add_argument("--max_length",         type=int,   default=1024)

    p.add_argument("--bf16",                   action="store_true", default=True)
    p.add_argument("--fp16",                   action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # logging/saving
    p.add_argument("--output_dir",     type=str, default='./arc_freeze_sn_finetune')
    p.add_argument("--logging_steps",  type=int, default=10)
    p.add_argument("--eval_steps",     type=int, default=500)
    p.add_argument("--report_to",      type=str, default="wandb")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--cache_dir",      type=str, default='./cache')
    p.add_argument("--upload_name",    type=str, default=None,
                   help="HF repo id. 설정 시 학습 후 자동 업로드")
    p.add_argument("--hf_token",       type=str, default=None)

    return p.parse_args()


# =====================================================================
# Helpers
# =====================================================================

def _select_first_n(ds, n: int):
    if n is None or n <= 0:
        return ds
    return ds.select(range(min(n, len(ds))))


def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower() or "chat" in str(model_ref).lower()


# =====================================================================
# ARC-C formatting & tokenization
# =====================================================================

def format_arc_question(question: str, choices: dict) -> str:
    """arc_challenge_chat.yaml 의 doc_to_text 포맷으로 문제 생성."""
    labels = choices["label"]
    texts  = choices["text"]
    choice_lines = []
    for lbl, txt in zip(labels, texts):
        letter = LETTER_MAP.get(str(lbl), str(lbl))
        choice_lines.append(f"{letter}. {txt}")
    return ARC_CHAT_PROMPT_TEMPLATE.format(
        question=question.strip(),
        choices="\n".join(choice_lines),
    )


def get_arc_answer_letter(answer_key: str) -> str:
    """answerKey를 A/B/C/D 레터로 반환."""
    return LETTER_MAP.get(str(answer_key), str(answer_key))


def tokenize_sft_example(
    question_with_choices: str,
    answer_text: str,
    tokenizer,
    max_length: int,
    model_ref: str,
) -> Dict[str, List[int]]:
    """SFT 형식으로 토큰화: instruct는 chat template, base는 plain prompt."""
    question_with_choices = str(question_with_choices).strip()
    answer_text = str(answer_text).strip()

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question_with_choices}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user",      "content": question_with_choices},
                    {"role": "assistant", "content": answer_text},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(
                prompt_text, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]
            labels = full_ids.copy()
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            return {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        except Exception:
            pass

    # base 모델: plain prompt
    plain_prompt = f"{question_with_choices}\n{ARC_GEN_PREFIX} "
    prompt_ids = tokenizer(
        plain_prompt, add_special_tokens=False, truncation=True, max_length=max_length
    )["input_ids"]
    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        answer_text, add_special_tokens=False, truncation=True, max_length=remain
    )["input_ids"]
    if (tokenizer.eos_token_id is not None
            and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id)
            and len(prompt_ids) + len(answer_ids) < max_length):
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels    = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


# =====================================================================
# Data collator
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
            input_ids.append(f["input_ids"]       + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0]      * pad_len)
            labels.append(f["labels"]              + [-100]  * pad_len)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


# =====================================================================
# Load Safety Neurons from Detection Output
# =====================================================================

def load_safety_neurons(output_file: str, logger) -> dict:
    """
    Load safety neurons from detection output file.

    Format:
        Line 0: ffn_up_common  (dict)
        Line 1: ffn_down_common (dict)
        Line 2: q_common       (dict)
        Line 3: k_common       (dict)
        Line 4: v_common       (dict)

    Returns:
        {
            'ffn_up':   {layer_idx: set(neuron_names)},
            'ffn_down': {layer_idx: set(neuron_names)},
            'q':        {layer_idx: set(neuron_names)},
            'k':        {layer_idx: set(neuron_names)},
            'v':        {layer_idx: set(neuron_names)},
        }
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    try:
        ffn_up_raw   = ast.literal_eval(lines[0].strip())
        ffn_down_raw = ast.literal_eval(lines[1].strip())
        q_raw        = ast.literal_eval(lines[2].strip())
        k_raw        = ast.literal_eval(lines[3].strip())
        v_raw        = ast.literal_eval(lines[4].strip())

        safety_neurons = {
            'ffn_up':   {int(k): v for k, v in ffn_up_raw.items()},
            'ffn_down': {int(k): v for k, v in ffn_down_raw.items()},
            'q':        {int(k): v for k, v in q_raw.items()},
            'k':        {int(k): v for k, v in k_raw.items()},
            'v':        {int(k): v for k, v in v_raw.items()},
        }
    except Exception as e:
        logger.error(f"Error parsing safety neurons file: {e}")
        raise

    logger.info(f"Loaded safety neurons from {output_file}")
    logger.info(f"\n{'='*70}")
    logger.info(f"Safety Neurons Loaded - Detailed Breakdown")
    logger.info(f"{'='*70}")

    total_neurons = 0
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        module_total = sum(len(v) for v in safety_neurons[module_type].values())
        logger.info(f"  {module_type:12} : {module_total:4} neurons")
        total_neurons += module_total
        layers_with = [l for l, ns in safety_neurons[module_type].items() if ns]
        if layers_with:
            logger.info(
                f"    └─ Layers with neurons: {layers_with[:5]}{'...' if len(layers_with) > 5 else ''}"
            )

    logger.info(f"\nTotal safety neurons: {total_neurons}")
    logger.info(f"{'='*70}\n")
    return safety_neurons


# =====================================================================
# Freeze Safety Neurons
# =====================================================================

def setup_safety_neuron_freezing(model, safety_neurons: dict, logger) -> list:
    """
    Safety neuron을 freeze하고 나머지 파라미터만 학습 가능하게 설정.

    finetune_gsm8k_freeze_sn.py와 동일한 방식:
    - 모든 파라미터 requires_grad=True
    - safety neuron 위치에 gradient hook으로 0 처리
    - AdamW weight-decay drift 방지용 frozen_param_specs 반환
    """
    total_params        = 0
    frozen_neuron_params = 0
    frozen_modules      = {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0}
    frozen_param_specs  = []  # (param, indices, axis)

    def _sanitize_indices(raw_indices, dim: int, module_name: str, layer_idx: int):
        parsed, dropped = [], 0
        for x in raw_indices:
            idx = None
            if isinstance(x, int):
                idx = x
            elif isinstance(x, str):
                s = x.strip()
                if s.lstrip("-").isdigit():
                    idx = int(s)
                else:
                    m = re.search(r"-?\d+", s)
                    if m:
                        idx = int(m.group(0))
            if idx is None:
                dropped += 1
                continue
            if 0 <= idx < dim:
                parsed.append(idx)
            else:
                dropped += 1
        uniq = sorted(set(parsed))
        if dropped > 0:
            logger.warning(
                f"[Index sanitize] layer={layer_idx}, module={module_name}, "
                f"kept={len(uniq)}, dropped={dropped}, dim={dim}"
            )
        return uniq

    def _make_zero_hook_rows(indices):
        def hook(grad):
            grad = grad.clone()
            grad[indices, :] = 0.0
            return grad
        return hook

    def _make_zero_hook_cols(indices):
        def hook(grad):
            grad = grad.clone()
            grad[:, indices] = 0.0
            return grad
        return hook

    # Step 1: 모든 파라미터 학습 가능
    for param in model.parameters():
        param.requires_grad = True

    # Step 2: safety neuron 위치에 gradient hook 등록
    for name, param in model.named_parameters():
        total_params += param.numel()
        parts = name.split('.')
        if len(parts) < 4 or parts[0] != 'model' or parts[1] != 'layers':
            continue
        try:
            layer_idx = int(parts[2])
        except ValueError:
            continue

        if 'mlp.up_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['ffn_up'].get(layer_idx, []), param.shape[0], 'ffn_up', layer_idx
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['ffn_up'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'mlp.down_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['ffn_down'].get(layer_idx, []), param.shape[1], 'ffn_down', layer_idx
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[0]
                frozen_modules['ffn_down'] += 1
                param.register_hook(_make_zero_hook_cols(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'cols'))

        elif 'self_attn.q_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['q'].get(layer_idx, []), param.shape[0], 'q', layer_idx
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['q'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'self_attn.k_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['k'].get(layer_idx, []), param.shape[0], 'k', layer_idx
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['k'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'self_attn.v_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['v'].get(layer_idx, []), param.shape[0], 'v', layer_idx
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['v'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n{'='*70}")
    logger.info(f"Safety Neuron Freezing Setup Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total parameters:                         {total_params:,}")
    logger.info(f"Frozen safety neuron parameters (eff.):   {frozen_neuron_params:,}")
    logger.info(f"Trainable parameters:                     {trainable_params:,}")
    logger.info(f"Trainable ratio:                          {trainable_params / total_params * 100:.4f}%")
    logger.info(f"Frozen safety neuron ratio:               {frozen_neuron_params / total_params * 100:.4f}%")
    logger.info(f"\nLayers with frozen safety neurons:")
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

    AdamW weight-decay(λθ)는 gradient hook과 독립적으로 적용되므로
    hook으로 gradient를 0으로 만들어도 weight-decay에 의해 값이 변한다.
    이 콜백이 매 step 후 초기 값으로 복원하여 완전한 freeze를 보장한다.
    """

    def __init__(self, frozen_param_specs):
        self._specs = frozen_param_specs
        self._frozen_vals = []
        for param, indices, axis in frozen_param_specs:
            with torch.no_grad():
                if axis == 'rows':
                    self._frozen_vals.append(param.data[indices, :].clone())
                else:
                    self._frozen_vals.append(param.data[:, indices].clone())

    def on_step_end(self, args, state, control, **kwargs):
        for (param, indices, axis), frozen_val in zip(self._specs, self._frozen_vals):
            with torch.no_grad():
                if axis == 'rows':
                    param.data[indices, :] = frozen_val
                else:
                    param.data[:, indices] = frozen_val
        return control


# =====================================================================
# Logging
# =====================================================================

def setup_logging(output_dir: str):
    log_dir = "./logs/safety_neuron_arc"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"finetune_arc_freeze_sn_{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    logger, log_file = setup_logging(args.output_dir)
    logger.info(f"\n{'='*70}")
    logger.info(f"  ARC-Challenge Fine-tuning with Safety Neuron Freezing")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")

    if not os.path.exists(args.safety_neurons_file):
        raise FileNotFoundError(f"Safety neurons file not found: {args.safety_neurons_file}")

    raw_path   = args.model_path
    is_local   = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logger.info(f"Configuration:")
    logger.info(f"  ├─ SN-Tuned model     : {model_path}")
    logger.info(f"  ├─ Safety neurons file: {args.safety_neurons_file}")
    logger.info(f"  ├─ Dataset            : {args.dataset_name} / {args.dataset_subset}")
    logger.info(f"  ├─ Input format       : {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f"  ├─ Train samples      : {args.num_train_samples} (0=all)")
    logger.info(f"  ├─ Safety mix ratio   : {args.safety_mix_ratio}")
    logger.info(f"  ├─ LR={args.learning_rate}, epochs={args.epochs}, batch={args.batch_size}x{args.grad_accum}")
    logger.info(f"  ├─ Strategy           : Freeze safety neurons, train others")
    logger.info(f"  └─ Output dir         : {args.output_dir}")

    if args.bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        args.bf16 = False
        if not args.fp16:
            args.fp16 = True

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    wandb.init(
        entity="gokms0509-yonsei-university",
        project="ARC-Challenge Freeze-SN Finetuning",
        name=run_name,
        config={
            "model_path": model_path,
            "safety_neurons_file": os.path.basename(args.safety_neurons_file),
            "strategy": "freeze_safety_neurons",
            "dataset": f"{args.dataset_name}/{args.dataset_subset}",
            "learning_rate": args.learning_rate,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "max_length": args.max_length,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": args.lr_scheduler_type,
            "safety_mix_ratio": args.safety_mix_ratio,
            "is_instruct": is_instruct_model(model_path),
        },
    )

    # ── [1/5] Tokenizer ──────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"  [1/5] Loading Tokenizer")
    logger.info(f"{'='*70}\n")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        logger.info("✓ Tokenizer loaded from local files")
    except Exception as e:
        logger.warning(f"Local tokenizer load failed: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        logger.info("✓ Tokenizer loaded from HuggingFace Hub")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"✅ Tokenizer ready  (vocab={len(tokenizer)}, pad='{tokenizer.pad_token}')")

    # ── [2/5] Model ───────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"  [2/5] Loading Model (bf16)")
    logger.info(f"{'='*70}\n")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto",
            local_files_only=True, trust_remote_code=False
        )
        logger.info("✓ Model loaded from local files")
    except Exception as e:
        logger.warning(f"Local model load failed: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=False
        )
        logger.info("✓ Model loaded from HuggingFace Hub")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model ready  ({total_params/1e9:.2f}B params, dtype={model.dtype})")

    # ── [3/5] Safety Neuron Freeze setup ─────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"  [3/5] Loading Safety Neurons and Setting up Freezing")
    logger.info(f"{'='*70}\n")
    safety_neurons    = load_safety_neurons(args.safety_neurons_file, logger)
    frozen_param_specs = setup_safety_neuron_freezing(model, safety_neurons, logger)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"✅ Freezing complete  "
        f"(trainable={trainable_params/1e9:.2f}B, {100*trainable_params/total_params:.2f}%)"
    )

    # ── [4/5] Dataset ─────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"  [4/5] Loading ARC-Challenge Dataset")
    logger.info(f"{'='*70}\n")
    train_ds = load_dataset(
        args.dataset_name, args.dataset_subset,
        split=args.train_split, cache_dir=args.cache_dir
    )
    train_ds = _select_first_n(train_ds, args.num_train_samples)

    eval_ds = None
    if args.num_eval_samples and args.num_eval_samples > 0:
        eval_ds = load_dataset(
            args.dataset_name, args.dataset_subset,
            split=args.eval_split, cache_dir=args.cache_dir
        )
        eval_ds = _select_first_n(eval_ds, args.num_eval_samples)

    logger.info(f"✅ Train: {len(train_ds)} samples" +
                (f"  /  Eval: {len(eval_ds)}" if eval_ds is not None else ""))

    def preprocess(ex):
        question_with_choices = format_arc_question(ex["question"], ex["choices"])
        answer_letter = get_arc_answer_letter(ex["answerKey"])
        answer_text   = f"{ARC_GEN_PREFIX} {answer_letter}"
        return tokenize_sft_example(
            question_with_choices, answer_text, tokenizer, args.max_length, model_path
        )

    train_tok = train_ds.map(
        preprocess, remove_columns=train_ds.column_names,
        num_proc=max(1, args.num_workers), desc="Tokenizing train",
    )
    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(
            preprocess, remove_columns=eval_ds.column_names,
            num_proc=max(1, args.num_workers), desc="Tokenizing eval",
        )

    # Safety data mixing
    if args.safety_mix_ratio > 0:
        if not os.path.exists(args.safety_data_path):
            raise FileNotFoundError(f"Safety dataset not found: {args.safety_data_path}")
        with open(args.safety_data_path, "r", encoding="utf-8") as f:
            safety_raw = json.load(f)
        num_safety = int(len(train_tok) * args.safety_mix_ratio)
        rng     = random.Random(args.seed)
        sampled = rng.sample(safety_raw, min(num_safety, len(safety_raw)))

        def preprocess_safety(ex):
            return tokenize_sft_example(
                ex["prompt"], ex["llama3_output"], tokenizer, args.max_length, model_path
            )

        safety_hf  = HFDataset.from_list(sampled)
        safety_tok = safety_hf.map(
            preprocess_safety, remove_columns=safety_hf.column_names,
            desc="Tokenizing safety data",
        )
        train_tok = concatenate_datasets([train_tok, safety_tok]).shuffle(seed=args.seed)
        logger.info(
            f"✅ Safety data mixed: {len(safety_tok)} samples (ratio={args.safety_mix_ratio}), "
            f"total={len(train_tok)}"
        )

    # ── [5/5] Training ────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"  [5/5] Training with Trainer + AdamW")
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
        eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps if do_eval else None,
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
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
        callbacks=[SafetyNeuronRestoreCallback(frozen_param_specs)],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("✓ Training completed")

    # ── Save ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f"  Saving Fine-tuned Model")
    logger.info(f"{'='*70}\n")

    try:
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = f"{args.output_dir}_{timestamp}"

        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Moving model to CPU for safe serialization...")
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        model.save_pretrained(
            final_output_dir,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(final_output_dir)
        model.config.save_pretrained(final_output_dir)
        if hasattr(model, 'generation_config'):
            model.generation_config.save_pretrained(final_output_dir)

        # 저장 파일 목록 로깅
        total_size = 0
        for fname in sorted(os.listdir(final_output_dir)):
            fpath = os.path.join(final_output_dir, fname)
            sz = os.path.getsize(fpath)
            total_size += sz
            logger.info(f"  {fname:50s}  {sz/1e6:8.2f} MB")
        logger.info(f"  Total: {total_size/1e9:.2f} GB")

        logger.info(f"✅ Model saved to: {os.path.abspath(final_output_dir)}")

    except Exception as e:
        logger.error(f"❌ Model saving failed: {e}")
        logger.error(traceback.format_exc())
        raise

    # Training config 저장
    config = {
        'base_model': args.model_path,
        'fine_tuning_type': 'ARC-Challenge Fine-tuning with Safety Neuron Freezing',
        'safety_neurons_file': args.safety_neurons_file,
        'dataset': args.dataset_name,
        'dataset_subset': args.dataset_subset,
        'num_train_samples': args.num_train_samples,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_ratio': args.warmup_ratio,
        'epochs': args.epochs,
        'max_length': args.max_length,
        'max_grad_norm': args.max_grad_norm,
        'lr_scheduler_type': args.lr_scheduler_type,
        'optimizer': args.optim,
        'gradient_checkpointing': args.gradient_checkpointing,
        'dtype': 'bf16',
        'trainer_type': 'Trainer',
        'strategy': 'Freeze safety neurons, train others',
        'safety_mix_ratio': args.safety_mix_ratio,
    }
    with open(os.path.join(final_output_dir, 'finetune_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    if args.upload_name:
        logger.info(f"\nUploading to HuggingFace: {args.upload_name}")
        try:
            from upload_sn_tuned_model import upload_to_huggingface
            upload_to_huggingface(final_output_dir, args.upload_name, args.hf_token)
            logger.info(f"✅ Upload completed: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")

    logger.info(f"\n{'='*70}")
    logger.info(f"  ✅ Fine-tuning Complete!")
    logger.info(f"{'='*70}\n")
    wandb.finish()


if __name__ == '__main__':
    main()
