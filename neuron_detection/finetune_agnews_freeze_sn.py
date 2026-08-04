"""
AG News 데이터셋으로 SN-Tuned 모델을 finetuning 한다 (Safety Neuron Freeze).

Safety neuron 은 freeze 하고 나머지 파라미터만 학습해 safety 성능을 유지한다.
- 데이터 로딩/프롬프트 포맷: Safety-WaRP-LLM/agnews_eval/finetune_agnews_full_params.py
- Safety neuron freezing: finetune_medqa_freeze_sn.py

medqa 스크립트와 달리 최종 모델을 `--output_dir` 에 **그대로** 저장한다
(`_<timestamp>` 접미사를 붙이지 않는다). downstream 체인이 경로를 추측할 필요가 없다.

GPU 는 호출하는 셸에서 CUDA_VISIBLE_DEVICES 로 지정한다. 이 모듈은 import 시점에
CUDA_VISIBLE_DEVICES 를 건드리지 않는다 (되살리지 말 것).

Example Usage:
CUDA_VISIBLE_DEVICES=1 python finetune_agnews_freeze_sn.py \
    --model_path /path/to/sn_tuned_llama2_7b-chat_lr5e-5_20260803_000000 \
    --safety_neurons_file ./output_neurons/safety_neuron_accelerated_20260728_044429.txt \
    --agnews_train_path /home/edgeai_lab/Safety-WaRP-LLM/data/agnews_train_8k_seed42.json \
    --output_dir ./agnews_sn_finetune_llama2_7b-chat \
    --learning_rate 3e-5 --epochs 3 \
    --upload_name kmseong/llama2_7b-chat-agnews-sn-tuned-lr3e-5
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# ⚠️ import 시점에 CUDA_VISIBLE_DEVICES 를 박아두면 이 모듈을 import 하는 모든 스크립트가
#    오염되고, SLURM 환경에서는 스케줄러가 설정한 값을 덮어쓴다.
#    GPU 지정은 호출하는 셸에서 하라. 되살리지 말 것.
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


AGNEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

AGNEWS_INSTRUCTION = (
    "Categorize the news article given in the input into one of the 4 categories:\n\n"
    "World\n"
    "Sports\n"
    "Business\n"
    "Sci/Tech"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="AG News SFT with safety-neuron freezing."
    )

    # ── Model / neurons ──────────────────────────────────────────────────────
    p.add_argument('--model_path', type=str, required=True,
                   help='SN-Tuned 모델 경로 또는 HF repo id')
    p.add_argument('--safety_neurons_file', type=str, required=True,
                   help='freeze 할 safety neuron 파일 (5줄 포맷)')

    # ── Dataset ──────────────────────────────────────────────────────────────
    p.add_argument('--agnews_train_path', type=str, required=True,
                   help='AG News 학습 데이터 (로컬 json/jsonl, HF dataset dir, 또는 hub id)')
    p.add_argument('--agnews_eval_path', type=str, default=None)
    p.add_argument('--agnews_train_split', type=str, default='train')
    p.add_argument('--agnews_eval_split', type=str, default='test')
    p.add_argument('--num_train_samples', type=int, default=0,
                   help='0 이면 전체 사용')
    p.add_argument('--num_eval_samples', type=int, default=0,
                   help='0 이면 eval 을 수행하지 않는다')
    p.add_argument('--seed', type=int, default=42)

    # ── Optional safety data mixing ──────────────────────────────────────────
    p.add_argument('--safety_data_path', type=str,
                   default='/home/edgeai_lab/Safety-WaRP-LLM/data/circuit_breakers_train.json')
    p.add_argument('--safety_mix_ratio', type=float, default=0.0,
                   help='>0 이면 safety 데이터를 비율만큼 섞는다 (SafeInstr baseline)')

    # ── Training ─────────────────────────────────────────────────────────────
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_batch_size', type=int, default=4)
    p.add_argument('--grad_accum', type=int, default=4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--learning_rate', type=float, default=3e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--lr_scheduler_type', type=str, default='cosine')
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--optim', type=str, default='adamw_torch')
    p.add_argument('--max_length', type=int, default=1024)

    p.add_argument('--bf16', action='store_true', default=True)
    p.add_argument('--fp16', action='store_true', default=False)
    p.add_argument('--gradient_checkpointing', action='store_true', default=False)

    # ── Output / logging ─────────────────────────────────────────────────────
    p.add_argument('--output_dir', type=str, default='./agnews_freeze_sn_finetune')
    p.add_argument('--logging_steps', type=int, default=10)
    p.add_argument('--eval_steps', type=int, default=500)
    p.add_argument('--report_to', type=str, default='none',
                   help="'wandb' 로 켤 수 있다. 기본값 none (wandb 로그인 없이도 동작)")
    p.add_argument('--wandb_project', type=str, default='AGNews Freeze SN Finetuning')
    p.add_argument('--wandb_entity', type=str, default=None)
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--cache_dir', type=str, default='./cache')

    p.add_argument('--upload_name', type=str, default=None,
                   help='설정하면 학습 후 HF Hub 로 업로드한다')
    p.add_argument('--hf_token', type=str, default=None)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging():
    log_dir = './logs/safety_neuron_agnews'
    os.makedirs(log_dir, exist_ok=True)

    log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'finetune_agnews_freeze_sn_{log_timestamp}.log')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# ─────────────────────────────────────────────────────────────────────────────
# Prompt formatting (finetune_agnews_full_params.py 와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def is_instruct_model(model_ref: str) -> bool:
    model_ref = str(model_ref).lower()
    return 'instruct' in model_ref or 'chat' in model_ref


def _as_text(value) -> str:
    return '' if value is None else str(value).strip()


def _select_random_n(ds, n: int, seed: int):
    if n is None or n <= 0 or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def render_chat_fallback(prompt: str, response: Optional[str], model_ref: str) -> Tuple[str, str]:
    prompt = prompt.strip()
    model_ref = str(model_ref).lower()
    if 'llama-2' in model_ref or 'llama2' in model_ref:
        prompt_text = f'<s>[INST] {prompt} [/INST]'
        if response is None:
            return prompt_text, prompt_text
        return prompt_text, f'{prompt_text} {response.strip()} </s>'

    prompt_text = f'User:\n{prompt}\n\nAssistant:\n'
    if response is None:
        return prompt_text, prompt_text
    return prompt_text, f'{prompt_text}{response.strip()}'


def _agnews_label_to_text(row: Dict) -> str:
    if row.get('completion') is not None:
        return _as_text(row.get('completion'))
    if row.get('output') is not None:
        return _as_text(row.get('output'))
    if row.get('label_text') is not None:
        return _as_text(row.get('label_text'))
    label = row.get('label')
    if label is None:
        return ''
    try:
        return AGNEWS_LABELS[int(label)]
    except (TypeError, ValueError, KeyError):
        return _as_text(label)


def agnews_prompt_response(row: Dict, prefer_chat: bool = False) -> Tuple[str, str]:
    if not prefer_chat:
        if row.get('prompt') is not None and row.get('completion') is not None:
            return _as_text(row.get('prompt')), _as_text(row.get('completion'))

        if row.get('instruction') is not None and row.get('input') is not None:
            prompt = (
                '### Instruction:\n'
                f"{_as_text(row.get('instruction'))}\n"
                '### Input:\n'
                f"{_as_text(row.get('input'))}\n\n"
                '### Response:\n'
            )
            return prompt, _agnews_label_to_text(row)

    messages = row.get('messages')
    if isinstance(messages, list) and messages:
        user_parts = [
            _as_text(message.get('content'))
            for message in messages
            if message.get('role') != 'assistant'
        ]
        assistant_parts = [
            _as_text(message.get('content'))
            for message in messages
            if message.get('role') == 'assistant'
        ]
        return '\n\n'.join(x for x in user_parts if x), assistant_parts[-1] if assistant_parts else ''

    if row.get('prompt') is not None and row.get('completion') is not None:
        return _as_text(row.get('prompt')), _as_text(row.get('completion'))

    if row.get('instruction') is not None and row.get('input') is not None:
        prompt = f"{_as_text(row.get('instruction'))}\n\nNews article:\n{_as_text(row.get('input'))}"
        return prompt, _agnews_label_to_text(row)

    if row.get('text') is not None:
        prompt = f"{AGNEWS_INSTRUCTION}\n\nNews article:\n{_as_text(row.get('text'))}"
        return prompt, _agnews_label_to_text(row)

    title = _as_text(row.get('title'))
    description = _as_text(row.get('description'))
    if title or description:
        article = ' '.join(x for x in (title, description) if x)
        prompt = f'{AGNEWS_INSTRUCTION}\n\nNews article:\n{article}'
        return prompt, _agnews_label_to_text(row)

    return '', ''


def _keep_answer_budget(
    prompt_ids: List[int],
    answer_ids: List[int],
    max_length: int,
) -> Tuple[List[int], List[int]]:
    if len(prompt_ids) + len(answer_ids) <= max_length:
        return prompt_ids, answer_ids

    if max_length <= 1:
        raise ValueError(f'max_length must be > 1, got {max_length}')

    answer_floor = max(1, max_length // 4)
    answer_budget = min(len(answer_ids), answer_floor)
    prompt_budget = max_length - answer_budget

    answer_ids = answer_ids[:answer_budget]
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]
    return prompt_ids, answer_ids


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
        raise ValueError('prompt and response must be non-empty')

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
            full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids):]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)['input_ids']
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
        except Exception:
            prompt_text, full_text = render_chat_fallback(prompt, response, model_ref)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
            full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']
            if full_ids[: len(prompt_ids)] == prompt_ids:
                answer_ids = full_ids[len(prompt_ids):]
            else:
                answer_ids = tokenizer(response, add_special_tokens=False)['input_ids']
                if tokenizer.eos_token_id is not None:
                    answer_ids.append(tokenizer.eos_token_id)
    else:
        prompt_ids = tokenizer(prompt + '\n', add_special_tokens=False)['input_ids']
        answer_ids = tokenizer(response, add_special_tokens=False)['input_ids']
        if tokenizer.eos_token_id is not None:
            answer_ids.append(tokenizer.eos_token_id)

    prompt_ids, answer_ids = _keep_answer_budget(prompt_ids, answer_ids, max_length=max_length)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if not any(label != -100 for label in labels):
        raise ValueError('tokenization produced no supervised response tokens')

    return {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'labels': labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataset_path(path: str, split: str, cache_dir: Optional[str] = None):
    path_obj = Path(path)
    if path_obj.exists() and path_obj.is_dir():
        loaded = load_from_disk(path)
        return loaded[split] if hasattr(loaded, 'keys') else loaded
    if path_obj.exists():
        return load_dataset('json', data_files=path, split='train', cache_dir=cache_dir)
    loaded = load_dataset(path, cache_dir=cache_dir)
    return loaded[split] if hasattr(loaded, 'keys') else loaded


def load_agnews_rows(args, logger, eval_dataset: bool = False):
    path = args.agnews_eval_path if eval_dataset else args.agnews_train_path
    split = args.agnews_eval_split if eval_dataset else args.agnews_train_split
    if eval_dataset and not path:
        return None
    if not path:
        raise ValueError('--agnews_train_path is required')

    ds = _load_dataset_path(path, split, args.cache_dir)
    n = args.num_eval_samples if eval_dataset else args.num_train_samples
    ds = _select_random_n(ds, n, args.seed + (1 if eval_dataset else 0))
    logger.info(f"Loaded AG News {'eval' if eval_dataset else 'train'} rows: {len(ds)}")
    return ds


def tokenize_dataset_rows(ds, args, tokenizer, model_path: str, logger):
    """행 단위 루프로 토크나이즈한다. 깨진 행은 datasets.map 처럼 죽지 않고 건너뛴다."""
    tokenized = []
    skipped = 0

    for row in ds:
        row = dict(row)
        prompt, response = agnews_prompt_response(
            row,
            prefer_chat=is_instruct_model(model_path),
        )

        if not prompt or not response:
            skipped += 1
            continue

        try:
            tokenized.append(
                tokenize_prompt_response(
                    prompt,
                    response,
                    tokenizer,
                    args.max_length,
                    model_path,
                )
            )
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                logger.warning(f'Skipping malformed row: {exc}')

    if not tokenized:
        raise ValueError('No valid tokenized examples for AG News')
    if skipped:
        logger.warning(f'Skipped {skipped} rows during tokenization')

    return HFDataset.from_dict(
        {
            'input_ids': [x['input_ids'] for x in tokenized],
            'attention_mask': [x['attention_mask'] for x in tokenized],
            'labels': [x['labels'] for x in tokenized],
        }
    )


def maybe_mix_safety(train_tok, args, tokenizer, model_path: str, logger, num_agnews: int = 0):
    if args.safety_mix_ratio <= 0:
        return train_tok

    if not os.path.exists(args.safety_data_path):
        raise FileNotFoundError(f'Safety dataset not found: {args.safety_data_path}')

    with open(args.safety_data_path, 'r', encoding='utf-8') as f:
        safety_raw = json.load(f)

    num_safety = int(len(train_tok) * args.safety_mix_ratio)
    rng = random.Random(args.seed)
    sampled = rng.sample(safety_raw, min(num_safety, len(safety_raw)))

    def preprocess_safety(ex):
        return tokenize_prompt_response(
            ex['prompt'],
            ex['llama3_output'],
            tokenizer,
            args.max_length,
            model_path,
        )

    safety_hf = HFDataset.from_list(sampled)
    safety_tok = safety_hf.map(
        preprocess_safety,
        remove_columns=safety_hf.column_names,
        desc='Tokenizing safety data',
    )

    mixed = concatenate_datasets([train_tok, safety_tok]).shuffle(seed=args.seed)
    logger.info(f'Safety data mixed: {len(safety_tok)} samples (ratio={args.safety_mix_ratio})')
    logger.info(
        f'Total training samples: {len(mixed)} '
        f'(AG News {num_agnews} + Safety {len(safety_tok)})'
    )
    return mixed


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f['input_ids']) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            input_ids.append(f['input_ids'] + [pad_id] * pad_len)
            attention_mask.append(f['attention_mask'] + [0] * pad_len)
            labels.append(f['labels'] + [-100] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Safety neuron loading (finetune_medqa_freeze_sn.py 와 동일)
# ─────────────────────────────────────────────────────────────────────────────

def load_safety_neurons(output_file, logger):
    """
    Load safety neurons from detection output file.

    Format:
        Line 0: ffn_up_common (dict)
        Line 1: ffn_down_common (dict)
        Line 2: q_common (dict)
        Line 3: k_common (dict)
        Line 4: v_common (dict)
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
        logger.error(f'Error parsing safety neurons file: {e}')
        raise

    logger.info(f'Loaded safety neurons from {output_file}')
    logger.info(f"\n{'='*70}")
    logger.info(f'Safety Neurons Loaded - Detailed Breakdown')
    logger.info(f"{'='*70}")

    total_neurons = 0
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        module_total = sum(len(neurons) for neurons in safety_neurons[module_type].values())
        logger.info(f'  {module_type:12} : {module_total:4} neurons')
        total_neurons += module_total
        layers_with_neurons = [l for l in safety_neurons[module_type] if safety_neurons[module_type][l]]
        if layers_with_neurons:
            logger.info(f"    └─ Layers with neurons: {layers_with_neurons[:5]}{'...' if len(layers_with_neurons) > 5 else ''}")

    logger.info(f'\nTotal safety neurons: {total_neurons}')
    logger.info(f"{'='*70}\n")

    if total_neurons == 0:
        raise ValueError(
            f'{output_file} 에서 검출된 뉴런이 0개입니다 — freeze 할 대상이 없어 '
            f'일반 full-parameter FT 와 동일해집니다. 탐지 결과를 확인하세요.'
        )
    return safety_neurons


# ─────────────────────────────────────────────────────────────────────────────
# Freeze Safety Neurons
# ─────────────────────────────────────────────────────────────────────────────

def setup_safety_neuron_freezing(model, safety_neurons, logger):
    """
    Freeze safety neurons and train only the remaining parameters.

    This is the REVERSE of sn_tune.py's setup_gradient_masking:
    - sn_tune.py: freeze all, train only safety neurons
    - This function: train all, freeze only safety neurons

    Returns frozen_param_specs: list of (param, indices, axis) used by
    SafetyNeuronRestoreCallback to undo weight-decay updates on safety neurons.
    """
    total_params         = 0
    frozen_neuron_params = 0
    frozen_modules       = {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0}
    frozen_param_specs   = []

    def _sanitize_indices(raw_indices, dim: int, module_name: str, layer_idx: int):
        parsed  = []
        dropped = 0
        for x in raw_indices:
            idx = None
            if isinstance(x, int):
                idx = x
            elif isinstance(x, str):
                s = x.strip()
                if s.lstrip('-').isdigit():
                    idx = int(s)
                else:
                    m = re.search(r'-?\d+', s)
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
                f'[Index sanitize] layer={layer_idx}, module={module_name}, '
                f'kept={len(uniq)}, dropped={dropped}, dim={dim}'
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

    # Step 1: Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Step 2: Freeze safety neurons via gradient hooks + track for weight-decay restore
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
                safety_neurons['ffn_up'].get(layer_idx, []),
                param.shape[0], 'ffn_up', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['ffn_up'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'mlp.down_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['ffn_down'].get(layer_idx, []),
                param.shape[1], 'ffn_down', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[0]
                frozen_modules['ffn_down'] += 1
                param.register_hook(_make_zero_hook_cols(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'cols'))

        elif 'self_attn.q_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['q'].get(layer_idx, []),
                param.shape[0], 'q', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['q'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'self_attn.k_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['k'].get(layer_idx, []),
                param.shape[0], 'k', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['k'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

        elif 'self_attn.v_proj.weight' in name:
            neuron_indices = _sanitize_indices(
                safety_neurons['v'].get(layer_idx, []),
                param.shape[0], 'v', layer_idx,
            )
            if neuron_indices:
                frozen_neuron_params += len(neuron_indices) * param.shape[1]
                frozen_modules['v'] += 1
                param.register_hook(_make_zero_hook_rows(neuron_indices))
                frozen_param_specs.append((param, neuron_indices, 'rows'))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n{'='*70}")
    logger.info(f'Safety Neuron Freezing Setup Summary')
    logger.info(f"{'='*70}")
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Frozen safety neuron parameters (effective): {frozen_neuron_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    logger.info(f'Trainable ratio: {trainable_params / total_params * 100:.4f}%')
    logger.info(f'Frozen safety neuron ratio: {frozen_neuron_params / total_params * 100:.4f}%')
    logger.info(f'\nLayers with frozen safety neurons:')
    for module_type, count in frozen_modules.items():
        if count > 0:
            logger.info(f'  {module_type:12} : {count} layers')
    logger.info(f"{'='*70}\n")

    if not frozen_param_specs:
        raise ValueError(
            'freeze 대상 파라미터를 하나도 잡지 못했습니다. '
            'safety neuron 파일의 layer index 와 모델 구조가 맞는지 확인하세요.'
        )
    return frozen_param_specs


# ─────────────────────────────────────────────────────────────────────────────
# Safety Neuron Restore Callback
# ─────────────────────────────────────────────────────────────────────────────

class SafetyNeuronRestoreCallback(TrainerCallback):
    """
    Restores safety neuron weights after every optimizer step.

    AdamW's weight-decay term (λθ) is applied independently of gradient hooks,
    so safety neuron weights would otherwise drift toward 0 even when the
    gradient hook zeros out the gradient signal.  This callback saves the
    initial (frozen) values at construction time and writes them back after
    every optimizer step, guaranteeing true parameter freezing.
    """

    def __init__(self, frozen_param_specs):
        self._specs       = frozen_param_specs
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

    def max_drift(self) -> float:
        """학습 후 frozen 값이 실제로 보존됐는지 확인하기 위한 최대 절대 편차."""
        worst = 0.0
        for (param, indices, axis), frozen_val in zip(self._specs, self._frozen_vals):
            with torch.no_grad():
                current = param.data[indices, :] if axis == 'rows' else param.data[:, indices]
                worst = max(worst, (current - frozen_val).abs().max().item())
        return worst


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace upload
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_hf(output_dir: str, upload_name: str, hf_token: Optional[str], logger):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error('huggingface_hub 설치 필요: pip install huggingface_hub')
        return

    token = hf_token or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=upload_name, repo_type='model', exist_ok=True, private=False)
        logger.info(f'Repo created/found: {upload_name}')
    except Exception as exc:
        logger.warning(f'Repo creation warning: {exc}')

    ignore_patterns = ['.wandb/*', 'wandb/*', '*.log', '__pycache__/*', 'cache/*']
    logger.info(f'Uploading {output_dir} → {upload_name} ...')
    try:
        api.upload_folder(
            folder_path=output_dir,
            repo_id=upload_name,
            repo_type='model',
            ignore_patterns=ignore_patterns,
            commit_message='AG News fine-tuned model (safety neuron freeze)',
        )
        logger.info(f'Upload completed: https://huggingface.co/{upload_name}')
    except Exception as exc:
        logger.error(f'Upload failed: {exc}')
        logger.error('Model was saved locally; upload manually if needed')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    logger, log_file = setup_logging()

    logger.info(f"\n{'='*70}")
    logger.info(f'  AG News Fine-tuning with Safety Neuron Freezing')
    logger.info(f"{'='*70}\n")
    logger.info(f'Log file: {log_file}')

    if not os.path.exists(args.safety_neurons_file):
        raise FileNotFoundError(f'Safety neurons file not found: {args.safety_neurons_file}')

    raw_path   = args.model_path
    is_local   = raw_path.startswith('./') or raw_path.startswith('/') or raw_path.startswith('../')
    model_path = os.path.abspath(raw_path) if is_local else raw_path
    if is_local and not os.path.exists(model_path):
        raise FileNotFoundError(f'Model path not found: {model_path}')

    use_wandb = bool(args.report_to) and args.report_to != 'none'

    logger.info(f'Configuration:')
    logger.info(f'   ├─ SN-Tuned model: {model_path}')
    logger.info(f'   ├─ Safety neurons file: {args.safety_neurons_file}')
    logger.info(f'   ├─ AG News train: {args.agnews_train_path}')
    logger.info(f'   ├─ AG News eval: {args.agnews_eval_path or "(none)"}')
    logger.info(f'   ├─ Training samples: {args.num_train_samples or "all"}')
    logger.info(f"   ├─ Input formatting: {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f'   ├─ Batch size: {args.batch_size}')
    logger.info(f'   ├─ Gradient accumulation: {args.grad_accum}')
    logger.info(f'   ├─ Epochs: {args.epochs}')
    logger.info(f'   ├─ Learning rate: {args.learning_rate}')
    logger.info(f'   ├─ Weight decay: {args.weight_decay}')
    logger.info(f'   ├─ Optimizer: {args.optim}')
    logger.info(f'   ├─ Warmup ratio: {args.warmup_ratio}')
    logger.info(f'   ├─ Max length: {args.max_length}')
    logger.info(f'   ├─ Safety mix ratio: {args.safety_mix_ratio}')
    logger.info(f'   ├─ Dtype: bf16')
    logger.info(f'   ├─ CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")}')
    logger.info(f'   ├─ Strategy: Freeze safety neurons, train others')
    logger.info(f'   └─ Output dir: {args.output_dir}\n')

    if use_wandb:
        import wandb
        run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.output_dir))
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                'model_path':           model_path,
                'safety_neurons_file':  os.path.basename(args.safety_neurons_file),
                'strategy':             'freeze_safety_neurons',
                'learning_rate':        args.learning_rate,
                'num_epochs':           args.epochs,
                'batch_size':           args.batch_size,
                'grad_accum':           args.grad_accum,
                'effective_batch_size': args.batch_size * args.grad_accum,
                'max_length':           args.max_length,
                'weight_decay':         args.weight_decay,
                'warmup_ratio':         args.warmup_ratio,
                'lr_scheduler':         args.lr_scheduler_type,
                'dataset':              'agnews',
                'safety_mix_ratio':     args.safety_mix_ratio,
                'is_instruct':          is_instruct_model(model_path),
            },
        )

    # ── [1/5] Tokenizer ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f'  [1/5] Loading Tokenizer')
    logger.info(f"{'='*70}\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
        )
        logger.info('✓ Tokenizer loaded from local files')
    except Exception as e:
        logger.warning(f'Failed to load tokenizer with local_files_only: {e}')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        logger.info('✓ Tokenizer loaded from HuggingFace Hub')

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f'Tokenizer loaded: {type(tokenizer).__name__}, vocab={len(tokenizer)}, pad={tokenizer.pad_token}')

    # ── [2/5] Model ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f'  [2/5] Loading Model (bf16)')
    logger.info(f"{'='*70}\n")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map='auto',
            local_files_only=True, trust_remote_code=False,
        )
        logger.info('✓ Model loaded from local files')
    except Exception as e:
        logger.warning(f'Failed to load with local_files_only: {e}')
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map='auto', trust_remote_code=False,
        )
        logger.info('✓ Model loaded from HuggingFace Hub')

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model loaded: {total_params / 1e9:.2f}B params, dtype={model.dtype}')

    # ── [3/5] Safety Neuron Freezing ─────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f'  [3/5] Loading Safety Neurons and Setting up Freezing')
    logger.info(f"{'='*70}\n")

    safety_neurons     = load_safety_neurons(args.safety_neurons_file, logger)
    frozen_param_specs = setup_safety_neuron_freezing(model, safety_neurons, logger)
    restore_callback   = SafetyNeuronRestoreCallback(frozen_param_specs)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Safety neuron freezing complete: trainable={trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)')

    # ── [4/5] Load & Tokenize AG News ────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f'  [4/5] Loading AG News Dataset')
    logger.info(f"{'='*70}\n")

    train_rows = load_agnews_rows(args, logger, eval_dataset=False)
    eval_rows  = load_agnews_rows(args, logger, eval_dataset=True)

    logger.info(f"\n{'='*70}")
    logger.info(f'  [4.5/5] Preprocessing Data')
    logger.info(f"{'='*70}\n")

    train_tok = tokenize_dataset_rows(train_rows, args, tokenizer, model_path, logger)
    eval_tok  = None
    if eval_rows is not None and args.num_eval_samples > 0:
        eval_tok = tokenize_dataset_rows(eval_rows, args, tokenizer, model_path, logger)

    num_agnews = len(train_tok)
    train_tok = maybe_mix_safety(train_tok, args, tokenizer, model_path, logger, num_agnews=num_agnews)

    logger.info(f'Data preprocessed: train={len(train_tok)}' + (f', eval={len(eval_tok)}' if eval_tok else ''))

    # ── [5/5] Train ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f'  [5/5] Training')
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
        save_strategy='no',
        eval_strategy=('steps' if do_eval else 'no'),
        eval_steps=(args.eval_steps if do_eval else None),
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=(args.report_to if use_wandb else 'none'),
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
        processing_class=tokenizer,   # transformers >=4.56: `tokenizer=` 인자는 제거됨
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
        callbacks=[restore_callback],
    )

    logger.info('Starting training...')
    trainer.train()

    drift = restore_callback.max_drift()
    logger.info(f'Frozen safety-neuron max drift after training: {drift:.3e} (0 이어야 정상)')
    if drift != 0.0:
        logger.warning('Safety neuron 가중치가 완전히 보존되지 않았습니다!')

    # ── Save model ───────────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info(f'  Saving Fine-tuned Model')
    logger.info(f"{'='*70}\n")

    final_output_dir = args.output_dir
    try:
        logger.info('Step 1: Preparing model for saving...')
        gc.collect()
        torch.cuda.empty_cache()

        logger.info('Step 2: Moving model to CPU for safe serialization...')
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        logger.info('Step 3: Saving model weights...')
        logger.info(f'   ├─ Output directory: {os.path.abspath(final_output_dir)}')
        model.save_pretrained(
            final_output_dir,
            safe_serialization=True,
            max_shard_size='4GB',
            push_to_hub=False,
        )
        logger.info(f'   └─ ✅ Model weights saved successfully')

        logger.info('Step 4: Saving tokenizer...')
        tokenizer.save_pretrained(final_output_dir)
        logger.info(f'   └─ ✅ Tokenizer saved')

        logger.info('Step 5: Saving model config and generation settings...')
        model.config.save_pretrained(final_output_dir)
        if hasattr(model, 'generation_config'):
            model.generation_config.save_pretrained(final_output_dir)
        logger.info(f'   └─ ✅ Configs saved')

        logger.info('Step 6: Verifying saved model integrity...')
        required_files = ['config.json', 'tokenizer_config.json']
        missing_files  = []
        for fname in required_files:
            fpath = os.path.join(final_output_dir, fname)
            if not os.path.exists(fpath):
                missing_files.append(fname)
            else:
                logger.info(f'   ├─ {fname}: {os.path.getsize(fpath)/1024:.2f} KB ✅')
        if missing_files:
            raise FileNotFoundError(f'Missing/corrupted files: {missing_files}')

        model_files = [f for f in os.listdir(final_output_dir) if f.endswith('.safetensors')]
        if not model_files:
            raise FileNotFoundError('No safetensors files found after save!')
        logger.info(f'   ├─ ✅ Found {len(model_files)} model shard file(s)')

        total_size = 0
        for fname in sorted(os.listdir(final_output_dir)):
            fpath = os.path.join(final_output_dir, fname)
            if os.path.isfile(fpath):
                total_size += os.path.getsize(fpath)
        logger.info(f'   └─ Total size: {total_size/1e9:.2f} GB ✅')

        logger.info('\nStep 7: Final verification - attempting to load saved model...')
        test_tok   = AutoTokenizer.from_pretrained(final_output_dir)
        test_model = AutoModelForCausalLM.from_pretrained(
            final_output_dir, torch_dtype=torch.bfloat16,
            device_map='cpu', local_files_only=True,
        )
        del test_tok, test_model
        gc.collect()
        logger.info(f'   └─ ✅ Model verified successfully!')

        logger.info(f'\n✅ Fine-tuned model saved and verified: {os.path.abspath(final_output_dir)}')

    except Exception as e:
        logger.error(f'\n❌ CRITICAL ERROR during model saving: {e}')
        logger.error(traceback.format_exc())
        raise

    # ── Save training config ─────────────────────────────────────────────────
    config = {
        'base_model':             args.model_path,
        'fine_tuning_type':       'AG News Fine-tuning with Safety Neuron Freezing',
        'safety_neurons_file':    args.safety_neurons_file,
        'dataset':                'agnews',
        'agnews_train_path':      args.agnews_train_path,
        'agnews_eval_path':       args.agnews_eval_path,
        'num_train_samples':      num_agnews,
        'safety_mix_ratio':       args.safety_mix_ratio,
        'safety_data_path':       args.safety_data_path if args.safety_mix_ratio > 0 else None,
        'batch_size':             args.batch_size,
        'grad_accum':             args.grad_accum,
        'learning_rate':          args.learning_rate,
        'weight_decay':           args.weight_decay,
        'warmup_ratio':           args.warmup_ratio,
        'epochs':                 args.epochs,
        'max_length':             args.max_length,
        'max_grad_norm':          args.max_grad_norm,
        'lr_scheduler_type':      args.lr_scheduler_type,
        'optimizer':              args.optim,
        'gradient_checkpointing': args.gradient_checkpointing,
        'dtype':                  'bf16',
        'trainer_type':           'Trainer',
        'strategy':               'Freeze safety neurons, train others',
        'frozen_neuron_max_drift': drift,
    }

    config_path = os.path.join(final_output_dir, 'finetune_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f'✅ Config saved to: {config_path}')

    if args.upload_name:
        logger.info(f'\nStarting upload to Hugging Face: {args.upload_name}')
        upload_to_hf(final_output_dir, args.upload_name, args.hf_token, logger)

    logger.info(f"\n{'='*70}")
    logger.info(f'  ✅ Fine-tuning Complete!')
    logger.info(f"{'='*70}\n")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
