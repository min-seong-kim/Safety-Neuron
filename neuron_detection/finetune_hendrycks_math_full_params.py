"""
Hendrycks MATH 데이터셋을 사용하여 SN-Tuned 모델의 전체 파라미터(Full Parameter) 파인튜닝

Instruct 모델 기준:
- tokenizer.apply_chat_template 사용
- user prompt는 labels=-100으로 마스킹
- assistant response만 loss 계산

Example Usage:
python finetune_hendrycks_math_full_params.py \
    --model_path kmseong/llama3.1_8b_instruct-Safety-FT-lr3e-5  \
    --output_dir ./full_finetune_MATH_8b_instruct-lr1e-5 \
    --upload_name kmseong/llama3.1_8b_instruct_MATH_full_ft-lr1e-5

    
python finetune_hendrycks_math_full_params.py \
    --model_path kmseong/llama3.1_8b_instruct-Safety-FT-lr3e-5 \
    --output_dir ./lora_math_llama3.1_8b \
    --lora --lora_r 16 --lora_alpha 32 \
    --upload_name kmseong/llama3.1_8b_instruct_MATH_lora_ft-lr1e-5
"""

import argparse
import os
import random
import re
import json
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime
import logging

import wandb
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    _peft_available = True
except ImportError:
    _peft_available = False

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    p = argparse.ArgumentParser(description="Full Parameter Finetune SN-Tuned Model on Hendrycks MATH")

    p.add_argument("--model_path", type=str, required=True, help="HuggingFace model ID or local path")

    p.add_argument("--math_dataset_source", type=str, default="official", choices=["official", "flat_competition_math"])
    p.add_argument("--math_official_dataset_path", type=str, default="EleutherAI/hendrycks_math")
    p.add_argument("--math_flat_dataset_path", type=str, default="qwedsacf/competition_math")
    p.add_argument("--math_subjects", type=str, default="all")
    p.add_argument("--math_levels", type=str, default="all")
    p.add_argument("--num_train_samples", type=int, default=0)
    p.add_argument("--num_eval_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--math_train_on_mixed_formats", action="store_true", default=False)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    p.add_argument("--output_dir", type=str, default="./math_sn_tune_full_finetune")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--upload_name", type=str, default=None,
                   help="Optional Hugging Face repo id (e.g., username/model-name). If set, upload after training")
    p.add_argument("--hf_token", type=str, default=None,
                   help="Optional Hugging Face token for upload")

    # LoRA 옵션
    p.add_argument("--lora", action="store_true",
                   help="LoRA를 사용하여 학습 (peft 필요)")
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank (default: 16)")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (default: 32)")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout (default: 0.05)")
    p.add_argument("--lora_target_modules", type=str, nargs='+',
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   help="LoRA를 적용할 모듈 이름 목록")

    return p.parse_args()


def is_instruct_model(model_ref: str) -> bool:
    model_ref = model_ref.lower()
    return any(tag in model_ref for tag in ('instruct', 'chat'))


def normalize_csv_arg(raw_value: str) -> str:
    value = str(raw_value).strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        value = value[1:-1].strip()
    return value


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
    return text[idx:right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if s is None:
        raise ValueError("remove_boxed received None")
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left):]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    left = "\\fbox{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s


def extract_final_answer_from_solution(solution: str) -> str:
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        raise ValueError(f"Could not find final boxed answer in solution: {solution[:300]!r}")
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


def tokenize_math_sft_example(problem: str, target_text: str, tokenizer, max_length: int, model_ref: str) -> Dict[str, List[int]]:
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

    if tokenizer.eos_token_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


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


def setup_logging():
    log_dir = "./logs/safety_neuron_math"
    os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_math_{log_timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path

    logger, log_file = setup_logging()

    logger.info(f"\n{'=' * 70}")
    logger.info("  🚀 Full Parameter Hendrycks MATH Fine-tuning (SN-Tuned Model)")
    logger.info(f"{'=' * 70}\n")
    logger.info(f"Log file: {log_file}")
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   ├─ Model: {model_path}")
    logger.info(f"   ├─ Input formatting: {'chat template' if is_instruct_model(model_path) else 'Question/Answer plain text'}")
    logger.info(f"   ├─ Subjects: {args.math_subjects}")
    logger.info(f"   ├─ Levels: {args.math_levels}")
    logger.info(f"   ├─ Train samples: {args.num_train_samples}")
    logger.info(f"   ├─ Batch size: {args.batch_size}")
    logger.info(f"   ├─ Grad accum: {args.grad_accum}")
    logger.info(f"   ├─ Epochs: {args.epochs}")
    logger.info(f"   ├─ LR: {args.learning_rate}")
    logger.info(f"   └─ Output dir: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA 적용
    if args.lora:
        if not _peft_available:
            logger.error("peft 라이브러리가 설치되지 않았습니다. 'pip install peft'를 실행하세요.")
            raise ImportError("peft is required for LoRA training")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"✓ LoRA 적용: r={args.lora_r}, alpha={args.lora_alpha}, "
                    f"dropout={args.lora_dropout}, target_modules={args.lora_target_modules}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

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
    if subjects_arg.lower() == "all":
        subjects = list(subject_to_config.keys())
    else:
        subjects = [normalize_csv_arg(s) for s in subjects_arg.split(",") if normalize_csv_arg(s)]

    if args.math_dataset_source == "official":
        datasets_per_subject = []
        for subject in subjects:
            config_name = subject_to_config[subject]
            ds = load_dataset(args.math_official_dataset_path, config_name, split="train", cache_dir=args.cache_dir)
            ds = ds.map(lambda ex, subject=subject: {"type": subject})
            datasets_per_subject.append(ds)
        train_ds = concatenate_datasets(datasets_per_subject)
    else:
        train_ds = load_dataset(args.math_flat_dataset_path, split="train", cache_dir=args.cache_dir)
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
        report_to="wandb",
        remove_unused_columns=False,
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    wandb.init(
        entity="gokms0509-yonsei-university",
        project="Hendrycks MATH Full Finetuning",
        name=run_name,
        config={
            "model_path": model_path,
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
            "lora": args.lora,
            "lora_r": args.lora_r if args.lora else None,
            "lora_alpha": args.lora_alpha if args.lora else None,
            "lora_dropout": args.lora_dropout if args.lora else None,
        },
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    if args.lora:
        logger.info("LoRA 어댑터를 base model에 merge 중...")
        model = model.merge_and_unload()
        logger.info("✓ Merge 완료 - 전체 모델 저장 중...")
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config = {
        "base_model": model_path,
        "fine_tuning_type": "LoRA Fine-tuning" if args.lora else "Full Parameter Fine-tuning",
        "dataset": "Hendrycks MATH",
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
        "optimizer": "AdamW (torch)",
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": "bf16" if args.bf16 else ("fp16" if args.fp16 else "fp32"),
        "input_formatting": "chat template" if is_instruct_model(model_path) else "Question/Answer plain text",
        "prompt_masking": "assistant-only loss",
    }

    config_path = os.path.join(args.output_dir, "finetune_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"✅ Fine-tuned model saved to {args.output_dir}")
    logger.info(f"✅ Config saved to {config_path}")

    if args.upload_name:
        logger.info(f"\nStarting upload to Hugging Face: {args.upload_name}")
        try:
            from upload_sn_tuned_model import upload_to_huggingface

            upload_to_huggingface(args.output_dir, args.upload_name, args.hf_token)
            logger.info(f"✅ Upload completed: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.error("Model was saved locally; you can upload manually with upload_sn_tuned_model.py")

    wandb.finish()


if __name__ == "__main__":
    main()
