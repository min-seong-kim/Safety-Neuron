"""
GSM8K 데이터셋을 사용하여 SN-Tuned 모델의 전체 파라미터(Full Parameter) 파인튜닝

Trainer + AdamW 8-bit optimizer (bitsandbytes) 사용으로 메모리 효율성 극대화

Example Usage:
python finetune_gsm8k_full_params.py \
    --model_path ./sn_tuned_model_20260205_010725 \
    --output_dir ./gsm8k_sn_tune_after_gsm8k_fullft 
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description='Full Parameter Finetune SN-Tuned Model on GSM8K')
    
    # model
    p.add_argument('--model_path', type=str, 
                    default=None,
                    required=True,
                    help='HuggingFace model ID or local path (SN-Tuned model)')
    
    # data
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_subset", type=str, default="main")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="test")
    p.add_argument("--num_train_samples", type=int, default=7473)
    p.add_argument("--num_eval_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    
    # training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # seq
    p.add_argument("--max_length", type=int, default=512)
    
    # memory/speed knobs
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    # logging/saving
    p.add_argument("--output_dir", type=str, default='./gsm8k_sn_tune_full_finetune')
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default='./cache')
    
    return p.parse_args()

def _select_first_n(ds, n: int):
    if n is None or n <= 0:
        return ds
    n = min(n, len(ds))
    return ds.select(range(n))


def build_chat_prompt(question: str, tokenizer) -> str:
    """Llama 3.2 Instruct chat template 사용"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves math problems step by step. Always show your reasoning and provide the final numerical answer after ####."
        },
        {
            "role": "user",
            "content": f"Solve this problem step by step:\n\n{question.strip()}"
        }
    ]
    # chat template 적용 (답변 부분 제외)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def tokenize_sft_example(prompt_text: str, answer_text: str, tokenizer, max_length: int) -> Dict[str, List[int]]:
    """SFT 형식으로 토큰화: 프롬프트는 attention, 답변만 loss 계산"""
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    # Ensure room for answer
    remain = max(1, max_length - len(prompt_ids))
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remain,
    )["input_ids"]

    # Add EOS if possible and fits
    if tokenizer.eos_token_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != tokenizer.eos_token_id):
        if len(prompt_ids) + len(answer_ids) < max_length:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    # Loss only on answer tokens (프롬프트는 -100으로 마스킹)
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForCausalLMWithPadding:
    """패딩된 배치 생성"""
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            l = len(f["input_ids"])
            pad_len = max_len - l
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def main():
    """Main fine-tuning pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"  🚀 Full Parameter GSM8K Fine-tuning (SN-Tuned Model)")
    print(f"{'='*70}\n")
    print(f"⚙️  Configuration:")
    print(f"   ├─ Base model: {args.model_path}")
    print(f"   ├─ Training samples: {args.num_train_samples}")
    print(f"   ├─ Batch size: {args.batch_size}")
    print(f"   ├─ Gradient accumulation: {args.grad_accum}")
    print(f"   ├─ Epochs: {args.epochs}")
    print(f"   ├─ Learning rate: {args.learning_rate}")
    print(f"   ├─ Weight decay: {args.weight_decay}")
    print(f"   ├─ Optimizer: adamw_bnb_8bit (memory efficient)")
    print(f"   ├─ Warmup ratio: {args.warmup_ratio}")
    print(f"   ├─ Max length: {args.max_length}")
    print(f"   ├─ Dtype: bf16")
    print(f"   └─ Output dir: {args.output_dir}\n")

    # Load tokenizer
    print(f"\n{'='*70}")
    print(f"  [1/4] Loading Tokenizer")
    print(f"{'='*70}\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        use_fast=True,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        fix_mistral_regex=True  # Fix regex pattern warning
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded")

    # Load model with bf16
    print(f"\n{'='*70}")
    print(f"  [2/4] Loading Model (bf16)")
    print(f"{'='*70}\n")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        dtype=dtype,  # Use dtype instead of torch_dtype (torch_dtype is deprecated)
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded")
    print(f"   ├─ Model size: {total_params / 1e9:.2f}B parameters")
    print(f"   ├─ Trainable: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")
    print(f"   ├─ Dtype: {model.dtype}")
    print(f"   └─ Gradient checkpointing: Enabled")

    # Load dataset
    print(f"\n{'='*70}")
    print(f"  [3/4] Loading GSM8K Dataset")
    print(f"{'='*70}\n")
    train_ds = load_dataset(
        args.dataset_name, 
        args.dataset_subset, 
        split=args.train_split,
        cache_dir=args.cache_dir
    )
    train_ds = _select_first_n(train_ds, args.num_train_samples)

    eval_ds = None
    if args.num_eval_samples and args.num_eval_samples > 0:
        eval_ds = load_dataset(
            args.dataset_name, 
            args.dataset_subset, 
            split=args.eval_split,
            cache_dir=args.cache_dir
        )
        eval_ds = _select_first_n(eval_ds, args.num_eval_samples)
    
    print(f"✅ Datasets loaded")
    print(f"   ├─ Train: {len(train_ds)} samples")
    if eval_ds is not None:
        print(f"   └─ Eval: {len(eval_ds)} samples")

    # Preprocess data
    print(f"\n{'='*70}")
    print(f"  [3.5/4] Preprocessing Data")
    print(f"{'='*70}\n")
    
    def preprocess(ex):
        prompt = build_chat_prompt(ex["question"], tokenizer)
        answer = ex["answer"]
        return tokenize_sft_example(prompt, answer, tokenizer, args.max_length)

    train_tok = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        num_proc=max(1, args.num_workers),
        desc="Tokenizing train",
    )

    eval_tok = None
    if eval_ds is not None:
        eval_tok = eval_ds.map(
            preprocess,
            remove_columns=eval_ds.column_names,
            num_proc=max(1, args.num_workers),
            desc="Tokenizing eval",
        )
    
    print(f"✅ Data preprocessed")

    # Training
    print(f"\n{'='*70}")
    print(f"  [4/4] Training with Trainer + AdamW 8-bit")
    print(f"{'='*70}\n")
    
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
        # 핵심: AdamW 8-bit optimizer (메모리 효율적)
        optim="adamw_bnb_8bit",
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
    )

    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"\n{'='*70}")
    print(f"  Saving Fine-tuned Model")
    print(f"{'='*70}\n")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Fine-tuned model saved!")
    print(f"   └─ Output directory: {args.output_dir}")
    
    # Save training config
    import json
    config = {
        'base_model': args.model_path,
        'fine_tuning_type': 'Full Parameter Fine-tuning',
        'dataset': 'GSM8K',
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
        'optimizer': 'adamw_bnb_8bit',
        'gradient_checkpointing': True,
        'dtype': 'bf16',
        'trainer_type': 'Trainer',
    }
    
    config_path = os.path.join(args.output_dir, 'finetune_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config saved to: {config_path}")
    
    print(f"\n{'='*70}")
    print(f"  ✅ Fine-tuning Complete!")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
