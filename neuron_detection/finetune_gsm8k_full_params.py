"""
Example Usage:
python finetune_gsm8k_full_params.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./full_finetune_llama2_7b_chat_gsm8k_full_finetune_lr3e-5

python finetune_gsm8k_full_params.py \
    --model_path kmseong/llama3.1_8b_instruct-Safety-FT-lr3e-5 \
    --output_dir ./full_finetune_llama3.1_8b_instruct_gsm8k_ssft3e-5_lr1e-5

python finetune_gsm8k_full_params.py \
    --model_path kmseong/llama2_7b-Safety-FT-lr3e-5 \
    --output_dir ./full_finetune_llama2_7b_base_gsm8k_lr5e-5 \
    --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/llama2_7b-base-gsm8k_ssft_lr5e-5
LoRA:
python finetune_gsm8k_full_params.py \
    --model_path kmseong/llama2_7b-Safety-FT-lr3e-5 \
    --output_dir ./lora_gsm8k_llama2_7b \
    --learning_rate 5e-5 --epochs 3 \
    --lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --upload_name kmseong/llama2_7b_base-gsm8k_lora_ft_lr5e-5

    
Safety 10% mixing + full parameter:
python finetune_gsm8k_full_params.py \
    --model_path kmseong/llama2_7b-chat-Safety-FT-lr5e-5 \
    --output_dir ./full_gsm8k_llama2_7b_safetymix \
    --learning_rate 5e-5 --epochs 3 \
    --safety_mix_ratio 0.05 \
    --upload_name kmseong/llama2_7b-chat-gsm8k_safelnstr_5p_lr5e-5


"""

import argparse
import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging

import wandb
import torch
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
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
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # seq
    p.add_argument("--max_length", type=int, default=1024)
    
    # memory/speed knobs
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    
    # logging/saving
    p.add_argument("--output_dir", type=str, default='./gsm8k_sn_tune_full_finetune')
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default='./cache')
    p.add_argument("--upload_name", type=str, default=None,
                    help="Optional Hugging Face repo id (e.g., username/model-name). If set, upload after training")
    p.add_argument("--hf_token", type=str, default=None,
                    help="Optional Hugging Face token for upload")

    # Safety data mixing
    p.add_argument("--safety_data_path", type=str,
                    default="/home/yonsei_jong/Safety-Neuron/neuron_detection/corpus_all/circuit_breakers_train.json",
                    help="Safety dataset JSON 경로 (circuit_breakers_train.json 형식)")
    p.add_argument("--safety_mix_ratio", type=float, default=0.0,
                    help="GSM8K 데이터 수 대비 safety 데이터 비율 (e.g. 0.1 = 10%%, 0=비활성화)")
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

def _select_first_n(ds, n: int):
    if n is None or n <= 0:
        return ds
    n = min(n, len(ds))
    return ds.select(range(n))


def is_instruct_model(model_ref: str) -> bool:
    return "instruct" in str(model_ref).lower() or "chat" in str(model_ref).lower()


def build_chat_prompt(question: str, tokenizer) -> str:
    """베이스 모델용 프롬프트 빌딩"""
    return f"Question: {question.strip()}\nAnswer:"


def tokenize_sft_example(question: str, answer_text: str, tokenizer, max_length: int, model_ref: str) -> Dict[str, List[int]]:
    """SFT 형식으로 토큰화: base는 plain prompt, instruct는 chat template 사용"""
    question = str(question).strip()
    answer_text = str(answer_text).strip()

    if is_instruct_model(model_ref):
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer_text},
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
        except Exception:
            pass

    prompt_text = build_chat_prompt(question, tokenizer)
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

def setup_logging(output_dir):
    """로깅 설정: 파일과 콘솔 모두에 출력"""
    log_dir = "./logs/safety_neuron_gsm8k"
    os.makedirs(log_dir, exist_ok=True)
    
    # 파일 이름: 현재 날짜 및 시간
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"finetune_gsm8k_{log_timestamp}.log")
    
    # 로거 설정
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def main():
    """Main fine-tuning pipeline."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    
    # 로컬 경로(./  또는 /로 시작)만 절대 경로로 변환, HuggingFace Hub ID는 그대로 유지
    raw_path = args.model_path
    is_local = raw_path.startswith("./") or raw_path.startswith("/") or raw_path.startswith("../")
    model_path = os.path.abspath(raw_path) if is_local else raw_path

    # 로깅 설정
    logger, log_file = setup_logging(args.output_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  🚀 Full Parameter GSM8K Fine-tuning (SN-Tuned Model)")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")
    
    # 로컬 경로인 경우에만 존재 여부 확인
    if is_local and not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   ├─ SN-Tuned model: {model_path}")
    logger.info(f"   ├─ Input formatting: {'chat template' if is_instruct_model(model_path) else 'base plain prompt'}")
    logger.info(f"   ├─ Training samples: {args.num_train_samples}")
    logger.info(f"   ├─ Batch size: {args.batch_size}")
    logger.info(f"   ├─ Gradient accumulation: {args.grad_accum}")
    logger.info(f"   ├─ Epochs: {args.epochs}")
    logger.info(f"   ├─ Learning rate: {args.learning_rate}")
    logger.info(f"   ├─ Weight decay: {args.weight_decay}")
    logger.info(f"   ├─ Optimizer: AdamW (torch)")
    logger.info(f"   ├─ Warmup ratio: {args.warmup_ratio}")
    logger.info(f"   ├─ Max length: {args.max_length}")
    logger.info(f"   ├─ Dtype: bf16")
    logger.info(f"   └─ Output dir: {args.output_dir}\n")

    # Load tokenizer
    logger.info(f"\n{'='*70}")
    logger.info(f"  [1/4] Loading Tokenizer")
    logger.info(f"{'='*70}\n")
    
    tokenizer = None
    
    # 시도 1: local_files_only=True (권장)
    try:
        logger.info("Attempting to load tokenizer (local files only)...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        logger.info("✓ Tokenizer loaded from local files")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer with local_files_only: {e}")
        logger.info("Attempting to load from HuggingFace Hub...")
        
        # 시도 2: Hub에서 로드 (fallback)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("✓ Tokenizer loaded from HuggingFace Hub")
        except Exception as e2:
            logger.error(f"Failed to load tokenizer: {e2}")
            raise RuntimeError(f"Could not load tokenizer from {model_path}") from e2
    
    if tokenizer is None:
        raise RuntimeError(f"Tokenizer loading failed for {model_path}")
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Tokenizer loaded successfully")
    logger.info(f"   ├─ Tokenizer type: {type(tokenizer).__name__}")
    logger.info(f"   ├─ Vocab size: {len(tokenizer)}")
    logger.info(f"   └─ Pad token: {tokenizer.pad_token}")

    # Load model with bf16
    logger.info(f"\n{'='*70}")
    logger.info(f"  [2/4] Loading Model (bf16)")
    logger.info(f"{'='*70}\n")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    
    model = None
    load_error = None
    
    # 시도 1: local_files_only=True (권장)
    try:
        logger.info("Attempting to load model (local files only)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=False,
        )
        logger.info("✓ Model loaded from local files")
    except Exception as e:
        load_error = str(e)
        logger.warning(f"Failed to load with local_files_only: {e}")
        logger.info("Attempting to load from HuggingFace Hub...")
        
        # 시도 2: Hub에서 로드 (fallback)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=False,
            )
            logger.info("✓ Model loaded from HuggingFace Hub")
        except Exception as e2:
            logger.error(f"Failed to load model from Hub: {e2}")
            logger.error(f"Original error: {load_error}")
            raise RuntimeError(f"Could not load model from {model_path}") from e2

    if model is None:
        raise RuntimeError(f"Model loading failed for {model_path}")

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
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   ├─ Model size: {total_params / 1e9:.2f}B parameters")
    logger.info(f"   ├─ Trainable: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"   ├─ Dtype: {model.dtype}")
    logger.info(f"   └─ Gradient checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")

    # Load dataset
    logger.info(f"\n{'='*70}")
    logger.info(f"  [3/4] Loading GSM8K Dataset")
    logger.info(f"{'='*70}\n")
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
    
    logger.info(f"✅ Datasets loaded")
    logger.info(f"   ├─ Train: {len(train_ds)} samples")
    if eval_ds is not None:
        logger.info(f"   └─ Eval: {len(eval_ds)} samples")

    # Preprocess data
    logger.info(f"\n{'='*70}")
    logger.info(f"  [3.5/4] Preprocessing Data")
    logger.info(f"{'='*70}\n")
    
    def preprocess(ex):
        question = ex["question"]
        answer = ex["answer"]
        return tokenize_sft_example(question, answer, tokenizer, args.max_length, model_path)

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

    # Safety data mixing
    if args.safety_mix_ratio > 0:
        safety_path = args.safety_data_path
        if not os.path.exists(safety_path):
            logger.error(f"Safety dataset not found: {safety_path}")
            raise FileNotFoundError(f"Safety dataset not found: {safety_path}")

        with open(safety_path, "r", encoding="utf-8") as f:
            safety_raw = json.load(f)

        num_safety = int(len(train_tok) * args.safety_mix_ratio)
        rng = random.Random(args.seed)
        sampled = rng.sample(safety_raw, min(num_safety, len(safety_raw)))

        def preprocess_safety(ex):
            return tokenize_sft_example(
                ex["prompt"], ex["llama3_output"], tokenizer, args.max_length, model_path
            )

        safety_hf = HFDataset.from_list(sampled)
        safety_tok = safety_hf.map(
            preprocess_safety,
            remove_columns=safety_hf.column_names,
            desc="Tokenizing safety data",
        )

        train_tok = concatenate_datasets([train_tok, safety_tok]).shuffle(seed=args.seed)
        logger.info(f"✅ Safety data mixed: {len(safety_tok)} samples (ratio={args.safety_mix_ratio})")
        logger.info(f"   Total training samples: {len(train_tok)} (GSM8K {len(train_ds)} + Safety {len(safety_tok)})")

    # Training
    logger.info(f"\n{'='*70}")
    logger.info(f"  [4/4] Training with Trainer + AdamW")
    logger.info(f"{'='*70}\n")
    
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
        # 핵심: Adam optimizer (메모리 효율적)
        optim="adamw_torch",
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    run_name = os.path.basename(os.path.normpath(args.output_dir))
    wandb.init(
        entity="gokms0509-yonsei-university",
        project="GSM8K Full Finetuning",
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
            "dataset": "gsm8k",
            "is_instruct": is_instruct_model(model_path),
            "lora": args.lora,
            "lora_r": args.lora_r if args.lora else None,
            "lora_alpha": args.lora_alpha if args.lora else None,
            "lora_dropout": args.lora_dropout if args.lora else None,
            "safety_mix_ratio": args.safety_mix_ratio,
            "safety_data_path": args.safety_data_path if args.safety_mix_ratio > 0 else None,
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
    
    # Save model
    logger.info(f"\n{'='*70}")
    logger.info(f"  Saving Fine-tuned Model")
    logger.info(f"{'='*70}\n")
    if args.lora:
        logger.info("LoRA 어댑터를 base model에 merge 중...")
        model = model.merge_and_unload()
        logger.info("✓ Merge 완료 - 전체 모델 저장 중...")
        model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"✅ Fine-tuned model saved successfully to {args.output_dir}")

    # Save training config
    config = {
        'base_model': model_path,
        'fine_tuning_type': 'LoRA Fine-tuning' if args.lora else 'Full Parameter Fine-tuning',
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
        'optimizer': 'AdamW (torch)',
        'gradient_checkpointing': args.gradient_checkpointing,
        'dtype': 'bf16',
        'trainer_type': 'Trainer',
        'safety_mix_ratio': args.safety_mix_ratio,
        'safety_data_path': args.safety_data_path if args.safety_mix_ratio > 0 else None,
    }
    
    config_path = os.path.join(args.output_dir, 'finetune_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Config saved to: {config_path}")

    if args.upload_name:
        logger.info(f"\nStarting upload to Hugging Face: {args.upload_name}")
        try:
            from upload_sn_tuned_model import upload_to_huggingface

            upload_to_huggingface(args.output_dir, args.upload_name, args.hf_token)
            logger.info(f"✅ Upload completed: https://huggingface.co/{args.upload_name}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.error("Model was saved locally; you can upload manually with upload_sn_tuned_model.py")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  ✅ Fine-tuning Complete!")
    logger.info(f"{'='*70}\n")
    wandb.finish()

if __name__ == '__main__':
    main()
