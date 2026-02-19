"""
GSM8K 데이터셋을 사용하여 SN-Tuned 모델(Llama-3.2-3B 기반)의 전체 파라미터(Full Parameter) 파인튜닝

Trainer + AdamW 8-bit optimizer (bitsandbytes) 사용으로 메모리 효율성 극대화

Example Usage:
python finetune_gsm8k_full_params.py \
    --model_path /home/gokms0509/Safety-Neuron/neuron_detection/sn_tuned_model_20260209_202808 \
    --output_dir ./gsm8k_sn_tune_after_gsm8k_fullft 
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import logging

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
    """베이스 모델용 프롬프트 빌딩 (finetune_gsm8k_SFT.py와 동일)"""
    system_msg = "You are a helpful assistant that solves math problems step by step. Always show your reasoning and provide the final numerical answer after ####."
    user_msg = f"Solve this problem step by step:\n\n{question.strip()}"
    prompt = f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"
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
    
    # 로컬 경로를 절대 경로로 변환 (transformers가 상대 경로를 Hub repo로 인식하는 문제 해결)
    model_path = os.path.abspath(args.model_path)
    
    # 로깅 설정
    logger, log_file = setup_logging(args.output_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  🚀 Full Parameter GSM8K Fine-tuning (SN-Tuned Model - Llama 3.2-3B Base)")
    logger.info(f"{'='*70}\n")
    logger.info(f"Log file: {log_file}")
    
    # 모델 경로 존재 확인
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   ├─ SN-Tuned model: {model_path}")
    logger.info(f"   ├─ Base model: meta-llama/Llama-3.2-3B")
    logger.info(f"   ├─ Training samples: {args.num_train_samples}")
    logger.info(f"   ├─ Batch size: {args.batch_size}")
    logger.info(f"   ├─ Gradient accumulation: {args.grad_accum}")
    logger.info(f"   ├─ Epochs: {args.epochs}")
    logger.info(f"   ├─ Learning rate: {args.learning_rate}")
    logger.info(f"   ├─ Weight decay: {args.weight_decay}")
    logger.info(f"   ├─ Optimizer: adamw_bnb_8bit (memory efficient)")
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
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   ├─ Model size: {total_params / 1e9:.2f}B parameters")
    logger.info(f"   ├─ Trainable: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"   ├─ Dtype: {model.dtype}")
    logger.info(f"   └─ Gradient checkpointing: Enabled")

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
    
    logger.info(f"✅ Data preprocessed")

    # Training
    logger.info(f"\n{'='*70}")
    logger.info(f"  [4/4] Training with Trainer + AdamW 8-bit")
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

    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"\n{'='*70}")
    logger.info(f"  Saving Fine-tuned Model")
    logger.info(f"{'='*70}\n")
    
    try:
        import gc
        
        # 1️⃣ 메모리 정리 및 최적화
        logger.info("Step 1: Preparing model for saving...")
        gc.collect()  # Python garbage collection
        torch.cuda.empty_cache()  # Clear GPU cache
        
        # 2️⃣ 모델을 CPU로 옮김 (가장 중요!)
        logger.info("Step 2: Moving model to CPU for safe serialization...")
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3️⃣ 모델 저장 (최대한 안전한 방식)
        logger.info("Step 3: Saving model weights directly (not via Trainer)...")
        logger.info(f"   ├─ Using safe_serialization=True (safetensors)")
        logger.info(f"   ├─ Output directory: {os.path.abspath(args.output_dir)}")
        
        # Trainer 거치지 않고 직접 저장 (더 안전)
        model.save_pretrained(
            args.output_dir,
            safe_serialization=True,
            max_shard_size="4GB",  # 4GB 이하로 분할
            push_to_hub=False,
        )
        logger.info(f"   └─ ✅ Model weights saved successfully")
        
        # 4️⃣ Tokenizer 저장
        logger.info("Step 4: Saving tokenizer...")
        tokenizer.save_pretrained(
            args.output_dir,
            safe_serialization=True
        )
        logger.info(f"   └─ ✅ Tokenizer saved")
        
        # 5️⃣ Config 및 생성 설정 저장
        logger.info("Step 5: Saving model config and generation settings...")
        model.config.save_pretrained(args.output_dir)
        if hasattr(model, 'generation_config'):
            model.generation_config.save_pretrained(args.output_dir)
        logger.info(f"   └─ ✅ Configs saved")
        
        # 6️⃣ 저장 검증
        logger.info("Step 6: Verifying saved model integrity...")
        required_files = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
        missing_files = []
        for fname in required_files:
            fpath = os.path.join(args.output_dir, fname)
            if not os.path.exists(fpath):
                missing_files.append(fname)
            else:
                size = os.path.getsize(fpath)
                if size == 0:
                    logger.warning(f"   ⚠️  {fname} is empty!")
                    missing_files.append(fname)
        
        if missing_files:
            raise FileNotFoundError(f"Missing/corrupted files: {missing_files}")
        
        # 모델 파일 존재 확인 (safetensors)
        model_files = [f for f in os.listdir(args.output_dir) 
                      if f.endswith('.safetensors')]
        if not model_files:
            raise FileNotFoundError("No safetensors files found after save!")
        
        logger.info(f"   ├─ ✅ Found {len(model_files)} model shard file(s)")
        
        # 7️⃣ 파일 크기 로깅 및 최종 확인
        logger.info(f"\n📦 Saved files:")
        total_size = 0
        for fname in sorted(os.listdir(args.output_dir)):
            fpath = os.path.join(args.output_dir, fname)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                total_size += size
                if size > 1e6:  # > 1MB인 파일만 표시
                    logger.info(f"   ├─ {fname:40} {size/1e9:>8.3f} GB")
                    
        logger.info(f"   └─ Total size: {total_size/1e9:.2f} GB ✅")
        
        # 8️⃣ 최종 검증: 모델 로드 가능 확인
        logger.info(f"\nStep 7: Final verification - attempting to load saved model...")
        try:
            test_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            # 메모리 절약을 위해 메타 데이터만 로드
            test_model = AutoModelForCausalLM.from_pretrained(
                args.output_dir,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            logger.info(f"   └─ ✅ Model loads successfully - integrity verified!")
            del test_model
            del test_tokenizer
            gc.collect()
        except Exception as load_err:
            logger.error(f"   ❌ CRITICAL: Saved model cannot be loaded: {load_err}")
            logger.error(f"      This means the save operation was incomplete!")
            raise RuntimeError(f"Model save verification failed: {load_err}") from load_err
        
        logger.info(f"\n✅✅✅ Fine-tuned model saved and verified successfully!")
        logger.info(f"   Output directory: {os.path.abspath(args.output_dir)}")
        logger.info(f"   Total size: {total_size/1e9:.2f} GB")
        logger.info(f"   Status: ✅ READY FOR EVALUATION")
        
    except Exception as e:
        logger.error(f"\n❌❌❌ CRITICAL ERROR during model saving: {e}")
        logger.error(f"   {type(e).__name__}: {str(e)}")
        logger.error(f"   Output directory may be incomplete: {args.output_dir}")
        logger.error(f"   Please check the directory contents before using this model!")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Save training config
    import json
    config = {
        'base_model': model_path,
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
    
    logger.info(f"✅ Config saved to: {config_path}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"  ✅ Fine-tuning Complete!")
    logger.info(f"{'='*70}\n")

if __name__ == '__main__':
    main()
