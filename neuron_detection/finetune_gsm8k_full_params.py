"""
GSM8K 데이터셋을 사용하여 SN-Tuned 모델의 전체 파라미터(Full Parameter) 파인튜닝

LoRA를 사용하지 않고 모든 파라미터를 업데이트합니다.

Example Usage:
python finetune_gsm8k_full_params.py \
    --model_path kmseong/Llama-3.2-3B-Instruct-SN-Tune_20251208_225036 \
    --num_train_samples 100 \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --max_length 256 \
    --output_dir ./gsm8k_sn_tune_full_finetune \
    --cache_dir ./cache
"""

import os
import torch
import json
import argparse
from datetime import datetime
from typing import Dict, List
import numpy as np
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer

# ==================== Configuration ====================
parser = argparse.ArgumentParser(description='Full Parameter Finetune SN-Tuned Model on GSM8K')
parser.add_argument('--model_path', type=str, 
                    default='kmseong/Llama-3.2-3B-Instruct-SN-Tune_20251208_225036',
                    help='HuggingFace model ID or local path (SN-Tuned model)')
parser.add_argument('--num_train_samples', type=int, default=100, 
                    help='Number of training samples')
parser.add_argument('--num_eval_samples', type=int, default=20, 
                    help='Number of evaluation samples')
parser.add_argument('--batch_size', type=int, default=2, 
                    help='Training batch size (reduced for full param fine-tuning)')
parser.add_argument('--eval_batch_size', type=int, default=8, 
                    help='Evaluation batch size')
parser.add_argument('--epochs', type=int, default=3, 
                    help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5, 
                    help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.05, 
                    help='Warmup ratio')
parser.add_argument('--max_length', type=int, default=256, 
                    help='Maximum sequence length')
parser.add_argument('--gradient_accumulation_steps', type=int, default=16, 
                    help='Gradient accumulation steps')
parser.add_argument('--device', type=str, default='cuda', 
                    help='Device to use (cuda/cpu)')
parser.add_argument('--cache_dir', type=str, default='./cache', 
                    help='HuggingFace cache directory')
parser.add_argument('--output_dir', type=str, 
                    default='./gsm8k_sn_tune_full_finetune',
                    help='Output directory for finetuned model')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ==================== Utility Functions ====================

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def print_section(text):
    """Print formatted section"""
    print(f"\n{text}")
    print(f"{'-'*70}")

def extract_answer(text: str) -> str:
    """Extract numerical answer from response."""
    import re
    if '####' in text:
        parts = text.split('####')
        answer_part = parts[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            return numbers[0]
    
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None

def get_expected_answer(gsm8k_answer: str) -> str:
    """Extract expected answer from GSM8K format."""
    import re
    if '####' in gsm8k_answer:
        parts = gsm8k_answer.split('####')
        answer_part = parts[-1].strip()
        numbers = re.findall(r'-?\d+\.?\d*', answer_part)
        if numbers:
            return numbers[0]
    return None

def create_prompt(question: str) -> str:
    """Create prompt for GSM8K problem."""
    return f"""Solve this math problem step by step:

{question}

Provide your final answer in the format:
[reasoning steps]
####
[final answer (just the number)]"""

def create_training_data(dataset, num_samples: int, tokenizer, max_length: int) -> Dataset:
    """
    Create training data from GSM8K dataset for SFTTrainer.
    
    Format: question + answer (in "text" field for SFTTrainer)
    """
    print_section(f"Creating training data ({num_samples} samples)")
    
    if num_samples > len(dataset):
        print(f"⚠️  Requested {num_samples} samples but only {len(dataset)} available")
        num_samples = len(dataset)
    
    training_samples = dataset.select(range(num_samples))
    
    def format_sample(sample):
        question = sample['question']
        answer = sample['answer']
        
        # Combine question and answer for training (SFTTrainer expects "text" field)
        text = f"Question: {question}\nAnswer: {answer}"
        
        return {"text": text}
    
    # Format all samples
    formatted_data = training_samples.map(
        format_sample,
        remove_columns=training_samples.column_names,
        desc="Formatting training data"
    )
    
    print(f"✅ Created {len(formatted_data)} training samples")
    return formatted_data

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from HuggingFace."""
    print_section("Loading SN-Tuned Model and Tokenizer")
    
    try:
        print(f"📂 Model ID: {model_path}")
        print(f"   Cache dir: {args.cache_dir}")
        
        # Load tokenizer
        print("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
            cache_dir=args.cache_dir
        )
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        # Load model for full parameter fine-tuning (no quantization needed)
        print("🔄 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        
        # Prepare model for fine-tuning
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        
        print("✅ Model loaded and prepared for full parameter fine-tuning")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model info:")
        print(f"   ├─ Model size: {total_params / 1e9:.2f}B parameters")
        print(f"   ├─ Dtype: {model.dtype}")
        print(f"   ├─ Device: {model.device}")
        print(f"   ├─ Trainable params: {trainable_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")
        print(f"   └─ Gradient checkpointing: Enabled")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def finetune(model, tokenizer, train_dataset):
    """
    Fine-tune the model using SFTTrainer (Full Parameter).
    
    Uses all parameters for fine-tuning without LoRA.
    """
    print_section("Fine-tuning Configuration (Full Parameters)")
    
    print(f"📊 Training settings:")
    print(f"   ├─ Batch size: {args.batch_size}")
    print(f"   ├─ Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"   ├─ Learning rate: {args.learning_rate}")
    print(f"   ├─ Epochs: {args.epochs}")
    print(f"   ├─ Warmup ratio: {args.warmup_ratio}")
    print(f"   ├─ Max length: {args.max_length}")
    print(f"   ├─ Gradient checkpointing: Enabled")
    print(f"   └─ Output dir: {args.output_dir}")
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        output_dir=args.output_dir,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        report_to="none",
        seed=42,
    )
    
    # Use SFTTrainer for full parameter fine-tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Train
    print("\n" + "="*70)
    print("  Starting Fine-tuning (Full Parameters, SFTTrainer)")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    seconds = training_time % 60
    
    print(f"\n{'='*70}")
    print(f"  Fine-tuning Complete!")
    print(f"{'='*70}")
    print(f"⏱️  Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return trainer

def save_model(model, tokenizer, trainer):
    """Save the fine-tuned model."""
    print_section("Saving Fine-tuned Model")
    
    # Get the trained model
    trained_model = trainer.model
    
    # Save model and tokenizer
    print(f"Saving fine-tuned model to {args.output_dir}...")
    trained_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"✅ Fine-tuned model saved successfully!")
    print(f"✅ Output directory: {args.output_dir}")
    
    # Save training config
    config = {
        'base_model': args.model_path,
        'fine_tuning_type': 'Full Parameter Fine-tuning',
        'dataset': 'GSM8K',
        'num_train_samples': args.num_train_samples,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'warmup_ratio': args.warmup_ratio,
        'epochs': args.epochs,
        'max_length': args.max_length,
        'use_lora': False,
        'gradient_checkpointing': True,
        'max_grad_norm': 0.3,
        'lr_scheduler': 'cosine',
        'optimizer': 'adamw_torch',
        'trainer_type': 'SFTTrainer',
        'timestamp': datetime.now().isoformat(),
    }
    
    config_path = os.path.join(args.output_dir, 'finetune_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Config saved to: {config_path}")

def evaluate_on_gsm8k(model, tokenizer, eval_dataset, num_samples: int = 10):
    """
    Evaluate fine-tuned model on GSM8K samples.
    """
    print_section(f"Evaluating on GSM8K ({num_samples} samples)")
    
    model.eval()
    correct = 0
    total = 0
    
    eval_size = min(num_samples, len(eval_dataset))
    
    for idx, sample in enumerate(tqdm(eval_dataset.select(range(eval_size)))):
        question = sample['question']
        expected_answer = get_expected_answer(sample['answer'])
        
        if expected_answer is None:
            continue
        
        # Create prompt
        prompt = create_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        predicted_answer = extract_answer(response)
        
        # Check if correct
        is_correct = False
        if predicted_answer is not None:
            try:
                is_correct = float(predicted_answer) == float(expected_answer)
            except:
                is_correct = predicted_answer == expected_answer
        
        if is_correct:
            correct += 1
        
        total += 1
        
        if (idx + 1) % 5 == 0 or idx == 0:
            status = "✅" if is_correct else "❌"
            print(f"{status} [{idx+1}/{eval_size}] Expected: {expected_answer}, Got: {predicted_answer}")
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n📊 Evaluation Results:")
    print(f"   ├─ Accuracy: {accuracy:.2f}%")
    print(f"   ├─ Correct: {correct}/{total}")
    print(f"   └─ Model path: {args.output_dir}")
    
    return accuracy

# ==================== Main ====================

def main():
    """Main fine-tuning pipeline."""
    print_header("🚀 Full Parameter GSM8K Fine-tuning Pipeline (SN-Tuned Model)")
    
    print(f"⚙️  Configuration:")
    print(f"   ├─ Base model: {args.model_path}")
    print(f"   ├─ Training samples: {args.num_train_samples}")
    print(f"   ├─ Batch size: {args.batch_size}")
    print(f"   ├─ Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   ├─ Epochs: {args.epochs}")
    print(f"   ├─ Learning rate: {args.learning_rate}")
    print(f"   ├─ Fine-tuning type: Full Parameter (no LoRA)")
    print(f"   └─ Output dir: {args.output_dir}\n")
    
    # Step 1: Load dataset
    print_section("Loading GSM8K Dataset")
    gsm8k_train = load_dataset('openai/gsm8k', 'main', split='train', cache_dir=args.cache_dir)
    gsm8k_test = load_dataset('openai/gsm8k', 'main', split='test', cache_dir=args.cache_dir)
    print(f"✅ Train dataset: {len(gsm8k_train)} samples")
    print(f"✅ Test dataset: {len(gsm8k_test)} samples")
    
    # Step 2: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    # Step 3: Create training dataset
    train_dataset = create_training_data(gsm8k_train, args.num_train_samples, tokenizer, args.max_length)
    
    # Step 4: Fine-tune
    trainer = finetune(model, tokenizer, train_dataset)
    
    # Step 5: Save model
    save_model(model, tokenizer, trainer)
    
    # Step 6: Evaluate
    print_section("Evaluating Fine-tuned Model")
    
    # Load the saved fine-tuned model for evaluation
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    finetuned_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
    
    # Evaluate on test set
    accuracy = evaluate_on_gsm8k(finetuned_model, finetuned_tokenizer, gsm8k_test, num_samples=50)
    
    print_header(f"✅ Fine-tuning Complete! Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
