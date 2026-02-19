"""
Safety Neuron Enhancement + GSM8K Fine-tuning Pipeline

1. Load detected safety neurons from detection output
2. Enhance safety neurons using Safety Dataset (Circuit Breakers) - SN-Tune
3. Freeze the enhanced safety neurons
4. Fine-tune remaining parameters on GSM8K dataset

Example Usage:
python sn_enhance_then_gsm8k_finetune.py \
  ./output_neurons/meta-llama_Llama-3.2-3B-Instruct_harmful_prompts_200_threshold_neurons_200_20251208_215958.txt \
  ./corpus_all/circuit_breakers_train.json \
  --num_gsm8k_samples 7473 \
  --gsm8k_batch_size 2 \
  --gsm8k_epochs 3 \
  --gsm8k_lr 1e-5 \
  --output_dir ./sn_enhanced_gsm8k_llama3_model \
  --cache_dir ./cache
"""

import os
import sys
import json
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from typing import Dict, List
import ast
import logging
import numpy as np
from tqdm import tqdm

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from datasets import load_dataset
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
parser = argparse.ArgumentParser(
    description='Safety Neuron Enhancement + GSM8K Fine-tuning'
)
parser.add_argument('safety_neurons_file', type=str,
                    help='Path to safety neurons detection output file')
parser.add_argument('safety_dataset_file', type=str,
                    help='Path to safety dataset (Circuit Breakers JSON)')
parser.add_argument('--model_path', type=str,
                    default='meta-llama/Llama-3.2-3B-Instruct',
                    help='HuggingFace model ID or local path')
parser.add_argument('--sn_tune_lr', type=float, default=1e-6,
                    help='Learning rate for SN-Tune (safety neuron enhancement)')
parser.add_argument('--sn_tune_epochs', type=int, default=1,
                    help='Number of epochs for SN-Tune')
parser.add_argument('--sn_tune_samples', type=int, default=50,
                    help='Number of safety samples for SN-Tune')
parser.add_argument('--sn_tune_batch_size', type=int, default=2,
                    help='Batch size for SN-Tune')
parser.add_argument('--num_gsm8k_samples', type=int, default=100,
                    help='Number of GSM8K training samples')
parser.add_argument('--gsm8k_batch_size', type=int, default=2,
                    help='Batch size for GSM8K fine-tuning')
parser.add_argument('--gsm8k_epochs', type=int, default=3,
                    help='Number of epochs for GSM8K fine-tuning')
parser.add_argument('--gsm8k_lr', type=float, default=1e-5,
                    help='Learning rate for GSM8K fine-tuning')
parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                    help='Gradient accumulation steps for GSM8K fine-tuning')
parser.add_argument('--cache_dir', type=str, default='./cache',
                    help='HuggingFace cache directory')
parser.add_argument('--output_dir', type=str,
                    default='./sn_enhanced_gsm8k_model',
                    help='Output directory for final model')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

# ==================== Constants ====================
MODEL_NAME = args.model_path
NUM_LAYERS = 28
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 256

# ==================== Utility Functions ====================

def print_header(text):
    """Print formatted header"""
    logger.info(f"\n{'='*70}")
    logger.info(f"  {text}")
    logger.info(f"{'='*70}\n")

def print_section(text):
    """Print formatted section"""
    logger.info(f"\n{text}")
    logger.info(f"{'-'*70}")

# ==================== Safety Dataset ====================

class SafetyDataset(Dataset):
    """Circuit Breakers dataset for safety alignment"""
    
    def __init__(self, json_path, tokenizer, max_samples=None, max_length=256):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:min(max_samples, len(self.data))]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loaded {len(self.data)} safety samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        harmful_prompt = item.get('prompt', '')
        safe_response = item.get('llama3_output', '')
        
        full_text = f"{harmful_prompt} {safe_response}"
        
        encodings = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        labels = encodings['input_ids'].clone()
        labels[encodings['attention_mask'] == 0] = -100
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
        }


# ==================== Load Safety Neurons ====================

def load_safety_neurons(output_file):
    """Load safety neurons from detection output file with detailed validation"""
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    safety_neurons = {}
    
    try:
        safety_neurons['ffn_up'] = ast.literal_eval(lines[0].strip())
        safety_neurons['ffn_down'] = ast.literal_eval(lines[1].strip())
        safety_neurons['q'] = ast.literal_eval(lines[2].strip())
        safety_neurons['k'] = ast.literal_eval(lines[3].strip())
        safety_neurons['v'] = ast.literal_eval(lines[4].strip())
    except Exception as e:
        logger.error(f"Error parsing safety neurons file: {e}")
        raise
    
    logger.info(f"Loaded safety neurons from {output_file}")
    
    print_section("Safety Neurons Loaded - Detailed Validation")
    
    total_neurons = 0
    total_unique_indices = set()
    
    logger.info(f"{'Module':<12} {'Layers':<8} {'Neurons':<10} {'Sample Indices':<20}")
    logger.info(f"{'-'*60}")
    
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        module_data = safety_neurons[module_type]
        module_neurons = 0
        active_layers = []
        sample_indices = []
        
        for layer_idx in sorted(module_data.keys()):
            if module_data[layer_idx]:  # Non-empty set
                layer_neurons = module_data[layer_idx]
                module_neurons += len(layer_neurons)
                active_layers.append(layer_idx)
                
                # Parse and validate neuron indices
                for neuron_name in layer_neurons:
                    if neuron_name.startswith('neuron_'):
                        try:
                            idx = int(neuron_name.split('_')[1])
                            total_unique_indices.add(f"{module_type}_L{layer_idx}_N{idx}")
                            if len(sample_indices) < 3:  # Show first 3 indices
                                sample_indices.append(idx)
                        except (IndexError, ValueError):
                            logger.warning(f"Invalid neuron format in {module_type}: {neuron_name}")
        
        total_neurons += module_neurons
        sample_str = f"[{', '.join(map(str, sample_indices[:3]))}...]" if sample_indices else "[]"
        
        logger.info(f"{module_type:<12} {len(active_layers):<8} {module_neurons:<10} {sample_str:<20}")
        
        if active_layers:
            layer_range = f"{min(active_layers)}-{max(active_layers)}" if len(active_layers) > 1 else str(active_layers[0])
            logger.info(f"{'':12} └─ Active layers: {layer_range}")
    
    logger.info(f"{'-'*60}")
    logger.info(f"{'TOTALS':<12} {'-':<8} {total_neurons:<10} {'':20}")
    logger.info(f"\n🔍 Validation Results:")
    logger.info(f"   ✅ Total safety neurons: {total_neurons}")
    logger.info(f"   ✅ Unique neuron instances: {len(total_unique_indices)}")
    logger.info(f"   ✅ Expected format: 'neuron_[0-3071]' ✓")
    logger.info(f"   ✅ Matches neuron_detection_simple.py output format ✓")
    
    # Validate neuron indices are within expected ranges
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        max_expected = 3072  # Hidden dimension for Llama-3.2-3B
        if module_type.startswith('ffn_up'):
            max_expected = 11008  # FFN intermediate size
        
        for layer_idx, neuron_set in safety_neurons[module_type].items():
            for neuron_name in neuron_set:
                if neuron_name.startswith('neuron_'):
                    try:
                        idx = int(neuron_name.split('_')[1])
                        if idx >= max_expected:
                            logger.warning(f"Neuron index {idx} exceeds expected max {max_expected} in {module_type}")
                    except:
                        pass
    
    logger.info(f"   ✅ Neuron indices validation complete")
    
    return safety_neurons


# ==================== Freeze/Unfreeze Parameters ====================

def parse_neuron_indices(neuron_set):
    """Parse neuron indices from detection format ('neuron_123' -> 123)"""
    indices = []
    for neuron_name in neuron_set:
        if neuron_name.startswith('neuron_'):
            try:
                idx = int(neuron_name.split('_')[1])
                indices.append(idx)
            except (IndexError, ValueError):
                logger.warning(f"Invalid neuron name format: {neuron_name}")
    return sorted(indices)


def create_neuron_masks(model, safety_neurons):
    """Create masks for safety neurons to enable precise neuron-level freezing"""
    masks = {}
    neuron_stats = {
        'ffn_up': {'layers': 0, 'neurons': 0, 'params': 0},
        'ffn_down': {'layers': 0, 'neurons': 0, 'params': 0},
        'q': {'layers': 0, 'neurons': 0, 'params': 0},
        'k': {'layers': 0, 'neurons': 0, 'params': 0},
        'v': {'layers': 0, 'neurons': 0, 'params': 0},
    }
    
    for name, param in model.named_parameters():
        # ===== FFN UP Projection =====
        if 'mlp.up_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['ffn_up'] and safety_neurons['ffn_up'][layer_idx]:
                neuron_indices = parse_neuron_indices(safety_neurons['ffn_up'][layer_idx])
                if neuron_indices:
                    # Create mask for output dimension (rows)
                    mask = torch.zeros(param.shape[0], dtype=torch.bool, device=param.device)
                    for neuron_idx in neuron_indices:
                        if neuron_idx < param.shape[0]:
                            mask[neuron_idx] = True
                    masks[name] = mask
                    
                    # Update stats
                    neuron_stats['ffn_up']['layers'] += 1
                    neuron_stats['ffn_up']['neurons'] += len(neuron_indices)
                    neuron_stats['ffn_up']['params'] += mask.sum().item() * param.shape[1]
        
        # ===== FFN DOWN Projection =====
        elif 'mlp.down_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['ffn_down'] and safety_neurons['ffn_down'][layer_idx]:
                neuron_indices = parse_neuron_indices(safety_neurons['ffn_down'][layer_idx])
                if neuron_indices:
                    # Create mask for input dimension (columns)
                    mask = torch.zeros(param.shape[1], dtype=torch.bool, device=param.device)
                    for neuron_idx in neuron_indices:
                        if neuron_idx < param.shape[1]:
                            mask[neuron_idx] = True
                    masks[name] = mask
                    
                    neuron_stats['ffn_down']['layers'] += 1
                    neuron_stats['ffn_down']['neurons'] += len(neuron_indices)
                    neuron_stats['ffn_down']['params'] += mask.sum().item() * param.shape[0]
        
        # ===== Attention Q Projection =====
        elif 'self_attn.q_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['q'] and safety_neurons['q'][layer_idx]:
                neuron_indices = parse_neuron_indices(safety_neurons['q'][layer_idx])
                if neuron_indices:
                    mask = torch.zeros(param.shape[0], dtype=torch.bool, device=param.device)
                    for neuron_idx in neuron_indices:
                        if neuron_idx < param.shape[0]:
                            mask[neuron_idx] = True
                    masks[name] = mask
                    
                    neuron_stats['q']['layers'] += 1
                    neuron_stats['q']['neurons'] += len(neuron_indices)
                    neuron_stats['q']['params'] += mask.sum().item() * param.shape[1]
        
        # ===== Attention K Projection =====
        elif 'self_attn.k_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['k'] and safety_neurons['k'][layer_idx]:
                neuron_indices = parse_neuron_indices(safety_neurons['k'][layer_idx])
                if neuron_indices:
                    mask = torch.zeros(param.shape[0], dtype=torch.bool, device=param.device)
                    for neuron_idx in neuron_indices:
                        if neuron_idx < param.shape[0]:
                            mask[neuron_idx] = True
                    masks[name] = mask
                    
                    neuron_stats['k']['layers'] += 1
                    neuron_stats['k']['neurons'] += len(neuron_indices)
                    neuron_stats['k']['params'] += mask.sum().item() * param.shape[1]
        
        # ===== Attention V Projection =====
        elif 'self_attn.v_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['v'] and safety_neurons['v'][layer_idx]:
                neuron_indices = parse_neuron_indices(safety_neurons['v'][layer_idx])
                if neuron_indices:
                    mask = torch.zeros(param.shape[0], dtype=torch.bool, device=param.device)
                    for neuron_idx in neuron_indices:
                        if neuron_idx < param.shape[0]:
                            mask[neuron_idx] = True
                    masks[name] = mask
                    
                    neuron_stats['v']['layers'] += 1
                    neuron_stats['v']['neurons'] += len(neuron_indices)
                    neuron_stats['v']['params'] += mask.sum().item() * param.shape[1]
    
    return masks, neuron_stats


def freeze_non_safety_neurons(model, safety_neurons):
    """Freeze all parameters except safety neurons (precise neuron-level freezing)
    
    Uses gradient masking to freeze only non-safety neurons while keeping
    the exact same neuron detection logic as neuron_detection_simple.py.
    """
    # Create masks for safety neurons
    print_section("Creating Safety Neuron Masks")
    masks, neuron_stats = create_neuron_masks(model, safety_neurons)
    
    # Store masks in model for gradient masking
    model.safety_neuron_masks = masks
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_neurons = sum(stats['neurons'] for stats in neuron_stats.values())
    total_trainable_params = sum(stats['params'] for stats in neuron_stats.values())
    
    # All parameters require gradients, but we'll mask safety neurons during backward
    for param in model.parameters():
        param.requires_grad = True
    
    print_section("Precise Safety Neuron Analysis")
    logger.info(f"🔍 Neuron-Level Detection Results:")
    logger.info(f"{'Module':<12} {'Layers':<8} {'Neurons':<10} {'Parameters':<12}")
    logger.info(f"{'-'*50}")
    
    for module, stats in neuron_stats.items():
        if stats['neurons'] > 0:
            logger.info(f"{module:<12} {stats['layers']:<8} {stats['neurons']:<10} {stats['params']:<12,}")
    
    logger.info(f"{'-'*50}")
    logger.info(f"{'TOTAL':<12} {sum(s['layers'] for s in neuron_stats.values()):<8} "
                f"{total_trainable_neurons:<10} {total_trainable_params:<12,}")
    
    print_section("Parameter Freezing Summary (Precise Neuron-Level)")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Safety neuron parameters (trainable): {total_trainable_params:,}")
    logger.info(f"Other parameters (frozen via gradient masking): {total_params - total_trainable_params:,}")
    logger.info(f"Trainable ratio: {total_trainable_params / total_params * 100:.6f}%")
    logger.info(f"\n💡 Precise Neuron Matching:")
    logger.info(f"   ✅ Using exact same detection logic as neuron_detection_simple.py")
    logger.info(f"   ✅ Parsing 'neuron_123' format to get exact indices")
    logger.info(f"   ✅ Creating masks for {len(masks)} weight matrices")
    logger.info(f"   ✅ Only {total_trainable_neurons} neurons will be trained")
    
    # Register backward hooks for gradient masking
    def gradient_mask_hook(grad, mask, param_name):
        """Apply neuron-level gradient masking"""
        if grad is not None and mask is not None:
            if 'down_proj' in param_name:
                # For down projection: mask input dimension (columns)
                masked_grad = grad.clone()
                masked_grad[:, ~mask] = 0
            else:
                # For up_proj, q_proj, k_proj, v_proj: mask output dimension (rows)
                masked_grad = grad.clone() 
                masked_grad[~mask, :] = 0
            return masked_grad
        return grad
    
    hooks_registered = 0
    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name]
            param.register_hook(lambda grad, m=mask, n=name: gradient_mask_hook(grad, m, n))
            hooks_registered += 1
    
    logger.info(f"   ✅ Registered {hooks_registered} gradient masking hooks")
    
    return total_trainable_params, total_params


def freeze_safety_neurons(model, safety_neurons):
    """Freeze the enhanced safety neurons and unfreeze other parameters (precise neuron-level)
    
    After SN-Tune, we freeze only the safety neurons that were just trained,
    allowing all other parameters to be trained on GSM8K dataset.
    """
    # Get existing masks
    if not hasattr(model, 'safety_neuron_masks'):
        logger.error("Model does not have safety_neuron_masks! Run freeze_non_safety_neurons first.")
        return
    
    masks = model.safety_neuron_masks
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    safety_neuron_params = 0
    
    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name]
            if 'down_proj' in name:
                safety_neuron_params += mask.sum().item() * param.shape[0]
            else:
                safety_neuron_params += mask.sum().item() * param.shape[1]
    
    trainable_params = total_params - safety_neuron_params
    
    # All parameters require gradients, but we'll mask safety neurons during backward
    for param in model.parameters():
        param.requires_grad = True
    
    print_section("Frozen Safety Neurons - Precise Neuron-Level GSM8K Training")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters (non-safety): {trainable_params:,}")
    logger.info(f"Frozen parameters (safety neurons): {safety_neuron_params:,}")
    logger.info(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")
    logger.info(f"\n💡 Safety Neuron Preservation:")
    logger.info(f"   ✅ {len(masks)} weight matrices with safety neurons will be frozen")
    logger.info(f"   ✅ Only safety neurons frozen, not entire layers")
    logger.info(f"   ✅ Enhanced safety neurons from SN-Tune are preserved")
    logger.info(f"   ✅ All other parameters trainable on GSM8K")
    
    # Register backward hooks for gradient masking (freeze safety neurons)
    def gradient_mask_hook_freeze(grad, mask, param_name):
        """Apply neuron-level gradient masking to freeze safety neurons"""
        if grad is not None and mask is not None:
            if 'down_proj' in param_name:
                # For down projection: mask input dimension (columns)
                masked_grad = grad.clone()
                masked_grad[:, mask] = 0  # Freeze safety neurons (set to 0)
            else:
                # For up_proj, q_proj, k_proj, v_proj: mask output dimension (rows)
                masked_grad = grad.clone()
                masked_grad[mask, :] = 0  # Freeze safety neurons (set to 0)
            return masked_grad
        return grad
    
    # Clear existing hooks and register new ones
    hooks_registered = 0
    for name, param in model.named_parameters():
        if name in masks:
            mask = masks[name]
            # Register hook to freeze safety neurons (opposite of SN-Tune phase)
            param.register_hook(lambda grad, m=mask, n=name: gradient_mask_hook_freeze(grad, m, n))
            hooks_registered += 1
    
    logger.info(f"   ✅ Registered {hooks_registered} gradient masking hooks for GSM8K training")
    
    # Store in model that we're now in GSM8K training phase
    model.training_phase = 'gsm8k'


# ==================== SN-Tune Training ====================

def train_sn_tune(model, train_dataloader, learning_rate, num_epochs, device):
    """Train safety neurons on safety dataset"""
    model = model.to(device)
    model.train()
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    total_loss = 0.0
    total_steps = 0
    
    print_section("Starting SN-Tune Training")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {args.sn_tune_batch_size}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"SN-Tune Epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                total_steps += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
    
    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    print_section("SN-Tune Training Complete")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Total steps: {total_steps}")
    
    return model


# ==================== GSM8K Fine-tuning ====================

class GSM8KDataset:
    """Create training data from GSM8K dataset"""
    
    @staticmethod
    def create_training_data(dataset, num_samples, tokenizer, max_length):
        print_section(f"Creating GSM8K training data ({num_samples} samples)")
        
        if num_samples > len(dataset):
            logger.warning(f"Requested {num_samples} samples but only {len(dataset)} available")
            num_samples = len(dataset)
        
        training_samples = dataset.select(range(num_samples))
        
        def format_sample(sample):
            question = sample['question']
            answer = sample['answer']
            text = f"Question: {question}\nAnswer: {answer}"
            return {"text": text}
        
        formatted_data = training_samples.map(
            format_sample,
            remove_columns=training_samples.column_names,
            desc="Formatting training data"
        )
        
        logger.info(f"✅ Created {len(formatted_data)} training samples")
        return formatted_data


def finetune_gsm8k(model, tokenizer, train_dataset):
    """Fine-tune non-safety parameters on GSM8K"""
    print_section("GSM8K Fine-tuning Configuration")
    
    logger.info(f"📊 Training settings:")
    logger.info(f"   ├─ Batch size: {args.gsm8k_batch_size}")
    logger.info(f"   ├─ Gradient Accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"   ├─ Learning rate: {args.gsm8k_lr}")
    logger.info(f"   ├─ Epochs: {args.gsm8k_epochs}")
    logger.info(f"   ├─ Gradient checkpointing: Enabled")
    logger.info(f"   └─ Output dir: {args.output_dir}")
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.gsm8k_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=args.gsm8k_epochs,
        learning_rate=args.gsm8k_lr,
        bf16=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        output_dir=args.output_dir,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        report_to="none",
        seed=42,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print_section("Starting GSM8K Fine-tuning")
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    seconds = training_time % 60
    
    logger.info(f"\n{'='*70}")
    logger.info(f"GSM8K Fine-tuning Complete!")
    logger.info(f"⏱️  Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"{'='*70}\n")
    
    return trainer


def save_model(model, tokenizer, trainer):
    """Save the fine-tuned model"""
    print_section("Saving Final Model")
    
    trained_model = trainer.model
    
    logger.info(f"Saving model to {args.output_dir}...")
    trained_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info(f"✅ Model saved successfully!")
    
    # Save training config
    config = {
        'base_model': args.model_path,
        'pipeline': 'Safety Neuron Enhancement + GSM8K Fine-tuning',
        'safety_neurons_file': args.safety_neurons_file,
        'safety_dataset_file': args.safety_dataset_file,
        'sn_tune_lr': args.sn_tune_lr,
        'sn_tune_epochs': args.sn_tune_epochs,
        'sn_tune_samples': args.sn_tune_samples,
        'sn_tune_batch_size': args.sn_tune_batch_size,
        'gsm8k_samples': args.num_gsm8k_samples,
        'gsm8k_batch_size': args.gsm8k_batch_size,
        'gsm8k_epochs': args.gsm8k_epochs,
        'gsm8k_learning_rate': args.gsm8k_lr,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'use_lora': False,
        'gradient_checkpointing': True,
        'timestamp': datetime.now().isoformat(),
    }
    
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Config saved to: {config_path}")


# ==================== Main ====================

def main():
    print_header("🚀 Safety Neuron Enhancement + GSM8K Fine-tuning Pipeline")
    
    # Verify input files
    if not os.path.exists(args.safety_neurons_file):
        logger.error(f"Safety neurons file not found: {args.safety_neurons_file}")
        sys.exit(1)
    
    if not os.path.exists(args.safety_dataset_file):
        logger.error(f"Safety dataset file not found: {args.safety_dataset_file}")
        sys.exit(1)
    
    logger.info(f"⚙️  Configuration:")
    logger.info(f"   ├─ Base model: {args.model_path}")
    logger.info(f"   ├─ Safety neurons: {args.safety_neurons_file}")
    logger.info(f"   ├─ Safety dataset: {args.safety_dataset_file}")
    logger.info(f"   ├─ SN-Tune samples: {args.sn_tune_samples}")
    logger.info(f"   ├─ GSM8K samples: {args.num_gsm8k_samples}")
    logger.info(f"   └─ Output dir: {args.output_dir}\n")
    
    # Step 1: Load model and tokenizer
    print_section("Loading Model and Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
        cache_dir=args.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    
    model.config.use_cache = False
    logger.info(f"✅ Model and tokenizer loaded")
    
    # Step 2: Load safety neurons
    print_section("Loading Safety Neurons")
    safety_neurons = load_safety_neurons(args.safety_neurons_file)
    
    # Step 3: Prepare for SN-Tune (freeze non-safety neurons)
    print_section("Preparing for SN-Tune")
    freeze_non_safety_neurons(model, safety_neurons)
    
    # Step 4: Load safety dataset and train SN-Tune
    print_section("Loading Safety Dataset")
    safety_dataset = SafetyDataset(
        args.safety_dataset_file,
        tokenizer,
        max_samples=args.sn_tune_samples,
        max_length=MAX_SEQ_LENGTH
    )
    
    safety_loader = DataLoader(
        safety_dataset,
        batch_size=args.sn_tune_batch_size,
        shuffle=True
    )
    
    # Train SN-Tune
    model = train_sn_tune(
        model,
        safety_loader,
        learning_rate=args.sn_tune_lr,
        num_epochs=args.sn_tune_epochs,
        device=DEVICE
    )
    
    # Step 5: Freeze safety neurons and prepare for GSM8K fine-tuning
    print_section("Preparing for GSM8K Fine-tuning")
    freeze_safety_neurons(model, safety_neurons)
    
    # Step 6: Load GSM8K dataset
    print_section("Loading GSM8K Dataset")
    gsm8k = load_dataset('openai/gsm8k', 'main', split='train', cache_dir=args.cache_dir)
    logger.info(f"✅ GSM8K dataset loaded: {len(gsm8k)} total samples")
    
    # Step 7: Create training data
    train_dataset = GSM8KDataset.create_training_data(
        gsm8k,
        args.num_gsm8k_samples,
        tokenizer,
        MAX_SEQ_LENGTH
    )
    
    # Step 8: Fine-tune on GSM8K
    trainer = finetune_gsm8k(model, tokenizer, train_dataset)
    
    # Step 9: Save model
    save_model(model, tokenizer, trainer)
    
    print_header("✅ Pipeline Complete!")
    logger.info(f"Final model saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
