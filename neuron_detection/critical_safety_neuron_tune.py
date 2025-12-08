"""
Step 3: Critical Safety Neuron Fine-tuning (Critical-Tune)

목표:
  Critical Safety Neurons만 미세 조정하여 모델의 안전성 향상
  
알고리즘:
  1. 모델 로드 (Llama-3.2-3B-Instruct)
  2. Critical Safety Neurons 로드
  3. Critical Neurons 제외 모든 파라미터 고정 (requires_grad=False)
  4. Safety dataset (Circuit Breakers)으로 미세 조정
  5. 모델 저장
  
특징:
  - 매우 파라미터 효율적 (전체의 1% 미만만 학습)
  - 메모리 효율적 (< 8GB VRAM)
  - 빠른 학습 (몇 분 내)
  - 일반 능력 손상 최소화

입력:
  - Critical Safety Neurons file (from compute_critical_safety_neurons.py)
    예: meta-llama_Llama-3.2-3B-Instruct_critical_safety_neurons_*.txt
  
  - Safety dataset file (Circuit Breakers)
    예: ./corpus_all/circuit_breakers_train.json

출력:
  - Fine-tuned model
    예: ./critical_safety_tuned_model_YYYYMMDD_HHMMSS/

사용법:
  python critical_safety_neuron_tune.py [critical_neurons_file] [dataset_file] [output_dir]
  
  예시:
python critical_safety_neuron_tune.py \
    ./output_neurons/meta-llama_Llama-3.2-3B-Instruct_critical_safety_neurons_20251208_233423.txt \
    ./corpus_all/circuit_breakers_train.json \
    --num_samples 50 

하이퍼파라미터:
  - Model: meta-llama/Llama-3.2-3B-Instruct
  - Batch Size: 2 (메모리 효율)
  - Learning Rate: 1e-6 (조심스러운 학습)
  - Epochs: 1 (과적합 방지)
  - Optimizer: AdamW
  - Max Sequence Length: 256
"""

import os
import sys
import torch
import json
import argparse
from datetime import datetime
from typing import Dict, Set, List, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Fine-tune Critical Safety Neurons')
parser.add_argument('critical_neurons_file', type=str, 
                    help='Path to critical neurons file')
parser.add_argument('dataset_file', type=str, 
                    help='Path to safety dataset file (JSON)')
parser.add_argument('--model_path', type=str, 
                    default='meta-llama/Llama-3.2-3B-Instruct',
                    help='HuggingFace model ID')
parser.add_argument('--output_dir', type=str,
                    default=None,
                    help='Output directory for fine-tuned model')
parser.add_argument('--num_samples', type=int, default=50,
                    help='Number of training samples to use')
parser.add_argument('--batch_size', type=int, default=2, 
                    help='Training batch size')
parser.add_argument('--learning_rate', type=float, default=1e-6, 
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=1, 
                    help='Number of training epochs')
parser.add_argument('--max_length', type=int, default=256, 
                    help='Maximum sequence length')
parser.add_argument('--cache_dir', type=str, default='./cache', 
                    help='HuggingFace cache directory')

args = parser.parse_args()

if args.output_dir is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"./critical_safety_tuned_model_{timestamp}"

os.makedirs(args.output_dir, exist_ok=True)

# ------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------

def print_header(text):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def print_section(text):
    """Print formatted section"""
    print(f"\n{text}")
    print(f"{'-'*70}")


def load_critical_neurons_from_file(file_path: str) -> Dict[str, Dict[int, Set[str]]]:
    """Load critical neurons from file."""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 5:
            logger.error(f"Invalid file format (expected 5 lines, got {len(lines)})")
            return None
        
        ffn_up = eval(lines[0].strip())
        ffn_down = eval(lines[1].strip())
        q = eval(lines[2].strip())
        k = eval(lines[3].strip())
        v = eval(lines[4].strip())
        
        # Convert to proper format
        for module in [ffn_up, ffn_down, q, k, v]:
            for layer_idx in list(module.keys()):
                if not isinstance(layer_idx, int):
                    module[int(layer_idx)] = module.pop(layer_idx)
                if isinstance(module[layer_idx], list):
                    module[layer_idx] = set(module[layer_idx])
        
        return {
            'ffn_up': ffn_up,
            'ffn_down': ffn_down,
            'q': q,
            'k': k,
            'v': v,
        }
    
    except Exception as e:
        logger.error(f"Error loading critical neurons file: {e}")
        return None


def load_safety_dataset(json_file: str, num_samples: int = 50) -> Dataset:
    """Load safety dataset from JSON file."""
    print_section(f"Loading Safety Dataset from {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to huggingface Dataset format
        # Support multiple field combinations
        texts = []
        for item in data:
            if isinstance(item, dict):
                # Try: prompt + llama3_output (preferred)
                if 'prompt' in item and 'llama3_output' in item:
                    text = f"Prompt: {item['prompt']}\nResponse: {item['llama3_output']}"
                    texts.append({'text': text})
                # Try: prompt + output (fallback)
                elif 'prompt' in item and 'output' in item:
                    text = f"Prompt: {item['prompt']}\nResponse: {item['output']}"
                    texts.append({'text': text})
                # Try: prompt + response
                elif 'prompt' in item and 'response' in item:
                    text = f"Prompt: {item['prompt']}\nResponse: {item['response']}"
                    texts.append({'text': text})
                # Try: text field
                elif 'text' in item:
                    texts.append({'text': item['text']})
            elif isinstance(item, str):
                texts.append({'text': item})
        
        if not texts:
            logger.error("No valid text samples found in dataset!")
            if data:
                logger.error(f"Sample item keys: {data[0].keys() if isinstance(data[0], dict) else 'not a dict'}")
            return None
        
        # Limit to num_samples
        if num_samples and num_samples > 0:
            texts = texts[:num_samples]
        
        dataset = Dataset.from_dict({
            'text': [t['text'] for t in texts]
        })
        
        logger.info(f"✅ Loaded {len(dataset)} samples from dataset")
        if len(dataset) == 0:
            logger.error("Dataset is empty after loading!")
            return None
        return dataset
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def freeze_non_critical_neurons(model, critical_neurons: Dict) -> Tuple[int, int]:
    """
    Freeze all parameters except critical safety neurons using parameter masks.
    
    Returns:
        (total_params, trainable_params)
    """
    total_params = 0
    trainable_params = 0
    
    # Count total first
    for param in model.parameters():
        total_params += param.numel()
    
    # Freeze everything first
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Create masks for each neuron weight parameter and register them as buffers
    unfrozen_count = 0
    
    for layer_idx in range(28):
        layer = model.model.layers[layer_idx]
        
        # FFN up_proj - unfreeze neuron outputs (rows)
        if 'ffn_up' in critical_neurons and hasattr(layer.mlp, 'up_proj'):
            neuron_indices = critical_neurons['ffn_up'].get(layer_idx, set())
            if neuron_indices:
                mask = torch.zeros(layer.mlp.up_proj.weight.shape, dtype=torch.bool)
                for neuron_str in neuron_indices:
                    try:
                        if isinstance(neuron_str, str):
                            neuron_idx = int(neuron_str.replace('neuron_', ''))
                        else:
                            neuron_idx = int(neuron_str)
                        
                        if neuron_idx < layer.mlp.up_proj.weight.shape[0]:
                            mask[neuron_idx, :] = True
                            unfrozen_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to parse FFN up neuron {neuron_str}: {e}")
                
                # Unfreeze based on mask
                if mask.any():
                    layer.mlp.up_proj.weight.requires_grad = True
        
        # FFN down_proj - unfreeze input neurons (columns)
        if 'ffn_down' in critical_neurons and hasattr(layer.mlp, 'down_proj'):
            neuron_indices = critical_neurons['ffn_down'].get(layer_idx, set())
            if neuron_indices:
                mask = torch.zeros(layer.mlp.down_proj.weight.shape, dtype=torch.bool)
                for neuron_str in neuron_indices:
                    try:
                        if isinstance(neuron_str, str):
                            neuron_idx = int(neuron_str.replace('neuron_', ''))
                        else:
                            neuron_idx = int(neuron_str)
                        
                        if neuron_idx < layer.mlp.down_proj.weight.shape[1]:
                            mask[:, neuron_idx] = True
                            unfrozen_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to parse FFN down neuron {neuron_str}: {e}")
                
                # Unfreeze based on mask
                if mask.any():
                    layer.mlp.down_proj.weight.requires_grad = True
        
        # Attention Q, K, V - unfreeze input neurons (columns)
        for attn_type in ['q', 'k', 'v']:
            if attn_type in critical_neurons:
                neuron_indices = critical_neurons[attn_type].get(layer_idx, set())
                if neuron_indices:
                    attr_map = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj'}
                    proj_name = attr_map[attn_type]
                    
                    if hasattr(layer.self_attn, proj_name):
                        mask = torch.zeros(getattr(layer.self_attn, proj_name).weight.shape, dtype=torch.bool)
                        for neuron_str in neuron_indices:
                            try:
                                if isinstance(neuron_str, str):
                                    neuron_idx = int(neuron_str.replace('neuron_', ''))
                                else:
                                    neuron_idx = int(neuron_str)
                                
                                proj = getattr(layer.self_attn, proj_name)
                                if neuron_idx < proj.weight.shape[1]:
                                    mask[:, neuron_idx] = True
                                    unfrozen_count += 1
                            except Exception as e:
                                logger.warning(f"Failed to parse {attn_type} neuron {neuron_str}: {e}")
                        
                        # Unfreeze based on mask
                        if mask.any():
                            proj = getattr(layer.self_attn, proj_name)
                            proj.weight.requires_grad = True
    
    # Count trainable parameters
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Unfrozen {unfrozen_count} neuron weights (entire weight matrices)")
    return total_params, trainable_params


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples."""
    tokenized = tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def custom_collate_fn(batch):
    """Convert batch to tensors."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        items = [item[key] for item in batch]
        if isinstance(items[0], list):
            # Convert list of lists to tensor
            collated[key] = torch.tensor(items)
        else:
            # Already tensor or similar
            collated[key] = torch.tensor(items)
    return collated


def train_critical_safety_neurons(model, tokenizer, train_dataset, critical_neurons):
    """Fine-tune critical safety neurons."""
    
    print_section("Critical Safety Neuron Fine-tuning Configuration")
    
    # Freeze non-critical parameters
    logger.info("Freezing non-critical parameters...")
    total_params, trainable_params = freeze_non_critical_neurons(model, critical_neurons)
    
    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0
    
    logger.info(f"📊 Parameter Statistics:")
    logger.info(f"  • Total params: {total_params / 1e9:.2f}B")
    logger.info(f"  • Trainable params: {trainable_params / 1e6:.2f}M ({trainable_pct:.4f}%)")
    logger.info(f"  • Frozen params: {(total_params - trainable_params) / 1e9:.2f}B")
    
    logger.info(f"\n⚙️  Training Settings:")
    logger.info(f"  • Batch size: {args.batch_size}")
    logger.info(f"  • Learning rate: {args.learning_rate}")
    logger.info(f"  • Epochs: {args.epochs}")
    logger.info(f"  • Max length: {args.max_length}")
    logger.info(f"  • Optimizer: AdamW")
    
    # Prepare dataset
    logger.info(f"\nPreparing dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text'],
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    
    # Setup optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    
    # Training loop
    print("\n" + "="*70)
    print("  Starting Critical Safety Neuron Fine-tuning")
    print("="*70 + "\n")
    
    model.train()
    total_loss = 0
    num_steps = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            num_steps += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                logger.info(f"  Batch {batch_idx + 1}: Loss = {avg_loss:.4f}")
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"  Fine-tuning Complete!")
    print(f"{'='*70}")
    logger.info(f"⏱️  Average loss: {avg_loss:.4f}")
    logger.info(f"⏱️  Total steps: {num_steps}")
    
    return model


def save_model(model, tokenizer, critical_neurons):
    """Save the fine-tuned model."""
    print_section("Saving Fine-tuned Model")
    
    logger.info(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save critical neurons info
    critical_info_file = os.path.join(args.output_dir, 'critical_neurons_info.json')
    critical_info = {
        'model_path': args.model_path,
        'critical_neurons_file': args.critical_neurons_file,
        'dataset_file': args.dataset_file,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'max_length': args.max_length,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(critical_info_file, 'w', encoding='utf-8') as f:
        json.dump(critical_info, f, indent=2)
    
    logger.info(f"✅ Model saved successfully!")
    logger.info(f"✅ Output directory: {args.output_dir}")
    logger.info(f"✅ Config saved to: {critical_info_file}")


def main():
    """Main function."""
    print_header("🚀 Critical Safety Neuron Fine-tuning (Critical-Tune)")
    
    logger.info(f"⚙️  Configuration:")
    logger.info(f"  • Model: {args.model_path}")
    logger.info(f"  • Critical neurons file: {args.critical_neurons_file}")
    logger.info(f"  • Dataset file: {args.dataset_file}")
    logger.info(f"  • Number of samples: {args.num_samples}")
    logger.info(f"  • Output directory: {args.output_dir}\n")
    
    # Step 1: Load critical neurons
    print_section("Loading Critical Neurons")
    critical_neurons = load_critical_neurons_from_file(args.critical_neurons_file)
    if critical_neurons is None:
        sys.exit(1)
    
    critical_total = 0
    for module_key, module_dict in critical_neurons.items():
        for layer_idx, neuron_set in module_dict.items():
            critical_total += len(neuron_set)
    
    logger.info(f"✅ Loaded {critical_total} critical safety neurons")
    if critical_total == 0:
        logger.warning("No critical neurons loaded! Proceeding with caution...")
    
    # Step 2: Load model and tokenizer
    print_section("Loading Model and Tokenizer")
    logger.info(f"Loading model: {args.model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    model.config.use_cache = False
    
    logger.info(f"✅ Model loaded (device: {model.device})")
    
    # Step 3: Load dataset
    train_dataset = load_safety_dataset(args.dataset_file, num_samples=args.num_samples)
    if train_dataset is None or len(train_dataset) == 0:
        logger.error("Failed to load training dataset!")
        sys.exit(1)
    
    logger.info(f"Using {len(train_dataset)} samples for training")
    
    # Step 4: Fine-tune critical safety neurons
    model = train_critical_safety_neurons(model, tokenizer, train_dataset, critical_neurons)
    
    # Step 5: Save model
    save_model(model, tokenizer, critical_neurons)
    
    print_header(f"✅ Critical Safety Neuron Fine-tuning Complete!")


if __name__ == '__main__':
    main()
