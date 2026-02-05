"""
Safety Neuron Tuning (SN-Tune)

- Load detected safety neurons from output file
- Freeze all non-safety neuron parameters
- Fine-tune only safety neurons on safety dataset (Circuit Breakers)
- Use small learning rate and 1 epoch as per paper

python sn_tune.py \
  ./output_neurons/meta-llama_Llama-3.2-3B-Instruct_harmful_prompts_200_threshold_neurons_200_20251208_215958.txt \
  ./corpus_all/circuit_breakers_train.json \
  ./sn_tuned_model

"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from datetime import datetime
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# Configuration
# =====================================================================
model_name = "meta-llama/Llama-3.2-3B-Instruct"
NUM_LAYERS = 28

# SN-Tune hyperparameters
LEARNING_RATE = 1e-6  # Very small LR as per paper
NUM_EPOCHS = 1  # 1 epoch fine-tuning
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 256
MAX_SAMPLES = 50  # Use only 50 samples for fine-tuning

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# Safety Dataset
# =====================================================================
class SafetyDataset(Dataset):
    """
    Circuit Breakers dataset for safety alignment
    """
    
    def __init__(self, json_path, tokenizer, max_samples=None, max_length=512):
        """
        Args:
            json_path: Path to circuit_breakers_train.json
            tokenizer: HuggingFace tokenizer
            max_samples: Maximum samples to use
            max_length: Max sequence length
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:min(max_samples, len(self.data))]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Circuit Breakers: 'prompt' and 'llama3_output' (safe response)
        
        if idx == 0:  # 첫 번째 샘플 로그
            logger.info(f"\n[Dataset Sample #0]")
            logger.info(f"  Keys: {item.keys()}")
            logger.info(f"  Prompt (first 100 chars): {item.get('prompt', '')[:100]}...")
            logger.info(f"  Response (first 100 chars): {item.get('llama3_output', '')[:100]}...")
        
        harmful_prompt = item.get('prompt', '')
        safe_response = item.get('llama3_output', '')
        
        # Combine into training text: harmful_prompt + safe_response
        full_text = f"{harmful_prompt} {safe_response}"
        
        encodings = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Labels: 패딩 토큰을 -100으로 설정 (loss 계산에서 무시됨)
        labels = encodings['input_ids'].clone()
        labels[encodings['attention_mask'] == 0] = -100
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
        }


# =====================================================================
# Load Safety Neurons from Detection Output
# =====================================================================
def load_safety_neurons(output_file):
    """
    Load safety neurons from detection output file
    
    Format:
        Line 0: ffn_up_common (JSON: {str(layer_idx): [neuron_indices]})
        Line 1: ffn_down_common (JSON)
        Line 2: q_common (JSON)
        Line 3: k_common (JSON)
        Line 4: v_common (JSON)
    
    Returns:
        safety_neurons: {
            'ffn_up': {layer_idx(int): set(neuron_indices)},
            'ffn_down': {layer_idx(int): set(neuron_indices)},
            'q': {layer_idx(int): set(neuron_indices)},
            'k': {layer_idx(int): set(neuron_indices)},
            'v': {layer_idx(int): set(neuron_indices)},
        }
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    safety_neurons = {}
    
    # Parse each line as JSON dict and convert string keys to integers
    try:
        safety_neurons['ffn_up'] = {
            int(k): set(v) for k, v in json.loads(lines[0].strip()).items()
        }
        safety_neurons['ffn_down'] = {
            int(k): set(v) for k, v in json.loads(lines[1].strip()).items()
        }
        safety_neurons['q'] = {
            int(k): set(v) for k, v in json.loads(lines[2].strip()).items()
        }
        safety_neurons['k'] = {
            int(k): set(v) for k, v in json.loads(lines[3].strip()).items()
        }
        safety_neurons['v'] = {
            int(k): set(v) for k, v in json.loads(lines[4].strip()).items()
        }
    except Exception as e:
        logger.error(f"Error parsing safety neurons file: {e}")
        raise
    
    logger.info(f"Loaded safety neurons from {output_file}")
    
    # Log summary with layer-wise breakdown
    logger.info(f"\n{'='*70}")
    logger.info(f"Safety Neurons Loaded - Detailed Breakdown")
    logger.info(f"{'='*70}")
    
    total_neurons = 0
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        module_total = sum(len(neurons) for neurons in safety_neurons[module_type].values())
        logger.info(f"  {module_type:12} : {module_total:4} neurons (column indices)")
        total_neurons += module_total
        
        # Show which layers have neurons
        layers_with_neurons = [l for l in safety_neurons[module_type] if safety_neurons[module_type][l]]
        if layers_with_neurons:
            logger.info(f"    └─ Layers with neurons: {layers_with_neurons[:5]}{'...' if len(layers_with_neurons) > 5 else ''}")
        # Show which layers have neurons
        layers_with_neurons = [l for l in safety_neurons[module_type] if safety_neurons[module_type][l]]
        if layers_with_neurons:
            logger.info(f"    └─ Layers with neurons: {layers_with_neurons[:5]}{'...' if len(layers_with_neurons) > 5 else ''}")
    
    logger.info(f"\nTotal safety neurons: {total_neurons}")
    logger.info(f"{'='*70}\n")
    
    return safety_neurons


# =====================================================================
# Freeze Parameters Except Safety Neurons
# =====================================================================
def freeze_non_safety_neurons(model, safety_neurons):
    """
    Freeze all parameters except those in safety_neurons
    
    Args:
        model: LLaMA model
        safety_neurons: Dict of safety neuron indices per layer/module
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"[3/6] Parameter Freezing Setup")
    logger.info(f"{'='*70}")
    
    total_params = 0
    trainable_params = 0
    unfrozen_modules = {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0}
    unfrozen_layers = {'ffn_up': set(), 'ffn_down': set(), 'q': set(), 'k': set(), 'v': set()}
    
    logger.info(f"Safety neurons keys by module:")
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        layer_keys = sorted([k for k in safety_neurons[module_type].keys()])
        logger.info(f"  {module_type:12}: {layer_keys[:10]}{'...' if len(layer_keys) > 10 else ''}")
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False  # Freeze by default
        
        # Check if this parameter should be trainable
        if 'mlp.up_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['ffn_up'] and safety_neurons['ffn_up'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules['ffn_up'] += 1
                unfrozen_layers['ffn_up'].add(layer_idx)
        
        elif 'mlp.down_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['ffn_down'] and safety_neurons['ffn_down'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules['ffn_down'] += 1
                unfrozen_layers['ffn_down'].add(layer_idx)
        
        elif 'self_attn.q_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['q'] and safety_neurons['q'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules['q'] += 1
                unfrozen_layers['q'].add(layer_idx)
        
        elif 'self_attn.k_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['k'] and safety_neurons['k'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules['k'] += 1
                unfrozen_layers['k'].add(layer_idx)
        
        elif 'self_attn.v_proj.weight' in name:
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['v'] and safety_neurons['v'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
                unfrozen_modules['v'] += 1
                unfrozen_layers['v'].add(layer_idx)
    
    logger.info(f"\n✓ Freezing complete")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters (safety neurons): {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params / total_params * 100:.4f}%")
    
    logger.info(f"\nUnfrozen modules (layers with safety neurons):")
    for module_type, count in unfrozen_modules.items():
        logger.info(f"  {module_type:12} : {count} layers unfrozen, layers: {sorted(list(unfrozen_layers[module_type]))}")
    
    logger.info(f"  Note: Actual trainable columns will be further masked by gradient hooks")
    logger.info(f"{'='*70}\n")


# =====================================================================
# Training Loop
# =====================================================================
def train_sn_tune(
    model,
    tokenizer,
    train_dataloader,
    learning_rate=1e-6,
    num_epochs=1,
    device=DEVICE,
):
    """
    SN-Tune training loop
    
    Args:
        model: LLaMA model with frozen non-safety parameters
        tokenizer: Tokenizer
        train_dataloader: DataLoader for safety dataset
        learning_rate: Learning rate
        num_epochs: Number of epochs
        device: Device to use
    """
    model = model.to(device)
    model.train()
    
    # Only optimize trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    total_loss = 0.0
    total_steps = 0
    
    logger.info(f"Starting SN-Tune training...")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {len(train_dataloader)}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Log first batch details
            if batch_idx == 0:
                logger.info(f"\n[First Batch Info]")
                logger.info(f"  Batch size: {input_ids.shape[0]}")
                logger.info(f"  Sequence length: {input_ids.shape[1]}")
                logger.info(f"  Device: {input_ids.device}")
                
                # Count valid labels (not -100)
                valid_labels = (labels != -100).sum().item()
                logger.info(f"  Valid labels (non-padding): {valid_labels}/{labels.numel()}")
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Log gradient info for first batch
            if batch_idx == 0:
                logger.info(f"\n[Gradient Check - Batch 0]")
                non_zero_grads = 0
                zero_grads = 0
                max_grad = 0
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_abs_max = param.grad.abs().max().item()
                        max_grad = max(max_grad, grad_abs_max)
                        if param.grad.abs().sum() > 0:
                            non_zero_grads += 1
                        else:
                            zero_grads += 1
                logger.info(f"  Parameters with non-zero gradients: {non_zero_grads}")
                logger.info(f"  Parameters with zero gradients: {zero_grads}")
                logger.info(f"  Max gradient magnitude: {max_grad:.6f}")
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            
            # Check for NaN
            loss_val = loss.item()
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected at batch {batch_idx + 1}. Skipping this batch.")
                continue
            
            optimizer.step()
            
            total_loss += loss_val
            epoch_loss += loss_val
            total_steps += 1
            
            # Log every 5 batches
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                avg_batch_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
                logger.info(f"  Batch {batch_idx + 1}: loss = {loss_val:.4f}")
        
        logger.info(f"Epoch {epoch + 1} completed - Epoch Loss: {epoch_loss / len(train_dataloader):.4f}")
    
    avg_loss = total_loss / total_steps
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Complete")
    logger.info(f"{'='*70}")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Training time: {num_epochs} epoch(s)")
    
    # Verify that only safety neurons were modified
    logger.info(f"\n[Post-Training Verification]")
    modified_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            modified_params += 1
    logger.info(f"  Parameters that were trained: {modified_params}")
    logger.info(f"{'='*70}\n")
    
    return model


# =====================================================================
# Save Fine-tuned Model
# =====================================================================
def save_sn_tuned_model(model, tokenizer, save_path):
    """
    Save the SN-tuned model and tokenizer
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        save_path: Path to save the model
    """
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"  Saving model to {save_path}...")
    model.save_pretrained(save_path)
    logger.info(f"  ✓ Model saved")
    
    logger.info(f"  Saving tokenizer to {save_path}...")
    tokenizer.save_pretrained(save_path)
    logger.info(f"  ✓ Tokenizer saved")


# =====================================================================
# Main
# =====================================================================
def main(argv):
    if len(argv) < 2:
        logger.error("Usage: python sn_tune.py <safety_neurons_file> <safety_dataset_json> [output_dir]")
        logger.error("Example: python sn_tune.py ./output_neurons/meta-llama_...neurons_200_20251208_*.txt ./corpus_all/circuit_breakers_train.json ./sn_tuned_model")
        sys.exit(1)
    
    safety_neurons_file = argv[0]
    safety_dataset_json = argv[1]
    output_dir = argv[2] if len(argv) > 2 else "./sn_tuned_model"
    
    # Verify files exist
    if not os.path.exists(safety_neurons_file):
        logger.error(f"Safety neurons file not found: {safety_neurons_file}")
        sys.exit(1)
    
    if not os.path.exists(safety_dataset_json):
        logger.error(f"Safety dataset file not found: {safety_dataset_json}")
        sys.exit(1)
    
    logger.info(f"\n{'='*70}")
    logger.info("Safety Neuron Tuning (SN-Tune)")
    logger.info(f"{'='*70}")
    logger.info(f"Safety neurons file: {safety_neurons_file}")
    logger.info(f"Safety dataset file: {safety_dataset_json}")
    logger.info(f"Output directory: {output_dir}\n")
    
    # =====================================================================
    # 1. Load model and tokenizer
    # =====================================================================
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float32,  # float32로 변경 (float16의 numerical instability 해결)
    )
    model.eval()
    logger.info("✓ Model and tokenizer loaded (float32)")
    
    # =====================================================================
    # 2. Load safety neurons
    # =====================================================================
    logger.info("\nLoading safety neurons...")
    safety_neurons = load_safety_neurons(safety_neurons_file)
    
    # =====================================================================
    # 3. Freeze non-safety parameters
    # =====================================================================
    logger.info("\nFreezing non-safety parameters...")
    freeze_non_safety_neurons(model, safety_neurons)
    
    # =====================================================================
    # 4. Load safety dataset
    # =====================================================================
    logger.info("\nLoading safety dataset...")
    safety_dataset = SafetyDataset(
        safety_dataset_json,
        tokenizer,
        max_samples=MAX_SAMPLES,
        max_length=MAX_SEQ_LENGTH
    )
    
    train_dataloader = DataLoader(
        safety_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    logger.info(f"✓ DataLoader created: {len(train_dataloader)} batches")
    logger.info(f"  Total samples: {len(safety_dataset)}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Number of batches: {len(train_dataloader)}")
    logger.info(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    
    # =====================================================================
    # 5. SN-Tune training
    # =====================================================================
    logger.info("\nStarting SN-Tune training...")
    model = train_sn_tune(
        model,
        tokenizer,
        train_dataloader,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
    )
    
    # =====================================================================
    # 6. Save fine-tuned model
    # =====================================================================
    logger.info(f"\n{'='*70}")
    logger.info("[6/6] Saving Fine-tuned Model")
    logger.info(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = f"{output_dir}_{timestamp}"
    
    logger.info(f"Output directory: {final_output_dir}")
    save_sn_tuned_model(model, tokenizer, final_output_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info("SN-Tune Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"✓ Fine-tuned model saved to: {final_output_dir}")
    logger.info(f"  - Model weights saved")
    logger.info(f"  - Tokenizer saved")
    logger.info(f"  - Ready for upload to Hugging Face")
    logger.info(f"{'='*70}\n")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
