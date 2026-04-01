"""
Safety Neuron Tuning (SN-Tune)

- Load detected safety neurons from output file
- Freeze all non-safety neuron parameters
- Fine-tune only safety neurons on safety dataset (Circuit Breakers)
- Use small learning rate and 1 epoch as per paper

# SN-Tune
python sn_tune.py \
  ./output_neurons/safety_neuron_threshold_20260331_085057.txt \
  ./corpus_all/circuit_breakers_train.json \
  ./only_sn_tuned_model

# RSN-Tune
python sn_tune.py \
  ./output_neurons/critical-safety-neuron_20260401_115452.txt \
  ./corpus_all/circuit_breakers_train.json \
  ./only_rsn_tuned_model
"""

import os
import sys
import json
import torch
import torch.nn as nn
from bitsandbytes.optim import AdamW8bit
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from datetime import datetime
import ast

logger = logging.getLogger(__name__)


def setup_logging(log_dir="./logs/sn_tuning"):
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError:
        log_dir = "./logs/sn_tuning"
        os.makedirs(log_dir, exist_ok=True)
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sn_tune_{log_timestamp}.log")

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file

# =====================================================================
# Configuration
# =====================================================================
model_name = "meta-llama/Llama-3.2-3B"
NUM_LAYERS = 28

# SN-Tune hyperparameters
LEARNING_RATE = 1e-5  # Very small LR as per paper
NUM_EPOCHS = 3  # 1 epoch fine-tuning
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 512
MAX_SAMPLES = 4994  # Use only 50 samples for fine-tuning

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
        self._logged_first = False
        
        logger.info(f"Loaded {len(self.data)} samples from {json_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Circuit Breakers: 'prompt' and 'llama3_output' (safe response)
        
        if not self._logged_first:  # 첫 번째 접근 샘플 로그 (shuffle 영향 방지)
            self._logged_first = True
            logger.info(f"\n[Dataset Sample #first]")
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
        Line 0: ffn_up_common (dict)
        Line 1: ffn_down_common (dict)
        Line 2: q_common (dict)
        Line 3: k_common (dict)
        Line 4: v_common (dict)
    
    Returns:
        safety_neurons: {
            'ffn_up': {layer_idx: set(neuron_names)},
            'ffn_down': {layer_idx: set(neuron_names)},
            'q': {layer_idx: set(neuron_names)},
            'k': {layer_idx: set(neuron_names)},
            'v': {layer_idx: set(neuron_names)},
        }
    """
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    safety_neurons = {}
    
    # Parse each line as a dict string and convert keys from string to int
    try:
        # Keys are stored as strings, need to convert to int
        ffn_up_raw = ast.literal_eval(lines[0].strip())
        ffn_down_raw = ast.literal_eval(lines[1].strip())
        q_raw = ast.literal_eval(lines[2].strip())
        k_raw = ast.literal_eval(lines[3].strip())
        v_raw = ast.literal_eval(lines[4].strip())
        
        # Convert string keys to int keys
        safety_neurons['ffn_up'] = {int(k): v for k, v in ffn_up_raw.items()}
        safety_neurons['ffn_down'] = {int(k): v for k, v in ffn_down_raw.items()}
        safety_neurons['q'] = {int(k): v for k, v in q_raw.items()}
        safety_neurons['k'] = {int(k): v for k, v in k_raw.items()}
        safety_neurons['v'] = {int(k): v for k, v in v_raw.items()}
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
        logger.info(f"  {module_type:12} : {module_total:4} neurons")
        total_neurons += module_total
        
        # Show which layers have neurons
        layers_with_neurons = [l for l in safety_neurons[module_type] if safety_neurons[module_type][l]]
        if layers_with_neurons:
            logger.info(f"    └─ Layers with neurons: {layers_with_neurons[:5]}{'...' if len(layers_with_neurons) > 5 else ''}")
    
    logger.info(f"\nTotal safety neurons: {total_neurons}")
    logger.info(f"{'='*70}\n")
    
    return safety_neurons


# =====================================================================
# Setup Gradient Masking for Safety Neurons
# =====================================================================
def setup_gradient_masking(model, safety_neurons):
    """
    Setup gradient masking to train only safety neurons.
    
    Neuron = specific row/column in weight matrix.
    We use backward hooks to zero out gradients for non-safety neurons.
    
    Args:
        model: LLaMA model
        safety_neurons: Dict of safety neuron indices per layer/module
    
    Returns:
        hooks: List of registered hooks (for cleanup)
    """
    hooks = []
    total_params = 0
    trainable_neuron_params = 0
    unfrozen_modules = {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0}
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False  # Freeze by default
        
        # Extract layer index from name
        # e.g., "model.layers.0.mlp.up_proj.weight" -> layer_idx = 0
        parts = name.split('.')
        if len(parts) < 4 or parts[0] != 'model' or parts[1] != 'layers':
            continue
        
        try:
            layer_idx = int(parts[2])
        except ValueError:
            continue
        
        # Check module type and setup gradient masking
        if 'mlp.up_proj.weight' in name:
            neuron_indices = safety_neurons['ffn_up'].get(layer_idx, [])
            if neuron_indices:
                param.requires_grad = True
                # up_proj: weight shape [intermediate_dim, hidden_dim]
                # neurons are rows in weight matrix
                trainable_neuron_params += len(neuron_indices) * param.shape[1]
                unfrozen_modules['ffn_up'] += 1
                
                # Register backward hook for gradient masking
                def make_hook(indices):
                    def hook(grad):
                        mask = torch.zeros_like(grad)
                        mask[indices, :] = 1.0  # Only keep gradients for safety neurons
                        return grad * mask
                    return hook
                
                hook_handle = param.register_hook(make_hook(neuron_indices))
                hooks.append(hook_handle)
        
        elif 'mlp.down_proj.weight' in name:
            neuron_indices = safety_neurons['ffn_down'].get(layer_idx, [])
            if neuron_indices:
                param.requires_grad = True
                # down_proj: weight shape [hidden_dim, intermediate_dim]
                # neurons are rows in weight matrix (output dimensions)
                trainable_neuron_params += len(neuron_indices) * param.shape[1]
                unfrozen_modules['ffn_down'] += 1
                
                def make_hook(indices):
                    def hook(grad):
                        mask = torch.zeros_like(grad)
                        mask[indices, :] = 1.0
                        return grad * mask
                    return hook
                
                hook_handle = param.register_hook(make_hook(neuron_indices))
                hooks.append(hook_handle)
        
        elif 'self_attn.q_proj.weight' in name:
            neuron_indices = safety_neurons['q'].get(layer_idx, [])
            if neuron_indices:
                param.requires_grad = True
                # q_proj: weight shape [hidden_dim, hidden_dim]
                # neurons are rows
                trainable_neuron_params += len(neuron_indices) * param.shape[1]
                unfrozen_modules['q'] += 1
                
                def make_hook(indices):
                    def hook(grad):
                        mask = torch.zeros_like(grad)
                        mask[indices, :] = 1.0
                        return grad * mask
                    return hook
                
                hook_handle = param.register_hook(make_hook(neuron_indices))
                hooks.append(hook_handle)
        
        elif 'self_attn.k_proj.weight' in name:
            neuron_indices = safety_neurons['k'].get(layer_idx, [])
            if neuron_indices:
                param.requires_grad = True
                # k_proj: neurons are rows
                trainable_neuron_params += len(neuron_indices) * param.shape[1]
                unfrozen_modules['k'] += 1
                
                def make_hook(indices):
                    def hook(grad):
                        mask = torch.zeros_like(grad)
                        mask[indices, :] = 1.0
                        return grad * mask
                    return hook
                
                hook_handle = param.register_hook(make_hook(neuron_indices))
                hooks.append(hook_handle)
        
        elif 'self_attn.v_proj.weight' in name:
            neuron_indices = safety_neurons['v'].get(layer_idx, [])
            if neuron_indices:
                param.requires_grad = True
                # v_proj: neurons are rows
                trainable_neuron_params += len(neuron_indices) * param.shape[1]
                unfrozen_modules['v'] += 1
                
                def make_hook(indices):
                    def hook(grad):
                        mask = torch.zeros_like(grad)
                        mask[indices, :] = 1.0
                        return grad * mask
                    return hook
                
                hook_handle = param.register_hook(make_hook(neuron_indices))
                hooks.append(hook_handle)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Gradient Masking Setup Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable neuron parameters (effective): {trainable_neuron_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_neuron_params:,}")
    logger.info(f"Trainable ratio: {trainable_neuron_params / total_params * 100:.4f}%")
    logger.info(f"Gradient masking hooks registered: {len(hooks)}")
    
    logger.info(f"\nLayers with gradient masking:")
    for module_type, count in unfrozen_modules.items():
        logger.info(f"  {module_type:12} : {count} layers")
    
    logger.info(f"{'='*70}\n")
    
    return hooks


# =====================================================================
# Training Loop
# =====================================================================
def train_sn_tune(
    model,
    tokenizer,
    train_dataloader,
    learning_rate=1e-6,
    num_epochs=1,
    grad_accum_steps=4,
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
    optimizer = AdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.0
    )
    
    total_loss = 0.0
    total_steps = 0
    optimizer_steps = 0
    
    logger.info(f"Starting SN-Tune training...")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Grad accum steps: {grad_accum_steps}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * grad_accum_steps}")
    logger.info(f"  Num batches: {len(train_dataloader)}")
    
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
            
            # Gradient accumulation 시작
            if batch_idx % grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
                loss = outputs.loss

            # NaN/Inf 처리 (backward 전에 체크)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected at batch {batch_idx + 1}. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass
            (loss / grad_accum_steps).backward()
            
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
            
            # Optimizer step (accumulation step 도달 시 또는 마지막 배치)
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                optimizer.step()
                optimizer_steps += 1

            loss_val = loss.item()
            
            total_loss += loss_val
            epoch_loss += loss_val
            total_steps += 1
            
            # Log every 5 batches
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                avg_batch_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
                logger.info(f"  Batch {batch_idx + 1}: loss = {loss_val:.4f}")
        
        logger.info(f"Epoch {epoch + 1} completed - Epoch Loss: {epoch_loss / len(train_dataloader):.4f}")
    
    avg_loss = total_loss / max(1, total_steps)
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Complete")
    logger.info(f"{'='*70}")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Optimizer steps: {optimizer_steps}")
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
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")


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

    log_file = setup_logging()
    
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
    logger.info(f"Log file: {log_file}\n")
    
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
        torch_dtype=torch.bfloat16,  # bfloat16로 변경 (float16의 numerical instability 해결)
    )
    model.eval()
    logger.info("✓ Model and tokenizer loaded (bfloat16)")
    
    # =====================================================================
    # 2. Load safety neurons
    # =====================================================================
    logger.info("\nLoading safety neurons...")
    safety_neurons = load_safety_neurons(safety_neurons_file)
    
    # =====================================================================
    # 3. Setup gradient masking for safety neurons
    # =====================================================================
    logger.info("\nSetting up gradient masking for safety neurons...")
    gradient_hooks = setup_gradient_masking(model, safety_neurons)
    
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
        grad_accum_steps=GRAD_ACCUM_STEPS,
        device=DEVICE,
    )
    
    # =====================================================================
    # 6. Save fine-tuned model
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = f"{output_dir}_{timestamp}"
    
    logger.info(f"\nSaving fine-tuned model...")
    save_sn_tuned_model(model, tokenizer, final_output_dir)
    
    # Clean up gradient hooks
    for hook in gradient_hooks:
        hook.remove()
    logger.info("✓ Gradient hooks cleaned up")
    
    logger.info(f"\n{'='*70}")
    logger.info("SN-Tune Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Fine-tuned model saved to: {final_output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
