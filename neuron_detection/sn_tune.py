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
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
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
    
    # Parse each line as a dict string
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
    
    # Log summary
    total_neurons = 0
    for module_type in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        module_total = sum(len(neurons) for neurons in safety_neurons[module_type].values())
        logger.info(f"  {module_type}: {module_total} neurons across layers")
        total_neurons += module_total
    
    logger.info(f"Total safety neurons: {total_neurons}")
    
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
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        param.requires_grad = False  # Freeze by default
        
        # Check if this parameter should be trainable
        # Format: model.layers.{layer_idx}.mlp.{up_proj|down_proj}.weight
        # Format: model.layers.{layer_idx}.self_attn.{q_proj|k_proj|v_proj}.weight
        
        if 'mlp.up_proj.weight' in name:
            # FFN up_proj
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['ffn_up'] and safety_neurons['ffn_up'][layer_idx]:
                # Only unfreeze the neurons in safety_neurons
                param.requires_grad = True
                trainable_params += param.numel()
        
        elif 'mlp.down_proj.weight' in name:
            # FFN down_proj
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['ffn_down'] and safety_neurons['ffn_down'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
        
        elif 'self_attn.q_proj.weight' in name:
            # Attention Q
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['q'] and safety_neurons['q'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
        
        elif 'self_attn.k_proj.weight' in name:
            # Attention K
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['k'] and safety_neurons['k'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
        
        elif 'self_attn.v_proj.weight' in name:
            # Attention V
            layer_idx = int(name.split('.')[2])
            if layer_idx in safety_neurons['v'] and safety_neurons['v'][layer_idx]:
                param.requires_grad = True
                trainable_params += param.numel()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Parameter Freezing Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters (safety neurons): {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"Trainable ratio: {trainable_params / total_params * 100:.4f}%")
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
        
        pbar = tqdm(train_dataloader, desc=f"Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                return_dict=True
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_steps += 1
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_steps
    logger.info(f"\n{'='*70}")
    logger.info(f"Training Complete")
    logger.info(f"{'='*70}")
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"{'='*70}\n")
    
    return model


# =====================================================================
# Save Fine-tuned Model
# =====================================================================
def save_sn_tuned_model(model, save_path):
    """
    Save the SN-tuned model
    
    Args:
        model: Fine-tuned model
        save_path: Path to save the model
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path}")


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
        torch_dtype=torch.float16,
    )
    model.eval()
    logger.info("✓ Model and tokenizer loaded")
    
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output_dir = f"{output_dir}_{timestamp}"
    
    logger.info(f"\nSaving fine-tuned model...")
    save_sn_tuned_model(model, final_output_dir)
    
    logger.info(f"\n{'='*70}")
    logger.info("SN-Tune Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Fine-tuned model saved to: {final_output_dir}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
