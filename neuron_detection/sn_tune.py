"""
Safety Neuron Tuning (SN-Tune)

- Load detected safety neurons from output file
- Freeze all non-safety neuron parameters
- Fine-tune only safety neurons on safety dataset (Circuit Breakers)
- Use small learning rate and 1 epoch as per paper

# SN-Tune

# SN-Tune with custom model
python sn_tune.py \
    --neuron_file ./output_neurons/llama_2_7b_chat_safety_neuron_accelerated_20260416_160653.txt \
    --dataset_file ./corpus_all/circuit_breakers_train.json \
    --local_model_name ./only_sn_tuned_model_llama2_7b_chat_lr3e-5 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --upload_name kmseong/llama2_7b_chat_only_sn_tuned_lr3e-5_shuffle
  

# RSN-Tune
python sn_tune.py \
    --neuron_file ./output_neurons/critical_safety_neuron_20260418_204636.txt \
    --dataset_file ./corpus_all/circuit_breakers_train.json \
    --local_model_name ./only_rsn_tuned_model_llama2_7b_chat_lr3e-5 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --upload_name kmseong/llama2_7b_chat_only_rsn_tuned_lr3e-5

"""

import argparse
import os
import sys
import json
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
from datetime import datetime
import ast
from contextlib import nullcontext
import wandb

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# =====================================================================
# Configuration
# =====================================================================
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
NUM_LAYERS = 32

# SN-Tune hyperparameters
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 1024
MAX_SAMPLES = 4994


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
# Helpers
# =====================================================================
def is_instruct_model(name: str) -> bool:
    """Return True if model name indicates an instruction-tuned model."""
    model_ref = name.lower()
    return any(tag in model_ref for tag in ('instruct', 'chat'))




# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# Safety Dataset
# =====================================================================
class SafetyDataset(Dataset):
    """
    Circuit Breakers dataset for safety alignment
    """
    
    def __init__(self, json_path, tokenizer, max_samples=None, max_length=1024, is_instruct=False):
        """
        Args:
            json_path: Path to circuit_breakers_train.json
            tokenizer: HuggingFace tokenizer
            max_samples: Maximum samples to use
            max_length: Max sequence length
            is_instruct: If True, use tokenizer.apply_chat_template() for formatting
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:min(max_samples, len(self.data))]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_instruct = is_instruct
        self._logged_first = False
        
        logger.info(f"Loaded {len(self.data)} samples from {json_path}")
        logger.info(f"  Format: {'chat template (instruct)' if is_instruct else 'plain text (base)'}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Circuit Breakers: 'prompt' and 'llama3_output' (safe response)

        harmful_prompt = item.get('prompt', '')
        safe_response = item.get('llama3_output', '')

        if self.is_instruct:
            # ── Instruct model: use chat template ──────────────────────────
            # Prompt-only with add_generation_prompt=True to get the assistant
            # turn header (<|start_header_id|>assistant<|end_header_id|>\n\n).
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": harmful_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_length = len(prompt_ids)

            # Full sequence (prompt + response + EOS)
            full_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": harmful_prompt},
                    {"role": "assistant", "content": safe_response},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )

            # Truncate to max_length
            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]

            seq_len = len(full_ids)
            pad_len = self.max_length - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            input_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len

            labels = list(input_ids)
            for i in range(min(prompt_length, self.max_length)):
                labels[i] = -100
            for i in range(self.max_length):
                if attention_mask[i] == 0:
                    labels[i] = -100

            if not self._logged_first:
                self._logged_first = True
                logger.info(f"\n[Dataset Sample #first] (instruct / chat template)")
                logger.info(f"  Keys: {item.keys()}")
                logger.info(f"  Prompt (first 100): {harmful_prompt[:100]}...")
                logger.info(f"  Response (first 100): {safe_response[:100]}...")
                logger.info(f"  prompt_length (tokens): {prompt_length}")
                logger.info(f"  full_ids length: {seq_len}")
                # Decode what the model will actually learn (labels != -100)
                learned_ids = [t for t, l in zip(input_ids, labels) if l != -100]
                logger.info(f"  Learned tokens ({len(learned_ids)}): {self.tokenizer.decode(learned_ids)[:200]}...")
                logger.info(f"  Masked (prompt) tokens: {self.tokenizer.decode(input_ids[:prompt_length])[:200]}...")

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
            }

        else:
            # ── Base model: plain Question/Answer format ────────────────────
            prompt_text = f"Question: {harmful_prompt}\nAnswer:"
            full_text = f"{prompt_text} {safe_response}"

            encodings = self.tokenizer(
                full_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            # Tokenize prompt alone to find exact token boundary
            prompt_encodings = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Labels: mask prompt tokens and padding
            labels = encodings['input_ids'].clone()
            prompt_length = prompt_encodings['input_ids'].size(1)
            labels[:, :prompt_length] = -100
            labels[encodings['attention_mask'] == 0] = -100

            if not self._logged_first:
                self._logged_first = True
                logger.info(f"\n[Dataset Sample #first] (base / plain text)")
                logger.info(f"  Keys: {item.keys()}")
                logger.info(f"  Prompt (first 100): {harmful_prompt[:100]}...")
                logger.info(f"  Response (first 100): {safe_response[:100]}...")
                logger.info(f"  prompt_length (tokens): {prompt_length}")
                input_ids_list = encodings['input_ids'][0].tolist()
                labels_list = labels[0].tolist()
                learned_ids = [t for t, l in zip(input_ids_list, labels_list) if l != -100]
                logger.info(f"  Learned tokens ({len(learned_ids)}): {self.tokenizer.decode(learned_ids)[:200]}...")
                logger.info(f"  Masked (prompt) tokens: {self.tokenizer.decode(input_ids_list[:prompt_length])[:200]}...")

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
                # neurons are COLUMNS (intermediate_dim axis), same index as up_proj rows
                trainable_neuron_params += len(neuron_indices) * param.shape[0]
                unfrozen_modules['ffn_down'] += 1
                
                def make_hook(indices):
                    def hook(grad):
                        mask = torch.zeros_like(grad)
                        mask[:, indices] = 1.0  # columns = intermediate_dim neurons
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
    learning_rate=1e-5,
    num_epochs=3,
    grad_accum_steps=4,
    warmup_ratio=0.1,
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
    # Match batch tensor device to the model device to avoid cuda:0/cuda:1 mismatch.
    model_device = next(model.parameters()).device
    if str(device) != str(model_device):
        logger.warning(f"Requested device={device}, but model is on {model_device}. Using model device.")
    device = model_device

    model.train()

    # Only optimize trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    total_optimization_steps = num_epochs * math.ceil(len(train_dataloader) / grad_accum_steps)
    warmup_steps = int(total_optimization_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimization_steps
    )

    total_loss = 0.0
    total_steps = 0
    optimizer_steps = 0
    global_step = 0

    logger.info(f"Starting SN-Tune training...")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Grad accum steps: {grad_accum_steps}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * grad_accum_steps}")
    logger.info(f"  Num batches: {len(train_dataloader)}")
    logger.info(f"  Total optimization steps: {total_optimization_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")

    optimizer.zero_grad(set_to_none=True)

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

                # Count valid labels (response-only, excluding prompt/padding)
                valid_labels = (labels != -100).sum().item()
                logger.info(f"  Valid labels (response-only): {valid_labels}/{labels.numel()}")

            # Forward pass (phase0_SSFT와 동일한 방식: loss를 grad_accum_steps로 나눠서 autocast 내부에서 처리)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss / grad_accum_steps

            # NaN/Inf 처리 (backward 전에 체크)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf detected at batch {batch_idx + 1}. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass
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
            
            # Optimizer step (accumulation step 도달 시 또는 마지막 배치)
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                    norm_type=2,
                ).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                wandb.log({
                    "train/loss": loss.item() * grad_accum_steps,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/optimizer_step": optimizer_steps,
                    "train/grad_norm": grad_norm,
                }, step=global_step)

            loss_val = loss.item() * grad_accum_steps
            
            total_loss += loss_val
            epoch_loss += loss_val
            total_steps += 1
            
            # Log every 5 batches
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                avg_batch_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
                logger.info(f"  Batch {batch_idx + 1}: loss = {loss_val:.4f}")
        
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} completed - Epoch Loss: {epoch_avg_loss:.4f}")
        wandb.log({"train/epoch_loss": epoch_avg_loss, "epoch": epoch + 1})
    
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
    parser = argparse.ArgumentParser(description="Safety Neuron Tuning (SN-Tune)")
    # Backward-compatible positional args
    parser.add_argument("safety_neurons_file", nargs="?", default=None, help="Path to safety neurons txt file (positional, optional)")
    parser.add_argument("safety_dataset_json", nargs="?", default=None, help="Path to circuit_breakers_train.json (positional, optional)")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory (positional, optional)")

    # Preferred named args
    parser.add_argument("--neuron_file", type=str, default=None, help="Path to safety neurons txt file")
    parser.add_argument("--dataset_file", type=str, default=None, help="Path to circuit_breakers_train.json")
    parser.add_argument("--local_model_name", type=str, default=None, help="Local output model directory name")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME, help="HuggingFace model name or local path")
    parser.add_argument("--upload_name", type=str, default=None, help="Optional Hugging Face repo id (e.g., username/model-name). If set, upload after training")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face token for upload")
    args = parser.parse_args(argv)

    safety_neurons_file = args.neuron_file or args.safety_neurons_file
    safety_dataset_json = args.dataset_file or args.safety_dataset_json
    output_dir = args.local_model_name or args.output_dir or "./sn_tuned_model"
    learning_rate = args.learning_rate
    model_name = args.model_name
    upload_name = args.upload_name
    hf_token = args.hf_token

    if safety_neurons_file is None or safety_dataset_json is None:
        parser.error("Provide neuron/dataset via --neuron_file and --dataset_file (or positional args).")

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
    logger.info(f"Upload target: {upload_name if upload_name else 'None'}")
    logger.info(f"Log file: {log_file}\n")

    _is_instruct = is_instruct_model(model_name)
    logger.info(f"Model: {model_name}")
    logger.info(f"Instruct model detected: {_is_instruct} → using {'chat template' if _is_instruct else 'plain text'} format\n")

    run_name = os.path.basename(output_dir)
    wandb.init(
        entity="gokms0509-yonsei-university",
        project="SN-Tune",
        name=run_name,
        config={
            "model_name": model_name,
            "learning_rate": learning_rate,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_samples": MAX_SAMPLES,
            "is_instruct": _is_instruct,
            "safety_neurons_file": os.path.basename(safety_neurons_file),
        },
    )
    logger.info(f"✓ wandb run initialized: {run_name}")
    
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
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    logger.info("✓ Model and tokenizer loaded (bfloat16)")
    model_device = next(model.parameters()).device
    logger.info(f"✓ Model loaded on device: {model_device}")
    
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
        max_length=MAX_SEQ_LENGTH,
        is_instruct=_is_instruct,
    )
    
    train_dataloader = DataLoader(
        safety_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(112),
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
        learning_rate=learning_rate,
        num_epochs=NUM_EPOCHS,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=0.1,
        device=model_device,
    )
    
    # =====================================================================
    # 6. Save fine-tuned model
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lr_str = f"lr{learning_rate:.0e}".replace("-0", "-").replace("+0", "")
    final_output_dir = f"{output_dir}_{lr_str}_{timestamp}"
    
    logger.info(f"\nSaving fine-tuned model...")
    save_sn_tuned_model(model, tokenizer, final_output_dir)
    
    # Clean up gradient hooks
    for hook in gradient_hooks:
        hook.remove()
    logger.info("✓ Gradient hooks cleaned up")

    wandb.finish()
    
    logger.info(f"\n{'='*70}")
    logger.info("SN-Tune Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Fine-tuned model saved to: {final_output_dir}")

    if upload_name:
        logger.info(f"\nStarting upload to Hugging Face: {upload_name}")
        try:
            from upload_sn_tuned_model import upload_to_huggingface

            upload_to_huggingface(final_output_dir, upload_name, hf_token)
            logger.info(f"✓ Upload completed: https://huggingface.co/{upload_name}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            logger.error("Model was saved locally; you can upload manually with upload_sn_tuned_model.py")

    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
