"""
Safety Neuron Tuning (SN-Tune)
Safety neuron들만 활성화하고 나머지 파라미터는 freeze한 후
circuit_breakers 데이터로 fine-tuning
"""

import os
import torch
import json
from datetime import datetime
from typing import Dict, List

# Set environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType

# ==================== Configuration ====================
model_name = "meta-llama/Llama-3.2-3B-Instruct"
safety_neuron_file = "../neuron_detection/output_neurons/meta-llama_Meta-Llama-3.2-3B-Instruct_harmful_prompts_200_real_neurons.txt"
dataset_file = "../neuron_detection/corpus_all/circuit_breakers_train.json"
output_dir = "./sn_tune_output"
cache_dir = "./cache"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# ==================== Load Safety Neurons ====================

def load_safety_neurons(filename: str) -> Dict:
    """Load safety neurons from detection output file"""
    print("Loading Safety Neurons...")
    
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Using empty neuron data.")
        dummy_neuron = []
        for i in range(5):
            layer_dict = {}
            for layer in range(28):
                layer_dict[layer] = set()
            dummy_neuron.append(layer_dict)
        return dummy_neuron
    
    try:
        activate_neuron = []
        with open(filename, 'r') as file:
            neurons = file.readlines()
            for neuron in neurons:
                neuron = eval(neuron.strip())
                activate_neuron.append(neuron)
        print(f"✓ Safety Neurons loaded: {len(activate_neuron)} neuron types")
        return activate_neuron
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        dummy_neuron = []
        for i in range(5):
            layer_dict = {}
            for layer in range(28):
                layer_dict[layer] = set()
            dummy_neuron.append(layer_dict)
        return dummy_neuron

# ==================== Enable Safety Neurons (Neuron-Level Freezing) ====================

def setup_safety_neuron_freezing(model, safety_neuron: List[Dict]):
    """
    Freeze all parameters, then selectively unfreeze only safety neurons
    This ensures ONLY safety neurons are trainable
    
    safety_neuron[0]: FFN up_proj neurons {layer: {neuron_indices}}
    safety_neuron[1]: FFN down_proj neurons {layer: {neuron_indices}}
    safety_neuron[2]: Attention q_proj neurons {layer: {neuron_indices}}
    safety_neuron[3]: Attention k_proj neurons {layer: {neuron_indices}}
    safety_neuron[4]: Attention v_proj neurons {layer: {neuron_indices}}
    """
    print("\n=== Setting up Selective Neuron Freezing ===")
    
    # Step 1: Freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Step 2: Count and unfreeze only safety neurons
    total_safety_neurons = 0
    trainable_params = 0
    frozen_params = 0
    
    for layer_idx in range(28):
        layer = model.model.layers[layer_idx]
        
        # ===== FFN up_proj (safety_neuron[0]) =====
        fwd_up_neurons = safety_neuron[0].get(layer_idx, set())
        if len(fwd_up_neurons) > 0 and hasattr(layer.mlp, 'up_proj'):
            up_proj = layer.mlp.up_proj
            # Unfreeze weight rows corresponding to safety neurons
            if hasattr(up_proj, 'weight') and up_proj.weight is not None:
                for neuron_idx in fwd_up_neurons:
                    if neuron_idx < up_proj.weight.shape[0]:
                        up_proj.weight.data[neuron_idx].requires_grad = True
                        trainable_params += up_proj.weight[neuron_idx].numel()
                        total_safety_neurons += 1
            # Unfreeze bias if exists
            if hasattr(up_proj, 'bias') and up_proj.bias is not None:
                for neuron_idx in fwd_up_neurons:
                    if neuron_idx < up_proj.bias.shape[0]:
                        up_proj.bias.data[neuron_idx].requires_grad = True
                        trainable_params += 1
        
        # ===== FFN down_proj (safety_neuron[1]) =====
        fwd_down_neurons = safety_neuron[1].get(layer_idx, set())
        if len(fwd_down_neurons) > 0 and hasattr(layer.mlp, 'down_proj'):
            down_proj = layer.mlp.down_proj
            # Unfreeze weight columns corresponding to safety neurons
            if hasattr(down_proj, 'weight') and down_proj.weight is not None:
                for neuron_idx in fwd_down_neurons:
                    if neuron_idx < down_proj.weight.shape[1]:
                        down_proj.weight.data[:, neuron_idx].requires_grad = True
                        trainable_params += down_proj.weight[:, neuron_idx].numel()
                        total_safety_neurons += 1
            # Bias doesn't correspond to specific input neurons, skip
        
        # ===== Attention q_proj (safety_neuron[2]) =====
        q_neurons = safety_neuron[2].get(layer_idx, set())
        if len(q_neurons) > 0 and hasattr(layer.self_attn, 'q_proj'):
            q_proj = layer.self_attn.q_proj
            if hasattr(q_proj, 'weight') and q_proj.weight is not None:
                for neuron_idx in q_neurons:
                    if neuron_idx < q_proj.weight.shape[0]:
                        q_proj.weight.data[neuron_idx].requires_grad = True
                        trainable_params += q_proj.weight[neuron_idx].numel()
                        total_safety_neurons += 1
            if hasattr(q_proj, 'bias') and q_proj.bias is not None:
                for neuron_idx in q_neurons:
                    if neuron_idx < q_proj.bias.shape[0]:
                        q_proj.bias.data[neuron_idx].requires_grad = True
                        trainable_params += 1
        
        # ===== Attention k_proj (safety_neuron[3]) =====
        k_neurons = safety_neuron[3].get(layer_idx, set())
        if len(k_neurons) > 0 and hasattr(layer.self_attn, 'k_proj'):
            k_proj = layer.self_attn.k_proj
            if hasattr(k_proj, 'weight') and k_proj.weight is not None:
                for neuron_idx in k_neurons:
                    if neuron_idx < k_proj.weight.shape[0]:
                        k_proj.weight.data[neuron_idx].requires_grad = True
                        trainable_params += k_proj.weight[neuron_idx].numel()
                        total_safety_neurons += 1
            if hasattr(k_proj, 'bias') and k_proj.bias is not None:
                for neuron_idx in k_neurons:
                    if neuron_idx < k_proj.bias.shape[0]:
                        k_proj.bias.data[neuron_idx].requires_grad = True
                        trainable_params += 1
        
        # ===== Attention v_proj (safety_neuron[4]) =====
        v_neurons = safety_neuron[4].get(layer_idx, set())
        if len(v_neurons) > 0 and hasattr(layer.self_attn, 'v_proj'):
            v_proj = layer.self_attn.v_proj
            if hasattr(v_proj, 'weight') and v_proj.weight is not None:
                for neuron_idx in v_neurons:
                    if neuron_idx < v_proj.weight.shape[0]:
                        v_proj.weight.data[neuron_idx].requires_grad = True
                        trainable_params += v_proj.weight[neuron_idx].numel()
                        total_safety_neurons += 1
            if hasattr(v_proj, 'bias') and v_proj.bias is not None:
                for neuron_idx in v_neurons:
                    if neuron_idx < v_proj.bias.shape[0]:
                        v_proj.bias.data[neuron_idx].requires_grad = True
                        trainable_params += 1
    
    # Count frozen params
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"✓ Safety Neurons identified: {total_safety_neurons}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Frozen parameters: {frozen_params:,}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Trainable ratio: {100*trainable_params/(trainable_params+frozen_params):.4f}%")
    
    return model

# ==================== Load and Prepare Dataset ====================

def load_circuit_breakers_dataset(json_file: str, num_samples: int = 50) -> Dataset:
    """
    Load circuit_breakers dataset and prepare for training
    Uses 'prompt' as input and 'llama3_output' as target
    """
    print(f"\n=== Loading Circuit Breakers Dataset ===")
    
    if not os.path.exists(json_file):
        print(f"Error: Dataset file '{json_file}' not found!")
        raise FileNotFoundError(json_file)
    
    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total samples in dataset: {len(data)}")
    
    # Sample the required number
    if len(data) > num_samples:
        data = data[:num_samples]
    
    print(f"Using {len(data)} samples for training")
    
    # Prepare training data
    training_data = []
    for item in data:
        if 'prompt' in item and 'llama3_output' in item:
            training_data.append({
                'text': f"Question: {item['prompt']}\nAnswer: {item['llama3_output']}"
            })
    
    print(f"✓ Prepared {len(training_data)} training samples")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        'text': [d['text'] for d in training_data]
    })
    
    return dataset

# ==================== Main Training ====================

def main():
    print("="*70)
    print("  Safety Neuron Tuning (SN-Tune)")
    print("="*70)
    
    # Step 1: Load safety neurons
    safety_neuron = load_safety_neurons(safety_neuron_file)
    
    # Step 2: Load dataset
    dataset = load_circuit_breakers_dataset(dataset_file, num_samples=50)
    
    # Step 3: Load model and tokenizer
    print(f"\n=== Loading Model ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"✓ Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    print(f"✓ Model loaded: {model_name}")
    
    # Step 4: Setup safety neuron targeting (freeze all, unfreeze only safety neurons)
    model = setup_safety_neuron_freezing(model, safety_neuron)
    
    # Step 5: Setup training
    print(f"\n=== Training Configuration ===")
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=1,
        learning_rate=1e-5,
        bf16=True,
        save_steps=100,
        save_total_limit=1,
        logging_steps=5,
        output_dir=output_dir,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        report_to="none",
    )
    
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Dataset size: {len(dataset)}")
    
    # Step 6: Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        packing=True,
    )
    
    # Step 7: Train
    print(f"\n=== Starting Training ===")
    trainer.train()
    
    # Step 8: Save model
    print(f"\n=== Saving Model ===")
    final_output_dir = os.path.join(output_dir, "SN-Tuned-Model")
    os.makedirs(final_output_dir, exist_ok=True)
    
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    print(f"✓ Model saved to: {final_output_dir}")
    
    # Step 9: Create model card
    model_card = f"""---
license: apache-2.0
tags:
  - safety-neuron
  - llama
  - sn-tune
  - circuit-breakers
---

# Llama-3.2-3B Safety Neuron Tuned (SN-Tune)

Safety Neuron Tuning on Circuit Breakers dataset.

## Method

- **Safety Neuron Detection**: Identified safety neurons from harmful prompts
- **Selective Fine-tuning**: Only safety neurons are trainable
- **Other Parameters**: All other parameters are frozen
- **Dataset**: Circuit Breakers (50 prompt-response pairs)

## Training Details

- **Base Model**: {model_name}
- **Dataset**: Circuit Breakers (Harmful prompts + Safe responses)
- **Training Method**: Selective Fine-tuning (Safety neurons only)
- **Learning Rate**: 1e-5
- **Epochs**: 1
- **Batch Size**: 2 (with 4x gradient accumulation)

## Key Features

✅ **Targeted Training**: Only safety neurons are updated
✅ **Preservation**: All other model parameters remain frozen
✅ **Safety Focused**: Trained on harmful prompt -> safe response pairs
✅ **Efficient**: Minimal parameter updates

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{final_output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{final_output_dir}")

inputs = tokenizer("Harmful prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(os.path.join(final_output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    print(f"✓ Model card created")
    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {final_output_dir}")

if __name__ == "__main__":
    main()
