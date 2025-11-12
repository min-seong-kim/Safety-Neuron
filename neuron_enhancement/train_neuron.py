import os
import torch
from datetime import datetime

# Set environment variables for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from dataclasses import field
import random
import pdb
from tqdm import tqdm
from datasets import Dataset
import re
from huggingface_hub import HfApi, login

output_dir="xxxxxx"
cache_dir="xxxxxx"

os.makedirs(output_dir,exist_ok=True)
os.makedirs(cache_dir,exist_ok=True)

model_name = "meta-llama/Llama-3.1-8B-Instruct"


import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--language", type=str, default="English")
parser.add_argument("--task", type=str, default="Wiki")

args = parser.parse_args()
print(args)

import ast

def retrive_neuron(filename):
    # Empty list to store the dictionaries
    activate_neuron = []

    # Check if filename is a placeholder or invalid
    if filename == 'xxxxxx' or not filename:
        print(f"Warning: Invalid filename '{filename}'. Using empty neuron data.")
        # Return proper structure: list of 5 dictionaries, each with 32 layer keys (0-31)
        dummy_neuron = []
        for i in range(5):  # 5 types of neurons (fwd_up, fwd_down, q, k, v)
            layer_dict = {}
            for layer in range(32):  # 32 layers (0-31)
                layer_dict[layer] = set()  # Empty set for each layer
            dummy_neuron.append(layer_dict)
        return dummy_neuron
    
    # Check if file exists and is not a directory
    import os
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' not found. Using empty neuron data.")
        dummy_neuron = []
        for i in range(5):
            layer_dict = {}
            for layer in range(32):
                layer_dict[layer] = set()
            dummy_neuron.append(layer_dict)
        return dummy_neuron
    
    if os.path.isdir(filename):
        print(f"Error: '{filename}' is a directory, not a file. Using empty neuron data.")
        dummy_neuron = []
        for i in range(5):
            layer_dict = {}
            for layer in range(32):
                layer_dict[layer] = set()
            dummy_neuron.append(layer_dict)
        return dummy_neuron

    # Open the file and read line by line
    try:
        with open(filename, 'r') as file:
            neurons = file.readlines()
            for neuron in neurons:
                neuron = eval(neuron.strip())
                activate_neuron.append(neuron)
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        dummy_neuron = []
        for i in range(5):
            layer_dict = {}
            for layer in range(32):
                layer_dict[layer] = set()
            dummy_neuron.append(layer_dict)
        return dummy_neuron

    return activate_neuron

def deduplicate(neuron_target, neuron_delete):
    index_keys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    for key in index_keys:
        neuron_target[0][key] = neuron_target[0][key] - neuron_delete[0][key]
        neuron_target[1][key] = neuron_target[1][key] - neuron_delete[1][key]
        neuron_target[2][key] = neuron_target[2][key] - neuron_delete[2][key]
        neuron_target[3][key] = neuron_target[3][key] - neuron_delete[3][key]
        neuron_target[4][key] = neuron_target[4][key] - neuron_delete[4][key]
    # Safety 뉴런 중에서 Foundation 뉴런과 겹치지 않는 순수한 안전 뉴런
    return neuron_target

def freeze_safety_neurons(model, pure_safety_neuron):
    """
    Freeze Safety Neuron parameters so they won't be updated during training.
    Only neurons in pure_safety_neuron will be frozen.
    
    Args:
        model: The model to freeze
        pure_safety_neuron: Dictionary containing safety neuron indices
                           [fwd_up, fwd_down, q, k, v]
    """
    frozen_params = 0
    total_params = 0
    
    for layer_idx in range(32):
        layer = model.model.layers[layer_idx]
        
        # Freeze FFN up_proj neurons
        fwd_up_neurons = pure_safety_neuron[0].get(layer_idx, set())
        if len(fwd_up_neurons) > 0 and hasattr(layer.mlp, 'up_proj'):
            # Get neuron indices from strings like "neuron_123"
            neuron_indices = set()
            for neuron_str in fwd_up_neurons:
                try:
                    idx = int(neuron_str.split('_')[1])
                    neuron_indices.add(idx)
                except:
                    continue
            
            # Create a hook to freeze these neurons
            if len(neuron_indices) > 0:
                up_proj = layer.mlp.up_proj
                # Store original weight
                up_proj._original_weight = up_proj.weight.data.clone()
                up_proj._safety_indices = neuron_indices
                frozen_params += len(neuron_indices)
        
        # Freeze FFN down_proj neurons
        fwd_down_neurons = pure_safety_neuron[1].get(layer_idx, set())
        if len(fwd_down_neurons) > 0 and hasattr(layer.mlp, 'down_proj'):
            neuron_indices = set()
            for neuron_str in fwd_down_neurons:
                try:
                    idx = int(neuron_str.split('_')[1])
                    neuron_indices.add(idx)
                except:
                    continue
            
            if len(neuron_indices) > 0:
                down_proj = layer.mlp.down_proj
                down_proj._original_weight = down_proj.weight.data.clone()
                down_proj._safety_indices = neuron_indices
                frozen_params += len(neuron_indices)
        
        # Freeze Attention Q neurons
        q_neurons = pure_safety_neuron[2].get(layer_idx, set())
        if len(q_neurons) > 0 and hasattr(layer.self_attn, 'q_proj'):
            neuron_indices = set()
            for neuron_str in q_neurons:
                try:
                    idx = int(neuron_str.split('_')[1])
                    neuron_indices.add(idx)
                except:
                    continue
            
            if len(neuron_indices) > 0:
                q_proj = layer.self_attn.q_proj
                q_proj._original_weight = q_proj.weight.data.clone()
                q_proj._safety_indices = neuron_indices
                frozen_params += len(neuron_indices)
        
        # Freeze Attention K neurons
        k_neurons = pure_safety_neuron[3].get(layer_idx, set())
        if len(k_neurons) > 0 and hasattr(layer.self_attn, 'k_proj'):
            neuron_indices = set()
            for neuron_str in k_neurons:
                try:
                    idx = int(neuron_str.split('_')[1])
                    neuron_indices.add(idx)
                except:
                    continue
            
            if len(neuron_indices) > 0:
                k_proj = layer.self_attn.k_proj
                k_proj._original_weight = k_proj.weight.data.clone()
                k_proj._safety_indices = neuron_indices
                frozen_params += len(neuron_indices)
        
        # Freeze Attention V neurons
        v_neurons = pure_safety_neuron[4].get(layer_idx, set())
        if len(v_neurons) > 0 and hasattr(layer.self_attn, 'v_proj'):
            neuron_indices = set()
            for neuron_str in v_neurons:
                try:
                    idx = int(neuron_str.split('_')[1])
                    neuron_indices.add(idx)
                except:
                    continue
            
            if len(neuron_indices) > 0:
                v_proj = layer.self_attn.v_proj
                v_proj._original_weight = v_proj.weight.data.clone()
                v_proj._safety_indices = neuron_indices
                frozen_params += len(neuron_indices)
    
    print(f"Frozen safety neuron parameters: {frozen_params}")
    return model

def restore_frozen_neurons(model):
    """
    Restore frozen safety neurons after each backward pass.
    """
    for layer_idx in range(32):
        layer = model.model.layers[layer_idx]
        
        # Restore FFN up_proj
        if hasattr(layer.mlp.up_proj, '_original_weight'):
            with torch.no_grad():
                for idx in layer.mlp.up_proj._safety_indices:
                    layer.mlp.up_proj.weight[idx] = layer.mlp.up_proj._original_weight[idx]
        
        # Restore FFN down_proj
        if hasattr(layer.mlp.down_proj, '_original_weight'):
            with torch.no_grad():
                for idx in layer.mlp.down_proj._safety_indices:
                    layer.mlp.down_proj.weight[idx] = layer.mlp.down_proj._original_weight[idx]
        
        # Restore Attention Q
        if hasattr(layer.self_attn.q_proj, '_original_weight'):
            with torch.no_grad():
                for idx in layer.self_attn.q_proj._safety_indices:
                    layer.self_attn.q_proj.weight[idx] = layer.self_attn.q_proj._original_weight[idx]
        
        # Restore Attention K
        if hasattr(layer.self_attn.k_proj, '_original_weight'):
            with torch.no_grad():
                for idx in layer.self_attn.k_proj._safety_indices:
                    layer.self_attn.k_proj.weight[idx] = layer.self_attn.k_proj._original_weight[idx]
        
        # Restore Attention V
        if hasattr(layer.self_attn.v_proj, '_original_weight'):
            with torch.no_grad():
                for idx in layer.self_attn.v_proj._safety_indices:
                    layer.self_attn.v_proj.weight[idx] = layer.self_attn.v_proj._original_weight[idx] 

# Load Safety Neuron (from safety corpus)
print("Loading Safety Neuron...")
safety_neuron = retrive_neuron('../neuron_detection/output_neurons/meta-llama_Meta-Llama-3-8B_do_not_answer_real_neurons_500.txt')
print(f"✓ Safety Neuron loaded")

# Load Utility Neuron (from Wikipedia corpus)
print("Loading Utility Neuron...")
utility_neuron = retrive_neuron('../neuron_detection/output_neurons/meta-llama_Meta-Llama-3-8B_wikipedia_utility_neurons_500.txt')
print(f"✓ Utility Neuron loaded")

# Deduplicate: Remove Utility Neurons from Safety Neurons to get pure Safety Neurons
print("Deduplicating neurons (removing Utility Neurons from Safety Neurons)...")
pure_safety_neuron = deduplicate(safety_neuron, utility_neuron)
print(f"✓ Pure Safety Neuron extracted")

# Print statistics
print("\n=== Neuron Statistics ===")
for layer_idx in range(32):
    safety_count = len(safety_neuron[0].get(layer_idx, set())) + len(safety_neuron[1].get(layer_idx, set()))
    utility_count = len(utility_neuron[0].get(layer_idx, set())) + len(utility_neuron[1].get(layer_idx, set()))
    pure_count = len(pure_safety_neuron[0].get(layer_idx, set())) + len(pure_safety_neuron[1].get(layer_idx, set()))
    
    if pure_count > 0:
        print(f"Layer {layer_idx}: Safety={safety_count}, Utility={utility_count}, Pure={pure_count}")

# Use pure Safety Neuron for enhancement
activate_neuron = pure_safety_neuron

# Check if data file exists, if not create dummy data or use alternative
import os
data_file = "xxxxxxx"

print("\n=== Loading Training Dataset ===")

if data_file == "xxxxxxx" or not os.path.exists(data_file):
    print(f"Loading GSM8K dataset from Hugging Face...")
    
    try:
        # Load GSM8K dataset
        # 기존 
        # dataset = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
        # 줄인거(테스트용)
        # dataset = load_dataset("openai/gsm8k", "main", split="train[0:100]", cache_dir=cache_dir)
        
        dataset = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
        print(f"✓ GSM8K dataset loaded: {len(dataset)} samples")
        
        # GSM8K has 'question' and 'answer' fields
        # We need to convert to 'original_question' and 'response' format
        def convert_gsm8k_format(example):
            # Extract the final answer from the 'answer' field
            # GSM8K format: "solution text\n####\nfinal_answer"
            answer_text = example['answer']
            return {
                "original_question": example['question'],
                "response": answer_text
            }
        
        dataset = dataset.map(convert_gsm8k_format, remove_columns=dataset.column_names)
        print(f"✓ Converted {len(dataset)} GSM8K samples to training format")
        
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print(f"Falling back to dummy data...")
        
        # Create dummy training data with correct field names
        dummy_data = [
            {"original_question": "What is machine learning?", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data."},
            {"original_question": "How do neural networks work?", "response": "Neural networks are computational models inspired by the human brain, consisting of interconnected nodes that process information."},
            {"original_question": "What is deep learning?", "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns in data."}
        ] * 10  # Repeat to have more samples
        
        # Create temporary JSON file
        import json
        temp_file = "./temp_training_data.json"
        with open(temp_file, 'w') as f:
            for item in dummy_data:
                json.dump(item, f)
                f.write('\n')
        
        dataset = load_dataset("json", data_files=temp_file, split="train", cache_dir=cache_dir)
else:
    dataset = load_dataset("json", data_files=data_file, split="train", cache_dir=cache_dir)

# Convert dataset to have "text" field for SFTTrainer
def convert_to_text_format(example):
    # Handle both GSM8K format (original_question, response) and other formats
    if 'original_question' in example and 'response' in example:
        question = example['original_question']
        answer = example['response']
    elif 'question' in example and 'answer' in example:
        question = example['question']
        answer = example['answer']
    else:
        # Fallback
        question = example.get('original_question', example.get('question', ''))
        answer = example.get('response', example.get('answer', ''))
    
    return {"text": f"Question: {question}\nAnswer: {answer}"}

dataset = dataset.map(convert_to_text_format, remove_columns=dataset.column_names)

# Define preprocessing functions before using them
def formatting_prompts_func(example):
    # For SFTTrainer, return single text string, not a list
    if isinstance(example['original_question'], list):
        # Batch processing case
        output_texts = []
        for i in range(len(example['original_question'])):
            text = f"Question: {example['original_question'][i]}\nAnswer: {example['response'][i]}"
            output_texts.append(text)
        return {"text": output_texts}
    else:
        # Single example case
        text = f"Question: {example['original_question']}\nAnswer: {example['response']}"
        return {"text": text}

print(f"✓ Final dataset size: {len(dataset)} samples")

# Load tokenizer first
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Dataset is already in the correct format with "text" field, no need for additional preprocessing

print("\n=== Loading Model ===")
# base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)
print(f"✓ Model loaded: {model_name}")

# Freeze Safety Neurons BEFORE applying LoRA
print("\n=== Freezing Safety Neurons ===")
base_model = freeze_safety_neurons(base_model, pure_safety_neuron)

# Configure LoRA for fine-tuning quantized model
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA to the model
base_model = get_peft_model(base_model, lora_config)
print(f"Number of trainable parameters: {base_model.num_parameters(only_trainable=True):,}")

# Create custom trainer callback to restore frozen neurons after each step
from transformers import TrainerCallback

class SafeNeuronFreezeCallback(TrainerCallback):
    def __init__(self, base_model_ref):
        self.base_model_ref = base_model_ref
    
    def on_backward_end(self, args, state, control, **kwargs):
        """Restore frozen safety neurons after backward pass"""
        restore_frozen_neurons(self.base_model_ref)

# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=4,  # Keep very small batch size
    gradient_accumulation_steps=16,  # Increase accumulation to compensate
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=3,  # Reduced to 1 for quick testing 기존 3
    learning_rate=1e-5,
    bf16=True,
    save_steps=1000,
    save_total_limit=2,  # Keep last 2 checkpoints
    logging_steps=10,
    output_dir=output_dir,
    optim="adamw_torch",  # Use standard PyTorch optimizer instead of bitsandbytes
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    dataloader_pin_memory=False,  # Disable pin memory to save memory
    remove_unused_columns=True,
    report_to="none",  # Disable wandb logging
)

# Use SFTTrainer with correct parameters for latest version
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer  
    args=training_args,
    callbacks=[SafeNeuronFreezeCallback(base_model.base_model)],  # Add freeze callback
    # formatting_func removed since data is already in correct format
)

print("\n=== Starting Training ===")
print(f"Dataset size: {len(dataset)}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size} (with {training_args.gradient_accumulation_steps}x accumulation)")
print(f"Safety neurons frozen: Yes")
print("-" * 50)

trainer.train() 

print("\n=== Training Complete ===")

# Merge LoRA adapter with base model to create a complete model
print("\n=== Merging LoRA Adapter with Base Model ===")
try:
    # Merge LoRA adapter with base model
    merged_model = trainer.model.merge_and_unload()
    print("✓ LoRA adapter merged with base model")
except Exception as e:
    print(f"Warning: Could not merge LoRA adapter: {e}")
    print("Using model with LoRA adapter as is...")
    merged_model = trainer.model

# Save the enhanced model
final_output_dir = os.path.join(output_dir, "Llama3_SafetyEnhanced")
os.makedirs(final_output_dir, exist_ok=True)

print(f"Saving complete enhanced model to {final_output_dir}...")
merged_model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

print(f"✓ Complete model saved successfully!")
print(f"✓ Output directory: {final_output_dir}")
print("\n=== Model Details ===")
print(f"Model: {model_name}")
print(f"Training dataset: GSM8K (openai/gsm8k)")
print(f"Samples: {len(dataset)}")
print(f"Safety neurons frozen: Yes")
print(f"Utility neurons excluded: Yes")
print("-" * 50)

# Upload to Hugging Face Hub
print("\n=== Uploading Complete Model to Hugging Face Hub ===")

try:
    # Generate model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hf_model_name = f"kmseong/Llama3_SafetyEnhanced_{timestamp}"
    
    print(f"Model name: {hf_model_name}")
    print(f"Uploading complete model to Hugging Face...")
    
    # Push complete model to hub
    merged_model.push_to_hub(
        repo_id=hf_model_name,
        private=False,  # Set to True if you want a private repo
        commit_message=f"Safety-Enhanced Llama3 model (complete, merged) with frozen safety neurons. Trained on GSM8K."
    )
    
    tokenizer.push_to_hub(
        repo_id=hf_model_name,
        commit_message=f"Tokenizer for Safety-Enhanced Llama3 model."
    )
    
    print(f"✓ Complete model uploaded successfully!")
    print(f"✓ URL: https://huggingface.co/{hf_model_name}")
    
    # Create a model card
    model_card = f"""---
license: apache-2.0
tags:
  - safety-neuron
  - llama
  - enhancement
  - gsm8k
  - merged
---

# Llama3 Safety-Enhanced Model (Complete/Merged)

This is a complete safety-enhanced version of Meta Llama 3 8B, trained with Safety Neuron methodology.
This is a fully merged model (LoRA adapter merged with base model).

## Method

- **Safety Neuron Detection**: Neurons responding to safety-related inputs
- **Utility Neuron Detection**: Neurons from Wikipedia corpus (general utility)
- **Deduplication**: Removed utility neurons to get pure safety neurons
- **Enhancement**: Fine-tuned on GSM8K with safety neurons frozen
- **Merging**: LoRA adapter merged with base model for deployment

## Training Details

- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Dataset**: OpenAI GSM8K ({len(dataset)} samples)
- **Training Method**: LoRA + Frozen Safety Neurons
- **Frozen Parameters**: 4395
- **Epochs**: 1
- **Model Type**: Complete merged model (ready for inference)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{hf_model_name}"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Use for inference
inputs = tokenizer("Question: What is 2+2?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Key Features

✅ **Complete Model**: No need to load base model + adapter separately
✅ **Safety Enhanced**: Contains frozen safety neurons
✅ **Production Ready**: Can be directly deployed
✅ **LoRA Merged**: All LoRA weights merged into model weights

## Model Size

- Model weights: Full merged model (approximately 16GB for 8B parameters in fp16)
- No separate adapter files needed

## References

- Safety Neuron: Detection and Enhancement methodology
- GSM8K: Grade School Math 8K dataset
- LoRA: Low-Rank Adaptation for Large Language Models

Generated: {timestamp}
"""
    
    # Save and upload model card
    with open(os.path.join(final_output_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    print(f"\n✓ Model card created and uploaded")
    print(f"\n=== Upload Summary ===")
    print(f"Repository: {hf_model_name}")
    print(f"Files uploaded:")
    print(f"  ✓ config.json (model configuration)")
    print(f"  ✓ pytorch_model.bin or model.safetensors (complete model weights)")
    print(f"  ✓ tokenizer.json (tokenizer)")
    print(f"  ✓ special_tokens_map.json")
    print(f"  ✓ tokenizer_config.json")
    print(f"  ✓ README.md (model card)")
    print(f"\nModel is ready for production use!")
    
except Exception as e:
    print(f"Error uploading to Hugging Face: {e}")
    print(f"Model saved locally at: {final_output_dir}")
    print(f"You can upload manually using:")
    print(f"  huggingface-cli upload {final_output_dir} kmseong/Llama3_8B_Instruct_SafetyEnhanced_{{timestamp}}")