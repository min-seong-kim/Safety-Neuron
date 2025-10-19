import os
import torch

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
# os.environ["WANDB_DISABLED"] = "true"

output_dir="xxxxxx"
cache_dir="xxxxxx"

os.makedirs(output_dir,exist_ok=True)
os.makedirs(cache_dir,exist_ok=True)

model_name = "meta-llama/Meta-Llama-3-8B"


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

    return neuron_target


activate_neuron = retrive_neuron('xxxxxx')

# Check if data file exists, if not create dummy data or use alternative
import os
data_file = "xxxxxxx"
if data_file == "xxxxxxx" or not os.path.exists(data_file):
    print(f"Warning: Data file '{data_file}' not found. Creating dummy dataset.")
    
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
    return {"text": f"Question: {example['original_question']}\nAnswer: {example['response']}"}

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

# Load tokenizer first
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Dataset is already in the correct format with "text" field, no need for additional preprocessing

# base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

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

# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Keep very small batch size
    gradient_accumulation_steps=16,  # Increase accumulation to compensate
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=1, 
    learning_rate=2e-6,
    bf16=True,
    save_steps=500,
    save_total_limit=0,
    logging_steps=10,
    output_dir=output_dir,
    optim="adamw_torch",  # Use standard PyTorch optimizer instead of bitsandbytes
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    dataloader_pin_memory=False,  # Disable pin memory to save memory
    remove_unused_columns=True,
)

# Use SFTTrainer with correct parameters for latest version
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    processing_class=tokenizer,  # Use processing_class instead of tokenizer  
    args=training_args,
    # formatting_func removed since data is already in correct format
)

trainer.train() 

output_dir = os.path.join(output_dir, "Llama3_Reason")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)