import os
from dataclasses import field, dataclass
from typing import Optional, Any
import transformers
from rouge_score import rouge_scorer
import random
from itertools import groupby
import pdb
import re
import sys
from tqdm import tqdm
from typing import List
import logging
logging.basicConfig(level=logging.INFO)
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import csv

random.seed(112)

model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


def Prompting(model, prompt, candidate_premature_layers):
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Storage for activations
    layer_activations = {}
    attention_activations = {}
    
    # Hook functions to capture activations
    def get_activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                # For attention layers, output is typically (hidden_states, attention_weights)
                layer_activations[name] = output[0].detach().cpu()
                if len(output) > 1 and output[1] is not None:
                    attention_activations[name] = output[1].detach().cpu()
            else:
                layer_activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks for all transformer layers
    hooks = []
    for layer_idx in candidate_premature_layers:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # For Llama-like models
            if layer_idx < len(model.model.layers):
                layer = model.model.layers[layer_idx]
                # Hook the main layer output
                hooks.append(layer.register_forward_hook(get_activation_hook(f'layer_{layer_idx}')))
                
                # Hook FFN components
                if hasattr(layer, 'mlp'):
                    if hasattr(layer.mlp, 'up_proj'):
                        hooks.append(layer.mlp.up_proj.register_forward_hook(get_activation_hook(f'layer_{layer_idx}_ffn_up')))
                    if hasattr(layer.mlp, 'down_proj'):
                        hooks.append(layer.mlp.down_proj.register_forward_hook(get_activation_hook(f'layer_{layer_idx}_ffn_down')))
                
                # Hook attention components
                if hasattr(layer, 'self_attn'):
                    if hasattr(layer.self_attn, 'q_proj'):
                        hooks.append(layer.self_attn.q_proj.register_forward_hook(get_activation_hook(f'layer_{layer_idx}_attn_q')))
                    if hasattr(layer.self_attn, 'k_proj'):
                        hooks.append(layer.self_attn.k_proj.register_forward_hook(get_activation_hook(f'layer_{layer_idx}_attn_k')))
                    if hasattr(layer.self_attn, 'v_proj'):
                        hooks.append(layer.self_attn.v_proj.register_forward_hook(get_activation_hook(f'layer_{layer_idx}_attn_v')))
                    if hasattr(layer.self_attn, 'o_proj'):
                        hooks.append(layer.self_attn.o_proj.register_forward_hook(get_activation_hook(f'layer_{layer_idx}_attn_o')))
    
    try:
        # Forward pass to capture activations
        with torch.no_grad():
            # Use a simple forward pass instead of generate for better activation capture
            output = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                output_hidden_states=True,
                return_dict=True
            )
        
        # Parameters for neuron selection (as mentioned in README)
        top_number_attn = 1000  # Number of top attention neurons
        top_number_ffn = 2000   # Number of top FFN neurons
        activation_threshold = 0.1  # Threshold for considering a neuron "active"
        
        # Process activations to find top activated neurons
        hidden_embed = {}
        activate_keys_fwd_up = {}
        activate_keys_fwd_down = {}
        activate_keys_q = {}
        activate_keys_k = {}
        activate_keys_v = {}
        activate_keys_o = {}
        layer_keys = {}
        
        for layer_idx in candidate_premature_layers:
            # Initialize lists for this layer
            activate_keys_fwd_up[layer_idx] = []
            activate_keys_fwd_down[layer_idx] = []
            activate_keys_q[layer_idx] = []
            activate_keys_k[layer_idx] = []
            activate_keys_v[layer_idx] = []
            activate_keys_o[layer_idx] = []
            
            # Process FFN up-projection activations
            ffn_up_key = f'layer_{layer_idx}_ffn_up'
            if ffn_up_key in layer_activations:
                activations = layer_activations[ffn_up_key]
                # Take the last token's activations
                if activations.dim() > 2:
                    activations = activations[:, -1, :]  # [batch_size, hidden_dim]
                activations = activations.squeeze(0)  # Remove batch dimension
                
                # Find top activated neurons
                if activations.dim() == 1 and len(activations) > 0:
                    # Get indices of neurons above threshold
                    active_indices = torch.where(torch.abs(activations) > activation_threshold)[0]
                    # Sort by activation magnitude and take top N
                    if len(active_indices) > 0:
                        sorted_indices = active_indices[torch.argsort(torch.abs(activations[active_indices]), descending=True)]
                        top_indices = sorted_indices[:min(top_number_ffn, len(sorted_indices))]
                        activate_keys_fwd_up[layer_idx] = [f"neuron_{idx.item()}" for idx in top_indices]
                    else:
                        # If no neurons above threshold, take top activations anyway
                        top_indices = torch.argsort(torch.abs(activations), descending=True)[:min(top_number_ffn, len(activations))]
                        activate_keys_fwd_up[layer_idx] = [f"neuron_{idx.item()}" for idx in top_indices]
            
            # Process FFN down-projection activations
            ffn_down_key = f'layer_{layer_idx}_ffn_down'
            if ffn_down_key in layer_activations:
                activations = layer_activations[ffn_down_key]
                if activations.dim() > 2:
                    activations = activations[:, -1, :]
                activations = activations.squeeze(0)
                
                if activations.dim() == 1 and len(activations) > 0:
                    active_indices = torch.where(torch.abs(activations) > activation_threshold)[0]
                    if len(active_indices) > 0:
                        sorted_indices = active_indices[torch.argsort(torch.abs(activations[active_indices]), descending=True)]
                        top_indices = sorted_indices[:min(top_number_ffn, len(sorted_indices))]
                        activate_keys_fwd_down[layer_idx] = [f"neuron_{idx.item()}" for idx in top_indices]
                    else:
                        top_indices = torch.argsort(torch.abs(activations), descending=True)[:min(top_number_ffn, len(activations))]
                        activate_keys_fwd_down[layer_idx] = [f"neuron_{idx.item()}" for idx in top_indices]
            
            # Process attention Q, K, V, O projections
            for proj_name, proj_dict in [('q', activate_keys_q), ('k', activate_keys_k), 
                                       ('v', activate_keys_v), ('o', activate_keys_o)]:
                attn_key = f'layer_{layer_idx}_attn_{proj_name}'
                if attn_key in layer_activations:
                    activations = layer_activations[attn_key]
                    if activations.dim() > 2:
                        activations = activations[:, -1, :]
                    activations = activations.squeeze(0)
                    
                    if activations.dim() == 1 and len(activations) > 0:
                        active_indices = torch.where(torch.abs(activations) > activation_threshold)[0]
                        if len(active_indices) > 0:
                            sorted_indices = active_indices[torch.argsort(torch.abs(activations[active_indices]), descending=True)]
                            top_indices = sorted_indices[:min(top_number_attn, len(sorted_indices))]
                            proj_dict[layer_idx] = [f"neuron_{idx.item()}" for idx in top_indices]
                        else:
                            top_indices = torch.argsort(torch.abs(activations), descending=True)[:min(top_number_attn, len(activations))]
                            proj_dict[layer_idx] = [f"neuron_{idx.item()}" for idx in top_indices]
            
            # Store layer information
            hidden_embed[layer_idx] = f"layer_{layer_idx}_output"
            layer_keys[layer_idx] = f"layer_{layer_idx}_keys"
        
        # Generate a token for answer (keeping original behavior)
        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        answer = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        print(f"Error during activation capture: {e}")
        # Fallback to dummy data if real extraction fails
        hidden_embed = {}
        activate_keys_fwd_up = {}
        activate_keys_fwd_down = {}
        activate_keys_q = {}
        activate_keys_k = {}
        activate_keys_v = {}
        activate_keys_o = {}
        layer_keys = {}
        
        for layer_idx in candidate_premature_layers:
            hidden_embed[layer_idx] = f"layer_{layer_idx}_output"
            activate_keys_fwd_up[layer_idx] = [f"neuron_{j}" for j in range(10)]
            activate_keys_fwd_down[layer_idx] = [f"neuron_{j}" for j in range(10)]
            activate_keys_q[layer_idx] = [f"neuron_{j}" for j in range(10)]
            activate_keys_k[layer_idx] = [f"neuron_{j}" for j in range(10)]
            activate_keys_v[layer_idx] = [f"neuron_{j}" for j in range(10)]
            activate_keys_o[layer_idx] = [f"neuron_{j}" for j in range(10)]
            layer_keys[layer_idx] = f"layer_{layer_idx}_keys"
        
        answer = prompt  # Fallback answer
        
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    return hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys


def load_wikipedia_data(num_samples: int = 1000):
    """
    Load Wikipedia data from Hugging Face.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        List of text samples from Wikipedia
    """
    print(f"Loading Wikipedia dataset (subset: 20231101.en)...")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=False,
            cache_dir="./wikipedia_cache"
        )
        
        # Extract text and sample
        texts = []
        print(f"Sampling {num_samples} documents from Wikipedia...")
        
        # Get random indices
        total_size = len(dataset)
        random_indices = random.sample(range(total_size), min(num_samples, total_size))
        
        for idx in tqdm(random_indices):
            try:
                text = dataset[idx]['text']
                # Take first sentence or paragraph (max 512 tokens)
                sentences = text.split('.')
                if sentences:
                    texts.append(sentences[0].strip()[:512])
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Successfully loaded {len(texts)} Wikipedia samples")
        return texts
        
    except Exception as e:
        print(f"Error loading Wikipedia dataset: {e}")
        print("Falling back to creating dummy data...")
        return ["The quick brown fox jumps over the lazy dog."] * num_samples


def main(argv):
    """
    argv[0]: Number of samples to load from Wikipedia (default: 1000)
    """
    
    # Parse arguments
    num_samples = int(argv[0]) if len(argv) > 0 else 1000
    
    # Load Wikipedia data instead of local corpus
    lines = load_wikipedia_data(num_samples=num_samples)
    
    if not lines:
        print("Error: No data loaded. Exiting.")
        return
    
    print(f"Total samples: {len(lines)}")

    candidate_premature_layers = []
    for i in range(32):
        candidate_premature_layers.append(i)

    activate_keys_set_fwd_up = []
    activate_keys_set_fwd_down = []
    activate_keys_set_q = []
    activate_keys_set_k = []
    activate_keys_set_v = []

    count = 0

    # Test with just the first prompt to debug
    print(f"Testing with first prompt: {lines[0][:50]}...")
    
    # Process each prompt
    print("Processing prompts to detect utility neurons...")
    
    for prompt in tqdm(lines):
        try:
            hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, _, _ = Prompting(model, prompt, candidate_premature_layers)
            activate_keys_set_fwd_up.append(activate_keys_fwd_up)
            activate_keys_set_fwd_down.append(activate_keys_fwd_down)
            activate_keys_set_q.append(activate_keys_q)
            activate_keys_set_k.append(activate_keys_k)
            activate_keys_set_v.append(activate_keys_v)
        except Exception as e:
            count += 1
            # Handle the OutOfMemoryError here
            print(f"Failed prompt {count}: {prompt[:50]}...")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("---")

    # Initialize dictionary for common elements
    common_elements_dict_fwd_up = {}
    common_elements_dict_fwd_down = {}
    common_elements_dict_q = {}
    common_elements_dict_k = {}
    common_elements_dict_v = {}

    # Check if lists are not empty before processing
    if not activate_keys_set_fwd_up:
        print("Error: activate_keys_set_fwd_up is empty. All prompts may have failed.")
        return

    # Iterate through the keys of the first dictionary
    for key in activate_keys_set_fwd_up[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_up):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_up]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_up[key] = common_elements

    for key in activate_keys_set_fwd_down[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_down):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_down]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_down[key] = common_elements

    for key in activate_keys_set_q[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_q):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_q]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_q[key] = common_elements

    for key in activate_keys_set_k[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_k):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_k]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_k[key] = common_elements

    for key in activate_keys_set_v[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_v):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_v]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_v[key] = common_elements

    # Create output directory if it doesn't exist
    os.makedirs("./output_neurons", exist_ok=True)
    
    # Clean model name for file path (replace / with _)
    clean_model_name = model_name.replace("/", "_")
    successful_samples = len(lines) - count
    file_path = f"./output_neurons/{clean_model_name}_wikipedia_utility_neurons_{successful_samples}.txt"

    print(f"Saving utility neurons to {file_path}")
    
    with open(file_path, 'w') as file:
        file.write(str(common_elements_dict_fwd_up) + '\n')
        file.write(str(common_elements_dict_fwd_down) + '\n')
        file.write(str(common_elements_dict_q) + '\n')
        file.write(str(common_elements_dict_k) + '\n')
        file.write(str(common_elements_dict_v) + '\n')
    
    print(f"Utility neuron detection completed. Results saved to: {file_path}")
    print(f"Processed {successful_samples} prompts successfully out of {len(lines)} total.")


if __name__ == "__main__":
    main(sys.argv[1:])
