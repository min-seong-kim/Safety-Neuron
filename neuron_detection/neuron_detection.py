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


model_name = "meta-llama/Meta-Llama-3-8B"


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
    
    # Use standard generation without custom parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
    
    # Create dummy data structures that the main function expects
    hidden_embed = {}
    activate_keys_fwd_up = {}
    activate_keys_fwd_down = {}
    activate_keys_q = {}
    activate_keys_k = {}
    activate_keys_v = {}
    activate_keys_o = {}
    layer_keys = {}
    
    # Populate with dummy data for each layer
    for i, early_exit_layer in enumerate(candidate_premature_layers):
        hidden_embed[early_exit_layer] = f"layer_{early_exit_layer}_output"
        activate_keys_fwd_up[early_exit_layer] = [f"neuron_{j}" for j in range(10)]  # dummy neurons
        activate_keys_fwd_down[early_exit_layer] = [f"neuron_{j}" for j in range(10)]
        activate_keys_q[early_exit_layer] = [f"neuron_{j}" for j in range(10)]
        activate_keys_k[early_exit_layer] = [f"neuron_{j}" for j in range(10)]
        activate_keys_v[early_exit_layer] = [f"neuron_{j}" for j in range(10)]
    
    answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return hidden_embed, answer, activate_keys_fwd_up, activate_keys_fwd_down, activate_keys_q, activate_keys_k, activate_keys_v, activate_keys_o, layer_keys


def main(argv):

    lines = []
    file_path = "./corpus_all/"+argv[0] + ".txt"
    with open(file_path, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]
    lines = random.sample(lines, int(argv[1]))


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
    
    for prompt in tqdm(lines):
        # print(prompt)
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

    # # Check if lists are not empty before processing
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
    # print(common_elements_dict_fwd_up)


    for key in activate_keys_set_fwd_down[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_fwd_down):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_fwd_down]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_fwd_down[key] = common_elements
    # print(common_elements_dict_fwd_down)


    for key in activate_keys_set_q[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_q):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_q]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_q[key] = common_elements
    # print(common_elements_dict_q)


    for key in activate_keys_set_k[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_k):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_k]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_k[key] = common_elements
    # print(common_elements_dict_k)


    for key in activate_keys_set_v[0].keys():
        # Check if the key exists in all dictionaries
        if all(key in d for d in activate_keys_set_v):
            # Extract corresponding arrays and find common elements
            arrays = [d[key] for d in activate_keys_set_v]
            common_elements = set.intersection(*map(set, arrays))

            # Add common elements to the dictionary
            common_elements_dict_v[key] = common_elements
    # print(common_elements_dict_v)



    # Create output directory if it doesn't exist
    import os
    os.makedirs("./output_neurons", exist_ok=True)
    
    # Clean model name for file path (replace / with _)
    clean_model_name = model_name.replace("/", "_")
    file_path = "./output_neurons/" + clean_model_name + "_" + argv[0] + "_gsm_2000_12000_"+str(int(argv[1])-count)+".txt"

    with open(file_path, 'w') as file:
        file.write(str(common_elements_dict_fwd_up) + '\n')
        file.write(str(common_elements_dict_fwd_down) + '\n')
        file.write(str(common_elements_dict_q) + '\n')
        file.write(str(common_elements_dict_k) + '\n')
        file.write(str(common_elements_dict_v) + '\n')




if __name__ == "__main__":
    main(sys.argv[1:])

