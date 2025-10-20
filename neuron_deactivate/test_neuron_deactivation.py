#!/usr/bin/env python3
"""
Test script for neuron deactivation with Meta-Llama-3-8B
Usage: python test_neuron_deactivation.py
"""

import sys
import os

# Test with simplified parameters
if __name__ == "__main__":
    # Simple test parameters - using 'zh' (Chinese) as it's supported
    test_args = [
        "zh",     # language (chinese - supported language) 
        "10",     # understanding_layer
        "20",     # generation_layer  
        "5",      # attn_deact_number
        "5",      # ffn_deact_number
        "True",   # under_attn
        "False",  # reason_attn
        "True",   # gen_attn
        "False",  # under_ffn
        "True",   # reason_ffn
        "False"   # gen_ffn
    ]
    
    print("Starting neuron deactivation test...")
    print(f"Arguments: {test_args}")
    
    # Import and run the main function
    try:
        from test_mistral_gsm import main
        main(test_args)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()