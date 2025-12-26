"""
Upload SN-Tuned Model to Hugging Face Hub

Usage:
    python upload_sn_tuned_model.py <model_local_path>

Example:
    python upload_sn_tuned_model.py ./sn_tuned_model_20251208_223350
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# Configuration
# =====================================================================
HF_USERNAME = "kmseong"
MODEL_NAME_PREFIX = "Llama-3.2-3B-Instruct-SN-Tune"


def get_model_name_with_timestamp():
    """Generate model name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_NAME_PREFIX}_{timestamp}"


def upload_to_huggingface(model_path):
    """
    Upload SN-tuned model to Hugging Face Hub
    
    Args:
        model_path: Local path to the model directory
    """
    
    # Verify model path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    # Check for required files
    required_files = ['config.json', 'generation_config.json']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            logger.warning(f"Warning: {file} not found in {model_path}")
    
    # Generate model name with timestamp
    model_name = get_model_name_with_timestamp()
    repo_id = f"{HF_USERNAME}/{model_name}"
    
    logger.info(f"\n{'='*70}")
    logger.info("Uploading SN-Tuned Model to Hugging Face Hub")
    logger.info(f"{'='*70}")
    logger.info(f"Local model path: {model_path}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Model name: {model_name}")
    
    try:
        # Step 1: Authenticate with Hugging Face
        logger.info("\n[Step 1] Authenticating with Hugging Face...")
        try:
            # Try to use existing cached token
            api = HfApi()
            # This will use the token from ~/.huggingface/token if it exists
            logger.info("✓ Using cached Hugging Face token")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.info("\nPlease login to Hugging Face:")
            logger.info("  Run: huggingface-cli login")
            logger.info("  Or set HUGGINGFACE_TOKEN environment variable")
            sys.exit(1)
        
        # Step 2: Load model and tokenizer locally to verify
        logger.info("\n[Step 2] Verifying model locally...")
        try:
            logger.info("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("  ✓ Tokenizer loaded")
            
            logger.info("  Loading model config...")
            model = AutoModelForCausalLM.from_pretrained(model_path)
            logger.info("  ✓ Model loaded")
            logger.info(f"  Model type: {type(model).__name__}")
            logger.info(f"  Model size: {model.num_parameters():,} parameters")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)
        
        # Step 3: Upload to Hugging Face
        logger.info(f"\n[Step 3] Uploading to Hugging Face...")
        logger.info(f"  Repository: {repo_id}")
        
        try:
            # Initialize API
            api = HfApi()
            
            # Step 3a: Create repository if it doesn't exist
            logger.info("  Creating repository on hub...")
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    exist_ok=True
                )
                logger.info("  ✓ Repository created/verified")
            except Exception as e:
                logger.warning(f"  Warning creating repo: {e}")
            
            # Step 3b: Push model and tokenizer to hub (excluding checkpoint directories)
            logger.info("  Pushing model to hub (this may take a few minutes)...")
            
            # Upload entire folder excluding checkpoint directories
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                ignore_patterns=["checkpoint-*", ".git*", ".DS_Store"],
                commit_message="SN-Tune (Safety Neuron Fine-tuning) model"
            )
            logger.info("  ✓ Model pushed to hub (checkpoints excluded)")
            
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")
            sys.exit(1)
        
        # Step 4: Create model card (README)
        logger.info(f"\n[Step 4] Creating model card...")
        
        readme_content = f"""---
license: apache-2.0
tags:
- safety
- fine-tuning
- llama
- safety-neurons
---

# {model_name}

This is a Safety Neuron-Tuned (SN-Tune) version of Llama-3.2-3B-Instruct.

## Model Description

- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning Method**: SN-Tune (Safety Neuron Tuning)
- **Training Data**: Circuit Breakers dataset (safety alignment data)
- **Upload Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## What is SN-Tune?

SN-Tune is a selective fine-tuning approach that:
1. Detects safety neurons - a small set of neurons critical for safety
2. Freezes all non-safety parameters
3. Fine-tunes only safety neurons on safety data

This approach allows for:
- Enhanced safety alignment
- Minimal impact on general capabilities
- Parameter-efficient fine-tuning

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "How can I help you today?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Safety Note

This model has been fine-tuned specifically for safety using the SN-Tune method.
It should provide improved safety alignment compared to the base model.

## License

This model is licensed under the Apache 2.0 License.
See the base model (meta-llama/Llama-3.2-3B-Instruct) for more details.

## References

- Base model: [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- Safety neurons detection methodology
"""
        
        try:
            readme_path = os.path.join(model_path, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            logger.info("  ✓ README.md created")
            
            # Push README to hub
            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add model card"
            )
            logger.info("  ✓ README.md pushed to hub")
            
        except Exception as e:
            logger.warning(f"Failed to upload README: {e}")
        
        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info("Upload Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"\n✓ Model successfully uploaded to Hugging Face")
        logger.info(f"\nRepository URL:")
        logger.info(f"  https://huggingface.co/{repo_id}")
        logger.info(f"\nYou can now use this model with:")
        logger.info(f"  from transformers import AutoModelForCausalLM")
        logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
        logger.info(f"\n{'='*70}\n")
        
        return repo_id
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def main(argv):
    if len(argv) < 1:
        logger.error("Usage: python upload_sn_tuned_model.py <model_local_path>")
        logger.error("\nExample:")
        logger.error("  python upload_sn_tuned_model.py ./sn_tuned_model_20251208_223350")
        logger.error("\nNote:")
        logger.error("  - Make sure you have huggingface_hub installed")
        logger.error("  - Authenticate with: huggingface-cli login")
        logger.error("  - Or set HUGGINGFACE_TOKEN environment variable")
        sys.exit(1)
    
    model_path = argv[0]
    upload_to_huggingface(model_path)


if __name__ == "__main__":
    main(sys.argv[1:])
