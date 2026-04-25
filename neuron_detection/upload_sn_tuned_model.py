"""
llama 3.1 8B
python upload_sn_tuned_model.py \
    --upload_pair ./full_finetune_llama3.1_8b_instruct_gsm8k_ssft3e-5_lr1e-5 kmseong/llama3.1_8b_instruct_gsm8k_full_ft_lr1e-5 \
    --upload_pair ./math_ft_8b_instruct_freeze_rsn_lr1e-5_20260420_004336 kmseong/llama3.1_8b_instruct_math_ft_freeze_rsn_lr1e-5_new 

llama 2 7B
python upload_sn_tuned_model.py \
    --upload_pair /home/yonsei_jong/SafeDelta/finetuned_models/gsm8k-llama2-7b-chat-safeft kmseong/llama2_7b_chat_safedelta_only_gsm8k \
"""

import argparse
import os
import sys
from datetime import datetime
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args(argv):
    p = argparse.ArgumentParser(description="Upload SN-tuned model to Hugging Face Hub")
    p.add_argument("model_path", type=str, nargs="?", help="Local model directory path")
    p.add_argument(
        "--repo_id",
        type=str,
        help="Target Hugging Face repository id, e.g. kmseong/my-model",
    )
    p.add_argument(
        "--upload_pair",
        nargs=2,
        action="append",
        metavar=("MODEL_PATH", "REPO_ID"),
        help="Upload multiple model/repo pairs in one run. Repeat this option for each pair.",
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face access token. If omitted, use HF_TOKEN/HUGGINGFACE_TOKEN or cached login.",
    )
    args = p.parse_args(argv)

    using_single = args.model_path is not None or args.repo_id is not None
    using_multi = bool(args.upload_pair)

    if using_single and using_multi:
        p.error("Use either single upload mode (<model_path> --repo_id ...) or multi upload mode (--upload_pair ...), not both.")

    if using_single:
        if not args.model_path or not args.repo_id:
            p.error("Single upload mode requires both <model_path> and --repo_id.")
    elif not using_multi:
        p.error("Provide either <model_path> --repo_id or one or more --upload_pair MODEL_PATH REPO_ID.")

    return args


def upload_to_huggingface(model_path, repo_id, hf_token=None):
    """
    Upload SN-tuned model to Hugging Face Hub
    
    Args:
        model_path: Local path to the model directory
        repo_id: Target Hugging Face repo id (username/model_name)
        hf_token: Optional Hugging Face token
    """
    
    # Resolve token once and reuse for all hub operations.
    effective_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # Verify model path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)
    
    # Check core model files
    required_files = ['config.json']
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
            logger.warning(f"Warning: {file} not found in {model_path}")

    # Llama tokenizer can be tokenizer.json or tokenizer.model depending on conversion.
    has_tokenizer_asset = any(
        os.path.exists(os.path.join(model_path, candidate))
        for candidate in ["tokenizer.json", "tokenizer.model"]
    )
    if not has_tokenizer_asset:
        missing_files.append("tokenizer.json|tokenizer.model")
        logger.warning(f"Warning: tokenizer.json/tokenizer.model not found in {model_path}")
    
    # If tokenizer files are missing, copy from base model.
    if missing_files:
        logger.info(f"\n⚠ Missing files detected: {missing_files}")
        logger.info(f"Copying complete tokenizer from base model...")
        try:
            base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
            temp_tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=effective_token)
            
            # Save tokenizer to the model directory
            temp_tokenizer.save_pretrained(model_path)
            logger.info(f"✓ Tokenizer files copied from base model")
            
        except Exception as e:
            logger.error(f"Failed to copy tokenizer files: {e}")
            logger.error(f"Please run SN-Tune again - it should save the tokenizer automatically")
            logger.error(f"Or manually copy tokenizer files from meta-llama/Llama-3.2-3B-Instruct")
            sys.exit(1)
    
    model_name = repo_id.split("/")[-1]
    
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
            api = HfApi(token=effective_token)
            whoami = api.whoami(token=effective_token)
            logger.info(f"✓ Authenticated as: {whoami.get('name', 'unknown')}")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.info("\nPlease login to Hugging Face:")
            logger.info("  1) Pass --hf_token <token>")
            logger.info("  2) Or set HF_TOKEN/HUGGINGFACE_TOKEN")
            logger.info("  3) Or run: huggingface-cli login")
            sys.exit(1)
        
        # Step 2: Load model and tokenizer locally to verify
        logger.info("\n[Step 2] Verifying model locally...")
        try:
            logger.info("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("  ✓ Tokenizer loaded")
            
            logger.info("  Loading model config...")
            # Load without device_map for verification (avoid GPU memory issues)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=None,  # Don't use device_map during verification
                    torch_dtype=None,  # Use default dtype
                )
            except Exception as e1:
                # Fallback: try with CPU
                logger.warning(f"  Failed with device_map=None: {e1}")
                logger.info("  Retrying with CPU...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="cpu"
                )
            
            logger.info("  ✓ Model loaded")
            logger.info(f"  Model type: {type(model).__name__}")
            logger.info(f"  Model size: {model.num_parameters():,} parameters")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(f"Full traceback:")
            logger.error(traceback.format_exc())
            logger.info("\n  Diagnosis: Checking model files...")
            logger.info(f"  Model directory: {model_path}")
            logger.info(f"  Files present:")
            import subprocess
            result = subprocess.run(['ls', '-lh', model_path], capture_output=True, text=True)
            logger.info(result.stdout)
            sys.exit(1)
        
        # Step 3: Upload to Hugging Face
        logger.info(f"\n[Step 3] Uploading to Hugging Face...")
        logger.info(f"  Repository: {repo_id}")
        
        try:
            # Step 3a: Create repository if it doesn't exist
            logger.info("  Creating repository on hub...")
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    exist_ok=True,
                    token=effective_token,
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
                commit_message="SN-Tune (Safety Neuron Fine-tuning) model",
                token=effective_token,
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
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add model card",
                token=effective_token,
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


def upload_multiple_to_huggingface(upload_pairs, hf_token=None):
    uploaded = []

    logger.info(f"Preparing to upload {len(upload_pairs)} models...")
    for idx, (model_path, repo_id) in enumerate(upload_pairs, start=1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"[{idx}/{len(upload_pairs)}] Uploading {model_path} -> {repo_id}")
        logger.info(f"{'#' * 70}")
        uploaded_repo = upload_to_huggingface(model_path, repo_id, hf_token)
        uploaded.append((model_path, uploaded_repo))

    logger.info(f"\n{'=' * 70}")
    logger.info("Batch upload complete")
    logger.info(f"{'=' * 70}")
    for model_path, repo_id in uploaded:
        logger.info(f"  - {model_path} -> https://huggingface.co/{repo_id}")
    logger.info(f"{'=' * 70}")

    return uploaded


def main(argv):
    args = parse_args(argv)
    if args.upload_pair:
        upload_multiple_to_huggingface(args.upload_pair, args.hf_token)
    else:
        upload_to_huggingface(args.model_path, args.repo_id, args.hf_token)


if __name__ == "__main__":
    main(sys.argv[1:])
