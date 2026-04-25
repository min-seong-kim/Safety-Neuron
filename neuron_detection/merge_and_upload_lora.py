"""
LoRA 어댑터를 base model에 merge하고 허깅페이스에 업로드하는 스크립트
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_path = "./lora_gsm8k_llama3.1_8b"
base_model_id = "kmseong/llama3.1_8b_base-Safety-FT-lr3e-5"
repo_id = "kmseong/llama3.1_8b_base-gsm8k_lora_ft_lr5e-5"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

print(f"Loading base model: {base_model_id} ...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print("Loading LoRA adapter and merging...")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
print("Merge complete.")

print(f"Uploading model to {repo_id}...")
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
print("Upload complete!")
