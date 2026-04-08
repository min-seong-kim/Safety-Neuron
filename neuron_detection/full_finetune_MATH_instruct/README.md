---
license: apache-2.0
tags:
- safety
- fine-tuning
- llama
- safety-neurons
---

# llama3.2_3b_instruct_full_finetune_MATH_3e-5

This is a Safety Neuron-Tuned (SN-Tune) version of Llama-3.2-3B-Instruct.

## Model Description

- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning Method**: SN-Tune (Safety Neuron Tuning)
- **Training Data**: Circuit Breakers dataset (safety alignment data)
- **Upload Date**: 2026-04-08 10:50:24

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

model_name = "kmseong/llama3.2_3b_instruct_full_finetune_MATH_3e-5"
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
