#!/usr/bin/env python3
"""
간단한 모델 업로드 스크립트
/home/hail/kms/Safety-Neuron/neuron_enhancement/xxxxxx/Llama3_SafetyEnhanced 을 Hugging Face에 업로드
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# 설정
model_path = "/home/hail/kms/Safety-Neuron/neuron_enhancement/xxxxxx/Llama3_SafetyEnhanced"
hf_username = "kmseong"  # 자신의 계정으로 변경

# 타임스탬프로 모델명 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
repo_name = f"{hf_username}/Llama3_SafetyEnhanced_{timestamp}"

print(f"모델 경로: {model_path}")
print(f"업로드 대상: {repo_name}")
print("-" * 50)

# 1. 모델과 토크나이저 로드
print("\n1️⃣ 모델 로드 중...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
print("✓ 모델 로드 완료")

print("2️⃣ 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("✓ 토크나이저 로드 완료")

# 2. Hugging Face에 업로드
print(f"\n3️⃣ Hugging Face에 업로드 중...")
print(f"   리포지토리: {repo_name}")

model.push_to_hub(
    repo_id=repo_name,
    private=False,
    commit_message="Safety-Enhanced Llama3 model with frozen safety neurons"
)
print("✓ 모델 업로드 완료")

tokenizer.push_to_hub(
    repo_id=repo_name,
    commit_message="Tokenizer for Safety-Enhanced Llama3"
)
print("✓ 토크나이저 업로드 완료")

print("\n" + "=" * 50)
print("✅ 완료!")
print("=" * 50)
print(f"📍 모델 URL: https://huggingface.co/{repo_name}")
