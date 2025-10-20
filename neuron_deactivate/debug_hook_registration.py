#!/usr/bin/env python3
import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# 뉴런 파일 로드 및 파싱 함수
def load_neurons_from_file(file_path):
    """뉴런 파일을 로드하고 파싱"""
    print(f"Loading neurons from: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    try:
        # 각 줄이 하나의 딕셔너리인지 확인
        print(f"File has {len(lines)} lines")
        
        # 첫 번째 줄을 파싱해보기
        if lines:
            print(f"First line preview: {lines[0][:100]}...")
            first_dict = ast.literal_eval(lines[0].strip())
            print(f"First dict type: {type(first_dict)}")
            print(f"First dict keys count: {len(first_dict)}")
            
            # 첫 번째 딕셔너리가 전체 레이어를 포함하는지 확인
            if len(first_dict) == 32:  # Llama-3-8B는 32 레이어
                print("Using first line as complete neuron data")
                neurons_dict = first_dict
            else:
                print("Multiple dictionaries detected, need to combine them")
                # 여러 딕셔너리를 합치는 로직이 필요할 수 있음
                neurons_dict = first_dict
        
        print(f"Successfully parsed neurons dict with {len(neurons_dict)} layers")
        
        # 각 레이어별 뉴런 개수 확인
        for layer_idx in sorted(neurons_dict.keys())[:5]:  # 처음 5개 레이어만 출력
            neuron_count = len(neurons_dict[layer_idx])
            print(f"Layer {layer_idx}: {neuron_count} neurons")
            if neuron_count > 0:
                # 첫 번째 뉴런 예시 출력
                first_neuron = next(iter(neurons_dict[layer_idx]))
                print(f"  Example neuron: {first_neuron}")
        
        return neurons_dict
    except Exception as e:
        print(f"Error parsing neurons file: {e}")
        import traceback
        traceback.print_exc()
        return None

# 뉴런 인덱스 추출 함수
def extract_neuron_index(neuron_name):
    """neuron_123 형태에서 123 추출"""
    try:
        return int(neuron_name.split('_')[1])
    except:
        return None

# 테스트 실행
def main():
    neuron_file = '/home/hail/kms/Safety-Neuron/neuron_detection/output_neurons/meta-llama_Meta-Llama-3-8B_english_real_neurons_50.txt'
    
    # 1. 뉴런 데이터 로드 테스트
    print("=== Step 1: Loading neuron data ===")
    neurons_dict = load_neurons_from_file(neuron_file)
    
    if neurons_dict is None:
        print("Failed to load neurons")
        return
    
    # 2. 특정 레이어의 뉴런 인덱스 추출 테스트
    print("\n=== Step 2: Testing neuron index extraction ===")
    test_layer = 5  # 레이어 5 테스트
    if test_layer in neurons_dict:
        layer_neurons = neurons_dict[test_layer]
        print(f"Layer {test_layer} has {len(layer_neurons)} neurons")
        
        # 뉴런 인덱스 추출
        neuron_indices = []
        for neuron_name in layer_neurons:
            idx = extract_neuron_index(neuron_name)
            if idx is not None:
                neuron_indices.append(idx)
            else:
                print(f"Failed to parse neuron: {neuron_name}")
        
        print(f"Extracted {len(neuron_indices)} valid neuron indices")
        print(f"Sample indices: {sorted(neuron_indices)[:10]}")
    
    # 3. 모델 로드 테스트 (가벼운 테스트)
    print("\n=== Step 3: Testing model layer structure ===")
    try:
        model_name = "meta-llama/Meta-Llama-3-8B"
        print(f"Loading model: {model_name}")
        
        # 토크나이저만 먼저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer loaded successfully")
        
        # 모델 구조 정보만 확인 (실제 로드는 생략)
        print("Model would have 32 layers with FFN components")
        print("Each layer has: up_proj, down_proj for FFN")
        
    except Exception as e:
        print(f"Error with model: {e}")
    
    # 4. 후크 등록 조건 시뮬레이션
    print("\n=== Step 4: Simulating hook registration conditions ===")
    total_hooks_would_register = 0
    
    for layer_idx in neurons_dict:
        layer_neurons = neurons_dict[layer_idx]
        if len(layer_neurons) > 0:
            neuron_indices = []
            for neuron_name in layer_neurons:
                idx = extract_neuron_index(neuron_name)
                if idx is not None:
                    neuron_indices.append(idx)
            
            if neuron_indices:
                total_hooks_would_register += 1
                print(f"Layer {layer_idx}: Would register hook for {len(neuron_indices)} neurons")
    
    print(f"\nTotal layers that would get hooks: {total_hooks_would_register}")

if __name__ == "__main__":
    main()