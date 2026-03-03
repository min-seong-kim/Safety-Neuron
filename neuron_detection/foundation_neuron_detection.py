"""
Step 1: Utility Neuron Detection from Wikipedia

목표:
  Wikipedia 데이터로부터 Utility Neurons (일반 언어 처리에 필수적인 뉴런) 검출
  
알고리즘:
  1. Wikipedia 문서 로드 (1000개 권장)
  2. 각 문서마다 neuron_detection_simple.py의 detect_safety_neurons_threshold() 실행
  3. 모든 문서에서 공통으로 활성화되는 뉴런들의 교집합 계산
  4. 결과 저장 (utility_neurons_*.txt)

사용법:
  python detect_utility_neurons.py [num_docs] [model_name]
  
  예시:
    python foundation_neuron_detection.py 800

시간/메모리:
  - 입력: Wikipedia 문서 (권장: 1000개)
  - 시간: ~15-20분
  - 메모리: 16GB
"""

import os
import sys
import torch
import random
import logging
import json
from typing import Dict, Set, List, Tuple
from datetime import datetime
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(112)
torch.manual_seed(112)

# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------
model_name = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

NUM_LAYERS = 28
HIDDEN_DIM = 3072
TOTAL_NEURONS = NUM_LAYERS * HIDDEN_DIM

# Threshold hyperparameters
FFN_ACTIVE_FRACTION = 0.05
ATTN_ACTIVE_FRACTION = 0.05
MIN_NEURONS_FOR_QUANTILE = 10


def select_by_threshold(importance: torch.Tensor,
                        active_fraction: float) -> Set[int]:
    """
    Given a 1D importance vector [D], select indices whose importance >= epsilon.
    """
    if importance.numel() == 0:
        return set()

    if importance.numel() < MIN_NEURONS_FOR_QUANTILE:
        return set()

    q = max(0.0, min(1.0, 1.0 - active_fraction))
    epsilon = torch.quantile(importance, q)

    active_mask = importance >= epsilon
    indices = torch.nonzero(active_mask, as_tuple=False).view(-1)

    return {idx.item() for idx in indices}


def detect_safety_neurons_threshold(
    prompt: str,
) -> Tuple[Dict[int, Set[int]],
           Dict[int, Set[int]],
           Dict[int, Set[int]],
           Dict[int, Set[int]],
           Dict[int, Set[int]]]:
    """
    Neuron detection based on activation magnitude thresholding.
    
    Returns:
        (ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict)
    """

    ffn_up_dict: Dict[int, Set[int]] = {}
    ffn_down_dict: Dict[int, Set[int]] = {}
    q_dict: Dict[int, Set[int]] = {}
    k_dict: Dict[int, Set[int]] = {}
    v_dict: Dict[int, Set[int]] = {}

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        activations_dict = {}

        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    act = output[0]
                else:
                    act = output
                activations_dict[name] = act.detach().cpu()
            return hook

        hooks = []
        for layer_idx in range(NUM_LAYERS):
            layer = model.model.layers[layer_idx]

            if hasattr(layer.mlp, "up_proj"):
                hooks.append(
                    layer.mlp.up_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_ffn_up")
                    )
                )
            if hasattr(layer.mlp, "down_proj"):
                hooks.append(
                    layer.mlp.down_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_ffn_down")
                    )
                )

            if hasattr(layer.self_attn, "q_proj"):
                hooks.append(
                    layer.self_attn.q_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_attn_q")
                    )
                )
            if hasattr(layer.self_attn, "k_proj"):
                hooks.append(
                    layer.self_attn.k_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_attn_k")
                    )
                )
            if hasattr(layer.self_attn, "v_proj"):
                hooks.append(
                    layer.self_attn.v_proj.register_forward_hook(
                        get_activation_hook(f"layer_{layer_idx}_attn_v")
                    )
                )

        try:
            with torch.no_grad():
                _ = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    output_hidden_states=False,
                    return_dict=True,
                )

            for layer_idx in range(NUM_LAYERS):
                # FFN up_proj
                ffn_up_key = f"layer_{layer_idx}_ffn_up"
                if ffn_up_key in activations_dict:
                    act = activations_dict[ffn_up_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))
                    indices_set = select_by_threshold(
                        importance, FFN_ACTIVE_FRACTION
                    )
                    ffn_up_dict[layer_idx] = indices_set
                else:
                    ffn_up_dict[layer_idx] = set()

                # FFN down_proj
                ffn_down_key = f"layer_{layer_idx}_ffn_down"
                if ffn_down_key in activations_dict:
                    act = activations_dict[ffn_down_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))
                    indices_set = select_by_threshold(
                        importance, FFN_ACTIVE_FRACTION
                    )
                    ffn_down_dict[layer_idx] = indices_set
                else:
                    ffn_down_dict[layer_idx] = set()

                # Attention Q
                q_key = f"layer_{layer_idx}_attn_q"
                if q_key in activations_dict:
                    act = activations_dict[q_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))
                    indices_set = select_by_threshold(
                        importance, ATTN_ACTIVE_FRACTION
                    )
                    q_dict[layer_idx] = indices_set
                else:
                    q_dict[layer_idx] = set()

                # Attention K
                k_key = f"layer_{layer_idx}_attn_k"
                if k_key in activations_dict:
                    act = activations_dict[k_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))
                    indices_set = select_by_threshold(
                        importance, ATTN_ACTIVE_FRACTION
                    )
                    k_dict[layer_idx] = indices_set
                else:
                    k_dict[layer_idx] = set()

                # Attention V
                v_key = f"layer_{layer_idx}_attn_v"
                if v_key in activations_dict:
                    act = activations_dict[v_key].float()
                    importance = torch.abs(act).mean(dim=(0, 1))
                    indices_set = select_by_threshold(
                        importance, ATTN_ACTIVE_FRACTION
                    )
                    v_dict[layer_idx] = indices_set
                else:
                    v_dict[layer_idx] = set()

        finally:
            for h in hooks:
                h.remove()

    except Exception as e:
        logger.error(f"Error in neuron detection: {e}")
        for layer_idx in range(NUM_LAYERS):
            ffn_up_dict[layer_idx] = set()
            ffn_down_dict[layer_idx] = set()
            q_dict[layer_idx] = set()
            k_dict[layer_idx] = set()
            v_dict[layer_idx] = set()

    return ffn_up_dict, ffn_down_dict, q_dict, k_dict, v_dict


def compute_intersection(neuron_sets_list: List[Dict[int, Set[int]]]) -> Dict[int, Set[int]]:
    """
    Compute intersection of neuron sets across all documents.
    """
    if not neuron_sets_list:
        return {}

    intersection_dict: Dict[int, Set[int]] = {}
    all_layers = range(NUM_LAYERS)

    for layer_idx in all_layers:
        layer_sets = []
        for neuron_dict in neuron_sets_list:
            layer_sets.append(neuron_dict.get(layer_idx, set()))

        if not layer_sets:
            intersection_dict[layer_idx] = set()
            continue

        common = set.intersection(*layer_sets) if layer_sets else set()
        intersection_dict[layer_idx] = common

    return intersection_dict


def load_wikipedia_data(num_samples: int = 1000) -> List[str]:
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
        
        for idx in tqdm(random_indices, desc="Loading Wikipedia docs"):
            try:
                text = dataset[idx]['text']
                # Take first sentence or paragraph (max 512 tokens)
                sentences = text.split('.')
                if sentences:
                    texts.append(sentences[0].strip()[:512])
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Successfully loaded {len(texts)} Wikipedia samples")
        return texts
        
    except Exception as e:
        logger.error(f"Error loading Wikipedia dataset: {e}")
        logger.error("Please check your internet connection or HuggingFace access")
        raise


def main(argv):
    """
    Main function to detect foundation neurons from Wikipedia.
    
    Usage:
        python neuron_detection_foundation.py [num_docs] [model_path]
        
    Example:
        python neuron_detection_foundation.py 1000
        python neuron_detection_foundation.py 500 meta-llama/Llama-3.2-3B-Instruct
    """
    
    num_docs = int(argv[0]) if len(argv) > 0 else 1000
    
    logger.info("="*70)
    logger.info("Foundation Neuron Detection from Wikipedia")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Num Wikipedia docs: {num_docs}")
    logger.info(f"FFN_ACTIVE_FRACTION: {FFN_ACTIVE_FRACTION}")
    logger.info(f"ATTN_ACTIVE_FRACTION: {ATTN_ACTIVE_FRACTION}\n")
    
    # Step 1: Load Wikipedia data
    wikipedia_docs = load_wikipedia_data(num_samples=num_docs)
    
    if not wikipedia_docs:
        logger.error("Failed to load Wikipedia data")
        sys.exit(1)
    
    # Step 2: Detect neurons for each document
    logger.info("\nDetecting foundation neurons for each Wikipedia document...")
    ffn_up_sets: List[Dict[int, Set[int]]] = []
    ffn_down_sets: List[Dict[int, Set[int]]] = []
    q_sets: List[Dict[int, Set[int]]] = []
    k_sets: List[Dict[int, Set[int]]] = []
    v_sets: List[Dict[int, Set[int]]] = []
    
    for idx, doc in enumerate(tqdm(wikipedia_docs, desc="Detecting neurons")):
        try:
            ffn_up, ffn_down, q, k, v = detect_safety_neurons_threshold(doc)
            ffn_up_sets.append(ffn_up)
            ffn_down_sets.append(ffn_down)
            q_sets.append(q)
            k_sets.append(k)
            v_sets.append(v)
        except Exception as e:
            logger.warning(f"Failed doc {idx}: {str(e)[:100]}")
    
    logger.info(f"Successfully processed {len(ffn_up_sets)}/{len(wikipedia_docs)} documents")
    
    # Step 3: Compute intersection (Foundation Neurons)
    logger.info("\nComputing foundation neuron intersections...")
    ffn_up_foundation = compute_intersection(ffn_up_sets)
    ffn_down_foundation = compute_intersection(ffn_down_sets)
    q_foundation = compute_intersection(q_sets)
    k_foundation = compute_intersection(k_sets)
    v_foundation = compute_intersection(v_sets)
    
    # Step 4: Save results
    os.makedirs("./output_neurons", exist_ok=True)
    clean_model_name = model_name.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        f"./output_neurons/"
        f"{clean_model_name}_utility_neurons_{len(ffn_up_sets)}_{timestamp}.txt"
    )
    
    logger.info(f"Saving results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        # safety_neuron_detection.py와 동일한 JSON line 포맷
        f.write(json.dumps({str(k): list(v) for k, v in ffn_up_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in ffn_down_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in q_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in k_foundation.items()}) + "\n")
        f.write(json.dumps({str(k): list(v) for k, v in v_foundation.items()}) + "\n")
    
    # Statistics
    logger.info("\n" + "="*70)
    logger.info("Utility Neuron Detection Results")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Wikipedia documents processed: {len(ffn_up_sets)}/{len(wikipedia_docs)}\n")
    
    total_foundation_neurons = 0
    total_ffn_neurons = 0
    total_attn_neurons = 0
    
    for layer_idx in range(NUM_LAYERS):
        ffn_up_count = len(ffn_up_foundation.get(layer_idx, set()))
        ffn_down_count = len(ffn_down_foundation.get(layer_idx, set()))
        q_count = len(q_foundation.get(layer_idx, set()))
        k_count = len(k_foundation.get(layer_idx, set()))
        v_count = len(v_foundation.get(layer_idx, set()))
        
        ffn_count = ffn_up_count + ffn_down_count
        attn_count = q_count + k_count + v_count
        layer_neurons = ffn_count + attn_count
        
        if layer_neurons > 0:
            logger.info(f"Layer {layer_idx}: {layer_neurons} foundation neurons (FFN: {ffn_count}, Attention: {attn_count})")
            total_foundation_neurons += layer_neurons
            total_ffn_neurons += ffn_count
            total_attn_neurons += attn_count
    
    foundation_sparsity = total_foundation_neurons / TOTAL_NEURONS if TOTAL_NEURONS > 0 else 0
    logger.info(f"\nTotal foundation neurons detected: {total_foundation_neurons} (FFN: {total_ffn_neurons}, Attention: {total_attn_neurons})")
    logger.info(f"Foundation sparsity: {foundation_sparsity*100:.4f}%")
    logger.info(f"Output saved to: {output_file}")
    logger.info("="*70)
    
    # Print next steps
    logger.info("\n📋 Next Steps:")
    logger.info(f"1. ✓ Safety Neurons: Already detected")
    logger.info(f"2. ✓ Foundation Neurons: Just detected (saved above)")
    logger.info(f"3. → Run: python neuron_detection_rsn.py")
    logger.info(f"   to compute RSN = Safety - (Safety ∩ Foundation)")


if __name__ == "__main__":
    main(sys.argv[1:])
