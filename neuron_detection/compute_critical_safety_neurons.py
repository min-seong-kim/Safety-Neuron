"""
Step 2: Compute Critical Safety Neurons

목표:
  Critical Safety Neurons = Safety Neurons - (Safety Neurons ∩ Utility Neurons)
  
설명:
  Safety Neurons와 Utility Neurons의 교집합을 제외하고, 
  Safety에만 필요한 뉴런들만 남긴다.
  
  이렇게 하면 downstream fine-tuning(예: GSM8K)에서도 
  안전성을 잃지 않으면서 성능을 높일 수 있다.

입력:
  - Safety Neurons file (from neuron_detection_simple.py)
    예: meta-llama_Llama-3.2-3B-Instruct_harmful_prompts_200_threshold_neurons_200_*.txt
  
  - Utility Neurons file (from neuron_detection_foundation.py)
    예: meta-llama_Llama-3.2-3B-Instruct_utility_neurons_1000_*.txt

출력:
  - Critical Safety Neurons file
    예: meta-llama_Llama-3.2-3B-Instruct_critical_safety_neurons_*.txt

수식:
  N_critical = N_safe - (N_safe ∩ N_utility)
  
  where:
    N_safe = Safety neurons from harmful prompts
    N_utility = Utility neurons from Wikipedia
    N_critical = Neurons critical for safety only

사용법:
  python compute_critical_safety_neurons.py [safety_file] [utility_file]
  
  예시:
    python compute_critical_safety_neurons.py
    # 자동으로 최신 파일 찾음
    
    또는
    
    python compute_critical_safety_neurons.py \
      ./output_neurons/meta-llama_Llama-3.2-3B-Instruct_harmful_prompts_200_*.txt \
      ./output_neurons/meta-llama_Llama-3.2-3B-Instruct_utility_neurons_1000_*.txt
"""

import os
import sys
import logging
from typing import Dict, Set
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_neurons_from_file(file_path: str) -> Dict[str, Dict[int, Set[str]]]:
    """
    Load neuron data from saved file.
    
    File format (5 lines):
    Line 1: ffn_up dictionary
    Line 2: ffn_down dictionary
    Line 3: q dictionary
    Line 4: k dictionary
    Line 5: v dictionary
    
    Returns:
        {'ffn_up': {layer_idx: set}, 'ffn_down': {...}, 'q': {...}, 'k': {...}, 'v': {...}}
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 5:
            logger.error(f"Invalid file format (expected 5 lines, got {len(lines)})")
            return None
        
        # Parse each line as a dictionary
        ffn_up = eval(lines[0].strip())
        ffn_down = eval(lines[1].strip())
        q = eval(lines[2].strip())
        k = eval(lines[3].strip())
        v = eval(lines[4].strip())
        
        # Convert layer indices to int and neuron names to set
        for module in [ffn_up, ffn_down, q, k, v]:
            for layer_idx in list(module.keys()):
                if not isinstance(layer_idx, int):
                    module[int(layer_idx)] = module.pop(layer_idx)
                if isinstance(module[layer_idx], list):
                    module[layer_idx] = set(module[layer_idx])
        
        return {
            'ffn_up': ffn_up,
            'ffn_down': ffn_down,
            'q': q,
            'k': k,
            'v': v,
        }
    
    except Exception as e:
        logger.error(f"Error loading neuron file: {e}")
        return None


def compute_critical_safety_neurons(safety_neurons: Dict, utility_neurons: Dict, num_layers: int = 28) -> Dict:
    """
    Compute Critical Safety Neurons = Safety - (Safety ∩ Utility)
    
    Args:
        safety_neurons: Dictionary with structure {'ffn_up': {...}, 'ffn_down': {...}, ...}
        utility_neurons: Same structure
        num_layers: Number of transformer layers (28 for Llama-3.2-3B)
    
    Returns:
        critical: Dictionary with same structure containing only Critical Safety Neurons
    """
    
    critical = {}
    module_keys = ['ffn_up', 'ffn_down', 'q', 'k', 'v']
    
    for module in module_keys:
        critical[module] = {}
        
        safety_module = safety_neurons.get(module, {})
        utility_module = utility_neurons.get(module, {})
        
        for layer_idx in range(num_layers):
            safety_set = safety_module.get(layer_idx, set())
            utility_set = utility_module.get(layer_idx, set())
            
            # Critical = Safety - (Safety ∩ Utility)
            overlap = safety_set & utility_set
            critical_layer = safety_set - overlap
            
            critical[module][layer_idx] = critical_layer
    
    return critical


def compute_statistics(safety_neurons: Dict, utility_neurons: Dict, critical_neurons: Dict) -> Dict:
    """
    Compute detailed statistics about Safety, Utility, Overlap, and Critical neurons.
    """
    
    stats = {
        'safety': {'ffn': 0, 'attn': 0, 'total': 0},
        'utility': {'ffn': 0, 'attn': 0, 'total': 0},
        'critical': {'ffn': 0, 'attn': 0, 'total': 0},
        'overlap': {'ffn': 0, 'attn': 0, 'total': 0},
        'layer_stats': {},
    }
    
    for layer_idx in range(28):
        layer_stats = {
            'safety': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
            'utility': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
            'critical': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
            'overlap': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
        }
        
        for module in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
            safety_set = safety_neurons[module].get(layer_idx, set())
            utility_set = utility_neurons[module].get(layer_idx, set())
            critical_set = critical_neurons[module].get(layer_idx, set())
            overlap_set = safety_set & utility_set
            
            layer_stats['safety'][module] = len(safety_set)
            layer_stats['utility'][module] = len(utility_set)
            layer_stats['critical'][module] = len(critical_set)
            layer_stats['overlap'][module] = len(overlap_set)
            
            # Update global stats
            module_type = 'ffn' if 'ffn' in module else 'attn'
            stats['safety'][module_type] += len(safety_set)
            stats['utility'][module_type] += len(utility_set)
            stats['critical'][module_type] += len(critical_set)
            stats['overlap'][module_type] += len(overlap_set)
        
        stats['layer_stats'][layer_idx] = layer_stats
        
        # Total counts
        for category in ['safety', 'utility', 'critical', 'overlap']:
            stats[category]['total'] += stats[category]['ffn'] + stats[category]['attn']
    
    return stats


def main(argv):
    """
    Main function to compute Critical Safety Neurons.
    
    Usage:
        python compute_critical_safety_neurons.py [safety_file] [utility_file]
    
    If files are not provided, the script will search for the latest ones in ./output_neurons/
    """
    
    # Find files if not provided
    if len(argv) < 2:
        logger.info("Searching for neuron detection files in ./output_neurons/...")
        
        if not os.path.exists("./output_neurons"):
            logger.error("Directory ./output_neurons/ does not exist")
            sys.exit(1)
        
        files = os.listdir("./output_neurons")
        
        # Find latest safety neurons file
        safety_files = [f for f in files if "harmful_prompts" in f and "threshold" in f]
        utility_files = [f for f in files if "utility_neurons" in f]
        
        if not safety_files:
            logger.error("No safety neuron files found in ./output_neurons/")
            logger.error("Please run: python neuron_detection_simple.py harmful_prompts 200")
            sys.exit(1)
        
        if not utility_files:
            logger.error("No utility neuron files found in ./output_neurons/")
            logger.error("Please run: python neuron_detection_foundation.py 1000")
            sys.exit(1)
        
        # Get latest files (by modification time)
        safety_file = sorted(safety_files, key=lambda f: os.path.getmtime(f"./output_neurons/{f}"))[-1]
        utility_file = sorted(utility_files, key=lambda f: os.path.getmtime(f"./output_neurons/{f}"))[-1]
        
        safety_file = f"./output_neurons/{safety_file}"
        utility_file = f"./output_neurons/{utility_file}"
        
        logger.info(f"Using safety file: {safety_file}")
        logger.info(f"Using utility file: {utility_file}")
    else:
        safety_file = argv[0]
        utility_file = argv[1]
    
    logger.info("\n" + "="*80)
    logger.info("Critical Safety Neuron Computation")
    logger.info("="*80)
    logger.info(f"\nFormula: N_critical = N_safe - (N_safe ∩ N_utility)")
    logger.info(f"Description: Safety neurons that do NOT overlap with utility neurons")
    
    # Load neurons
    logger.info("\nLoading safety neurons...")
    safety_neurons = load_neurons_from_file(safety_file)
    if safety_neurons is None:
        sys.exit(1)
    
    logger.info("Loading utility neurons...")
    utility_neurons = load_neurons_from_file(utility_file)
    if utility_neurons is None:
        sys.exit(1)
    
    # Compute Critical Safety Neurons
    logger.info("\nComputing Critical Safety Neurons...")
    critical_neurons = compute_critical_safety_neurons(safety_neurons, utility_neurons)
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_statistics(safety_neurons, utility_neurons, critical_neurons)
    
    # Save Critical Safety Neurons
    os.makedirs("./output_neurons", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    critical_output_file = f"./output_neurons/meta-llama_Llama-3.2-3B-Instruct_critical_safety_neurons_{timestamp}.txt"
    
    logger.info(f"\nSaving Critical Safety Neurons to {critical_output_file}...")
    with open(critical_output_file, "w", encoding="utf-8") as f:
        f.write(str(critical_neurons['ffn_up']) + "\n")
        f.write(str(critical_neurons['ffn_down']) + "\n")
        f.write(str(critical_neurons['q']) + "\n")
        f.write(str(critical_neurons['k']) + "\n")
        f.write(str(critical_neurons['v']) + "\n")
    
    # Print detailed statistics
    logger.info("\n" + "="*80)
    logger.info("Neuron Statistics")
    logger.info("="*80)
    
    logger.info(f"\n📊 Overall Summary:")
    logger.info(f"{'Category':<20} {'FFN':<10} {'Attention':<10} {'Total':<10}")
    logger.info(f"{'-'*50}")
    
    for category in ['safety', 'utility', 'critical', 'overlap']:
        ffn_count = stats[category]['ffn']
        attn_count = stats[category]['attn']
        total_count = stats[category]['total']
        logger.info(f"{category:<20} {ffn_count:<10} {attn_count:<10} {total_count:<10}")
    
    # Per-layer breakdown
    logger.info(f"\n🔍 Per-Layer Breakdown (showing layers with critical neurons > 0):")
    logger.info(f"{'Layer':<10} {'Safety':<10} {'Utility':<10} {'Overlap':<10} {'Critical':<10}")
    logger.info(f"{'-'*52}")
    
    for layer_idx in range(28):
        layer_stat = stats['layer_stats'][layer_idx]
        safety_total = sum(layer_stat['safety'].values())
        utility_total = sum(layer_stat['utility'].values())
        overlap_total = sum(layer_stat['overlap'].values())
        critical_total = sum(layer_stat['critical'].values())
        
        if critical_total > 0:
            logger.info(f"{layer_idx:<10} {safety_total:<10} {utility_total:<10} {overlap_total:<10} {critical_total:<10}")
    
    # Key insights
    logger.info(f"\n💡 Key Insights:")
    safety_total = stats['safety']['total']
    utility_total = stats['utility']['total']
    overlap_total = stats['overlap']['total']
    critical_total = stats['critical']['total']
    
    if safety_total > 0:
        overlap_pct = (overlap_total / safety_total) * 100
        logger.info(f"  • Overlap between Safety and Utility: {overlap_pct:.2f}% of Safety neurons")
    
    if safety_total > 0:
        critical_pct = (critical_total / safety_total) * 100
        logger.info(f"  • Critical neurons retained: {critical_pct:.2f}% of Safety neurons")
    
    if utility_total > 0:
        safety_pct = (safety_total / utility_total) * 100
        logger.info(f"  • Safety vs Utility neurons ratio: {safety_pct:.2f}%")
    
    logger.info(f"\n📈 Counts:")
    logger.info(f"  • Safety neurons: {safety_total}")
    logger.info(f"  • Utility neurons: {utility_total}")
    logger.info(f"  • Overlapping neurons: {overlap_total}")
    logger.info(f"  • Critical neurons (Safety - Overlap): {critical_total}")
    
    logger.info(f"\n✅ Critical Safety Neurons saved to: {critical_output_file}")
    logger.info("="*80)
    
    # Next steps
    logger.info("\n📋 Next Steps:")
    logger.info(f"  1. Review Critical Safety Neuron statistics above")
    logger.info(f"  2. Run Critical Safety Neuron fine-tuning with:")
    logger.info(f"     python critical_safety_neuron_tune.py {critical_output_file} ./corpus_all/circuit_breakers_train.json")


if __name__ == "__main__":
    main(sys.argv[1:])
