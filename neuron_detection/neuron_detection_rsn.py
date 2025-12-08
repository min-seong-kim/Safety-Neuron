"""
RSN (Robust Safety Neurons) Computation

목표:
  N_RSN = N_safe - (N_safe ∩ N_foundation)
  
  => Safety Neurons 중에서 Foundation Neurons와 겹치지 않는 부분만 선택
  => 이렇게 하면 downstream fine-tuning에서도 안전성이 보장됨

입력:
  - Safety Neurons file (from neuron_detection_simple.py)
  - Foundation Neurons file (from neuron_detection_foundation.py)

출력:
  - RSN file with statistics
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


def compute_rsn(safety_neurons: Dict, foundation_neurons: Dict, num_layers: int = 28) -> Dict:
    """
    Compute RSN = Safety - (Safety ∩ Foundation)
    
    Args:
        safety_neurons: Dictionary with structure {'ffn_up': {...}, 'ffn_down': {...}, ...}
        foundation_neurons: Same structure
        num_layers: Number of transformer layers (28 for Llama-3.2-3B)
    
    Returns:
        rsn: Dictionary with same structure containing only RSN
    """
    
    rsn = {}
    module_keys = ['ffn_up', 'ffn_down', 'q', 'k', 'v']
    
    for module in module_keys:
        rsn[module] = {}
        
        safety_module = safety_neurons.get(module, {})
        foundation_module = foundation_neurons.get(module, {})
        
        for layer_idx in range(num_layers):
            safety_set = safety_module.get(layer_idx, set())
            foundation_set = foundation_module.get(layer_idx, set())
            
            # RSN = Safety - (Safety ∩ Foundation)
            overlap = safety_set & foundation_set
            rsn_layer = safety_set - overlap
            
            rsn[module][layer_idx] = rsn_layer
    
    return rsn


def compute_statistics(safety_neurons: Dict, foundation_neurons: Dict, rsn_neurons: Dict) -> Dict:
    """
    Compute detailed statistics about Safety, Foundation, and RSN neurons.
    """
    
    stats = {
        'safety': {'ffn': 0, 'attn': 0, 'total': 0},
        'foundation': {'ffn': 0, 'attn': 0, 'total': 0},
        'rsn': {'ffn': 0, 'attn': 0, 'total': 0},
        'overlap': {'ffn': 0, 'attn': 0, 'total': 0},
        'layer_stats': {},
    }
    
    for layer_idx in range(28):
        layer_stats = {
            'safety': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
            'foundation': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
            'rsn': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
            'overlap': {'ffn_up': 0, 'ffn_down': 0, 'q': 0, 'k': 0, 'v': 0},
        }
        
        for module in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
            safety_set = safety_neurons[module].get(layer_idx, set())
            foundation_set = foundation_neurons[module].get(layer_idx, set())
            rsn_set = rsn_neurons[module].get(layer_idx, set())
            overlap_set = safety_set & foundation_set
            
            layer_stats['safety'][module] = len(safety_set)
            layer_stats['foundation'][module] = len(foundation_set)
            layer_stats['rsn'][module] = len(rsn_set)
            layer_stats['overlap'][module] = len(overlap_set)
            
            # Update global stats
            module_type = 'ffn' if 'ffn' in module else 'attn'
            stats['safety'][module_type] += len(safety_set)
            stats['foundation'][module_type] += len(foundation_set)
            stats['rsn'][module_type] += len(rsn_set)
            stats['overlap'][module_type] += len(overlap_set)
        
        stats['layer_stats'][layer_idx] = layer_stats
        
        # Total counts
        for category in ['safety', 'foundation', 'rsn', 'overlap']:
            stats[category]['total'] += stats[category]['ffn'] + stats[category]['attn']
    
    return stats


def main(argv):
    """
    Main function to compute RSN from Safety and Foundation neurons.
    
    Usage:
        python neuron_detection_rsn.py <safety_file> <foundation_file>
    
    Example:
        python neuron_detection_rsn.py \
            ./output_neurons/meta-llama_Llama-3.2-3B-Instruct_harmful_prompts_200_*.txt \
            ./output_neurons/meta-llama_Llama-3.2-3B-Instruct_foundation_neurons_1000_*.txt
    
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
        foundation_files = [f for f in files if "foundation_neurons" in f]
        
        if not safety_files:
            logger.error("No safety neuron files found in ./output_neurons/")
            logger.error("Please run: python neuron_detection_simple.py harmful_prompts 200")
            sys.exit(1)
        
        if not foundation_files:
            logger.error("No foundation neuron files found in ./output_neurons/")
            logger.error("Please run: python neuron_detection_foundation.py 1000")
            sys.exit(1)
        
        # Get latest files (by modification time)
        safety_file = sorted(safety_files, key=lambda f: os.path.getmtime(f"./output_neurons/{f}"))[-1]
        foundation_file = sorted(foundation_files, key=lambda f: os.path.getmtime(f"./output_neurons/{f}"))[-1]
        
        safety_file = f"./output_neurons/{safety_file}"
        foundation_file = f"./output_neurons/{foundation_file}"
        
        logger.info(f"Using safety file: {safety_file}")
        logger.info(f"Using foundation file: {foundation_file}")
    else:
        safety_file = argv[0]
        foundation_file = argv[1]
    
    logger.info("\n" + "="*80)
    logger.info("RSN (Robust Safety Neurons) Computation")
    logger.info("="*80)
    
    # Load neurons
    logger.info("\nLoading safety neurons...")
    safety_neurons = load_neurons_from_file(safety_file)
    if safety_neurons is None:
        sys.exit(1)
    
    logger.info("Loading foundation neurons...")
    foundation_neurons = load_neurons_from_file(foundation_file)
    if foundation_neurons is None:
        sys.exit(1)
    
    # Compute RSN
    logger.info("\nComputing RSN = Safety - (Safety ∩ Foundation)...")
    rsn_neurons = compute_rsn(safety_neurons, foundation_neurons)
    
    # Compute statistics
    logger.info("Computing statistics...")
    stats = compute_statistics(safety_neurons, foundation_neurons, rsn_neurons)
    
    # Save RSN
    os.makedirs("./output_neurons", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rsn_output_file = f"./output_neurons/meta-llama_Llama-3.2-3B-Instruct_rsn_neurons_{timestamp}.txt"
    
    logger.info(f"\nSaving RSN to {rsn_output_file}...")
    with open(rsn_output_file, "w", encoding="utf-8") as f:
        f.write(str(rsn_neurons['ffn_up']) + "\n")
        f.write(str(rsn_neurons['ffn_down']) + "\n")
        f.write(str(rsn_neurons['q']) + "\n")
        f.write(str(rsn_neurons['k']) + "\n")
        f.write(str(rsn_neurons['v']) + "\n")
    
    # Print detailed statistics
    logger.info("\n" + "="*80)
    logger.info("Neuron Statistics")
    logger.info("="*80)
    
    logger.info(f"\n📊 Overall Summary:")
    logger.info(f"{'Category':<20} {'FFN':<10} {'Attention':<10} {'Total':<10}")
    logger.info(f"{'-'*50}")
    
    for category in ['safety', 'foundation', 'rsn', 'overlap']:
        ffn_count = stats[category]['ffn']
        attn_count = stats[category]['attn']
        total_count = stats[category]['total']
        logger.info(f"{category:<20} {ffn_count:<10} {attn_count:<10} {total_count:<10}")
    
    # Per-layer breakdown
    logger.info(f"\n🔍 Per-Layer Breakdown (showing layers with RSN > 0):")
    logger.info(f"{'Layer':<10} {'Safety':<10} {'Foundation':<12} {'Overlap':<10} {'RSN':<10}")
    logger.info(f"{'-'*52}")
    
    for layer_idx in range(28):
        layer_stat = stats['layer_stats'][layer_idx]
        safety_total = sum(layer_stat['safety'].values())
        foundation_total = sum(layer_stat['foundation'].values())
        overlap_total = sum(layer_stat['overlap'].values())
        rsn_total = sum(layer_stat['rsn'].values())
        
        if rsn_total > 0:
            logger.info(f"{layer_idx:<10} {safety_total:<10} {foundation_total:<12} {overlap_total:<10} {rsn_total:<10}")
    
    # Key insights
    logger.info(f"\n💡 Key Insights:")
    safety_total = stats['safety']['total']
    foundation_total = stats['foundation']['total']
    overlap_total = stats['overlap']['total']
    rsn_total = stats['rsn']['total']
    
    if safety_total > 0:
        overlap_pct = (overlap_total / safety_total) * 100
        logger.info(f"  • Overlap between Safety and Foundation: {overlap_pct:.2f}% of Safety neurons")
    
    if safety_total > 0:
        rsn_pct = (rsn_total / safety_total) * 100
        logger.info(f"  • RSN retained: {rsn_pct:.2f}% of Safety neurons")
    
    if foundation_total > 0:
        safety_pct = (safety_total / foundation_total) * 100
        logger.info(f"  • Safety neurons vs Foundation neurons: {safety_pct:.2f}%")
    
    logger.info(f"\n📈 Counts:")
    logger.info(f"  • Safety neurons: {safety_total}")
    logger.info(f"  • Foundation neurons: {foundation_total}")
    logger.info(f"  • Overlapping neurons: {overlap_total}")
    logger.info(f"  • RSN neurons (Safety - Overlap): {rsn_total}")
    
    logger.info(f"\n✅ RSN saved to: {rsn_output_file}")
    logger.info("="*80)
    
    # Next steps
    logger.info("\n📋 Next Steps:")
    logger.info(f"  1. Review RSN statistics above")
    logger.info(f"  2. Run RSN-Tune fine-tuning with:")
    logger.info(f"     python rsn_tune.py {rsn_output_file} ./corpus_all/circuit_breakers_train.json")


if __name__ == "__main__":
    main(sys.argv[1:])
