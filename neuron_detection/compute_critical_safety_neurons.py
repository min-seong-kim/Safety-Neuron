"""
Step 2: Compute Critical Safety Neurons

목표:
  Critical Safety Neurons = Safety Neurons - (Safety Neurons ∩ Utility Neurons)
    
python compute_critical_safety_neurons.py \
    ./output_neurons/llama_2_7b_base_safety_neuron_accelerated_20260417_003734.txt \
    ./output_neurons/utility_neurons_1000_20260417_125034.txt

python compute_critical_safety_neurons.py \
    ./output_neurons/llama_2_7b_chat_safety_neuron_accelerated_20260416_160653.txt \
    ./output_neurons/utility_neurons_1000_20260417_125319.txt

python compute_critical_safety_neurons.py \
    ./output_neurons/llama_31_8b_base_safety_neuron_accelerated_20260418_195615.txt \
    ./output_neurons/utility_neurons_1000_20260418_161554.txt

python compute_critical_safety_neurons.py \
    ./output_neurons/llama_31_8b_instruct_safety_neuron_accelerated_20260418_195634.txt \
    ./output_neurons/utility_neurons_1000_20260418_162404.txt


"""

import os
import sys
import logging
import json
import ast
from typing import Dict, Set
from datetime import datetime
from transformers import AutoConfig

from neuron_percentage_utils import (
    LLAMA31_8B_FALLBACK_DIMS,
    calculate_total_model_neurons_from_config,
    calculate_total_model_neurons_from_dims,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"


def setup_logging() -> str:
    """Configure file + console logging and return log file path."""
    log_dir = os.path.join(SCRIPT_DIR, "logs", "critical_safety_neuron")
    os.makedirs(log_dir, exist_ok=True)

    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"compute_critical_safety_neurons_{log_timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log file: {log_file}")
    return log_file


def calculate_model_total_neurons(model_name: str = DEFAULT_MODEL_NAME) -> int:
    """Model-wide neuron denominator: q/k/v/o + gate/up/down output channels."""
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        logger.info(f"Using model config for denominator: {model_name}")
        return calculate_total_model_neurons_from_config(cfg)
    except Exception as e:
        # Fallback for Llama-3.1-8B architecture
        logger.warning(f"Failed to load AutoConfig ({e}); using Llama-3.1-8B fallback config")
        return calculate_total_model_neurons_from_dims(**LLAMA31_8B_FALLBACK_DIMS)


def _parse_dict_line(raw: str) -> Dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)


def load_neurons_from_file(file_path: str) -> Dict[str, Dict[int, Set[int]]]:
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
        
        # Parse each line as a dictionary (JSON lines preferred)
        ffn_up = _parse_dict_line(lines[0])
        ffn_down = _parse_dict_line(lines[1])
        q = _parse_dict_line(lines[2])
        k = _parse_dict_line(lines[3])
        v = _parse_dict_line(lines[4])
        
        # Convert layer indices to int and neuron indices to set[int]
        for module in [ffn_up, ffn_down, q, k, v]:
            for original_layer_idx in list(module.keys()):
                layer_idx = int(original_layer_idx)
                values = module.pop(original_layer_idx)
                if isinstance(values, list):
                    values = set(int(v) for v in values)
                elif isinstance(values, set):
                    values = set(int(v) for v in values)
                else:
                    values = set()
                module[layer_idx] = values
        
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


def compute_critical_safety_neurons(safety_neurons: Dict, utility_neurons: Dict, num_layers: int = None) -> Dict:
    """
    Compute Critical Safety Neurons = Safety - (Safety ∩ Utility)
    
    Args:
        safety_neurons: Dictionary with structure {'ffn_up': {...}, 'ffn_down': {...}, ...}
        utility_neurons: Same structure
        num_layers: Number of transformer layers (inferred from data if None)
    
    Returns:
        critical: Dictionary with same structure containing only Critical Safety Neurons
    """
    
    if num_layers is None:
        all_layers = set()
        for module in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
            all_layers.update(safety_neurons.get(module, {}).keys())
            all_layers.update(utility_neurons.get(module, {}).keys())
        num_layers = max(all_layers) + 1 if all_layers else 32

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

    all_layers = set()
    for module in ['ffn_up', 'ffn_down', 'q', 'k', 'v']:
        all_layers.update(safety_neurons.get(module, {}).keys())
        all_layers.update(utility_neurons.get(module, {}).keys())
        all_layers.update(critical_neurons.get(module, {}).keys())
    num_layers = max(all_layers) + 1 if all_layers else 32

    for layer_idx in range(num_layers):
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
        
    # Total counts (한 번만 계산)
    for category in ['safety', 'utility', 'critical', 'overlap']:
        stats[category]['total'] = stats[category]['ffn'] + stats[category]['attn']
    
    return stats


def main(argv):
    """
    Main function to compute Critical Safety Neurons.
    
    Usage:
        python compute_critical_safety_neurons.py [safety_file] [utility_file]
    
    If files are not provided, the script will search for the latest ones in ./output_neurons/
    """
    
    log_file = setup_logging()

    output_dir = os.path.join(SCRIPT_DIR, "output_neurons")

    # Find files if not provided
    if len(argv) < 2:
        logger.info(f"Searching for neuron detection files in {output_dir}...")
        
        if not os.path.exists(output_dir):
            logger.error(f"Directory does not exist: {output_dir}")
            sys.exit(1)
        
        files = os.listdir(output_dir)
        
        # Find latest safety neurons file
        safety_files = [
            f for f in files
            if (
                ("threshold_neurons" in f or "safety-neuron_threshold" in f)
                and "utility_neurons" not in f
                and "critical_safety_neurons" not in f
                and "critical-safety-neuron" not in f
            )
        ]
        utility_files = [f for f in files if "utility_neurons" in f]
        
        if not safety_files:
            logger.error(f"No safety neuron files found in {output_dir}")
            logger.error("Please run: python neuron_detection_simple.py harmful_prompts 200")
            sys.exit(1)
        
        if not utility_files:
            logger.error(f"No utility neuron files found in {output_dir}")
            logger.error("Please run: python neuron_detection_foundation.py 1000")
            sys.exit(1)
        
        # Get latest files (by modification time)
        safety_file = sorted(safety_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))[-1]
        utility_file = sorted(utility_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))[-1]
        
        safety_file = os.path.join(output_dir, safety_file)
        utility_file = os.path.join(output_dir, utility_file)
        
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
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    critical_output_file = os.path.join(output_dir, f"critical_safety_neuron_{timestamp}.txt")
    
    logger.info(f"\nSaving Critical Safety Neurons to {critical_output_file}...")
    with open(critical_output_file, "w", encoding="utf-8") as f:
        # safety/foundation 파일과 동일한 JSON lines 포맷
        f.write(json.dumps({str(k): sorted(list(v)) for k, v in critical_neurons['ffn_up'].items()}) + "\n")
        f.write(json.dumps({str(k): sorted(list(v)) for k, v in critical_neurons['ffn_down'].items()}) + "\n")
        f.write(json.dumps({str(k): sorted(list(v)) for k, v in critical_neurons['q'].items()}) + "\n")
        f.write(json.dumps({str(k): sorted(list(v)) for k, v in critical_neurons['k'].items()}) + "\n")
        f.write(json.dumps({str(k): sorted(list(v)) for k, v in critical_neurons['v'].items()}) + "\n")
    
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
    
    for layer_idx in sorted(stats['layer_stats'].keys()):
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

    total_model_neurons = calculate_model_total_neurons(DEFAULT_MODEL_NAME)
    critical_pct_total = (critical_total / total_model_neurons * 100) if total_model_neurons > 0 else 0.0
    logger.info(f"  • Total model neurons (q/k/v/o + gate/up/down): {total_model_neurons:,}")
    logger.info(f"  • Critical neurons over total model neurons: {critical_pct_total:.4f}%")
    
    logger.info(f"\n✅ Critical Safety Neurons saved to: {critical_output_file}")
    logger.info(f"📝 Log saved to: {log_file}")
    logger.info("="*80)
    
    # Next steps
    logger.info("\n📋 Next Steps:")
    logger.info(f"  1. Review Critical Safety Neuron statistics above")
    logger.info(f"  2. Run Critical Safety Neuron fine-tuning with:")
    logger.info(f"     python critical_safety_neuron_tune.py {critical_output_file} ./corpus_all/circuit_breakers_train.json")


if __name__ == "__main__":
    main(sys.argv[1:])
