# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **Safety-Specific Neurons in LLMs** (ICLR 2025, "Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron"). It detects the small subset (<1%) of parameter-space neurons that implement an LLM's safety behavior, then uses that knowledge to (a) install/strengthen safety via **SN-Tune** and (b) preserve safety during downstream fine-tuning via **RSN-Tune** and safety-neuron-frozen fine-tuning.

The git repo root is the **parent** directory (`Safety-Neuron/`); active work happens in this `neuron_detection/` subdirectory. Code, papers, and a README live at the repo root (`../README.md`, `../SN-paper.md`, `../Safety_Neuron_Detection.md` — read these for the methodology and math).

## Core concepts

- **Neuron = parameter-space row/column**, not an activation. Specifically: rows of `up_proj.weight` (FFN), and rows of `q_proj`/`k_proj`/`v_proj.weight` (attention). `ffn_down` reuses the up-projection indices (deactivating `W_up[:,k]` ≡ deactivating `W_down[:,k]`).
- **Importance** = L2 change in hidden representation when a neuron is removed, computed *in parallel* via the patched transformers (mask matrix for FFN/V, rank-1 softmax-delta for Q/K). Scores are accumulated inside the forward pass.
- **Safety neurons** `N_safe` = intersection of per-layer top-k important neurons across many harmful prompts (consistency criterion).
- **Foundation/Utility neurons** `N_foundation` = same detection run on a general corpus (Wikipedia).
- **Critical/Robust safety neurons** `N_robust = N_safe \ N_foundation` — the subset safe to tune without harming downstream task ability.

## Architecture & data flow

The pipeline is a sequence of standalone scripts that pass data via `.txt`/`.json` neuron files in `output_neurons/`:

1. **Detection** — `safety_neuron_detection_v2_basis_rotation.py` (safety neurons), `foundation_neuron_detection.py` (utility/foundation neurons). Output: 5-section neuron file (`ffn_up`, `ffn_down`, `attn_q`, `attn_k`, `attn_v`), each a JSON dict `{layer_idx: [neuron_indices]}`.
2. **Critical neuron computation** — `compute_critical_safety_neurons.py <safety_file> <utility_file>` → `critical_safety_neuron_*.txt` (set difference).
3. **SN-Tune / RSN-Tune** — `sn_tune.py`: loads a neuron file, **freezes all params except** the listed neurons, fine-tunes on a safety corpus (Circuit Breakers). SN-Tune uses the safety file; RSN-Tune uses the critical file. Same script, different `--neuron_file`.
4. **Downstream fine-tune with frozen safety neurons** — `finetune_gsm8k_freeze_sn.py`, `finetune_arc-c_freeze_sn.py`, `finetune_hendrycks_math_freeze_sn.py`, `finetune_mbpp_freeze_sn.py`, `finetune_medqa_freeze_sn.py`. **Inverse of SN-Tune**: trains all params but freezes the safety neurons (gradient-zeroing hooks on weight rows/cols) so safety survives task fine-tuning. `*_full_params.py` variants are baselines (no freezing).
5. **Upload** — `upload_sn_tuned_model.py` (supports `--upload_pair MODEL_PATH REPO_ID` repeated), or via `--upload_name` on the training scripts.

End-to-end orchestration: `run_sn_pipeline.sh` (rotation-space variant) and `run_sn_rsn_then_gsm8k_freeze.sh` (SN → RSN → two GSM8K freezes). Edit the Configuration block at the top of these scripts.

### Patched transformers — REQUIRED

Detection depends on a **patched `transformers` package**. The patched modeling files live in `./transformers/models/{llama,mistral,gemma2,qwen2}/` and `./transformers/generation/`. These compute and return the extra `*_score` tensors (e.g. `modeling_llama.py` returns `hidden_score_fwd_up/down`, `hidden_score_q/k/v/o` from each layer; detection reads `_last_ffn_up_score`, `_last_q_score`, etc. off the modules). **You must copy these over the installed `transformers` package's modeling files** for detection to work — a stock transformers install will not produce the scores and detection will fail with "Ensure patched modeling_llama.py is loaded." Supported model families: Llama, Mistral, Gemma2, Qwen2.

### Basis rotation

`safety_neuron_detection_v2_basis_rotation.py` can detect on a **basis-rotated** model (`--use_basis_rotation_score --basis_dir <dir>`) for a larger, more reliable intersection. Rotation is right-multiplication (`W @ V`), so **row indices are preserved** — neurons detected in rotation space map 1:1 to original-space rows. `map_rotated_to_original_neurons.py` just reformats the file (no index change). Critically: detect on the rotated model, but **SN-Tune and downstream fine-tuning run on the ORIGINAL model** (rotation breaks inter-layer consistency).

## Running

There is no test suite, linter, or build step — these are research scripts run directly with `python`. Always run from this `neuron_detection/` directory.

```sh
# 1. Detect safety neurons (first positional arg = num prompts for intersection)
python safety_neuron_detection_v2_basis_rotation.py 4994 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --top_number_ffn 1800 --top_number_attn 300 \
    --safety_neuron --attn_implementation flash_attention_2

# 2. Detect foundation/utility neurons (fractions ~0.01–0.05)
python foundation_neuron_detection.py 1000 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --ffn_active_fraction 0.01 --attn_active_fraction 0.01

# 3. Critical (robust) safety neurons
python compute_critical_safety_neurons.py <safety_file>.txt <utility_file>.txt

# 4. SN-Tune (freeze everything except safety neurons)
python sn_tune.py \
    --neuron_file ./output_neurons/<safety_file>.txt \
    --dataset_file ./corpus_all/circuit_breakers_train.json \
    --local_model_name ./only_sn_tuned_model_llama2_7b_chat_lr5e-5 \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --upload_name kmseong/<repo>

# 5. Downstream fine-tune freezing safety neurons
python finetune_gsm8k_freeze_sn.py \
    --model_path <sn_tuned_model_or_hf_id> \
    --safety_neurons_file <abs_path_to_safety_file>.txt \
    --output_dir ./out --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/<repo>
```

Most scripts hardcode `os.environ["CUDA_VISIBLE_DEVICES"]` near the top (e.g. `"7"` in detection/tuning, `"1"` in foundation detection) — **change this to target a different GPU**, it is not a CLI flag.

## Conventions & gotchas

- **The README's top section is the source of truth for hyperparameters**: full-FT lr is 3e-5 (base) / 5e-5 (instruct), 3 epochs. SN-Tune in the paper uses tiny lr (1e-6) / 1 epoch / ~50 docs, but the scripts default higher (e.g. `sn_tune.py` defaults lr 5e-5, 3 epochs) — set explicitly per experiment.
- **Detection target fraction**: aim for safety neurons ≈ ≤1% of model params; tune `--top_number_ffn/--top_number_attn` (fixed per-layer top-k) or `--*_active_fraction` to hit it. `calculate_safety_neuron_percentage.py` / `neuron_percentage_utils.py` compute the percentage.
- **Model type auto-detection**: `is_instruct_model()` keys on "instruct"/"chat" in the name → controls whether `apply_chat_template` is used. Naming matters.
- **Neuron file format** is shared across all stages; when editing one stage's reader/writer keep the 5-key (`ffn_up/ffn_down/attn_q/attn_k/attn_v`) `{layer: [indices]}` structure intact.
- **Paths in example docstrings are absolute and machine-specific** (`/home/yonsei_jong/...`, `/NHNHOME/...`) — they are stale references, not literal; use paths relative to this dir.
- Outputs/logs are git-ignored (`logs/`, `output_neurons/` content, `wandb/`, `*.safetensors`, model dirs). `corpus_all/`, `cache/`, `wikipedia_cache/` are also ignored and must be populated locally.
- `../.gitignore` currently has **unresolved merge conflict markers** (`<<<<<<< HEAD` / `=======` / `>>>>>>>`) around the `logs/` allow-list rules — resolve before committing anything that touches it.
