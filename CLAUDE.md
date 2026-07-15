# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for **Safety-Specific Neurons in LLMs** (ICLR 2025, "Understanding and Enhancing Safety Mechanisms of LLMs via Safety-Specific Neuron"). It identifies the small subset (<1% of params) of parameter-space neurons implementing an LLM's safety behavior, then uses that to (a) install/strengthen safety via **SN-Tune** and (b) preserve safety across downstream fine-tuning via **RSN-Tune** and safety-neuron-frozen fine-tuning.

Nearly all code lives in `neuron_detection/`. The repo root holds the papers (`SN-paper.md`, `Safety_Neuron_Detection.md`, `Safety_Neuron_Alignment_paper.md` — read these for methodology and math), the upstream `README.md`, and `experiment_neuron_detection.sh` (the current end-to-end pipeline).

There is **no test suite, linter, or build step** — these are research scripts run directly with `python`, from the `neuron_detection/` directory.

## Core concepts

- **Neuron = a parameter-space row/column**, not an activation: rows of `up_proj.weight` (FFN) and of `q_proj`/`k_proj`/`v_proj.weight` (attention). `ffn_down` reuses the up-projection indices, since deactivating `W_up[:,k]` ≡ deactivating `W_down[:,k]` — hence the axis flip (rows for up/q/k/v, **columns** for down) in every masking routine.
- **Importance** = L2 change in the hidden representation when a neuron is removed, computed *in parallel* rather than by ablating neurons one at a time (diagonal mask matrix for FFN/V; rank-1 softmax-delta for Q/K).
- **Safety neurons** `N_safe` = intersection of per-layer top-k important neurons across many harmful prompts (the consistency criterion).
- **Foundation/utility neurons** `N_foundation` = the same detection run over a general corpus (Wikipedia).
- **Critical/robust safety neurons** `N_robust = N_safe \ N_foundation` — the subset tunable without harming downstream task ability. RSN-Tune trains these; SN-Tune trains all of `N_safe`.

## Pipeline

Stages are standalone scripts that hand off via neuron files in `neuron_detection/output_neurons/`. `experiment_neuron_detection.sh` (repo root) wires the whole thing together and is the best reference for how stages actually compose — it also shows the two-conda-env split (detection in `hb_sn`, training in `hb`).

1. **Detection** — `safety_neuron_detection_v2_revised.py <num_prompts> --safety_neuron|--utility_neuron`. One script, both roles; the mode flags are mutually exclusive and **required**. Writes `safety_neuron_accelerated_<ts>.txt` or `utility_neurons_<n>_<ts>.txt`.
2. **Critical neurons** — `compute_critical_safety_neurons.py <safety_file> <utility_file>` (plain `sys.argv`, not argparse) → `critical_safety_neuron_<ts>.txt`. With no args it auto-discovers the newest matching files in `output_neurons/`.
3. **Percentage check** — `calculate_safety_neuron_percentage.py --neuron_file X --model_name Y`. Aim for ≤1% of params; tune `--top_number_ffn/--top_number_attn` to hit it.
4. **SN-Tune / RSN-Tune** — `sn_tune.py`: freezes everything **except** the listed neurons and fine-tunes on a safety corpus (Circuit Breakers). Same script for both; SN-Tune passes the safety file, RSN-Tune the critical file.
5. **Downstream FT with frozen safety neurons** — `finetune_{gsm8k,arc-c,hendrycks_math,mbpp,medqa}_freeze_sn.py`. **Inverse of SN-Tune**: trains all params but freezes the safety neurons so safety survives task fine-tuning. The `*_full_params.py` variants are the no-freezing baselines.
6. **Upload** — via `--upload_name` on the training scripts, or `upload_sn_tuned_model.py` standalone.

### Freezing mechanisms differ between the two directions

Both use `param.register_hook` gradient masks, but they are not symmetric, and the asymmetry is deliberate:

- `sn_tune.py` sets `requires_grad=False` on everything, re-enables only tensors containing safety neurons, and multiplies gradients by a **keep-mask** (`grad * mask`, mask=1 at safety indices).
- `finetune_*_freeze_sn.py` sets `requires_grad=True` everywhere and **zeroes** gradients at safety indices — plus registers a `SafetyNeuronRestoreCallback` that rewrites the frozen weights after each optimizer step. This exists because AdamW's weight-decay term (λθ) is applied independently of gradient hooks, so zeroed gradients alone would still let frozen weights drift toward 0.
- `sn_tune.py` has **no** such restore callback, so non-safety rows inside a `requires_grad=True` tensor can still be weight-decayed there.

### Neuron file format

Five lines, one JSON/Python-repr dict per line, `{layer_idx: [neuron_indices]}`. **Order is the contract — keys are never read from the file**, only assigned by line position:

```
line 0: ffn_up   line 1: ffn_down   line 2: q   line 3: k   line 4: v
```

In code the dict keys are `ffn_up, ffn_down, q, k, v` — **not** `attn_q/attn_k/attn_v`. (`--basis_layer_types` in the rotation scripts uses the `attn_*` spelling for a different purpose; don't conflate them.) `neuron_percentage_utils.py` silently skips unknown keys, so a mis-keyed dict counts as **zero** rather than erroring. Keep the 5-line structure intact when touching any reader/writer.

### Patched transformers — required for detection

The top-k detection scripts depend on a **patched `transformers`**. The patched modeling files live in `neuron_detection/transformers/models/{llama,mistral,gemma2}/` and `transformers/generation/`; they return extra `*_score` tensors that detection reads off the modules (`_last_ffn_up_score`, `_last_q_score`, …). Nothing manipulates `sys.path` — **you must copy these over the installed package's modeling files**, or detection fails with "Ensure patched modeling_llama.py is loaded."

**Which env has which patch (as of 2026-07-15):** conda env `hb_sn` (detection) has a patched `models/qwen2/modeling_qwen2.py` — byte-identical to the repo's `transformers/models/modeling_qwen2 (1).py`. Its `llama` is **stock**, so Llama detection would need the patch copied in first. Env `hb` (training) is stock throughout, which is correct — training must not see the patch.

**The patch targets an older transformers than what is installed (4.57.3).** One incompatibility was found and fixed on 2026-07-15: `modeling_qwen2.py:465` used `@check_model_inputs` where 4.57.3 requires `@check_model_inputs()` — in 4.57.3 that decorator became a factory taking `tie_last_hidden_states`, so the bare form replaces `Qwen2Model.forward` with the inner `wrapped_fn` and every forward dies with `TypeError: wrapped_fn() got an unexpected keyword argument 'input_ids'`. Fixed in both the installed copy and the repo copy. **Assume more version drift lurks** in the other patched families — if a new family or a transformers upgrade is introduced, diff the patch against the stock file of that exact version before trusting it. Sanity check: `inspect.signature(Qwen2Model.forward)` must start with `['self', 'input_ids', ...]`, not `['func']`.

The patched files use **CRLF line endings** throughout — `sed` patterns anchored with `$` silently fail to match. Anchor on the token instead, or account for `\r`.

Note the two algorithm families and their differing dependencies:

- **Fixed per-layer top-k** — `safety_neuron_detection_v2_revised.py`, `safety_neuron_detection_v2_basis_rotation.py`. Read scores stashed by the patch. Take `--top_number_ffn/--top_number_attn`. **Require the patch.**
- **Global-fraction** — `foundation_neuron_detection.py`, `safety_neuron_detection_rotation.py`. Register their own forward hooks and compute importance themselves. Take `--ffn_active_fraction/--attn_active_fraction`. **Do not need the patch.**

`experiment_neuron_detection.sh` has an opt-in patch check (`EXPECTED_MODELING_LLAMA`) that compares `modeling_llama.__file__` against an expected path; it is **skipped by default** since the variable defaults to empty.

### Basis rotation

`safety_neuron_detection_v2_basis_rotation.py` is a strict superset of `_revised`'s CLI, adding `--use_basis_rotation_score --basis_dir <dir>` to detect on a **basis-rotated** model for a larger, more reliable intersection, plus `--attn_implementation` (`_revised` hardcodes slow `eager`). Rotation is right-multiplication (`W @ V`), so **row indices are preserved** and rotation-space neurons map 1:1 to original-space rows; `map_rotated_to_original_neurons.py` only reformats. Critically: detect on the rotated model, but **run SN-Tune and downstream FT on the ORIGINAL model** — rotation breaks inter-layer consistency.

`safety_neuron_detection_rotation.py` is a different animal: it scores **WaRP basis directions** (columns of `U`), so its indices mean something other than the other three despite the identical file format. It requires `--basis_dir`.

## Running

```sh
cd neuron_detection

# 1. Foundation/utility neurons
python safety_neuron_detection_v2_revised.py 1000 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --top_number_ffn 1200 --top_number_attn 200 --utility_neuron

# 2. Safety neurons
python safety_neuron_detection_v2_revised.py 4994 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --top_number_ffn 600 --top_number_attn 100 --safety_neuron

# 3. Critical (robust) safety neurons — positional, no flags
python compute_critical_safety_neurons.py <safety_file>.txt <utility_file>.txt

# 4. SN-Tune (safety file) / RSN-Tune (critical file)
python sn_tune.py \
    --neuron_file ./output_neurons/<file>.txt \
    --dataset_file <path>/circuit_breakers_train.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --local_model_name ./sn_tuned_out \
    --learning_rate 5e-5 --upload_name kmseong/<repo>

# 5. Downstream FT freezing safety neurons
python finetune_gsm8k_freeze_sn.py \
    --model_path <sn_tuned_model_or_hf_id> \
    --safety_neurons_file <path_to_neuron_file>.txt \
    --output_dir ./out --learning_rate 5e-5 --epochs 3 \
    --upload_name kmseong/<repo>
```

## Gotchas

- **Detection swallows its own failures and emits empty neuron files.** Each prompt runs in a `try/except` that logs and continues, so a 100%-failure run still exits 0 and writes `{"0": [], "1": [], ...}` for all 5 sections. Nothing downstream validates this: `compute_critical_safety_neurons.py` happily returns an empty difference, and the first real error surfaces hours later as `ValueError: optimizer got an empty parameter list` in `sn_tune.py`. **Always verify a detection run before letting it proceed** — the output filename carries the count (`utility_neurons_0_...` means zero detected), and the log line `Detection complete: success=N, failed=M` is the authoritative check.
- **`corpus_all/circuit_breakers_train.json` is git-ignored and must be supplied locally** (copied there on 2026-07-15 from `/home/edgeai_lab/Safety-WaRP-LLM/data/`, 4994 entries — matching the `4994` the scripts expect). The detection scripts hardcode that exact path with **no CLI override** and `sys.exit(1)` if absent. Note `experiment_neuron_detection.sh`'s `SAFETY_DATASET_FILE` only reaches `sn_tune.py`, never detection — the two read the corpus from different places.
- **`sn_tune.py` appends `_lr<lr>_<timestamp>` to `--local_model_name`** (`:913`), so the saved directory never equals the name you passed. The original script therefore relied on the HF upload succeeding for downstream stages to find the model; `experiment_neuron_detection.sh` now resolves the real directory and passes it to `--model_path` directly.
- **`CUDA_VISIBLE_DEVICES` is hardcoded per script** and the form matters: `sn_tune.py` (7), `finetune_gsm8k_freeze_sn.py` (5) and `_revised.py` (7) use `setdefault`, so a shell `export` wins; `_basis_rotation.py` (7), `_rotation.py` (0), `foundation_neuron_detection.py` (1) and `finetune_gsm8k_full_params.py` (2,3) use hard assignment and **silently override the shell**. All then load with `device_map={"": 0}`.
- **`sn_tune.py` has no `--epochs` flag** — `NUM_EPOCHS = 3` is hardcoded, alongside `BATCH_SIZE=4`, `GRAD_ACCUM_STEPS=4`, `MAX_SEQ_LENGTH=1024`, `MAX_SAMPLES=4994`, `NUM_LAYERS=32`.
- **Defaults drift from the paper and from each other.** The paper's SN-Tune is lr 1e-6 / 1 epoch / ~50 docs, but `sn_tune.py` defaults to 5e-5 / 3 epochs; `finetune_gsm8k_freeze_sn.py` defaults to lr **7e-5** while `finetune_gsm8k_full_params.py` uses 5e-5. The README's top section gives full-FT lr as 3e-5 (base) / 5e-5 (instruct), 3 epochs. **Set these explicitly per experiment** rather than trusting a default.
- **Detection output filenames collide**: `_revised.py` and `_basis_rotation.py` emit byte-identical `safety_neuron_accelerated_<ts>.txt` / `utility_neurons_<n>_<ts>.txt` patterns, as does `foundation_neuron_detection.py` for the utility one. The shell's `find_latest_neuron_file` picks by `ls -1t | head -1`, so it can silently grab a stale file from a different producer.
- **`.gitignore` has unresolved merge conflict markers** (~lines 137-148, around the `logs/` allow-list rules). Resolve before committing anything touching it.
- **Model type auto-detection**: `is_instruct_model()` keys on "instruct"/"chat" in the model name to decide whether to apply the chat template. Naming matters.
- **Absolute paths in docstrings/defaults are stale** (`/home/yonsei_jong/...`, `/NHNHOME/...`) — e.g. `finetune_gsm8k_full_params.py --safety_data_path` and `_basis_rotation.py --basis_dir` default to non-existent machine-specific paths. Treat them as illustrative.
- **wandb**: only `finetune_gsm8k_full_params.py` uses it, and it hardcodes `report_to="wandb"` in `TrainingArguments`, making its own `--report_to` flag dead. The freeze/SN-Tune scripts don't use wandb.
- Outputs are git-ignored (`logs/`, `output_neurons/` content, `wandb/`, `*.safetensors`, model dirs). `corpus_all/`, `cache/`, `wikipedia_cache/` are ignored and must be populated locally.
