---
name: gpu-lol-debug
description: Debug gpu-lol analysis or cluster issues. Use when the wrong GPU is selected, VRAM estimate is off, packages are wrong, or a cluster isn't working right.
argument-hint: [repo_path | cluster_name]
allowed-tools: Bash, Read, Grep, Glob
context: fork
agent: general-purpose
---

Debug gpu-lol issue: $ARGUMENTS

## Current environment
!`cd /home/mb/Desktop/new-projects/gpu-lol && echo "=== Clusters ===" && .venv/bin/gpu-lol ls 2>&1 | head -15 && echo "=== LLM Config ===" && cat ~/.gpu-lol/config 2>/dev/null | grep -v KEY | grep -v key || echo "(no config)"`

## Debugging Runbook

### Is it an analysis problem? (wrong dry-run output)

Run with LLM disabled to isolate heuristic vs LLM issues:
```bash
GPU_LOL_LLM_URL="" gpu-lol up $ARGUMENTS --dry-run
```

Then with LLM enabled:
```bash
gpu-lol up $ARGUMENTS --dry-run
```

Compare. If heuristics are wrong → fix `analyzer.py`. If LLM is wrong → it's overriding correctly but with bad values.

**Trace the analysis pipeline:**

1. **Package extraction** (`_extract_packages`)
   - Reads `requirements.txt` in root + 1 level of subdirectories
   - Falls back to scanning `.py` imports if no requirements found
   - Filters stdlib, local modules, non-identifiers
   - Bug pattern: local module names leaking in (check `local_names` filter)

2. **Workload detection** (`_detect_workload_type`)
   - Checks `.py` filenames for: train/finetune/grpo/sft/serve/infer/vllm
   - Falls back to package names: trl/peft/accelerate → training, vllm → inference
   - Bug pattern: inference repo with a file called `train_utils.py` → misdetected as training

3. **VRAM estimation** (`_estimate_vram`)
   - Scans for size patterns: `70b`, `llama-7b`, `13B params`
   - Uses largest found size × precision overhead × workload overhead + 2GB buffer
   - Bug pattern: unrelated "7b" string in a comment triggering wrong estimate

4. **LLM override** (`llm.py`)
   - Validates gpu_type against GPU_CATALOG before accepting
   - Floors vram at 8GB
   - Bug pattern: LLM returns `"none"` for gpu_type (polymarket case — valid)

### Is it a cluster problem? (pod not working)

```bash
# Check what's actually on the pod
gpu-lol ssh $ARGUMENTS -- "nvidia-smi && python3 --version && pip list | head -20"
gpu-lol ssh $ARGUMENTS -- "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"

# Check setup script ran correctly
gpu-lol logs $ARGUMENTS

# Check code is there
gpu-lol ssh $ARGUMENTS -- "ls -la ~/sky_workdir/"
```

### Known issues and fixes

| Issue | Root cause | Fix |
|-------|------------|-----|
| `sky exec` hangs forever | `sky exec` queues jobs; `sleep infinity` is running | Use `sky ssh <cluster> -- <cmd>` instead (already fixed in skypilot.py) |
| RTX3090 unavailable | RunPod CZ/CA/US all sold out | Create `.gpu-lol.yaml` override with `gpu_type: RTX4090` |
| Container not found | Old RunPod image tag format | New format: `runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204` |
| `--detach-setup` error | SkyPilot 0.11 API change | Use `--detach-run` (already fixed in skypilot.py) |
| RunPod auth fails | Wrong config.toml format | Must use `[default]` not `[credentials]` |
| Double `/v1` in OpenRouter | URL already contains `/v1` | llm.py strips trailing `/v1` before appending path |

## Key source files
- `gpu_lol/analyzer.py` — `_extract_packages`, `_detect_workload_type`, `_estimate_vram`, `_select_gpu`
- `gpu_lol/llm.py` — `analyze_repo()`, `_load_config()`
- `gpu_lol/skypilot.py` — `exec()` uses `sky ssh`, `_sky()` resolver
- `~/.gpu-lol/config` — LLM URL, API keys
- `~/.runpod/config.toml` — must have `[default]` profile
