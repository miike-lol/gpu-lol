---
name: gpu-lol-llm
description: Configure, test, or debug the LLM integration for smarter repo analysis. Use when asked to set up LLM, change models, switch to local LLM, or debug why analysis is wrong.
argument-hint: [test | config | reset]
allowed-tools: Bash, Read, Write
---

Manage gpu-lol LLM integration: $ARGUMENTS

## Current LLM config
!`cat ~/.gpu-lol/config 2>/dev/null | grep -E "LLM_URL|LLM_MODEL" || echo "(no LLM configured — heuristics only)"`

## How the LLM works

The LLM gets a directory listing + file samples and returns:
```json
{
  "workload_type": "training",
  "vram_required_gb": 24,
  "gpu_type": "RTX4090",
  "reasoning": "QLoRA fine-tuning of 7B model..."
}
```

It **overrides** the heuristic result, but with guardrails:
- `gpu_type` must be in the GPU catalog or it's rejected
- `vram_required_gb` is floored at 8GB
- Returns `None` → heuristics used

Source: `gpu_lol/llm.py` — `analyze_repo()`

## Configure

Edit `~/.gpu-lol/config`:

**OpenRouter (recommended — any model, pay per token):**
```
GPU_LOL_LLM_URL=https://openrouter.ai/api/v1
GPU_LOL_LLM_KEY=sk-or-v1-YOUR_KEY
GPU_LOL_LLM_MODEL=x-ai/grok-4.1-fast
```

**Anthropic direct:**
```
GPU_LOL_LLM_URL=https://api.anthropic.com
GPU_LOL_LLM_KEY=sk-ant-YOUR_KEY
GPU_LOL_LLM_MODEL=claude-sonnet-4-6
```

**Ollama (100% local, free):**
```
GPU_LOL_LLM_URL=http://localhost:11434
GPU_LOL_LLM_MODEL=llama3.2
```

**Disable LLM (heuristics only):**
```bash
unset GPU_LOL_LLM_URL
# or just don't set it in ~/.gpu-lol/config
```

## Test the LLM

```bash
# See what the LLM says vs heuristics
GPU_LOL_LLM_URL="" gpu-lol up ~/my-project --dry-run  # heuristics only
gpu-lol up ~/my-project --dry-run                      # with LLM

# The LLM reasoning is printed during analysis:
#   LLM: QLoRA fine-tuning of 7B model needs 24GB VRAM...
```

## Debug LLM issues

**LLM picking wrong GPU:**
The LLM's `gpu_type` is validated against `GPU_CATALOG` in `analyzer.py`. If it returns an unknown GPU ID, it's silently rejected and heuristics are used. Check what the LLM is actually returning:

```python
# In gpu_lol/llm.py, temporarily add:
print("LLM raw response:", result)
```

**LLM overriding to wrong VRAM:**
VRAM is floored at 8GB but not capped. If the LLM returns an unreasonably high number, check if it's reading the repo content correctly.

**Double /v1 in URL:**
If using OpenRouter, the URL `https://openrouter.ai/api/v1` already contains `/v1`. `llm.py` strips trailing `/v1` before appending the endpoint path. If you get 404s, check `gpu_lol/llm.py` `analyze_repo()`.
