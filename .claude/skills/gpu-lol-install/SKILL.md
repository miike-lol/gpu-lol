---
name: gpu-lol-install
description: First-time setup of gpu-lol — install dependencies, configure credentials, and verify everything works. Use when someone is setting up gpu-lol for the first time or when credentials are broken.
argument-hint: [--from-source]
allowed-tools: Bash, Read, Write
---

Set up gpu-lol from scratch.

## Current state
!`echo "=== Python ===" && python3 --version 2>&1 && echo "=== gpu-lol ===" && gpu-lol --version 2>/dev/null || echo "not installed" && echo "=== sky ===" && sky --version 2>/dev/null || echo "not installed" && echo "=== RunPod ===" && cat ~/.runpod/config.toml 2>/dev/null || echo "not configured" && echo "=== gpu-lol credentials ===" && if [ -f ~/.gpu-lol/.env ]; then grep -q DOTENV_PUBLIC_KEY ~/.gpu-lol/.env && echo "encrypted (.env)" || echo "plain text (.env)"; else echo "not configured"; fi`

## Steps

### 1. Install gpu-lol

```bash
pip install git+https://github.com/miike-lol/gpu-lol
```

From source (development):
```bash
cd /home/mb/Desktop/new-projects/gpu-lol
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Install SkyPilot

```bash
pip install "skypilot[runpod,vast,lambda]"
```

### 3. Configure credentials

```bash
gpu-lol secrets init
```

This interactive wizard configures all three cloud providers in one go:
- Prompts for RunPod API key → https://www.runpod.io/console/user/settings
- Prompts for Vast.ai API key → writes `~/.vast_api_key`
- Prompts for Lambda Cloud API key → writes `~/.lambda_cloud/lambda_keys.yaml`
- Prompts for LLM URL + key (optional — heuristics work without it)
- Encrypts everything with dotenvx automatically
- Writes `~/.runpod/config.toml` for SkyPilot

To update a single key later:
```bash
gpu-lol secrets set RUNPOD_API_KEY=rpa_newkey
gpu-lol secrets show   # verify (keys masked)
```

### 4. Verify

```bash
gpu-lol check                          # cloud credentials
gpu-lol up . --dry-run                 # test analysis pipeline
GPU_LOL_LLM_URL="" gpu-lol up . --dry-run  # test heuristics without LLM
```

If `gpu-lol check` shows RunPod as disabled, run `gpu-lol secrets init` again — it will fix `~/.runpod/config.toml`.

### 5. Run tests (developers only)

```bash
pytest tests/ -v
```

All 32 tests should pass with no GPU or network required.
