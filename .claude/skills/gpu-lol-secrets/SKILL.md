---
name: gpu-lol-secrets
description: Manage gpu-lol credentials — RunPod key, LLM config, encryption. Use when asked to set up keys, rotate credentials, check what's configured, or fix auth issues.
argument-hint: [init | show | set KEY=VALUE | encrypt]
allowed-tools: Bash
disable-model-invocation: true
---

Manage gpu-lol secrets: $ARGUMENTS

## Current credentials
!`cd /home/mb/Desktop/new-projects/gpu-lol && .venv/bin/gpu-lol secrets show 2>&1`

## CLI commands

```bash
gpu-lol secrets init           # interactive wizard — set all keys, encrypt
gpu-lol secrets show           # display config (keys masked)
gpu-lol secrets set KEY=value  # update a single key
gpu-lol secrets encrypt        # encrypt plain .env (if not already encrypted)
```

## Common tasks

### First-time setup
```bash
gpu-lol secrets init
```
Prompts for RunPod API key, Vast.ai API key, Lambda Cloud API key, and LLM config, then encrypts everything automatically. One command configures all three cloud providers.

### Rotate RunPod key
```bash
gpu-lol secrets set RUNPOD_API_KEY=rpa_newkey
gpu-lol check   # verify it works
```

### Switch LLM model
```bash
gpu-lol secrets set GPU_LOL_LLM_MODEL=claude-sonnet-4-6
```

### Switch LLM provider
```bash
gpu-lol secrets set GPU_LOL_LLM_URL=https://api.anthropic.com
gpu-lol secrets set GPU_LOL_LLM_KEY=sk-ant-newkey
gpu-lol secrets set GPU_LOL_LLM_MODEL=claude-sonnet-4-6
```

### Use local Ollama (no key needed)
```bash
gpu-lol secrets set GPU_LOL_LLM_URL=http://localhost:11434
gpu-lol secrets set GPU_LOL_LLM_KEY=
gpu-lol secrets set GPU_LOL_LLM_MODEL=llama3.2
```

### Disable LLM (heuristics only)
```bash
gpu-lol secrets set GPU_LOL_LLM_URL=
```

### Set Hugging Face token (auto-injected into pods)
```bash
gpu-lol secrets set HF_TOKEN=hf_yourtoken
```
When set, gpu-lol automatically writes this token to `~/.cache/huggingface/token` on every pod at launch time — no manual `huggingface-cli login` needed.

### Set template pattern
```bash
gpu-lol secrets set GPU_LOL_TEMPLATE_PATTERN=my_template_prefix
```
Controls which RunPod template family is preferred when launching pods. Default: `competitive_salmon_porpoise`. Run `gpu-lol templates` to list available templates.

## If credentials are broken

1. Check what's currently set:
   ```bash
   gpu-lol secrets show
   ```
2. Re-run init to reset everything:
   ```bash
   gpu-lol secrets init
   ```
3. Verify RunPod is enabled:
   ```bash
   gpu-lol check
   ```

## Security model

| File | Commit? | Notes |
|------|---------|-------|
| `~/.gpu-lol/.env` | Never | Encrypted ciphertext — safe to backup |
| `~/.gpu-lol/.env.keys` | **Never** | Private key — back up to password manager |
| `.env.example` | Yes | Template with no real values |

Private key at `~/.gpu-lol/.env.keys` — if you lose this, you'll need to re-run `gpu-lol secrets init`.
