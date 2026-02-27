# gpu-lol

AI-powered GPU environment manager. Reads your repo, picks the right GPU, launches it.

## Install

```bash
pip install git+https://github.com/miike-lol/gpu-lol
```

Or with uv (faster):
```bash
uv pip install git+https://github.com/miike-lol/gpu-lol
```

**Requirements:** Python 3.10+, a [RunPod account](https://www.runpod.io) with a payment method added.

## Setup

```bash
gpu-lol secrets init
```

This walks you through:
- RunPod API key (required — get it at runpod.io/console/user/settings)
- Vast.ai / Lambda Labs keys (optional — more GPU availability)
- LLM config (optional — makes project analysis smarter)

Then in any ML project:

```bash
cd my-llm-project
gpu-lol up
```

```
🔍 Analyzing my-llm-project...
  LLM: QLoRA fine-tuning of 7B model needs 24GB VRAM...
✓ Environment detected:
  Workload:  training
  GPU:       RTX4090 (needs 24GB VRAM)
  Image:     runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204
  Packages:  8 dependencies

🚀 Launching cluster 'gr-my-llm-project-48291'...

✨ Ready!

  SSH:       ssh -F ~/.sky/generated/ssh/gr-my-llm-project-48291 gr-my-llm-project-48291
  Snapshot:  gpu-lol snapshot gr-my-llm-project-48291
  Stop:      gpu-lol stop gr-my-llm-project-48291
  Down:      gpu-lol down gr-my-llm-project-48291

  Your code is at ~/sky_workdir/ on the pod
```

Under the hood, gpu-lol just:

- Detected training workload from your code ✅
- Picked the right GPU for the exact VRAM requirement ✅
- Selected the correct RunPod PyTorch image ✅
- Routed across RunPod → Vast.ai → Lambda for cheapest price ✅
- Synced your local repo to `~/sky_workdir/` on the pod ✅
- Auto-installed Claude Code on the pod ✅

No zip files. No JupyterLab. No manual installs.

## How it works

1. Reads your repo (`requirements.txt`, imports, filenames)
2. Scans `from_pretrained()` calls and maps known model IDs (Llama, Qwen, Mistral, DeepSeek, etc.) to exact VRAM
3. Detects multi-GPU patterns (DDP, FSDP, torchrun) and sets gpu_count automatically
4. LLM analyzes the repo for smarter detection (optional — falls back to heuristics)
5. Detects workload type (training / inference / interactive)
6. Estimates VRAM needed (scans for model size hints like `llama-70b`)
7. Parses Dockerfile for base images and packages
8. Picks cheapest GPU with enough VRAM across RunPod, Vast.ai, Lambda
9. Generates SkyPilot task YAML and launches
10. Auto-symlinks large assets (HF cache, datasets) from RunPod network volumes
11. Validates CUDA + packages before handing over SSH

## Commands

```bash
# Launch (updated)
gpu-lol up                          # analyze + launch (shows cost confirmation)
gpu-lol up ~/my-project             # specific repo
gpu-lol up --dry-run                # preview without spending
gpu-lol up --yes                    # skip cost confirmation prompt
gpu-lol up --gpu A40                # force GPU type
gpu-lol up --gpus 4                 # force GPU count (multi-GPU)
gpu-lol up --detach                 # fire-and-forget, return immediately
gpu-lol up --detach --watch         # fire-and-forget but tail the launch log
gpu-lol up --stop-after 4           # auto-stop after 4 idle hours
gpu-lol up --template porpoise      # use specific RunPod template (fast boot)
gpu-lol up --assets /workspace/models:/root/.cache/huggingface/hub  # symlink assets

# Templates (new)
gpu-lol templates                   # list RunPod templates (grouped, highlighted)
gpu-lol templates --filter porpoise # filter by name

# Interact
gpu-lol ssh <cluster>          # drop into cluster shell
gpu-lol exec <cluster> -- cmd  # run a command on the cluster
gpu-lol logs <cluster>         # stream setup/job logs
gpu-lol validate <cluster>     # check CUDA, VRAM, packages
gpu-lol info <cluster>         # show GPU type, IP, port, cost/hr, SSH command

# Lifecycle
gpu-lol ls                     # list running clusters
gpu-lol start <cluster>        # restart stopped cluster
gpu-lol stop <cluster>         # pause billing (state preserved)
gpu-lol down <cluster>         # terminate permanently
gpu-lol down --all             # kill everything (lists clusters, requires typing "yes")

# Environment
gpu-lol snapshot <cluster>     # capture running env to .gpu-lol.yaml
gpu-lol resume                 # restore from .gpu-lol.yaml
gpu-lol check                  # verify cloud credentials

# Credentials
gpu-lol secrets init           # first-time setup wizard
gpu-lol secrets show           # display config (keys masked)
gpu-lol secrets set KEY=value  # update a single credential
```

## Launch behavior

Before launching, gpu-lol shows the estimated cost and asks to confirm:

```
Launch 1x RTX4090 @ ~$0.34/hr? [Y/n]
```

Skip with `--yes` / `-y`. If no ML packages are detected, you'll also get a warning before it proceeds.

If `~/.gpu-lol/.env` doesn't exist yet, `gpu-lol up` redirects automatically to `gpu-lol secrets init` instead of failing.

If `HF_TOKEN` is set in your secrets, it is automatically written to `~/.cache/huggingface/token` on the pod at launch time.

## Save your environment

After getting your environment right, snapshot it:

```bash
gpu-lol snapshot my-cluster
git add .gpu-lol.yaml && git commit -m "save gpu env"
```

Next time, skip analysis entirely:

```bash
gpu-lol resume   # exact same environment, new pod
```

## Credentials

```bash
gpu-lol secrets init   # prompts for RunPod key + LLM config, encrypts automatically
```

Credentials are encrypted at rest with [dotenvx](https://dotenvx.com). Your keys never sit in plaintext.

`gpu-lol secrets init` also handles Vast.ai (prompts for API key → writes `~/.vast_api_key`) and Lambda Labs (prompts for API key → writes `~/.lambda_cloud/lambda_keys.yaml`).

## Fast boot with RunPod templates

gpu-lol automatically selects your pre-cached RunPod templates (they boot in seconds vs minutes).

```bash
gpu-lol templates                    # see all your templates
gpu-lol up --template competitive_salmon_porpoise  # pin a specific one
```

To set your preferred template family:
```bash
gpu-lol secrets set GPU_LOL_TEMPLATE_PATTERN=my_template_prefix
```

## LLM integration

gpu-lol uses an LLM to analyze your repo for smarter workload detection — understanding context like "this is QLoRA fine-tuning of a 70B model" rather than just counting imports.

Configure any OpenAI-compatible endpoint:

```bash
gpu-lol secrets set GPU_LOL_LLM_URL=https://openrouter.ai/api/v1
gpu-lol secrets set GPU_LOL_LLM_KEY=sk-or-yourkey
gpu-lol secrets set GPU_LOL_LLM_MODEL=anthropic/claude-sonnet-4-6
```

Or use a local Ollama instance (no key needed):
```bash
gpu-lol secrets set GPU_LOL_LLM_URL=http://localhost:11434
gpu-lol secrets set GPU_LOL_LLM_KEY=
gpu-lol secrets set GPU_LOL_LLM_MODEL=llama3.2
```

To disable LLM and use heuristics only:
```bash
gpu-lol secrets set GPU_LOL_LLM_URL=
```

Without LLM, gpu-lol falls back to heuristics: package names, filenames, `from_pretrained()` calls, and known model sizes. It still works well for most projects.

## Environment variables

All config lives in `~/.gpu-lol/.env` (encrypted). You can set any key via `gpu-lol secrets set KEY=value`.

| Variable | Description |
|----------|-------------|
| `RUNPOD_API_KEY` | RunPod API key (required) |
| `VAST_API_KEY` | Vast.ai API key (optional) |
| `LAMBDA_API_KEY` | Lambda Labs API key (optional) |
| `HF_TOKEN` | HuggingFace token — auto-written to pod at launch |
| `GPU_LOL_LLM_URL` | LLM endpoint (any OpenAI-compatible URL) |
| `GPU_LOL_LLM_KEY` | LLM API key |
| `GPU_LOL_LLM_MODEL` | LLM model name |
| `GPU_LOL_TEMPLATE_PATTERN` | RunPod template name prefix to prefer |

## Providers

gpu-lol uses [SkyPilot](https://skypilot.co) to automatically find the cheapest available GPU across:
- RunPod
- Vast.ai
- Lambda Labs

You never pick a provider. gpu-lol picks for you.

## GPU catalog

| GPU | VRAM | ~$/hr |
|-----|------|-------|
| RTX3090 | 24GB | $0.22 |
| RTX4090 | 24GB | $0.34 |
| A40 | 48GB | $0.40 |
| A6000 | 48GB | $0.50 |
| A100-SXM4 | 80GB | $1.19 |
| H100-SXM | 80GB | $2.49 |
