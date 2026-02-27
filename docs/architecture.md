# gpu-lol Architecture

## The Pipeline

```
repo/ ──► analyzer.py ──► EnvironmentSpec ──► skypilot.py ──► SkyPilot YAML ──► cloud GPU
              │                                     │
           llm.py                             validator.py
          (override)                          (smoke test)
```

## EnvironmentSpec — the universal artifact

`gpu_lol/config.py` — everything flows through this dataclass.

```python
EnvironmentSpec(
    gpu_type="RTX4090",        # SkyPilot accelerator ID
    vram_required_gb=24,       # minimum VRAM
    base_image="runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204",
    requirements=["torch", "peft", "trl"],
    setup_commands=["pip install -r requirements.txt"],
    workload_type="training",  # training | inference | interactive
)
```

Serializes to/from `.gpu-lol.yaml`. That file is the reproducibility artifact — commit it.

## analyzer.py — the moat

Analysis priority (highest wins):

1. `.gpu-lol.yaml` in repo root — user config wins, **stop here**
2. LLM override via `GPU_LOL_LLM_URL` — smarter analysis if configured
3. Heuristics — always available, no external dependencies

### Heuristic pipeline

```
_extract_packages()
  → requirements.txt (root + subdirs)
  → pyproject.toml [project.dependencies] + [project.optional-dependencies]
  → setup.py install_requires
  → Dockerfile RUN pip install lines        ← NEW
  → fallback: scan .py imports (filtered against stdlib + local modules)

_detect_workload_type()
  → .py filenames: train/finetune/grpo/sft → "training"
                   serve/inference/vllm → "inference"
  → package names: trl/peft/accelerate → "training"
                   vllm/ray → "inference"
  → default: "interactive"

_estimate_vram()
  → KNOWN_MODEL_SIZES: from_pretrained("llama-3-70b") → 70B → 140GB fp16  ← NEW
  → scan files for size patterns: "70b", "llama-7b", "13B params"
  → VRAM_ESTIMATES[size][precision] × overhead + 2GB buffer
  → fallback: training=40GB, inference=16GB, interactive=16GB

_detect_gpu_count()                         ← NEW
  → scan for: torchrun, DDP, FSDP, deepspeed, accelerate launch
  → extract nproc_per_node / num_gpus if explicit
  → default multi-GPU: 2

_detect_dockerfile_image()                  ← NEW
  → FROM cuda/pytorch/tensorflow/nvidia images → use as base image

_select_gpu(vram)
  → GPU_CATALOG ordered cheapest-first
  → return first with vram >= required
  → fallback: H100-SXM (biggest)
```

## skypilot.py — provider abstraction

Never call RunPod/Vast.ai APIs directly. SkyPilot handles routing.

```python
resources:
  image_id: docker:runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204
  accelerators: RTX4090:1
  any_of:
    - cloud: runpod
    - cloud: vast
    - cloud: lambda
```

SkyPilot tries each provider in order, picks cheapest available.

**Key: `exec()` uses `ssh -F ~/.sky/generated/ssh/<cluster>` — `sky ssh` redesigned in 0.11 for node pools, not cluster SSH**

`sky exec` queues a job — hangs when `sleep infinity` is running.
`ssh -F ~/.sky/generated/ssh/<cluster> <cluster> -- <cmd>` runs inline over SSH, returns immediately.

## runpod_api.py — fast boot via cached templates

Fetches user's RunPod pod templates via GraphQL API. Pre-cached images boot in seconds.

Priority in `best_ml_template()`:
1. User's preferred template pattern (`GPU_LOL_TEMPLATE_PATTERN`, default: `competitive_salmon_porpoise`)
2. Official RunPod pytorch templates
3. `None` → `_select_base_image()` fallback

Note: Cloudflare blocks Python's default urllib UA → must send browser User-Agent.

## Asset symlinking

Large datasets and model checkpoints don't get copied — they get symlinked.

```python
EnvironmentSpec.asset_links: list[str]  # "remote:local" pairs
```

`skypilot.py` generates in setup:
```bash
mkdir -p /root/.cache && ln -sf /workspace/huggingface /root/.cache/huggingface
```

Auto-detected for HF projects: `/workspace/huggingface` → `~/.cache/huggingface`

Override with: `gpu-lol up --assets /workspace/data:/data`

## llm.py — optional intelligence layer

Sends repo context to any OpenAI-compatible endpoint:

```python
{
  "model": "x-ai/grok-4.1-fast",
  "messages": [{
    "role": "user",
    "content": "Analyze this ML repo and return JSON: {workload_type, vram_required_gb, gpu_type, reasoning}\n\n<repo_context>..."
  }],
  "response_format": {"type": "json_object"}
}
```

Guardrails applied to LLM output before accepting:
- `gpu_type` validated against `GPU_CATALOG` (unknown types rejected)
- `vram_required_gb` floored at 8
- Any exception → returns `None` → heuristics used

## Base image format

RunPod changed their tag format. New format only:
```
runpod/pytorch:{version}-cu{cuda_no_dots}-torch{torch_no_dots}-ubuntu{os}

Examples:
  runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204   (CUDA 12.9, PyTorch 2.9.1, Ubuntu 22.04)
  runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204   (CUDA 12.8.1)
  runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404   (CUDA 13.0, Ubuntu 24.04)
```

Old format (`runpod/pytorch:2.1.0-py3.10-cuda12.1-devel-ubuntu22.04`) no longer exists.

## GPU Catalog

```python
GPU_CATALOG = [                                     # ordered cheapest-first
    {"skypilot_id": "RTX3090",   "vram": 24, "cost_hr": 0.22},
    {"skypilot_id": "RTX4090",   "vram": 24, "cost_hr": 0.34},
    {"skypilot_id": "A40",       "vram": 48, "cost_hr": 0.40},
    {"skypilot_id": "A6000",     "vram": 48, "cost_hr": 0.50},
    {"skypilot_id": "A100-SXM4", "vram": 80, "cost_hr": 1.19},
    {"skypilot_id": "H100-SXM",  "vram": 80, "cost_hr": 2.49},
]
```

## File map

```
gpu_lol/
  config.py      EnvironmentSpec dataclass + .gpu-lol.yaml serialization
  analyzer.py    THE MOAT — repo analysis, heuristics, GPU selection
  llm.py         Optional LLM override via OpenAI-compatible API
  skypilot.py    SkyPilot YAML generation + sky subprocess calls
  runpod_api.py  RunPod GraphQL API — template listing and selection
  validator.py   Smoke tests (CUDA, VRAM, packages, code sync)
  snapshot.py    Reverse-engineer running pod → EnvironmentSpec
  cli.py         Click CLI — up, ssh, exec, logs, validate, snapshot, resume, ls, start, stop, down, check

tests/
  test_analyzer.py   11 heuristic tests (no GPU, no network, LLM disabled)
  test_skypilot.py   YAML generation tests
  test_validator.py  Validator logic tests
  fixtures/          Fake repos for testing

.claude/skills/      Claude Code slash commands for this project
docs/                Architecture + provider guides
```
