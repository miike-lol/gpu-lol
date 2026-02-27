---
name: gpu-lol-add-gpu
description: Add a new GPU to the gpu-lol catalog or add a new base image. Use when asked to support a new GPU type, update prices, or add a Docker base image.
argument-hint: [gpu_name] [vram_gb] [cost_per_hr]
allowed-tools: Bash, Read, Edit, Grep
---

Add GPU or base image to gpu-lol: $ARGUMENTS

## Current GPU catalog
!`cd /home/mb/Desktop/new-projects/gpu-lol && grep -A 20 "GPU_CATALOG" gpu_lol/analyzer.py | head -25`

## Current base images
!`cd /home/mb/Desktop/new-projects/gpu-lol && grep -A 10 "BASE_IMAGES" gpu_lol/analyzer.py | head -15`

## Adding a new GPU to the catalog

Edit `gpu_lol/analyzer.py` — `GPU_CATALOG` list (ordered cheapest first):

```python
GPU_CATALOG = [
    {"skypilot_id": "RTX3090",    "vram": 24,  "cost_hr": 0.22},
    {"skypilot_id": "RTX4090",    "vram": 24,  "cost_hr": 0.34},
    # Add new GPU here in cost order
    {"skypilot_id": "NEW_GPU_ID", "vram": XX,  "cost_hr": X.XX},
    ...
]
```

Rules:
- `skypilot_id` must match SkyPilot's accelerator name exactly (check `sky show-gpus`)
- Insert in ascending `cost_hr` order — `_select_gpu()` picks the first one with enough VRAM
- Also add to `_name_to_skypilot_id()` in `snapshot.py` if it might appear in `nvidia-smi` output

## Adding a new base image

Edit `BASE_IMAGES` dict in `gpu_lol/analyzer.py`:

```python
BASE_IMAGES = {
    ("pytorch", "12.9"): "runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204",
    # RunPod tag format: {version}-cu{cuda_no_dots}-torch{torch_no_dots}-ubuntu{os}
    ("pytorch", "13.0"): "runpod/pytorch:X.X.X-cu1300-torchXXX-ubuntuXXXX",
}
```

Also update `DEFAULT_IMAGE` if the new image should be the default.

## After any change

Run tests to verify nothing broke:
```bash
cd /home/mb/Desktop/new-projects/gpu-lol
.venv/bin/python -m pytest tests/ -v
```

Update fixture assertions in `tests/test_analyzer.py` if GPU selection or image assertions changed.

## Check available GPUs on SkyPilot/RunPod
```bash
.venv/bin/sky show-gpus 2>&1 | head -40
```
