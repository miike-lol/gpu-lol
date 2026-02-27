---
name: gpu-lol-validate
description: Validate a running GPU cluster — checks CUDA, VRAM, packages, and code sync. Use after launching a cluster or when debugging environment issues.
argument-hint: <cluster_name>
allowed-tools: Bash, Read
---

Validate cluster: $ARGUMENTS

## Current clusters
!`cd /home/mb/Desktop/new-projects/gpu-lol && .venv/bin/gpu-lol ls 2>&1`

## Run validation
```bash
gpu-lol validate $ARGUMENTS
```

Validation checks:
- `torch.cuda.is_available()` — CUDA visible
- VRAM ≥ spec requirement (with 15% tolerance)
- Critical ML packages importable (torch, transformers, peft, etc.)
- Code synced to `~/sky_workdir/`

## If checks fail, diagnose inline via SSH

```bash
# GPU/CUDA
gpu-lol ssh $ARGUMENTS -- "nvidia-smi"
gpu-lol ssh $ARGUMENTS -- "python3 -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'"

# Packages
gpu-lol ssh $ARGUMENTS -- "pip list | grep -E 'torch|transformers|peft|vllm|accelerate'"

# Code sync
gpu-lol ssh $ARGUMENTS -- "ls -la ~/sky_workdir/"

# Setup script output (check for install errors)
gpu-lol logs $ARGUMENTS
```

## Common failure causes

| Failure | Likely cause | Fix |
|---------|-------------|-----|
| CUDA not available | Wrong base image | Check `base_image` in `.gpu-lol.yaml` |
| Package import fails | Not installed | `gpu-lol ssh <cluster> -- pip install <pkg>` |
| Code not synced | Workdir path wrong | Check `workdir` in generated SkyPilot YAML |
| VRAM too low | Wrong GPU selected | `gpu-lol down` + `gpu-lol up --gpu A40` |

## Source files
- `gpu_lol/validator.py` — validation logic, auto-fix
- `gpu_lol/skypilot.py` — `exec()` uses `sky ssh` for inline execution (not queued jobs)
