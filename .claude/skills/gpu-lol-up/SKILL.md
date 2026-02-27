---
name: gpu-lol-up
description: Analyze a repo and launch the optimal GPU environment. Use when asked to spin up a GPU pod, launch a cluster, or run an ML workload on cloud GPU.
argument-hint: [repo_path] [--name cluster-name] [--gpu GPU_TYPE] [--detach] [--template NAME] [--gpus N] [--assets REMOTE:LOCAL]
allowed-tools: Bash, Read, Grep, Glob
---

Launch a GPU environment for: $ARGUMENTS

## Current project state
!`cd /home/mb/Desktop/new-projects/gpu-lol && .venv/bin/gpu-lol ls 2>&1 | head -20`

## Steps

1. Run dry-run first and show the detected environment:
   ```
   gpu-lol up $ARGUMENTS --dry-run
   ```

2. Critically evaluate the output — before launching, verify:
   - **Workload type** correct? (training/inference/interactive)
   - **VRAM estimate** reasonable for the actual model/task?
   - **GPU selection** cost-effective? (RTX3090=$0.22 → RTX4090=$0.34 → A40=$0.40 → A100=$1.19 → H100=$2.49/hr)
   - **Package list** clean? No stdlib or local modules, no garbage
   - **Base image** right? (`runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204` for PyTorch)

3. If analysis looks wrong, fix `gpu_lol/analyzer.py` first — don't launch with bad config.

4. If analysis looks good, confirm with the user and launch:
   ```
   gpu-lol up $ARGUMENTS
   ```

   Consider optional flags before launching:
   - Use `--detach` for a non-blocking launch that returns immediately (fire-and-forget, shows a next-step panel).
   - Use `--detach --watch` to fire-and-forget but still stream the log (Ctrl-C safe — the cluster keeps running).
   - Use `--stop-after HOURS` to auto-stop the cluster after N idle hours (e.g. `--stop-after 4` for a 4-hour job).
   - Use `--yes` / `-y` to skip the cost confirmation prompt.
   - Use `--assets` for projects with large datasets or model caches, e.g.:
     ```
     gpu-lol up $ARGUMENTS --assets /workspace/models:/root/.cache/huggingface/hub
     ```
     HuggingFace projects get `/workspace/huggingface:/root/.cache/huggingface` added automatically.
   - Use `--gpus 4` if multi-GPU training was detected (torchrun/DDP/FSDP/deepspeed) or explicitly needed.
   - Use `--template` to pin a porpoise template for faster boot (run `gpu-lol templates` to list options):
     ```
     gpu-lol up $ARGUMENTS --template competitive_salmon_porpoise
     ```

   Examples:
   ```
   gpu-lol up $ARGUMENTS --stop-after 4    # auto-stops after 4 idle hours
   gpu-lol up $ARGUMENTS --detach --watch  # non-blocking but streams log
   gpu-lol up $ARGUMENTS --yes             # skip cost confirmation
   ```

   Note: A cost confirmation prompt is shown before every launch (unless `--yes` is passed). If no credentials are configured, gpu-lol redirects to `gpu-lol secrets init` automatically.

5. After cluster is READY, run `gpu-lol validate <cluster_name>` automatically.

6. Present the SSH panel to the user.

## GPU Catalog (cheapest first)
| GPU | VRAM | Cost/hr |
|-----|------|---------|
| RTX3090 | 24GB | $0.22 |
| RTX4090 | 24GB | $0.34 |
| A40 | 48GB | $0.40 |
| A6000 | 48GB | $0.50 |
| A100-SXM4 | 80GB | $1.19 |
| H100-SXM | 80GB | $2.49 |

## Key source files
- `gpu_lol/analyzer.py` — repo analysis, GPU selection, VRAM estimation
- `gpu_lol/config.py` — EnvironmentSpec, `.gpu-lol.yaml` serialization
- `gpu_lol/skypilot.py` — SkyPilot YAML generation, cluster launch
