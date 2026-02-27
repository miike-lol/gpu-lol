---
name: gpu-lol-logs
description: Stream and interpret logs from a running GPU cluster. Use when asked to see logs, debug a training run, check setup output, or monitor a job.
argument-hint: <cluster_name> [job_id]
allowed-tools: Bash
---

Stream logs from cluster: $ARGUMENTS

## Running clusters
!`cd /home/mb/Desktop/new-projects/gpu-lol && .venv/bin/gpu-lol ls 2>&1 | head -15`

## Stream logs

```bash
gpu-lol logs $ARGUMENTS
```

Job IDs: `1` = setup script, `2+` = subsequent exec jobs.

## What to look for

**Setup succeeded:**
```
✅ gpu-lol: environment ready
```

**Package install issues:**
```
ERROR: Could not find a version that satisfies the requirement...
ERROR: pip's dependency resolver...
```
→ Fix: update `requirements.txt` or pin different versions

**CUDA/driver mismatch:**
```
RuntimeError: CUDA error: no kernel image is available for execution
torch.cuda.is_available() returning False
```
→ Fix: wrong base image — check `base_image` in `.gpu-lol.yaml` matches the GPU's CUDA version

**Out of memory:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
→ Fix: need bigger GPU. Run `gpu-lol down <cluster>` + `gpu-lol up --gpu A40`

**Node install issues (claude-code):**
```
npm install -g @anthropic-ai/claude-code
```
→ This is non-fatal, cluster still works without it

## Run a specific command and see output

```bash
gpu-lol ssh $ARGUMENTS -- "nvidia-smi"
gpu-lol ssh $ARGUMENTS -- "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
gpu-lol ssh $ARGUMENTS -- "cat ~/sky_workdir/requirements.txt"
```
