---
name: gpu-lol-snapshot
description: Capture a running cluster's environment to .gpu-lol.yaml for reproducibility. Use when asked to save, snapshot, or preserve a GPU environment.
argument-hint: <cluster_name> [save_path]
allowed-tools: Bash, Read, Write
---

Snapshot cluster environment: $ARGUMENTS

## Running clusters
!`cd /home/mb/Desktop/new-projects/gpu-lol && .venv/bin/gpu-lol ls 2>&1`

## Steps

1. Run the snapshot:
   ```bash
   gpu-lol snapshot $ARGUMENTS
   ```
   This captures: pip freeze, CUDA version, Python version, GPU info.

2. Read and show the generated `.gpu-lol.yaml`.

3. Verify it's complete — check for:
   - `gpu_type` and `vram_required_gb` are correct
   - `requirements` list looks complete (not truncated at 100 packages)
   - `cuda_version` matches what's actually installed
   - `base_image` uses the new RunPod format: `runpod/pytorch:1.0.3-cu{cuda}-torch{torch}-ubuntu{os}`
   - No editable installs (`-e git+...`) or `file://` paths that won't reproduce

4. Suggest committing to the repo:
   ```bash
   git add .gpu-lol.yaml
   git commit -m "chore: save gpu-lol environment spec"
   ```

5. Tell the user: anyone can now run `gpu-lol resume` from this file to get the exact same environment on any cloud provider.

## Source files
- `gpu_lol/snapshot.py` — EnvironmentSnapshotter, pip freeze, CUDA/Python/GPU capture
- `gpu_lol/config.py` — EnvironmentSpec.save() → .gpu-lol.yaml serialization
