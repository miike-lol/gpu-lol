---
name: gpu-lol-resume
description: Resume a saved GPU environment from a .gpu-lol.yaml spec file. Use when asked to restore, recreate, or resume a previous environment.
argument-hint: [spec_path] [--name cluster-name]
allowed-tools: Bash, Read
---

Resume GPU environment from spec: $ARGUMENTS

## Available specs
!`find . ~/.claude -name ".gpu-lol.yaml" 2>/dev/null | head -10 && echo "---" && ls *.yaml 2>/dev/null || true`

## Spec content
!`cat ${ARGUMENTS:-.gpu-lol.yaml} 2>/dev/null || echo "No .gpu-lol.yaml found in current directory"`

## Steps

1. Show the user the spec above — confirm it's the right environment:
   - GPU type and VRAM
   - Key packages
   - Workload type

2. If the spec looks right, resume it:
   ```bash
   gpu-lol resume $ARGUMENTS
   ```
   Or with a custom cluster name:
   ```bash
   gpu-lol resume $ARGUMENTS --name my-cluster
   ```

3. Once READY, run validation:
   ```bash
   gpu-lol validate <cluster_name>
   ```

4. Provide the SSH command.

## If no spec exists

If there's no `.gpu-lol.yaml`, the user needs to either:
- Run `gpu-lol up <repo>` to create a fresh environment
- Run `gpu-lol snapshot <running-cluster>` to capture an existing one

## Dry-run first

To see the YAML that will be submitted without launching:
```bash
gpu-lol resume $ARGUMENTS --dry-run
```
