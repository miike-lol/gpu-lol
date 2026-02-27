---
name: gpu-lol-down
description: Tear down GPU cluster(s) to stop billing. Use when asked to kill, terminate, wind down, or stop a GPU pod. Always checks for running clusters first.
argument-hint: [cluster_name | --all]
allowed-tools: Bash
disable-model-invocation: true
---

Wind down GPU environment: $ARGUMENTS

## Running clusters (check billing impact)
!`cd /home/mb/Desktop/new-projects/gpu-lol && .venv/bin/gpu-lol ls 2>&1`

## Decision flow

1. Show the user the running clusters above.

2. If no clusters are running, say so and stop.

3. **Ask if they want to snapshot first** — this preserves the environment for later:
   ```
   gpu-lol snapshot <cluster_name> --save .
   ```
   Recommend if they've installed packages or made changes beyond the original repo.

4. Clarify: **stop** vs **down**?
   - `gpu-lol stop <cluster>` — pauses billing, state preserved, can `gpu-lol start` later
   - `gpu-lol down <cluster>` — permanent, all data lost, billing stops immediately

5. Confirm, then execute:
   ```bash
   # Specific cluster (--yes skips confirmation)
   gpu-lol down $ARGUMENTS --yes

   # Everything (interactive — shows all running clusters and requires typing "yes")
   gpu-lol down --all
   ```

6. Verify with `gpu-lol ls` that nothing is left running.
