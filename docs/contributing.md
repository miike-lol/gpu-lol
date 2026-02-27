# Contributing to gpu-lol

## Install (regular users)

```bash
pip install git+https://github.com/miike-lol/gpu-lol
```

## Dev setup (contributors)

```bash
git clone https://github.com/miike-lol/gpu-lol
cd gpu-lol
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install "skypilot[runpod,vast,lambda]"
```

## The one rule

**Fix `analyzer.py` before anything else.** The whole tool is only as good as its analysis. Before working on launch, validation, snapshot — make sure dry-run output is correct.

Test against real repos:
```bash
gpu-lol up ~/Desktop/new-projects/polymarket-001 --dry-run
gpu-lol up ~/Desktop/new-projects/gpt-oss-and-qwen3-5-lean --dry-run
gpu-lol up ~/Desktop/new-projects/gpt-oss-120B-complete-lean --dry-run
```

## Running tests

```bash
pytest tests/ -v              # full suite (32 tests, no GPU needed)
pytest tests/test_analyzer.py # just analysis logic
```

**LLM is always disabled in tests** via autouse fixture in each test file:
```python
@pytest.fixture(autouse=True)
def disable_llm():
    with patch("gpu_lol.llm.analyze_repo", return_value=None):
        yield
```

New test files must include this fixture.

## Adding known model sizes

gpu-lol maps HF model IDs to parameter counts in `analyzer.py` `KNOWN_MODEL_SIZES`:

```python
KNOWN_MODEL_SIZES = {
    "my-new-model-70b": 70,  # lowercase substring match against from_pretrained() calls
}
```

Format: lowercase substring of the model ID → parameter count in billions.

## Adding a GPU

1. Add to `GPU_CATALOG` in `analyzer.py` — keep sorted by `cost_hr` ascending
2. Add to `_name_to_skypilot_id()` in `snapshot.py` — maps `nvidia-smi` names
3. Run `pytest tests/ -v` — update any assertions that changed
4. Verify `skypilot_id` matches SkyPilot exactly: `sky show-gpus`

## Adding a base image

1. Add to `BASE_IMAGES` in `analyzer.py`
2. Update `DEFAULT_IMAGE` if it should be the new default
3. Update `config.py` default `base_image` field
4. Update `snapshot._infer_base_image()` if new CUDA version
5. Update test assertion: `assert "cu{ver}" in spec.base_image`

RunPod image format: `runpod/pytorch:{version}-cu{cuda_nodots}-torch{torch_nodots}-ubuntu{os}`
Check available tags: https://hub.docker.com/r/runpod/pytorch/tags

## Adding a new CLI command

1. Add method to `SkyPilotLauncher` in `skypilot.py` if it needs sky calls
2. Add `@cli.command()` in `cli.py`
3. If it runs commands on the cluster, use `launcher.exec()` — it uses `ssh -F ~/.sky/generated/ssh/<cluster>` (inline), not `sky exec` (queued jobs)
4. Add a skill in `.claude/skills/gpu-lol-{name}/SKILL.md`

## Key patterns

**Never call provider APIs directly** — always go through SkyPilot.

**exec() uses `ssh -F ~/.sky/generated/ssh/<cluster>`, not sky ssh** — `sky exec` queues a job and hangs when `sleep infinity` is running. `sky ssh` was redesigned in 0.11 for node pools, not cluster SSH. `exec()` in `skypilot.py` uses `ssh -F ~/.sky/generated/ssh/<cluster> <cluster> -- <cmd>` for inline execution.

**EnvironmentSpec is the contract** — analyzer produces it, skypilot consumes it, snapshot reproduces it. Don't pass raw dicts between components.

**LLM is always optional** — if `GPU_LOL_LLM_URL` is unset or LLM fails, heuristics run. Never make LLM required.

## Skills

Claude Code skills live in `.claude/skills/`. Each skill is a directory with `SKILL.md`.

Skills use live context injection with `!` `` `command` `` — the command runs before Claude sees the content. Keep injected commands fast (< 2s).

## Credentials in development

```bash
gpu-lol secrets init   # first-time setup — prompts, encrypts, writes ~/.runpod/config.toml
gpu-lol secrets show   # verify what's set
gpu-lol secrets set RUNPOD_API_KEY=rpa_newkey  # rotate a key
```

Credentials live in `~/.gpu-lol/.env` (dotenvx-encrypted) and `~/.gpu-lol/.env.keys` (private key).
Neither is ever committed — both are gitignored. The `.env.example` in the repo is the safe template.

If the LLM is interfering with tests or dry-runs:
```bash
GPU_LOL_LLM_URL="" gpu-lol up . --dry-run   # disable LLM for this invocation
```

## Commit style

```
fix: handle RTX3090 sold-out by falling back to RTX4090
feat: add H200 to GPU catalog
chore: update RunPod base image tags
test: add fixture for qwen 72b inference
```
