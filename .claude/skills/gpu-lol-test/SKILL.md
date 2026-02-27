---
name: gpu-lol-test
description: Run the gpu-lol test suite. Use when asked to run tests, check if tests pass, or verify a fix didn't break anything.
argument-hint: [test_file_or_filter]
allowed-tools: Bash, Read
---

Run gpu-lol tests: $ARGUMENTS

## Test suite
!`cd /home/mb/Desktop/new-projects/gpu-lol && .venv/bin/python -m pytest tests/ -v --tb=short $ARGUMENTS 2>&1`

## Test breakdown

| File | Tests | What it covers |
|------|-------|---------------|
| `tests/test_analyzer.py` | 11 | Heuristic analysis — workload detection, VRAM estimates, GPU selection, package extraction |
| `tests/test_skypilot.py` | varies | SkyPilot YAML generation |
| `tests/test_validator.py` | varies | Validator logic |

## Fixture repos (in `tests/fixtures/`)

| Fixture | Expected workload | Expected GPU | Expected VRAM |
|---------|------------------|-------------|---------------|
| `pytorch-training/` | training | A40 | 40GB |
| `vllm-inference/` | inference | RTX3090 or RTX4090 | 16GB |
| `plain-ml/` | interactive | any | 16GB |
| `llama-finetune/` | training | H100-SXM | >100GB |

## Critical: LLM must be disabled in tests

All test files need this autouse fixture or the LLM will override heuristics and break assertions:
```python
@pytest.fixture(autouse=True)
def disable_llm():
    with patch("gpu_lol.llm.analyze_repo", return_value=None):
        yield
```

If adding a new test file, add this fixture.

## If tests fail

1. Run the specific failing test with `-s` for full output:
   ```bash
   pytest tests/test_analyzer.py::test_pytorch_training_detection -v -s
   ```

2. Check if you changed GPU catalog, VRAM estimates, or base images — update fixture assertions to match.

3. Check the image assertion uses the new format:
   ```python
   assert "cu129" in spec.base_image  # Not "cuda12.1"
   ```

All 32 tests must pass. No GPU or network required.
