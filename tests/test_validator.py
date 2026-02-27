"""
Unit tests for EnvironmentValidator.
Uses mock SkyPilotLauncher — no actual GPU or network required.
"""
import pytest
from unittest.mock import MagicMock
from gpu_lol.config import EnvironmentSpec
from gpu_lol.validator import EnvironmentValidator, ValidationResult, TestResult


def make_launcher(responses: dict[str, tuple[str, int]]) -> MagicMock:
    """
    Create a mock launcher whose exec() returns predefined responses.
    responses: {command_substring: (output, exit_code)}
    """
    launcher = MagicMock()

    def exec_side_effect(cluster_name, command):
        for key, value in responses.items():
            if key in command:
                return value
        return ("", 0)  # Default: success

    launcher.exec.side_effect = exec_side_effect
    return launcher


def test_all_pass():
    launcher = make_launcher({
        "torch.cuda.is_available": ("", 0),
        "total_memory": ("", 0),
        "sky_workdir": ("Code present", 0),
    })
    spec = EnvironmentSpec(vram_required_gb=16, requirements=["torch"])
    validator = EnvironmentValidator(launcher, "test-cluster")
    result = validator.validate(spec)

    assert result.passed
    assert len(result.failures) == 0


def test_cuda_not_available():
    launcher = make_launcher({
        "torch.cuda.is_available": ("CUDA not available", 1),
        "total_memory": ("", 0),
        "sky_workdir": ("Code present", 0),
    })
    spec = EnvironmentSpec(vram_required_gb=16, requirements=["torch"])
    validator = EnvironmentValidator(launcher, "test-cluster")
    result = validator.validate(spec)

    assert not result.passed
    failure_names = [f.name for f in result.failures]
    assert "cuda_available" in failure_names


def test_vram_insufficient():
    launcher = make_launcher({
        "torch.cuda.is_available": ("", 0),
        "total_memory": ("Only 8.0GB VRAM, need 40GB", 1),
        "sky_workdir": ("Code present", 0),
    })
    spec = EnvironmentSpec(vram_required_gb=40, requirements=["torch"])
    validator = EnvironmentValidator(launcher, "test-cluster")
    result = validator.validate(spec)

    assert not result.passed
    failure_names = [f.name for f in result.failures]
    assert "vram_sufficient" in failure_names


def test_code_not_synced():
    launcher = make_launcher({
        "torch.cuda.is_available": ("", 0),
        "total_memory": ("", 0),
        "sky_workdir": ("", 1),
    })
    spec = EnvironmentSpec(vram_required_gb=16, requirements=["torch"])
    validator = EnvironmentValidator(launcher, "test-cluster")
    result = validator.validate(spec)

    assert not result.passed
    failure_names = [f.name for f in result.failures]
    assert "code_synced" in failure_names


def test_critical_package_import_tested():
    """Critical ML packages should be individually import-tested."""
    tested_commands = []

    launcher = MagicMock()
    launcher.exec.side_effect = lambda cluster, cmd: (tested_commands.append(cmd), ("", 0))[1]

    spec = EnvironmentSpec(
        vram_required_gb=16,
        requirements=["torch>=2.1.0", "transformers>=4.36", "peft>=0.6"]
    )
    validator = EnvironmentValidator(launcher, "test-cluster")
    validator.validate(spec)

    # Should have tested imports for torch, transformers, peft
    import_cmds = [c for c in tested_commands if "import torch" in c or "import transformers" in c or "import peft" in c]
    assert len(import_cmds) >= 2


def test_auto_fix_installs_missing_packages():
    install_calls = []

    launcher = MagicMock()
    # First validate: all fail for missing packages
    # After fix: all pass
    call_count = [0]

    def exec_side_effect(cluster, command):
        call_count[0] += 1
        if "pip install" in command:
            install_calls.append(command)
            return ("", 0)
        if call_count[0] <= 10:
            return ("ModuleNotFoundError", 1)
        return ("", 0)

    launcher.exec.side_effect = exec_side_effect

    failures = [
        TestResult(name="import_peft", passed=False, error="ModuleNotFoundError"),
    ]
    spec = EnvironmentSpec(vram_required_gb=16, requirements=["peft"])
    validator = EnvironmentValidator(launcher, "test-cluster")
    validator.auto_fix(failures, spec)

    # Should have tried to pip install peft
    assert any("peft" in cmd for cmd in install_calls)


def test_validation_result_summary_pass():
    result = ValidationResult(
        passed=True,
        successes=[
            TestResult("cuda_available", True),
            TestResult("vram_sufficient", True),
        ],
        failures=[]
    )
    assert "2 checks passed" in result.summary


def test_validation_result_summary_fail():
    result = ValidationResult(
        passed=False,
        successes=[TestResult("cuda_available", True)],
        failures=[TestResult("vram_sufficient", False, "Only 8GB")]
    )
    assert "1/2 checks failed" in result.summary
    assert "vram_sufficient" in result.summary
