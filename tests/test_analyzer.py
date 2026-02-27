"""
Unit tests for CodebaseAnalyzer.
No GPU or network required — pure codebase analysis.
LLM is disabled for all tests (heuristics only).
"""
import pytest
from pathlib import Path
from unittest.mock import patch

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def disable_llm():
    """Disable LLM for all analyzer tests — we're testing heuristics only."""
    with patch("gpu_lol.llm.analyze_repo", return_value=None):
        yield


def test_pytorch_training_detection():
    from gpu_lol.analyzer import CodebaseAnalyzer
    analyzer = CodebaseAnalyzer(str(FIXTURES / "pytorch-training"))
    spec = analyzer.analyze()

    assert spec.workload_type == "training"
    assert spec.gpu_type == "A40"  # 40GB default for training
    assert spec.vram_required_gb == 40
    assert "torch>=2.1.0" in spec.requirements
    assert "transformers>=4.36.0" in spec.requirements
    assert "runpod/pytorch" in spec.base_image
    # Image comes from RunPod templates (fast-boot); CUDA version varies by template


def test_vllm_inference_detection():
    from gpu_lol.analyzer import CodebaseAnalyzer
    analyzer = CodebaseAnalyzer(str(FIXTURES / "vllm-inference"))
    spec = analyzer.analyze()

    assert spec.workload_type == "inference"
    assert spec.vram_required_gb == 16
    assert spec.gpu_type in ("RTX3090", "RTX4090")  # 16GB fits on 24GB cards
    assert "vllm>=0.2.0" in spec.requirements


def test_plain_ml_interactive():
    from gpu_lol.analyzer import CodebaseAnalyzer
    analyzer = CodebaseAnalyzer(str(FIXTURES / "plain-ml"))
    spec = analyzer.analyze()

    assert spec.workload_type == "interactive"
    assert spec.vram_required_gb == 16
    assert "numpy" in spec.requirements
    assert "scikit-learn" in spec.requirements


def test_llama_70b_vram_estimation():
    from gpu_lol.analyzer import CodebaseAnalyzer
    analyzer = CodebaseAnalyzer(str(FIXTURES / "llama-finetune"))
    spec = analyzer.analyze()

    assert spec.workload_type == "training"
    # 70B fp16 = 140GB * 1.25 overhead + 2 = 177GB → H100-SXM (80GB is too small, but catalog maxes at 80)
    # Actually H100-SXM is 80GB which is < 177GB, so it falls back to "H100-SXM" (largest in catalog)
    assert spec.gpu_type == "H100-SXM"
    assert spec.vram_required_gb > 100  # Should be large


def test_gpu_router_yaml_overrides_analysis(tmp_path):
    from gpu_lol.analyzer import CodebaseAnalyzer
    from gpu_lol.config import EnvironmentSpec

    # Create a fake repo with both requirements.txt and .gpu-lol.yaml
    (tmp_path / "requirements.txt").write_text("torch\n")
    spec = EnvironmentSpec(gpu_type="RTX3090", vram_required_gb=24, name="override-test")
    spec.save(str(tmp_path / ".gpu-lol.yaml"))

    analyzer = CodebaseAnalyzer(str(tmp_path))
    loaded = analyzer.analyze()

    assert loaded.gpu_type == "RTX3090"
    assert loaded.vram_required_gb == 24
    assert loaded.name == "override-test"


def test_package_extraction_from_requirements(tmp_path):
    from gpu_lol.analyzer import CodebaseAnalyzer

    (tmp_path / "requirements.txt").write_text("torch==2.1.0\ntransformers>=4.36\nnumpy\n")
    analyzer = CodebaseAnalyzer(str(tmp_path))
    packages = analyzer._extract_packages()

    assert "torch==2.1.0" in packages
    assert "transformers>=4.36" in packages
    assert "numpy" in packages


def test_package_extraction_fallback_imports(tmp_path):
    from gpu_lol.analyzer import CodebaseAnalyzer

    # No requirements.txt, should scan .py imports
    (tmp_path / "model.py").write_text("import torch\nfrom transformers import AutoModel\n")
    analyzer = CodebaseAnalyzer(str(tmp_path))
    packages = analyzer._extract_packages()

    assert "torch" in packages
    assert "transformers" in packages


def test_closest_size_key():
    from gpu_lol.analyzer import CodebaseAnalyzer
    a = CodebaseAnalyzer.__new__(CodebaseAnalyzer)

    assert a._closest_size_key(1) == "1b"
    assert a._closest_size_key(3) == "3b"
    assert a._closest_size_key(7) == "7b"
    assert a._closest_size_key(13) == "13b"
    assert a._closest_size_key(30) == "30b"
    assert a._closest_size_key(70) == "70b"
    assert a._closest_size_key(405) == "405b"


def test_gpu_selection_by_vram():
    from gpu_lol.analyzer import CodebaseAnalyzer
    a = CodebaseAnalyzer.__new__(CodebaseAnalyzer)

    assert a._select_gpu(16) == "RTX3090"   # 24GB fits 16GB
    assert a._select_gpu(24) == "RTX3090"   # exactly 24GB
    assert a._select_gpu(25) == "A40"        # needs 48GB
    assert a._select_gpu(48) == "A40"        # exactly 48GB
    assert a._select_gpu(49) == "A100-SXM4" # needs 80GB
    assert a._select_gpu(200) == "H100-SXM"  # fallback to biggest


def test_setup_commands_include_requirements(tmp_path):
    from gpu_lol.analyzer import CodebaseAnalyzer

    (tmp_path / "requirements.txt").write_text("transformers>=4.36\n")
    analyzer = CodebaseAnalyzer(str(tmp_path))
    cmds = analyzer._build_setup_commands(["transformers"])

    assert any("transformers" in cmd for cmd in cmds)
    assert any("claude-code" in cmd for cmd in cmds)


def test_setup_commands_always_include_claude_code(tmp_path):
    from gpu_lol.analyzer import CodebaseAnalyzer

    analyzer = CodebaseAnalyzer(str(tmp_path))
    cmds = analyzer._build_setup_commands([])

    assert any("claude-code" in cmd for cmd in cmds)
