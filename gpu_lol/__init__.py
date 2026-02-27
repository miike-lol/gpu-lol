"""
gpu-lol: AI-powered GPU environment manager

Usage:
    # CLI
    gpu-lol up
    gpu-lol snapshot my-cluster
    gpu-lol resume

    # Python API (for use in agents / Claude Code)
    from gpu_lol import up, EnvironmentSpec
"""

from .config import EnvironmentSpec
from .analyzer import CodebaseAnalyzer
from .skypilot import SkyPilotLauncher
from .validator import EnvironmentValidator
from .snapshot import EnvironmentSnapshotter

__version__ = "0.1.0"
__all__ = [
    "EnvironmentSpec",
    "CodebaseAnalyzer",
    "SkyPilotLauncher",
    "EnvironmentValidator",
    "EnvironmentSnapshotter",
]


def up(repo_path: str = ".", cluster_name: str = None, gpu: str = None) -> str:
    """
    Programmatic API: analyze repo and spin up GPU environment.
    Returns cluster_name.

    For use in Claude Code agents:
        from gpu_lol import up
        cluster = up(".")
    """
    import time
    from pathlib import Path

    spec = CodebaseAnalyzer(repo_path).analyze()
    if gpu:
        spec.gpu_type = gpu

    name = cluster_name or f"gr-{Path(repo_path).resolve().name}-{int(time.time()) % 100000}"
    launcher = SkyPilotLauncher()
    launcher.launch(spec, name, repo_path)
    launcher.wait_for_ready(name)

    validator = EnvironmentValidator(launcher, name)
    result = validator.validate(spec)
    if not result.passed:
        validator.auto_fix(result.failures, spec)

    return name
