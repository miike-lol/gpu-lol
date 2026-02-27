from pathlib import Path
from .config import EnvironmentSpec
from .skypilot import SkyPilotLauncher


class EnvironmentSnapshotter:

    def __init__(self, launcher: SkyPilotLauncher, cluster_name: str):
        self.launcher = launcher
        self.cluster_name = cluster_name

    def snapshot(self) -> EnvironmentSpec:
        """
        Reverse-engineer the running environment into a reproducible EnvironmentSpec.
        Captures pip packages, CUDA version, Python version, GPU info.
        """
        print("  Capturing pip packages...")
        packages = self._capture_pip_freeze()

        print("  Capturing CUDA version...")
        cuda_version = self._capture_cuda_version()

        print("  Capturing Python version...")
        python_version = self._capture_python_version()

        print("  Capturing GPU info...")
        gpu_info = self._capture_gpu_info()

        # Build a reproduction script from what we found
        setup_commands = [
            "pip install --upgrade pip --quiet",
            "pip install " + " ".join(packages[:100]) + " --quiet",  # Cap at 100 packages
            "command -v node >/dev/null 2>&1 || (curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs)",
            "npm install -g @anthropic-ai/claude-code --quiet 2>/dev/null || true",
        ]

        return EnvironmentSpec(
            gpu_type=gpu_info.get("skypilot_id"),
            vram_required_gb=gpu_info.get("vram", 16),
            base_image=self._infer_base_image(cuda_version, python_version),
            requirements=packages,
            cuda_version=cuda_version,
            python_version=python_version,
            setup_commands=setup_commands,
            created_from="snapshot",
        )

    def save(self, spec: EnvironmentSpec, repo_path: str = ".") -> str:
        """Save snapshot as .gpu-lol.yaml in repo root."""
        save_path = str(Path(repo_path) / ".gpu-lol.yaml")
        spec.save(save_path)
        print(f"\n✅ Saved to {save_path}")
        print("   Commit this file to reproduce your environment anywhere:")
        print("   git add .gpu-lol.yaml && git commit -m 'chore: save gpu-lol environment spec'")
        return save_path

    def _capture_pip_freeze(self) -> list[str]:
        out, _ = self.launcher.exec(self.cluster_name, "pip freeze 2>/dev/null")
        lines = [l.strip() for l in out.splitlines() if l.strip() and not l.startswith("#")]
        # Filter out internal/editable installs
        return [l for l in lines if not l.startswith("-e") and not l.startswith("file://")]

    def _capture_cuda_version(self) -> str:
        out, code = self.launcher.exec(
            self.cluster_name,
            "python3 -c \"import torch; print(torch.version.cuda)\" 2>/dev/null"
        )
        if code == 0 and out.strip():
            # Normalize to major.minor
            parts = out.strip().split(".")
            return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else "12.1"
        return "12.1"

    def _capture_python_version(self) -> str:
        out, _ = self.launcher.exec(
            self.cluster_name,
            "python3 -c \"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')\""
        )
        return out.strip() if out.strip() else "3.10"

    def _capture_gpu_info(self) -> dict:
        out, code = self.launcher.exec(
            self.cluster_name,
            "python3 -c \"import torch; p=torch.cuda.get_device_properties(0); print(p.name, int(p.total_memory/1e9))\" 2>/dev/null"
        )
        if code == 0 and out.strip():
            parts = out.strip().split()
            if len(parts) >= 2:
                vram = int(parts[-1])
                gpu_name = " ".join(parts[:-1])
                skypilot_id = self._name_to_skypilot_id(gpu_name, vram)
                return {"name": gpu_name, "vram": vram, "skypilot_id": skypilot_id}
        return {"name": "unknown", "vram": 16, "skypilot_id": "A40"}

    def _name_to_skypilot_id(self, gpu_name: str, vram: int) -> str:
        """Map GPU name string to SkyPilot accelerator ID."""
        name_lower = gpu_name.lower()
        if "h100" in name_lower: return "H100-SXM"
        if "a100" in name_lower: return "A100-SXM4"
        if "a40" in name_lower: return "A40"
        if "a6000" in name_lower: return "A6000"
        if "4090" in name_lower: return "RTX4090"
        if "3090" in name_lower: return "RTX3090"
        # Fallback: pick from catalog by VRAM
        from .analyzer import CodebaseAnalyzer
        analyzer = CodebaseAnalyzer.__new__(CodebaseAnalyzer)
        return analyzer._select_gpu(vram)

    def _infer_base_image(self, cuda_version: str, python_version: str) -> str:
        from .analyzer import CodebaseAnalyzer
        from . import runpod_api

        # Prefer user's cached templates — match on python version if possible
        try:
            template = runpod_api.best_ml_template("interactive", python_version)
            if template:
                return template["imageName"]
        except Exception:
            pass

        images = CodebaseAnalyzer.BASE_IMAGES
        cuda_major_minor = ".".join(cuda_version.split(".")[:2])
        return images.get(("pytorch", cuda_major_minor), CodebaseAnalyzer.DEFAULT_IMAGE)
