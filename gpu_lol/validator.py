from dataclasses import dataclass, field
from .analyzer import CRITICAL_ML_PACKAGES, PACKAGE_TO_IMPORT
from .config import EnvironmentSpec
from .skypilot import SkyPilotLauncher


@dataclass
class TestResult:
    name: str
    passed: bool
    error: str = ""
    duration_ms: int = 0


@dataclass
class ValidationResult:
    passed: bool
    failures: list[TestResult] = field(default_factory=list)
    successes: list[TestResult] = field(default_factory=list)

    @property
    def summary(self) -> str:
        total = len(self.failures) + len(self.successes)
        if self.passed:
            return f"✓ All {total} checks passed"
        return f"✗ {len(self.failures)}/{total} checks failed: {', '.join(f.name for f in self.failures)}"


class EnvironmentValidator:

    def __init__(self, launcher: SkyPilotLauncher, cluster_name: str):
        self.launcher = launcher
        self.cluster_name = cluster_name

    def validate(self, spec: EnvironmentSpec) -> ValidationResult:
        """Run all smoke tests. Returns ValidationResult."""
        results = []

        # Test 1: CUDA is visible
        results.append(self._run_test(
            "cuda_available",
            "python3 -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available'\""
        ))

        # Test 2: VRAM is sufficient
        vram_check = (
            f"python3 -c \""
            f"import torch; "
            f"mem = torch.cuda.get_device_properties(0).total_memory / 1e9; "
            f"assert mem >= {spec.vram_required_gb * 0.85}, "
            f"f'Only {{mem:.1f}}GB VRAM, need {spec.vram_required_gb}GB'\""
        )
        results.append(self._run_test("vram_sufficient", vram_check))

        # Test 3: Critical ML packages importable
        critical = [
            p.split("==")[0].split(">=")[0].lower()
            for p in spec.requirements
            if p.split("==")[0].split(">=")[0].lower() in CRITICAL_ML_PACKAGES
        ]
        for pkg in critical[:5]:  # Test top 5 critical packages
            import_name = PACKAGE_TO_IMPORT.get(pkg, pkg.replace("-", "_"))
            results.append(self._run_test(
                f"import_{pkg}",
                f"python3 -c 'import {import_name}'"
            ))

        # Test 4: Code was synced
        results.append(self._run_test(
            "code_synced",
            "test -d ~/sky_workdir && echo 'Code present'"
        ))

        failures = [r for r in results if not r.passed]
        successes = [r for r in results if r.passed]
        return ValidationResult(passed=len(failures) == 0, failures=failures, successes=successes)

    def auto_fix(self, failures: list[TestResult], spec: EnvironmentSpec) -> bool:
        """
        Attempt to fix validation failures automatically.
        Currently handles: missing packages (install via pip).
        Returns True if all failures resolved.
        """
        for failure in failures:
            if failure.name.startswith("import_"):
                pkg = failure.name.replace("import_", "")
                print(f"  Installing missing package: {pkg}")
                self.launcher.exec(self.cluster_name, f"pip install {pkg} --quiet")

        # Re-validate
        result = self.validate(spec)
        return result.passed

    def _run_test(self, name: str, command: str) -> TestResult:
        import time
        start = time.time()
        output, code = self.launcher.exec(self.cluster_name, command)
        duration = int((time.time() - start) * 1000)
        return TestResult(
            name=name,
            passed=(code == 0),
            error=output.strip() if code != 0 else "",
            duration_ms=duration,
        )
