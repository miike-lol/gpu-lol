import re
from pathlib import Path
from .config import EnvironmentSpec

# Python standard library modules to exclude from package detection
STDLIB_MODULES = {
    "os", "sys", "re", "json", "time", "datetime", "pathlib", "typing",
    "collections", "itertools", "functools", "math", "random", "string",
    "io", "abc", "copy", "dataclasses", "enum", "logging", "warnings",
    "subprocess", "threading", "multiprocessing", "asyncio", "socket",
    "http", "urllib", "email", "html", "xml", "csv", "sqlite3", "pickle",
    "hashlib", "hmac", "base64", "struct", "array", "queue", "heapq",
    "bisect", "weakref", "contextlib", "inspect", "traceback", "unittest",
    "tempfile", "shutil", "glob", "fnmatch", "stat", "platform", "signal",
    "gc", "dis", "ast", "token", "tokenize", "keyword", "builtins",
    "argparse", "configparser", "getopt", "getpass", "textwrap", "pprint",
}

# Map pip package names to their import names where they differ
PACKAGE_TO_IMPORT = {
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "beautifulsoup4": "bs4",
    "pyyaml": "yaml",
    "python-dotenv": "dotenv",
    "typing-extensions": "typing_extensions",
    "pyzmq": "zmq",
    "pytorch-lightning": "lightning",
    "huggingface-hub": "huggingface_hub",
}

# Critical ML packages — these get validated in smoke tests
CRITICAL_ML_PACKAGES = {
    "torch", "tensorflow", "transformers", "datasets", "accelerate",
    "peft", "trl", "vllm", "unsloth", "bitsandbytes", "flash-attn",
    "xformers", "deepspeed", "fairscale",
}


class CodebaseAnalyzer:

    # VRAM estimates by model parameter count and precision (GB)
    VRAM_ESTIMATES = {
        "1b":  {"fp32": 4,   "fp16": 2,   "int8": 1,  "int4": 0.5},
        "3b":  {"fp32": 12,  "fp16": 6,   "int8": 3,  "int4": 1.5},
        "7b":  {"fp32": 28,  "fp16": 14,  "int8": 7,  "int4": 4},
        "13b": {"fp32": 52,  "fp16": 26,  "int8": 13, "int4": 7},
        "30b": {"fp32": 120, "fp16": 60,  "int8": 30, "int4": 15},
        "70b": {"fp32": 280, "fp16": 140, "int8": 70, "int4": 35},
        "405b": {"fp32": 1600, "fp16": 810, "int8": 405, "int4": 202},
    }

    # Map HF model ID substrings (lowercase) to parameter count in billions
    KNOWN_MODEL_SIZES = {
        # Llama
        "llama-3.1-405b": 405, "llama-3.1-70b": 70, "llama-3.1-8b": 8,
        "llama-3-70b": 70, "llama-3-8b": 8,
        "llama-2-70b": 70, "llama-2-13b": 13, "llama-2-7b": 7,
        # Qwen
        "qwen2.5-72b": 72, "qwen2.5-32b": 32, "qwen2.5-14b": 14,
        "qwen2.5-7b": 7, "qwen2.5-3b": 3, "qwen2-72b": 72, "qwen2-7b": 7,
        # Mistral / Mixtral
        "mixtral-8x22b": 141, "mixtral-8x7b": 47,
        "mistral-7b": 7, "mistral-nemo": 12,
        # Phi
        "phi-4": 14, "phi-3.5": 3.8, "phi-3-medium": 14, "phi-3-mini": 3.8,
        # Gemma
        "gemma-2-27b": 27, "gemma-2-9b": 9, "gemma-2-2b": 2,
        "gemma-7b": 7, "gemma-2b": 2,
        # DeepSeek
        "deepseek-v3": 671, "deepseek-r1": 671,
        "deepseek-coder-33b": 33, "deepseek-coder-6.7b": 6.7,
        # Falcon
        "falcon-180b": 180, "falcon-40b": 40, "falcon-7b": 7,
        # Other
        "command-r-plus": 104, "dbrx": 132, "yi-34b": 34, "yi-6b": 6,
    }

    # Training needs extra VRAM for optimizer states + gradients
    TRAINING_OVERHEAD = 1.25
    INFERENCE_OVERHEAD = 1.05

    # Ordered cheapest first. SkyPilot will select from these.
    GPU_CATALOG = [
        {"skypilot_id": "RTX3090",    "vram": 24,  "cost_hr": 0.22},
        {"skypilot_id": "RTX4090",    "vram": 24,  "cost_hr": 0.34},
        {"skypilot_id": "A40",        "vram": 48,  "cost_hr": 0.40},
        {"skypilot_id": "A6000",      "vram": 48,  "cost_hr": 0.50},
        {"skypilot_id": "A100-SXM4",  "vram": 80,  "cost_hr": 1.19},
        {"skypilot_id": "H100-SXM",   "vram": 80,  "cost_hr": 2.49},
    ]

    # Docker base images — runpod/pytorch new tag format: {version}-cu{cuda}-torch{torch}-ubuntu{os}
    BASE_IMAGES = {
        ("pytorch", "12.9"): "runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204",
        ("pytorch", "12.8"): "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2204",
        ("pytorch", "13.0"): "runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404",
        ("tensorflow",):     "tensorflow/tensorflow:2.15.0-gpu",
    }
    DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204"

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def analyze(self, template_override: str | None = None) -> EnvironmentSpec:
        """
        Main entry point. Returns EnvironmentSpec.
        Priority:
          1. .gpu-lol.yaml in repo (user config wins, stop here)
          2. LLM analysis via GPU_LOL_LLM_URL (if configured)
          3. Heuristic analysis (always available, no external deps)
        """
        config_path = self.repo_path / ".gpu-lol.yaml"
        if config_path.exists():
            print(f"  Found .gpu-lol.yaml — using saved spec")
            return EnvironmentSpec.from_yaml_file(str(config_path))

        # Heuristics run unconditionally — LLM can override below
        packages = self._extract_packages()
        workload = self._detect_workload_type()
        vram = self._estimate_vram(packages, workload)
        gpu = self._select_gpu(vram)
        gpu_count = self._detect_gpu_count()

        # Resolve template override early so LLM result can still override workload/vram/gpu
        _template_image = None
        if template_override:
            from . import runpod_api
            t = runpod_api.get_template(template_override)
            if t:
                _template_image = t["imageName"]
                print(f"  Template: {t['name']} → {t['imageName']}")
            else:
                print(f"  ⚠ Template '{template_override}' not found — using auto-selected image")

        dockerfile_image = self._detect_dockerfile_image()

        # LLM override: smarter analysis when configured
        from . import llm
        llm_result = llm.analyze_repo(str(self.repo_path))
        llm_reasoning = None
        if llm_result:
            workload = llm_result.get("workload_type", workload)
            llm_vram = llm_result.get("vram_required_gb", vram)
            vram = max(llm_vram, 8)  # floor: interactive pods always need some VRAM
            # Only accept gpu_type if it's a known catalog value
            valid_gpus = {g["skypilot_id"] for g in self.GPU_CATALOG}
            llm_gpu = llm_result.get("gpu_type")
            if llm_gpu in valid_gpus:
                gpu = llm_gpu
            else:
                gpu = self._select_gpu(vram)  # recompute from validated vram
            llm_gpu_count = llm_result.get("gpu_count")
            if isinstance(llm_gpu_count, int) and llm_gpu_count > 1:
                gpu_count = llm_gpu_count
            llm_reasoning = llm_result.get("reasoning")
            if llm_reasoning:
                print(f"  LLM: {llm_reasoning}")

        image = _template_image or dockerfile_image or self._select_base_image(packages, workload)
        setup = self._build_setup_commands(packages)

        return EnvironmentSpec(
            gpu_type=gpu,
            gpu_count=gpu_count,
            vram_required_gb=vram,
            base_image=image,
            requirements=packages,
            setup_commands=setup,
            workload_type=workload,
            name=self.repo_path.name,
            created_from=str(self.repo_path),
        )

    def _extract_packages(self) -> list[str]:
        """Extract pip packages from requirements files and import scanning."""
        packages = set()

        # 1. requirements.txt — search root then one level of subdirectories
        req_files = [self.repo_path / "requirements.txt"]
        req_files += sorted(self.repo_path.glob("*/requirements.txt"))[:3]
        for req_file in req_files:
            if req_file.exists():
                for line in req_file.read_text().splitlines():
                    line = line.split("#")[0].strip()  # strip inline comments
                    if line and not line.startswith("-"):
                        packages.add(line)

        # 2. pyproject.toml
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            try:
                data = tomllib.loads(pyproject.read_text())
                deps = data.get("project", {}).get("dependencies", [])
                packages.update(deps)
                # Also grab optional dependencies, but skip dev/test groups
                opt_deps = data.get("project", {}).get("optional-dependencies", {})
                for group_name, group_deps in opt_deps.items():
                    if group_name.lower() in ("dev", "test", "tests", "lint", "typing", "docs"):
                        continue
                    if isinstance(group_deps, list):
                        packages.update(group_deps)
            except Exception:
                pass

        # 3. setup.py
        setup_py = self.repo_path / "setup.py"
        if setup_py.exists():
            content = setup_py.read_text()
            matches = re.findall(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            for match in matches:
                pkgs = re.findall(r'["\']([^"\']+)["\']', match)
                packages.update(pkgs)

        # 4b. Dockerfile — pip install lines
        for dname in ["Dockerfile", "Dockerfile.gpu", "Dockerfile.cuda"]:
            dockerfile = self.repo_path / dname
            if dockerfile.exists():
                for line in dockerfile.read_text(errors="ignore").splitlines():
                    line = line.strip()
                    if line.upper().startswith("RUN ") and "pip install" in line:
                        pip_part = re.sub(r'.*pip install\s+', '', line, flags=re.IGNORECASE)
                        pip_part = pip_part.split("&&")[0].split("||")[0].strip()
                        for token in pip_part.split():
                            token = token.split("#")[0].strip()
                            if token and not token.startswith("-") and not token.startswith("http"):
                                if re.match(r'^[a-zA-Z0-9]', token):
                                    packages.add(token)
                break  # Only read first Dockerfile found

        # 4. Scan .py imports — only if no requirements found anywhere
        if not packages:
            # Build set of local module names to exclude (repo's own files/dirs)
            local_names = {p.stem for p in self.repo_path.rglob("*.py")}
            local_names |= {p.name for p in self.repo_path.iterdir() if p.is_dir()}

            skip_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git"}
            for py_file in self.repo_path.rglob("*.py"):
                if any(s in py_file.parts for s in skip_dirs):
                    continue
                try:
                    for line in py_file.read_text(errors="ignore").splitlines():
                        line = line.strip()
                        if line.startswith("import "):
                            # Handle `import os, json, sys` — split on commas
                            names_part = line[len("import "):].split("#")[0]
                            for token in names_part.split(","):
                                pkg = token.strip().split(".")[0].split(" ")[0]
                                pkg = pkg.rstrip(";")
                                if self._is_real_package(pkg, local_names):
                                    packages.add(pkg)
                        elif line.startswith("from ") and " import " in line:
                            pkg = line.split()[1].split(".")[0]
                            if self._is_real_package(pkg, local_names):
                                packages.add(pkg)
                except Exception:
                    pass

        return sorted(list(packages))

    def _is_real_package(self, pkg: str, local_names: set) -> bool:
        """Return True if pkg looks like a real pip package (not stdlib or local)."""
        if not pkg or not pkg.isidentifier():
            return False
        if pkg.startswith("_"):  # __future__, _thread, etc.
            return False
        if pkg in STDLIB_MODULES:
            return False
        if pkg in local_names:  # repo's own modules
            return False
        return True

    def _detect_workload_type(self) -> str:
        """Detect training vs inference vs interactive from file names, content, and packages."""
        all_files_lower = [str(f).lower() for f in self.repo_path.rglob("*.py")
                           if ".venv" not in str(f)]

        training_signals = ["train", "finetune", "fine_tune", "grpo", "sft", "rlhf",
                            "ppo", "dpo", "orpo", "trainer", "pretrain"]
        inference_signals = ["serve", "inference", "infer", "api", "server",
                             "vllm", "tgi", "endpoint", "deploy"]

        # Package names that strongly imply workload type
        training_packages = {"trl", "peft", "accelerate", "deepspeed", "unsloth",
                             "bitsandbytes", "fairscale"}
        inference_packages = {"vllm", "tgi", "text-generation-inference", "ray", "triton"}

        files_str = " ".join(all_files_lower)

        # Check .py file names and content
        if any(signal in files_str for signal in training_signals):
            return "training"
        if any(signal in files_str for signal in inference_signals):
            return "inference"

        # Fall back to checking installed packages
        packages = self._extract_packages()
        pkg_names = {p.split("==")[0].split(">=")[0].split("<=")[0].lower() for p in packages}

        if pkg_names & training_packages:
            return "training"
        if pkg_names & inference_packages:
            return "inference"

        return "interactive"

    def _estimate_vram(self, packages: list[str], workload: str) -> int:
        """
        Estimate VRAM requirement.
        Scans for model size hints in filenames, configs, and argparse defaults.
        Falls back to safe defaults by workload type.
        """
        # Look for model size patterns in all text files
        size_pattern = re.compile(
            r'(\d+(?:\.\d+)?)\s*[bB](?:illion)?\s*(?:param|parameter|model)?|'
            r'(\d+)b[-_\s]|[-_\s](\d+)b\b',
            re.IGNORECASE
        )

        found_sizes = []
        search_files = list(self.repo_path.rglob("*.py"))[:50]  # Limit scan
        search_files += list(self.repo_path.rglob("*.yaml"))[:20]
        search_files += list(self.repo_path.rglob("*.json"))[:10]

        for f in search_files:
            if any(skip in str(f) for skip in [".venv", "__pycache__"]):
                continue
            try:
                content = f.read_text(errors="ignore")
                matches = size_pattern.findall(content)
                for match in matches:
                    size_str = next(m for m in match if m)
                    try:
                        size = float(size_str)
                        if 0.5 <= size <= 500:  # Reasonable model size range
                            found_sizes.append(size)
                    except ValueError:
                        pass
            except Exception:
                pass

        # Also check common HF model name patterns (e.g. "llama-3-70b", "qwen-7b")
        hf_pattern = re.compile(r'(?:llama|qwen|mistral|phi|gemma|deepseek|falcon)[-_]?(\d+)b', re.IGNORECASE)
        for f in search_files[:30]:
            try:
                content = f.read_text(errors="ignore")
                for match in hf_pattern.finditer(content):
                    size = float(match.group(1))
                    if 0.5 <= size <= 500:
                        found_sizes.append(size)
            except Exception:
                pass

        # Scan for from_pretrained("model-id") and map to known model sizes
        pretrained_pattern = re.compile(
            r'from_pretrained\s*\(\s*["\']([^"\']+)["\']', re.IGNORECASE
        )
        for f in search_files[:30]:
            try:
                content = f.read_text(errors="ignore")
                for match in pretrained_pattern.finditer(content):
                    model_id = match.group(1).lower()
                    for known_id, size_b in self.KNOWN_MODEL_SIZES.items():
                        if known_id in model_id:
                            found_sizes.append(float(size_b))
                            break
            except Exception:
                pass

        if found_sizes:
            # Use the largest model size found (conservative)
            max_size = max(found_sizes)
            size_key = self._closest_size_key(max_size)
            precision = "fp16"  # Default assumption

            # Check for quantization hints
            content_sample = ""
            for f in search_files[:10]:
                try:
                    content_sample += f.read_text(errors="ignore")[:5000]
                except Exception:
                    pass

            if "int4" in content_sample or "4bit" in content_sample or "bnb_4bit" in content_sample:
                precision = "int4"
            elif "int8" in content_sample or "8bit" in content_sample:
                precision = "int8"

            base_vram = self.VRAM_ESTIMATES.get(size_key, {}).get(precision, 16)
            overhead = self.TRAINING_OVERHEAD if workload == "training" else self.INFERENCE_OVERHEAD
            return max(int(base_vram * overhead) + 2, 8)  # +2GB buffer

        # Safe defaults by workload
        defaults = {"training": 40, "inference": 16, "interactive": 16}
        return defaults.get(workload, 16)

    def _closest_size_key(self, size_b: float) -> str:
        """Map float parameter count to size key."""
        if size_b <= 2: return "1b"
        if size_b <= 5: return "3b"
        if size_b <= 10: return "7b"
        if size_b <= 20: return "13b"
        if size_b <= 50: return "30b"
        if size_b <= 100: return "70b"
        return "405b"

    def _select_gpu(self, vram_required: int) -> str:
        """Pick cheapest GPU with enough VRAM."""
        for gpu in self.GPU_CATALOG:
            if gpu["vram"] >= vram_required:
                return gpu["skypilot_id"]
        return "H100-SXM"  # Fallback to biggest

    def _select_base_image(self, packages: list[str], workload: str = "interactive") -> str:
        """
        Pick Docker base image.
        Priority:
          1. TensorFlow detected → TF image (user's templates won't have TF)
          2. User's RunPod templates → fast boot from cached image
          3. Default runpod/pytorch image
        """
        pkg_names = [p.split("==")[0].split(">=")[0].split("<=")[0].lower() for p in packages]

        if "tensorflow" in pkg_names or "tf-nightly" in pkg_names:
            return self.BASE_IMAGES.get(("tensorflow",), self.DEFAULT_IMAGE)

        # Try user's pre-cached RunPod templates — much faster than fresh pulls
        try:
            from . import runpod_api
            template = runpod_api.best_ml_template(workload)
            if template:
                return template["imageName"]
        except Exception:
            pass

        return self.BASE_IMAGES.get(("pytorch", "12.9"), self.DEFAULT_IMAGE)

    def _build_setup_commands(self, packages: list[str]) -> list[str]:
        """Build ordered setup commands for the pod."""
        cmds = ["pip install --upgrade pip --quiet"]

        # Install from requirements file if it exists
        if (self.repo_path / "requirements.txt").exists():
            cmds.append("pip install -r ~/sky_workdir/requirements.txt --quiet")
        elif packages:
            # Filter out packages that are likely already in the base image
            skip = {"torch", "torchvision", "torchaudio", "numpy", "scipy"}
            to_install = [p for p in packages if p.split("==")[0].lower() not in skip]
            if to_install:
                cmds.append(f"pip install {' '.join(to_install[:50])} --quiet")  # Cap at 50

        # Always install claude-code — this is Mikey's workflow
        cmds.append("command -v node >/dev/null 2>&1 || (curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs)")
        cmds.append("npm install -g @anthropic-ai/claude-code --quiet 2>/dev/null || true")

        return cmds

    def _detect_dockerfile_image(self) -> str | None:
        """Extract ML base image from Dockerfile FROM line."""
        for name in ["Dockerfile", "Dockerfile.gpu", "Dockerfile.cuda", "Dockerfile.train"]:
            p = self.repo_path / name
            if p.exists():
                for line in p.read_text(errors="ignore").splitlines():
                    line = line.strip()
                    if line.upper().startswith("FROM ") and "scratch" not in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            image = parts[1]
                            if any(kw in image.lower() for kw in
                                   ["cuda", "pytorch", "torch", "tensorflow", "nvidia", "runpod"]):
                                return image
        return None

    def _detect_gpu_count(self) -> int:
        """Detect multi-GPU usage (DDP, FSDP, torchrun, deepspeed). Returns 1 if single-GPU."""
        multi_gpu_signals = [
            "torchrun", "torch.distributed", "DistributedDataParallel", "FSDP",
            "FullyShardedDataParallel", "deepspeed.init_distributed",
            "accelerate launch", "nproc_per_node", "WORLD_SIZE", "LOCAL_RANK",
            "torch.multiprocessing.spawn",
        ]
        search_files = list(self.repo_path.rglob("*.py"))[:50]
        search_files += list(self.repo_path.rglob("*.sh"))[:10]
        for f in search_files:
            if any(skip in str(f) for skip in [".venv", "__pycache__", "node_modules", "tests/", "test_", "gpu_lol/"]):
                continue
            try:
                content = f.read_text(errors="ignore")
                if any(signal in content for signal in multi_gpu_signals):
                    # Look for explicit nproc hints
                    m = re.search(r'nproc[_-]per[_-]node["\s:=,]+(\d+)', content)
                    if m:
                        return int(m.group(1))
                    m = re.search(r'(?:num_gpus|n_gpus|num_processes)["\s:=,]+(\d+)', content)
                    if m and int(m.group(1)) > 1:
                        return int(m.group(1))
                    return 2  # Default: 2 GPUs for multi-GPU projects
            except Exception:
                pass
        return 1
