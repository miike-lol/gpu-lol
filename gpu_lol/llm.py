"""
LLM-powered repo analysis via any OpenAI-compatible endpoint.

Configure via environment variables or ~/.gpu-lol/config:
    GPU_LOL_LLM_URL=http://localhost:11434   # Ollama, LM Studio, vLLM, etc.
    GPU_LOL_LLM_KEY=                          # empty for local, API key for hosted
    GPU_LOL_LLM_MODEL=qwen2.5-coder:7b

Fully optional. If not configured or unreachable, analyzer falls back to heuristics.
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path


def _load_config() -> dict:
    """
    Load config from ~/.gpu-lol/.env (dotenvx-encrypted or plain) or
    ~/.gpu-lol/config (legacy). Environment variables always win.

    Priority:
      1. Environment variables (highest)
      2. ~/.gpu-lol/.env via dotenvx decrypt (if dotenvx installed + file encrypted)
      3. ~/.gpu-lol/.env plain text
      4. ~/.gpu-lol/config (legacy plain text)
    """
    env_file = Path.home() / ".gpu-lol" / ".env"
    config_file = Path.home() / ".gpu-lol" / "config"

    if env_file.exists():
        config = _decrypt_dotenvx(env_file) or _parse_plain(env_file)
    elif config_file.exists():
        config = _parse_plain(config_file)
    else:
        config = {}

    # Environment always wins
    for key in ("GPU_LOL_LLM_URL", "GPU_LOL_LLM_KEY", "GPU_LOL_LLM_MODEL", "RUNPOD_API_KEY"):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


def _decrypt_dotenvx(path: Path) -> dict | None:
    """Try dotenvx CLI to decrypt an encrypted .env file. Returns None if unavailable."""
    import shutil
    import subprocess
    dotenvx = shutil.which("dotenvx") or shutil.which("dotenvx", path=os.path.expanduser("~/.local/bin"))
    if not dotenvx:
        return None
    try:
        result = subprocess.run(
            [dotenvx, "get", "-f", str(path), "--format", "json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return None


def _parse_plain(path: Path) -> dict:
    """Parse a plain KEY=VALUE .env file."""
    config = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, _, val = line.partition("=")
            config[key.strip()] = val.strip().strip('"').strip("'")
    return config


def is_configured() -> bool:
    config = _load_config()
    return bool(config.get("GPU_LOL_LLM_URL"))


def analyze_repo(repo_path: str) -> dict | None:
    """
    Send repo context to the configured LLM.
    Returns dict with keys: workload_type, vram_required_gb, gpu_type, reasoning
    Returns None if LLM is not configured, unreachable, or returns bad output.
    """
    config = _load_config()
    base_url = config.get("GPU_LOL_LLM_URL", "").rstrip("/")
    api_key = config.get("GPU_LOL_LLM_KEY", "")
    model = config.get("GPU_LOL_LLM_MODEL", "qwen2.5-coder:7b")

    if not base_url:
        return None

    context = _build_context(repo_path)
    prompt = _build_prompt(context)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
    }

    # Normalize: strip trailing /v1 so we don't double it
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/v1/chat/completions"
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"].strip()
        return _parse_response(content)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
        return None


def _build_context(repo_path: str) -> dict:
    """Collect relevant repo context to send to the LLM."""
    root = Path(repo_path)
    context = {"name": root.name, "files": [], "requirements": "", "code_sample": ""}

    # File listing (exclude noise)
    skip = {".venv", "venv", "__pycache__", ".git", "node_modules", ".egg-info"}
    files = []
    for f in root.rglob("*"):
        if f.is_file() and not any(s in f.parts for s in skip):
            files.append(str(f.relative_to(root)))
    context["files"] = sorted(files)[:60]

    # requirements.txt content
    req = root / "requirements.txt"
    if req.exists():
        context["requirements"] = req.read_text()[:2000]

    # First 100 lines of the most relevant .py file
    for name in ["train.py", "main.py", "run.py", "inference.py", "serve.py"]:
        candidate = root / name
        if candidate.exists():
            lines = candidate.read_text(errors="ignore").splitlines()[:100]
            context["code_sample"] = "\n".join(lines)
            break

    return context


def _build_prompt(context: dict) -> str:
    files_str = "\n".join(f"  {f}" for f in context["files"])
    parts = [
        f"Analyze this ML repository and return GPU requirements as JSON.\n",
        f"Repository: {context['name']}",
        f"\nFiles:\n{files_str}",
    ]
    if context["requirements"]:
        parts.append(f"\nrequirements.txt:\n{context['requirements']}")
    if context["code_sample"]:
        parts.append(f"\nCode sample:\n{context['code_sample']}")
    parts.append("""
Return ONLY valid JSON, no explanation:
{
  "workload_type": "training" | "inference" | "interactive",
  "vram_required_gb": <integer, minimum GPU VRAM needed>,
  "gpu_type": "RTX3090" | "RTX4090" | "A40" | "A6000" | "A100-SXM4" | "H100-SXM",
  "reasoning": "<one sentence>"
}""")
    return "\n".join(parts)


def _parse_response(content: str) -> dict | None:
    """Extract JSON from LLM response."""
    # Strip markdown code fences if present
    if "```" in content:
        lines = content.splitlines()
        inner = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(inner)
    try:
        data = json.loads(content.strip())
        required = {"workload_type", "vram_required_gb", "gpu_type"}
        if required.issubset(data.keys()):
            return data
    except json.JSONDecodeError:
        pass
    return None
