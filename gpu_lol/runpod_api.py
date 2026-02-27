"""
RunPod API client — fetches user's pod templates for fast image selection.
Templates are pre-cached on RunPod nodes and boot significantly faster than fresh pulls.
"""

import json
import os
import urllib.request
import urllib.error
from functools import lru_cache

GRAPHQL_URL = "https://api.runpod.io/graphql"

TEMPLATES_QUERY = """
{
  myself {
    podTemplates {
      id
      name
      imageName
      containerDiskInGb
      ports
    }
  }
}
"""

# Templates that are good for ML work — scored by preference
ML_IMAGE_SIGNALS = ["pytorch", "cuda", "torch", "tensorflow", "rocm"]


@lru_cache(maxsize=1)
def fetch_templates() -> list[dict]:
    """
    Fetch user's RunPod pod templates. Cached for the process lifetime.
    Returns [] if API key is missing or request fails — callers must handle gracefully.
    """
    api_key = _get_api_key()
    if not api_key:
        return []
    try:
        payload = json.dumps({"query": TEMPLATES_QUERY}).encode()
        req = urllib.request.Request(
            f"{GRAPHQL_URL}?api_key={api_key}",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return data.get("data", {}).get("myself", {}).get("podTemplates", [])
    except Exception:
        return []


def get_template(name_or_id: str) -> dict | None:
    """
    Find a template by exact id or partial case-insensitive name match.
    Exact id wins, then first name match.
    """
    templates = fetch_templates()
    # Exact id
    for t in templates:
        if t["id"] == name_or_id:
            return t
    # Partial name (case-insensitive)
    needle = name_or_id.lower()
    for t in templates:
        if needle in t["name"].lower():
            return t
    return None


def best_ml_template(workload: str, python_version: str = "3.11") -> dict | None:
    """
    Pick the best pre-cached template for the given workload.
    Prioritizes templates the user has already used (competitive_salmon_porpoise variants)
    since these are cached on RunPod nodes and boot in seconds vs minutes.

    Priority:
      1. User's personal porpoise templates (known fast)
      2. Official RunPod pytorch templates
      3. None → caller falls back to DEFAULT_IMAGE
    """
    templates = fetch_templates()
    if not templates:
        return None

    pattern = _get_template_pattern()

    # --- Tier 1: personal porpoise templates (user's go-to dev env) ---
    preferred = [t for t in templates if pattern in t["name"].lower()]

    if preferred:
        # Python 3.10 requested → pick that variant
        if python_version.startswith("3.10"):
            for t in preferred:
                if "python3.10" in t["name"]:
                    return t

        # Training → newest torch (2.8) for best perf, skip 5090-specific
        if workload == "training":
            for t in preferred:
                name = t["name"].lower()
                if ("torch2.8" in name or "torch28" in name) and "5090" not in name:
                    return t

        # Default: exact pattern match (e.g. CUDA 12.4, torch 2.4, Python 3.11)
        for t in preferred:
            if t["name"].lower() == pattern:
                return t

        return preferred[0]

    # --- Tier 2: official RunPod pytorch templates (still fast, well-cached) ---
    official = [t for t in templates if t.get("id", "").startswith("runpod-torch")]
    if official:
        # Pick newest (last in list)
        return official[-1]

    return None


def _get_api_key() -> str:
    """Get RunPod API key — env var, then ~/.gpu-lol config."""
    key = os.environ.get("RUNPOD_API_KEY", "")
    if key:
        return key
    try:
        from .llm import _load_config
        return _load_config().get("RUNPOD_API_KEY", "")
    except Exception:
        return ""


def _get_template_pattern() -> str:
    """
    Get the user's preferred template name pattern.
    Checks GPU_LOL_TEMPLATE_PATTERN env var, then ~/.gpu-lol config.
    Defaults to 'competitive_salmon_porpoise'.
    """
    pattern = os.environ.get("GPU_LOL_TEMPLATE_PATTERN", "")
    if pattern:
        return pattern.lower()
    try:
        from .llm import _load_config
        config = _load_config()
        pattern = config.get("GPU_LOL_TEMPLATE_PATTERN", "")
        if pattern:
            return pattern.lower()
    except Exception:
        pass
    return "competitive_salmon_porpoise"
