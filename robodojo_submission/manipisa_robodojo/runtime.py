"""Resolve native WSL Codex settings without reading or copying credentials."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tomllib

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_MODEL = "gpt-6-astra"
SUBMISSION_REASONING = "high"


def result_path(value):
    """Keep all live policy artifacts in this independent submission tree."""
    base = (SUBMISSION_ROOT / "results").resolve()
    path = Path(value)
    if not path.is_absolute():
        path = SUBMISSION_ROOT / path
    path = path.resolve()
    if path != base and base not in path.parents:
        raise ValueError("Outputs must stay inside robodojo_submission/results")
    return path


def resolve_codex(config):
    if sys.platform != "linux":
        raise RuntimeError("Run the policy and Codex inside WSL; run only the simulator on Windows")
    executable = config.get("codex_path") or shutil.which("codex")
    if not executable:
        raise FileNotFoundError("Native WSL codex was not found on PATH")
    executable = Path(executable).expanduser().resolve()
    if executable.suffix.lower() in (".exe", ".cmd", ".bat", ".ps1"):
        raise ValueError("A Windows Codex executable cannot be used for the WSL policy")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(f"Codex is not executable: {executable}")
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    config_path = codex_home / "config.toml"
    settings = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if config.get("codex_profile"):
        raise ValueError("Submission configuration is fixed; Codex profiles are not supported")
    # This runner uses the existing ChatGPT login with the default OpenAI provider.
    # Fail instead of silently switching a user's configured custom provider.
    if settings.get("model_provider", "openai") != "openai":
        raise ValueError("Custom Codex provider detected; explicit provider integration is required")
    model = config.get("model") or SUBMISSION_MODEL
    effort = config.get("reasoning_effort") or SUBMISSION_REASONING
    if (model, effort) != (SUBMISSION_MODEL, SUBMISSION_REASONING):
        raise ValueError("This submission is fixed to gpt-6-astra / high")
    return {"codex_path": str(executable), "model": model, "reasoning_effort": effort,
            "codex_home": str(codex_home), "config_path": str(config_path),
            "configured_reasoning_effort": settings.get("model_reasoning_effort"),
            "runtime": "native-linux", "profile": None}
