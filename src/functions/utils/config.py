from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _read_yaml(path: Path) -> Dict[str, Any]:
    """
    Read a YAML file and return a dict.

    Behavior:
    - If file missing: return {}
    - If YAML parses to None: treat as {}
    - If YAML parses to non-dict: raise ValueError (schema contract violation)

    This strictness helps catch misconfigured YAML early and keeps downstream code simpler.
    """
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping/dict: {path}")

    return data

def load_parameters() -> Dict[str, Any]:
    """
    Load the main runtime parameters file: parameters/parameters.yaml.

    Returns:
      Dict[str, Any] (empty if missing)
    """
    return _read_yaml(PROJECT_ROOT / "parameters" / "parameters.yaml")

def load_credentials() -> Dict[str, Any]:
    """
    Load credentials for local dev with environment-variable overrides.

    Supported keys
    --------------
    - GOOGLE_API_KEY:
        Used by google-genai client (HyDE generation / embeddings in offline pipelines).

    Precedence
    ----------
    env var > credentials.yaml > empty

    Returns
    -------
    Dict[str, Any] containing:
      - GOOGLE_API_KEY: str (may be empty)
      - has_google_api_key: bool
      - source: "env" | "file" | "none"
    """
    file_creds = _read_yaml(PROJECT_ROOT / "parameters" / "credentials.yaml")

    # Cloud Run best practice: secrets injected via env vars (Secret Manager, etc.)
    env_google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    # If env is unset, fall back to file-based key (local only).
    file_google_key = str(file_creds.get("GOOGLE_API_KEY", "")).strip()

    google_key = env_google_key or file_google_key

    if env_google_key:
        source = "env"
    elif file_google_key:
        source = "file"
    else:
        source = "none"

    return {
        "GOOGLE_API_KEY": google_key,
        "has_google_api_key": bool(google_key),
        "source": source,
    }


def load_config() -> Dict[str, Any]:
    """
    Unified config loader used across the repo.

    This function:
    1) Loads parameters.yaml
    2) Injects credentials into cfg["credentials"] (without overwriting other keys)
    3) Normalizes some fields for safety (e.g., stripping whitespace)

    Returns:
    cfg: Dict[str, Any]
    """
    cfg = load_parameters()

    # Ensure stable structure for consumers (callers can rely on cfg["credentials"] existing).
    if not isinstance(cfg, dict):
        # Defensive: should never happen because _read_yaml enforces dict, but keep safe.
        cfg = {}

    cfg.setdefault("credentials", {})
    if not isinstance(cfg.get("credentials"), dict):
        cfg["credentials"] = {}

    # Merge credentials (env override already applied inside load_credentials)
    creds = load_credentials()
    cfg["credentials"].update(creds)

    # Optional safety guard: normalize and ensure string type (avoid None surprises)
    cfg["credentials"]["GOOGLE_API_KEY"] = str(cfg["credentials"].get("GOOGLE_API_KEY", "")).strip()

    return cfg

