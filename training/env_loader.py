"""Load .env from project root so env vars are available everywhere.

Secrets live ONLY in .env (gitignored). Code references them via os.environ.

Env vars used:
    RUNPOD_API_KEY          – RunPod API key
    GEE_SERVICE_ACCOUNT     – Google Earth Engine service account email
    GEE_KEY_FILE            – path to GEE service account JSON key (relative to project root)
    GOOGLE_MAPS_API_KEY     – Google Maps / Places API key
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> None:
    """Load .env from the project root. Idempotent and import-safe."""
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    _load(PROJECT_ROOT / ".env")


def load_dotenv_notebook() -> None:
    """Load .env when called from a notebook (auto-detects project root)."""
    try:
        from dotenv import load_dotenv as _load
    except ImportError:
        return
    cwd = Path.cwd()
    root = cwd if (cwd / "training").is_dir() else cwd.parent
    _load(root / ".env")


def get_gee_credentials():
    """Return (service_account, key_path) from env vars."""
    sa = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    key = os.environ.get("GEE_KEY_FILE", "")
    if not sa:
        raise EnvironmentError(
            "GEE_SERVICE_ACCOUNT not set. Add it to .env (see .env.example)."
        )
    if not key:
        raise EnvironmentError(
            "GEE_KEY_FILE not set. Add it to .env (see .env.example)."
        )
    key_path = Path(key)
    if not key_path.is_absolute():
        key_path = PROJECT_ROOT / key_path
    if not key_path.exists():
        raise FileNotFoundError(
            f"GEE key file not found: {key_path}. "
            "Download it from GCP Console and update GEE_KEY_FILE in .env."
        )
    return sa, str(key_path)
