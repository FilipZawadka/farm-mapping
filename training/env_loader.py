"""Load .env from project root so env vars are available everywhere.

Secrets live ONLY in .env (gitignored). Code references them via os.environ.

Env vars used:
    RUNPOD_API_KEY          – RunPod API key
    GEE_SERVICE_ACCOUNT     – Google Earth Engine service account email
    GEE_KEY_FILE            – path to GEE service account JSON key
    GEE_PRIVATE_KEY_JSON    – (alternative) full JSON key content as a string,
                              or base64-encoded JSON (avoids special-char issues
                              in secret stores). A temp file is created automatically
                              so you never need to copy key files to pods.
    GOOGLE_MAPS_API_KEY     – Google Maps / Places API key
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_GEE_KEY_TMPFILE: str | None = None


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


def _materialize_key_json() -> str:
    """Write GEE_PRIVATE_KEY_JSON to a temp file and return its path.

    Accepts either raw JSON or base64-encoded JSON (useful when the secret
    store mangles special characters like newlines inside private keys).
    The file is created once per process and reused on subsequent calls.
    """
    global _GEE_KEY_TMPFILE  # noqa: PLW0603
    if _GEE_KEY_TMPFILE and Path(_GEE_KEY_TMPFILE).exists():
        return _GEE_KEY_TMPFILE

    raw = os.environ["GEE_PRIVATE_KEY_JSON"]
    if not raw.lstrip().startswith("{"):
        raw = base64.b64decode(raw).decode("utf-8")
    data = json.loads(raw)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="gee_key_")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    _GEE_KEY_TMPFILE = path
    return path


def get_gee_credentials() -> tuple[str, str]:
    """Return ``(service_account, key_path)`` from env vars.

    Supports two modes:

    * **File mode** (local dev): set ``GEE_SERVICE_ACCOUNT`` + ``GEE_KEY_FILE``.
    * **Inline mode** (RunPod / CI): set ``GEE_SERVICE_ACCOUNT`` +
      ``GEE_PRIVATE_KEY_JSON`` (the full JSON key content as a string).
      A temp file is created automatically.
    """
    sa = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    if not sa:
        raise EnvironmentError(
            "GEE_SERVICE_ACCOUNT not set. Add it to .env (see .env.example)."
        )

    inline_json = os.environ.get("GEE_PRIVATE_KEY_JSON", "")
    key_file = os.environ.get("GEE_KEY_FILE", "")

    if inline_json:
        return sa, _materialize_key_json()

    if not key_file:
        raise EnvironmentError(
            "Neither GEE_KEY_FILE nor GEE_PRIVATE_KEY_JSON is set. "
            "Add one of them to .env (see .env.example)."
        )
    key_path = Path(key_file)
    if not key_path.is_absolute():
        key_path = PROJECT_ROOT / key_path
    if not key_path.exists():
        raise FileNotFoundError(
            f"GEE key file not found: {key_path}. "
            "Download it from GCP Console and update GEE_KEY_FILE in .env."
        )
    return sa, str(key_path)
