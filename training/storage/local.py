"""Local filesystem storage backend (copy/sync to a base path)."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Store cache under a local base path. put_dir copies to base/remote_key; get_dir copies from base/remote_key."""

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path).resolve()

    def put_dir(self, local_path: Path, remote_key: str) -> None:
        dest = self.base_path / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local_path, dest)

    def get_dir(self, remote_key: str, local_path: Path) -> None:
        src = self.base_path / remote_key
        if not src.is_dir():
            raise FileNotFoundError(f"Cache key not found: {remote_key}")
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            dest = local_path / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    def exists(self, remote_key: str) -> bool:
        return (self.base_path / remote_key).is_dir()
