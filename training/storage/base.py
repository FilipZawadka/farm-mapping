"""Storage backend interface for data cache.

Backends support put_dir (upload), get_dir (download), and exists for
a cache key. Used to persist patch data (e.g. on RunPod volume or S3)
between runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Abstract backend for cache storage."""

    @abstractmethod
    def put_dir(self, local_path: Path, remote_key: str) -> None:
        """Upload a directory to the backend under *remote_key*."""
        ...

    @abstractmethod
    def get_dir(self, remote_key: str, local_path: Path) -> None:
        """Download from *remote_key* to *local_path* (directory)."""
        ...

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Return True if *remote_key* exists in the backend."""
        ...
