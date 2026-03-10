"""Storage backends for data cache (local, S3, GCS, RunPod)."""

from .base import StorageBackend
from .local import LocalStorageBackend

__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "get_cache_backend",
]


def get_cache_backend(cfg: "PipelineConfig") -> StorageBackend | None:
    """Build the configured cache backend, or None if cache disabled."""
    from ..config import PipelineConfig
    if not getattr(cfg, "cache", None) or not cfg.cache.enabled:
        return None
    backend_type = cfg.cache.backend
    if backend_type == "local":
        return LocalStorageBackend(cfg.cache.local.base_path)
    if backend_type == "runpod":
        return LocalStorageBackend(cfg.cache.runpod.volume_mount)
    if backend_type == "s3":
        from .s3 import S3StorageBackend
        return S3StorageBackend(
            bucket=cfg.cache.s3.bucket,
            prefix=cfg.cache.s3.prefix,
        )
    if backend_type == "gcs":
        from .gcs import GCSStorageBackend
        return GCSStorageBackend(
            bucket=cfg.cache.gcs.bucket,
            prefix=cfg.cache.gcs.prefix,
        )
    return None
