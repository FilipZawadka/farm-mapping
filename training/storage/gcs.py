"""GCS storage backend for data cache.

Requires optional dependency: pip install google-cloud-storage
Credentials via env GOOGLE_APPLICATION_CREDENTIALS or default.
"""

from __future__ import annotations

from pathlib import Path

from .base import StorageBackend


class GCSStorageBackend(StorageBackend):
    """Store cache in a GCS bucket under prefix/remote_key/."""

    def __init__(self, bucket: str, prefix: str = "cache"):
        self.bucket_name = bucket
        self.prefix = prefix.rstrip("/")

    def _bucket(self):
        try:
            from google.cloud import storage
        except ImportError as e:
            raise ImportError(
                "GCS backend requires google-cloud-storage. "
                "Install with: pip install google-cloud-storage"
            ) from e
        return storage.Client().bucket(self.bucket_name)

    def _key(self, remote_key: str, filename: str = "") -> str:
        base = f"{self.prefix}/{remote_key}"
        return f"{base}/{filename}" if filename else base + "/"

    def put_dir(self, local_path: Path, remote_key: str) -> None:
        bucket = self._bucket()
        local_path = Path(local_path)
        for f in local_path.rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_path)
                blob_name = self._key(remote_key, str(rel).replace("\\", "/"))
                bucket.blob(blob_name).upload_from_filename(str(f))

    def get_dir(self, remote_key: str, local_path: Path) -> None:
        bucket = self._bucket()
        local_path = Path(local_path)
        prefix = self._key(remote_key)
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith("/"):
                continue
            rel = blob.name[len(prefix) :].lstrip("/")
            dest = local_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))

    def exists(self, remote_key: str) -> bool:
        bucket = self._bucket()
        prefix = self._key(remote_key)
        it = bucket.list_blobs(prefix=prefix, max_results=1)
        return next(it, None) is not None
