"""S3 storage backend for data cache.

Requires optional dependency: pip install boto3
Credentials via env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or default chain.
"""

from __future__ import annotations

from pathlib import Path

from .base import StorageBackend


class S3StorageBackend(StorageBackend):
    """Store cache in an S3 bucket under prefix/remote_key/."""

    def __init__(self, bucket: str, prefix: str = "cache"):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    def _client(self):
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "S3 backend requires boto3. Install with: pip install boto3"
            ) from e
        return boto3.client("s3")

    def _key(self, remote_key: str, filename: str = "") -> str:
        base = f"{self.prefix}/{remote_key}"
        return f"{base}/{filename}" if filename else base + "/"

    def put_dir(self, local_path: Path, remote_key: str) -> None:
        client = self._client()
        local_path = Path(local_path)
        for f in local_path.rglob("*"):
            if f.is_file():
                rel = f.relative_to(local_path)
                key = self._key(remote_key, str(rel).replace("\\", "/"))
                client.upload_file(str(f), self.bucket, key)

    def get_dir(self, remote_key: str, local_path: Path) -> None:
        client = self._client()
        local_path = Path(local_path)
        prefix = self._key(remote_key)
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix) :].lstrip("/")
                dest = local_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(self.bucket, key, str(dest))

    def exists(self, remote_key: str) -> bool:
        client = self._client()
        prefix = self._key(remote_key)
        resp = client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1)
        return resp.get("KeyCount", 0) > 0
