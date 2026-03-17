"""Extract image patches for each candidate via imagery providers (e.g. Earth Engine).

Uses ``ee.data.computePixels`` (REST API) for EE providers to download numpy arrays.
Multiple imagery sources are stacked along the channel axis. Patches are saved as
``.npy`` files with **relative** patch_path in metadata for cache/portability.

Usage::

    python -m training.patch_extraction --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

from src.config import init_ee

from .config import (
    PatchConfig,
    imagery_config_hash,
    imagery_metadata,
    load_config,
    resolve_paths,
    cache_key,
)
from .imagery import resolve_imagery_sources
from .imagery.base import ResolvedSource

log = logging.getLogger(__name__)


def _make_region(lat: float, lng: float, patch_cfg: PatchConfig) -> ee.Geometry:
    """Build an EE geometry rectangle centred on (lat, lng) with patch extent."""
    half_extent = patch_cfg.patch_extent_m / 2
    deg_offset = half_extent / 111_000
    return ee.Geometry.Rectangle([
        lng - deg_offset, lat - deg_offset,
        lng + deg_offset, lat + deg_offset,
    ])


def _build_grid(lat: float, lng: float, patch_cfg: PatchConfig) -> dict:
    """Build the EE computePixels grid specification in UTM projection."""
    half = patch_cfg.patch_extent_m / 2
    utm_zone = int((lng + 180) / 6) + 1
    crs = f"EPSG:326{utm_zone:02d}" if lat >= 0 else f"EPSG:327{utm_zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    easting, northing = transformer.transform(lng, lat)
    return {
        "dimensions": {
            "width": patch_cfg.patch_size_px,
            "height": patch_cfg.patch_size_px,
        },
        "affineTransform": {
            "scaleX": patch_cfg.resolution_m,
            "shearX": 0,
            "translateX": easting - half,
            "scaleY": -patch_cfg.resolution_m,
            "shearY": 0,
            "translateY": northing + half,
        },
        "crsCode": crs,
    }


def _unpack_structured(arr: np.ndarray, band_names: list[str]) -> np.ndarray:
    """Extract channels from a structured (named-field) numpy array."""
    channels = [arr[n] for n in band_names if n in arr.dtype.names]
    if not channels:
        channels = [arr[n] for n in arr.dtype.names]
    return np.stack(channels, axis=0).astype(np.float32)


def _reshape_array(
    arr: np.ndarray, band_names: list[str], n_bands: int,
) -> np.ndarray:
    """Normalise raw computePixels output to shape ``(C, H, W)``."""
    if arr.dtype.names:
        return _unpack_structured(arr, band_names)
    if arr.ndim == 3 and arr.shape[2] == n_bands:
        return np.transpose(arr, (2, 0, 1)).astype(np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :].astype(np.float32)
    return arr.astype(np.float32)


def _extract_one_patch(
    candidate_id: str,
    lat: float,
    lng: float,
    state: str,
    country: str,
    patch_cfg: PatchConfig,
    sources: list[ResolvedSource],
    output_dir: Path,
    patches_root: Path,
    date_start: str,
    date_end: str,
    imagery_hash: str,
    imagery_meta: dict[str, str],
) -> dict | None:
    """Download one patch (multi-source stacked) as a .npy file. Returns metadata or None."""
    try:
        region = _make_region(lat, lng, patch_cfg)
        grid = _build_grid(lat, lng, patch_cfg)
        all_bands: list[str] = []
        arrays: list[np.ndarray] = []

        for source in sources:
            img = source.build_image(region, date_start, date_end)
            band_names = source.band_names()
            request = {
                "expression": img,
                "fileFormat": "NUMPY_NDARRAY",
                "grid": grid,
            }
            result = ee.data.computePixels(request)
            raw = result if isinstance(result, np.ndarray) else np.load(
                io.BytesIO(result), allow_pickle=True,
            )
            arr = _reshape_array(raw, band_names, len(band_names))
            all_bands.extend(band_names)
            arrays.append(arr)

        stacked = np.concatenate(arrays, axis=0).astype(np.float32)
        nan_frac = np.isnan(stacked).sum() / max(stacked.size, 1)

        state_part = state if state else "_"
        country_part = country if country else "_"
        patch_dir = output_dir / country_part / state_part / imagery_hash
        patch_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{candidate_id}.npy"
        out_path = patch_dir / filename
        np.save(out_path, stacked)

        rel_path = out_path.relative_to(patches_root)

        meta = {
            "candidate_id": candidate_id,
            "lat": lat,
            "lng": lng,
            "state": state,
            "n_channels": stacked.shape[0],
            "height": stacked.shape[1],
            "width": stacked.shape[2],
            "clear_pixel_fraction": round(1.0 - float(nan_frac), 4),
            "patch_path": str(rel_path),
            "imagery_config_hash": imagery_hash,
            **imagery_meta,
        }
        return meta
    except Exception as exc:
        log.warning(
            "Failed to extract patch for %s: %s",
            candidate_id,
            str(exc)[:200],
        )
        return None


_FLUSH_EVERY = 1000  # flush patch_meta.csv every N successful patches


def _flush_meta(rows: list[dict], patches_root: Path) -> list[dict]:
    """Append *rows* to ``patch_meta.csv`` and return an empty list."""
    if not rows:
        return rows
    meta_path = patches_root / "patch_meta.csv"
    write_header = not meta_path.exists()
    pd.DataFrame(rows).to_csv(meta_path, mode="a", header=write_header, index=False)
    log.info("Checkpoint: flushed %d rows to %s", len(rows), meta_path)
    return []


def _record_failed(candidate_id: str, error: str, patches_root: Path) -> None:
    """Append a failed candidate to ``failed_patches.csv``."""
    failed_path = patches_root / "failed_patches.csv"
    write_header = not failed_path.exists()
    pd.DataFrame([{"candidate_id": candidate_id, "error": error[:200]}]).to_csv(
        failed_path, mode="a", header=write_header, index=False
    )


def _load_failed_ids(patches_root: Path) -> set[str]:
    """Load previously failed candidate IDs."""
    failed_path = patches_root / "failed_patches.csv"
    if not failed_path.exists():
        return set()
    return set(pd.read_csv(failed_path, usecols=["candidate_id"])["candidate_id"].astype(str))


def _extract_sequential(
    candidates: pd.DataFrame,
    patch_cfg: PatchConfig,
    sources: list[ResolvedSource],
    output_dir: Path,
    patches_root: Path,
    imagery_hash: str,
    imagery_meta: dict[str, str],
) -> list[dict]:
    date_start = patch_cfg.date_range[0]
    date_end = patch_cfg.date_range[1]
    rows: list[dict] = []
    total = len(candidates)
    for i, (_, row) in enumerate(candidates.iterrows()):
        cid = str(row.get("id", i))
        meta = _extract_one_patch(
            cid,
            float(row["lat"]),
            float(row["lng"]),
            str(row.get("state", "")),
            str(row.get("country", "")),
            patch_cfg,
            sources,
            output_dir,
            patches_root,
            date_start,
            date_end,
            imagery_hash,
            imagery_meta,
        )
        if meta:
            rows.append(meta)
        else:
            _record_failed(cid, "extraction failed", patches_root)
        if (i + 1) % 10 == 0:
            log.info("  %d / %d patches extracted", i + 1, total)
        if len(rows) >= _FLUSH_EVERY:
            rows = _flush_meta(rows, patches_root)
    return rows


def _extract_parallel(
    candidates: pd.DataFrame,
    patch_cfg: PatchConfig,
    sources: list[ResolvedSource],
    output_dir: Path,
    patches_root: Path,
    imagery_hash: str,
    imagery_meta: dict[str, str],
) -> list[dict]:
    date_start = patch_cfg.date_range[0]
    date_end = patch_cfg.date_range[1]
    rows: list[dict] = []
    total = len(candidates)
    with ThreadPoolExecutor(max_workers=patch_cfg.num_workers) as pool:
        cid_map = {}
        for i, (_, row) in enumerate(candidates.iterrows()):
            cid = str(row.get("id", i))
            fut = pool.submit(
                _extract_one_patch,
                cid,
                float(row["lat"]),
                float(row["lng"]),
                str(row.get("state", "")),
                str(row.get("country", "")),
                patch_cfg,
                sources,
                output_dir,
                patches_root,
                date_start,
                date_end,
                imagery_hash,
                imagery_meta,
            )
            cid_map[fut] = cid
        done = 0
        for fut in as_completed(cid_map):
            meta = fut.result()
            if meta:
                rows.append(meta)
            else:
                _record_failed(cid_map[fut], "extraction failed", patches_root)
            done += 1
            if done % 10 == 0:
                log.info("  %d / %d patches extracted", done, total)
            if len(rows) >= _FLUSH_EVERY:
                rows = _flush_meta(rows, patches_root)
    return rows


PATCHES_ROOT_NAME = "patches"


def _get_patches_root(output_dir: Path) -> Path:
    """Walk up from *output_dir* to find the ``patches/`` ancestor."""
    for parent in [output_dir, *output_dir.parents]:
        if parent.name == PATCHES_ROOT_NAME:
            return parent
    return output_dir.parent


def extract_patches(
    candidates: pd.DataFrame,
    patch_cfg: PatchConfig,
    max_patches: int | None = None,
    patches_root: Path | None = None,
) -> pd.DataFrame:
    """Extract patches for all candidates. Returns metadata DataFrame.

    Uses imagery_sources from config (or legacy single EE S2). Writes
    patch_path as relative to *patches_root* (default: ``data/patches/``).
    Appends metadata rows to ``{patches_root}/patch_meta.csv``.
    """
    output_dir = Path(patch_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if patches_root is None:
        patches_root = _get_patches_root(output_dir)

    if max_patches is not None:
        candidates = candidates.head(max_patches)

    sources = resolve_imagery_sources(patch_cfg)
    imagery_hash = imagery_config_hash(patch_cfg)
    band_names = [b for s in sources for b in s.band_names()]
    imagery_meta = imagery_metadata(patch_cfg, band_names)

    # Skip candidates already present in patch_meta.csv (resume support)
    meta_path = patches_root / "patch_meta.csv"
    skip_ids: set[str] = set()
    if meta_path.exists():
        skip_ids = set(pd.read_csv(meta_path, usecols=["candidate_id"])["candidate_id"].astype(str))

    # Also skip previously failed patches (unless retry_failed is set)
    if not patch_cfg.retry_failed:
        failed_ids = _load_failed_ids(patches_root)
        if failed_ids:
            log.info("Skipping %d previously failed patches (set retry_failed: true to retry).", len(failed_ids))
            skip_ids |= failed_ids
    else:
        # Clear failed log so they get a fresh attempt
        failed_path = patches_root / "failed_patches.csv"
        if failed_path.exists():
            failed_path.unlink()
            log.info("Cleared failed_patches.csv — retrying all previously failed patches.")

    if skip_ids:
        before = len(candidates)
        candidates = candidates[~candidates["id"].astype(str).isin(skip_ids)]
        skipped = before - len(candidates)
        if skipped:
            log.info("Skipping %d already-extracted/failed patches, %d remaining.", skipped, len(candidates))

    if len(candidates) == 0:
        log.info("All patches already extracted.")
        return pd.read_csv(meta_path)

    log.info(
        "Extracting %d patches (workers=%d, sources=%d, imagery_hash=%s) ...",
        len(candidates),
        patch_cfg.num_workers,
        len(sources),
        imagery_hash,
    )

    if patch_cfg.num_workers <= 1:
        rows = _extract_sequential(
            candidates, patch_cfg, sources, output_dir, patches_root,
            imagery_hash, imagery_meta,
        )
    else:
        rows = _extract_parallel(
            candidates, patch_cfg, sources, output_dir, patches_root,
            imagery_hash, imagery_meta,
        )

    # Flush any remaining rows not yet written by incremental checkpoints
    _flush_meta(rows, patches_root)

    meta_path = patches_root / "patch_meta.csv"
    return pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()


def _load_candidates_csv(candidates_dir: str, countries: list[str]) -> pd.DataFrame:
    """Load candidate CSVs from ``{candidates_dir}/{country}.csv``."""
    frames: list[pd.DataFrame] = []
    for country in countries:
        csv_path = Path(candidates_dir) / f"{country}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            frames.append(df)
            log.info("Loaded %d candidates from %s", len(df), csv_path)
        else:
            log.warning("Candidates file not found: %s", csv_path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    from .env_loader import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract image patches (Sentinel-2 / Sentinel-1 / multi-source)"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    cfg = resolve_paths(load_config(args.config))
    init_ee()

    candidates = _load_candidates_csv(cfg.data.candidates_dir, cfg.data.countries)
    if len(candidates) == 0:
        log.error(
            "No candidates found in %s for countries %s -- run candidates.py first",
            cfg.data.candidates_dir,
            cfg.data.countries,
        )
        return

    extract_patches(candidates, cfg.patches, max_patches=args.max_patches)

    if getattr(cfg, "cache", None) and cfg.cache.enabled:
        from .storage import get_cache_backend
        backend = get_cache_backend(cfg)
        if backend:
            key = cache_key(cfg)
            backend.put_dir(Path(cfg.patches.output_dir), key)
            log.info("Cached patches to %s under key %s", cfg.cache.backend, key)


if __name__ == "__main__":
    main()
