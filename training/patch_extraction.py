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
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import init_ee

from .config import PatchConfig, load_config, resolve_paths, cache_key
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
    """Build the EE computePixels grid specification."""
    half = patch_cfg.patch_extent_m / 2
    cos_lat = np.cos(np.radians(lat))
    utm_zone = int((lng + 180) / 6) + 1
    crs = f"EPSG:326{utm_zone:02d}" if lat >= 0 else f"EPSG:327{utm_zone:02d}"
    return {
        "dimensions": {
            "width": patch_cfg.patch_size_px,
            "height": patch_cfg.patch_size_px,
        },
        "affineTransform": {
            "scaleX": patch_cfg.resolution_m,
            "shearX": 0,
            "translateX": lng * 111_000 * cos_lat - half,
            "scaleY": -patch_cfg.resolution_m,
            "shearY": 0,
            "translateY": lat * 111_000 + half,
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
    patch_cfg: PatchConfig,
    sources: list[ResolvedSource],
    output_dir: Path,
    date_start: str,
    date_end: str,
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
            arr = _reshape_array(
                np.load(io.BytesIO(result)), band_names, len(band_names)
            )
            all_bands.extend(band_names)
            arrays.append(arr)

        stacked = np.concatenate(arrays, axis=0).astype(np.float32)
        nan_frac = np.isnan(stacked).sum() / max(stacked.size, 1)

        filename = f"{candidate_id}.npy"
        out_path = output_dir / filename
        np.save(out_path, stacked)

        return {
            "candidate_id": candidate_id,
            "lat": lat,
            "lng": lng,
            "n_channels": stacked.shape[0],
            "height": stacked.shape[1],
            "width": stacked.shape[2],
            "clear_pixel_fraction": round(1.0 - float(nan_frac), 4),
            "patch_path": filename,
        }
    except Exception as exc:
        log.warning(
            "Failed to extract patch for %s: %s",
            candidate_id,
            str(exc)[:200],
        )
        return None


def _extract_sequential(
    candidates: gpd.GeoDataFrame,
    patch_cfg: PatchConfig,
    sources: list[ResolvedSource],
    output_dir: Path,
) -> list[dict]:
    date_start = patch_cfg.date_range[0]
    date_end = patch_cfg.date_range[1]
    rows: list[dict] = []
    total = len(candidates)
    for i, (_, row) in enumerate(candidates.iterrows()):
        meta = _extract_one_patch(
            str(row.get("id", i)),
            float(row["lat"]),
            float(row["lng"]),
            patch_cfg,
            sources,
            output_dir,
            date_start,
            date_end,
        )
        if meta:
            rows.append(meta)
        if (i + 1) % 10 == 0:
            log.info("  %d / %d patches extracted", i + 1, total)
    return rows


def _extract_parallel(
    candidates: gpd.GeoDataFrame,
    patch_cfg: PatchConfig,
    sources: list[ResolvedSource],
    output_dir: Path,
) -> list[dict]:
    date_start = patch_cfg.date_range[0]
    date_end = patch_cfg.date_range[1]
    rows: list[dict] = []
    total = len(candidates)
    with ThreadPoolExecutor(max_workers=patch_cfg.num_workers) as pool:
        futures = {
            pool.submit(
                _extract_one_patch,
                str(row.get("id", i)),
                float(row["lat"]),
                float(row["lng"]),
                patch_cfg,
                sources,
                output_dir,
                date_start,
                date_end,
            ): i
            for i, (_, row) in enumerate(candidates.iterrows())
        }
        done = 0
        for fut in as_completed(futures):
            meta = fut.result()
            if meta:
                rows.append(meta)
            done += 1
            if done % 10 == 0:
                log.info("  %d / %d patches extracted", done, total)
    return rows


def extract_patches(
    candidates: gpd.GeoDataFrame,
    patch_cfg: PatchConfig,
    max_patches: int | None = None,
) -> pd.DataFrame:
    """Extract patches for all candidates. Returns metadata DataFrame.

    Uses imagery_sources from config (or legacy single EE S2). Writes
    patch_path as relative filename for cache/portability.
    """
    output_dir = Path(patch_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_patches is not None:
        candidates = candidates.head(max_patches)

    sources = resolve_imagery_sources(patch_cfg)
    log.info(
        "Extracting %d patches (workers=%d, sources=%d) ...",
        len(candidates),
        patch_cfg.num_workers,
        len(sources),
    )

    if patch_cfg.num_workers <= 1:
        rows = _extract_sequential(
            candidates, patch_cfg, sources, output_dir
        )
    else:
        rows = _extract_parallel(
            candidates, patch_cfg, sources, output_dir
        )

    meta_df = pd.DataFrame(rows)
    if len(meta_df) > 0:
        meta_path = output_dir / "patch_meta.parquet"
        meta_df.to_parquet(meta_path, index=False)
        log.info("Saved patch metadata: %s (%d patches)", meta_path, len(meta_df))

    return meta_df


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

    candidates_path = Path(cfg.patches.output_dir) / "candidates.parquet"
    if not candidates_path.exists():
        log.error(
            "No candidates.parquet found at %s -- run candidates.py first",
            candidates_path,
        )
        return
    candidates = gpd.read_parquet(candidates_path)
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
