"""Pipeline orchestrator -- ties data loading, detection, dedup, and validation together."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import ee
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, shape

from .config import CountryConfig, DetectionParams
from .detection import (
    DETECTION_METHODS,
    compute_spectral_indices,
    get_sentinel2_composite,
)
from .geometry import fetch_ee_features, generate_tiles, spatial_dedup

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tile detection
# ---------------------------------------------------------------------------

def _cos_lat_for_tile(tile: ee.Geometry.Rectangle) -> float:
    """Approximate cos(latitude) for a tile, used in property computation."""
    coords = tile.bounds().coordinates().getInfo()[0]
    mid_lat = (coords[0][1] + coords[2][1]) / 2
    return float(np.cos(np.radians(mid_lat)))


def run_tile(
    tile: ee.Geometry,
    methods: Sequence[str],
    params: DetectionParams,
) -> dict[str, ee.FeatureCollection]:
    """Run selected detection methods on a single tile.

    Returns a dict mapping method name to its ee.FeatureCollection (server-side).
    Methods that need a Sentinel-2 composite share one; methods that don't (e.g.
    Google Open Buildings) skip the composite step.
    """
    cos_lat = _cos_lat_for_tile(tile)

    needs_composite = {"NDBI", "MetalRoof"}
    needs_s2 = bool(set(methods) & needs_composite)

    composite = None
    indices = None
    if needs_s2:
        composite = get_sentinel2_composite(
            tile, params.start_date, params.end_date, params.max_cloud_cover,
        )
        indices = compute_spectral_indices(composite)

    shared_kwargs = {
        "composite": composite,
        "indices": indices,
        "cos_lat": cos_lat,
    }

    results: dict[str, ee.FeatureCollection] = {}
    for method_name in methods:
        fn = DETECTION_METHODS.get(method_name)
        if fn is None:
            log.warning("Unknown method '%s' -- skipping", method_name)
            continue
        try:
            fc = fn(tile, params, **shared_kwargs)
            results[method_name] = fc
        except Exception as exc:
            log.warning("  %s failed on tile: %s", method_name, str(exc)[:100])

    return results


# ---------------------------------------------------------------------------
# Per-country pipeline
# ---------------------------------------------------------------------------

def _features_to_gdf(features: list[dict]) -> gpd.GeoDataFrame:
    """Convert a list of GeoJSON feature dicts to a GeoDataFrame."""
    if not features:
        return gpd.GeoDataFrame(
            columns=["geometry", "source", "area_m2", "length_m", "width_m",
                     "aspect_ratio", "centroid_lon", "centroid_lat"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    rows = []
    geometries = []
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry")
        if geom is not None:
            geometries.append(shape(geom))
        else:
            geometries.append(Point(
                float(props.get("centroid_lon", 0)),
                float(props.get("centroid_lat", 0)),
            ))
        rows.append(props)

    gdf = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")
    return gdf


def run_country(
    config: CountryConfig,
    known_farms: gpd.GeoDataFrame,
    methods: Optional[Sequence[str]] = None,
) -> gpd.GeoDataFrame:
    """Run the full detection pipeline for one country.

    1. Generate tiles around known farm locations.
    2. Run detection methods on each tile.
    3. Fetch results from EE.
    4. Merge + spatial dedup.

    Returns a GeoDataFrame of candidate detections.
    """
    if methods is None:
        methods = list(DETECTION_METHODS.keys())

    params = config.detection_params

    country_farms = known_farms[known_farms["country"] == config.name] if len(known_farms) > 0 else known_farms
    tiles = generate_tiles(
        config.bounds,
        config.tile_size_deg,
        seed_points=country_farms if len(country_farms) > 0 else None,
        buffer_factor=1.5,
    )

    log.info("Country %s: %d tiles to process, methods=%s", config.name, len(tiles), methods)

    all_features: list[dict] = []
    for tile_idx, tile in enumerate(tiles):
        log.info("  Tile %d/%d ...", tile_idx + 1, len(tiles))
        try:
            method_fcs = run_tile(tile, methods, params)
        except Exception as exc:
            log.warning("  Tile %d failed entirely: %s", tile_idx + 1, str(exc)[:100])
            continue

        for method_name, fc in method_fcs.items():
            feats = fetch_ee_features(fc, params.max_per_method_per_tile, method_name)
            all_features.extend(feats)

    log.info("  Total raw features: %d", len(all_features))

    candidates = _features_to_gdf(all_features)
    if len(candidates) > 0:
        candidates = spatial_dedup(candidates, radius_m=50)
        log.info("  After dedup: %d", len(candidates))

    return candidates


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    candidates: gpd.GeoDataFrame,
    known: gpd.GeoDataFrame,
    radius_m: float = 200,
) -> dict:
    """Compute per-method and combined precision / recall / F1.

    A known farm is matched if any candidate is within *radius_m* metres.
    A candidate is a true positive if it is within *radius_m* of any known farm.
    """
    if len(candidates) == 0 or len(known) == 0:
        return {"per_method": {}, "combined": _empty_metrics()}

    known_coords = np.column_stack([known.geometry.x, known.geometry.y])

    if "centroid_lon" in candidates.columns:
        cand_lons = candidates["centroid_lon"].astype(float).values
        cand_lats = candidates["centroid_lat"].astype(float).values
    else:
        cand_lons = candidates.geometry.centroid.x.values
        cand_lats = candidates.geometry.centroid.y.values
    cand_coords = np.column_stack([cand_lons, cand_lats])

    all_methods = candidates["source"].unique() if "source" in candidates.columns else ["all"]

    def _min_dist_to_set(point, ref_coords):
        dlat = (ref_coords[:, 1] - point[1]) * 111_000
        dlon = (ref_coords[:, 0] - point[0]) * 111_000 * np.cos(np.radians(point[1]))
        return np.min(np.sqrt(dlat**2 + dlon**2))

    def _count_matches(points_a, points_b):
        """Count how many points in *a* are within radius_m of any point in *b*."""
        if len(points_a) == 0 or len(points_b) == 0:
            return 0
        return sum(1 for pt in points_a if _min_dist_to_set(pt, points_b) < radius_m)

    def _calc(mask):
        cc = cand_coords[mask]
        n_cand, n_known = int(mask.sum()), len(known_coords)
        tp_cand = _count_matches(cc, known_coords)
        matched_known = _count_matches(known_coords, cc)
        return _build_metrics(n_cand, tp_cand, matched_known, n_known)

    per_method = {}
    for method in all_methods:
        mask = (candidates["source"] == method).values if "source" in candidates.columns else np.ones(len(candidates), dtype=bool)
        per_method[method] = _calc(mask)

    combined = _calc(np.ones(len(candidates), dtype=bool))

    return {"per_method": per_method, "combined": combined}


def _build_metrics(n_cand: int, tp: int, known_matched: int, n_known: int) -> dict:
    precision = tp / n_cand if n_cand else 0
    recall = known_matched / n_known if n_known else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "candidates": n_cand, "true_positives": tp,
        "known_matched": known_matched, "known_total": n_known,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
    }


def _empty_metrics() -> dict:
    return {
        "candidates": 0, "true_positives": 0, "known_matched": 0,
        "known_total": 0, "precision": 0, "recall": 0, "f1": 0,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_results(
    candidates: gpd.GeoDataFrame,
    known: gpd.GeoDataFrame,
    country_name: str,
    output_dir: str | Path = "output",
) -> dict[str, Path]:
    """Save candidates and known farms as GeoJSON and CSV.

    Returns a dict of output file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    slug = country_name.lower().replace(" ", "_")
    paths: dict[str, Path] = {}

    if len(candidates) > 0:
        geojson_path = out / f"{slug}_candidates.geojson"
        candidates.to_file(geojson_path, driver="GeoJSON")
        paths["candidates_geojson"] = geojson_path

        csv_path = out / f"{slug}_candidates.csv"
        cols = [c for c in candidates.columns if c != "geometry"]
        candidates[cols].to_csv(csv_path, index=False)
        paths["candidates_csv"] = csv_path

    if len(known) > 0:
        known_path = out / f"{slug}_known_farms.geojson"
        known.to_file(known_path, driver="GeoJSON")
        paths["known_geojson"] = known_path

    log.info("Exported %d files for %s to %s", len(paths), country_name, out)
    return paths
