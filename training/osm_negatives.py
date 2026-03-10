"""Fetch non-farm buildings from OpenStreetMap as hard-negative candidates.

For each configured region the module:

1. Resolves a bounding box (country-level from ``COUNTRIES`` or US-state-level
   from a built-in lookup table).
2. Queries the Overpass API for buildings matching the configured ``osm_tags``
   (warehouse, industrial, hangar, ...) that are **not** tagged as farms.
3. Filters results to area > 500 m^2.
4. Removes any point within ``min_distance_m`` of a known positive farm.
5. Caches the cleaned result to ``data/osm_negatives/{region_slug}.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.config import COUNTRIES

from .config import build_region_string, parse_region

log = logging.getLogger(__name__)

# US state bounding boxes: (min_lon, min_lat, max_lon, max_lat)
US_STATE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    "AL": (-88.47, 30.22, -84.89, 35.01),
    "AK": (-179.15, 51.21, -129.97, 71.39),
    "AZ": (-114.81, 31.33, -109.04, 37.00),
    "AR": (-94.62, 33.00, -89.64, 36.50),
    "CA": (-124.41, 32.53, -114.13, 42.01),
    "CO": (-109.06, 36.99, -102.04, 41.00),
    "CT": (-73.73, 40.95, -71.79, 42.05),
    "DE": (-75.79, 38.45, -75.05, 39.84),
    "FL": (-87.63, 24.40, -79.97, 31.00),
    "GA": (-85.61, 30.36, -80.84, 35.00),
    "HI": (-160.24, 18.91, -154.81, 22.24),
    "ID": (-117.24, 41.99, -111.04, 49.00),
    "IL": (-91.51, 36.97, -87.02, 42.51),
    "IN": (-88.10, 37.77, -84.78, 41.76),
    "IA": (-96.64, 40.38, -90.14, 43.50),
    "KS": (-102.05, 36.99, -94.59, 40.00),
    "KY": (-89.57, 36.50, -81.96, 39.15),
    "LA": (-94.04, 28.93, -88.82, 33.02),
    "ME": (-71.08, 43.06, -66.95, 47.46),
    "MD": (-79.49, 37.91, -75.05, 39.72),
    "MA": (-73.51, 41.24, -69.93, 42.89),
    "MI": (-90.42, 41.70, -82.12, 48.31),
    "MN": (-97.24, 43.50, -89.49, 49.38),
    "MS": (-91.66, 30.17, -88.10, 34.99),
    "MO": (-95.77, 36.00, -89.10, 40.61),
    "MT": (-116.05, 44.36, -104.04, 49.00),
    "NE": (-104.05, 40.00, -95.31, 43.00),
    "NV": (-120.00, 35.00, -114.04, 42.00),
    "NH": (-72.56, 42.70, -70.70, 45.30),
    "NJ": (-75.56, 38.93, -73.89, 41.36),
    "NM": (-109.05, 31.33, -103.00, 37.00),
    "NY": (-79.76, 40.50, -71.86, 45.02),
    "NC": (-84.32, 33.84, -75.46, 36.59),
    "ND": (-104.05, 45.94, -96.55, 49.00),
    "OH": (-84.82, 38.40, -80.52, 41.98),
    "OK": (-103.00, 33.62, -94.43, 37.00),
    "OR": (-124.57, 41.99, -116.46, 46.29),
    "PA": (-80.52, 39.72, -74.69, 42.27),
    "RI": (-71.86, 41.15, -71.12, 42.02),
    "SC": (-83.35, 32.03, -78.54, 35.22),
    "SD": (-104.06, 42.48, -96.44, 45.94),
    "TN": (-90.31, 34.98, -81.65, 36.68),
    "TX": (-106.65, 25.84, -93.51, 36.50),
    "UT": (-114.05, 37.00, -109.04, 42.00),
    "VT": (-73.44, 42.73, -71.46, 45.02),
    "VA": (-83.68, 36.54, -75.24, 39.47),
    "WA": (-124.85, 45.54, -116.92, 49.00),
    "WV": (-82.64, 37.20, -77.72, 40.64),
    "WI": (-92.89, 42.49, -86.25, 47.08),
    "WY": (-111.06, 40.99, -104.05, 45.01),
}


def get_region_bounds(region: str) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) for a region string."""
    country_key, state = parse_region(region)

    if state and country_key == "united_states":
        bounds = US_STATE_BOUNDS.get(state)
        if bounds:
            return bounds

    country_cfg = COUNTRIES.get(country_key)
    if country_cfg is None:
        raise ValueError(f"Unknown region: {region}")
    return country_cfg.bounds


def _region_slug(region: str) -> str:
    return region.replace("/", "_")


def _is_far_from_farms(
    lon: float, lat: float, pos_coords: np.ndarray, min_dist_m: float,
) -> bool:
    if len(pos_coords) == 0:
        return True
    dlat = (pos_coords[:, 1] - lat) * 111_000
    dlon = (pos_coords[:, 0] - lon) * 111_000 * np.cos(np.radians(lat))
    return float(np.min(np.sqrt(dlat**2 + dlon**2))) >= min_dist_m


_MAX_RETRIES = 4
_INITIAL_BACKOFF_S = 30


def _query_overpass_single(
    bbox: str, tag: str,
) -> list[dict]:
    """Run a single Overpass query for one building tag with retry."""
    import time

    import overpy

    query = (
        f'[out:json][timeout:180];\n'
        f'(\n'
        f'  way["building"="{tag}"]'
        f'["building"!="farm"]'
        f'["landuse"!="farmyard"]'
        f'({bbox});\n'
        f');\n'
        f'out center;'
    )

    api = overpy.Overpass()
    for attempt in range(_MAX_RETRIES):
        try:
            result = api.query(query)
            break
        except (overpy.exception.OverpassGatewayTimeout, overpy.exception.OverpassTooManyRequests):
            wait = _INITIAL_BACKOFF_S * (2 ** attempt)
            log.warning("  Overpass timeout/busy (attempt %d/%d), retrying in %ds ...",
                        attempt + 1, _MAX_RETRIES, wait)
            time.sleep(wait)
    else:
        log.error("  Overpass failed after %d retries for tag=%s", _MAX_RETRIES, tag)
        return []

    rows: list[dict] = []
    for way in result.ways:
        if way.center_lat is None or way.center_lon is None:
            continue
        rows.append({
            "osm_id": way.id,
            "lat": float(way.center_lat),
            "lng": float(way.center_lon),
            "building_type": way.tags.get("building", ""),
        })
    return rows


def _query_overpass(
    bounds: tuple[float, float, float, float],
    osm_tags: list[str],
) -> list[dict]:
    """Query Overpass for buildings matching *osm_tags* within *bounds*.

    Queries one tag at a time to avoid timeouts on large areas.
    """
    import time

    min_lon, min_lat, max_lon, max_lat = bounds
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    all_rows: list[dict] = []
    for tag in osm_tags:
        log.info("    querying tag=%s ...", tag)
        rows = _query_overpass_single(bbox, tag)
        log.info("    tag=%s: %d results", tag, len(rows))
        all_rows.extend(rows)
        time.sleep(2)

    seen: set[int] = set()
    deduped: list[dict] = []
    for r in all_rows:
        if r["osm_id"] not in seen:
            seen.add(r["osm_id"])
            deduped.append(r)
    return deduped


def _fetch_single_region(
    region: str,
    osm_tags: list[str],
    min_distance_m: float,
    pos_coords: np.ndarray,
    cache_dir: Path,
) -> gpd.GeoDataFrame:
    """Fetch (or load cached) OSM negatives for one region."""
    slug = _region_slug(region)
    cache_path = cache_dir / f"{slug}.parquet"

    if cache_path.exists():
        log.info("  %s: loading cached OSM negatives from %s", region, cache_path)
        return gpd.read_parquet(cache_path)

    bounds = get_region_bounds(region)
    log.info("  %s: querying Overpass (bounds=%s, tags=%s) ...", region, bounds, osm_tags)
    raw = _query_overpass(bounds, osm_tags)
    log.info("  %s: %d raw OSM buildings returned", region, len(raw))

    if not raw:
        return gpd.GeoDataFrame(
            columns=["osm_id", "lat", "lng", "building_type", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    filtered = [
        r for r in raw
        if _is_far_from_farms(r["lng"], r["lat"], pos_coords, min_distance_m)
    ]
    log.info("  %s: %d after farm-distance filter (%.0f m)",
             region, len(filtered), min_distance_m)

    if not filtered:
        df = pd.DataFrame(columns=["osm_id", "lat", "lng", "building_type"])
    else:
        df = pd.DataFrame(filtered)
    geometry = [Point(r["lng"], r["lat"]) for r in filtered]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache_path, index=False)
    log.info("  %s: cached %d negatives -> %s", region, len(gdf), cache_path)
    return gdf


_EMPTY_NEG_COLS = [
    "id", "lat", "lng", "label", "source", "country",
    "state", "region", "species", "name", "category", "geometry",
]


def _raw_to_candidate_row(idx: int, r: pd.Series) -> dict:
    """Convert a single raw OSM row to a candidate dict."""
    country_key, state = parse_region(_infer_region(r))
    country_cfg = COUNTRIES.get(country_key)
    country_name = country_cfg.name if country_cfg else country_key
    return {
        "id": f"osm_{r.get('osm_id', idx)}",
        "lat": r["lat"], "lng": r["lng"],
        "label": 0,
        "source": "osm_buildings",
        "country": country_name,
        "state": state or "",
        "region": build_region_string(country_key, state or ""),
        "species": "", "name": "",
        "category": r.get("building_type", ""),
    }


def _merge_region_frames(
    frames: list[gpd.GeoDataFrame], max_total: int,
) -> gpd.GeoDataFrame:
    """Merge per-region GDFs, cap total count, and convert to candidate schema."""
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    if len(merged) > max_total:
        merged = merged.sample(n=max_total, random_state=42).reset_index(drop=True)

    rows = [_raw_to_candidate_row(i, r) for i, r in merged.iterrows()]
    df = pd.DataFrame(rows)
    geometry = [Point(r["lng"], r["lat"]) for r in rows]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def fetch_osm_negatives(
    regions: list[str],
    osm_tags: list[str],
    max_total: int,
    min_distance_m: float,
    pos_coords: np.ndarray,
    cache_dir: str | Path = "data/osm_negatives",
) -> gpd.GeoDataFrame:
    """Fetch OSM building negatives for all *regions* and return a combined GDF.

    The returned GeoDataFrame has the standard candidate columns so it can be
    concatenated directly with positive candidates.
    """
    cache_path = Path(cache_dir)
    frames: list[gpd.GeoDataFrame] = []
    for region in regions:
        gdf = _fetch_single_region(region, osm_tags, min_distance_m, pos_coords, cache_path)
        if len(gdf) > 0:
            frames.append(gdf)

    if not frames:
        return gpd.GeoDataFrame(columns=_EMPTY_NEG_COLS, geometry="geometry", crs="EPSG:4326")

    return _merge_region_frames(frames, max_total)


def _infer_region(row: pd.Series) -> str:
    """Best-effort region inference from a raw OSM row's lat/lng."""
    lat, lng = row["lat"], row["lng"]

    for state_code, (smin_lon, smin_lat, smax_lon, smax_lat) in US_STATE_BOUNDS.items():
        if smin_lat <= lat <= smax_lat and smin_lon <= lng <= smax_lon:
            return f"united_states/{state_code}"

    for key, cfg in COUNTRIES.items():
        cmin_lon, cmin_lat, cmax_lon, cmax_lat = cfg.bounds
        if cmin_lat <= lat <= cmax_lat and cmin_lon <= lng <= cmax_lon:
            return key

    return "unknown"
