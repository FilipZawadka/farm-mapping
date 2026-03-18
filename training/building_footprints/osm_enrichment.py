"""Enrich building footprints with OSM building tags.

Batch-queries Overpass API for building tags in tile-sized regions, then
matches OSM buildings to footprint centroids by proximity. This classifies
negatives as "warehouse", "industrial", "residential", etc.

Cached per tile to avoid repeated API calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from .taxonomy import unify_label

log = logging.getLogger(__name__)

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_M_PER_DEG = 111_000.0

# Tags we're interested in for classification
_BUILDING_QUERY_TAGS = [
    "building", "landuse", "industrial", "amenity", "shop",
]


def _query_overpass_buildings(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float,
    timeout: int = 60,
) -> list[dict]:
    """Query Overpass for all tagged buildings in a bounding box.

    Returns list of dicts with keys: osm_id, lat, lng, tags (dict).
    """
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    query = f"""
    [out:json][timeout:{timeout}];
    (
      way["building"]({bbox});
      relation["building"]({bbox});
    );
    out center;
    """
    for attempt in range(3):
        try:
            resp = requests.post(
                _OVERPASS_URL,
                data={"data": query},
                timeout=timeout + 10,
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                log.warning("Overpass rate limit, waiting %ds ...", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, json.JSONDecodeError) as exc:
            log.warning("Overpass query failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    else:
        return []

    results = []
    for elem in data.get("elements", []):
        center = elem.get("center", {})
        lat = center.get("lat") or elem.get("lat")
        lng = center.get("lon") or elem.get("lon")
        if lat is None or lng is None:
            continue
        results.append({
            "osm_id": elem.get("id", 0),
            "lat": lat,
            "lng": lng,
            "tags": elem.get("tags", {}),
        })

    return results


def _tile_cache_key(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> str:
    s = f"{min_lat:.3f}_{min_lon:.3f}_{max_lat:.3f}_{max_lon:.3f}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


def enrich_with_osm_tags(
    buildings: pd.DataFrame,
    cache_dir: Path,
    tile_size_deg: float = 0.5,
    match_radius_m: float = 50,
) -> pd.DataFrame:
    """Add OSM building tags to unclassified buildings.

    Only enriches buildings where ``unified_label == "unknown"``.
    Queries Overpass in tiles, caches results, and matches by proximity.

    Args:
        buildings: DataFrame with lat, lng, unified_label columns.
        cache_dir: Directory for caching Overpass results.
        tile_size_deg: Tile size for batching Overpass queries.
        match_radius_m: Max distance to match an OSM building to a footprint.

    Returns:
        DataFrame with updated unified_group, unified_label, source_label
        for buildings that got OSM matches.
    """
    cache_dir = Path(cache_dir)
    osm_cache = cache_dir / "osm_tiles"
    osm_cache.mkdir(parents=True, exist_ok=True)

    unknown_mask = buildings["unified_label"] == "unknown"
    if not unknown_mask.any():
        log.info("No unknown buildings to enrich with OSM tags")
        return buildings

    buildings = buildings.copy()
    unknown_lats = buildings.loc[unknown_mask, "lat"].values
    unknown_lngs = buildings.loc[unknown_mask, "lng"].values

    # Determine tiles needed
    min_lat, max_lat = float(unknown_lats.min()), float(unknown_lats.max())
    min_lng, max_lng = float(unknown_lngs.min()), float(unknown_lngs.max())

    all_osm: list[dict] = []
    n_tiles = 0

    for lat in np.arange(min_lat, max_lat + tile_size_deg, tile_size_deg):
        for lon in np.arange(min_lng, max_lng + tile_size_deg, tile_size_deg):
            tile_max_lat = min(lat + tile_size_deg, max_lat + 0.01)
            tile_max_lon = min(lon + tile_size_deg, max_lng + 0.01)

            # Skip tiles with no unknown buildings
            in_tile = (
                (unknown_lats >= lat) & (unknown_lats < tile_max_lat) &
                (unknown_lngs >= lon) & (unknown_lngs < tile_max_lon)
            )
            if not in_tile.any():
                continue

            cache_key = _tile_cache_key(lat, lon, tile_max_lat, tile_max_lon)
            cache_path = osm_cache / f"{cache_key}.json"

            if cache_path.exists():
                with open(cache_path) as f:
                    osm_buildings = json.load(f)
            else:
                osm_buildings = _query_overpass_buildings(lat, lon, tile_max_lat, tile_max_lon)
                with open(cache_path, "w") as f:
                    json.dump(osm_buildings, f)
                time.sleep(1)  # rate limit

            all_osm.extend(osm_buildings)
            n_tiles += 1

    if not all_osm:
        log.info("No OSM buildings found in %d tiles", n_tiles)
        return buildings

    log.info("Queried %d OSM buildings from %d tiles", len(all_osm), n_tiles)

    # Match unknown buildings to OSM buildings by proximity
    osm_lats = np.array([b["lat"] for b in all_osm])
    osm_lngs = np.array([b["lng"] for b in all_osm])
    mean_lat = float(np.mean(osm_lats))
    cos_lat = np.cos(np.radians(mean_lat))

    osm_xy = np.column_stack([osm_lngs * _M_PER_DEG * cos_lat, osm_lats * _M_PER_DEG])
    tree = cKDTree(osm_xy)

    bld_xy = np.column_stack([
        unknown_lngs * _M_PER_DEG * cos_lat,
        unknown_lats * _M_PER_DEG,
    ])
    dists, idxs = tree.query(bld_xy)

    n_enriched = 0
    unknown_indices = np.where(unknown_mask)[0]

    for k, i in enumerate(unknown_indices):
        if dists[k] > match_radius_m:
            continue
        osm_bld = all_osm[idxs[k]]
        tags = osm_bld.get("tags", {})

        # Build a tag string for taxonomy lookup
        building_tag = tags.get("building", "")
        landuse_tag = tags.get("landuse", "")

        # Try building tag first, then landuse
        unified = None
        if building_tag:
            unified = unify_label(source="osm", raw_category=f"building={building_tag}")
        if (unified is None or unified.label == "unknown") and landuse_tag:
            unified = unify_label(source="osm", raw_category=f"landuse={landuse_tag}")

        if unified and unified.label != "unknown":
            buildings.iloc[i, buildings.columns.get_loc("unified_group")] = unified.group
            buildings.iloc[i, buildings.columns.get_loc("unified_label")] = unified.label
            buildings.iloc[i, buildings.columns.get_loc("source_label")] = "osm_enrichment"
            # If the OSM tag says it's a farm, update label
            if unified.is_farm and buildings.iloc[i, buildings.columns.get_loc("label")] == 0:
                buildings.iloc[i, buildings.columns.get_loc("label")] = 1
                buildings.iloc[i, buildings.columns.get_loc("species")] = unified.species
            n_enriched += 1

    log.info(
        "OSM enrichment: %d/%d unknown buildings got a category",
        n_enriched, int(unknown_mask.sum()),
    )

    return buildings
