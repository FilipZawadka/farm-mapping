"""Building footprint-based candidate generation.

Queries building footprint databases (Google Open Buildings, Microsoft Global
ML Building Footprints) via Earth Engine, then labels each building as
positive (near a known farm) or negative (not near any known farm).

This produces better training data than random rural negatives because
negatives are real buildings that visually resemble farms.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.config import COUNTRIES, init_ee

from ..config import (
    BuildingFootprintConfig,
    DataConfig,
    build_country_key_map,
    build_region_string,
)
from .labeler import label_buildings
from .providers import get_provider, query_country_buildings

log = logging.getLogger(__name__)


def _infer_states(
    lats: np.ndarray, lngs: np.ndarray, country_keys: np.ndarray,
) -> list[str]:
    """Infer US state codes from coordinates using bounding-box lookup.

    Non-US buildings get empty state. US buildings outside any state bbox
    also get empty state (will be treated as country-wide).
    """
    from ..osm_negatives import US_STATE_BOUNDS

    states = [""] * len(lats)
    us_mask = country_keys == "united_states"
    if not us_mask.any():
        return states

    # Pre-compute state bounds arrays for vectorised check
    state_codes = list(US_STATE_BOUNDS.keys())
    bounds = np.array([US_STATE_BOUNDS[s] for s in state_codes])  # (n_states, 4)

    for i in np.where(us_mask)[0]:
        lat, lng = lats[i], lngs[i]
        # Check which state bbox contains this point
        in_state = (
            (lng >= bounds[:, 0]) & (lng <= bounds[:, 2]) &
            (lat >= bounds[:, 1]) & (lat <= bounds[:, 3])
        )
        matches = np.where(in_state)[0]
        if len(matches) > 0:
            states[i] = state_codes[matches[0]]

    n_resolved = sum(1 for s in states if s)
    n_us = int(us_mask.sum())
    log.info("State inference: %d/%d US buildings resolved to a state", n_resolved, n_us)

    return states


def fetch_building_candidates(
    cfg: DataConfig,
    known_farms: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Query building footprints and label them as farm / non-farm.

    Args:
        cfg: Data configuration (must have ``building_footprints`` section).
        known_farms: GeoDataFrame of known farm locations used for labelling.

    Returns:
        GeoDataFrame with standard candidate schema:
        [id, lat, lng, label, source, country, state, region, species,
         category, name, area_m2, geometry].
    """
    bf_cfg = cfg.building_footprints
    prov_cfg = bf_cfg.provider
    cache_dir = Path(bf_cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    init_ee()

    provider = get_provider(
        prov_cfg.name,
        min_confidence=prov_cfg.min_confidence,
    )

    name_to_key = build_country_key_map()
    all_buildings: list[pd.DataFrame] = []

    for country_key in cfg.countries:
        country_cfg = COUNTRIES.get(country_key)
        if country_cfg is None:
            log.warning("Country %r not in registry — skipping", country_key)
            continue

        if not provider.covers_country(country_cfg.iso_code):
            log.warning(
                "Provider %r has no data for %s (%s) — skipping",
                prov_cfg.name, country_key, country_cfg.iso_code,
            )
            continue

        # Check cache
        cache_key = (
            f"{country_key}_{prov_cfg.name}"
            f"_{int(prov_cfg.min_area_m2)}_{int(prov_cfg.max_area_m2)}"
            f"_{prov_cfg.min_confidence}"
        )
        cache_path = cache_dir / f"{cache_key}.parquet"

        if cache_path.exists():
            log.info("Loading cached buildings for %s from %s", country_key, cache_path)
            df = pd.read_parquet(cache_path)
        else:
            log.info(
                "Querying %s buildings for %s (bounds=%s, area=%d-%d m²) ...",
                prov_cfg.name, country_key, country_cfg.bounds,
                int(prov_cfg.min_area_m2), int(prov_cfg.max_area_m2),
            )
            rows = query_country_buildings(
                provider=provider,
                bounds=country_cfg.bounds,
                iso_code=country_cfg.iso_code,
                min_area_m2=prov_cfg.min_area_m2,
                max_area_m2=prov_cfg.max_area_m2,
                tile_size_deg=bf_cfg.tile_size_deg,
                max_buildings=bf_cfg.max_buildings_per_country,
            )
            if not rows:
                log.info("  No buildings found for %s", country_key)
                continue

            df = pd.DataFrame(rows)
            df.to_parquet(cache_path, index=False)
            log.info("  Cached %d buildings to %s", len(df), cache_path)

        df["country"] = country_cfg.name
        df["country_key"] = country_key
        all_buildings.append(df)

    if not all_buildings:
        log.warning("No buildings found for any country")
        return gpd.GeoDataFrame(
            columns=["id", "lat", "lng", "label", "source", "country",
                     "state", "region", "species", "category", "name",
                     "area_m2", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )

    buildings = pd.concat(all_buildings, ignore_index=True)
    log.info("Total buildings queried: %d", len(buildings))

    # Label by proximity to known farms (FTP + OSM)
    buildings = label_buildings(
        buildings, known_farms, proximity_radius_m=bf_cfg.proximity_radius_m,
    )

    # Enrich unclassified buildings with OSM building tags
    n_unknown = int((buildings["unified_label"] == "unknown").sum())
    if n_unknown > 0:
        from .osm_enrichment import enrich_with_osm_tags
        log.info("Enriching %d unclassified buildings with OSM tags ...", n_unknown)
        buildings = enrich_with_osm_tags(
            buildings,
            cache_dir=Path(bf_cfg.cache_dir),
            tile_size_deg=bf_cfg.tile_size_deg,
        )

    # Convert to candidate schema
    buildings["id"] = buildings["building_id"]
    buildings["source"] = buildings["provider"]

    # Infer state from coordinates (critical for region-based splits)
    buildings["state"] = _infer_states(
        buildings["lat"].values,
        buildings["lng"].values,
        buildings["country_key"].values,
    )
    buildings["region"] = [
        build_region_string(k, s)
        for k, s in zip(buildings["country_key"], buildings["state"])
    ]

    geometry = [Point(row.lng, row.lat) for row in buildings.itertuples()]
    gdf = gpd.GeoDataFrame(buildings, geometry=geometry, crs="EPSG:4326")

    # Keep standard columns + unified labels for analysis / future multi-class
    keep_cols = [
        "id", "lat", "lng", "label", "source", "country", "state",
        "region", "species", "category", "name", "area_m2",
        "unified_group", "unified_label", "source_label", "geometry",
    ]
    for col in keep_cols:
        if col not in gdf.columns:
            gdf[col] = ""
    gdf = gdf[keep_cols]

    n_pos = (gdf["label"] == 1).sum()
    n_neg = (gdf["label"] == 0).sum()
    log.info("Building footprint candidates: %d positive, %d negative", n_pos, n_neg)

    return gdf
