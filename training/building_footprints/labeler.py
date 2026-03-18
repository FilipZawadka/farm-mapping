"""Label buildings by cross-referencing all available data sources.

For every building footprint, checks (in priority order):
1. Farm Transparency Project — proximity match to known farms
2. OSM farm tags — proximity match to OSM-tagged farms
3. OSM building tags — Overpass reverse-lookup for building type

Labels are unified via the taxonomy module so that the same real-world
category from different sources isn't counted twice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .taxonomy import UnifiedLabel, normalise_species, unify_label

log = logging.getLogger(__name__)

_M_PER_DEG = 111_000.0


def _build_tree(lats: np.ndarray, lons: np.ndarray) -> tuple[cKDTree, float]:
    """Build a KDTree in approximate meter coordinates."""
    mean_lat = float(np.mean(lats)) if len(lats) > 0 else 0.0
    cos_lat = np.cos(np.radians(mean_lat))
    xy = np.column_stack([lons * _M_PER_DEG * cos_lat, lats * _M_PER_DEG])
    return cKDTree(xy), cos_lat


def _proximity_match(
    bld_lats: np.ndarray,
    bld_lngs: np.ndarray,
    ref_lats: np.ndarray,
    ref_lngs: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match buildings to reference points within radius.

    Returns:
        is_match: bool array — True if building is within radius of a ref point
        distances: float array — distance to nearest ref point
        indices: int array — index of nearest ref point (valid only where is_match)
    """
    if len(ref_lats) == 0:
        return (
            np.zeros(len(bld_lats), dtype=bool),
            np.full(len(bld_lats), np.inf),
            np.zeros(len(bld_lats), dtype=int),
        )

    tree, cos_lat = _build_tree(ref_lats, ref_lngs)
    bld_xy = np.column_stack([
        bld_lngs * _M_PER_DEG * cos_lat,
        bld_lats * _M_PER_DEG,
    ])
    distances, indices = tree.query(bld_xy)
    is_match = distances <= radius_m
    return is_match, distances, indices


def _extract_coords(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract lat/lng arrays from a GeoDataFrame."""
    if "lat" in gdf.columns:
        return gdf["lat"].values, gdf["lng"].values
    return gdf.geometry.y.values, gdf.geometry.x.values


def _safe_col(df, col: str, dtype=str) -> np.ndarray:
    """Get column values with fallback to empty strings."""
    if col in df.columns:
        return df[col].fillna("").astype(dtype).values
    return np.full(len(df), "", dtype=object)


def label_buildings(
    buildings: pd.DataFrame,
    known_farms: gpd.GeoDataFrame,
    proximity_radius_m: float = 200,
    osm_farms: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """Assign unified labels to buildings from all available sources.

    For each building, checks:
    1. Farm Transparency farms (proximity match → positive with species/category)
    2. OSM farms (proximity match → positive, may add category)
    3. No match → negative (label=0, group="unknown")

    All labels are unified via the taxonomy so that "Farm (eggs)" from FTP
    and "building=chicken_coop" from OSM resolve to the same canonical label.

    Args:
        buildings: DataFrame with [lat, lng, area_m2, building_id, provider].
        known_farms: Known farm locations from Farm Transparency + OSM data sources.
        proximity_radius_m: Match threshold in meters.
        osm_farms: Optional separate OSM farm GeoDataFrame (if loaded independently).
            If None, OSM farms are assumed to be included in known_farms.

    Returns:
        DataFrame with added columns: label, unified_group, unified_label,
        species, category, name, matched_farm_id, matched_distance_m, source_label.
    """
    if len(buildings) == 0:
        for col in ["label", "unified_group", "unified_label", "species",
                     "category", "name", "matched_farm_id", "matched_distance_m",
                     "source_label"]:
            buildings[col] = pd.Series(dtype=str if col != "label" else int)
        return buildings

    buildings = buildings.copy()
    n = len(buildings)
    bld_lats = buildings["lat"].values
    bld_lngs = buildings["lng"].values

    # Initialise label columns
    buildings["label"] = 0
    buildings["unified_group"] = "unknown"
    buildings["unified_label"] = "unknown"
    buildings["species"] = ""
    buildings["category"] = ""
    buildings["name"] = ""
    buildings["matched_farm_id"] = ""
    buildings["matched_distance_m"] = np.inf
    buildings["source_label"] = ""

    # --- Source 1: Farm Transparency farms ---
    if known_farms is not None and len(known_farms) > 0:
        # Split by source to apply correct taxonomy mapping
        ftp_mask = known_farms["source"].astype(str).str.contains(
            "FarmTransparency|farm_transparency", case=False, na=False,
        )
        osm_mask = ~ftp_mask

        # Check FTP farms first (higher quality labels)
        ftp_farms = known_farms[ftp_mask]
        if len(ftp_farms) > 0:
            ftp_lats, ftp_lngs = _extract_coords(ftp_farms)
            is_match, dists, idxs = _proximity_match(
                bld_lats, bld_lngs, ftp_lats, ftp_lngs, proximity_radius_m,
            )

            ftp_ids = _safe_col(ftp_farms, "id")
            ftp_species = _safe_col(ftp_farms, "species")
            ftp_categories = _safe_col(ftp_farms, "category")
            ftp_names = _safe_col(ftp_farms, "name")

            for i in np.where(is_match)[0]:
                j = idxs[i]
                unified = unify_label(
                    source="farm_transparency",
                    raw_category=ftp_categories[j],
                    raw_species=ftp_species[j],
                )
                buildings.iloc[i, buildings.columns.get_loc("label")] = 1
                buildings.iloc[i, buildings.columns.get_loc("unified_group")] = unified.group
                buildings.iloc[i, buildings.columns.get_loc("unified_label")] = unified.label
                buildings.iloc[i, buildings.columns.get_loc("species")] = unified.species
                buildings.iloc[i, buildings.columns.get_loc("category")] = ftp_categories[j]
                buildings.iloc[i, buildings.columns.get_loc("name")] = ftp_names[j]
                buildings.iloc[i, buildings.columns.get_loc("matched_farm_id")] = ftp_ids[j]
                buildings.iloc[i, buildings.columns.get_loc("matched_distance_m")] = dists[i]
                buildings.iloc[i, buildings.columns.get_loc("source_label")] = "farm_transparency"

            log.info("  FTP match: %d buildings matched", int(is_match.sum()))

        # Check OSM farms for buildings not yet labelled
        osm_farm_data = known_farms[osm_mask]
        if osm_farms is not None and len(osm_farms) > 0:
            osm_farm_data = pd.concat([osm_farm_data, osm_farms], ignore_index=True)

        if len(osm_farm_data) > 0:
            unlabelled = buildings["label"] == 0
            if unlabelled.any():
                ul_lats = bld_lats[unlabelled]
                ul_lngs = bld_lngs[unlabelled]
                osm_lats, osm_lngs = _extract_coords(
                    gpd.GeoDataFrame(osm_farm_data, crs="EPSG:4326")
                    if not isinstance(osm_farm_data, gpd.GeoDataFrame)
                    else osm_farm_data
                )
                is_match, dists, idxs = _proximity_match(
                    ul_lats, ul_lngs, osm_lats, osm_lngs, proximity_radius_m,
                )

                osm_ids = _safe_col(osm_farm_data, "id")
                osm_species = _safe_col(osm_farm_data, "species")
                osm_categories = _safe_col(osm_farm_data, "category")
                osm_names = _safe_col(osm_farm_data, "name")

                ul_indices = np.where(unlabelled)[0]
                for k, i in enumerate(ul_indices):
                    if not is_match[k]:
                        continue
                    j = idxs[k]
                    unified = unify_label(
                        source="osm",
                        raw_category=osm_categories[j],
                        raw_species=osm_species[j],
                    )
                    buildings.iloc[i, buildings.columns.get_loc("label")] = 1 if unified.is_farm else 0
                    buildings.iloc[i, buildings.columns.get_loc("unified_group")] = unified.group
                    buildings.iloc[i, buildings.columns.get_loc("unified_label")] = unified.label
                    buildings.iloc[i, buildings.columns.get_loc("species")] = unified.species
                    buildings.iloc[i, buildings.columns.get_loc("category")] = osm_categories[j]
                    buildings.iloc[i, buildings.columns.get_loc("name")] = osm_names[j]
                    buildings.iloc[i, buildings.columns.get_loc("matched_farm_id")] = osm_ids[j]
                    buildings.iloc[i, buildings.columns.get_loc("matched_distance_m")] = dists[k]
                    buildings.iloc[i, buildings.columns.get_loc("source_label")] = "osm"

                log.info("  OSM farm match: %d buildings matched", int(is_match.sum()))

    n_pos = int((buildings["label"] == 1).sum())
    n_neg = n - n_pos
    n_labelled = int((buildings["unified_label"] != "unknown").sum())
    log.info(
        "Labelled %d buildings: %d positive, %d negative (%d with known category)",
        n, n_pos, n_neg, n_labelled,
    )

    return buildings
