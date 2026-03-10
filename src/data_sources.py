"""Pluggable data-source loaders that produce a unified GeoDataFrame schema.

Every loader returns a GeoDataFrame with columns:
    id, name, lat, lng, species, category, source, country, geometry
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .config import CountryConfig

SCHEMA_COLUMNS = [
    "id", "name", "lat", "lng", "species", "category", "source", "country", "state", "geometry",
]

# Species values that represent poultry
_POULTRY = {"Chickens", "Turkeys", "Ducks", "Geese"}
# Species values that represent pigs
_PIGS = {"Pigs"}


# ---------------------------------------------------------------------------
# Farm Transparency Project loader
# ---------------------------------------------------------------------------

def load_farm_transparency(
    csv_path: Path | str,
    country: str,
    species_filter: Optional[Sequence[str]] = None,
    categories_include: Optional[Sequence[str]] = None,
) -> gpd.GeoDataFrame:
    """Load a Farm Transparency Project CSV and return a standardised GeoDataFrame.

    Parameters
    ----------
    csv_path : path to the CSV file
    country : country name to stamp on every row
    species_filter : keep only rows whose Species is in this list (None = keep all)
    categories_include : keep only rows whose Categories contain one of these
        substrings (e.g. ``["Farm"]``).  None = keep all.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Id": "id", "Name": "name", "Lat": "lat", "Lng": "lng",
                            "Species": "species", "Categories": "category"})

    # Drop rows without valid coordinates
    df = df.dropna(subset=["lat", "lng"])
    df = df[(df["lat"] != 0) & (df["lng"] != 0)]

    # Filter to actual farms (exclude zoos, slaughterhouses, etc.)
    if categories_include is not None:
        mask = pd.Series(False, index=df.index)
        for substr in categories_include:
            mask |= df["category"].fillna("").str.contains(substr, case=False, na=False, regex=False)
        df = df[mask]

    if species_filter is not None:
        df = df[df["species"].isin(species_filter)]

    df["source"] = "FarmTransparency"
    df["country"] = country
    if "State" in df.columns:
        df["state"] = df["State"]
    elif "state" not in df.columns:
        df["state"] = ""

    geometry = [Point(lng, lat) for lng, lat in zip(df["lng"], df["lat"])]
    gdf = gpd.GeoDataFrame(df[["id", "name", "lat", "lng", "species", "category",
                                "source", "country", "state"]], geometry=geometry, crs="EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# OpenStreetMap farm loader
# ---------------------------------------------------------------------------

def load_osm_farms(
    csv_path: Path | str,
    country: str,
    species_filter: Optional[Sequence[str]] = None,
) -> gpd.GeoDataFrame:
    """Load an OSM farms CSV (same header format as Farm Transparency)."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Id": "id", "Name": "name", "Lat": "lat", "Lng": "lng",
                            "Species": "species", "Categories": "category"})

    df = df.dropna(subset=["lat", "lng"])
    df = df[(df["lat"] != 0) & (df["lng"] != 0)]

    if species_filter is not None:
        df = df[df["species"].isin(species_filter)]

    df["source"] = "OSM"
    df["country"] = country
    if "State" in df.columns:
        df["state"] = df["State"]
    elif "state" not in df.columns:
        df["state"] = ""

    geometry = [Point(lng, lat) for lng, lat in zip(df["lng"], df["lat"])]
    gdf = gpd.GeoDataFrame(df[["id", "name", "lat", "lng", "species", "category",
                                "source", "country", "state"]], geometry=geometry, crs="EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# Merge & deduplicate
# ---------------------------------------------------------------------------

def merge_sources(
    gdfs: list[gpd.GeoDataFrame],
    dedup_radius_m: float = 100,
) -> gpd.GeoDataFrame:
    """Concatenate multiple GeoDataFrames and remove near-duplicates.

    When two points from different sources fall within *dedup_radius_m* of each
    other, only the first (highest-priority source) is kept.
    """
    if not gdfs:
        return gpd.GeoDataFrame(columns=SCHEMA_COLUMNS, geometry="geometry", crs="EPSG:4326")

    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    if len(merged) <= 1:
        return merged

    coords = np.column_stack([merged.geometry.x, merged.geometry.y])
    keep = np.ones(len(coords), dtype=bool)

    for i in range(len(coords)):
        if not keep[i]:
            continue
        dlat = (coords[i + 1:, 1] - coords[i, 1]) * 111_000
        dlon = (coords[i + 1:, 0] - coords[i, 0]) * 111_000 * np.cos(np.radians(coords[i, 1]))
        dists = np.sqrt(dlat**2 + dlon**2)
        too_close = np.where(dists < dedup_radius_m)[0] + i + 1
        keep[too_close] = False

    return merged[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: load all available sources for a country
# ---------------------------------------------------------------------------

def load_known_farms(
    config: CountryConfig,
    categories_include: Optional[Sequence[str]] = ("Farm",),
) -> gpd.GeoDataFrame:
    """Load and merge all available data sources for a single country.

    Returns a deduplicated GeoDataFrame in the unified schema.
    """
    gdfs: list[gpd.GeoDataFrame] = []

    if config.ft_path is not None:
        gdfs.append(
            load_farm_transparency(
                config.ft_path,
                country=config.name,
                species_filter=config.species_filter,
                categories_include=categories_include,
            )
        )

    if config.osm_full_path is not None:
        gdfs.append(
            load_osm_farms(
                config.osm_full_path,
                country=config.name,
                species_filter=config.species_filter,
            )
        )

    return merge_sources(gdfs)
