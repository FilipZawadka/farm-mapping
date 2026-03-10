"""Shared geospatial utilities -- method-agnostic.

Functions here work with Earth Engine objects (server-side) or local GeoDataFrames
(client-side) and are reused across all detection methods and the pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

import ee
import geopandas as gpd
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Earth Engine: geometry property computation
# ---------------------------------------------------------------------------

def add_geometry_properties(feature: ee.Feature, cos_lat: float = 1.0) -> ee.Feature:
    """Compute area, length, width, aspect ratio, and centroid for *feature*.

    This is designed to be used as ``fc.map(lambda f: add_geometry_properties(f, cos))``.
    Because EE lambda closures only accept a single argument, callers should
    use :func:`make_property_adder` to bake in *cos_lat*.
    """
    geom = feature.geometry()
    area = geom.area(maxError=1)

    bounds = geom.bounds(maxError=1)
    coords = ee.List(bounds.coordinates().get(0))
    p0 = ee.List(coords.get(0))
    p1 = ee.List(coords.get(1))
    p2 = ee.List(coords.get(2))

    width_deg = ee.Number(p1.get(0)).subtract(ee.Number(p0.get(0))).abs()
    height_deg = ee.Number(p2.get(1)).subtract(ee.Number(p1.get(1))).abs()

    meters_per_deg = 111_000
    width_m = width_deg.multiply(meters_per_deg).multiply(cos_lat)
    height_m = height_deg.multiply(meters_per_deg)

    length = width_m.max(height_m)
    width = width_m.min(height_m)
    aspect_ratio = length.divide(width.max(0.1))

    centroid = geom.centroid(maxError=1)

    return feature.set({
        "area_m2": area,
        "length_m": length,
        "width_m": width,
        "aspect_ratio": aspect_ratio,
        "centroid_lon": centroid.coordinates().get(0),
        "centroid_lat": centroid.coordinates().get(1),
    })


def make_property_adder(cos_lat: float):
    """Return a one-argument callable suitable for ``fc.map(...)``."""
    def _adder(feature):
        return add_geometry_properties(feature, cos_lat=cos_lat)
    return _adder


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------

def generate_tiles(
    bounds: tuple[float, float, float, float],
    tile_size_deg: float,
    seed_points: Optional[gpd.GeoDataFrame] = None,
    buffer_factor: float = 1.5,
) -> list[ee.Geometry.Rectangle]:
    """Grid *bounds* into tiles, optionally restricting to tiles near *seed_points*.

    Parameters
    ----------
    bounds : (min_lon, min_lat, max_lon, max_lat)
    tile_size_deg : size of each square tile in degrees
    seed_points : if provided, only tiles that contain at least one seed point
        (after buffering) are returned.
    buffer_factor : expand each tile's inclusion zone by this factor when
        checking seed membership (e.g. 1.5 = 50% larger).
    """
    min_lon, min_lat, max_lon, max_lat = bounds

    tiles: list[ee.Geometry.Rectangle] = []
    lon = min_lon
    while lon < max_lon:
        lat = min_lat
        while lat < max_lat:
            t_min_lon = lon
            t_min_lat = lat
            t_max_lon = min(lon + tile_size_deg, max_lon)
            t_max_lat = min(lat + tile_size_deg, max_lat)

            if seed_points is not None and len(seed_points) > 0:
                half = tile_size_deg * buffer_factor / 2
                cx = (t_min_lon + t_max_lon) / 2
                cy = (t_min_lat + t_max_lat) / 2
                pt_x = seed_points.geometry.x
                pt_y = seed_points.geometry.y
                in_tile = (
                    pt_x.between(cx - half, cx + half)
                    & pt_y.between(cy - half, cy + half)
                )
                if not in_tile.any():
                    lat += tile_size_deg
                    continue

            tiles.append(ee.Geometry.Rectangle([t_min_lon, t_min_lat, t_max_lon, t_max_lat]))
            lat += tile_size_deg
        lon += tile_size_deg

    return tiles


# ---------------------------------------------------------------------------
# Spatial deduplication (client-side)
# ---------------------------------------------------------------------------

def spatial_dedup(gdf: gpd.GeoDataFrame, radius_m: float = 50) -> gpd.GeoDataFrame:
    """Remove near-duplicate points/centroids within *radius_m* metres."""
    if len(gdf) <= 1:
        return gdf

    if "centroid_lon" in gdf.columns and "centroid_lat" in gdf.columns:
        lons = gdf["centroid_lon"].values.astype(float)
        lats = gdf["centroid_lat"].values.astype(float)
    else:
        lons = gdf.geometry.centroid.x.values
        lats = gdf.geometry.centroid.y.values

    keep = np.ones(len(gdf), dtype=bool)
    for i, (lon_i, lat_i) in enumerate(zip(lons, lats)):
        if not keep[i]:
            continue
        dlat = (lats[i + 1:] - lat_i) * 111_000
        dlon = (lons[i + 1:] - lon_i) * 111_000 * np.cos(np.radians(lat_i))
        dists = np.sqrt(dlat**2 + dlon**2)
        too_close = np.where(dists < radius_m)[0] + i + 1
        keep[too_close] = False

    return gdf[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Safe EE feature fetching
# ---------------------------------------------------------------------------

def fetch_ee_features(
    fc: ee.FeatureCollection,
    max_per_call: int = 2000,
    method_name: str = "unknown",
) -> list[dict]:
    """Fetch features from an ee.FeatureCollection with size guard and error handling.

    Returns a list of GeoJSON feature dicts (client-side).
    """
    try:
        count = fc.size().getInfo()
    except Exception as exc:
        log.warning("  %s: size() failed -- %s", method_name, str(exc)[:80])
        return []

    if count == 0:
        log.info("  %s: 0 candidates", method_name)
        return []

    fetched = fc if count <= max_per_call else fc.limit(max_per_call)

    try:
        geojson = fetched.getInfo()
    except Exception as exc:
        log.warning("  %s: getInfo() failed -- %s", method_name, str(exc)[:80])
        return []

    features = geojson.get("features", [])
    for feat in features:
        feat["properties"]["source"] = method_name

    actual = len(features)
    if count > max_per_call:
        log.info("  %s: %d candidates (limited to %d)", method_name, count, actual)
    else:
        log.info("  %s: %d candidates", method_name, actual)

    return features
