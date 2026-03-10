"""Pluggable Earth Engine detection methods.

Each public ``detect_*`` function has the signature::

    (geometry, params, **kwargs) -> ee.FeatureCollection

and tags every output feature with ``source = <MethodName>``.

The :data:`DETECTION_METHODS` registry maps human-readable names to callables.
The pipeline iterates over this dict, so adding a new method is just:

    1. Write a ``detect_foo(...)`` function.
    2. Add ``"Foo": detect_foo`` to :data:`DETECTION_METHODS`.
"""

from __future__ import annotations

import logging
from typing import Callable

import ee

from .config import DetectionParams
from .geometry import make_property_adder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel-2 composite & spectral indices (shared by several methods)
# ---------------------------------------------------------------------------

def get_sentinel2_composite(
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud: int = 15,
) -> ee.Image:
    """Cloud-free Sentinel-2 SR median composite clipped to *geometry*."""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )
    return s2.median().clip(geometry)


def compute_spectral_indices(composite: ee.Image) -> dict[str, ee.Image]:
    """Return a dict of spectral index images derived from a Sentinel-2 composite."""
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    brightness = composite.select(["B4", "B3", "B2"]).reduce(ee.Reducer.mean())
    swir = composite.select("B11")
    return {
        "ndbi": ndbi, "ndvi": ndvi, "ndwi": ndwi, "mndwi": mndwi,
        "brightness": brightness, "swir": swir,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _loose_area_filter(fc: ee.FeatureCollection, params: DetectionParams) -> ee.FeatureCollection:
    """Keep features whose area_m2 falls within the loose range."""
    return fc.filter(
        ee.Filter.And(
            ee.Filter.gte("area_m2", params.min_area_m2),
            ee.Filter.lte("area_m2", params.max_area_m2),
        )
    )


def _vectorize(
    mask: ee.Image,
    geometry: ee.Geometry,
    params: DetectionParams,
    label: str = "class",
) -> ee.FeatureCollection:
    """Convert a binary mask to vector polygons."""
    return mask.selfMask().reduceToVectors(
        geometry=geometry,
        scale=params.vectorize_scale,
        geometryType="polygon",
        eightConnected=True,
        labelProperty=label,
        maxPixels=params.max_pixels,
        tileScale=params.tile_scale,
    )


def _morphological_open(mask: ee.Image, radius: int) -> ee.Image:
    """Focal-min then focal-max to remove isolated bright pixels."""
    if radius <= 0:
        return mask
    kernel = ee.Kernel.circle(radius=radius)
    return mask.focalMin(kernel=kernel).focalMax(kernel=kernel)


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def detect_ndbi(
    geometry: ee.Geometry,
    params: DetectionParams,
    *,
    indices: dict[str, ee.Image],
    cos_lat: float = 1.0,
    **_kwargs,
) -> ee.FeatureCollection:
    """NDBI-based built-up area detection with vegetation/water exclusion."""
    built_mask = (
        indices["ndbi"].gt(params.ndbi_threshold)
        .And(indices["ndvi"].lt(params.ndbi_ndvi_max))
        .And(indices["ndwi"].lt(params.ndbi_water_max))
        .And(indices["mndwi"].lt(params.ndbi_water_max))
    )
    built_mask = _morphological_open(built_mask, params.morph_kernel_radius)
    vectors = _vectorize(built_mask, geometry, params, label="ndbi")
    candidates = _loose_area_filter(vectors.map(make_property_adder(cos_lat)), params)
    return candidates.map(lambda f: f.set("source", "NDBI"))


def detect_metal_roof(
    geometry: ee.Geometry,
    params: DetectionParams,
    *,
    composite: ee.Image,
    indices: dict[str, ee.Image],
    cos_lat: float = 1.0,
    **_kwargs,
) -> ee.FeatureCollection:
    """Metal-roof spectral detection (high brightness + high SWIR + low vegetation)."""
    metal_mask = (
        indices["brightness"].gt(params.metal_brightness_min)
        .And(indices["ndvi"].lt(params.metal_ndvi_max))
        .And(indices["swir"].gt(params.metal_swir_min))
        .And(indices["mndwi"].lt(0.1))
    )
    metal_mask = _morphological_open(metal_mask, params.morph_kernel_radius)
    vectors = _vectorize(metal_mask, geometry, params, label="metal")
    candidates = _loose_area_filter(vectors.map(make_property_adder(cos_lat)), params)
    return candidates.map(lambda f: f.set("source", "MetalRoof"))


def detect_google_open_buildings(
    geometry: ee.Geometry,
    params: DetectionParams,
    *,
    cos_lat: float = 1.0,
    **_kwargs,
) -> ee.FeatureCollection:
    """Google Open Buildings v3 filtered by confidence and loose area."""
    ob = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
    ob_area = ob.filterBounds(geometry)

    if params.ob_min_confidence > 0:
        ob_area = ob_area.filter(ee.Filter.gte("confidence", params.ob_min_confidence))

    ob_area = ob_area.filter(
        ee.Filter.And(
            ee.Filter.gte("area_in_meters", params.min_area_m2),
            ee.Filter.lte("area_in_meters", params.max_area_m2),
        )
    )

    def _add_props(feature):
        geom = feature.geometry()
        centroid = geom.centroid()
        area = feature.get("area_in_meters")
        bounds = geom.bounds()
        coords = ee.List(bounds.coordinates().get(0))
        p0 = ee.List(coords.get(0))
        p1 = ee.List(coords.get(1))
        p2 = ee.List(coords.get(2))
        w = ee.Geometry.Point(p0).distance(ee.Geometry.Point(p1))
        h = ee.Geometry.Point(p1).distance(ee.Geometry.Point(p2))
        return feature.set({
            "area_m2": area,
            "length_m": w.max(h),
            "width_m": w.min(h),
            "aspect_ratio": w.max(h).divide(w.min(h).add(0.001)),
            "centroid_lon": centroid.coordinates().get(0),
            "centroid_lat": centroid.coordinates().get(1),
            "source": "GoogleOpenBuildings",
        })

    return ob_area.map(_add_props)


def detect_dynamic_world(
    geometry: ee.Geometry,
    params: DetectionParams,
    *,
    cos_lat: float = 1.0,
    **_kwargs,
) -> ee.FeatureCollection:
    """Dynamic World built-up class detection."""
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(geometry)
        .filterDate(params.start_date, params.end_date)
    )
    built_class = dw.select("label").mode().eq(params.dw_built_class).clip(geometry)
    vectors = _vectorize(built_class, geometry, params, label="dw")
    candidates = _loose_area_filter(vectors.map(make_property_adder(cos_lat)), params)
    return candidates.map(lambda f: f.set("source", "DynamicWorld"))


def detect_sar(
    geometry: ee.Geometry,
    params: DetectionParams,
    *,
    cos_lat: float = 1.0,
    **_kwargs,
) -> ee.FeatureCollection:
    """Sentinel-1 SAR backscatter detection (VV polarisation)."""
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geometry)
        .filterDate(params.start_date, params.end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )
    s1_median = s1.median().clip(geometry)
    sar_mask = s1_median.gt(params.sar_threshold_db)
    sar_mask = _morphological_open(sar_mask, params.morph_kernel_radius)
    vectors = _vectorize(sar_mask, geometry, params, label="sar")
    candidates = _loose_area_filter(vectors.map(make_property_adder(cos_lat)), params)
    return candidates.map(lambda f: f.set("source", "SAR"))


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

# Keys are the canonical method names used throughout the pipeline.
# Values are callables; the pipeline passes them (geometry, params, **shared_kwargs).
DETECTION_METHODS: dict[str, Callable] = {
    "NDBI": detect_ndbi,
    "MetalRoof": detect_metal_roof,
    "GoogleOpenBuildings": detect_google_open_buildings,
    "DynamicWorld": detect_dynamic_world,
    "SAR": detect_sar,
}
