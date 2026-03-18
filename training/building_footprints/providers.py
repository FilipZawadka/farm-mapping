"""Building footprint providers for Earth Engine.

Each provider queries a building footprint dataset (Google Open Buildings,
Microsoft Global ML Building Footprints) and returns centroids + area for
buildings above a minimum size threshold.

Usage is internal — call via :func:`fetch_building_candidates` in ``__init__.py``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import ee
import numpy as np

log = logging.getLogger(__name__)

# Countries covered by Google Open Buildings v3
# (Africa, Latin America, Caribbean, South Asia, Southeast Asia)
_GOOGLE_OB_COUNTRIES: set[str] = {
    # Africa
    "DZ", "AO", "BJ", "BW", "BF", "BI", "CM", "CV", "CF", "TD", "KM",
    "CD", "CG", "CI", "DJ", "EG", "GQ", "ER", "SZ", "ET", "GA", "GM",
    "GH", "GN", "GW", "KE", "LS", "LR", "LY", "MG", "MW", "ML", "MR",
    "MU", "MA", "MZ", "NA", "NE", "NG", "RW", "ST", "SN", "SC", "SL",
    "SO", "ZA", "SS", "SD", "TZ", "TG", "TN", "UG", "ZM", "ZW",
    # Latin America / Caribbean
    "AR", "BO", "BR", "CL", "CO", "CR", "CU", "DO", "EC", "SV", "GT",
    "HN", "HT", "JM", "MX", "NI", "PA", "PY", "PE", "PR", "TT", "UY",
    "VE",
    # South & Southeast Asia
    "AF", "BD", "BT", "KH", "IN", "ID", "LA", "MY", "MM", "NP", "PK",
    "PH", "LK", "TH", "TL", "VN",
}

# Microsoft Buildings collection uses country display names
# MS Buildings asset names use the country's display name with spaces.
# Verify via: ee.data.listAssets("projects/sat-io/open-datasets/MSBuildings")
_MS_COUNTRY_NAMES: dict[str, str] = {
    "US": "United States",
    "GB": "United Kingdom",
    "AU": "Australia",
    "CA": "Canada",
    "DE": "Germany",
    "BR": "Brazil",
    "MX": "Mexico",
    "CL": "Chile",
    "AR": "Argentina",
    "TH": "Thailand",
    "PH": "Philippines",
    "IN": "India",
    "ID": "Indonesia",
    "MY": "Malaysia",
    "VN": "Vietnam",
    "BD": "Bangladesh",
    "PK": "Pakistan",
    "CO": "Colombia",
    "PE": "Peru",
    "VE": "Venezuela",
    "ZA": "South Africa",
    "NG": "Nigeria",
    "KE": "Kenya",
    "EG": "Egypt",
    "TZ": "Tanzania",
    "ET": "Ethiopia",
    "GH": "Ghana",
    "FR": "France",
    "ES": "Spain",
    "IT": "Italy",
    "PL": "Poland",
    "NL": "Netherlands",
    "BE": "Belgium",
    "AT": "Austria",
    "SE": "Sweden",
    "NO": "Norway",
    "DK": "Denmark",
    "FI": "Finland",
    "JP": "Japan",
    "KR": "South Korea",
    "NZ": "New Zealand",
}


class BuildingProvider(ABC):
    """Base class for building footprint data sources."""

    @abstractmethod
    def query_tile(
        self,
        geometry: ee.Geometry,
        min_area_m2: float,
        max_area_m2: float,
        **kwargs,
    ) -> list[dict]:
        """Query buildings in a tile geometry.

        Returns list of dicts with keys: lat, lng, area_m2, building_id, provider.
        """

    @abstractmethod
    def covers_country(self, iso_code: str) -> bool:
        """Return True if this provider has data for the given ISO country code."""


class GoogleOpenBuildingsProvider(BuildingProvider):
    """Google Open Buildings v3 — 1.8B buildings in Africa, LatAm, S/SE Asia."""

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self._collection = ee.FeatureCollection(
            "GOOGLE/Research/open-buildings/v3/polygons"
        )

    def covers_country(self, iso_code: str) -> bool:
        return iso_code.upper() in _GOOGLE_OB_COUNTRIES

    def query_tile(
        self,
        geometry: ee.Geometry,
        min_area_m2: float,
        max_area_m2: float,
        **kwargs,
    ) -> list[dict]:
        fc = self._collection.filterBounds(geometry)
        if self.min_confidence > 0:
            fc = fc.filter(ee.Filter.gte("confidence", self.min_confidence))
        fc = fc.filter(
            ee.Filter.And(
                ee.Filter.gte("area_in_meters", min_area_m2),
                ee.Filter.lte("area_in_meters", max_area_m2),
            )
        )

        # Extract centroid + area server-side, pull as lists
        def _extract(f):
            centroid = f.geometry().centroid()
            return f.set({
                "centroid_lat": centroid.coordinates().get(1),
                "centroid_lon": centroid.coordinates().get(0),
            })

        fc = fc.map(_extract)

        # Pull data in one batch (tile should be small enough)
        try:
            info = fc.reduceColumns(
                ee.Reducer.toList(3),
                ["centroid_lat", "centroid_lon", "area_in_meters"],
            ).getInfo()
        except ee.EEException as exc:
            log.warning("Google OB query failed for tile: %s", exc)
            return []

        rows = info.get("list", [])
        return [
            {
                "lat": r[0],
                "lng": r[1],
                "area_m2": r[2],
                "building_id": f"gob_{r[1]:.6f}_{r[0]:.6f}",
                "provider": "google_open_buildings",
            }
            for r in rows
        ]


class MSBuildingsProvider(BuildingProvider):
    """Microsoft Global ML Building Footprints — 777M buildings worldwide."""

    def covers_country(self, iso_code: str) -> bool:
        return iso_code.upper() in _MS_COUNTRY_NAMES

    def _collection_id(self, iso_code: str) -> str:
        name = _MS_COUNTRY_NAMES.get(iso_code.upper(), iso_code)
        return f"projects/sat-io/open-datasets/MSBuildings/{name}"

    def query_tile(
        self,
        geometry: ee.Geometry,
        min_area_m2: float,
        max_area_m2: float,
        *,
        iso_code: str = "",
        **kwargs,
    ) -> list[dict]:
        collection_id = self._collection_id(iso_code)
        fc = ee.FeatureCollection(collection_id).filterBounds(geometry)

        # Compute area server-side and filter
        def _add_area(f):
            area = f.geometry().area()
            centroid = f.geometry().centroid()
            return f.set({
                "area_m2": area,
                "centroid_lat": centroid.coordinates().get(1),
                "centroid_lon": centroid.coordinates().get(0),
            })

        fc = fc.map(_add_area).filter(
            ee.Filter.And(
                ee.Filter.gte("area_m2", min_area_m2),
                ee.Filter.lte("area_m2", max_area_m2),
            )
        )

        try:
            info = fc.reduceColumns(
                ee.Reducer.toList(3),
                ["centroid_lat", "centroid_lon", "area_m2"],
            ).getInfo()
        except ee.EEException as exc:
            msg = str(exc)
            if "not found" in msg or "does not exist" in msg:
                log.error("MS Buildings collection not found: %s", collection_id)
                return None  # signals collection-level error, not just empty tile
            log.warning("MS Buildings query failed for tile: %s", exc)
            return []

        rows = info.get("list", [])
        return [
            {
                "lat": r[0],
                "lng": r[1],
                "area_m2": r[2],
                "building_id": f"msb_{r[1]:.6f}_{r[0]:.6f}",
                "provider": "ms_buildings",
            }
            for r in rows
        ]


class AutoProvider(BuildingProvider):
    """Auto-select: Google Open Buildings if covered, else Microsoft Buildings."""

    def __init__(self, min_confidence: float = 0.65):
        self._google = GoogleOpenBuildingsProvider(min_confidence=min_confidence)
        self._ms = MSBuildingsProvider()

    def covers_country(self, iso_code: str) -> bool:
        return self._google.covers_country(iso_code) or self._ms.covers_country(iso_code)

    def _pick(self, iso_code: str) -> BuildingProvider:
        if self._google.covers_country(iso_code):
            return self._google
        return self._ms

    def query_tile(
        self,
        geometry: ee.Geometry,
        min_area_m2: float,
        max_area_m2: float,
        *,
        iso_code: str = "",
        **kwargs,
    ) -> list[dict]:
        provider = self._pick(iso_code)
        return provider.query_tile(
            geometry, min_area_m2, max_area_m2, iso_code=iso_code, **kwargs,
        )


def get_provider(name: str, min_confidence: float = 0.65) -> BuildingProvider:
    """Factory for building footprint providers."""
    if name == "google_open_buildings":
        return GoogleOpenBuildingsProvider(min_confidence=min_confidence)
    elif name == "ms_buildings":
        return MSBuildingsProvider()
    elif name == "auto":
        return AutoProvider(min_confidence=min_confidence)
    raise ValueError(f"Unknown building footprint provider: {name!r}")


def query_country_buildings(
    provider: BuildingProvider,
    bounds: tuple[float, float, float, float],
    iso_code: str,
    min_area_m2: float,
    max_area_m2: float,
    tile_size_deg: float = 0.5,
    max_buildings: int = 50_000,
) -> list[dict]:
    """Query buildings for an entire country, tiled to avoid EE limits.

    Args:
        provider: Building footprint provider instance.
        bounds: (min_lon, min_lat, max_lon, max_lat) country bounding box.
        iso_code: ISO 3166-1 alpha-2 country code.
        min_area_m2: Minimum building footprint area.
        max_area_m2: Maximum building footprint area.
        tile_size_deg: Size of query tiles in degrees.
        max_buildings: Stop after collecting this many buildings.

    Returns:
        List of building dicts with keys: lat, lng, area_m2, building_id, provider.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    all_buildings: list[dict] = []

    # Validate the collection exists before tiling thousands of queries
    test_geom = ee.Geometry.Rectangle([min_lon, min_lat, min_lon + tile_size_deg, min_lat + tile_size_deg])
    test_result = provider.query_tile(test_geom, min_area_m2, max_area_m2, iso_code=iso_code)
    if test_result is None:
        log.error("  %s: provider returned None for test tile — collection likely not found", iso_code)
        return []

    lats = np.arange(min_lat, max_lat, tile_size_deg)
    lons = np.arange(min_lon, max_lon, tile_size_deg)
    n_tiles = len(lats) * len(lons)
    tile_idx = 0

    for lat in lats:
        for lon in lons:
            if len(all_buildings) >= max_buildings:
                break
            tile_idx += 1
            tile_geom = ee.Geometry.Rectangle([
                lon, lat,
                min(lon + tile_size_deg, max_lon),
                min(lat + tile_size_deg, max_lat),
            ])
            buildings = provider.query_tile(
                tile_geom, min_area_m2, max_area_m2, iso_code=iso_code,
            )
            if buildings is None:
                # Collection-level error (not found) — bail immediately
                log.error("  %s: provider returned None — aborting", iso_code)
                return all_buildings
            all_buildings.extend(buildings)

            if tile_idx % 20 == 0 or buildings:
                log.info(
                    "  %s tile %d/%d: %d buildings (total: %d)",
                    iso_code, tile_idx, n_tiles, len(buildings), len(all_buildings),
                )

            # Rate-limit EE queries
            time.sleep(0.1)

        if len(all_buildings) >= max_buildings:
            log.info("  %s: hit max_buildings=%d, stopping", iso_code, max_buildings)
            break

    return all_buildings[:max_buildings]
