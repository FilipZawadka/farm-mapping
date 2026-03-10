"""Earth Engine Sentinel-1 SAR imagery provider."""

from __future__ import annotations

from typing import Any

import ee

DEFAULT_COLLECTION = "COPERNICUS/S1_GRD"
DEFAULT_POLARISATIONS = ["VV"]


class EarthEngineSentinel1Provider:
    """Build Sentinel-1 GRD median composite (VV/VH) as an ee.Image."""

    def __init__(
        self,
        *,
        collection: str = DEFAULT_COLLECTION,
        polarisations: list[str] | None = None,
        **_kwargs: Any,
    ):
        self.collection = collection
        self.polarisations = polarisations or DEFAULT_POLARISATIONS

    def band_names(self) -> list[str]:
        return list(self.polarisations)

    def build_image(
        self,
        region: ee.Geometry,
        date_start: str,
        date_end: str,
    ) -> ee.Image:
        """Build S1 median composite for IW mode and selected polarisations."""
        s1 = (
            ee.ImageCollection(self.collection)
            .filterBounds(region)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        for pol in self.polarisations:
            s1 = s1.filter(
                ee.Filter.listContains("transmitterReceiverPolarisation", pol)
            )
        return s1.select(self.polarisations).median().clip(region)
