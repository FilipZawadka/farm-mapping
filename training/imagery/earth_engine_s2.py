"""Earth Engine Sentinel-2 imagery provider."""

from __future__ import annotations

import logging
from typing import Any

import ee

from src.detection import get_sentinel2_composite

log = logging.getLogger(__name__)

INDEX_FORMULAS: dict[str, tuple[str, str]] = {
    "NDVI": ("B8", "B4"),
    "NDBI": ("B11", "B8"),
    "NDWI": ("B3", "B8"),
    "MNDWI": ("B3", "B11"),
}

DEFAULT_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
DEFAULT_INDICES = ["NDVI", "NDBI", "NDWI"]


class EarthEngineSentinel2Provider:
    """Build Sentinel-2 SR composite + indices as an ee.Image for patch extraction."""

    def __init__(
        self,
        *,
        collection: str = DEFAULT_COLLECTION,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        max_cloud_cover: int = 15,
        composite: str = "median",
        cloud_mask: str = "none",
        **_kwargs: Any,
    ):
        self.collection = collection
        self.bands = bands or DEFAULT_BANDS
        self.indices = indices or DEFAULT_INDICES
        self.max_cloud_cover = max_cloud_cover
        self.composite = composite
        self.cloud_mask = cloud_mask

    def band_names(self) -> list[str]:
        return list(self.bands) + [
            i for i in self.indices if i in INDEX_FORMULAS
        ]

    @staticmethod
    def _mask_scl_clouds(img: ee.Image) -> ee.Image:
        """Mask cloud shadow (3), cloud medium/high probability (8, 9) and
        cirrus (10) using the Scene Classification Layer."""
        scl = img.select("SCL")
        clear = (
            scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        )
        return img.updateMask(clear)

    def build_image(
        self,
        region: ee.Geometry,
        date_start: str,
        date_end: str,
    ) -> ee.Image:
        """Build S2 composite + selected bands and indices clipped to region."""
        if self.cloud_mask == "scl":
            s2 = (
                ee.ImageCollection(self.collection)
                .filterBounds(region)
                .filterDate(date_start, date_end)
                .filter(
                    ee.Filter.lt(
                        "CLOUDY_PIXEL_PERCENTAGE", self.max_cloud_cover
                    )
                )
                .map(self._mask_scl_clouds)
            )
            composite_img = s2.median().clip(region)
        elif self.collection != DEFAULT_COLLECTION:
            s2 = (
                ee.ImageCollection(self.collection)
                .filterBounds(region)
                .filterDate(date_start, date_end)
                .filter(
                    ee.Filter.lt(
                        "CLOUDY_PIXEL_PERCENTAGE", self.max_cloud_cover
                    )
                )
            )
            composite_img = s2.median().clip(region)
        else:
            composite_img = get_sentinel2_composite(
                region,
                date_start,
                date_end,
                max_cloud=self.max_cloud_cover,
            )

        bands_img = composite_img.select(self.bands)
        index_bands: list[ee.Image] = []
        for idx_name in self.indices:
            formula = INDEX_FORMULAS.get(idx_name)
            if formula is None:
                log.warning("Unknown index '%s' -- skipping", idx_name)
                continue
            idx_img = composite_img.normalizedDifference(
                list(formula)
            ).rename(idx_name)
            index_bands.append(idx_img)

        if index_bands:
            return bands_img.addBands(index_bands)
        return bands_img
