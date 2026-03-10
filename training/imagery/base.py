"""Protocol and types for imagery providers.

Providers build an image (ee.Image or later np.ndarray) for a region with
date range and provider-specific options. Used by patch extraction to
stack multiple sources into one (C, H, W) array per location.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Earth Engine image type; avoid importing ee at protocol level for optional deps
try:
    import ee
    EEImage = ee.Image
except ImportError:
    EEImage = Any


@runtime_checkable
class ImageryProvider(Protocol):
    """Protocol for building a multi-band image for a region."""

    def band_names(self) -> list[str]:
        """Return ordered list of band names this provider produces."""
        ...

    def build_image(
        self,
        region: EEImage,  # type: ignore[name-defined]
        date_start: str,
        date_end: str,
        **options: Any,
    ) -> EEImage:  # type: ignore[name-defined]
        """Build image for the region and date range. Returns ee.Image (or np.ndarray later)."""
        ...


class ResolvedSource:
    """A provider instance (configured with options) ready for extraction."""

    def __init__(self, provider: ImageryProvider):
        self.provider = provider

    def band_names(self) -> list[str]:
        return self.provider.band_names()

    def build_image(self, region: Any, date_start: str, date_end: str) -> Any:
        return self.provider.build_image(region, date_start, date_end)
