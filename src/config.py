"""Country definitions, detection parameters, and Earth Engine authentication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class DetectionParams:
    """Thresholds for each detection method. Override per-country or per-run."""

    # Sentinel-2 imagery
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    max_cloud_cover: int = 15

    # Loose area filter (applied to all raster-based methods after vectorization)
    min_area_m2: float = 100
    max_area_m2: float = 50_000

    # NDBI
    ndbi_threshold: float = 0.15
    ndbi_ndvi_max: float = 0.3
    ndbi_water_max: float = 0.0

    # Metal roof spectral
    metal_brightness_min: float = 2000
    metal_swir_min: float = 1500
    metal_ndvi_max: float = 0.25

    # SAR (Sentinel-1 VV backscatter in dB)
    sar_threshold_db: float = -8

    # Google Open Buildings
    ob_min_confidence: float = 0.5

    # Dynamic World
    dw_built_class: int = 6

    # Vectorization
    vectorize_scale: int = 10
    max_pixels: float = 1e9
    tile_scale: int = 4

    # Morphological cleanup (kernel radius for focalMin/focalMax)
    morph_kernel_radius: int = 1

    # Fetching
    max_per_method_per_tile: int = 2000


@dataclass
class CountryConfig:
    """Configuration for one country."""

    name: str
    iso_code: str
    bounds: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    tile_size_deg: float = 0.25

    farm_transparency_path: Optional[str] = None
    osm_path: Optional[str] = None

    species_filter: list[str] = field(default_factory=lambda: [
        "Chickens", "Turkeys", "Ducks", "Pigs",
    ])

    detection_params: DetectionParams = field(default_factory=DetectionParams)

    @property
    def ft_path(self) -> Optional[Path]:
        if self.farm_transparency_path is None:
            return None
        p = DATA_DIR / self.farm_transparency_path
        return p if p.exists() else None

    @property
    def osm_full_path(self) -> Optional[Path]:
        if self.osm_path is None:
            return None
        p = DATA_DIR / self.osm_path
        return p if p.exists() else None


# ---------------------------------------------------------------------------
# Pre-registered countries
# ---------------------------------------------------------------------------

COUNTRIES: dict[str, CountryConfig] = {
    "thailand": CountryConfig(
        name="Thailand",
        iso_code="TH",
        bounds=(97.3, 5.6, 105.6, 20.5),
        farm_transparency_path="farm_transparency_maps/All facilities in Thailand.csv",
        osm_path="osm_farms/All facilities in Thailand.csv",
    ),
    "united_states": CountryConfig(
        name="United States",
        iso_code="US",
        bounds=(-125.0, 24.5, -66.9, 49.4),
        tile_size_deg=0.5,
        farm_transparency_path="farm_transparency_maps/All facilities in United States.csv",
    ),
    "united_kingdom": CountryConfig(
        name="United Kingdom",
        iso_code="GB",
        bounds=(-8.2, 49.9, 1.8, 60.9),
        farm_transparency_path="farm_transparency_maps/All facilities in United Kingdom.csv",
    ),
    "brazil": CountryConfig(
        name="Brazil",
        iso_code="BR",
        bounds=(-73.9, -33.7, -34.8, 5.3),
        tile_size_deg=0.5,
        osm_path="osm_farms/All facilities in Brazil.csv",
    ),
    "australia": CountryConfig(
        name="Australia",
        iso_code="AU",
        bounds=(113.3, -43.6, 153.6, -10.7),
        tile_size_deg=0.5,
        farm_transparency_path="farm_transparency_maps/All facilities in Australia.csv",
    ),
}


def init_ee() -> None:
    """Initialize Earth Engine using credentials from environment variables.

    Requires GEE_SERVICE_ACCOUNT and GEE_KEY_FILE to be set in .env
    (see .env.example).
    """
    import ee
    from training.env_loader import load_dotenv, get_gee_credentials

    load_dotenv()
    service_account, key_path = get_gee_credentials()
    credentials = ee.ServiceAccountCredentials(service_account, key_path)
    ee.Initialize(credentials)
