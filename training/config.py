"""Pydantic config models -- single YAML file drives the entire pipeline.

Region strings
--------------
A *region* is ``"country_key"`` or ``"country_key/state"``.

* ``"thailand"``          -- every candidate whose country_key is thailand
* ``"united_states/AL"``  -- only candidates with country_key united_states **and** state AL
* ``"united_states"``     -- every candidate in the US regardless of state
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------------

def parse_region(region: str) -> tuple[str, str | None]:
    """Parse ``'country_key'`` or ``'country_key/state'``."""
    parts = region.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else None)


def candidate_matches_region(country_key: str, state: str, region: str) -> bool:
    """Return *True* when a candidate's (country_key, state) matches *region*.

    Country-wide candidates (state is empty/None) match any region for that
    country.  This ensures negatives sampled at the country level are included
    in region-based splits.
    """
    r_country, r_state = parse_region(region)
    if r_country != country_key:
        return False
    # Country-wide candidate (no state) → matches any region in that country
    if not state:
        return True
    if r_state is not None and r_state != state:
        return False
    return True


def matches_any_region(country_key: str, state: str, regions: list[str]) -> bool:
    """Return *True* when a candidate matches **any** region in *regions*."""
    return any(candidate_matches_region(country_key, state, r) for r in regions)


def build_country_key_map() -> dict[str, str]:
    """Return ``{display_name: country_key}`` built from the COUNTRIES registry."""
    from src.config import COUNTRIES
    return {cfg.name: key for key, cfg in COUNTRIES.items()}


def build_region_string(country_key: str, state: str) -> str:
    """Derive a canonical region string for a candidate row."""
    if state:
        return f"{country_key}/{state}"
    return country_key


class BuildingFootprintProviderConfig(BaseModel):
    """Config for a single building footprint data source."""
    name: Literal["google_open_buildings", "ms_buildings", "auto"] = "auto"
    min_area_m2: float = 500
    max_area_m2: float = 50_000
    min_confidence: float = 0.65  # Google Open Buildings only


class BuildingFootprintConfig(BaseModel):
    """Generate candidates from building footprint databases via Earth Engine.

    Buildings near known farms become positives; the rest become negatives.
    This produces harder negatives (real buildings) and more precise positives
    (actual structures, not just point coordinates).
    """
    enabled: bool = False
    provider: BuildingFootprintProviderConfig = Field(
        default_factory=BuildingFootprintProviderConfig
    )
    proximity_radius_m: float = 200  # match building to farm if within this distance
    cache_dir: str = "data/cache/building_footprints"
    max_buildings_per_country: int = 50_000
    tile_size_deg: float = 0.5  # for chunked EE queries


class NegativeSamplingConfig(BaseModel):
    strategy: Literal[
        "random_rural", "hard_negative", "stratified",
        "osm_buildings", "building_footprints",
    ] = "random_rural"
    ratio: float = 1.0
    min_distance_m: float = 2000
    seed: int = 42
    osm_tags: list[str] = Field(
        default_factory=lambda: [
            "warehouse", "industrial", "commercial", "retail",
            "church", "school", "hangar",
        ]
    )
    osm_cache_dir: str = "data/cache/osm_negatives"


class DataConfig(BaseModel):
    countries: list[str] = Field(default_factory=list)
    species_filter: list[str] = Field(default_factory=list)
    categories_include: list[str] = Field(default_factory=list)
    train_regions: Optional[list[str]] = None
    val_regions: Optional[list[str]] = None
    test_regions: Optional[list[str]] = None
    candidates_dir: str = "data/candidates"
    # Optional: load candidates directly from a parquet file instead of
    # generating them from Farm Transparency / OSM / building footprints.
    parquet_source: Optional[str] = None
    # Optional: extra parquet sources to merge with BFD/FTP candidates.
    # Each path should be a parquet file with pre-labeled clusters.
    extra_parquet_sources: list[str] = Field(default_factory=list)
    # Include unlabeled clusters from parquet_source (for inference on all data)
    include_unlabeled: bool = False
    # Force candidates with viz_status=inspected into test set (excluded from train/val).
    inspected_as_test: bool = False
    # Label mode: "binary" (farm=1, not-farm=0) or "poultry" (poultry=1, else=0)
    label_mode: str = "binary"
    # Drop rows with these modified_label values entirely
    exclude_labels: list[str] = Field(default_factory=list)
    # Drop rows where original_label contains "OSM" and row appears to be a farm
    exclude_osm_farms: bool = False
    osm_farm_cache_dir: str = "data/cache/osm_farm_finder"
    osm_farm_tags: list[str] = Field(
        default_factory=lambda: [
            "landuse=farmyard", "building=farm", "building=barn",
            "building=sty", "building=cowshed", "building=farm_auxiliary",
            "building=chicken_coop", "building=stable", "building=hatchery",
            "industrial=livestock", "place=farm",
        ]
    )
    osm_farm_species_keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "Chickens": [
                "chicken", "poultry", "broiler", "layer", "egg", "hen",
                "chicken_coop", "hatchery",
            ],
            "Pigs": ["pig", "swine", "pork", "sow", "boar", "sty"],
            "Cattle": ["cattle", "cow", "beef", "dairy", "buffalo", "cowshed"],
            "Ducks": ["duck"],
            "Turkeys": ["turkey"],
        }
    )
    negative_sampling: NegativeSamplingConfig = Field(
        default_factory=NegativeSamplingConfig
    )
    building_footprints: BuildingFootprintConfig = Field(
        default_factory=BuildingFootprintConfig
    )

    def all_regions(self) -> list[str]:
        """Union of train + val + test regions (empty list if none configured)."""
        out: list[str] = []
        for lst in (self.train_regions, self.val_regions, self.test_regions):
            if lst:
                out.extend(lst)
        return out


class PatchConfig(BaseModel):
    patch_size_px: int = 128
    resolution_m: int = 10
    bands: list[str] = Field(
        default_factory=lambda: ["B2", "B3", "B4", "B8", "B11", "B12"]
    )
    indices: list[str] = Field(default_factory=lambda: ["NDVI", "NDBI", "NDWI"])
    composite: Literal["median", "least_cloudy"] = "median"
    date_range: list[str] = Field(
        default_factory=lambda: ["2023-01-01", "2023-12-31"]
    )
    max_cloud_cover: int = 15
    output_dir: str = "data/patches"
    num_workers: int = 4
    retry_failed: bool = False
    # Optional: list of imagery sources (provider + options). If absent, single EE S2 source is used from bands/indices/date_range above.
    imagery_sources: Optional[list[dict[str, Any]]] = None

    @property
    def n_channels(self) -> int:
        if self.imagery_sources:
            from .imagery import ResolvedSource, get_provider
            total = 0
            for raw in self.imagery_sources:
                provider_name = raw.get("provider", "earth_engine_s2")
                opts = {k: v for k, v in raw.items() if k != "provider"}
                provider_cls = get_provider(provider_name)
                provider = provider_cls(**opts)
                total += len(ResolvedSource(provider).band_names())
            return total
        return len(self.bands) + len(self.indices)

    @property
    def patch_extent_m(self) -> float:
        return self.patch_size_px * self.resolution_m


def resolve_channel_indices(
    channel_subset: list[str],
    all_bands: list[str],
    all_indices: list[str],
) -> tuple[list[int], int]:
    """Convert channel names to positional indices in the .npy array.

    Returns (channel_indices, n_spectral_in_subset).
    Band order in .npy: bands[0..n-1], indices[n..n+m-1].
    Example: ["B2","B3","B4","NDWI"] with bands=[B2..B12], indices=[NDVI,NDBI,NDWI]
             -> ([0, 1, 2, 8], 3)
    """
    indices: list[int] = []
    n_spectral = 0
    for ch in channel_subset:
        if ch in all_bands:
            indices.append(all_bands.index(ch))
            n_spectral += 1
        elif ch in all_indices:
            indices.append(len(all_bands) + all_indices.index(ch))
        else:
            raise ValueError(f"Channel {ch!r} not in bands {all_bands} or indices {all_indices}")
    return indices, n_spectral


def imagery_config_hash(patch_cfg: PatchConfig) -> str:
    """Stable hash of imagery-affecting config so different bands/date_range don't overwrite patches."""
    payload: dict[str, Any] = {}
    if patch_cfg.imagery_sources:
        payload["imagery_sources"] = patch_cfg.imagery_sources
    else:
        payload["bands"] = patch_cfg.bands
        payload["indices"] = patch_cfg.indices
        payload["composite"] = patch_cfg.composite
        payload["max_cloud_cover"] = patch_cfg.max_cloud_cover
    payload["date_range"] = patch_cfg.date_range
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def imagery_metadata(patch_cfg: PatchConfig, band_names: list[str]) -> dict[str, str]:
    """Metadata dict for patch_meta.csv describing the imagery source."""
    if patch_cfg.imagery_sources:
        providers = [
            s.get("provider", "earth_engine_s2")
            for s in patch_cfg.imagery_sources
        ]
        provider = "+".join(providers)
    else:
        provider = "earth_engine_s2"
    bands_str = ",".join(band_names) if band_names else ",".join(patch_cfg.bands + patch_cfg.indices)
    return {
        "bands": bands_str,
        "date_range": ",".join(patch_cfg.date_range),
        "composite": patch_cfg.composite,
        "provider": provider,
    }


class ModelConfig(BaseModel):
    architecture: str = "resnet50"
    hub_name: str = "microsoft/resnet-50"
    pretrained: bool = True
    num_classes: int = 2
    input_channels: int = 9
    freeze_backbone_epochs: int = 3


class AugFlipConfig(BaseModel):
    enabled: bool = True
    probability: float = 0.5


class AugRotation90Config(BaseModel):
    enabled: bool = True
    probability: float = 0.75


class AugContinuousRotationConfig(BaseModel):
    enabled: bool = False
    probability: float = 0.3
    max_degrees: float = 15.0
    fill_mode: Literal["reflect", "zero"] = "reflect"


class AugResizedCropConfig(BaseModel):
    enabled: bool = False
    probability: float = 0.3
    scale_min: float = 0.8
    scale_max: float = 1.0


class AugBrightnessConfig(BaseModel):
    enabled: bool = True
    probability: float = 0.5
    range_min: float = 0.85
    range_max: float = 1.15


class AugPerBandJitterConfig(BaseModel):
    enabled: bool = False
    probability: float = 0.3
    range_min: float = 0.95
    range_max: float = 1.05


class AugGaussianNoiseConfig(BaseModel):
    enabled: bool = False
    probability: float = 0.3
    sigma: float = 0.02


class AugChannelDropoutConfig(BaseModel):
    enabled: bool = False
    probability: float = 0.1
    max_channels: int = 1


class AugCutoutConfig(BaseModel):
    enabled: bool = False
    probability: float = 0.2
    n_holes: int = 1
    hole_size: int = 16


class AugmentationConfig(BaseModel):
    enabled: bool = True
    horizontal_flip: AugFlipConfig = Field(default_factory=AugFlipConfig)
    vertical_flip: AugFlipConfig = Field(default_factory=AugFlipConfig)
    random_rotation_90: AugRotation90Config = Field(default_factory=AugRotation90Config)
    continuous_rotation: AugContinuousRotationConfig = Field(
        default_factory=AugContinuousRotationConfig
    )
    random_resized_crop: AugResizedCropConfig = Field(
        default_factory=AugResizedCropConfig
    )
    brightness_jitter: AugBrightnessConfig = Field(default_factory=AugBrightnessConfig)
    per_band_jitter: AugPerBandJitterConfig = Field(
        default_factory=AugPerBandJitterConfig
    )
    gaussian_noise: AugGaussianNoiseConfig = Field(
        default_factory=AugGaussianNoiseConfig
    )
    channel_dropout: AugChannelDropoutConfig = Field(
        default_factory=AugChannelDropoutConfig
    )
    cutout: AugCutoutConfig = Field(default_factory=AugCutoutConfig)
    recompute_indices: bool = False


class TrainingConfig(BaseModel):
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    scheduler: Literal["cosine", "step", "plateau"] = "cosine"
    early_stopping_patience: int = 5
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    mixed_precision: bool = True
    # Class weights [neg, pos] to penalize false positives when model predicts all positive
    class_weight: Optional[list[float]] = None
    # Upsample minority regions so each country contributes equally per epoch
    upsample_minority_regions: bool = False
    balanced_country_splits: bool = False
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    # Ablation: use only a subset of channels at training time (by band name).
    # None = use all channels from patches. E.g. ["B2","B3","B4","NDWI"] for RGB+NDWI.
    channel_subset: Optional[list[str]] = None
    # Ablation: center-crop patches to this size (pixels) at training time.
    # None = use full patch_size_px. E.g. 64 for 640m context instead of 1.28km.
    crop_center_px: Optional[int] = None


class MLflowConfig(BaseModel):
    tracking_uri: str = "./mlruns"
    experiment_name: str = "farm_detection_v1"
    log_model: bool = True


class RunPodConfig(BaseModel):
    gpu_type: str = "NVIDIA A40"
    gpu_fallbacks: list[str] = []
    cloud_type: Literal["SECURE", "COMMUNITY", "ALL"] = "ALL"
    docker_image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    volume_mount: str = "/workspace"
    code_dir: str = "/workspace/farm-mapping"
    github_repo: str = ""
    github_branch: str = "main"
    api_key_env: str = "RUNPOD_API_KEY"
    network_volume_id: Optional[str] = None
    cpu_instance_id: str = "cpu3g-4-16"
    cpu_fallbacks: list[str] = []
    auto_terminate: bool = True


class CacheLocalConfig(BaseModel):
    base_path: str = "data/cache"


class CacheS3Config(BaseModel):
    bucket: str = ""
    prefix: str = "cache"
    # Credentials via env (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) or default chain


class CacheGCSConfig(BaseModel):
    bucket: str = ""
    prefix: str = "cache"
    # Credentials via env GOOGLE_APPLICATION_CREDENTIALS or default


class CacheRunPodConfig(BaseModel):
    volume_mount: str = "/workspace/data"


class CacheConfig(BaseModel):
    enabled: bool = False
    backend: Literal["local", "s3", "gcs", "runpod"] = "local"
    local: CacheLocalConfig = Field(default_factory=CacheLocalConfig)
    s3: CacheS3Config = Field(default_factory=CacheS3Config)
    gcs: CacheGCSConfig = Field(default_factory=CacheGCSConfig)
    runpod: CacheRunPodConfig = Field(default_factory=CacheRunPodConfig)


class ConfidenceTiers(BaseModel):
    high: float = 0.8
    medium: float = 0.5
    low: float = 0.3


class InferenceConfig(BaseModel):
    checkpoint: str = "output/best_model.pt"
    threshold: float = 0.5
    confidence_tiers: ConfidenceTiers = Field(default_factory=ConfidenceTiers)


class VizConfig(BaseModel):
    output_dir: str = "output/maps"
    show_true_positives: bool = True
    show_false_positives: bool = True
    show_false_negatives: bool = True
    show_true_negatives: bool = False


class PipelineConfig(BaseModel):
    """Root config that aggregates all sections."""

    run_name: str = ""
    _config_stem: str = "default"  # set by load_config from the yaml filename

    data: DataConfig = Field(default_factory=DataConfig)
    patches: PatchConfig = Field(default_factory=PatchConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    runpod: RunPodConfig = Field(default_factory=RunPodConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    visualization: VizConfig = Field(default_factory=VizConfig)


def load_config(yaml_path: str | Path) -> PipelineConfig:
    """Load and validate a YAML config file into a :class:`PipelineConfig`."""
    with open(yaml_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = PipelineConfig.model_validate(raw or {})
    cfg._config_stem = Path(yaml_path).stem
    return cfg


def resolve_paths(cfg: PipelineConfig, root: Optional[Path] = None) -> PipelineConfig:
    """Resolve relative paths in the config against *root* (default: project root)."""
    if root is None:
        root = Path(__file__).resolve().parent.parent

    cfg.patches.output_dir = str((root / cfg.patches.output_dir).resolve())
    cfg.inference.checkpoint = str((root / cfg.inference.checkpoint).resolve())
    cfg.visualization.output_dir = str((root / cfg.visualization.output_dir).resolve())
    cfg.data.negative_sampling.osm_cache_dir = str(
        (root / cfg.data.negative_sampling.osm_cache_dir).resolve()
    )
    cfg.data.osm_farm_cache_dir = str(
        (root / cfg.data.osm_farm_cache_dir).resolve()
    )
    cfg.data.candidates_dir = str(
        (root / cfg.data.candidates_dir).resolve()
    )
    cfg.data.building_footprints.cache_dir = str(
        (root / cfg.data.building_footprints.cache_dir).resolve()
    )
    if cfg.data.parquet_source:
        cfg.data.parquet_source = str((root / cfg.data.parquet_source).resolve())
    cfg.data.extra_parquet_sources = [
        str((root / p).resolve()) for p in cfg.data.extra_parquet_sources
    ]
    if cfg.cache.enabled and cfg.cache.backend == "local":
        cfg.cache.local.base_path = str(
            (root / cfg.cache.local.base_path).resolve()
        )
    return cfg


def cache_key(cfg: PipelineConfig) -> str:
    """Stable cache key from data + patches config (content-affecting fields only)."""
    data_cfg = cfg.data.model_dump()
    patches_cfg = cfg.patches.model_dump()
    for key in ("output_dir", "num_workers"):
        patches_cfg.pop(key, None)
    data = {"data": data_cfg, "patches": patches_cfg}
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
