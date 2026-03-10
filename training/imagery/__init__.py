"""Imagery providers for patch extraction.

Each provider can build an Earth Engine image (or later, numpy array) for a
region with provider-specific options. Multiple sources are stacked along the
channel axis during extraction.
"""

from .base import ImageryProvider, ResolvedSource
from .earth_engine_s2 import EarthEngineSentinel2Provider
from .earth_engine_s1 import EarthEngineSentinel1Provider

PROVIDER_REGISTRY: dict[str, type[ImageryProvider]] = {
    "earth_engine_s2": EarthEngineSentinel2Provider,
    "earth_engine_s1": EarthEngineSentinel1Provider,
}


def resolve_imagery_sources(patch_cfg: "PatchConfig") -> list[ResolvedSource]:
    """Convert PatchConfig to a list of ResolvedSource for extraction.

    If imagery_sources is set, one ResolvedSource per entry. Otherwise a
    single EE S2 source from bands/indices/date_range/max_cloud_cover.
    """
    from ..config import PatchConfig
    if patch_cfg.imagery_sources:
        out: list[ResolvedSource] = []
        for raw in patch_cfg.imagery_sources:
            provider_name = raw.get("provider", "earth_engine_s2")
            opts = {k: v for k, v in raw.items() if k != "provider"}
            provider_cls = get_provider(provider_name)
            provider = provider_cls(**opts)
            out.append(ResolvedSource(provider))
        return out
    # Legacy: single EE S2 from top-level config
    provider = EarthEngineSentinel2Provider(
        bands=patch_cfg.bands,
        indices=patch_cfg.indices,
        max_cloud_cover=patch_cfg.max_cloud_cover,
        composite=patch_cfg.composite,
    )
    return [ResolvedSource(provider)]


def get_provider(name: str) -> type[ImageryProvider]:
    """Return the provider class for the given name."""
    if name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown imagery provider: {name}. Known: {list(PROVIDER_REGISTRY)}"
        )
    return PROVIDER_REGISTRY[name]
