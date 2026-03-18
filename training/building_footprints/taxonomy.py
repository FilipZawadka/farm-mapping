"""Unified label taxonomy for building classification.

Maps labels from different sources (Farm Transparency Project, OpenStreetMap)
to a canonical set so that the same real-world category isn't counted twice
when a building appears in multiple databases.

The taxonomy is hierarchical: each label has a ``group`` (broad category) and
a ``label`` (specific type).  This supports both binary classification
(farm vs non-farm) and future multi-class (farm type, building type).

Usage::

    from training.building_footprints.taxonomy import unify_label

    unified = unify_label(source="farm_transparency", raw_category="Farm (eggs)", raw_species="Chickens")
    # UnifiedLabel(group="farm", label="farm_eggs", species="chickens", is_farm=True)

    unified = unify_label(source="osm", raw_category="building=warehouse")
    # UnifiedLabel(group="industrial", label="warehouse", species="", is_farm=False)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnifiedLabel:
    """Canonical label for a building."""
    group: str          # broad: "farm", "industrial", "commercial", "residential", "agricultural", "unknown"
    label: str          # specific: "farm_meat", "warehouse", "school", etc.
    species: str = ""   # normalised species (lowercase), empty for non-farm
    is_farm: bool = False


# ---------------------------------------------------------------------------
# Farm Transparency Project mappings
# ---------------------------------------------------------------------------

_FTP_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    # (group, label)
    "farm (meat)": ("farm", "farm_meat"),
    "farm (dairy)": ("farm", "farm_dairy"),
    "farm (eggs)": ("farm", "farm_eggs"),
    "farm (wool)": ("farm", "farm_wool"),
    "farm (skins/fur)": ("farm", "farm_fur"),
    "farm": ("farm", "farm_generic"),
    "hatchery": ("farm", "hatchery"),
    "slaughterhouse": ("processing", "slaughterhouse"),
    "rendering plant": ("processing", "rendering_plant"),
    "saleyard": ("processing", "saleyard"),
    "depot / holding yard": ("processing", "holding_yard"),
    "live market": ("processing", "live_market"),
    "experimentation": ("other", "experimentation"),
    "zoo": ("other", "zoo"),
    "rodeo": ("other", "rodeo"),
    "wildlife": ("other", "wildlife"),
    "race training/breeding": ("other", "racing"),
}

# ---------------------------------------------------------------------------
# OSM tag mappings
# ---------------------------------------------------------------------------

_OSM_TAG_MAP: dict[str, tuple[str, str, bool]] = {
    # (group, label, is_farm)
    # Farm-related
    "building=farm": ("farm", "farm_generic", True),
    "building=barn": ("farm", "barn", True),
    "building=sty": ("farm", "farm_meat", True),
    "building=cowshed": ("farm", "farm_dairy", True),
    "building=farm_auxiliary": ("farm", "farm_auxiliary", True),
    "building=chicken_coop": ("farm", "farm_eggs", True),
    "building=stable": ("farm", "stable", True),
    "building=hatchery": ("farm", "hatchery", True),
    "building=greenhouse": ("agricultural", "greenhouse", False),
    "landuse=farmyard": ("farm", "farm_generic", True),
    "landuse=farmland": ("agricultural", "farmland", False),
    "industrial=livestock": ("farm", "farm_generic", True),
    "place=farm": ("farm", "farm_generic", True),
    # Industrial / commercial (common negatives)
    "building=warehouse": ("industrial", "warehouse", False),
    "building=industrial": ("industrial", "industrial", False),
    "building=commercial": ("commercial", "commercial", False),
    "building=retail": ("commercial", "retail", False),
    "building=office": ("commercial", "office", False),
    "building=church": ("institutional", "church", False),
    "building=school": ("institutional", "school", False),
    "building=hospital": ("institutional", "hospital", False),
    "building=hangar": ("industrial", "hangar", False),
    "building=garage": ("residential", "garage", False),
    "building=house": ("residential", "house", False),
    "building=residential": ("residential", "residential", False),
    "building=apartments": ("residential", "apartments", False),
    "building=shed": ("agricultural", "shed", False),
    "building=yes": ("unknown", "building_unclassified", False),
}

# ---------------------------------------------------------------------------
# Species normalisation
# ---------------------------------------------------------------------------

_SPECIES_ALIASES: dict[str, str] = {
    "cows/cattle": "cattle",
    "cows/cattle (unconfirmed)": "cattle",
    "chickens": "chickens",
    "chickens (unconfirmed)": "chickens",
    "pigs": "pigs",
    "pigs (unconfirmed)": "pigs",
    "turkeys": "turkeys",
    "ducks": "ducks",
    "sheep": "sheep",
    "goats": "goats",
    "minks": "minks",
    "rabbits": "rabbits",
    "horses": "horses",
    "fish/sealife": "fish",
    "alligators": "reptiles",
    "crocodiles": "reptiles",
    "pigeons": "pigeons",
    "dogs": "dogs",
    "deer": "deer",
    "greyhounds": "dogs",
}


def normalise_species(raw: str) -> str:
    """Normalise species string to canonical lowercase form."""
    if not raw:
        return ""
    return _SPECIES_ALIASES.get(raw.strip().lower(), raw.strip().lower())


def _match_ftp_category(raw_category: str) -> tuple[str, str] | None:
    """Match a Farm Transparency category string to (group, label)."""
    cat = raw_category.strip().lower()
    # Remove "(unconfirmed)" suffix
    cat = re.sub(r"\s*\(unconfirmed\)\s*$", "", cat)
    # Handle composite categories like "Farm (meat), Hatchery"
    # or "Live market, Slaughterhouse" — use the first part
    if "," in cat:
        cat = cat.split(",")[0].strip()
    return _FTP_CATEGORY_MAP.get(cat)


def unify_label(
    source: str,
    raw_category: str = "",
    raw_species: str = "",
    osm_tags: dict[str, str] | None = None,
) -> UnifiedLabel:
    """Map a raw label from any source to the unified taxonomy.

    Args:
        source: One of "farm_transparency", "osm", "osm_buildings", or
            a building footprint provider name.
        raw_category: Category string from the source (FTP category or
            OSM tag like "building=warehouse").
        raw_species: Species string (FTP or inferred).
        osm_tags: Dict of OSM tags for the building (optional, for richer
            classification of OSM features).

    Returns:
        UnifiedLabel with canonical group/label/species.
    """
    species = normalise_species(raw_species)

    # Try Farm Transparency mapping
    if source == "farm_transparency" or source == "FarmTransparency":
        match = _match_ftp_category(raw_category)
        if match:
            group, label = match
            is_farm = group == "farm"
            return UnifiedLabel(group=group, label=label, species=species, is_farm=is_farm)

    # Try OSM tag mapping
    if osm_tags:
        for key, value in osm_tags.items():
            tag_str = f"{key}={value}"
            if tag_str in _OSM_TAG_MAP:
                group, label, is_farm = _OSM_TAG_MAP[tag_str]
                return UnifiedLabel(group=group, label=label, species=species, is_farm=is_farm)

    # Try raw_category as an OSM tag string (e.g. "building=warehouse")
    if raw_category and "=" in raw_category:
        tag_str = raw_category.strip().lower()
        if tag_str in _OSM_TAG_MAP:
            group, label, is_farm = _OSM_TAG_MAP[tag_str]
            return UnifiedLabel(group=group, label=label, species=species, is_farm=is_farm)

    # Fallback: if we have a species, it's probably a farm
    if species:
        return UnifiedLabel(group="farm", label="farm_generic", species=species, is_farm=True)

    return UnifiedLabel(group="unknown", label="unknown", species="", is_farm=False)


def unify_labels_batch(
    categories: list[str],
    species_list: list[str],
    sources: list[str],
) -> list[UnifiedLabel]:
    """Batch version of unify_label for DataFrames."""
    return [
        unify_label(source=src, raw_category=cat, raw_species=sp)
        for cat, sp, src in zip(categories, species_list, sources)
    ]
