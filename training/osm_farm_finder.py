"""Find livestock farms in a country via the Overpass API.

Queries OSM for farm-tagged features, classifies each by species using
keyword matching on tags and names, filters by the pipeline config, caches
raw results, and generates an interactive HTML map.

Usage::

    python -m training.osm_farm_finder --config configs/chicken_meat_thailand.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.config import COUNTRIES
from src.visualization import _HTML_TEMPLATE, _legend_item, save_map

from .config import PipelineConfig, load_config, resolve_paths

log = logging.getLogger(__name__)

_MAX_RETRIES = 4
_INITIAL_BACKOFF_S = 30

SPECIES_COLORS: dict[str, str] = {
    "Chickens": "#e41a1c",
    "Pigs": "#377eb8",
    "Cattle": "#4daf4a",
    "Ducks": "#ff7f00",
    "Turkeys": "#984ea3",
    "Unknown": "#999999",
}


# ---------------------------------------------------------------------------
# Species classification
# ---------------------------------------------------------------------------

def _categorize_farm(
    tags: dict[str, str],
    name: str,
    species_keywords: dict[str, list[str]],
) -> tuple[str, str]:
    """Return ``(category, species)`` for an OSM feature.

    *species_keywords* maps species names to keyword lists, e.g.
    ``{"Chickens": ["chicken", "poultry", "ไก่", ...], ...}``.
    """
    combined = (name + " " + " ".join(tags.values())).lower()

    for species, keywords in species_keywords.items():
        if any(kw in combined for kw in keywords):
            return "Farm (meat)", species
    return "Farm", "Unknown"


# ---------------------------------------------------------------------------
# Overpass queries
# ---------------------------------------------------------------------------

def _parse_tag(tag_str: str) -> tuple[str, str]:
    """Parse ``'key=value'`` into ``(key, value)``."""
    key, _, value = tag_str.partition("=")
    return key.strip(), value.strip()


def _build_overpass_query(iso_code: str, tags: list[str]) -> str:
    """Build an Overpass QL query using the ISO country area."""
    tag_lines: list[str] = []
    for tag_str in tags:
        key, value = _parse_tag(tag_str)
        tag_lines.append(f'  node["{key}"="{value}"](area.country);')
        tag_lines.append(f'  way["{key}"="{value}"](area.country);')

    body = "\n".join(tag_lines)
    return (
        f'[out:json][timeout:300];\n'
        f'area["ISO3166-1"="{iso_code}"][admin_level=2]->.country;\n'
        f'(\n{body}\n);\n'
        f'out center;'
    )


def _run_overpass(query: str) -> object | None:
    """Execute an Overpass query with retry."""
    import overpy

    api = overpy.Overpass()
    for attempt in range(_MAX_RETRIES):
        try:
            return api.query(query)
        except (overpy.exception.OverpassGatewayTimeout,
                overpy.exception.OverpassTooManyRequests):
            wait = _INITIAL_BACKOFF_S * (2 ** attempt)
            log.warning("Overpass timeout/busy (attempt %d/%d), retrying in %ds ...",
                        attempt + 1, _MAX_RETRIES, wait)
            time.sleep(wait)
    log.error("Overpass failed after %d retries", _MAX_RETRIES)
    return None


def _parse_results(result: object) -> list[dict]:
    """Convert Overpass result (nodes + ways with center) into row dicts."""
    import overpy

    rows: list[dict] = []
    seen: set[str] = set()

    if not isinstance(result, overpy.Result):
        return rows

    for node in result.nodes:
        uid = f"node/{node.id}"
        if uid in seen:
            continue
        seen.add(uid)
        tags = dict(node.tags)
        name = tags.get("name", "")
        rows.append({
            "osm_id": uid,
            "lat": float(node.lat),
            "lng": float(node.lon),
            "name": name,
            "tags": json.dumps(tags, ensure_ascii=False),
        })

    for way in result.ways:
        uid = f"way/{way.id}"
        if uid in seen:
            continue
        seen.add(uid)
        if way.center_lat is None or way.center_lon is None:
            continue
        tags = dict(way.tags)
        name = tags.get("name", "")
        rows.append({
            "osm_id": uid,
            "lat": float(way.center_lat),
            "lng": float(way.center_lon),
            "name": name,
            "tags": json.dumps(tags, ensure_ascii=False),
        })

    return rows


def _classify_rows(
    rows: list[dict], species_keywords: dict[str, list[str]],
) -> list[dict]:
    """Add ``category`` and ``species`` columns to raw rows."""
    for row in rows:
        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else row["tags"]
        cat, species = _categorize_farm(tags, row.get("name", ""), species_keywords)
        row["category"] = cat
        row["species"] = species
    return rows


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------

def _build_species_layer_js(species: str, features: list[dict], color: str) -> str:
    """JS code for one species layer."""
    fc = {"type": "FeatureCollection", "features": features}
    fc_json = json.dumps(fc, ensure_ascii=False)
    safe = species.replace("'", "\\'")
    n = len(features)
    return f"""
    (function() {{
        var layer = L.geoJSON({fc_json}, {{
            pointToLayer: function(f, ll) {{
                return L.circleMarker(ll, {{radius: 6, fillColor: '{color}', color: '{color}',
                                            weight: 2, fillOpacity: 0.7}});
            }},
            onEachFeature: function(f, layer) {{
                var p = f.properties, html = '<b>{safe}</b><br>';
                if(p.name) html += 'Name: ' + p.name + '<br>';
                if(p.category) html += 'Category: ' + p.category + '<br>';
                if(p.osm_id) html += 'OSM: ' + p.osm_id + '<br>';
                layer.bindPopup(html);
            }}
        }}).addTo(map);
        overlays['{safe} ({n})'] = layer;
    }})();
"""


def _safe_val(v):
    """JSON-safe conversion."""
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _generate_farm_map(
    gdf: gpd.GeoDataFrame,
    title: str,
    center: tuple[float, float],
    output_path: Path,
) -> Path:
    """Generate and save an interactive HTML map colour-coded by species."""
    layers_js = ""
    legend_html = ""

    for species in sorted(gdf["species"].unique()):
        subset = gdf[gdf["species"] == species]
        color = SPECIES_COLORS.get(species, "#888888")
        features = []
        for _, row in subset.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["lng"], row["lat"]]},
                "properties": {
                    k: _safe_val(v) for k, v in row.items()
                    if k not in ("geometry",)
                },
            })
        layers_js += _build_species_layer_js(species, features, color)
        legend_html += _legend_item(species, color, len(features))

    html = _HTML_TEMPLATE.format(
        title=title,
        center_lat=center[0],
        center_lon=center[1],
        zoom=7,
        layers_js=layers_js,
        legend_html=legend_html,
    )
    return save_map(html, output_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _load_or_query(
    country_key: str,
    iso_code: str,
    tags: list[str],
    species_keywords: dict[str, list[str]],
    cache_dir: Path,
) -> pd.DataFrame:
    """Return raw OSM farm rows, using cache when available.

    Cached files store the raw Overpass results but species classification is
    **re-applied** on every load so that keyword changes take effect without
    requiring a fresh Overpass query.
    """
    cache_path = cache_dir / f"{country_key}_raw.parquet"

    if cache_path.exists():
        log.info("Loading cached raw results from %s", cache_path)
        df = pd.read_parquet(cache_path)
        rows = df.to_dict("records")
        rows = _classify_rows(rows, species_keywords)
        return pd.DataFrame(rows)

    log.info("Querying Overpass for farms in %s (ISO=%s) ...", country_key, iso_code)
    query = _build_overpass_query(iso_code, tags)
    result = _run_overpass(query)
    if result is None:
        return pd.DataFrame(columns=["osm_id", "lat", "lng", "name", "tags"])

    rows = _parse_results(result)
    log.info("  %d raw features returned", len(rows))
    rows = _classify_rows(rows, species_keywords)

    df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("  Cached raw results -> %s", cache_path)
    return df


def find_farms(cfg: PipelineConfig) -> gpd.GeoDataFrame:
    """Query OSM for farms, filter, generate map, and return filtered GDF."""
    cache_dir = Path(cfg.data.osm_farm_cache_dir)
    all_frames: list[pd.DataFrame] = []

    for country_key in cfg.data.countries:
        country_cfg = COUNTRIES.get(country_key)
        if country_cfg is None:
            log.warning("Country '%s' not in COUNTRIES registry -- skipping", country_key)
            continue

        df = _load_or_query(
            country_key, country_cfg.iso_code,
            cfg.data.osm_farm_tags, cfg.data.osm_farm_species_keywords,
            cache_dir,
        )
        if len(df) == 0:
            continue
        df["country"] = country_cfg.name
        df["country_key"] = country_key
        all_frames.append(df)

    if not all_frames:
        log.warning("No OSM farm results found for any configured country")
        return gpd.GeoDataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    log.info("Total raw OSM farms (all countries): %d", len(combined))
    _log_species_summary(combined)

    filtered = _apply_filters(combined, cfg)
    log.info("After filtering: %d farms", len(filtered))

    geometry = [Point(lng, lat) for lng, lat in zip(filtered["lng"], filtered["lat"])]
    gdf = gpd.GeoDataFrame(filtered, geometry=geometry, crs="EPSG:4326")

    _save_outputs(gdf, cfg, cache_dir)
    return gdf


def _log_species_summary(df: pd.DataFrame) -> None:
    for species, count in df["species"].value_counts().items():
        log.info("  %s: %d", species, count)


def _apply_filters(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    if cfg.data.species_filter:
        df = df[df["species"].isin(cfg.data.species_filter)]
    if cfg.data.categories_include:
        mask = pd.Series(False, index=df.index)
        for substr in cfg.data.categories_include:
            mask |= df["category"].fillna("").str.contains(
                substr, case=False, na=False, regex=False,
            )
        df = df[mask]
    return df.reset_index(drop=True)


def _save_outputs(
    gdf: gpd.GeoDataFrame, cfg: PipelineConfig, cache_dir: Path,
) -> None:
    species_slug = "_".join(s.lower() for s in cfg.data.species_filter) or "all"
    country_slug = "_".join(cfg.data.countries)

    parquet_path = cache_dir / f"{country_slug}_{species_slug}_farms.parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(parquet_path, index=False)
    log.info("Saved %d filtered farms -> %s", len(gdf), parquet_path)

    map_path = cache_dir / f"{country_slug}_{species_slug}_farms_map.html"
    center_lat = float(gdf["lat"].mean()) if len(gdf) > 0 else 0.0
    center_lon = float(gdf["lng"].mean()) if len(gdf) > 0 else 0.0
    title = f"OSM Farms: {country_slug} ({species_slug})"
    _generate_farm_map(gdf, title, (center_lat, center_lon), map_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find farms via OSM Overpass API")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    find_farms(cfg)


if __name__ == "__main__":
    main()
