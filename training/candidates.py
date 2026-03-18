"""Build a labelled candidate set: positives from existing data sources + negatives.

Every candidate row carries a ``region`` column (e.g. ``"united_states/AL"`` or
``"thailand"``) so that downstream splitting can assign rows to train / val / test
by geographic region rather than random sampling.

Usage::

    python -m training.candidates --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.config import COUNTRIES
from src.data_sources import load_known_farms

from .config import (
    DataConfig,
    PipelineConfig,
    build_country_key_map,
    build_region_string,
    load_config,
    matches_any_region,
    resolve_paths,
)

log = logging.getLogger(__name__)

_EMPTY_COLUMNS = [
    "id", "lat", "lng", "label", "source", "country", "state", "region", "geometry",
]


def _empty_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns=_EMPTY_COLUMNS, geometry="geometry", crs="EPSG:4326")


def _rows_to_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    if not rows:
        return _empty_gdf()
    df = pd.DataFrame(rows)
    geometry = [Point(r["lng"], r["lat"]) for r in rows]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def _is_far_from_farms(
    lon: float, lat: float, pos_coords: np.ndarray, min_dist_m: float
) -> bool:
    """Return True if (lon, lat) is at least *min_dist_m* from all farms."""
    if len(pos_coords) == 0:
        return True
    dlat = (pos_coords[:, 1] - lat) * 111_000
    dlon = (pos_coords[:, 0] - lon) * 111_000 * np.cos(np.radians(lat))
    return float(np.min(np.sqrt(dlat**2 + dlon**2))) >= min_dist_m


def _make_neg_row(
    prefix: str, country_key: str, idx: int,
    lat: float, lon: float, country_name: str,
    source: str, state: str = "",
) -> dict:
    return {
        "id": f"{prefix}_{country_key}_{idx}",
        "lat": lat, "lng": lon, "label": 0,
        "source": source, "country": country_name,
        "state": state,
        "region": build_region_string(country_key, state),
        "species": "", "name": "", "category": "",
    }


def _get_country_pos_coords(
    positives: gpd.GeoDataFrame, country_name: str,
) -> np.ndarray:
    country_pos = positives[positives["country"] == country_name]
    if len(country_pos) == 0:
        return np.empty((0, 2))
    return np.column_stack([country_pos.geometry.x, country_pos.geometry.y])


# ---------------------------------------------------------------------------
# Region tagging
# ---------------------------------------------------------------------------

def _add_region_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Derive a ``region`` column from (country, state) using the COUNTRIES map."""
    name_to_key = build_country_key_map()
    keys = gdf["country"].map(name_to_key).fillna("")
    states = gdf["state"].fillna("").astype(str)
    gdf["region"] = [
        build_region_string(k, s) for k, s in zip(keys, states)
    ]
    return gdf


def _filter_by_regions(
    gdf: gpd.GeoDataFrame, regions: list[str],
) -> gpd.GeoDataFrame:
    """Keep only rows whose region matches at least one entry in *regions*."""
    name_to_key = build_country_key_map()
    keys = gdf["country"].map(name_to_key).fillna("").astype(str)
    states = gdf["state"].fillna("").astype(str)
    mask = [
        matches_any_region(k, s, regions) for k, s in zip(keys, states)
    ]
    return gdf[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Positive candidates
# ---------------------------------------------------------------------------

def _positive_candidates(cfg: DataConfig) -> gpd.GeoDataFrame:
    """Load known farms for all configured countries and label them positive."""
    frames: list[gpd.GeoDataFrame] = []
    for country_key in cfg.countries:
        country_cfg = COUNTRIES.get(country_key)
        if country_cfg is None:
            log.warning("Country '%s' not in COUNTRIES registry -- skipping", country_key)
            continue
        gdf = load_known_farms(country_cfg, categories_include=cfg.categories_include)
        if len(gdf) > 0:
            frames.append(gdf)
            log.info("  %s: %d positive candidates", country_key, len(gdf))

    if not frames:
        return _empty_gdf()

    positives = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    positives["label"] = 1

    if "state" not in positives.columns:
        positives["state"] = ""
    positives = _add_region_column(positives)

    all_regions = cfg.all_regions()
    if all_regions:
        before = len(positives)
        positives = _filter_by_regions(positives, all_regions)
        log.info("  Filtered positives to configured regions: %d -> %d", before, len(positives))

    return positives


# ---------------------------------------------------------------------------
# Negative strategies
# ---------------------------------------------------------------------------

def _sample_rural_for_country(
    country_key: str, positives: gpd.GeoDataFrame,
    cfg: DataConfig, per_country: int, rng: np.random.Generator,
) -> list[dict]:
    """Sample random rural negatives for a single country."""
    country_cfg = COUNTRIES.get(country_key)
    if country_cfg is None:
        return []

    min_lon, min_lat, max_lon, max_lat = country_cfg.bounds
    pos_coords = _get_country_pos_coords(positives, country_cfg.name)
    min_d = cfg.negative_sampling.min_distance_m

    rows: list[dict] = []
    max_attempts = per_country * 20
    attempts = 0
    while len(rows) < per_country and attempts < max_attempts:
        batch = min(per_country * 4, max_attempts - attempts)
        lons = rng.uniform(min_lon, max_lon, size=batch)
        lats = rng.uniform(min_lat, max_lat, size=batch)
        for lon, lat in zip(lons, lats):
            if not _is_far_from_farms(lon, lat, pos_coords, min_d):
                continue
            rows.append(_make_neg_row("neg", country_key, len(rows), lat, lon,
                                      country_cfg.name, "random_rural"))
            if len(rows) >= per_country:
                break
        attempts += batch

    log.info("  %s: %d negative candidates (random rural)", country_key, len(rows))
    return rows


def _random_rural_negatives(
    positives: gpd.GeoDataFrame,
    cfg: DataConfig,
    n_negatives: int,
    rng: np.random.Generator,
) -> gpd.GeoDataFrame:
    """Sample random points within country bounds, far from known farms."""
    per_country = max(1, n_negatives // len(cfg.countries))
    rows: list[dict] = []
    for country_key in cfg.countries:
        rows.extend(_sample_rural_for_country(
            country_key, positives, cfg, per_country, rng))
    return _rows_to_gdf(rows)


def _offset_point(
    origin: np.ndarray, rng: np.random.Generator, min_dist_m: float,
) -> tuple[float, float]:
    """Offset a point by a random bearing and distance."""
    bearing = rng.uniform(0, 2 * np.pi)
    dist_m = rng.uniform(min_dist_m, min_dist_m * 3)
    dlat = dist_m * np.cos(bearing) / 111_000
    dlon = dist_m * np.sin(bearing) / (111_000 * np.cos(np.radians(origin[1])))
    return float(origin[0] + dlon), float(origin[1] + dlat)


def _sample_hard_for_country(
    country_key: str, positives: gpd.GeoDataFrame,
    cfg: DataConfig, per_country: int, rng: np.random.Generator,
) -> list[dict]:
    """Sample hard negatives for a single country by offsetting known farms."""
    country_cfg = COUNTRIES.get(country_key)
    if country_cfg is None:
        return []

    pos_coords = _get_country_pos_coords(positives, country_cfg.name)
    if len(pos_coords) == 0:
        return []

    min_lon, min_lat, max_lon, max_lat = country_cfg.bounds
    n_to_sample = min(per_country, len(pos_coords))
    indices = rng.choice(len(pos_coords), size=n_to_sample, replace=True)

    rows: list[dict] = []
    for idx_i, idx in enumerate(indices):
        lon, lat = _offset_point(pos_coords[idx], rng, cfg.negative_sampling.min_distance_m)
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            continue
        rows.append(_make_neg_row("hn", country_key, idx_i, lat, lon,
                                  country_cfg.name, "hard_negative"))

    log.info("  %s: %d negative candidates (hard negatives)", country_key, len(rows))
    return rows


def _hard_negatives(
    positives: gpd.GeoDataFrame,
    cfg: DataConfig,
    n_negatives: int,
    rng: np.random.Generator,
) -> gpd.GeoDataFrame:
    """Sample negatives by offsetting known farm locations."""
    per_country = max(1, n_negatives // len(cfg.countries))
    rows: list[dict] = []
    for country_key in cfg.countries:
        rows.extend(_sample_hard_for_country(
            country_key, positives, cfg, per_country, rng))
    return _rows_to_gdf(rows)


def _osm_building_negatives(
    positives: gpd.GeoDataFrame,
    cfg: DataConfig,
    n_negatives: int,
    _rng: np.random.Generator,
) -> gpd.GeoDataFrame:
    """Fetch non-farm buildings from OSM that visually resemble farm structures."""
    from .osm_negatives import fetch_osm_negatives

    all_regions = cfg.all_regions()
    if not all_regions:
        all_regions = [k for k in cfg.countries]

    pos_coords = np.column_stack([positives.geometry.x, positives.geometry.y]) if len(positives) > 0 else np.empty((0, 2))

    return fetch_osm_negatives(
        regions=all_regions,
        osm_tags=cfg.negative_sampling.osm_tags,
        max_total=n_negatives,
        min_distance_m=cfg.negative_sampling.min_distance_m,
        pos_coords=pos_coords,
        cache_dir=cfg.negative_sampling.osm_cache_dir,
    )


NEGATIVE_STRATEGIES = {
    "random_rural": _random_rural_negatives,
    "hard_negative": _hard_negatives,
    "osm_buildings": _osm_building_negatives,
}


def _building_footprint_candidates(
    positives: gpd.GeoDataFrame,
    cfg: DataConfig,
    n_negatives: int,
    _rng: np.random.Generator,
) -> gpd.GeoDataFrame:
    """Generate both positives and negatives from building footprint databases.

    ALL candidates use BFD building centroids as their coordinates.
    FTP/OSM data is used ONLY for labelling (which buildings are farms).
    This ensures the model can't learn to distinguish the coordinate source.
    """
    from .building_footprints import fetch_building_candidates

    bf_candidates = fetch_building_candidates(cfg, positives)
    if len(bf_candidates) == 0:
        log.warning("No building footprint candidates found, falling back to random_rural")
        return _random_rural_negatives(positives, cfg, n_negatives, _rng)

    # Balance: keep all positives, downsample negatives to match ratio
    bf_pos = bf_candidates[bf_candidates["label"] == 1]
    bf_neg = bf_candidates[bf_candidates["label"] == 0]

    # Target: ratio * n_positives negatives (based on BFD positives, not FTP)
    target_neg = int(len(bf_pos) * cfg.negative_sampling.ratio) if len(bf_pos) > 0 else n_negatives
    if len(bf_neg) > target_neg:
        bf_neg = bf_neg.sample(n=target_neg, random_state=cfg.negative_sampling.seed)

    log.info(
        "BFD balanced: %d positives, %d negatives (ratio=%.1f)",
        len(bf_pos), len(bf_neg), cfg.negative_sampling.ratio,
    )

    return gpd.GeoDataFrame(
        pd.concat([bf_pos, bf_neg], ignore_index=True), crs="EPSG:4326"
    )


def build_candidates(cfg: PipelineConfig) -> gpd.GeoDataFrame:
    """Build a combined candidate GeoDataFrame (positives + negatives).

    Returns a GeoDataFrame with columns including ``label`` (1 = positive,
    0 = negative) and ``region``.
    """
    log.info("Building positive candidates ...")
    positives = _positive_candidates(cfg.data)
    n_pos = len(positives)
    log.info("Total positives: %d", n_pos)

    n_neg = int(n_pos * cfg.data.negative_sampling.ratio)
    rng = np.random.default_rng(cfg.data.negative_sampling.seed)

    strategy = cfg.data.negative_sampling.strategy

    if strategy == "building_footprints":
        # ALL candidates come from building footprint centroids.
        # FTP/OSM positives are used ONLY as labels, never as patch centres.
        # This prevents the model from learning source-specific centering
        # differences instead of actual farm features.
        bf_candidates = _building_footprint_candidates(positives, cfg.data, n_neg, rng)
        n_bf_pos = int((bf_candidates["label"] == 1).sum())
        n_bf_neg = int((bf_candidates["label"] == 0).sum())
        log.info("Building footprint candidates: %d positive, %d negative (all BFD-centred)", n_bf_pos, n_bf_neg)
        combined = bf_candidates

    elif strategy == "stratified":
        n_half = n_neg // 2
        neg_rural = _random_rural_negatives(positives, cfg.data, n_half, rng)
        neg_hard = _hard_negatives(positives, cfg.data, n_neg - n_half, rng)
        negatives = gpd.GeoDataFrame(
            pd.concat([neg_rural, neg_hard], ignore_index=True), crs="EPSG:4326"
        )
        combined = gpd.GeoDataFrame(
            pd.concat([positives, negatives], ignore_index=True), crs="EPSG:4326"
        )
    else:
        neg_fn = NEGATIVE_STRATEGIES.get(strategy, _random_rural_negatives)
        negatives = neg_fn(positives, cfg.data, n_neg, rng)
        log.info("Total negatives: %d", len(negatives))
        combined = gpd.GeoDataFrame(
            pd.concat([positives, negatives], ignore_index=True), crs="EPSG:4326"
        )

    if "state" not in combined.columns:
        combined["state"] = ""
    if "region" not in combined.columns:
        combined = _add_region_column(combined)

    return combined


def save_candidates(
    gdf: gpd.GeoDataFrame,
    candidates_dir: str | Path,
    countries: list[str],
) -> list[Path]:
    """Persist candidate set as human-readable CSVs (one per country).

    Saves to ``{candidates_dir}/{country_key}.csv``.
    """
    out = Path(candidates_dir)
    out.mkdir(parents=True, exist_ok=True)

    name_to_key = build_country_key_map()
    gdf_keys = gdf["country"].map(name_to_key).fillna("")

    saved: list[Path] = []
    for country_key in countries:
        subset = gdf[gdf_keys == country_key]
        if len(subset) == 0:
            continue
        csv_cols = [c for c in subset.columns if c != "geometry"]
        path = out / f"{country_key}.csv"
        subset[csv_cols].to_csv(path, index=False)
        log.info("Saved %d candidates to %s", len(subset), path)
        saved.append(path)

    return saved


def main() -> None:
    from .env_loader import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build candidate set")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cfg = resolve_paths(load_config(args.config))
    candidates = build_candidates(cfg)
    save_candidates(candidates, cfg.data.candidates_dir, cfg.data.countries)

    from src.data_sources import generate_all_farms_csv
    generate_all_farms_csv()


if __name__ == "__main__":
    main()
