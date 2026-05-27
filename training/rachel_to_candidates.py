"""Convert Rachel's cluster parquet into candidate CSVs for the training pipeline.

Reads selected_clusters_relabeled.parquet, filters to labelled clusters,
computes centroids, infers US states, and writes per-country candidate CSVs.

Usage::

    python -m training.rachel_to_candidates --config configs/rachel_clusters.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from .config import load_config, resolve_paths, build_region_string
from .osm_negatives import US_STATE_BOUNDS

log = logging.getLogger(__name__)

# Map ADM0 codes to our country keys
_ADM0_TO_KEY = {
    "USA": "united_states",
    "BRA": "brazil",
    "MEX": "mexico",
    "THA": "thailand",
    "CHL": "chile",
    "ARG": "argentina",
    "CAN": "canada",
    "GBR": "united_kingdom",
    "AUS": "australia",
    "DEU": "germany",
    "ZAF": "south_africa",
}

_KEY_TO_NAME = {
    "united_states": "United States",
    "brazil": "Brazil",
    "mexico": "Mexico",
    "thailand": "Thailand",
    "chile": "Chile",
    "argentina": "Argentina",
    "canada": "Canada",
    "united_kingdom": "United Kingdom",
    "australia": "Australia",
    "germany": "Germany",
    "south_africa": "South Africa",
}


def _infer_us_state(lat: float, lng: float) -> str:
    """Infer US state code from coordinates."""
    for state_code, (min_lon, min_lat, max_lon, max_lat) in US_STATE_BOUNDS.items():
        if min_lon <= lng <= max_lon and min_lat <= lat <= max_lat:
            return state_code
    return ""


def convert(
    parquet_path: str | Path,
    candidates_dir: str | Path,
    include_unlabeled: bool = False,
    label_mode: str = "binary",
    exclude_labels: list[str] | None = None,
    exclude_osm_farms: bool = False,
    inspected_only: bool = False,
) -> pd.DataFrame:
    """Convert parquet to candidate CSVs.

    Args:
        parquet_path: Path to the Rachel clusters parquet.
        candidates_dir: Output directory for per-country CSVs.
        include_unlabeled: If True, keep unlabeled clusters (label=-1).
        label_mode: "binary" (farm=1, not-farm=0) or "poultry" (poultry=1, else=0).
        exclude_labels: List of label values to drop entirely.
        exclude_osm_farms: If True, drop rows where original_label contains "OSM"
            and the row is tagged as a farm (via standardized_label or OSM farm tags).
        inspected_only: If True, keep only rows with viz_status == "inspected".

    Returns the full DataFrame for inspection.
    """
    parquet_path = Path(parquet_path)
    candidates_dir = Path(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("Total clusters: %d", len(df))

    # Pick label column: new files use "final_label", old files use "modified_label"
    label_col = "final_label" if "final_label" in df.columns else "modified_label"
    log.info("Using label column: %s", label_col)

    if inspected_only:
        if "viz_status" not in df.columns:
            raise ValueError("inspected_only=True but viz_status column missing")
        before = len(df)
        df = df[df["viz_status"] == "inspected"].copy()
        log.info("Inspected only: %d -> %d", before, len(df))

    # Exclude ambiguous
    df = df[df[label_col] != "Ambiguous"].copy() if label_col in df.columns else df.copy()

    # Exclude specific labels
    if exclude_labels:
        before = len(df)
        df = df[~df[label_col].isin(exclude_labels)]
        log.info("Excluded labels %s: %d -> %d", exclude_labels, before, len(df))

    # Exclude OSM-tagged farm rows (inaccurate labels)
    if exclude_osm_farms:
        osm_mask = df["original_label"].fillna("").str.contains("OSM", case=False)
        # OSM rows that are farms: either standardized_label starts with Farm,
        # or standardized_label is NaN (OSM farm tags like sty, chicken_shed, etc.)
        osm_farm_mask = osm_mask & (
            df["standardized_label"].fillna("").str.startswith("Farm")
            | (df["standardized_label"].isna() & df["original_label"].fillna("").str.contains(
                "sty|chicken|poultry|cowshed|livestock|animal_breeding|pig|farm", case=False
            ))
        )
        before = len(df)
        df = df[~osm_farm_mask]
        log.info("Excluded OSM-tagged farms: %d -> %d (%d removed)", before, len(df), before - len(df))

    if not include_unlabeled:
        df = df[df[label_col].notna()].copy()
        log.info("Labelled only: %d", len(df))
    else:
        log.info("All clusters (incl. unlabeled): %d", len(df))

    # Compute centroids
    df["geom"] = df["geometry"].apply(shapely.from_wkb)
    df["lat"] = df["geom"].apply(lambda g: g.centroid.y)
    df["lng"] = df["geom"].apply(lambda g: g.centroid.x)

    # Assign labels based on mode
    if label_mode == "poultry":
        def _to_label(x):
            if pd.isna(x):
                return -1
            if "Poultry" in x:
                return 1
            return 0  # Pigs, Cattle, NotFarm, PigsOrPoultry → 0
        df["label"] = df[label_col].apply(_to_label)
        log.info("Poultry mode: %d poultry, %d other, %d unlabeled",
                 (df["label"] == 1).sum(), (df["label"] == 0).sum(), (df["label"] == -1).sum())
    elif label_mode == "multiclass":
        # 7-class farm-type taxonomy
        _MULTICLASS_MAP = {
            "NotFarm": 0,
            "Farm: Poultry: Meat Chickens": 1,
            "Farm: Poultry: Eggs": 2,
            "Farm: Poultry: Unspecified/Other": 3,
            "Farm: Pigs": 4,
            "Farm: Cattle": 5,
            "Farm: Mixed": 6,
            "Farm: Other": 6,
            "Farm: Unknown": 6,
            "Farm: PigsOrPoultry": 6,  # ambiguous → bucket with Other
        }
        def _to_label(x):
            if pd.isna(x):
                return -1
            return _MULTICLASS_MAP.get(x, -1)
        df["label"] = df[label_col].apply(_to_label)
        counts = df["label"].value_counts().sort_index()
        log.info("Multiclass labels: %s",
                 {int(k): int(v) for k, v in counts.items()})
    else:
        def _to_label(x):
            if pd.isna(x):
                return -1
            return 0 if x == "NotFarm" else 1
        df["label"] = df[label_col].apply(_to_label)

    # Map to our schema. Unknown ADM0 codes → lowercase code (e.g. "AFG" → "afg").
    df["id"] = df["cluster_id"]
    df["country_key"] = df["ADM0"].map(_ADM0_TO_KEY).fillna(df["ADM0"].str.lower())
    df["country"] = df["country_key"].map(_KEY_TO_NAME).fillna(df["ADM0"])
    df["source"] = "rachel_clusters"
    df["category"] = df[label_col]
    df["species"] = ""
    df["name"] = ""

    # Pass through viz_status for test set splitting
    if "viz_status" in df.columns:
        df["viz_status"] = df["viz_status"].fillna("")

    # Infer US states from coordinates
    us_mask = df["country_key"] == "united_states"
    df["state"] = ""
    if us_mask.any():
        df.loc[us_mask, "state"] = [
            _infer_us_state(row.lat, row.lng)
            for row in df.loc[us_mask].itertuples()
        ]
        n_resolved = (df.loc[us_mask, "state"] != "").sum()
        log.info("US state inference: %d/%d resolved", n_resolved, us_mask.sum())

    df["region"] = [
        build_region_string(k, s) for k, s in zip(df["country_key"], df["state"])
    ]

    # Keep candidate columns
    keep = ["id", "name", "lat", "lng", "species", "category", "source",
            "country", "state", "label", "region", "viz_status",
            "num_bldgs", "total_area_m2", "median_area", "template_score_if"]
    out_df = df[[c for c in keep if c in df.columns]].copy()

    # Save per country
    for country_key, grp in out_df.groupby(df["country_key"]):
        path = candidates_dir / f"{country_key}.csv"
        grp.to_csv(path, index=False)
        n_pos = (grp["label"] == 1).sum()
        n_neg = (grp["label"] == 0).sum()
        n_unk = (grp["label"] == -1).sum()
        log.info("Saved %d candidates to %s (pos=%d, neg=%d, unlabeled=%d)", len(grp), path, n_pos, n_neg, n_unk)

    log.info("Done. Total: %d candidates", len(out_df))
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Rachel clusters to candidates")
    parser.add_argument("--config", default="configs/rachel_clusters.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = resolve_paths(load_config(args.config))
    parquet_path = cfg.data.parquet_source
    if not parquet_path:
        parquet_path = Path(cfg.data.candidates_dir).parent / "selected_clusters_relabeled.parquet"
    convert(
        parquet_path,
        cfg.data.candidates_dir,
        include_unlabeled=getattr(cfg.data, "include_unlabeled", False),
        label_mode=getattr(cfg.data, "label_mode", "binary"),
        exclude_labels=getattr(cfg.data, "exclude_labels", None),
        exclude_osm_farms=getattr(cfg.data, "exclude_osm_farms", False),
        inspected_only=getattr(cfg.data, "inspected_only", False),
    )


if __name__ == "__main__":
    main()
