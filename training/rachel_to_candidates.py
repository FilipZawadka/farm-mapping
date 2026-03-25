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
) -> pd.DataFrame:
    """Convert parquet to candidate CSVs.

    If *include_unlabeled* is True, keeps all clusters (unlabeled get label=-1).
    Returns the full DataFrame for inspection.
    """
    parquet_path = Path(parquet_path)
    candidates_dir = Path(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("Total clusters: %d", len(df))

    if include_unlabeled:
        # Keep everything — unlabeled get label=-1
        df = df[df["modified_label"] != "Ambiguous"].copy() if "modified_label" in df.columns else df.copy()
        log.info("All clusters (excl. Ambiguous): %d", len(df))
    else:
        # Filter to labelled, non-ambiguous
        df = df[df["modified_label"].notna() & (df["modified_label"] != "Ambiguous")].copy()
        log.info("Labelled (excl. Ambiguous): %d", len(df))

    # Compute centroids
    df["geom"] = df["geometry"].apply(shapely.from_wkb)
    df["lat"] = df["geom"].apply(lambda g: g.centroid.y)
    df["lng"] = df["geom"].apply(lambda g: g.centroid.x)

    # Binary label: farm=1, not-farm=0, unlabeled=-1
    def _to_label(x):
        if pd.isna(x):
            return -1
        return 0 if x == "NotFarm" else 1
    df["label"] = df["modified_label"].apply(_to_label)

    # Map to our schema
    df["id"] = df["cluster_id"]
    df["country_key"] = df["ADM0"].map(_ADM0_TO_KEY)
    df["country"] = df["country_key"].map(_KEY_TO_NAME)
    df["source"] = "rachel_clusters"
    df["category"] = df["modified_label"]
    df["species"] = ""
    df["name"] = ""

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
            "country", "state", "label", "region",
            "num_bldgs", "total_area_m2", "median_area", "template_score_if"]
    out_df = df[[c for c in keep if c in df.columns]].copy()

    # Save per country
    for country_key, grp in out_df.groupby(df["country_key"]):
        path = candidates_dir / f"{country_key}.csv"
        grp.to_csv(path, index=False)
        n_pos = (grp["label"] == 1).sum()
        n_neg = (grp["label"] == 0).sum()
        log.info("Saved %d candidates to %s (pos=%d, neg=%d)", len(grp), path, n_pos, n_neg)

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
    if cfg.data.parquet_source:
        parquet_path = Path(cfg.data.parquet_source)
    else:
        parquet_path = Path(cfg.data.candidates_dir).parent / "selected_clusters_relabeled.parquet"
    convert(parquet_path, cfg.data.candidates_dir)


if __name__ == "__main__":
    main()
