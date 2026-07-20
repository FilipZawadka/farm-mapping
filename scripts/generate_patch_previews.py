"""Render small true-color JPEG previews of the actual Sentinel-2 patches fed
to the model, for the website's "model input patch" panel.

Reads directly from patch_meta.csv (deduped + coordinate-validated the same
way training/inference do -- see training/config.py validate_patch_locations)
so the preview is guaranteed to be the literal array the model scored, not a
basemap lookup at the label's coordinates.

Usage (run on a machine/pod with access to the patch store)::

    python scripts/generate_patch_previews.py \
        --patches-root data/patches \
        --imagery-hash cc5a6ebb502a \
        --ids-file /tmp/ids.txt \
        --out /tmp/patch_previews \
        --candidates-parquet data/rachel_geometry_candidates/all_countries/all_clusters_v5.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.config import validate_patch_locations  # noqa: E402

# True-color bands: patch channel order is always [bands..., indices...] and
# every release to date uses bands [B2, B3, B4, ...] -> R=idx2 (B4), G=idx1
# (B3), B=idx0 (B2). Raw S2 SR values are 0-10000 reflectance*1e4.
RGB_BAND_IDX = (2, 1, 0)
REFLECTANCE_CLIP = 0.30  # matches common S2 true-color quicklook stretch (GEE default min=0,max=3000)
GAMMA = 0.9


def to_thumbnail(arr: np.ndarray) -> Image.Image:
    rgb = arr[list(RGB_BAND_IDX)].astype(np.float32) / 10_000.0
    rgb = np.clip(rgb, 0.0, REFLECTANCE_CLIP) / REFLECTANCE_CLIP
    rgb = np.power(rgb, GAMMA)
    rgb = np.transpose(rgb, (1, 2, 0))  # C,H,W -> H,W,C
    rgb = np.nan_to_num(rgb, nan=0.0)
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8), mode="RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches-root", required=True)
    ap.add_argument("--imagery-hash", required=True)
    ap.add_argument("--ids-file", required=True, help="newline-separated candidate_ids to render")
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates-parquet", required=True,
                     help="all_clusters parquet -- used to validate patch coords still match")
    ap.add_argument("--jpeg-quality", type=int, default=82)
    args = ap.parse_args()

    patches_root = Path(args.patches_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(Path(args.ids_file).read_text().splitlines())
    wanted = {w.strip() for w in wanted if w.strip()}
    print(f"requested {len(wanted)} candidate previews")

    meta = pd.read_csv(patches_root / "patch_meta.csv", low_memory=False)
    meta = meta[meta["imagery_config_hash"].astype(str) == args.imagery_hash]
    meta = meta[meta["candidate_id"].astype(str).isin(wanted)]

    clusters = pd.read_parquet(args.candidates_parquet, columns=["cluster_id", "geometry"])
    import shapely
    geoms = shapely.from_wkb(clusters["geometry"].values)
    cents = shapely.centroid(geoms)
    cand = pd.DataFrame({
        "id": clusters["cluster_id"].astype(str),
        "lat": shapely.get_y(cents),
        "lng": shapely.get_x(cents),
    })

    meta = validate_patch_locations(meta, cand, context="generate_patch_previews")
    print(f"{len(meta)}/{len(wanted)} candidates have a coordinate-valid patch")

    missing = wanted - set(meta["candidate_id"].astype(str))
    if missing:
        print(f"WARNING: {len(missing)} requested ids have no valid patch (skipped)")

    n_ok = 0
    n_fail = 0
    for _, row in meta.iterrows():
        cid = str(row["candidate_id"])
        rel = row["patch_path"]
        path = (patches_root / rel) if not Path(rel).is_absolute() else Path(rel)
        try:
            arr = np.load(path)
            img = to_thumbnail(arr)
            img.save(out_dir / f"{cid}.jpg", format="JPEG", quality=args.jpeg_quality)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {cid}: {exc}")
            n_fail += 1

    print(f"wrote {n_ok} previews to {out_dir} ({n_fail} failed)")


if __name__ == "__main__":
    main()
