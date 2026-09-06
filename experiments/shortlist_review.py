"""Sample shortlist-country candidates (IDN/PHL/VNM) and fetch satellite imagery
for visual review.

Produces:
  experiments/results/shortlist_review_manifest.csv
  <out>/{ISO}/{stratum}__{candidate_id}__z{zoom}.jpg   (Google Static Maps)

Usage:
  python3 experiments/shortlist_review.py --out /tmp/shortlist_imgs [--fetch]
  python3 experiments/shortlist_review.py --grid 10.123,105.456 --iso VNM --out DIR  # 3x3 zoom-16 grid
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

ISOS = ["IDN", "PHL", "VNM"]
SEED = 42
CLASS_NAMES = ["NotFarm", "Poultry", "Pigs", "Cattle"]


def build_manifest() -> pd.DataFrame:
    df = lib.load(lib.FOURCLASS["v9"])
    df = df[df.ADM0.isin(ISOS)].copy()
    df["p_farm"] = 1.0 - df["prob_class0"]
    rng = np.random.default_rng(SEED)
    rows = []

    def take(d: pd.DataFrame, stratum: str, n: int | None = None, sort: str | None = None):
        if sort:
            d = d.sort_values(sort, ascending=False)
        if n is not None and len(d) > n:
            d = d.sample(n, random_state=SEED) if not sort else d.head(n)
        for _, r in d.iterrows():
            rows.append({
                "candidate_id": r.candidate_id, "iso": r.ADM0, "stratum": stratum,
                "lat": r.lat, "lng": r.lng,
                "pred": CLASS_NAMES[int(r.predicted_label)],
                "p_farm": round(float(r.p_farm), 3),
                "p_poultry": round(float(r.prob_class1), 3),
                "p_pigs": round(float(r.prob_class2), 3),
                "true": CLASS_NAMES[int(r.true_label)] if r.true_label >= 0 else "",
                "split": r.cnn_split_assigned,
            })

    for iso in ISOS:
        d = df[df.ADM0 == iso]
        lab = d[d.true_label >= 0]
        unl = d[d.true_label < 0]
        take(unl[unl.predicted_label == 1], f"poultry_top", 10, sort="prob_class1")
        take(unl[unl.predicted_label == 2], f"pigs_top", 10, sort="prob_class2")
        take(unl[(unl.p_farm >= 0.4) & (unl.p_farm <= 0.6)], "borderline", 10)
        take(unl[unl.predicted_label == 0], "notfarm_top", 8, sort="prob_class0")
        dis = lab[lab.true_label != lab.predicted_label].copy()
        dis["conf_wrong"] = dis[["prob_class0","prob_class1","prob_class2","prob_class3"]].max(axis=1)
        take(dis, "disagreement", 12, sort="conf_wrong")
        agr = lab[lab.true_label == lab.predicted_label]
        take(agr, "agreement", 5)

    m = pd.DataFrame(rows).drop_duplicates("candidate_id")
    return m


ESRI = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}")


def _tile_xy(lat: float, lng: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom
    x = (lng + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2.0 * n
    return x, y


def tile_mosaic(lat: float, lng: float, zoom: int, grid: int = 3):
    """Esri World Imagery mosaic centred on (lat,lng): grid x grid 256px tiles."""
    from PIL import Image
    import io
    xf, yf = _tile_xy(lat, lng, zoom)
    cx, cy = int(xf), int(yf)
    half = grid // 2
    img = Image.new("RGB", (256 * grid, 256 * grid))
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            url = ESRI.format(z=zoom, y=cy + dy, x=cx + dx)
            req = urllib.request.Request(url, headers={"User-Agent": "farm-mapping-review"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                tile = Image.open(io.BytesIO(resp.read())).convert("RGB")
            img.paste(tile, ((dx + half) * 256, (dy + half) * 256))
            time.sleep(0.03)
    # crop so the point is centred, not the tile
    offx = int((xf - cx) * 256) + (half * 256) - 320
    offy = int((yf - cy) * 256) + (half * 256) - 320
    offx = max(0, min(256 * grid - 640, offx))
    offy = max(0, min(256 * grid - 640, offy))
    return img.crop((offx, offy, offx + 640, offy + 640))


def fetch(m: pd.DataFrame, out: Path, key: str, wide_strata=("borderline", "disagreement")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    for _, r in m.iterrows():
        zooms = [17] + ([15] if r.stratum in wide_strata else [])
        for z in zooms:
            dst = out / r.iso / f"{r.stratum}__{r.candidate_id}__z{z}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue
            try:
                tile_mosaic(r.lat, r.lng, z).save(dst, quality=85)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {r.candidate_id} z{z}: {exc}")
                n_fail += 1
    print(f"fetched {n_ok} images, {n_fail} failures -> {out}")


def fetch_grid(center: str, iso: str, out: Path, key: str, zoom: int = 16) -> None:
    """3x3 grid of tiles around a hotspot for unseen-farm scanning."""
    lat0, lng0 = (float(x) for x in center.split(","))
    # zoom-16 640px covers ~1000m at low latitudes; step ~0.85 of the footprint
    step_m = 156543.03 / (2 ** zoom) * 640 * 0.85 * np.cos(np.radians(lat0))
    dlat = step_m / 111_320
    dlng = step_m / (111_320 * np.cos(np.radians(lat0)))
    out.mkdir(parents=True, exist_ok=True)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            lat, lng = lat0 + i * dlat, lng0 + j * dlng
            dst = out / f"grid_{iso}_{lat0:.4f}_{lng0:.4f}__r{i+1}c{j+1}.jpg"
            if dst.exists():
                continue
            tile_mosaic(lat, lng, zoom).save(dst, quality=85)
    print(f"grid written -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/shortlist_imgs")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--grid", help="lat,lng hotspot center for 3x3 zoom-16 grid")
    ap.add_argument("--iso", default="XXX")
    args = ap.parse_args()

    key = ""  # imagery source is Esri World Imagery tiles; no key needed
    if args.grid:
        fetch_grid(args.grid, args.iso, Path(args.out), key)
        return

    m = build_manifest()
    dst = lib.RESULTS / "shortlist_review_manifest.csv"
    m.to_csv(dst, index=False)
    print(f"manifest: {len(m)} rows -> {dst}")
    print(m.groupby(["iso", "stratum"]).size().unstack(fill_value=0).to_string())
    if args.fetch:
        fetch(m, Path(args.out), key)


if __name__ == "__main__":
    main()
