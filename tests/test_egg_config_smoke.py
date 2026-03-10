"""End-to-end test for US egg farms config with synthetic data.

Validates config load, region-based train/val/test split, and a short training run
without Earth Engine or real farm data. Run with:  python tests/test_egg_config_smoke.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("egg_config_test")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Train/val/test states from configs/us_egg_farms.yaml
TRAIN_STATES = ["IA", "OH", "IN"]
VAL_STATES = ["PA"]
TEST_STATES = ["TX"]


def test_egg_config_loads():
    log.info("=== Step 1: Load US egg farms config ===")
    from training.config import load_config, resolve_paths

    cfg = load_config(ROOT / "configs" / "us_egg_farms.yaml")
    cfg = resolve_paths(cfg, root=ROOT)
    assert cfg.data.countries == ["united_states"]
    assert "Farm (eggs)" in cfg.data.categories_include
    assert cfg.data.train_regions == [f"united_states/{s}" for s in TRAIN_STATES]
    assert cfg.data.val_regions == [f"united_states/{s}" for s in VAL_STATES]
    assert cfg.data.test_regions == [f"united_states/{s}" for s in TEST_STATES]
    assert cfg.patches.output_dir
    assert "egg" in cfg.patches.output_dir or "us_egg" in cfg.patches.output_dir
    log.info("  Config OK: train=%s val=%s test=%s",
             cfg.data.train_regions, cfg.data.val_regions, cfg.data.test_regions)
    return cfg


def test_synthetic_candidates_egg(cfg):
    log.info("=== Step 2: Synthetic candidates (US states) ===")
    from training.config import build_region_string

    patches_dir = Path(cfg.patches.output_dir)
    patches_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    rows = []
    # Positives and negatives per state so we have train/val/test split
    for state, label, n in [
        ("IA", 1, 8), ("OH", 1, 8), ("IN", 1, 8),
        ("PA", 1, 5), ("TX", 1, 5),
        ("IA", 0, 5), ("OH", 0, 5), ("IN", 0, 5),
        ("PA", 0, 4), ("TX", 0, 4),
    ]:
        for i in range(n):
            lat = 40.0 + rng.uniform(-2, 2)
            lng = -95.0 + rng.uniform(-5, 5)
            region = build_region_string("united_states", state)
            rows.append({
                "id": f"{'pos' if label else 'neg'}_{state}_{i}",
                "lat": lat, "lng": lng, "label": label,
                "species": "Chickens" if label else "",
                "category": "Farm (eggs)" if label else "",
                "source": "FarmTransparency" if label else "random_rural",
                "country": "United States",
                "state": state,
                "region": region,
            })

    from shapely.geometry import Point
    import geopandas as gpd
    geometry = [Point(r["lng"], r["lat"]) for r in rows]
    candidates = gpd.GeoDataFrame(rows, geometry=geometry, crs="EPSG:4326")
    candidates.to_parquet(patches_dir / "candidates.parquet", index=False)
    log.info("  Created %d synthetic candidates across US states", len(candidates))
    return candidates


def test_synthetic_patches_egg(cfg, candidates):
    log.info("=== Step 3: Synthetic patches (relative paths) ===")
    patches_dir = Path(cfg.patches.output_dir)
    rng = np.random.default_rng(42)
    n_channels = cfg.model.input_channels
    size = cfg.patches.patch_size_px

    meta_rows = []
    for _, row in candidates.iterrows():
        cid = str(row["id"])
        arr = rng.standard_normal((n_channels, size, size)).astype(np.float32)
        if row["label"] == 1:
            arr += 0.5
        filename = f"{cid}.npy"
        patch_path = patches_dir / filename
        np.save(patch_path, arr)
        meta_rows.append({
            "candidate_id": cid,
            "lat": row["lat"],
            "lng": row["lng"],
            "n_channels": n_channels,
            "height": size,
            "width": size,
            "clear_pixel_fraction": 0.95,
            "patch_path": filename,
        })

    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(patches_dir / "patch_meta.parquet", index=False)
    log.info("  Created %d patches with relative paths", len(meta))
    return meta


def test_training_egg(cfg):
    log.info("=== Step 4: Training (1 epoch, CPU) ===")
    # Override for a fast local test
    cfg.training.epochs = 1
    cfg.training.mixed_precision = False

    from training.train import train

    best_path = train(cfg)
    assert best_path.exists(), f"Best model not found at {best_path}"
    log.info("  Training complete: %s", best_path)

    output_dir = best_path.parent
    metrics_path = output_dir / "training_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    log.info("  Test metrics: %s", metrics)
    return best_path


def main():
    log.info("Starting US egg farms config smoke test ...")

    cfg = test_egg_config_loads()
    candidates = test_synthetic_candidates_egg(cfg)
    test_synthetic_patches_egg(cfg, candidates)
    test_training_egg(cfg)

    log.info("=" * 50)
    log.info("EGG CONFIG SMOKE TEST PASSED")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
