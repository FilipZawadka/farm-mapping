"""End-to-end smoke test: config -> candidates -> synthetic patches -> train -> infer -> viz.

Uses synthetic .npy patches to avoid needing Earth Engine credentials.
Run with:  python tests/smoke_test_pipeline.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke_test")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_config_loading():
    log.info("=== Step 1: Config loading ===")
    from training.config import load_config, resolve_paths

    cfg = load_config(ROOT / "configs" / "smoke_test.yaml")
    cfg = resolve_paths(cfg, root=ROOT)
    assert cfg.training.epochs == 2
    assert cfg.patches.patch_size_px == 32
    assert cfg.model.input_channels == 9
    log.info("  Config loaded OK: %d epochs, %dpx patches, %d channels",
             cfg.training.epochs, cfg.patches.patch_size_px, cfg.model.input_channels)
    return cfg


def test_candidates(cfg):
    log.info("=== Step 2: Candidate building (synthetic) ===")
    patches_dir = Path(cfg.patches.output_dir)
    patches_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_pos, n_neg = 25, 25
    rows = []
    for i in range(n_pos):
        rows.append({
            "id": f"pos_{i}", "name": f"farm_{i}",
            "lat": 13.0 + rng.uniform(-1, 1),
            "lng": 100.0 + rng.uniform(-1, 1),
            "label": 1, "species": "Chickens", "category": "Farm",
            "source": "FarmTransparency", "country": "Thailand",
            "state": "", "region": "thailand",
        })
    for i in range(n_neg):
        rows.append({
            "id": f"neg_{i}", "name": "",
            "lat": 14.0 + rng.uniform(-1, 1),
            "lng": 101.0 + rng.uniform(-1, 1),
            "label": 0, "species": "", "category": "",
            "source": "random_rural", "country": "Thailand",
            "state": "", "region": "thailand",
        })

    from shapely.geometry import Point
    import geopandas as gpd
    geometry = [Point(r["lng"], r["lat"]) for r in rows]
    candidates = gpd.GeoDataFrame(rows, geometry=geometry, crs="EPSG:4326")
    candidates.to_parquet(patches_dir / "candidates.parquet", index=False)
    log.info("  Created %d synthetic candidates", len(candidates))
    return candidates


def test_patches(cfg, candidates):
    log.info("=== Step 3: Synthetic patch generation ===")
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
        patch_path = patches_dir / f"{cid}.npy"
        np.save(patch_path, arr)
        meta_rows.append({
            "candidate_id": cid,
            "lat": row["lat"],
            "lng": row["lng"],
            "n_channels": n_channels,
            "height": size,
            "width": size,
            "clear_pixel_fraction": 0.95,
            "patch_path": str(patch_path),
        })

    meta = pd.DataFrame(meta_rows)
    meta.to_parquet(patches_dir / "patch_meta.parquet", index=False)
    log.info("  Created %d synthetic patches (%d x %d x %d)", len(meta), n_channels, size, size)
    return meta


def test_training(cfg):
    log.info("=== Step 4: Training (2 epochs, CPU) ===")
    from training.train import train

    best_path = train(cfg)
    assert best_path.exists(), f"Best model not found at {best_path}"
    log.info("  Training complete, best model: %s", best_path)

    output_dir = best_path.parent
    metrics_path = output_dir / "training_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    log.info("  Test metrics: %s", metrics)
    return best_path


def test_inference(cfg, best_path):
    log.info("=== Step 5: Inference ===")
    cfg.inference.checkpoint = str(best_path)

    from training.inference import score_candidates

    scored = score_candidates(cfg)
    assert len(scored) > 0
    assert "predicted_score" in scored.columns
    assert "confidence_tier" in scored.columns
    log.info("  Scored %d candidates, columns: %s", len(scored), list(scored.columns))
    return scored


def test_visualization(cfg):
    log.info("=== Step 6: Visualization ===")
    from training.visualize import visualize

    map_path = visualize(cfg)
    assert map_path.exists()
    html = map_path.read_text()
    assert "leaflet" in html.lower()
    log.info("  Map saved: %s (%d bytes)", map_path, len(html))
    return map_path


def test_mlflow_logged():
    log.info("=== Step 7: MLflow verification ===")
    import mlflow

    mlflow.set_tracking_uri("./mlruns")
    experiment = mlflow.get_experiment_by_name("smoke_test")
    assert experiment is not None, "MLflow experiment 'smoke_test' not found"

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0, "No MLflow runs found"
    log.info("  MLflow experiment found: %d run(s)", len(runs))
    latest = runs.iloc[0]
    log.info("  Latest run: status=%s", latest.get("status", "unknown"))


def main():
    log.info("Starting smoke test ...")

    cfg = test_config_loading()
    candidates = test_candidates(cfg)
    test_patches(cfg, candidates)
    best_path = test_training(cfg)
    test_inference(cfg, best_path)
    test_visualization(cfg)
    test_mlflow_logged()

    log.info("=" * 50)
    log.info("ALL SMOKE TESTS PASSED")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
