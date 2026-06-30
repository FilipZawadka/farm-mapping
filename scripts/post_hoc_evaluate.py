"""Run inspected / eval / generalization evaluators against an existing
checkpoint. Use when a training run wrote `training_metrics.json` but
crashed before writing the post-training evaluators (e.g. disk quota hit
mid-run).

Mirrors the same evaluators that `training.train` would have produced, so
the JSON shape matches `inspected_metrics.json`, `eval_metrics.json`,
`generalization_metrics.json`, `eval_metrics_per_country.json`,
`generalization_metrics_per_country.json`.

Usage::

    python scripts/post_hoc_evaluate.py \
        --config configs/rachel_clusters/world_v4_three_class.yaml
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.config import load_config, resolve_paths
from training.dataset import build_splits
from training.model import build_model
from training.train import (
    _evaluate, load_checkpoint, _write_per_country_metrics,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = resolve_paths(load_config(args.config))

    train_ds, val_ds, test_ds, inspected_ds, eval_ds, gen_ds = build_splits(cfg, patches_dir=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve effective input channels
    channel_subset = getattr(cfg.training, "channel_subset", None)
    if channel_subset:
        cfg.model.input_channels = len(channel_subset)

    model = build_model(cfg.model).to(device)
    output_dir = Path(cfg.patches.output_dir).parent / "output" / cfg._config_stem
    best_path = output_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(best_path)
    model.load_state_dict(load_checkpoint(best_path, device)["model_state_dict"])

    criterion = torch.nn.CrossEntropyLoss()
    bs = cfg.training.batch_size

    def run(name, ds, write_per_country):
        if ds is None:
            print(f"  {name}: dataset empty, skipped")
            return
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        _, metrics = _evaluate(model, loader, criterion, device)
        print(f"  {name}: {metrics}")
        (output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
        if write_per_country:
            _write_per_country_metrics(
                ds, model, criterion, device, bs,
                output_dir / f"{name}_metrics_per_country.json", cfg,
            )

    run("inspected", inspected_ds, write_per_country=False)
    run("eval", eval_ds, write_per_country=True)
    run("generalization", gen_ds, write_per_country=True)
    print("Done.")


if __name__ == "__main__":
    main()
