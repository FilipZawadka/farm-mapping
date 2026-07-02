"""AdaBN: adapt BatchNorm running stats to out-of-distribution countries.

Li et al. 2016 (https://arxiv.org/abs/1603.04779): a ResNet's BN running
mean/var ARE source-domain statistics. For OOD targets (Bangladesh, Nigeria)
re-estimate them with label-free forward passes over target patches, keeping
every learned weight untouched. Then evaluate the adapted model on the
generalization split.

No labels are used for adaptation — only pixel statistics.

Usage::

    python -m scripts.adabn_adapt --config configs/rachel_clusters/world_v8_three_class.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.config import load_config, resolve_paths
from training.dataset import build_splits
from training.model import build_model
from training.train import _evaluate, load_checkpoint, save_checkpoint

log = logging.getLogger("adabn")


@torch.no_grad()
def adapt_bn_stats(model: nn.Module, loader: DataLoader, device: torch.device,
                   max_batches: int | None = None) -> int:
    """Reset + re-estimate BN running stats from *loader* (cumulative average)."""
    n_bn = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average over all batches
            m.train()
            n_bn += 1
    if n_bn == 0:
        raise SystemExit("Model has no BatchNorm layers — AdaBN is a no-op "
                         "(e.g. ViT backbones use LayerNorm)")
    for i, (batch_x, _) in enumerate(loader):
        model(batch_x.to(device))
        if max_batches is not None and i + 1 >= max_batches:
            break
    model.eval()
    return n_bn


def main() -> None:
    parser = argparse.ArgumentParser(description="AdaBN adaptation + generalization eval")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap adaptation forward passes (default: all gen patches)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, inspected_ds, eval_ds, gen_ds = build_splits(cfg)
    if gen_ds is None or len(gen_ds) == 0:
        raise SystemExit("No generalization split — set data.generalization_countries")

    channel_subset = getattr(cfg.training, "channel_subset", None)
    if channel_subset:
        cfg.model.input_channels = len(channel_subset)
        cfg.model.in_channel_names = list(channel_subset)
    else:
        cfg.model.in_channel_names = list(cfg.patches.bands) + list(cfg.patches.indices)

    output_dir = Path(cfg.patches.output_dir).parent / "output" / cfg._config_stem
    best_path = output_dir / "best_model.pt"
    model = build_model(cfg.model).to(device)
    model.load_state_dict(load_checkpoint(best_path, device)["model_state_dict"])
    model.eval()

    bs = cfg.training.batch_size
    criterion = nn.CrossEntropyLoss()
    gen_loader = DataLoader(gen_ds, batch_size=bs, shuffle=False, num_workers=0)

    _, before = _evaluate(model, gen_loader, criterion, device)
    log.info("Generalization BEFORE AdaBN: %s", before)

    # Adapt on the SAME target-country patches (shuffled for stable BN batches).
    adapt_loader = DataLoader(gen_ds, batch_size=bs, shuffle=True, num_workers=0)
    n_bn = adapt_bn_stats(model, adapt_loader, device, max_batches=args.max_batches)
    log.info("Re-estimated running stats for %d BN layers", n_bn)

    _, after = _evaluate(model, gen_loader, criterion, device)
    log.info("Generalization AFTER AdaBN: %s", after)

    # In-distribution safety check: AdaBN'd model on the training-country test
    # split (expected to get worse — quantify the trade-off).
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    _, test_after = _evaluate(model, test_loader, criterion, device)
    log.info("Training-country test WITH target BN stats: %s", test_after)

    adapted_path = output_dir / "best_model_adabn.pt"
    save_checkpoint(adapted_path, model, None, None, None, epoch=-1, best_val_loss=0.0)
    report = {
        "generalization_before": before,
        "generalization_after": after,
        "test_with_target_stats": test_after,
        "checkpoint": str(adapted_path),
    }
    (output_dir / "adabn_report.json").write_text(json.dumps(report, indent=2))
    log.info("Saved adapted checkpoint to %s and adabn_report.json", adapted_path)


if __name__ == "__main__":
    main()
