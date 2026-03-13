"""Main training loop with MLflow logging, mixed precision, and early stopping.

Usage::

    python -m training.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import PipelineConfig, cache_key, load_config, resolve_paths
from .dataset import build_splits
from .model import build_model

log = logging.getLogger(__name__)


def _compute_metrics(
    preds: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 for the positive class."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


@torch.no_grad()
def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, dict]:
    """Run one eval pass and return (loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * len(batch_y)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch_y.cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = _compute_metrics(all_preds, all_labels)
    metrics["loss"] = round(avg_loss, 6)
    return avg_loss, metrics


def _make_optimizer(model, cfg: PipelineConfig, lr_scale: float = 1.0):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate * lr_scale,
        weight_decay=cfg.training.weight_decay,
    )


def _train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler):
    """Run one training epoch and return the average loss."""
    model.train()
    running = 0.0
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        running += loss.item() * len(batch_y)
    return running / max(len(loader.dataset), 1)


def _step_scheduler(scheduler, cfg: PipelineConfig, val_loss: float) -> None:
    if scheduler is None:
        return
    if cfg.training.scheduler == "plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def _log_mlflow_params(cfg: PipelineConfig, sizes: dict) -> None:
    mlflow.log_params({
        "architecture": cfg.model.architecture,
        "hub_name": cfg.model.hub_name,
        "input_channels": cfg.model.input_channels,
        "patch_size_px": cfg.patches.patch_size_px,
        "resolution_m": cfg.patches.resolution_m,
        "bands": ",".join(cfg.patches.bands),
        "indices": ",".join(cfg.patches.indices),
        "epochs": cfg.training.epochs,
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "scheduler": cfg.training.scheduler,
        "freeze_backbone_epochs": cfg.model.freeze_backbone_epochs,
        **sizes,
    })


def _save_test_results(
    model, test_loader, criterion, device, output_dir, best_path, cfg,
):
    model.load_state_dict(torch.load(best_path, weights_only=True))
    _, test_metrics = _evaluate(model, test_loader, criterion, device)
    mlflow.log_metrics({
        f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))
    })
    log.info("Test metrics: %s", test_metrics)

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2))
    mlflow.log_artifact(str(metrics_path))

    if cfg.mlflow.log_model:
        mlflow.log_artifact(str(best_path))


class _TrainCtx:
    """Bundles training state to keep function signatures short."""

    __slots__ = ("model", "criterion", "device", "use_amp", "scaler", "cfg", "best_path")

    def __init__(self, model, criterion, device, use_amp, scaler, cfg, best_path):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.scaler = scaler
        self.cfg = cfg
        self.best_path = best_path


def _run_epoch_loop(ctx: _TrainCtx, train_loader, val_loader):
    """Inner training loop across all epochs."""
    optimizer = _make_optimizer(ctx.model, ctx.cfg)
    scheduler = _build_scheduler(optimizer, ctx.cfg)
    best_val_loss, patience = float("inf"), 0

    for epoch in range(1, ctx.cfg.training.epochs + 1):
        if ctx.cfg.model.freeze_backbone_epochs > 0 and epoch == ctx.cfg.model.freeze_backbone_epochs + 1:
            ctx.model.unfreeze_backbone()
            optimizer = _make_optimizer(ctx.model, ctx.cfg, lr_scale=0.1)
            scheduler = _build_scheduler(optimizer, ctx.cfg)

        t_loss = _train_one_epoch(ctx.model, train_loader, ctx.criterion, optimizer, ctx.device, ctx.use_amp, ctx.scaler)
        v_loss, vm = _evaluate(ctx.model, val_loader, ctx.criterion, ctx.device)
        _step_scheduler(scheduler, ctx.cfg, v_loss)

        mlflow.log_metrics({"train_loss": t_loss, "val_loss": v_loss,
                            "val_f1": vm["f1"], "lr": optimizer.param_groups[0]["lr"]}, step=epoch)
        log.info("Epoch %d/%d  loss=%.4f  val=%.4f  f1=%.4f", epoch, ctx.cfg.training.epochs, t_loss, v_loss, vm["f1"])

        if v_loss < best_val_loss:
            best_val_loss, patience = v_loss, 0
            torch.save(ctx.model.state_dict(), ctx.best_path)
        else:
            patience += 1
        if patience >= ctx.cfg.training.early_stopping_patience:
            log.info("Early stopping at epoch %d", epoch)
            break


def train(cfg: PipelineConfig) -> Path:
    """Full training run. Returns path to the best model checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    patches_dir = None
    if getattr(cfg, "cache", None) and cfg.cache.enabled:
        from .storage import get_cache_backend
        backend = get_cache_backend(cfg)
        key = cache_key(cfg)
        if backend and backend.exists(key):
            download_dir = Path(cfg.patches.output_dir).parent / ".cache" / key
            download_dir.mkdir(parents=True, exist_ok=True)
            backend.get_dir(key, download_dir)
            patches_dir = download_dir
            log.info("Using cached patches from %s", patches_dir)

    train_ds, val_ds, test_ds = build_splits(cfg, patches_dir=patches_dir)
    bs = cfg.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = build_model(cfg.model).to(device)
    if cfg.model.freeze_backbone_epochs > 0:
        model.freeze_backbone()

    weights = None
    if cfg.training.class_weight is not None and len(cfg.training.class_weight) >= 2:
        weights = torch.tensor(cfg.training.class_weight, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    output_dir = Path(cfg.patches.output_dir).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"

    ctx = _TrainCtx(model, criterion, device, use_amp, scaler, cfg, best_path)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        _log_mlflow_params(cfg, {"train_size": len(train_ds), "val_size": len(val_ds), "test_size": len(test_ds)})
        _run_epoch_loop(ctx, train_loader, val_loader)
        _save_test_results(model, test_loader, criterion, device, output_dir, best_path, cfg)

    return best_path


def _build_scheduler(optimizer, cfg: PipelineConfig):
    if cfg.training.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    if cfg.training.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if cfg.training.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
    return None


def main() -> None:
    from .env_loader import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train farm detection model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = resolve_paths(load_config(args.config))
    train(cfg)


if __name__ == "__main__":
    main()
