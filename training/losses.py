"""Loss functions for imbalanced multi-class training.

Alternatives to plain class-weighted CrossEntropyLoss:

- FocalLoss (Lin et al. 2017): down-weights easy examples via (1-p_t)^gamma,
  concentrating gradient on hard/minority samples. ``alpha`` (optional)
  applies per-class weights on top, same semantics as CE ``weight``.
- LogitAdjustedLoss (Menon et al. 2021, "Long-tail learning via logit
  adjustment"): subtracts ``tau * log(prior_c)`` from each logit during
  training so the decision boundary is shifted toward rare classes in a
  Fisher-consistent way. At eval/inference the raw logits are used as-is.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        # Registered as buffer so .to(device) moves it with the module.
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal = (1.0 - pt).pow(self.gamma) * (-log_pt)
        if self.alpha is not None:
            focal = focal * self.alpha.to(logits.dtype)[target]
        return focal.mean()


class LogitAdjustedLoss(nn.Module):
    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        adjustment = tau * torch.log(class_priors.clamp_min(1e-12))
        self.register_buffer("adjustment", adjustment)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adjustment.to(logits.dtype), target)


def build_criterion(cfg, train_labels=None, device=None) -> nn.Module:
    """Build the training criterion from ``cfg.training``.

    ``train_labels`` (array-like of int) is required for logit_adjusted —
    priors are computed from the actual train split composition.
    """
    import numpy as np

    weights = None
    cw = cfg.training.class_weight
    if cw is not None and len(cw) >= 2:
        weights = torch.tensor(cw, dtype=torch.float32)

    loss_name = getattr(cfg.training, "loss", "cross_entropy")
    if loss_name == "focal":
        criterion = FocalLoss(
            gamma=getattr(cfg.training, "focal_gamma", 2.0), alpha=weights,
        )
    elif loss_name == "logit_adjusted":
        if train_labels is None:
            raise ValueError("logit_adjusted loss needs train_labels to compute priors")
        counts = np.bincount(
            np.asarray(train_labels, dtype=np.int64),
            minlength=cfg.model.num_classes,
        ).astype(np.float64)
        priors = torch.tensor(counts / counts.sum(), dtype=torch.float32)
        criterion = LogitAdjustedLoss(
            priors, tau=getattr(cfg.training, "logit_adjust_tau", 1.0),
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)

    return criterion.to(device) if device is not None else criterion
