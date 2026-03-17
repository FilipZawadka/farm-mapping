"""HuggingFace model builder with input-channel adaptation and registry pattern.

Supports any image classification backbone from the Hub. The builder:
1. Loads the pretrained backbone.
2. Adapts the first conv layer when ``input_channels != 3``.
3. Replaces the classification head with the correct number of classes.
4. Exposes ``freeze_backbone`` / ``unfreeze_backbone`` for staged training.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .config import ModelConfig

log = logging.getLogger(__name__)


def _adapt_first_conv(module: nn.Module, target_channels: int) -> nn.Module:
    """Replace the first Conv2d so it accepts *target_channels* instead of 3.

    Copies the original RGB weights where possible; new channels are initialised
    from the mean of the original weights.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels in (3, 1):
            old_weight = child.weight.data
            out_c, _, kh, kw = old_weight.shape

            new_conv = nn.Conv2d(
                target_channels, out_c, kernel_size=(kh, kw),
                stride=child.stride, padding=child.padding,
                dilation=child.dilation, groups=child.groups, bias=child.bias is not None,
            )

            with torch.no_grad():
                mean_weight = old_weight.mean(dim=1, keepdim=True)
                new_weight = mean_weight.repeat(1, target_channels, 1, 1)
                copy_c = min(old_weight.shape[1], target_channels)
                new_weight[:, :copy_c, :, :] = old_weight[:, :copy_c, :, :]
                new_conv.weight.copy_(new_weight)
                if child.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(child.bias)

            setattr(module, name, new_conv)
            log.info("Adapted first conv: %d -> %d input channels", child.in_channels, target_channels)
            return module

        result = _adapt_first_conv(child, target_channels)
        if result is not child:
            return module

    return module


def _try_replace_classifier(model: nn.Module, num_classes: int) -> bool:
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        in_features = _head_in_features(model.classifier)
        model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_features, num_classes))
        return True
    return False


def _try_replace_fc(model: nn.Module, num_classes: int) -> bool:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return True
    return False


def _try_replace_head_attr(model: nn.Module, num_classes: int) -> bool:
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return True
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return True
    return False


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the classifier head with one having the right number of outputs."""
    for fn in (_try_replace_classifier, _try_replace_fc, _try_replace_head_attr):
        if fn(model, num_classes):
            return model
    log.warning("Could not locate classification head -- model returned as-is")
    return model


def _head_in_features(module: nn.Module) -> int:
    """Walk backwards through a head module to find the last Linear's in_features."""
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if linears:
        return linears[-1].in_features
    for child in reversed(list(module.children())):
        if isinstance(child, nn.AdaptiveAvgPool2d):
            continue
        if hasattr(child, "in_features"):
            return child.in_features
    return 2048


class FarmDetector(nn.Module):
    """Wraps a HuggingFace backbone with channel adaptation and a new head."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self._uses_pixel_values = hasattr(backbone, "config")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._uses_pixel_values:
            out = self.backbone(pixel_values=x)
        else:
            out = self.backbone(x)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        return out

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for head in self._head_parameters():
            head.requires_grad = True
        log.info("Backbone frozen; only head is trainable")

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        log.info("Backbone unfrozen; all parameters trainable")

    def _head_parameters(self):
        for attr in ("classifier", "fc", "head", "heads"):
            sub = getattr(self.backbone, attr, None)
            if sub is not None and isinstance(sub, nn.Module):
                yield from sub.parameters()
                return
        yield from self.backbone.parameters()


def _update_config_channels(backbone: nn.Module, n_channels: int) -> None:
    """Update the HuggingFace model config so the forward-pass channel check passes.

    Some HuggingFace models cache ``num_channels`` in sub-modules at init time
    (e.g. ResNetEmbeddings), so we walk the tree and patch them all.
    """
    if hasattr(backbone, "config") and hasattr(backbone.config, "num_channels"):
        backbone.config.num_channels = n_channels

    for module in backbone.modules():
        if hasattr(module, "num_channels"):
            module.num_channels = n_channels


def _build_from_hub(cfg: ModelConfig) -> FarmDetector:
    """Shared builder: load from Hub, adapt channels/head, return FarmDetector."""
    from transformers import AutoModelForImageClassification

    backbone = AutoModelForImageClassification.from_pretrained(
        cfg.hub_name,
        num_labels=cfg.num_classes,
        ignore_mismatched_sizes=True,
    )
    if cfg.input_channels != 3:
        _adapt_first_conv(backbone, cfg.input_channels)
        _update_config_channels(backbone, cfg.input_channels)
    _replace_head(backbone, cfg.num_classes)
    return FarmDetector(backbone, cfg.num_classes)


def build_resnet(cfg: ModelConfig) -> FarmDetector:
    return _build_from_hub(cfg)


def build_vit(cfg: ModelConfig) -> FarmDetector:
    return _build_from_hub(cfg)


def build_generic(cfg: ModelConfig) -> FarmDetector:
    """Fallback builder for any HuggingFace image model."""
    return _build_from_hub(cfg)


def build_torchgeo_resnet(cfg: ModelConfig) -> FarmDetector:
    """Load a ResNet50 with torchgeo pretrained weights (e.g. Satlas, SSL4EO)."""
    import torchgeo.models
    import torchvision.models as tv_models

    weights_name = cfg.hub_name  # e.g. "SENTINEL2_SI_MS_SATLAS"
    weights_enum = getattr(torchgeo.models.ResNet50_Weights, weights_name)
    backbone = torchgeo.models.resnet50(weights=weights_enum)

    # torchgeo returns a torchvision ResNet — adapt channels if needed
    pretrained_channels = backbone.conv1.in_channels
    if cfg.input_channels != pretrained_channels:
        _adapt_first_conv(backbone, cfg.input_channels)
        log.info("Adapted first conv: %d -> %d input channels", pretrained_channels, cfg.input_channels)

    backbone.fc = nn.Linear(backbone.fc.in_features, cfg.num_classes)

    return FarmDetector(backbone, cfg.num_classes)


MODEL_BUILDERS: dict[str, callable] = {
    "resnet50": build_resnet,
    "resnet50_satlas": build_torchgeo_resnet,
    "resnet50_ssl4eo": build_torchgeo_resnet,
    "vit_small": build_vit,
    "vit_base": build_vit,
    "prithvi_eo": build_generic,
    # New architectures — all use the generic HuggingFace builder
    "convnext_tiny": build_generic,
    "efficientnet_b0": build_generic,
    "swin_tiny": build_generic,
}


def build_model(cfg: ModelConfig) -> FarmDetector:
    """Build a model from config, using the architecture registry."""
    builder = MODEL_BUILDERS.get(cfg.architecture, build_generic)
    model = builder(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model built: %s (%.1fM params, %.1fM trainable)",
        cfg.architecture,
        total / 1e6,
        trainable / 1e6,
    )
    return model
