"""Generate GPU experiment configs from the production baseline.

Every config is the production recipe (world_v10_fourclass_v9) with ONE lever
changed, so results are attributable. Writes to configs/experiments/.

Run:  python3 experiments/gen_configs.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "configs" / "rachel_clusters" / "world_v10_fourclass_v9.yaml"
OUT = REPO / "configs" / "experiments"

# EU-RO-1 (where the network volume lives) has no A40/A5000/A6000.
# 4090 is High stock at ~$0.34/hr; L4 is the only other config-listed fallback there.
GPU = "NVIDIA GeForce RTX 4090"
GPU_FALLBACKS = ["NVIDIA L4"]

BANDS6 = ["B2", "B3", "B4", "B8", "B11", "B12"]

# name -> (experiment id, description, deltas applied to the base config)
MATRIX: dict[str, tuple[str, str, dict]] = {}


def add(name: str, exp: str, desc: str, deltas: dict) -> None:
    MATRIX[name] = (exp, desc, deltas)


# ---------------------------------------------------------------- E0.3 seeds
for s in (43, 44, 45, 46):
    add(f"e03_seed{s}", "E0.3", f"seed variance, seed={s}", {"training": {"seed": s}})

# -------------------------------------------------------------- E1.2 context
add("e12_crop48", "E1.2", "context 48 px (480 m)", {"training": {"crop_center_px": 48}})
add("e12_crop128", "E1.2", "context 128 px (1.28 km, full patch)",
    {"training": {"crop_center_px": 128}})

# ---------------------------------------------------------------- E1.1 bands
add("e11_rgb", "E1.1", "RGB only", {
    "model": {"input_channels": 3},
    "training": {"channel_subset": ["B2", "B3", "B4"]}})
add("e11_rgb_nir", "E1.1", "RGB + NIR", {
    "model": {"input_channels": 4},
    "training": {"channel_subset": ["B2", "B3", "B4", "B8"]}})
add("e11_rgb_ndwi", "E1.1", "RGB + NDWI", {
    "model": {"input_channels": 4},
    "training": {"channel_subset": ["B2", "B3", "B4", "NDWI"]}})
add("e11_6bands", "E1.1", "6 spectral bands, no indices", {
    "model": {"input_channels": 6},
    "training": {"channel_subset": BANDS6}})
add("e11_recompute_idx", "E1.1", "9ch with indices recomputed after jitter", {
    "training": {"augmentation": {"recompute_indices": True}}})

# --------------------------------------------------------------- E1.6 backbone
# No resnet18 builder exists, so this tests the SoftCon-vs-SSL4EO choice that the
# paper flags as resting on external benchmark evidence rather than our own.
add("e16_ssl4eo", "E1.6", "SSL4EO-S12 MoCo backbone instead of SoftCon", {
    "model": {"architecture": "resnet50_ssl4eo", "hub_name": "SENTINEL2_ALL_MOCO"},
    "training": {"normalization": "none"}})

# -------------------------------------------------------- E1.7 checkpoint metric
add("e17_val_loss", "E1.7", "select checkpoint on val_loss instead of val_f1",
    {"training": {"checkpoint_metric": "val_loss"}})

# -------------------------------------------------------------- E1.9 taxonomy
# label_mode is baked into the candidate CSVs by the `candidates` step, so this
# run MUST regenerate them (--steps candidates train inference). Writing to its
# own candidates_dir keeps the shared four-class store intact; without both, the
# 3-class head meets 4-class labels and training dies on a CUDA device assert
# (`t >= 0 && t < n_classes`).
add("e19_three_class", "E1.9", "3-class taxonomy on identical data", {
    "model": {"num_classes": 3, "class_names": ["NotFarm", "Poultry", "OtherFarm"]},
    "data": {"label_mode": "three_class",
             "candidates_dir": "data/rachel_geometry_candidates/candidates_world_v10_v9_threeclass"}})

# -------------------------------------------------------------- E1.4 optimiser
add("e14_lr3e-5", "E1.4", "learning rate 3e-5", {"training": {"learning_rate": 3e-5}})
add("e14_lr3e-4", "E1.4", "learning rate 3e-4", {"training": {"learning_rate": 3e-4}})
add("e14_freeze0", "E1.4", "no backbone freeze phase", {"model": {"freeze_backbone_epochs": 0}})

# ----------------------------------------------------------- E1.5 augmentation
add("e15_no_photometric", "E1.5", "drop all 4 photometric augs", {"training": {"augmentation": {
    "brightness_jitter": {"enabled": False},
    "per_band_jitter": {"enabled": False},
    "gaussian_noise": {"enabled": False},
    "channel_dropout": {"enabled": False}}}})
add("e15_geometric_only", "E1.5", "flips + rot90 only", {"training": {"augmentation": {
    "continuous_rotation": {"enabled": False},
    "random_resized_crop": {"enabled": False},
    "brightness_jitter": {"enabled": False},
    "per_band_jitter": {"enabled": False},
    "gaussian_noise": {"enabled": False},
    "channel_dropout": {"enabled": False}}}})
add("e15_cutout_only", "E1.5", "production recipe + cutout as a single lever",
    {"training": {"augmentation": {"cutout": {
        "enabled": True, "probability": 0.5, "n_holes": 2, "hole_size": 16}}}})

# ================= WAVE 2 (2026-08-15): remaining training-config decisions ====

# -- E2.8 combination: do the wave-1 winners compose? Measured individually,
#    freeze0 was +5.9 sigma and 6-bands was a tie; additivity is NOT assumed
#    (softcon x ctx128 failed to stack), hence these explicit combo runs.
add("e28_combo2", "E2.8", "freeze0 + 6 bands (no indices)", {
    "model": {"freeze_backbone_epochs": 0, "input_channels": 6},
    "training": {"channel_subset": BANDS6}})
add("e28_combo3", "E2.8", "freeze0 + 6 bands + 3-class (candidate production recipe)", {
    "model": {"freeze_backbone_epochs": 0, "input_channels": 6,
              "num_classes": 3, "class_names": ["NotFarm", "Poultry", "OtherFarm"]},
    "training": {"channel_subset": BANDS6},
    "data": {"label_mode": "three_class",
             "candidates_dir": "data/rachel_geometry_candidates/candidates_world_v10_v9_threeclass"}})

# -- E1.4b regularisation / schedule knobs never tested
add("e14_wd0", "E1.4", "weight decay 0", {"training": {"weight_decay": 0.0}})
add("e14_wd01", "E1.4", "weight decay 0.1 (10x)", {"training": {"weight_decay": 0.1}})
add("e14_bs64", "E1.4", "batch size 64", {"training": {"batch_size": 64}})
add("e14_plateau", "E1.4", "ReduceLROnPlateau instead of cosine",
    {"training": {"scheduler": "plateau"}})

# -- E1.5b augmentation, finer grain: the two geometric extras individually,
#    plus the all-off anchor that bounds the whole stack's value.
add("e15_no_controt", "E1.5", "drop continuous rotation only",
    {"training": {"augmentation": {"continuous_rotation": {"enabled": False}}}})
add("e15_no_rrc", "E1.5", "drop random resized crop only",
    {"training": {"augmentation": {"random_resized_crop": {"enabled": False}}}})
add("e15_aug_off", "E1.5", "ALL augmentation disabled",
    {"training": {"augmentation": {"enabled": False}}})

# -- E1.8 TTA: inference-only on the existing production checkpoint (launch with
#    --steps inference). Keeps checkpoint/norm_stats_stem, unlike training runs.
add("e18_tta", "E1.8", "8-fold dihedral TTA on the production v9 checkpoint", {
    "inference": {"tta": True,
                  "checkpoint": "data/output/world_v10_fourclass_v9/best_model.pt",
                  "norm_stats_stem": "world_v10_fourclass_v9"}})

# Configs whose inference block must be preserved verbatim (scoring-only runs).
KEEP_INFERENCE = {"e18_tta"}


def deep_merge(base: dict, delta: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    for name, (exp, desc, deltas) in MATRIX.items():
        cfg = deep_merge(base, deltas)
        # Every run gets its own output dir / split file / norm stats via the stem.
        cfg["runpod"] = deep_merge(cfg.get("runpod", {}), {
            "gpu_type": GPU, "gpu_fallbacks": GPU_FALLBACKS, "auto_terminate": True,
        })
        # Score the labeled rows only: that covers every held-out slice including
        # the frozen blind benchmark, and skips ~123k unlabeled candidates.
        cfg["inference"] = deep_merge(cfg.get("inference", {}), {"labeled_only": True})
        if name not in KEEP_INFERENCE:
            cfg["inference"].pop("checkpoint", None)      # use this run's own best_model.pt
            cfg["inference"].pop("norm_stats_stem", None)  # use this run's own stats

        path = OUT / f"{name}.yaml"
        path.write_text(
            f"# {exp}: {desc}\n"
            f"# Generated by experiments/gen_configs.py from {BASE.relative_to(REPO)}\n"
            f"# Single-lever delta: {deltas}\n\n"
            + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        )
        written.append((exp, name, desc))

    print(f"wrote {len(written)} configs -> {OUT.relative_to(REPO)}\n")
    for exp, name, desc in sorted(written):
        print(f"  {exp:<6} {name:<22} {desc}")


if __name__ == "__main__":
    main()


def _selftest_startup_scripts() -> None:
    """Bash-parse the generated startup script for every experiment config.

    A malformed fragment kills the pod's script on line 1, and the pod then sits
    idle and billing with no error visible through the RunPod API. This check is
    cheap and catches that class of defect before any money is spent.
    Run: python3 experiments/gen_configs.py --selftest
    """
    import subprocess, sys, tempfile
    sys.path.insert(0, str(REPO))
    from training.config import load_config, resolve_paths
    from training import runpod_launch as rl

    bad = []
    for path in sorted(OUT.glob("*.yaml")):
        cfg = resolve_paths(load_config(str(path)))
        name = f"experiments/{path.name}"
        for steps in (["train", "inference"], ["candidates", "train", "inference"],
                      ["inference"]):
            script = rl._build_startup_script(cfg, name, steps=steps)
            with tempfile.NamedTemporaryFile("w", suffix=".sh") as fh:
                fh.write(script)
                fh.flush()
                r = subprocess.run(["bash", "-n", fh.name], capture_output=True, text=True)
            if r.returncode != 0:
                bad.append((path.name, steps, r.stderr.strip()[:200]))
    if bad:
        for n, s, e in bad:
            print(f"  SYNTAX ERROR {n} steps={s}: {e}")
        raise SystemExit(f"{len(bad)} startup scripts fail bash -n")
    print(f"startup-script self-test: {len(list(OUT.glob('*.yaml')))} configs x 3 step-sets OK")
