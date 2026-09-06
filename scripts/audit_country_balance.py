#!/usr/bin/env python3
"""Audit the country x label composition of a training split, and simulate
what each region-balancing scheme (training/balancing.py) would do to it.

Why
---
The training pool is geographically lopsided in a way that a pooled class
count hides: most farm positives come from five registry-labelled countries
(USA above all), while the world-wide review pool that round_4 folded into
train contributes NotFarm rows from ~100 countries, many of them 100% NotFarm
(RUS, UKR, BLR, MYS, IND, TUR, ...). This script prints that structure and the
statistic that matters for shortcut learning -- how much of the label a model
could predict from the country alone -- before and after each sampler.

Inputs (one of)
---------------
  --config CFG      pipeline YAML: reads the candidate CSVs in data.candidates_dir
                    (cnn_split_assigned / label / ADM0) and, if enabled, the
                    training.region_balancing block
  --parquet PATH    Rachel master parquet (cluster_id, final_label,
                    cnn_split_assigned, ADM0); labels mapped four_class
  --scored PATH     a scored_candidates.parquet (candidate_id, true_label, split)

Usage
-----
  python scripts/audit_country_balance.py --config configs/rachel_clusters/world_v10_fourclass_r4.yaml --schemes all
  python scripts/audit_country_balance.py --parquet data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet
  python scripts/audit_country_balance.py --config configs/rachel_clusters/world_v10_fourclass_r4_g_s42.yaml

No torch needed; runs anywhere the candidate files are mounted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.balancing import (  # noqa: E402
    DEFAULT_BUCKETS,
    UNKNOWN_ISO3,
    assign_groups,
    compute_region_balanced_weights,
    derive_iso3,
    format_report,
    macro_region,
    mutual_information,
)
from training.config import RegionBalancingConfig  # noqa: E402

CLASS_NAMES = {0: "NotFarm", 1: "Poultry", 2: "Pigs", 3: "Cattle"}
# four_class mapping, mirrors training/rachel_to_candidates.py
MAP4 = {
    "NotFarm": 0,
    "Farm: Poultry: Meat Chickens": 1,
    "Farm: Poultry: Eggs": 1,
    "Farm: Poultry: Unspecified/Other": 1,
    "Farm: Pigs": 2,
    "Farm: Cattle": 3,
}

# The default simulation set: the three campaign arms plus the share-law-only
# ablation, so the printout shows what class conditioning adds.
DEFAULT_SCHEMES: dict[str, dict] = {
    "G grouped_country (uniform, class-conditional)": dict(
        scheme="grouped_country", min_country_rows=300, class_conditional=True),
    "H bucket us/europe/rest (uniform, class-conditional)": dict(
        scheme="bucket", buckets=DEFAULT_BUCKETS, class_conditional=True),
    "I capped 20%/country (class-conditional)": dict(
        scheme="capped", max_share=0.20, class_conditional=True),
    "J grouped_country WITHOUT class conditioning (ablation)": dict(
        scheme="grouped_country", min_country_rows=300, class_conditional=False),
}


# ---------------------------------------------------------------- loading
def _load_config_rows(cfg_path: Path) -> tuple[pd.DataFrame, RegionBalancingConfig | None]:
    from training.config import load_config, resolve_paths
    cfg = resolve_paths(load_config(cfg_path), root=REPO)
    cdir = Path(cfg.data.candidates_dir)
    files = sorted(cdir.glob("*.csv"))
    if cfg.data.countries:
        files = [cdir / f"{c}.csv" for c in cfg.data.countries if (cdir / f"{c}.csv").exists()]
    if not files:
        sys.exit(f"no candidate CSVs under {cdir} -- run the candidates step or point --parquet at the master file")
    cand = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    split_col = "cnn_split_assigned" if "cnn_split_assigned" in cand.columns else None
    rows = pd.DataFrame({
        "cid": cand["id"].astype(str),
        "label": cand["label"].fillna(-1).astype(int),
        "split": cand[split_col].fillna("").astype(str) if split_col else "train",
        "iso3": derive_iso3(cand["id"], cand),
    })
    rb = cfg.training.region_balancing if cfg.training.region_balancing.enabled else None
    return rows, rb


def _load_parquet_rows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    label = df["final_label"].map(MAP4).fillna(-1).astype(int) if "final_label" in df.columns else \
        df["label"].fillna(-1).astype(int)
    iso = df["ADM0"].astype(str).str.upper() if "ADM0" in df.columns else \
        df["cluster_id"].astype(str).str.extract(r"^([A-Z]{3})_", expand=False).fillna(UNKNOWN_ISO3)
    return pd.DataFrame({
        "cid": df["cluster_id"].astype(str),
        "label": label,
        "split": df["cnn_split_assigned"].fillna("").astype(str) if "cnn_split_assigned" in df.columns else "train",
        "iso3": iso,
    })


def _load_scored_rows(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    cand = df.rename(columns={"candidate_id": "id"})
    split_col = "cnn_split_assigned" if "cnn_split_assigned" in df.columns else "split"
    return pd.DataFrame({
        "cid": df["candidate_id"].astype(str),
        "label": df["true_label"].fillna(-1).astype(int),
        "split": df[split_col].fillna("").astype(str) if split_col in df.columns else "train",
        "iso3": derive_iso3(df["candidate_id"], cand),
    })


# ---------------------------------------------------------------- statistics
def region_only_accuracy(groups: np.ndarray, farm: np.ndarray, w: np.ndarray | None = None) -> float:
    """Accuracy of a classifier that predicts each group's majority farm/not-farm label.

    Under the natural distribution this is the accuracy a model reaches by
    recognising *where* a patch is rather than *what* it shows; under a
    sampler's achieved distribution it is what that shortcut would still buy.
    """
    w = np.ones(len(farm)) if w is None else np.asarray(w, dtype=float)
    d = pd.DataFrame({"g": groups, "y": farm.astype(int), "w": w})
    agg = d.groupby(["g", "y"])["w"].sum().unstack(fill_value=0.0)
    return float(agg.max(axis=1).sum() / w.sum())


def majority_accuracy(farm: np.ndarray, w: np.ndarray | None = None) -> float:
    """Accuracy of always predicting the (weighted) majority class -- the floor
    the region-only classifier must be compared against, since a sampler also
    moves the class marginal."""
    w = np.ones(len(farm)) if w is None else np.asarray(w, dtype=float)
    rate = float(w[farm.astype(bool)].sum() / w.sum())
    return max(rate, 1.0 - rate)


def composition_tables(rows: pd.DataFrame, top: int = 40) -> dict:
    lab = rows[rows.label >= 0].copy()
    lab["farm"] = (lab.label != 0).astype(int)
    lab["region"] = lab.iso3.map(macro_region)
    n = len(lab)
    out: dict = {"n_rows": int(n), "n_countries": int(lab.iso3.nunique())}

    cls = lab.label.value_counts().sort_index()
    out["class_counts"] = {CLASS_NAMES.get(int(c), str(c)): int(v) for c, v in cls.items()}
    out["farm_rate"] = round(float(lab.farm.mean()), 4)
    print(f"\nlabelled rows: {n:,} across {out['n_countries']} countries | classes: "
          + ", ".join(f"{k}={v:,}" for k, v in out["class_counts"].items())
          + f" | farm rate {out['farm_rate']:.3f}")

    ct = pd.crosstab(lab.iso3, lab.label).reindex(columns=[0, 1, 2, 3], fill_value=0)
    ct.columns = [CLASS_NAMES[c] for c in ct.columns]
    ct["n"] = ct.sum(axis=1)
    ct["share%"] = (100 * ct.n / n).round(2)
    ct["farm_rate"] = ((ct.n - ct.NotFarm) / ct.n).round(3)
    ct["pos%"] = (100 * (ct.n - ct.NotFarm) / max(1, (lab.farm == 1).sum())).round(1)
    ct["neg%"] = (100 * ct.NotFarm / max(1, (lab.farm == 0).sum())).round(1)
    ct["region"] = [macro_region(c) for c in ct.index]
    ct["single_label"] = ((ct.farm_rate == 0) | (ct.farm_rate == 1)).map({True: "*", False: ""})
    ct = ct.sort_values("n", ascending=False)
    print(f"\nper-country composition of the split (top {top} by rows; '*' = single-label country;"
          f" pos%/neg% = share of ALL positives / negatives):")
    print(ct.head(top).to_string())
    if len(ct) > top:
        rest = ct.iloc[top:]
        print(f"... {len(rest)} more countries with {int(rest.n.sum()):,} rows "
              f"({int(rest.NotFarm.sum()):,} NotFarm / {int(rest.n.sum() - rest.NotFarm.sum()):,} farm)")
    out["per_country"] = ct.reset_index().rename(columns={"iso3": "country"}).to_dict(orient="records")

    # concentration
    pos = lab[lab.farm == 1].iso3.value_counts()
    neg = lab[lab.farm == 0].iso3.value_counts()
    print("\nwhere the positives come from: "
          + ", ".join(f"{c} {100 * v / pos.sum():.1f}%" for c, v in pos.head(6).items()))
    print("where the negatives come from: "
          + ", ".join(f"{c} {100 * v / neg.sum():.1f}%" for c, v in neg.head(8).items()))
    single_neg = ct[(ct.farm_rate == 0)]
    single_pos = ct[(ct.farm_rate == 1)]
    print(f"single-label countries: {len(single_neg)} all-NotFarm ({int(single_neg.n.sum()):,} rows = "
          f"{100 * single_neg.NotFarm.sum() / max(1, neg.sum()):.1f}% of all negatives), "
          f"{len(single_pos)} all-farm ({int(single_pos.n.sum()):,} rows)")
    out["concentration"] = {
        "positives_top": {c: int(v) for c, v in pos.head(10).items()},
        "negatives_top": {c: int(v) for c, v in neg.head(10).items()},
        "all_notfarm_countries": int(len(single_neg)),
        "all_notfarm_rows": int(single_neg.n.sum()),
        "all_farm_countries": int(len(single_pos)),
    }

    # macro-region view
    rt = pd.crosstab(lab.region, lab.farm).reindex(columns=[0, 1], fill_value=0)
    rt.columns = ["NotFarm", "farm"]
    rt["n"] = rt.sum(axis=1)
    rt["farm_rate"] = (rt.farm / rt.n).round(3)
    rt = rt.sort_values("n", ascending=False)
    print("\nby macro-region:")
    print(rt.to_string())
    out["per_region"] = rt.reset_index().to_dict(orient="records")

    # dependence statistics
    def _joint(keys):
        return pd.crosstab(keys, lab.farm).to_numpy(dtype=float)
    mi_c, nmi_c = mutual_information(_joint(lab.iso3))
    mi_r, nmi_r = mutual_information(_joint(lab.region))
    majority = max(lab.farm.mean(), 1 - lab.farm.mean())
    acc_c = region_only_accuracy(lab.iso3.to_numpy(), lab.farm.to_numpy())
    acc_r = region_only_accuracy(lab.region.to_numpy(), lab.farm.to_numpy())
    print(f"\nlabel ~ region dependence (binary farm label):")
    print(f"  NMI(label; country) = {nmi_c:.3f}   NMI(label; macro-region) = {nmi_r:.3f}"
          f"   (0 = independent, 1 = region determines the label)")
    print(f"  accuracy of a country-only classifier = {acc_c:.3f}   macro-region-only = {acc_r:.3f}"
          f"   vs majority class = {majority:.3f}  (shortcut gain: country {acc_c - majority:+.3f}, "
          f"macro-region {acc_r - majority:+.3f})")
    out["dependence"] = {
        "nmi_country": round(nmi_c, 4), "nmi_region": round(nmi_r, 4),
        "mi_country_nats": round(mi_c, 4), "mi_region_nats": round(mi_r, 4),
        "acc_country_only": round(acc_c, 4), "acc_region_only": round(acc_r, 4),
        "acc_majority": round(float(majority), 4),
    }
    return out


def simulate(rows: pd.DataFrame, schemes: dict[str, dict]) -> dict:
    lab = rows[rows.label >= 0]
    iso = lab.iso3.to_numpy()
    labels = lab.label.to_numpy()
    farm = labels != 0
    out = {}
    for name, kw in schemes.items():
        rb = kw if isinstance(kw, RegionBalancingConfig) else RegionBalancingConfig(enabled=True, **kw)
        w, rep = compute_region_balanced_weights(iso, labels, rb)
        groups = assign_groups(iso, rb)
        # Shortcut gain = what knowing the group buys over the majority class,
        # measured under the same distribution (natural vs the sampler's).
        gain_nat = region_only_accuracy(groups, farm) - majority_accuracy(farm)
        gain_ach = region_only_accuracy(groups, farm, w) - majority_accuracy(farm, w)
        gain_c_nat = region_only_accuracy(iso, farm) - majority_accuracy(farm)
        gain_c_ach = region_only_accuracy(iso, farm, w) - majority_accuracy(farm, w)
        print("\n" + "=" * 100 + f"\n{name}\n" + "=" * 100)
        print(format_report(rep))
        print(f"  shortcut gain (region-only classifier minus majority class): "
              f"by group natural {gain_nat:+.3f} -> under sampler {gain_ach:+.3f}; "
              f"by country natural {gain_c_nat:+.3f} -> under sampler {gain_c_ach:+.3f}")
        rep["shortcut_gain"] = {"group_natural": round(gain_nat, 4), "group_achieved": round(gain_ach, 4),
                                "country_natural": round(gain_c_nat, 4), "country_achieved": round(gain_c_ach, 4)}
        out[name] = rep
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config")
    src.add_argument("--parquet")
    src.add_argument("--scored")
    ap.add_argument("--split", default="train", help="split to audit (default train)")
    ap.add_argument("--schemes", default="config",
                    help="'config' = the config's region_balancing block if enabled, "
                         "'all' = the four default schemes, 'none' = composition only")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--json", help="write tables + simulation reports to this JSON file")
    args = ap.parse_args()

    rb_cfg = None
    if args.config:
        rows, rb_cfg = _load_config_rows(Path(args.config))
    elif args.parquet:
        rows = _load_parquet_rows(Path(args.parquet))
    else:
        rows = _load_scored_rows(Path(args.scored))

    avail = rows.split.value_counts().to_dict()
    print(f"rows loaded: {len(rows):,} | splits: {avail}")
    sel = rows[rows.split == args.split]
    if sel.empty:
        sys.exit(f"no rows with split == {args.split!r}")
    n_unknown = int((sel.iso3 == UNKNOWN_ISO3).sum())
    if n_unknown:
        print(f"! {n_unknown} rows have no resolvable country (grouped as {UNKNOWN_ISO3})")
    print(f"\n### split = {args.split}: {len(sel):,} rows")
    payload = {"split": args.split, "composition": composition_tables(sel, top=args.top)}

    schemes: dict[str, dict] = {}
    if args.schemes == "all":
        schemes = dict(DEFAULT_SCHEMES)
    elif args.schemes == "config":
        if rb_cfg is not None:
            schemes = {f"config: {rb_cfg.scheme}": rb_cfg}
        elif args.config:
            print("\n(config has region_balancing disabled; pass --schemes all to simulate the defaults)")
    if schemes:
        payload["simulations"] = simulate(sel, schemes)

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
