"""Unit tests for training/balancing.py -- the region-balanced sampler.

No torch, no patches, no Earth Engine: the module is pure numpy/pandas.
Run with:  python tests/test_balancing.py      (also collectable by pytest)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.balancing import (  # noqa: E402
    DEFAULT_BUCKETS,
    MACRO_REGION,
    UNKNOWN_ISO3,
    assign_buckets,
    assign_groups,
    class_family,
    compute_region_balanced_weights,
    derive_iso3,
    expand_bucket_members,
    format_report,
    macro_region,
    mutual_information,
    target_group_shares,
    water_fill_cap,
)
from training.config import RegionBalancingConfig, TrainingConfig, load_config  # noqa: E402

logging.basicConfig(level=logging.WARNING)


# Approximate round_4 train composition, reconstructed from the repo record
# (docs/EXPERIMENTS_LOG.md, experiments/results/e01_blind_benchmark_frozen.csv):
# country -> (NotFarm rows, farm rows). Whole countries are single-label.
R4_LIKE = {
    "USA": (1400, 5600), "MEX": (130, 1500), "THA": (170, 420), "BRA": (150, 320),
    "CHL": (60, 270), "RUS": (2300, 0), "DEU": (660, 110), "UKR": (610, 0),
    "AUS": (30, 510), "POL": (240, 250), "FRA": (170, 270), "ITA": (280, 120),
    "BLR": (330, 0), "CZE": (190, 60), "MYS": (190, 0), "ROU": (35, 150),
    "IND": (180, 0), "TUR": (160, 0), "GBR": (150, 0), "KAZ": (120, 0),
    "ARG": (30, 65), "EGY": (90, 0), "ESP": (90, 0), "HUN": (35, 50), "ZAF": (80, 2),
    "JPN": (60, 0), "CAN": (48, 0), "KEN": (38, 0), "VNM": (33, 0), "CUB": (24, 0),
    "CHN": (23, 0), "PER": (5, 3), "AFG": (2, 0), "ETH": (1, 0),
}


def synthetic(spec=R4_LIKE):
    iso, lab = [], []
    for c, (n0, n1) in spec.items():
        iso += [c] * (n0 + n1)
        farm = [1] * n1
        # sprinkle Pigs / Cattle into the farm rows so the 4-class path is exercised
        for k in range(n1):
            if k % 40 == 0:
                farm[k] = 3
            elif k % 7 == 0:
                farm[k] = 2
        lab += [0] * n0 + farm
    return np.array(iso), np.array(lab)


def rb(**kw) -> RegionBalancingConfig:
    return RegionBalancingConfig(enabled=True, **kw)


def _shares(iso, w, groups):
    s = pd.Series(w).groupby(groups).sum()
    return s / s.sum()


# ---------------------------------------------------------------- resolution
def test_macro_regions():
    assert macro_region("RUS") == "europe"
    assert macro_region("TUR") == "asia"
    assert macro_region("usa") == "north_america"
    assert macro_region("MEX") == "latin_america"
    assert macro_region("NGA") == "africa"
    assert macro_region("AUS") == "oceania"
    assert macro_region("XXX") == "unknown"
    assert len(MACRO_REGION) > 240


def test_derive_iso3_resolution_order():
    cands = pd.DataFrame({
        "id": ["RUS_cluster_1", "cand_a", "cand_b", "cand_c", "cand_d", "ARG_cluster_9", "cand_e"],
        "ADM0": ["RUS", None, None, None, None, None, "nan"],
        "country_key": ["rus", "united_states", "afg", "", "", "", ""],
        "country": ["RUS", "United States", "AFG", "Brazil", "nowhere", "", ""],
    })
    got = derive_iso3(cands["id"], cands)
    assert list(got) == ["RUS", "USA", "AFG", "BRA", UNKNOWN_ISO3, "ARG", UNKNOWN_ISO3]
    # ids absent from candidates resolve by prefix only
    got = derive_iso3(["POL_cluster_5", "no_prefix"], cands)
    assert list(got) == ["POL", UNKNOWN_ISO3]


# ------------------------------------------------------------------ grouping
def test_expand_bucket_members():
    isos, catch = expand_bucket_members(["EUROPE", "tur"])
    assert catch is False and "RUS" in isos and "DEU" in isos and "TUR" in isos
    isos, catch = expand_bucket_members(["*"])
    assert catch is True and isos == set()
    try:
        expand_bucket_members(["Europe!"])
    except ValueError:
        pass
    else:
        raise AssertionError("bad token accepted")


def test_assign_buckets_default_and_custom():
    iso = np.array(["USA", "RUS", "DEU", "THA", "NGA", "XXX"])
    got = assign_buckets(iso, DEFAULT_BUCKETS)
    assert list(got) == ["us", "europe", "europe", "rest", "rest", "rest"]
    # first bucket listing a code wins; explicit RUS-in-rest override
    got = assign_buckets(iso, {"rest": ["RUS", "*"], "us": ["USA"], "europe": ["EUROPE"]})
    assert list(got) == ["us", "rest", "europe", "rest", "rest", "rest"]
    # no catch-all -> 'other'
    got = assign_buckets(iso, {"us": ["USA"]})
    assert list(got) == ["us", "other", "other", "other", "other", "other"]
    try:
        assign_buckets(iso, {"a": ["*"], "b": ["*"]})
    except ValueError:
        pass
    else:
        raise AssertionError("two catch-alls accepted")


def test_grouped_country_pools_small_countries():
    iso, lab = synthetic()
    groups = assign_groups(iso, rb(scheme="grouped_country", min_country_rows=300))
    counts = pd.Series(iso).value_counts()
    for c, n in counts.items():
        g = set(groups[iso == c])
        assert len(g) == 1
        if n >= 300:
            assert g == {c}, (c, g)
        else:
            assert g == {f"rest_{macro_region(c)}"}, (c, g)
    assert "rest_europe" in set(groups) and "rest_asia" in set(groups)
    # capped: every country is its own group
    assert set(assign_groups(iso, rb(scheme="capped"))) == set(iso)


# ---------------------------------------------------------------- share laws
def test_water_fill_cap():
    p = pd.Series([0.5, 0.3, 0.1, 0.1], index=list("abcd"))
    q = water_fill_cap(p, 0.35)
    assert abs(q.sum() - 1) < 1e-12
    assert np.allclose(q.values, [0.35, 0.35, 0.15, 0.15])
    # single cap binding
    q = water_fill_cap(p, 0.45)
    assert abs(q["a"] - 0.45) < 1e-12 and abs(q.sum() - 1) < 1e-12
    assert np.allclose(q[["b", "c", "d"]].values, np.array([0.3, 0.1, 0.1]) * 0.55 / 0.5)
    # identity when nothing exceeds the cap or cap >= 1
    assert np.allclose(water_fill_cap(p, 0.6).values, p.values)
    assert np.allclose(water_fill_cap(p, 1.0).values, p.values)
    # infeasible: 4 groups cannot each stay <= 0.2 and sum to 1
    try:
        water_fill_cap(p, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible cap accepted")
    # a cascade: capping the first pushes the second over the cap too
    p = pd.Series([0.6, 0.25, 0.15])
    q = water_fill_cap(p, 0.35)
    assert np.allclose(q.values, [0.35, 0.35, 0.30]) and abs(q.sum() - 1) < 1e-12


def test_target_shares_uniform_and_tempered():
    counts = pd.Series({"a": 8000, "b": 2000, "c": 500, "d": 100})
    u = target_group_shares(counts, rb(scheme="grouped_country"))
    assert np.allclose(u.values, 0.25)
    nat = target_group_shares(counts, rb(scheme="bucket", temperature=1.0))
    assert np.allclose(nat.values, counts.values / counts.sum())
    sq = target_group_shares(counts, rb(scheme="grouped_country", temperature=2.0))
    expect = np.sqrt(counts.values / counts.sum())
    assert np.allclose(sq.values, expect / expect.sum())
    hot = target_group_shares(counts, rb(scheme="grouped_country", temperature=1e6))
    assert np.allclose(hot.values, 0.25, atol=1e-4)
    capped = target_group_shares(counts, rb(scheme="capped", max_share=0.4))
    assert capped["a"] <= 0.4 + 1e-12 and abs(capped.sum() - 1) < 1e-12


# --------------------------------------------------------------- information
def test_mutual_information():
    # independent -> 0; deterministic -> NMI 1
    mi, nmi = mutual_information(np.array([[0.25, 0.25], [0.25, 0.25]]))
    assert abs(mi) < 1e-12 and abs(nmi) < 1e-12
    mi, nmi = mutual_information(np.array([[0.5, 0.0], [0.0, 0.5]]))
    assert abs(mi - np.log(2)) < 1e-12 and abs(nmi - 1) < 1e-12


def test_class_family():
    lab = np.array([0, 1, 2, 3, -1])
    assert list(class_family(lab, "binary")) == [0, 1, 1, 1, -1]
    assert list(class_family(lab, "full")) == [0, 1, 2, 3, -1]


# ------------------------------------------------------------------- weights
def test_class_conditional_makes_label_independent_of_group():
    # Every group has both classes here, and no clipping, so the target is
    # exactly reachable: uniform group shares and the global farm rate inside
    # every group => NMI(label; group) == 0.
    spec = {"USA": (1000, 4000), "RUS": (2000, 100), "POL": (300, 300), "THA": (200, 400)}
    iso, lab = synthetic(spec)
    cfg = rb(scheme="grouped_country", min_country_rows=1, class_conditional=True, max_weight=1e9)
    w, rep = compute_region_balanced_weights(iso, lab, cfg)
    assert abs(w.mean() - 1) < 1e-9
    groups = assign_groups(iso, cfg)
    shares = _shares(iso, w, groups)
    assert np.allclose(shares.values, 0.25)
    farm = (lab != 0)
    global_rate = farm.mean()
    for g in shares.index:
        m = groups == g
        rate = w[m & farm].sum() / w[m].sum()
        assert abs(rate - global_rate) < 1e-9, (g, rate, global_rate)
    assert rep["label_region_dependence"]["nmi_achieved"] < 1e-9
    assert rep["label_region_dependence"]["nmi_natural"] > 0.05
    # class marginal preserved (report values are rounded to 4 decimals)
    assert abs(rep["class_marginal"]["achieved"][1] - global_rate) < 1e-3
    assert rep["n_clipped"] == 0


def test_rake_and_marginal_preservation_with_single_label_groups():
    from training.balancing import rake
    seed = np.array([[0.2, 0.3], [0.5, 0.0]])          # second row: structural zero
    Q, resid = rake(seed, np.array([0.5, 0.5]), np.array([0.6, 0.4]))
    assert resid < 1e-9
    assert np.allclose(Q.sum(axis=1), [0.5, 0.5]) and np.allclose(Q.sum(axis=0), [0.6, 0.4])
    assert Q[1, 1] == 0.0
    # infeasible: row 2 must carry 0.5 but column 1 allows only 0.4
    _, resid = rake(seed, np.array([0.5, 0.5]), np.array([0.4, 0.6]), iters=50)
    assert resid > 1e-3

    # Real case: RUS / UKR are all-NotFarm. With class conditioning the
    # sampled class prior must still equal the natural one (no v9_bal-style
    # prior shift); the single-label groups can only claim p(NotFarm) of their
    # nominal (uniform) share, the mixable groups share the rest equally.
    spec = {"USA": (1000, 4000), "RUS": (2000, 0), "UKR": (600, 0), "POL": (300, 300), "THA": (200, 400)}
    iso, lab = synthetic(spec)
    cfg = rb(scheme="grouped_country", min_country_rows=1, class_conditional=True, max_weight=1e9)
    w, rep = compute_region_balanced_weights(iso, lab, cfg)
    farm = lab != 0
    assert abs(w[farm].sum() / w.sum() - farm.mean()) < 1e-6
    shares = _shares(iso, w, iso)
    p_notfarm = 1 - farm.mean()
    expect_mix = 1 / (3 + 2 * p_notfarm)
    assert np.allclose([shares[c] for c in ("USA", "POL", "THA")], expect_mix)
    assert np.allclose([shares[c] for c in ("RUS", "UKR")], expect_mix * p_notfarm)
    assert rep["groups"]["RUS"]["nominal_share"] == 0.2
    assert abs(rep["groups"]["RUS"]["target_share"] - expect_mix * p_notfarm) < 1e-4
    # the mixable groups compensate, but far less than with full single-label shares
    for c in ("USA", "POL", "THA"):
        m = iso == c
        rate = w[m & farm].sum() / w[m].sum()
        assert farm.mean() < rate < 0.8, (c, rate)
    dep = rep["label_region_dependence"]
    assert dep["nmi_achieved"] < dep["nmi_natural"], dep
    # the capped scheme keeps the cap on the FINAL shares
    cfg = rb(scheme="capped", max_share=0.3, class_conditional=True, max_weight=1e9)
    w, rep = compute_region_balanced_weights(iso, lab, cfg)
    assert _shares(iso, w, iso).max() <= 0.3 + 1e-9
    assert abs(w[farm].sum() / w.sum() - farm.mean()) < 1e-6


def test_class_conditional_off_keeps_each_groups_mix():
    iso, lab = synthetic()
    cfg = rb(scheme="bucket", class_conditional=False, max_weight=1e9)
    w, rep = compute_region_balanced_weights(iso, lab, cfg)
    groups = assign_buckets(iso, DEFAULT_BUCKETS)
    farm = lab != 0
    for g in ("us", "europe", "rest"):
        m = groups == g
        assert abs(w[m & farm].sum() / w[m].sum() - farm[m].mean()) < 1e-9
        assert abs(rep["groups"][g]["achieved_share"] - 1 / 3) < 1e-4
    # bucket balancing alone leaves the region->label dependence intact
    assert rep["label_region_dependence"]["nmi_achieved"] > 0.05


def test_capped_scheme_only_shrinks_dominant_countries():
    iso, lab = synthetic()
    cfg = rb(scheme="capped", max_share=0.2, class_conditional=False, max_weight=1e9)
    w, rep = compute_region_balanced_weights(iso, lab, cfg)
    shares = _shares(iso, w, iso)
    natural = pd.Series(iso).value_counts(normalize=True)
    assert shares.max() <= 0.2 + 1e-9
    assert abs(shares["USA"] - 0.2) < 1e-9
    # every uncapped country gains a little, proportionally
    for c in shares.index:
        if natural[c] < 0.2:
            assert shares[c] >= natural[c] - 1e-12
    ratios = {c: shares[c] / natural[c] for c in shares.index if shares[c] < 0.2 - 1e-9}
    assert max(ratios.values()) - min(ratios.values()) < 1e-9
    # tiny countries are NOT repeated: the largest weight is the pro-rata
    # redistribution factor (1 - cap) / (1 - share of the capped country)
    assert abs(rep["weight_max"] - (1 - 0.2) / (1 - natural["USA"])) < 1e-3


def test_clipping_bounds_and_report():
    iso, lab = synthetic()
    cfg = rb(scheme="grouped_country", min_country_rows=300, class_conditional=True, max_weight=10.0)
    w, rep = compute_region_balanced_weights(iso, lab, cfg)
    assert abs(w.mean() - 1) < 1e-6
    assert w.min() >= 0.1 - 1e-9 and w.max() <= 10.0 + 1e-9
    assert rep["n_clipped"] > 0 and 0 < rep["clipped_frac"] < 1
    assert 0 < rep["ess_ratio"] <= 1
    assert rep["n_groups"] == len(set(assign_groups(iso, cfg)))
    # single-label countries can only be down-weighted, never fixed
    assert rep["groups"]["RUS"]["natural_farm_rate"] == 0.0
    assert rep["groups"]["RUS"]["achieved_farm_rate"] == 0.0
    assert rep["groups"]["RUS"]["achieved_share"] < rep["groups"]["RUS"]["natural_share"]
    # the pooled groups list their members
    pooled = [g for g in rep["groups"] if g.startswith("rest_")]
    assert pooled and all("members" in rep["groups"][g] for g in pooled)
    # dependence is reduced even though clipping keeps it above zero
    dep = rep["label_region_dependence"]
    assert dep["nmi_achieved"] < dep["nmi_natural"]
    json.dumps(rep)  # JSON-serialisable
    text = format_report(rep)
    assert "scheme=grouped_country" in text and "RUS" in text


def test_unlabeled_rows_get_zero_weight():
    iso = np.array(["USA", "USA", "RUS", "RUS"])
    lab = np.array([1, 0, 0, -1])
    w, rep = compute_region_balanced_weights(iso, lab, rb(scheme="bucket", max_weight=1e9))
    assert w[3] == 0.0 and rep["n_valid"] == 3 and rep["n_rows"] == 4
    assert abs(w[:3].mean() - 1) < 1e-9


def test_full_class_axis():
    iso, lab = synthetic({"USA": (500, 2000), "POL": (400, 200)})
    cfg = rb(scheme="bucket", class_axis="full", max_weight=1e9)
    w, rep = compute_region_balanced_weights(iso, lab, cfg)
    assert set(rep["class_marginal"]["natural"]) == {0, 1, 2, 3}
    for g in ("us", "europe"):
        m = assign_buckets(iso, DEFAULT_BUCKETS) == g
        for c in (0, 1, 2, 3):
            share = w[m & (lab == c)].sum() / w[m].sum()
            assert abs(share - (lab == c).mean()) < 1e-9


# -------------------------------------------------------------------- config
def test_config_defaults_and_validation():
    t = TrainingConfig()
    assert t.region_balancing.enabled is False
    t = TrainingConfig(region_balancing={"enabled": True, "scheme": "capped", "max_share": 0.15})
    assert t.region_balancing.scheme == "capped" and t.region_balancing.max_share == 0.15
    for bad in ({"scheme": "nope"}, {"max_share": 0.0}, {"max_weight": 0.5}, {"temperature": 0.0}):
        try:
            TrainingConfig(region_balancing=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    try:
        TrainingConfig(upsample_minority_regions=True, region_balancing={"enabled": True})
    except ValueError:
        pass
    else:
        raise AssertionError("both region samplers accepted")
    # the round_4 anchor config is untouched by the new block
    cfg = load_config(ROOT / "configs" / "rachel_clusters" / "world_v10_fourclass_r4.yaml")
    assert cfg.training.region_balancing.enabled is False
    assert cfg.training.upsample_minority_regions is False


def main() -> None:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  ok  {name}")
    print(f"\n{len(tests)} balancing tests passed")


if __name__ == "__main__":
    main()
