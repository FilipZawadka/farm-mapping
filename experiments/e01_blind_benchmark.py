"""E0.1 -- Freeze a blind benchmark and measure its detection power.

Two parts, both restricted to rows no model in the comparison trained on:

  (a) Four-class blind benchmark. qual_eval, further filtered to clusters that are
      spatially far from any training cluster (E0.2 filter). Measures achieved
      detection power and re-ranks v6..v9 with PAIRED bootstrap.

  (b) Three-class ctx128 vs softcon on the 1,072 labeled `unassigned` rows present
      in the archived releases. This is the locally-available stand-in for the
      3,297-row slice cited in the roadmap; that larger slice needs full-world
      3-class scores which are not in the local cache.

Composition skew is handled by reporting pooled AND country-macro-averaged
metrics; the roadmap's caveat is that pooled figures on these slices are
prior-skewed by one or two dominant countries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import lib

PATCH_M = 1280.0
WEB = lib.REPO / "web" / "public" / "data"


def country_macro_auc(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, min_n: int = 20) -> float:
    """Mean per-country AUC, so one dominant country cannot carry the metric."""
    aucs = []
    for c, idx in df.groupby("country").indices.items():
        if len(idx) < min_n:
            continue
        a = lib.safe_auc(y[idx], p[idx])
        if not np.isnan(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def part_a() -> dict:
    lib.header("E0.1a  Four-class blind benchmark (qual_eval, leakage-filtered)")
    v9 = lib.load(lib.FOURCLASS["v9"])
    lab9 = lib.labeled(v9)
    train_xy = lib.latlng(lab9[lab9.cnn_split_assigned == "train"])

    q = lab9[lab9.cnn_split_assigned == "qual_eval"].copy()
    q["dist_to_train_m"] = lib.nearest_distance_m(lib.latlng(q), train_xy)
    blind = q[q.dist_to_train_m >= PATCH_M].copy()
    print(f"qual_eval labeled: {len(q):,}  ->  blind (>= {PATCH_M:.0f} m from train): {len(blind):,}")
    print(f"countries: {blind.country.nunique()}   class mix: {blind.true_label.value_counts().to_dict()}")

    keep = set(blind.candidate_id)

    # detection power: CI half-width on this slice
    y, p = lib.farm_binary(blind)
    auc, ci = lib.bootstrap_ci(lib.safe_auc, y, p)
    half = (ci[1] - ci[0]) / 2
    mf, mf_ci = lib.bootstrap_ci(
        lambda yy, pp: float(f1_score(yy, (pp >= 0.5).astype(int), average="macro", zero_division=0)), y, p
    )
    print(f"\nfarm ROC-AUC {auc:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  half-width {half:.4f}")
    print(f"binary macro-F1 {mf:.4f} 95% CI [{mf_ci[0]:.4f}, {mf_ci[1]:.4f}] "
          f"half-width {(mf_ci[1]-mf_ci[0])/2:.4f}")

    # re-rank all four label rounds on the frozen slice, paired against v9
    print("\nPaired re-ranking on the frozen blind slice (delta vs v9, paired bootstrap):")
    ref = blind.set_index("candidate_id")
    y_ref = (ref["true_label"].to_numpy().astype(int) != 0).astype(int)
    p_v9 = 1.0 - ref["prob_class0"].to_numpy(dtype=float)

    ranking = []
    for name, run in lib.FOURCLASS.items():
        d = lib.labeled(lib.load(run))
        d = d[d.candidate_id.isin(keep)].drop_duplicates("candidate_id").set_index("candidate_id")
        d = d.reindex(ref.index)
        if d["prob_class0"].isna().any():
            print(f"  {name}: skipped (missing {int(d['prob_class0'].isna().sum())} rows)")
            continue
        p_m = 1.0 - d["prob_class0"].to_numpy(dtype=float)
        auc_m = lib.safe_auc(y_ref, p_m)
        cmacro = country_macro_auc(ref.reset_index(), y_ref, p_m)
        if name == "v9":
            print(f"  {name:<4} AUC={auc_m:.4f}  country-macro={cmacro:.4f}   (reference)")
            ranking.append({"model": name, "auc": auc_m, "country_macro_auc": cmacro,
                            "delta_vs_v9": 0.0, "ci": [0.0, 0.0], "p_value": 1.0})
            continue
        delta, dci, pval = lib.paired_bootstrap_delta(lib.safe_auc, y_ref, p_v9, p_m)
        sig = "yes" if (dci[0] > 0 or dci[1] < 0) else "no"
        print(f"  {name:<4} AUC={auc_m:.4f}  country-macro={cmacro:.4f}   "
              f"delta={delta:+.4f} [{dci[0]:+.4f},{dci[1]:+.4f}] p={pval:.3f}  separates={sig}")
        ranking.append({"model": name, "auc": auc_m, "country_macro_auc": cmacro,
                        "delta_vs_v9": delta, "ci": list(dci), "p_value": pval,
                        "separates": sig == "yes"})

    frozen = blind[["candidate_id", "country", "lat", "lng", "true_label", "dist_to_train_m"]].copy()
    out = lib.RESULTS / "e01_blind_benchmark_frozen.csv"
    frozen.to_csv(out, index=False)
    print(f"\nFrozen benchmark -> {out}  ({len(frozen):,} rows)")

    return {"n": len(blind), "countries": int(blind.country.nunique()),
            "auc": auc, "auc_ci": list(ci), "auc_ci_halfwidth": half,
            "macro_f1": mf, "macro_f1_ci": list(mf_ci),
            "ranking": ranking, "frozen_csv": str(out)}


def part_b() -> dict:
    lib.header("E0.1b  Three-class ctx128 vs softcon on the blind `unassigned` rows")
    a = pd.read_csv(WEB / "world_v9_softcon" / "points.csv", low_memory=False)
    b = pd.read_csv(WEB / "world_v9_ctx128" / "points.csv", low_memory=False)
    sel = lambda d: d[(d.split == "unassigned") & (d.true_label >= 0)]
    m = sel(a).merge(sel(b), on="id", suffixes=("_s", "_c"))
    assert (m.true_label_s == m.true_label_c).all()
    print(f"paired blind rows: {len(m):,}   countries: {m.country_s.nunique()}")
    print(f"class mix: {m.true_label_s.value_counts().to_dict()}")
    top = m.country_s.value_counts().head(3)
    print(f"composition skew -> top 3 countries = {100*top.sum()/len(m):.1f}% of rows: {top.to_dict()}")

    y = m.true_label_s.to_numpy().astype(int)
    P_s = m[["prob_class0_s", "prob_class1_s", "prob_class2_s"]].to_numpy(dtype=float)
    P_c = m[["prob_class0_c", "prob_class1_c", "prob_class2_c"]].to_numpy(dtype=float)

    def macro_f1_from_probs(yy, pred):
        return float(f1_score(yy, pred, average="macro", labels=[0, 1, 2], zero_division=0))

    f_s = macro_f1_from_probs(y, P_s.argmax(1))
    f_c = macro_f1_from_probs(y, P_c.argmax(1))
    print(f"\npooled macro-F1   softcon={f_s:.4f}   ctx128={f_c:.4f}   delta={f_c-f_s:+.4f}")

    # paired bootstrap on the macro-F1 difference
    rng = np.random.default_rng(lib.SEED)
    pred_s, pred_c = P_s.argmax(1), P_c.argmax(1)
    deltas = np.empty(lib.BOOT_N)
    for i in range(lib.BOOT_N):
        idx = rng.integers(0, len(y), len(y))
        deltas[i] = macro_f1_from_probs(y[idx], pred_c[idx]) - macro_f1_from_probs(y[idx], pred_s[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    pval = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    sep = lo > 0 or hi < 0
    print(f"paired delta 95% CI [{lo:+.4f}, {hi:+.4f}]  p={pval:.3f}  separates={'yes' if sep else 'no'}")

    # country-macro to neutralise the Australia-dominated composition
    rows = []
    for c, g in m.groupby("country_s"):
        if len(g) < 20:
            continue
        yy = g.true_label_s.to_numpy().astype(int)
        ps = g[["prob_class0_s", "prob_class1_s", "prob_class2_s"]].to_numpy(dtype=float).argmax(1)
        pc = g[["prob_class0_c", "prob_class1_c", "prob_class2_c"]].to_numpy(dtype=float).argmax(1)
        rows.append({"country": c, "n": len(g),
                     "softcon": macro_f1_from_probs(yy, ps), "ctx128": macro_f1_from_probs(yy, pc)})
    cdf = pd.DataFrame(rows)
    if not cdf.empty:
        print("\nper-country (n>=20):")
        print(cdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"country-macro   softcon={cdf.softcon.mean():.4f}   ctx128={cdf.ctx128.mean():.4f}   "
              f"delta={cdf.ctx128.mean()-cdf.softcon.mean():+.4f}")

    return {"n": len(m), "pooled": {"softcon": f_s, "ctx128": f_c, "delta": f_c - f_s,
                                     "ci": [float(lo), float(hi)], "p_value": float(pval),
                                     "separates": bool(sep)},
            "per_country": cdf.to_dict("records") if not cdf.empty else [],
            "country_macro": {"softcon": float(cdf.softcon.mean()) if not cdf.empty else None,
                              "ctx128": float(cdf.ctx128.mean()) if not cdf.empty else None}}


if __name__ == "__main__":
    res = {"part_a_fourclass": part_a(), "part_b_threeclass": part_b()}
    lib.save("e01_blind_benchmark", res)
