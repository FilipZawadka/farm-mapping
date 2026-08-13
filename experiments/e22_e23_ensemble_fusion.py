"""E2.2 (ensembling) and E2.3 (geometry fusion), both judged on the E0.1 frozen
blind benchmark plus the out-of-domain slice.

E2.2 decision rule: ship only if the paired delta separates from zero.
E2.3 decision rule: ship only if the OOD gain separates from zero. Geometry is
     evaluated OOD-first because the in-domain probe previously showed +0.004,
     while the hypothesis is that morphometry transfers where imagery does not.

Both blends are fitted with grouped cross-validation by country, so a country's
blend weight is never fitted on that country's own rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import lib

PATCH_M = 1280.0
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (blind, ood) frames carrying v8/v9 farm probabilities and IF score."""
    v9 = lib.load(lib.FOURCLASS["v9"])
    v8 = lib.load(lib.FOURCLASS["v8"])
    lab9 = lib.labeled(v9)
    train_xy = lib.latlng(lab9[lab9.cnn_split_assigned == "train"])

    v8p = lib.labeled(v8)[["candidate_id", "prob_class0"]].drop_duplicates("candidate_id")
    v8p = v8p.rename(columns={"prob_class0": "prob0_v8"})

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        d = df.merge(v8p, on="candidate_id", how="inner").copy()
        d["p_v9"] = 1.0 - d["prob_class0"]
        d["p_v8"] = 1.0 - d["prob0_v8"]
        d["y"] = (d["true_label"].astype(int) != 0).astype(int)
        return d

    q = lab9[lab9.cnn_split_assigned == "qual_eval"].copy()
    q["dist_to_train_m"] = lib.nearest_distance_m(lib.latlng(q), train_xy)
    blind = prep(q[q.dist_to_train_m >= PATCH_M])
    ood = prep(lab9[lab9.cnn_split_assigned == "generalization"])
    return blind, ood


def grouped_cv_blend(df: pd.DataFrame, col_a: str, col_b: str) -> tuple[np.ndarray, list]:
    """Leave-one-country-out blend: p = (1-w)*a + w*b, w fitted off-country."""
    out = np.empty(len(df), dtype=float)
    picks = []
    idx_of = {c: i for i, c in enumerate(df.index)}
    for c, g in df.groupby("country"):
        rest = df[df.country != c]
        if rest["y"].nunique() < 2:
            w = 0.0
        else:
            best_w, best_auc = 0.0, -np.inf
            for w in WEIGHTS:
                blend = (1 - w) * rest[col_a] + w * rest[col_b]
                a = lib.safe_auc(rest["y"].to_numpy(), blend.to_numpy())
                if not np.isnan(a) and a > best_auc:
                    best_auc, best_w = a, w
            w = best_w
        picks.append({"country": c, "n": len(g), "weight": float(w)})
        pos = [idx_of[i] for i in g.index]
        out[pos] = ((1 - w) * g[col_a] + w * g[col_b]).to_numpy()
    return out, picks


def evaluate(name: str, y: np.ndarray, base: np.ndarray, cand: np.ndarray) -> dict:
    a_base = lib.safe_auc(y, base)
    a_cand = lib.safe_auc(y, cand)
    delta, ci, pval = lib.paired_bootstrap_delta(lib.safe_auc, y, base, cand)
    sep = ci[0] > 0 or ci[1] < 0
    print(f"  {name:<28} base={a_base:.4f}  cand={a_cand:.4f}  "
          f"delta={delta:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] p={pval:.3f}  "
          f"{'SEPARATES' if sep else 'tie'}")
    return {"slice": name, "auc_base": a_base, "auc_cand": a_cand,
            "delta": delta, "ci": list(ci), "p_value": pval, "separates": bool(sep)}


def main() -> None:
    blind, ood = build_frames()

    # ------------------------------------------------------------------ E2.2
    lib.header("E2.2  v8+v9 probability ensemble vs v9 alone")
    print(f"blind n={len(blind):,}   ood n={len(ood):,}")
    e22 = []
    for name, d in [("blind_benchmark", blind), ("generalization", ood)]:
        y = d["y"].to_numpy()
        ens = 0.5 * (d["p_v8"].to_numpy() + d["p_v9"].to_numpy())
        e22.append(evaluate(name, y, d["p_v9"].to_numpy(), ens))

    # Two-part rule. The frozen benchmark is powerful enough that trivial effects
    # reach significance, so statistical separation alone is not sufficient:
    # doubling inference cost needs a materially useful gain as well.
    MIN_USEFUL_AUC = 0.005
    sep22 = any(r["separates"] and r["delta"] > 0 for r in e22)
    useful22 = any(r["separates"] and r["delta"] >= MIN_USEFUL_AUC for r in e22)
    print(f"\n  statistically separating: {sep22}   "
          f"exceeds {MIN_USEFUL_AUC:.3f} AUC practical floor: {useful22}")
    ship22 = sep22 and useful22
    print(f"DECISION: {'SHIP' if ship22 else 'DO NOT SHIP'} the ensemble -- "
          f"{'gain justifies 2x inference' if ship22 else 'gain is real but too small to justify 2x inference cost'}")

    # ------------------------------------------------------------------ E2.3
    lib.header("E2.3  Geometry fusion (Isolation Forest score + CNN)")
    e23 = []
    for name, d in [("blind_benchmark", blind), ("generalization", ood)]:
        g = d[d["template_score_if"].notna()].copy()
        cov = len(g) / len(d) if len(d) else 0.0
        print(f"\n  {name}: IF-score coverage {len(g):,}/{len(d):,} ({cov:.1%})")
        if len(g) < 100 or g["y"].nunique() < 2:
            print("    skipped (insufficient coverage)")
            e23.append({"slice": name, "skipped": True, "coverage": cov, "n": len(g)})
            continue
        # normalise IF score to [0,1] so the blend weight is interpretable
        s = g["template_score_if"].to_numpy(dtype=float)
        g["if_norm"] = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
        y = g["y"].to_numpy()
        print(f"    IF alone AUC={lib.safe_auc(y, g['if_norm'].to_numpy()):.4f}   "
              f"CNN alone AUC={lib.safe_auc(y, g['p_v9'].to_numpy()):.4f}")
        blended, picks = grouped_cv_blend(g, "p_v9", "if_norm")
        r = evaluate(f"{name} (LOCO blend)", y, g["p_v9"].to_numpy(), blended)
        r["coverage"] = cov
        r["weights"] = picks[:10]
        r["mean_weight"] = float(np.mean([p["weight"] for p in picks]))
        print(f"    mean LOCO IF weight = {r['mean_weight']:.2f}")
        e23.append(r)

    ship23 = any(r.get("separates") and r.get("delta", 0) >= MIN_USEFUL_AUC for r in e23)
    ood_tested = not any(r["slice"].startswith("generalization") and r.get("skipped") for r in e23)
    print(f"\nDECISION: {'SHIP' if ship23 else 'DO NOT SHIP'} geometry fusion")
    if not ood_tested:
        print("  CAVEAT: the actual hypothesis (geometry transfers OOD) is UNTESTED --")
        print("  Isolation Forest scores cover too few out-of-domain rows to evaluate.")

    lib.save("e22_e23_ensemble_fusion", {
        "practical_floor_auc": MIN_USEFUL_AUC,
        "e22_ensemble": {"results": e22, "separates": bool(sep22),
                         "exceeds_practical_floor": bool(useful22), "ship": bool(ship22)},
        "e23_geometry_fusion": {"results": e23, "ship": bool(ship23),
                                "ood_hypothesis_tested": bool(ood_tested)},
    })


if __name__ == "__main__":
    main()
