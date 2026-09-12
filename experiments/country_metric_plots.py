"""Figures for the reports produced by ``training/country_metrics.py``.

Re-creates the panels in Rachel Mason's CAFO-AI_v2 ``evaluate.py`` from a
report dict -- precision/recall-vs-threshold grids, TP/TN/FP/FN score
histograms, FP/FN-rate bars and the poultry-gate confusion matrices -- plus
two cross-run views of our own (mean-F1-vs-threshold curves per arm, and
per-country F1 by arm with seeds shown as points).

One deliberate difference: her PR-vs-threshold lines come from
``sklearn.precision_recall_curve`` (a knot at every distinct score); ours are
evaluated on the same 200-point grid the threshold search uses, so a figure
can be redrawn from the JSON alone. Visually identical at this resolution.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SPLIT_STYLES = {
    "test":           ("--", 0.4, 1.5, "EVAL1 (held-out test set)"),
    "eval":           ("-", 1.0, 1.6, "EVAL2 (representative sample)"),
    "generalization": ("-", 1.0, 1.6, "EVAL3 (generalization)"),
}
PREC_COLOR, REC_COLOR = "tab:blue", "tab:orange"
REFERENCE_LINE = 0.85


def _grid_axes(n: int, maxcols: int = 5, w: float = 4, h: float = 4, sharey: bool = True):
    nrows = max(1, math.ceil(n / maxcols))
    ncols = min(max(n, 1), maxcols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), sharey=sharey, squeeze=False)
    return fig, axes.flatten()


def _save(fig, fname):
    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _pr_legend(splits):
    elems = [Line2D([0], [0], color=PREC_COLOR, lw=2, label="precision"),
             Line2D([0], [0], color=REC_COLOR, lw=2, label="recall")]
    if len(splits) > 1:
        for sp in splits:
            style, _, lw, label = SPLIT_STYLES[sp]
            elems.append(Line2D([0], [0], color="gray", ls=style, lw=2, label=label))
    return elems


# ------------------------------------------------------------------ hers
def plot_pr_vs_threshold(entry: dict, countries, splits, fname, title=None, maxcols: int = 5):
    """Her ``_plot_pr_group``: one panel per country, precision and recall vs
    threshold, dotted guides at the chosen threshold and at 0.85."""
    grid = np.asarray(entry["mean_f1_curve"]["grid"])
    t = entry["threshold"]["mean_f1"]
    curves = entry["pr_curves"]
    countries = [c for c in countries if c in curves and any(sp in curves[c] for sp in splits)]
    if not countries:
        return None
    fig, axes = _grid_axes(len(countries), maxcols)
    for ax, c in zip(axes, countries):
        shown = None
        for sp in splits:
            cv = curves[c].get(sp)
            if not cv:
                continue
            style, alpha, lw, _ = SPLIT_STYLES[sp]
            ax.plot(grid, cv["precision"], color=PREC_COLOR, ls=style, alpha=alpha, lw=lw)
            ax.plot(grid, cv["recall"], color=REC_COLOR, ls=style, alpha=alpha, lw=lw)
            if style == "-":
                shown = cv
        ax.axvline(t, color="gray", ls=":", lw=1)
        ax.axhline(REFERENCE_LINE, color="gray", ls=":", lw=1)
        sub = f"  (n={shown['n']}, pos={shown['n_pos']})" if shown else ""
        ax.set_title(f"{c}{sub}")
        ax.set_xlabel("threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    for ax in axes[len(countries):]:
        ax.axis("off")
    axes[0].set_ylabel("score")
    axes[0].legend(handles=_pr_legend(splits), fontsize=8, loc="lower left")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _save(fig, fname)


def _draw_hist(ax, s, y_true, y_pred, density, t):
    bins = np.linspace(0, 1, 21)
    ax.hist(s[~y_true & ~y_pred], bins=bins, density=density, color="tab:blue", alpha=0.3, label="TN")
    ax.hist(s[y_true & y_pred], bins=bins, density=density, color="tab:green", alpha=0.3, label="TP")
    ax.hist(s[~y_true & y_pred], bins=bins, density=density, histtype="step", color="tab:red", lw=1.8, label="FP")
    ax.hist(s[y_true & ~y_pred], bins=bins, density=density, histtype="step", color="tab:orange", lw=1.8, label="FN")
    ax.axvline(t, color="gray", ls="--", lw=1)


def plot_score_hist(d, countries, score_col: str, pos, universe, t: float, fname,
                    mask=None, title=None, maxcols: int = 5):
    """Her ``_plot_hist_group``: per country, TP/TN filled and FP/FN outlined,
    counts on the top row and per-class densities below. ``mask`` restricts
    the rows (her version pools every labelled row, train included)."""
    g_all = d[d["truth"].isin(universe)]
    if mask is not None:
        g_all = g_all[mask.reindex(g_all.index).fillna(False).astype(bool)]
    countries = [c for c in countries if (g_all["ADM0"] == c).any()]
    if not countries:
        return None
    n = len(countries)
    ncols = min(n, maxcols)
    nblocks = (n + maxcols - 1) // maxcols
    fig, axes = plt.subplots(2 * nblocks, ncols, figsize=(4 * ncols, 3.5 * 2 * nblocks), sharex=True, squeeze=False)
    for i, c in enumerate(countries):
        block, col = divmod(i, maxcols)
        r0, r1 = 2 * block, 2 * block + 1
        g = g_all[g_all["ADM0"] == c]
        y_true = g["truth"].isin(pos).to_numpy()
        s = g[score_col].to_numpy()
        y_pred = s >= t
        _draw_hist(axes[r0, col], s, y_true, y_pred, False, t)
        _draw_hist(axes[r1, col], s, y_true, y_pred, True, t)
        axes[r0, col].set_title(f"{c} (n={len(g)})")
        axes[r1, col].set_xlabel(score_col)
        axes[r0, 0].set_ylabel("count")
        axes[r1, 0].set_ylabel("density (per class)")
    for i in range(n, nblocks * maxcols):
        block, col = divmod(i, maxcols)
        if col < ncols:
            axes[2 * block, col].axis("off")
            axes[2 * block + 1, col].axis("off")
    axes[0, 0].legend(fontsize=7)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return _save(fig, fname)


def plot_fp_fn_rates(rows, fname, min_labels: int = 5, title=None):
    """Her ``_plot_fp_fn_rates``: countries sorted by FP rate, FN rate overlaid,
    bars resting on fewer than ``min_labels`` labels drawn faded and every bar
    annotated with its label count."""
    rows = [r for r in rows if r["country"] != "TOTAL" and (r["n_neg"] > 0 or r["n_pos"] > 0)]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (r["FP_rate"] if r["FP_rate"] == r["FP_rate"] else -1.0))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(8, 0.16 * len(rows)), 5))

    def draw(rate_key, denom_key, width, strong, faded, label):
        first = True
        for xi, r in zip(x, rows):
            v = r[rate_key]
            if v != v:                       # NaN: denominator 0, bar skipped
                continue
            solid = r[denom_key] >= min_labels
            ax.bar(xi, v, width=width, color=strong if solid else faded,
                   label=label if (first and solid) else None)
            first = first and not solid
            ax.text(xi, v, str(int(r[denom_key])), ha="center", va="bottom", fontsize=6,
                    color="dimgray" if solid else "gray")

    draw("FP_rate", "n_neg", 0.8, "tab:red", "mistyrose", "FP rate")
    draw("FN_rate", "n_pos", 0.5, "tab:blue", "lightsteelblue", "FN rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["country"] for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("rate")
    ax.set_title(title or f"FP rate (sorted) with FN rate overlaid -- faded bars have < {min_labels} labels")
    ax.legend()
    fig.tight_layout()
    return _save(fig, fname)


def plot_confusion(conf: dict, fname, title=None):
    """Her ``class_cm``: per country, truth class x poultry gate."""
    countries = [c for c, v in conf.items() if v["n"] > 0]
    if not countries:
        return None
    fig, axes = plt.subplots(1, len(countries), figsize=(2.2 * len(countries), 3.6), sharey=True,
                             squeeze=False, gridspec_kw={"wspace": 0.05})
    for k, (ax, c) in enumerate(zip(axes[0], countries)):
        v = conf[c]
        m = np.asarray(v["matrix"])
        ax.imshow(m, cmap="Blues")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, m[i, j], ha="center", va="center", color="orange")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(v["cols"])
        ax.set_yticks(range(len(v["rows"])))
        if k == 0:
            ax.set_yticklabels(v["rows"])
        ax.set_xlabel(c)
    if title:
        fig.suptitle(title, y=1.02)
    return _save(fig, fname)


# ------------------------------------------------------------------ ours
def plot_mean_f1_curves(curves: dict, fname, title=None, ref_lines=(0.4,)):
    """[ours] Mean per-country F1 vs threshold for several models on one axis,
    each argmax marked. Flat curves mean the operating point hardly matters;
    a model whose peak sits far from the others needs its own threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cv in curves.items():
        grid = np.asarray(cv["grid"])
        y = np.asarray(cv["mean_f1"])
        line, = ax.plot(grid, y, lw=cv.get("lw", 1.6), alpha=cv.get("alpha", 1.0), label=label,
                        color=cv.get("color"))
        i = int(np.argmax(y))
        ax.plot(grid[i], y[i], "o", color=line.get_color(), ms=5)
    for r in ref_lines:
        ax.axvline(r, color="gray", ls=":", lw=1)
    ax.set_xlabel("threshold")
    ax.set_ylabel("mean per-country F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, fname)


def plot_country_f1_by_arm(points, fname, title=None, countries=None, arms=None, ref=None):
    """[ours] Per-country F1 (at each run's own mean-F1 threshold): one column
    per country, one colour per arm, a point per seed and a bar at the arm
    mean. ``ref`` = {country: f1} draws a grey reference marker (archived v9)."""
    countries = countries or sorted({p["country"] for p in points})
    arms = arms or sorted({p["arm"] for p in points})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(countries)), 5))
    width = 0.8 / max(len(arms), 1)
    for k, arm in enumerate(arms):
        color = cmap(k % 10)
        for i, c in enumerate(countries):
            vals = [p["f1"] for p in points if p["arm"] == arm and p["country"] == c]
            if not vals:
                continue
            x0 = i - 0.4 + width * (k + 0.5)
            ax.bar(x0, float(np.mean(vals)), width=width * 0.9, color=color, alpha=0.35,
                   label=arm if i == 0 else None)
            ax.scatter([x0] * len(vals), vals, color=color, s=14, zorder=3)
    if ref:
        for i, c in enumerate(countries):
            if c in ref:
                ax.hlines(ref[c], i - 0.4, i + 0.4, color="black", lw=1.2, ls="--",
                          label="reference" if i == 0 else None)
    ax.set_xticks(range(len(countries)))
    ax.set_xticklabels(countries)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("F1 at model's own mean-F1 threshold")
    ax.legend(fontsize=8, ncol=min(4, len(arms) + 1))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, fname)


# ------------------------------------------------------------------ per run
def render_run(report: dict, d, out_dir, name: str, heldout_mask=None) -> list:
    """All of her per-model figures for one run. ``d`` is the prepared frame
    the report was computed from (needed only for the histograms)."""
    out_dir = Path(out_dir)
    files = []
    focal = report["meta"]["focal_countries"]
    gen = report["meta"]["generalization_countries"]
    for task, e in report["tasks"].items():
        t = e["threshold"]["mean_f1"]
        f = plot_pr_vs_threshold(e, focal, ("test", "eval"), out_dir / f"score_thresh_{task}_eval.png",
                                 title=f"{name} -- {task}: precision/recall vs threshold, focal countries (t={t:.3f})")
        files.append(f)
        f = plot_pr_vs_threshold(e, gen, ("generalization",), out_dir / f"score_thresh_{task}_gen.png",
                                 title=f"{name} -- {task}: generalization countries (t={t:.3f})")
        files.append(f)
        if d is not None:
            for grp, cs in (("eval", focal), ("gen", gen)):
                f = plot_score_hist(d, cs, e["score_col"], e["positive_classes"], e["universe"], t,
                                    out_dir / f"prob_hist_{task}_{grp}.png", mask=heldout_mask,
                                    title=f"{name} -- {task} score histograms, held-out rows only")
                files.append(f)
        f = plot_fp_fn_rates(e["fp_fn_heldout"], out_dir / f"fp_fn_rates_{task}_heldout.png",
                             title=f"{name} -- {task}: FP rate (sorted) with FN rate overlaid, held-out rows, t={t:.3f}")
        files.append(f)
        if "confusion_eval" in e:
            f = plot_confusion(e["confusion_eval"], out_dir / f"class_cm_{task}_eval.png",
                               title=f"{name} -- poultry gate at t={t:.3f}, eval split")
            files.append(f)
    return [f for f in files if f is not None]
