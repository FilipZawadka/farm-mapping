"""Per-country, threshold-based evaluation of a scored candidate table.

This is a port of the evaluation Rachel Mason runs on our full-world prediction
CSVs in CAFO-AI_v2 (``evaluate.py`` dated 2026-09-06, driven by
``05_review-cnn-predictions-{1..4}.ipynb``; a snapshot of that repo lives in
``notebooks_rachel/``, gitignored). Producing the same numbers inside the
pipeline means a training run can be judged the way the delivered model is
judged, without a round-trip through Colab.

What she reports as most useful (2026-09):

* precision and recall **as a function of threshold**, per country, for each
  model -- the five focal countries on their ``eval`` split (``test`` drawn
  faintly alongside), the generalization countries on ``generalization``;
  the held-out countries IDN / MOZ / PER are excluded;
* precision, recall and F1 at **one global threshold per model**, chosen to
  maximise the **unweighted mean of per-country F1**: F1 is computed for each
  country at every candidate threshold, averaged across countries (each
  country counts once, however many rows it has) and the argmax is taken.

Two tasks are scored, exactly as in her notebook:

``farm``
    score = P(any farm) = sum of the non-NotFarm class probabilities.
    Positives are every ``Farm: *`` label, including the ones the model cannot
    name (Unknown / Mixed / Other / PigsOrPoultry, collapsed to
    ``GenericFarm``). ``Ambiguous`` and unlabelled rows are outside the
    universe and ignored.
``poultry``
    score = P(Poultry). Positives are the three poultry labels; the universe
    is the specific animal classes {NotFarm, Poultry, Pigs, Cattle}, so a
    generic farm is neither a hit nor a miss for the poultry question.

Everything here is plain pandas / numpy / scikit-learn so it can run on a pod
at the end of scoring (``training/inference.py`` calls :func:`write_report`)
and on a laptop over archived parquets
(``experiments/evaluate_country_metrics.py``). Plotting lives in
``experiments/country_metric_plots.py`` so this module never imports
matplotlib.

Additions beyond her notebook are marked ``[ours]`` and kept in separate keys,
so her numbers stay hers:

* a leave-one-country-out threshold -- the in-sample argmax is optimistic,
  because the threshold is tuned on the very rows it is then scored on;
* a held-out-only FP/FN table -- hers pools every labelled row, train
  included, which is fine for reviewing inference countries but not for
  judging a model.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants mirrored from CAFO-AI_v2/config.py (Rachel Mason, 2026-09-09).
# --------------------------------------------------------------------------
FOCAL_COUNTRIES = ("USA", "BRA", "CHL", "MEX", "THA")
GENERALIZATION_COUNTRIES = ("ALB", "BGD", "COD", "IND", "MAR", "NGA")
HELD_OUT_COUNTRIES = ("IDN", "MOZ", "PER")

# Collapse the 11 dataset labels into the six truth categories she scores.
LABEL_MAPPING = {
    "NotFarm": "NotFarm",
    "Farm: Poultry: Meat Chickens": "Poultry",
    "Farm: Poultry: Eggs": "Poultry",
    "Farm: Poultry: Unspecified/Other": "Poultry",
    "Farm: Pigs": "Pigs",
    "Farm: Cattle": "Cattle",
    "Farm: PigsOrPoultry": "GenericFarm",
    "Farm: Unknown": "GenericFarm",
    "Farm: Mixed": "GenericFarm",
    "Farm: Other": "GenericFarm",
    "Ambiguous": "Ambiguous",
}
FARM_TRUTH = ("Poultry", "Pigs", "Cattle", "GenericFarm")   # farm-task positives
POULTRY_UNIVERSE = ("NotFarm", "Poultry", "Pigs", "Cattle")  # poultry-task universe
CONFUSION_ROWS = POULTRY_UNIVERSE
DEFAULT_CLASS_NAMES = ("NotFarm", "Poultry", "Pigs", "Cattle")

GRID_N = 200                      # her n_grid; thresholds are `score >= t`
PR_REFERENCE_LINE = 0.85          # horizontal guide she draws on every PR panel
PLATEAU_TOL = (0.01, 0.02)        # [ours] "within tol of the best mean F1"
PRECISION_FLOOR = 0.8             # [ours] her by-eye rule from the first review notebook
FIXED_THRESHOLDS = (0.4, 0.5, 0.75)  # [ours] shipped OOD point (E2.1), default, her first hand-pick
TRAIN_SPLITS = ("train", "val")
UNLABELED_SPLITS = ("predict", "unlabeled", "unknown", "", "nan", "None")
TASKS = ("farm", "poultry")


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame, class_names=None) -> pd.DataFrame:
    """Return a copy with the columns every function below relies on.

    Adds ``cluster_id`` (str), ``ADM0`` (ISO3), ``split`` (Rachel's
    ``cnn_split_assigned`` where present, else the pipeline ``split``),
    ``truth`` (collapsed label, NaN when unlabelled) and one ``prob_<Class>``
    column per model class plus ``prob_Farm``. The class names used are
    stored in ``out.attrs["class_names"]``.
    """
    d = df.copy()
    if "geometry" in d.columns:
        d = d.drop(columns=["geometry"])

    if "cluster_id" in d.columns:
        d["cluster_id"] = d["cluster_id"].astype(str)
    elif "candidate_id" in d.columns:
        d["cluster_id"] = d["candidate_id"].astype(str)
    else:
        raise KeyError("scored table needs cluster_id or candidate_id")

    if "ADM0" not in d.columns:
        if "country" not in d.columns:
            raise KeyError("scored table needs ADM0 or country")
        d["ADM0"] = d["country"]
    d["ADM0"] = d["ADM0"].astype(str)

    split = pd.Series("unknown", index=d.index, dtype=object)
    if "split" in d.columns:
        split = d["split"].astype(str).str.strip().replace({"": "unknown", "nan": "unknown"})
    if "cnn_split_assigned" in d.columns:
        explicit = d["cnn_split_assigned"].astype(str).str.strip()
        has = ~explicit.isin(("", "nan", "None"))
        split = split.where(~has, explicit)
    d["split"] = split

    if "final_label" not in d.columns:
        raise KeyError("scored table needs final_label")
    lab = d["final_label"].astype(object).map(
        lambda x: None if x is None or (isinstance(x, float) and np.isnan(x))
        or str(x).strip() in ("", "nan", "None") else str(x).strip())
    d["truth"] = lab.map(LABEL_MAPPING)
    unmapped = sorted(set(lab.dropna()) - set(LABEL_MAPPING))
    if unmapped:
        log.warning("country_metrics: %d unmapped final_label values ignored: %s",
                    len(unmapped), unmapped[:8])

    names = list(class_names or DEFAULT_CLASS_NAMES)
    pcols = sorted([c for c in d.columns if c.startswith("prob_class")],
                   key=lambda c: int(c.replace("prob_class", "")))
    if pcols:
        if len(pcols) != len(names):
            raise ValueError(f"{len(pcols)} prob_class columns but {len(names)} class names {names}")
        for c, n in zip(pcols, names):
            d[f"prob_{n}"] = d[c].astype(float)
    elif any(f"prob_{n}" in d.columns for n in names):
        pass                                             # already renamed
    elif "predicted_score" in d.columns:                 # legacy binary model
        names = ["NotFarm", "Farm"]
        d["prob_Farm"] = d["predicted_score"].astype(float)
        d["prob_NotFarm"] = 1.0 - d["prob_Farm"]
    else:
        raise ValueError("no prob_class* / prob_<Class> / predicted_score columns")

    farm_classes = [n for n in names if n != "NotFarm"]
    d["prob_Farm"] = d[[f"prob_{n}" for n in farm_classes]].sum(axis=1)
    d.attrs["class_names"] = names
    return d


def task_spec(task: str, class_names) -> tuple[str, list[str], list[str]] | None:
    """(score column, positive truth classes, scoreable universe) for a task.

    Returns None when the model has no class that can answer the task
    (e.g. ``poultry`` for a binary farm/not-farm model).
    """
    if task == "farm":
        pos = list(FARM_TRUTH)
        return "prob_Farm", pos, pos + ["NotFarm"]
    if task == "poultry":
        if "Poultry" not in class_names:
            return None
        return "prob_Poultry", ["Poultry"], list(POULTRY_UNIVERSE)
    raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")


def reporting_split(gc: pd.DataFrame) -> str:
    """Her rule: a country reports on ``eval`` if it has any, else ``generalization``."""
    return "eval" if (gc["split"] == "eval").any() else "generalization"


def country_parts(d: pd.DataFrame, countries, universe) -> dict[str, tuple[pd.DataFrame, str]]:
    """country -> (eligible rows of its reporting split, split name)."""
    parts = {}
    for c in countries:
        gc = d[d["ADM0"] == c]
        if gc.empty:
            continue
        sp = reporting_split(gc)
        g = gc[(gc["split"] == sp) & gc["truth"].isin(universe)]
        if not g.empty:
            parts[c] = (g, sp)
    return parts


def heldout_mask(d: pd.DataFrame) -> pd.Series:
    """[ours] rows in a labelled split the model was not fitted on."""
    return ~d["split"].isin(TRAIN_SPLITS) & ~d["split"].isin(UNLABELED_SPLITS)


# --------------------------------------------------------------------------
# Core arithmetic
# --------------------------------------------------------------------------
def default_grid(n: int = GRID_N) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def f1_curve(scores, y, grid: np.ndarray):
    """F1 / precision / recall of ``score >= t`` for every t in ``grid``.

    Matches ``sklearn.metrics.f1_score(..., zero_division=0)`` at each t, but
    vectorised (n x len(grid)) so a 200-point sweep over eleven countries is
    instantaneous.
    """
    s = np.asarray(scores, dtype=float)[:, None]
    yb = np.asarray(y, dtype=bool)[:, None]
    pred = s >= grid[None, :]
    tp = (pred & yb).sum(0)
    fp = (pred & ~yb).sum(0)
    fn = (~pred & yb).sum(0)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return f1, prec, rec


def metrics_at(g: pd.DataFrame, score_col: str, pos, t: float) -> dict:
    """Confusion counts and rates for one group at one threshold."""
    y = g["truth"].isin(pos).to_numpy()
    p = g[score_col].to_numpy() >= t
    tp = int((y & p).sum()); fp = int((~y & p).sum())
    fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    n_pos, n_neg = tp + fn, fp + tn
    return {
        "n": int(len(g)), "n_pos": n_pos, "n_neg": n_neg,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / n_pos if n_pos else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "fp_rate": fp / n_neg if n_neg else float("nan"),
        "fn_rate": fn / n_pos if n_pos else float("nan"),
    }


def choose_threshold(parts: dict, score_col: str, pos, mode: str = "mean",
                     grid: np.ndarray | None = None) -> tuple[float, dict]:
    """One global threshold, her two ways.

    ``mean``   -- argmax over a shared grid of the unweighted mean per-country
                  F1 (every country weighted equally). Her choice.
    ``pooled`` -- argmax F1 of the pooled rows (big countries dominate).
    Returns (threshold, details); for ``mean`` the details carry the full
    curve so it can be plotted or compared across models.
    """
    grid = default_grid() if grid is None else grid
    if not parts:
        raise ValueError("no eligible rows in any country")
    if mode == "mean":
        curves = {c: f1_curve(g[score_col], g["truth"].isin(pos), grid)[0]
                  for c, (g, _) in parts.items()}
        mean_f1 = np.vstack(list(curves.values())).mean(axis=0)
        i = int(np.argmax(mean_f1))                      # first max, like np/hers
        return float(grid[i]), {"grid": grid, "mean_f1": mean_f1, "per_country": curves,
                                "best_mean_f1": float(mean_f1[i])}
    if mode == "pooled":
        pool = pd.concat([g for g, _ in parts.values()], ignore_index=True)
        y = pool["truth"].isin(pos).astype(int).to_numpy()
        prec, rec, thr = precision_recall_curve(y, pool[score_col].to_numpy())
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
        i = int(np.argmax(f1))
        return float(thr[i]), {"best_pooled_f1": float(f1[i])}
    if mode == "mean_pfloor":
        # [ours] Her criterion with a precision floor: the argmax of mean
        # per-country F1 restricted to thresholds where the unweighted mean
        # per-country precision is at least PRECISION_FLOOR. On eval sets that
        # are 70-90 % farms, plain F1 is maximised by accepting nearly
        # everything; this is the "precision >= 0.8" rule she applied by eye
        # in the first review notebook, made explicit. Falls back to the
        # threshold with the highest mean precision when no t clears the floor.
        curves = {c: f1_curve(g[score_col], g["truth"].isin(pos), grid) for c, (g, _) in parts.items()}
        mean_f1 = np.vstack([v[0] for v in curves.values()]).mean(axis=0)
        mean_p = np.vstack([v[1] for v in curves.values()]).mean(axis=0)
        ok = np.flatnonzero(mean_p >= PRECISION_FLOOR)
        if len(ok):
            i = int(ok[np.argmax(mean_f1[ok])])
            met = True
        else:
            i = int(np.argmax(mean_p))
            met = False
        return float(grid[i]), {"mean_f1": float(mean_f1[i]), "mean_precision": float(mean_p[i]),
                                "floor": PRECISION_FLOOR, "floor_met": met}
    raise ValueError("mode must be 'mean', 'pooled' or 'mean_pfloor'")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def pr_table(parts: dict, score_col: str, pos, t: float) -> list[dict]:
    """Her per-country score table at one threshold (plus the raw counts)."""
    rows = []
    for c, (g, sp) in parts.items():
        m = metrics_at(g, score_col, pos, t)
        rows.append({"country": c, "split": sp, "f_pos": m["n_pos"] / m["n"] if m["n"] else 0.0, **m})
    return rows


def loco_table(parts: dict, score_col: str, pos, grid: np.ndarray | None = None) -> list[dict]:
    """[ours] Each country scored at the threshold that maximises the mean F1
    of the *other* countries -- what a global threshold does on a country it
    was not tuned on."""
    grid = default_grid() if grid is None else grid
    names = list(parts)
    if len(names) < 2:
        return []
    M = np.vstack([f1_curve(parts[c][0][score_col], parts[c][0]["truth"].isin(pos), grid)[0]
                   for c in names])
    rows = []
    for i, c in enumerate(names):
        t_c = float(grid[int(np.argmax(np.delete(M, i, axis=0).mean(axis=0)))])
        rows.append({"country": c, "split": parts[c][1], "threshold": t_c,
                     **metrics_at(parts[c][0], score_col, pos, t_c)})
    return rows


def summarise_table(rows: list[dict]) -> dict:
    """Unweighted means across countries plus the pooled figures."""
    if not rows:
        return {}
    f1s = np.array([r["f1"] for r in rows])
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); fn = sum(r["fn"] for r in rows)
    worst = min(rows, key=lambda r: r["f1"])
    return {
        "n_countries": len(rows),
        "mean_f1": float(f1s.mean()),
        "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_recall": float(np.mean([r["recall"] for r in rows])),
        "min_f1": float(worst["f1"]), "min_f1_country": worst["country"],
        "pooled_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pooled_recall": tp / (tp + fn) if tp + fn else 0.0,
        "pooled_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "n_rows": int(sum(r["n"] for r in rows)),
    }


def fp_fn_summary(d: pd.DataFrame, score_col: str, pos, universe, t: float,
                  mask: pd.Series | None = None) -> list[dict]:
    """Her ``find_fp_fn`` summary: per country FP / n_neg / FN / n_pos and the
    two rates, sorted by FP count, with a TOTAL row last. ``mask`` restricts
    the rows (e.g. :func:`heldout_mask`); None pools every labelled row like
    her notebook does."""
    g = d[d["truth"].isin(universe)]
    if mask is not None:
        g = g[mask.reindex(g.index).fillna(False).astype(bool)]
    rows = []
    for c, gc in g.groupby("ADM0", sort=True):
        m = metrics_at(gc, score_col, pos, t)
        rows.append({"country": c, "FP": m["fp"], "n_neg": m["n_neg"], "FN": m["fn"], "n_pos": m["n_pos"],
                     "FP_rate": m["fp_rate"], "FN_rate": m["fn_rate"]})
    rows.sort(key=lambda r: (-r["FP"], r["country"]))
    if rows:
        FP = sum(r["FP"] for r in rows); n_neg = sum(r["n_neg"] for r in rows)
        FN = sum(r["FN"] for r in rows); n_pos = sum(r["n_pos"] for r in rows)
        rows.append({"country": "TOTAL", "FP": FP, "n_neg": n_neg, "FN": FN, "n_pos": n_pos,
                     "FP_rate": FP / n_neg if n_neg else float("nan"),
                     "FN_rate": FN / n_pos if n_pos else float("nan")})
    return rows


def class_confusion(d: pd.DataFrame, countries, splits, t: float,
                    score_col: str = "prob_Poultry", rows=CONFUSION_ROWS) -> dict:
    """Her ``class_cm``: per country, truth class (rows) x gate ``score >= t``
    (columns) -- shows what the poultry gate lets through from each class."""
    name = score_col.split("_", 1)[1]
    out = {}
    for c in countries:
        g = d[(d["ADM0"] == c) & d["truth"].isin(rows)]
        if splits is not None:
            g = g[g["split"].isin(splits)]
        gate = (g[score_col] >= t).astype(int)
        cm = pd.crosstab(g["truth"], gate).reindex(index=list(rows), columns=[0, 1], fill_value=0)
        out[c] = {"rows": list(rows), "cols": [f"not {name}", name],
                  "matrix": cm.to_numpy().tolist(), "n": int(len(g))}
    return out


def pr_curves(d: pd.DataFrame, countries, score_col: str, pos, universe,
              splits=("test", "eval", "generalization"), grid: np.ndarray | None = None) -> dict:
    """Precision and recall vs threshold on the shared grid, per country and
    split -- the data behind her PR-vs-threshold panels, stored so the plots
    can be redrawn from the JSON alone."""
    grid = default_grid() if grid is None else grid
    out = {}
    for c in countries:
        gc = d[(d["ADM0"] == c) & d["truth"].isin(universe)]
        for sp in splits:
            g = gc[gc["split"] == sp]
            if len(g) < 2:
                continue
            _, prec, rec = f1_curve(g[score_col], g["truth"].isin(pos), grid)
            out.setdefault(c, {})[sp] = {"n": int(len(g)), "n_pos": int(g["truth"].isin(pos).sum()),
                                         "precision": prec.round(4).tolist(),
                                         "recall": rec.round(4).tolist()}
    return out


def eval_set_table(d: pd.DataFrame, countries) -> list[dict]:
    """Her ``eval_set_table``: what each country's reporting split contains,
    expressed as the two task universes (her column names in brackets)."""
    rows = []
    for c in countries:
        gc = d[d["ADM0"] == c]
        if gc.empty:
            continue
        sp = reporting_split(gc)
        g = gc[gc["split"] == sp]
        labelled = g[g["truth"].notna()]
        farm_universe = labelled["truth"].isin(list(FARM_TRUTH) + ["NotFarm"]).sum()
        poultry_universe = labelled["truth"].isin(POULTRY_UNIVERSE).sum()
        rows.append({
            "country": c, "split": sp,
            "n_trainval": int(gc["split"].isin(TRAIN_SPLITS).sum()),
            "n_labelled": int(len(labelled)),                          # [Total clusters in evaluation set]
            "n_ambiguous": int((labelled["truth"] == "Ambiguous").sum()),
            "n_farm_task": int(farm_universe),                         # [Farm evaluation]
            "n_generic_farm": int(farm_universe - poultry_universe),   # [Unresolved Animal]
            "n_poultry_task": int(poultry_universe),                   # [Poultry evaluation]
            "n_farm_pos": int(labelled["truth"].isin(FARM_TRUTH).sum()),
            "n_poultry_pos": int((labelled["truth"] == "Poultry").sum()),
        })
    return rows


def cluster_fractions(d: pd.DataFrame, countries) -> list[dict]:
    """Her ``cluster_fractions``: label mix of each reporting split."""
    rows = []
    for c in countries:
        gc = d[d["ADM0"] == c]
        if gc.empty:
            continue
        g = gc[gc["split"] == reporting_split(gc)]
        lab = g["final_label"].astype(str)
        n = int(g["truth"].notna().sum())
        if n == 0:
            continue
        is_farm = lab.str.contains("Farm: ", na=False)
        is_poultry = lab.str.contains("Poultry", na=False)
        is_pigs = lab == "Farm: Pigs"
        is_cattle = lab == "Farm: Cattle"
        rows.append({
            "country": c, "n": n,
            "f_Poultry": float(is_poultry.sum() / n), "f_Pigs": float(is_pigs.sum() / n),
            "f_Cattle": float(is_cattle.sum() / n),
            "f_UnresolvedAnimals": float((is_farm & ~is_poultry & ~is_pigs & ~is_cattle).sum() / n),
            "f_Ambiguous": float((lab == "Ambiguous").sum() / n),
            "f_Farm": float(is_farm.sum() / n),
        })
    return rows


def headline_counts(d: pd.DataFrame, thresholds: dict) -> dict:
    """Her closing cell: how many candidates clear each gate, overall and
    among the unlabelled rows the map will show."""
    unl = d["truth"].isna()
    out = {"n_candidates": int(len(d)), "n_unlabelled": int(unl.sum())}
    for task, t in thresholds.items():
        col = "prob_Farm" if task == "farm" else f"prob_{task.capitalize()}"
        if col not in d.columns:
            continue
        hit = d[col] >= t
        out[f"n_{task}_above"] = int(hit.sum())
        out[f"n_unlabelled_{task}_above"] = int((hit & unl).sum())
        out[f"{task}_threshold"] = float(t)
    return out


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------
def full_report(df: pd.DataFrame, class_names=None, focal=FOCAL_COUNTRIES,
                generalization=GENERALIZATION_COUNTRIES, exclude=HELD_OUT_COUNTRIES,
                grid_n: int = GRID_N, tasks=TASKS) -> dict:
    """Everything her review notebook computes, as one JSON-able dict."""
    d = prepare(df, class_names)
    names = d.attrs["class_names"]
    grid = default_grid(grid_n)
    present = set(d["ADM0"].unique())
    focal_p = [c for c in focal if c in present and c not in exclude]
    gen_p = [c for c in generalization if c in present and c not in exclude]
    countries = focal_p + gen_p
    ho = heldout_mask(d)

    report = {
        "meta": {
            "source": "port of Rachel Mason, CAFO-AI_v2/evaluate.py (2026-09-06) + 05_review-cnn-predictions-4.ipynb",
            "class_names": names, "grid_n": grid_n, "rule": "score >= t",
            "focal_countries": focal_p, "generalization_countries": gen_p,
            "excluded_countries": [c for c in exclude if c in present],
            "n_rows": int(len(d)), "n_labelled": int(d["truth"].notna().sum()),
            "n_heldout_labelled": int((ho & d["truth"].notna()).sum()),
            "split_counts": {k: int(v) for k, v in d["split"].value_counts().items()},
        },
        "eval_sets": {"composition": eval_set_table(d, countries),
                      "fractions": cluster_fractions(d, countries)},
        "tasks": {},
    }
    thresholds = {}
    for task in tasks:
        spec = task_spec(task, names)
        if spec is None:
            continue
        score_col, pos, universe = spec
        parts = country_parts(d, countries, universe)
        if not parts:
            log.warning("country_metrics: no eligible rows for task %s", task)
            continue
        t_mean, curve = choose_threshold(parts, score_col, pos, "mean", grid)
        t_pool, pooled = choose_threshold(parts, score_col, pos, "pooled", grid)
        t_pf, pfloor = choose_threshold(parts, score_col, pos, "mean_pfloor", grid)
        table = pr_table(parts, score_col, pos, t_mean)
        table_pf = pr_table(parts, score_col, pos, t_pf)
        loco = loco_table(parts, score_col, pos, grid)
        j_pool = int(np.argmin(np.abs(grid - t_pool)))
        # [ours] How much the operating point matters: the span of thresholds
        # whose mean F1 is within `tol` of the best, and the mean F1 at a few
        # fixed thresholds. On positive-heavy eval sets the argmax can sit
        # very low (accept nearly everything) while the curve is almost flat
        # up to 0.5+, and this is the only way to see that from the numbers.
        plateau = {}
        for tol in PLATEAU_TOL:
            ok = np.flatnonzero(curve["mean_f1"] >= curve["best_mean_f1"] - tol)
            plateau[str(tol)] = [float(grid[ok.min()]), float(grid[ok.max()])]
        at_fixed = {str(t): float(curve["mean_f1"][int(np.argmin(np.abs(grid - t)))]) for t in FIXED_THRESHOLDS}
        entry = {
            "score_col": score_col, "positive_classes": pos, "universe": universe,
            "countries": list(parts),
            "threshold": {
                "mean_f1": t_mean, "pooled_f1": t_pool,
                "best_mean_f1": curve["best_mean_f1"],
                "best_pooled_f1": pooled["best_pooled_f1"],
                "mean_f1_at_pooled_t": float(curve["mean_f1"][j_pool]),
                "plateau": plateau,                # [ours] {tol: [t_lo, t_hi]}
                "mean_f1_at_fixed": at_fixed,      # [ours] {t: mean F1}
                "precision_floor": {"t": t_pf, **pfloor},   # [ours]
            },
            "per_country_pfloor": table_pf,               # [ours]
            "summary_pfloor": summarise_table(table_pf),  # [ours]
            "mean_f1_curve": {"grid": grid.round(4).tolist(),
                              "mean_f1": curve["mean_f1"].round(4).tolist(),
                              "per_country": {c: v.round(4).tolist() for c, v in curve["per_country"].items()}},
            "per_country": table,
            "summary": summarise_table(table),
            "per_country_loco": loco,                     # [ours]
            "summary_loco": summarise_table(loco),        # [ours]
            "pr_curves": pr_curves(d, countries, score_col, pos, universe, grid=grid),
            "fp_fn_heldout": fp_fn_summary(d, score_col, pos, universe, t_mean, mask=ho),   # [ours]
            "fp_fn_all_labelled": fp_fn_summary(d, score_col, pos, universe, t_mean),       # hers
        }
        if task == "poultry":
            entry["confusion_eval"] = class_confusion(d, focal_p, ["eval"], t_mean, score_col)
        report["tasks"][task] = entry
        thresholds[task] = t_mean
    report["headline"] = headline_counts(d, thresholds)
    return report


def flat_summary(report: dict) -> dict:
    """The scalars worth putting side by side across runs."""
    out = {"n_rows": report["meta"]["n_rows"], "n_labelled": report["meta"]["n_labelled"]}
    for task, e in report.get("tasks", {}).items():
        s, sl, th = e["summary"], e.get("summary_loco", {}), e["threshold"]
        out[f"{task}_t_mean"] = th["mean_f1"]
        out[f"{task}_t_pooled"] = th["pooled_f1"]
        out[f"{task}_mean_f1"] = s.get("mean_f1")
        out[f"{task}_mean_precision"] = s.get("mean_precision")
        out[f"{task}_mean_recall"] = s.get("mean_recall")
        out[f"{task}_min_f1"] = s.get("min_f1")
        out[f"{task}_min_f1_country"] = s.get("min_f1_country")
        out[f"{task}_pooled_f1"] = s.get("pooled_f1")
        out[f"{task}_mean_f1_loco"] = sl.get("mean_f1")
        for tol, span in th.get("plateau", {}).items():
            out[f"{task}_plateau_{tol}"] = span
        pf, spf = th.get("precision_floor", {}), e.get("summary_pfloor", {})
        if pf:
            out[f"{task}_t_pfloor"] = pf["t"]
            out[f"{task}_pfloor_met"] = pf["floor_met"]
            out[f"{task}_mean_f1_pfloor"] = spf.get("mean_f1")
            out[f"{task}_mean_precision_pfloor"] = spf.get("mean_precision")
            out[f"{task}_mean_recall_pfloor"] = spf.get("mean_recall")
            out[f"{task}_f1_by_country_pfloor"] = {r["country"]: r["f1"] for r in e.get("per_country_pfloor", [])}
        for t, v in th.get("mean_f1_at_fixed", {}).items():
            out[f"{task}_mean_f1_at_{t}"] = v
        out[f"{task}_f1_by_country"] = {r["country"]: r["f1"] for r in e["per_country"]}
        out[f"{task}_precision_by_country"] = {r["country"]: r["precision"] for r in e["per_country"]}
        out[f"{task}_recall_by_country"] = {r["country"]: r["recall"] for r in e["per_country"]}
        tot = next((r for r in e["fp_fn_heldout"] if r["country"] == "TOTAL"), None)
        if tot:
            out[f"{task}_heldout_FP_rate"] = tot["FP_rate"]
            out[f"{task}_heldout_FN_rate"] = tot["FN_rate"]
            out[f"{task}_heldout_n"] = tot["n_neg"] + tot["n_pos"]
    out.update({k: v for k, v in report.get("headline", {}).items()})
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_report(df: pd.DataFrame, out_dir, class_names=None,
                 fname: str = "country_threshold_metrics.json") -> Path:
    """Compute :func:`full_report` and write it next to the scored parquet."""
    report = full_report(df, class_names=class_names)
    path = Path(out_dir) / fname
    path.write_text(json.dumps(report, indent=1, default=_jsonable))
    return path


def format_summary(report: dict, ndigits: int = 3) -> str:
    """Compact text rendering for logs: threshold + per-country P/R/F1 per task."""
    lines = []
    for task, e in report.get("tasks", {}).items():
        th, s = e["threshold"], e["summary"]
        lines.append(f"[{task}] threshold(mean-F1)={th['mean_f1']:.3f} (pooled {th['pooled_f1']:.3f})  "
                     f"mean F1={s['mean_f1']:.3f} P={s['mean_precision']:.3f} R={s['mean_recall']:.3f}  "
                     f"min F1={s['min_f1']:.3f} ({s['min_f1_country']})")
        for r in e["per_country"]:
            lines.append(f"    {r['country']:<4} {r['split']:<15} n={r['n']:>5} f_pos={r['f_pos']:.2f}  "
                         f"P={r['precision']:.{ndigits}f} R={r['recall']:.{ndigits}f} F1={r['f1']:.{ndigits}f}")
    h = report.get("headline", {})
    if h:
        lines.append("headline: " + ", ".join(f"{k}={v}" for k, v in h.items()))
    return "\n".join(lines)
