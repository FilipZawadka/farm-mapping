"""Region-balanced sampling weights for the training DataLoader.

Why this exists
---------------
The round_4 training set (~21.5k rows) has a two-sided geographic confound:
nearly every farm positive comes from a handful of registry-labelled countries
(USA, MEX, THA, BRA, CHL), while a large share of the NotFarm rows come from
the world-wide review pool that was folded into train (RUS, UKR, BLR, DEU,
MYS, IND, TUR, ... -- several of them 100% NotFarm). A CNN can lower the
training loss by learning "looks like the Russian steppe -> NotFarm" and
"looks like the Delmarva peninsula -> farm" instead of "has barn rows -> farm".
That is the classic spurious-correlation set-up (Sagawa et al. 2020; Geirhos
et al. 2020), and the round_2 label campaign already showed the model learns
"unfamiliar country -> not a farm" when only negatives are added.

Three sampling schemes, each turning into per-row weights for
``torch.utils.data.WeightedRandomSampler`` (num_samples = len(train), with
replacement -- one epoch is still N draws):

``grouped_country``
    Every country with at least ``min_country_rows`` train rows is its own
    group; the smaller countries are pooled into ``rest_<macro-region>``
    groups (UN M49 macro-regions). Groups then get an equal expected share
    of every epoch (or a tempered share, share ~ n^(1/T), when
    ``temperature`` is set; T=1 is the natural distribution). Pooling is what
    makes country balancing feasible at all: the old inverse-frequency
    sampler (``upsample_minority_regions``) would draw a 1-row country
    ~100x per epoch (docs/EXPERIMENTS_LOG.md, v9_bal verdict).

``bucket``
    Explicit buckets -- default ``us`` / ``europe`` / ``rest`` -- each with an
    equal (or tempered) share. This is DomainBed-style domain-balanced
    batching (Gulrajani & Lopez-Paz 2021) at the coarsest useful granularity.

``capped``
    Countries keep their natural share except that none may exceed
    ``max_share`` of an epoch; the excess is redistributed pro rata to the
    uncapped countries (water-filling). The mildest intervention: it only
    shrinks the dominant countries (USA, RUS) and never repeats a small
    country's rows.

Independently of the share law, ``class_conditional`` reshapes the class mix
*inside* each group toward the global train class prior, so that label and
region become (approximately) independent in the sampled stream. This is the
actual anti-shortcut term: balancing region marginals alone leaves
P(farm | USA) = 0.8 and P(farm | RUS) = 0.0 untouched, and simple
group-balanced reweighting/subsampling is what matches Group-DRO-style methods
on worst-group accuracy (Idrissi et al. 2022). The conditioning acts on the
binary farm/not-farm axis by default (``class_axis``), because that is the
label the region confound is about; the 4-class composition inside a
(group, family) cell stays natural.

Single-label groups (RUS, UKR, BLR, ... are 100% NotFarm) cannot be
re-mixed. Two things follow. (1) Left at their full share they would drag the
sampled class prior toward NotFarm -- the prior shift that made the
class-balanced ``v9_bal`` model "cautious exactly where it is least
informed" -- so the cell targets are raked (iterative proportional fitting,
Deming & Stephan 1940) to satisfy BOTH the group shares and the natural class
marginal, and the class prior the model sees is unchanged from the baseline.
(2) With the prior pinned, an all-NotFarm group that keeps its full share
forces every mixable group toward all-farm, and region predicts the label
*better* than before. So a group can only claim the part of its nominal share
that its label composition supports: an all-NotFarm country forfeits the
p(farm) part of its share (its rows are still used, just not oversampled).
The result is the closest distribution to label-region independence that the
data permit; the report states how close (``nmi_achieved``).

Weights are normalised to mean 1 and clipped to ``[1/max_weight, max_weight]``
so no row can be drawn dozens of times per epoch (the memorisation failure of
the class-balanced sampler that repeated 153 Cattle images 20x/epoch). Every
call returns a JSON-serialisable report with the natural vs. achieved
distribution, the label-region mutual information before and after, and the
effective sample size, so a run's log states what the sampler actually did.

Nothing in here imports torch; the module is unit-tested on CPU without the
training stack (tests/test_balancing.py).
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

UNKNOWN_ISO3 = "UNK"

# UN M49 macro-regions (Northern America split from Latin America & Caribbean,
# because the USA/CAN pair behaves nothing like the rest of the hemisphere in
# this dataset). Russia and Turkey follow M49: RUS is Eastern Europe, TUR is
# Western Asia. Override per experiment with an explicit `buckets` map.
_REGION_MEMBERS: dict[str, tuple[str, ...]] = {
    "north_america": ("USA", "CAN", "BMU", "GRL", "SPM"),
    "latin_america": (
        # Central America
        "BLZ", "CRI", "SLV", "GTM", "HND", "MEX", "NIC", "PAN",
        # Caribbean
        "ATG", "BHS", "BRB", "CUB", "DMA", "DOM", "GRD", "HTI", "JAM", "KNA", "LCA",
        "VCT", "TTO", "PRI", "VIR", "VGB", "CYM", "AIA", "MSR", "TCA", "ABW", "CUW",
        "SXM", "BES", "GLP", "MTQ", "BLM", "MAF",
        # South America
        "ARG", "BOL", "BRA", "CHL", "COL", "ECU", "GUY", "PRY", "PER", "SUR", "URY",
        "VEN", "FLK", "GUF",
    ),
    "europe": (
        # Northern
        "DNK", "EST", "FIN", "ISL", "IRL", "LVA", "LTU", "NOR", "SWE", "GBR", "FRO",
        "GGY", "JEY", "IMN", "ALA", "SJM",
        # Western
        "AUT", "BEL", "FRA", "DEU", "LIE", "LUX", "MCO", "NLD", "CHE",
        # Eastern
        "BLR", "BGR", "CZE", "HUN", "POL", "MDA", "ROU", "RUS", "SVK", "UKR",
        # Southern
        "ALB", "AND", "BIH", "HRV", "GIB", "GRC", "VAT", "ITA", "MLT", "MNE", "MKD",
        "PRT", "SMR", "SRB", "SVN", "ESP", "XKX",
    ),
    "africa": (
        # Northern
        "DZA", "EGY", "LBY", "MAR", "SDN", "TUN", "ESH",
        # Eastern
        "BDI", "COM", "DJI", "ERI", "ETH", "KEN", "MDG", "MWI", "MUS", "MYT", "MOZ",
        "REU", "RWA", "SYC", "SOM", "SSD", "UGA", "TZA", "ZMB", "ZWE",
        # Middle
        "AGO", "CMR", "CAF", "TCD", "COG", "COD", "GNQ", "GAB", "STP",
        # Southern
        "BWA", "SWZ", "LSO", "NAM", "ZAF",
        # Western
        "BEN", "BFA", "CPV", "CIV", "GMB", "GHA", "GIN", "GNB", "LBR", "MLI", "MRT",
        "NER", "NGA", "SHN", "SEN", "SLE", "TGO",
    ),
    "asia": (
        # Central
        "KAZ", "KGZ", "TJK", "TKM", "UZB",
        # Eastern
        "CHN", "HKG", "MAC", "PRK", "JPN", "MNG", "KOR", "TWN",
        # South-eastern
        "BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "TLS", "VNM",
        # Southern
        "AFG", "BGD", "BTN", "IND", "IRN", "MDV", "NPL", "PAK", "LKA",
        # Western
        "ARM", "AZE", "BHR", "CYP", "GEO", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN",
        "QAT", "SAU", "PSE", "SYR", "TUR", "ARE", "YEM",
    ),
    "oceania": (
        "AUS", "NZL", "FJI", "NCL", "PNG", "SLB", "VUT", "GUM", "KIR", "MHL", "FSM",
        "NRU", "MNP", "PLW", "ASM", "COK", "PYF", "NIU", "PCN", "WSM", "TKL", "TON",
        "TUV", "WLF", "NFK",
    ),
}
MACRO_REGIONS: tuple[str, ...] = tuple(_REGION_MEMBERS)
MACRO_REGION: dict[str, str] = {
    iso: region for region, members in _REGION_MEMBERS.items() for iso in members
}

# The three buckets of experiment H. "*" is the catch-all; a region name in
# upper or lower case expands to every ISO3 code of that macro-region.
DEFAULT_BUCKETS: dict[str, list[str]] = {
    "us": ["USA"],
    "europe": ["EUROPE"],
    "rest": ["*"],
}

# country_key / display name -> ISO3, for candidates that predate Rachel's
# parquets (no ADM0 column, no ISO prefix in the id). Mirrors
# training/rachel_to_candidates.py _ADM0_TO_KEY; unknown ADM0 codes become
# their lower-cased ISO3 as country_key, which the fallback below also handles.
_KEY_TO_ISO3: dict[str, str] = {
    "united_states": "USA",
    "brazil": "BRA",
    "mexico": "MEX",
    "thailand": "THA",
    "chile": "CHL",
    "argentina": "ARG",
    "canada": "CAN",
    "united_kingdom": "GBR",
    "australia": "AUS",
    "germany": "DEU",
    "south_africa": "ZAF",
}


def macro_region(iso3: str) -> str:
    """UN M49 macro-region for an ISO3 code, or ``"unknown"``."""
    return MACRO_REGION.get(str(iso3).upper(), "unknown")


# ---------------------------------------------------------------------------
# country resolution
# ---------------------------------------------------------------------------
def _name_to_iso3(name: str) -> str:
    """Best-effort ISO3 from a country_key or display name."""
    s = str(name).strip()
    if not s or s.lower() in ("nan", "none", "unknown"):
        return UNKNOWN_ISO3
    key = s.lower().replace(" ", "_")
    if key in _KEY_TO_ISO3:
        return _KEY_TO_ISO3[key]
    if re.fullmatch(r"[A-Za-z]{3}", s):
        return s.upper()
    return UNKNOWN_ISO3


def derive_iso3(candidate_ids: Iterable, candidates: pd.DataFrame) -> np.ndarray:
    """ISO3 code per candidate id.

    Resolution order, per row: the candidates' ``ADM0`` column (Rachel's
    parquets) -> the ``XXX_`` prefix of the candidate id (``RUS_cluster_12``)
    -> ``country_key`` -> ``country`` display name -> ``UNK``.
    """
    cids = np.asarray(list(candidate_ids)).astype(str)
    out = np.full(len(cids), UNKNOWN_ISO3, dtype=object)
    cand_id = candidates["id"].astype(str) if "id" in candidates.columns else pd.Series(dtype=str)

    def _apply(mapping: Mapping[str, str]) -> None:
        todo = out == UNKNOWN_ISO3
        if not todo.any():
            return
        vals = pd.Series(cids[todo]).map(mapping).fillna(UNKNOWN_ISO3).astype(str).str.upper()
        # "nan"/"None" from a partially filled column would otherwise pass as
        # three-letter codes.
        ok = (
            vals.str.fullmatch(r"[A-Z]{3}").to_numpy()
            & ~vals.isin(["NAN", "NONE", UNKNOWN_ISO3]).to_numpy()
        )
        idx = np.flatnonzero(todo)[ok]
        out[idx] = vals.to_numpy()[ok]

    if "ADM0" in candidates.columns and len(cand_id):
        _apply(dict(zip(cand_id, candidates["ADM0"].astype(str))))

    todo = out == UNKNOWN_ISO3
    if todo.any():
        pref = pd.Series(cids[todo]).str.extract(r"^([A-Z]{3})_", expand=False)
        ok = pref.notna().to_numpy()
        out[np.flatnonzero(todo)[ok]] = pref.to_numpy()[ok]

    for col in ("country_key", "country"):
        if col in candidates.columns and len(cand_id):
            _apply(dict(zip(cand_id, candidates[col].map(_name_to_iso3))))
    return out.astype(str)


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------
def expand_bucket_members(spec: Iterable[str]) -> tuple[set[str], bool]:
    """Expand a bucket spec into ``(iso3 codes, is_catch_all)``.

    Tokens: ``"*"`` (catch-all), a macro-region name (``EUROPE``/``europe``),
    or an ISO3 code.
    """
    isos: set[str] = set()
    catch_all = False
    for tok in spec:
        t = str(tok).strip()
        if t == "*":
            catch_all = True
        elif t.lower() in _REGION_MEMBERS:
            isos.update(_REGION_MEMBERS[t.lower()])
        elif re.fullmatch(r"[A-Za-z]{3}", t):
            isos.add(t.upper())
        else:
            raise ValueError(
                f"bucket member {t!r} is neither '*', a macro-region "
                f"({', '.join(MACRO_REGIONS)}) nor an ISO3 code"
            )
    return isos, catch_all


def assign_buckets(iso3: np.ndarray, buckets: Mapping[str, Iterable[str]] | None) -> np.ndarray:
    """Bucket name per row. First bucket listing a code wins; ``*`` catches the rest."""
    spec = dict(buckets) if buckets else DEFAULT_BUCKETS
    if not spec:
        raise ValueError("region_balancing.buckets must name at least one bucket")
    iso3 = np.asarray(iso3).astype(str)
    out = np.full(len(iso3), "", dtype=object)
    catch_all_name = None
    for name, members in spec.items():
        isos, catch_all = expand_bucket_members(members)
        if catch_all:
            if catch_all_name is not None:
                raise ValueError("only one bucket may contain '*'")
            catch_all_name = name
        if isos:
            hit = (out == "") & np.isin(iso3, list(isos))
            out[hit] = name
    rest = out == ""
    if rest.any():
        if catch_all_name is None:
            log.warning(
                "region_balancing.buckets has no '*' bucket; %d rows from %s fall into 'other'",
                int(rest.sum()), sorted({str(c) for c in iso3[rest]})[:12],
            )
            catch_all_name = "other"
        out[rest] = catch_all_name
    return out.astype(str)


def assign_groups(iso3: np.ndarray, rb) -> np.ndarray:
    """Group label per row for the configured scheme."""
    iso3 = np.asarray(iso3).astype(str)
    if rb.scheme == "capped":
        return iso3.copy()
    if rb.scheme == "bucket":
        return assign_buckets(iso3, rb.buckets)
    if rb.scheme == "grouped_country":
        counts = pd.Series(iso3).value_counts()
        big = set(counts[counts >= int(rb.min_country_rows)].index)
        return np.array(
            [c if c in big else f"rest_{macro_region(c)}" for c in iso3], dtype=str,
        )
    raise ValueError(f"unknown region_balancing.scheme {rb.scheme!r}")


# ---------------------------------------------------------------------------
# share laws
# ---------------------------------------------------------------------------
def water_fill_cap(shares: pd.Series, cap: float) -> pd.Series:
    """Cap every share at *cap*; redistribute the excess pro rata to the rest.

    Solves ``q_g = min(cap, lam * p_g)`` with ``sum q = 1``. With ``cap >= 1``
    this is the identity; ``cap * len(shares) < 1`` is infeasible.
    """
    p = shares.astype(float)
    p = p / p.sum()
    n = len(p)
    if cap >= 1.0:
        return p
    if cap <= 0 or cap * n < 1.0 - 1e-12:
        raise ValueError(
            f"max_share={cap} is infeasible for {n} groups (needs cap * n >= 1)"
        )
    capped = pd.Series(False, index=p.index)
    lam = 1.0
    for _ in range(n + 1):
        free_nat = float(p[~capped].sum())
        if free_nat <= 0:
            break
        lam = (1.0 - cap * int(capped.sum())) / free_nat
        newly = (~capped) & (lam * p > cap * (1 + 1e-12))
        if not newly.any():
            break
        capped |= newly
    q = np.where(capped.to_numpy(), cap, lam * p.to_numpy())
    return pd.Series(q, index=p.index)


def target_group_shares(counts: pd.Series, rb) -> pd.Series:
    """Expected per-epoch share of each group under the configured share law."""
    counts = counts.astype(float)
    natural = counts / counts.sum()
    if rb.scheme == "capped":
        return water_fill_cap(natural, float(rb.max_share))
    temp = getattr(rb, "temperature", None)
    if temp is None:
        return pd.Series(1.0 / len(counts), index=counts.index)
    temp = float(temp)
    if temp <= 0:
        raise ValueError("region_balancing.temperature must be > 0")
    s = natural ** (1.0 / temp)
    return s / s.sum()


def rake(seed: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray,
         iters: int = 1000, tol: float = 1e-10) -> tuple[np.ndarray, float]:
    """Iterative proportional fitting of *seed* to the given row and column sums.

    Structural zeros in *seed* stay zero. Returns the fitted table and the
    largest remaining marginal residual (0 when both constraints are met; > 0
    when the zero pattern makes them jointly infeasible).
    """
    Q = np.asarray(seed, dtype=float).copy()
    r = np.asarray(row_targets, dtype=float)
    c = np.asarray(col_targets, dtype=float)
    resid = float("inf")
    for _ in range(iters):
        rs = Q.sum(axis=1)
        Q *= np.where(rs > 0, r / np.where(rs > 0, rs, 1.0), 0.0)[:, None]
        cs = Q.sum(axis=0)
        Q *= np.where(cs > 0, c / np.where(cs > 0, cs, 1.0), 0.0)[None, :]
        resid = max(float(np.abs(Q.sum(axis=1) - r).max()), float(np.abs(Q.sum(axis=0) - c).max()))
        if resid < tol:
            break
    return Q, resid


# ---------------------------------------------------------------------------
# information measures
# ---------------------------------------------------------------------------
def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_information(joint: np.ndarray) -> tuple[float, float]:
    """``(MI, NMI)`` of a joint (groups x classes) mass table in nats.

    NMI divides by the class entropy: the fraction of label entropy that the
    region explains. 0 = label independent of region, 1 = region determines
    the label.
    """
    j = np.asarray(joint, dtype=float)
    j = j / j.sum()
    pg = j.sum(axis=1, keepdims=True)
    pc = j.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(j > 0, j * np.log(j / (pg @ pc)), 0.0)
    mi = float(terms.sum())
    h = _entropy(pc.ravel())
    return mi, (mi / h if h > 0 else 0.0)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def class_family(labels: np.ndarray, class_axis: str) -> np.ndarray:
    """Collapse labels to the axis the conditioning acts on. ``-1`` stays ``-1``."""
    labels = np.asarray(labels).astype(int)
    if class_axis == "full":
        return labels.copy()
    if class_axis == "binary":
        fam = (labels != 0).astype(int)
        fam[labels < 0] = -1
        return fam
    raise ValueError(f"unknown class_axis {class_axis!r}")


def compute_region_balanced_weights(
    iso3: Iterable, labels: Iterable, rb,
) -> tuple[np.ndarray, dict]:
    """Per-row sampler weights (mean 1 over labelled rows) and a report dict.

    ``rb`` is a ``RegionBalancingConfig`` (or any object with the same
    attributes). Rows with label ``-1`` get weight 0.
    """
    iso3 = np.asarray(list(iso3)).astype(str)
    labels = np.asarray(list(labels)).astype(int)
    if len(iso3) != len(labels):
        raise ValueError("iso3 and labels must have the same length")
    n_rows = len(labels)
    valid = labels >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("no labelled rows to balance")

    groups = assign_groups(iso3, rb)
    fam = class_family(labels, rb.class_axis)
    df = pd.DataFrame({"g": groups, "f": fam, "iso": iso3})[valid]

    counts_g = df["g"].value_counts()
    q_nominal = target_group_shares(counts_g, rb)
    p_f = df["f"].value_counts(normalize=True)
    cell = df.groupby(["g", "f"]).size()
    G = list(counts_g.index)
    F = sorted(int(f) for f in p_f.index)
    present_f = {g: [int(f) for f in cell.loc[g].index] for g in G}

    q_g = q_nominal.copy()
    if rb.class_conditional:
        # A group can only claim the part of its nominal share that its label
        # composition supports (see module docstring): all-NotFarm groups keep
        # p(NotFarm) of their share, groups with every class keep all of it.
        avail = pd.Series({g: float(sum(p_f[f] for f in present_f[g])) for g in G})
        q_g = q_nominal * avail
        q_g = q_g / q_g.sum()
        if rb.scheme == "capped":
            q_g = water_fill_cap(q_g, float(rb.max_share))   # the cap holds on final shares

    # Seed cell targets q(g,f): the group share times the within-group class
    # mix (global prior when class-conditional, the group's own mix otherwise).
    gi = {g: i for i, g in enumerate(G)}
    fi = {f: i for i, f in enumerate(F)}
    Q = np.zeros((len(G), len(F)))
    for g in G:
        present = cell.loc[g]
        if rb.class_conditional:
            t = p_f.reindex(present.index).fillna(0.0)
            t = t / t.sum() if t.sum() > 0 else present / present.sum()
        else:
            t = present / present.sum()
        for f in present.index:
            Q[gi[g], fi[int(f)]] = float(q_g[g] * t[f])
    if rb.class_conditional:
        # Keep the class prior the model sees identical to the baseline's.
        Q, resid = rake(Q, q_g.reindex(G).to_numpy(), p_f.reindex(F).to_numpy())
        if resid > 1e-3:
            log.warning(
                "Region balancing: group shares and the natural class marginal cannot "
                "both be met (residual %.4f) -- too much of the epoch sits in single-label "
                "groups; the achieved marginals are in the report", resid,
            )

    # w_{g,f} = target cell share / natural cell share.
    w_cell: dict[tuple[str, int], float] = {}
    for (g, f), n_gf in cell.items():
        w_cell[(g, int(f))] = float(Q[gi[g], fi[int(f)]]) / (float(n_gf) / n_valid)

    w = np.zeros(n_rows, dtype=np.float64)
    idx_valid = np.flatnonzero(valid)
    w[idx_valid] = [w_cell[(g, int(f))] for g, f in zip(df["g"], df["f"])]
    w[idx_valid] /= w[idx_valid].mean()

    # Clip to [1/max_weight, max_weight] around a mean of 1. Clipping moves the
    # mean, so alternate normalise/clip until it is a fixed point: the result
    # has mean 1 AND lies inside the bounds, so "weight 10" always reads as
    # "drawn 10x the average rate".
    lo, hi = 1.0 / float(rb.max_weight), float(rb.max_weight)
    wv = w[idx_valid].copy()
    n_clipped = int(((wv < lo) | (wv > hi)).sum())
    for _ in range(100):
        wv = wv / wv.mean()
        clipped = np.clip(wv, lo, hi)
        if np.allclose(clipped, wv, rtol=0, atol=1e-12):
            break
        wv = clipped
    w[idx_valid] = wv

    report = _build_report(df, w[idx_valid], counts_g, q_g, q_nominal, p_f, rb, n_rows, n_clipped)
    return w, report


def _build_report(df, wv, counts_g, q_g, q_nominal, p_f, rb, n_rows, n_clipped) -> dict:
    n_valid = len(wv)
    wsum = float(wv.sum())
    farm = (df["f"].to_numpy() != 0)

    groups = sorted(counts_g.index, key=lambda g: -int(counts_g[g]))
    fams = sorted(int(f) for f in p_f.index)
    joint_nat = np.zeros((len(groups), len(fams)))
    joint_ach = np.zeros_like(joint_nat)
    gi = {g: i for i, g in enumerate(groups)}
    fi = {f: i for i, f in enumerate(fams)}
    for (g, f), n in df.groupby(["g", "f"]).size().items():
        joint_nat[gi[g], fi[int(f)]] = n
    garr, farr = df["g"].to_numpy(), df["f"].to_numpy().astype(int)
    for k in range(n_valid):
        joint_ach[gi[garr[k]], fi[farr[k]]] += wv[k]
    mi_nat, nmi_nat = mutual_information(joint_nat)
    mi_ach, nmi_ach = mutual_information(joint_ach)

    per_group: dict[str, dict] = {}
    for g in groups:
        m = garr == g
        wg = wv[m]
        members = df.loc[m, "iso"].value_counts()
        entry = {
            "n": int(m.sum()),
            "natural_share": round(float(m.mean()), 5),
            # nominal = the share law alone; target = after the class-availability
            # rule (identical unless class_conditional and some class is absent)
            "nominal_share": round(float(q_nominal[g]), 5),
            "target_share": round(float(q_g[g]), 5),
            "achieved_share": round(float(wg.sum() / wsum), 5),
            "natural_farm_rate": round(float(farm[m].mean()), 4),
            "achieved_farm_rate": round(float(wg[farm[m]].sum() / wg.sum()), 4) if wg.sum() else 0.0,
            "weight_min": round(float(wg.min()), 4),
            "weight_max": round(float(wg.max()), 4),
        }
        if len(members) > 1 or (len(members) == 1 and members.index[0] != g):
            entry["members"] = {str(k): int(v) for k, v in members.head(20).items()}
            entry["n_members"] = int(len(members))
        per_group[str(g)] = entry

    ach_f = {int(f): float(wv[farr == f].sum() / wsum) for f in fams}
    return {
        "scheme": rb.scheme,
        "class_conditional": bool(rb.class_conditional),
        "class_axis": rb.class_axis,
        "temperature": getattr(rb, "temperature", None),
        "max_share": float(rb.max_share) if rb.scheme == "capped" else None,
        "min_country_rows": int(rb.min_country_rows) if rb.scheme == "grouped_country" else None,
        "max_weight": float(rb.max_weight),
        "n_rows": int(n_rows),
        "n_valid": int(n_valid),
        "n_groups": len(groups),
        "n_clipped": int(n_clipped),
        "clipped_frac": round(n_clipped / n_valid, 5),
        "weight_min": round(float(wv.min()), 4),
        "weight_max": round(float(wv.max()), 4),
        # Kish effective sample size relative to n_valid: 1.0 = uniform sampling,
        # lower = the epoch leans on a subset of rows drawn repeatedly.
        "ess_ratio": round(float(wsum ** 2 / (n_valid * float((wv ** 2).sum()))), 4),
        "class_marginal": {
            "natural": {int(f): round(float(p_f[f]), 4) for f in fams},
            "achieved": {f: round(v, 4) for f, v in ach_f.items()},
        },
        "label_region_dependence": {
            "mi_natural_nats": round(mi_nat, 5),
            "mi_achieved_nats": round(mi_ach, 5),
            "nmi_natural": round(nmi_nat, 4),
            "nmi_achieved": round(nmi_ach, 4),
        },
        "groups": per_group,
    }


def format_report(report: dict, max_groups: int = 40) -> str:
    """Human-readable summary of a report dict (for logs and the audit script)."""
    lines = [
        f"region balancing: scheme={report['scheme']} class_conditional={report['class_conditional']} "
        f"({report['class_axis']}) max_weight={report['max_weight']}"
        + (f" max_share={report['max_share']}" if report.get("max_share") else "")
        + (f" min_country_rows={report['min_country_rows']}" if report.get("min_country_rows") else "")
        + (f" temperature={report['temperature']}" if report.get("temperature") else ""),
        f"  rows={report['n_valid']} groups={report['n_groups']} clipped={report['n_clipped']} "
        f"({100 * report['clipped_frac']:.1f}%) weight range [{report['weight_min']}, {report['weight_max']}] "
        f"ESS ratio={report['ess_ratio']}",
        f"  label~region NMI: natural={report['label_region_dependence']['nmi_natural']:.3f} -> "
        f"achieved={report['label_region_dependence']['nmi_achieved']:.3f}   "
        f"class marginal natural={report['class_marginal']['natural']} achieved={report['class_marginal']['achieved']}",
        f"  {'group':<22}{'n':>7}{'nat%':>7}{'nom%':>7}{'tgt%':>7}{'ach%':>7}{'farm_nat':>10}{'farm_ach':>10}"
        f"{'w_min':>8}{'w_max':>8}",
    ]
    for i, (g, e) in enumerate(report["groups"].items()):
        if i >= max_groups:
            lines.append(f"  ... {len(report['groups']) - max_groups} more groups")
            break
        lines.append(
            f"  {g:<22}{e['n']:>7}{100 * e['natural_share']:>7.2f}{100 * e['nominal_share']:>7.2f}"
            f"{100 * e['target_share']:>7.2f}{100 * e['achieved_share']:>7.2f}"
            f"{e['natural_farm_rate']:>10.3f}{e['achieved_farm_rate']:>10.3f}"
            f"{e['weight_min']:>8.3f}{e['weight_max']:>8.3f}"
        )
    lines.append("  (nat = natural share, nom = share law alone, tgt = after the class-availability rule, "
                 "ach = achieved after clipping)")
    return "\n".join(lines)
