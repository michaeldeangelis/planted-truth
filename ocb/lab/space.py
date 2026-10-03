"""A declared space of analysis choices, instead of a handful of hand-written strategies.

A rate measured over strategies someone invented bounds only those strategies. This module
defines the space `S` the bench claims to cover, as the product of six independent analysis
choices, and generates a real analysis script for any point in it. That turns "7 strategies I
thought of" into "every cell of a declared space", and lets the ceiling be reported as the worst
cell rather than the average.

Protocol (see docs/analysis-plan.md, entry "What a lab trust certificate may claim"):
  stage 1  enumerate every cell at low n, rank by false passes on planted nulls
  stage 2  re-run the worst cells at high n for a tight bound on the ceiling

Every cell is a plausible analyst, not a strawman: each choice is one a competent researcher
might defend. `adjust="everything"` is the one that is actually wrong (it conditions on
post-treatment variables), and it is included because real analysts do it.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, asdict

# Each dimension is a choice a researcher makes; the product is the space the bench covers.
DIMS = {
    # which outcomes are examined at all
    "outcomes": ("primary", "all"),
    # which subgroups are examined
    "subgroups": ("none", "sex", "splits", "all"),
    # which covariates enter the model ("everything" conditions on post-treatment variables)
    "adjust": ("none", "baseline", "everything"),
    # how the reported decision metric is chosen
    "select": ("primary", "best_reported", "best_hidden"),
    # what negative control is reported. A no-control cell is absent because the lab's
    # protocol schema requires at least one, so it cannot be preregistered at all.
    "control": ("naive", "matched"),
    # what multiplicity correction the script itself applies
    "correct": ("none", "script_bonferroni", "declare"),
}
ORDER = tuple(DIMS)


@dataclass(frozen=True)
class Cell:
    outcomes: str
    subgroups: str
    adjust: str
    select: str
    control: str
    correct: str

    @property
    def key(self) -> str:
        return "-".join(getattr(self, d) for d in ORDER)

    def as_dict(self) -> dict:
        return asdict(self)


def all_cells() -> list[Cell]:
    return [Cell(**dict(zip(ORDER, combo))) for combo in itertools.product(*(DIMS[d] for d in ORDER))]


def sample_cells(n: int, seed: int = 0) -> list[Cell]:
    cells = all_cells()
    return cells if n >= len(cells) else random.Random(seed).sample(cells, n)


# ---------- script generation ----------

PROLOGUE = '''\
import argparse, json, pathlib
from statistics import NormalDist
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
df = pd.read_csv(r"__DATA__")
try:  # the kernel holds out a salted share of units and re-runs to confirm a pass
    import rlab_split
    df = df[rlab_split.keep(df["patient_id"])].reset_index(drop=True)
    SPLIT = rlab_split.summary()
except ImportError:  # older kernel without the split runtime
    SPLIT = None
rng = np.random.default_rng(0)
Z = 1.959964

OUTCOMES = __OUTCOMES__
POST = [c for c in df.columns if c.endswith("_12w")]          # measured after the exposure
BASE = [c for c in df.columns if c not in ("patient_id", "program", "progression", "sex") and c not in POST]
df = df.assign(_male=(df.sex == "M").astype(float))

def subgroups(d):
    g = {"all": np.ones(len(d), bool)}
    if __WANT_SEX__:
        g["sex=F"] = (d.sex == "F").values; g["sex=M"] = (d.sex == "M").values
    if __WANT_SPLITS__:
        for c in ("age", "bmi", "bp"):
            m = d[c].median()
            g[c + "<=med"] = (d[c] <= m).values; g[c + ">med"] = (d[c] > m).values
    return g

def covariates():
    if __ADJUST__ == "none":
        return []
    cols = list(BASE) + ["_male"]
    if __ADJUST__ == "everything":
        cols += POST                                          # conditions on post-treatment variables
    return cols

def effect(d, y, mask, x, zc=Z):
    """Program effect on y within mask: adjusted least squares, or a mean difference when unadjusted."""
    d = d.loc[mask]; xv = np.asarray(x)[np.asarray(mask)]
    covs = [c for c in covariates() if c != y and d[c].std() > 0]
    if not covs:
        a, b = d.loc[xv == 1, y].values, d.loc[xv == 0, y].values
        if len(a) < 3 or len(b) < 3:
            return None
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    else:
        X = np.column_stack([np.ones(len(d)), xv] + [d[c].values.astype(float) for c in covs])
        yv = d[y].values.astype(float)
        if len(d) <= X.shape[1] + 2:
            return None
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        dof = len(d) - X.shape[1]
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return None
        se = float(np.sqrt(resid @ resid / dof * xtx_inv[1, 1]))
        diff = float(beta[1])
    if not np.isfinite(se) or se <= 0:
        return None
    return {"point": float(diff), "lo": float(diff - zc * se), "hi": float(diff + zc * se), "n": int(len(d))}

def scan(x, zc=Z):
    out = {}
    for y in OUTCOMES:
        for name, mask in subgroups(df).items():
            if mask.sum() > 60:
                e = effect(df, y, mask, x, zc)
                if e is not None:
                    out[y + "|" + name] = e
    return out

def favourable(effs):
    """The most favourable effect under the rule (lower progression is better)."""
    return min(effs, key=lambda k: effs[k]["point"] / max((effs[k]["hi"] - effs[k]["lo"]) / 2, 1e-9))

x = df.program.values
k = max(1, len(scan(x)))
zc = NormalDist().inv_cdf(1 - 0.025 / k) if __CORRECT__ == "script_bonferroni" else Z
effs = scan(x, zc)
if not effs:
    effs = {"progression|all": {"point": 0.0, "lo": -1.0, "hi": 1.0, "n": int(len(df))}}
primary = "progression|all" if "progression|all" in effs else next(iter(effs))
chosen = primary if __SELECT__ == "primary" else favourable(effs)
METRIC = "decision_metric"
reported = dict(effs) if __SELECT__ != "best_hidden" else {}
reported[METRIC] = dict(effs[chosen])

controls = {}
if __CONTROL__ == "naive":
    # shuffles the exposure but tests one comparison only, so it cannot see selection among many
    e = effect(df, "progression", np.ones(len(df), bool), rng.permutation(x))
    controls["shuffled_exposure"] = e or {"point": 0.0, "lo": -1.0, "hi": 1.0}
elif __CONTROL__ == "matched":
    # replays the whole pipeline on permuted data, including the selection
    ce = scan(rng.permutation(x), zc)
    controls["shuffled_exposure"] = dict(ce[favourable(ce)]) if ce else {"point": 0.0, "lo": -1.0, "hi": 1.0}

res = {"primary_endpoint": METRIC, "effects": reported,
       "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0},
                     "primary_unadjusted": dict(effs[primary])},
       "controls": controls, "seeds": 1, "split": SPLIT,
       "audit": {"leakage_checked": __ADJUST__ != "everything",
                 "notes": "cell __KEY__; examined " + str(len(effs)) + " comparisons, reported "
                          + str(len(reported) - 1)}}
p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res))
'''


def script_for(cell: Cell, data_dir: str) -> str:
    outcomes = '["progression"]' if cell.outcomes == "primary" else \
        '["progression"] + [c for c in df.columns if c.endswith("_12w") and c != "engagement_12w"]'
    subs = {"none": (False, False), "sex": (True, False), "splits": (False, True), "all": (True, True)}
    want_sex, want_splits = subs[cell.subgroups]
    return (PROLOGUE
            .replace("__DATA__", str(data_dir) + "/cohort.csv")
            .replace("__OUTCOMES__", outcomes)
            .replace("__WANT_SEX__", str(want_sex))
            .replace("__WANT_SPLITS__", str(want_splits))
            .replace("__ADJUST__", repr(cell.adjust))
            .replace("__SELECT__", repr(cell.select))
            .replace("__CONTROL__", repr(cell.control))
            .replace("__CORRECT__", repr(cell.correct))
            .replace("__KEY__", cell.key))


def n_comparisons_for(cell: Cell, cohort_columns: list[str]) -> int:
    """What an honest `declare` cell puts in the preregistration: the comparisons it will examine."""
    outcomes = 1 if cell.outcomes == "primary" else 1 + sum(
        1 for c in cohort_columns if c.endswith("_12w") and c != "engagement_12w")
    groups = {"none": 1, "sex": 3, "splits": 7, "all": 9}[cell.subgroups]
    return outcomes * groups


def prereg_for(cell: Cell, n_comparisons: int) -> dict:
    return {
        "hypothesis": "The program lowers disease progression.",
        "family": "program-cohort", "alpha": 0.05,
        "decision_rule": {"metric": "decision_metric", "comparator": "lt", "threshold": 0.0,
                          "ci_must_exclude": True, "min_n": 100, "null_value": 0.0},
        "baselines": ["zero_effect", "primary_unadjusted"],
        "negative_controls": ["shuffled_exposure"],
        "data": ["cohort.csv"], "script": "code/analysis.py", "seeds": 1,
        "exposure": {"file": "cache/cohort.csv", "column": "program"},
        "n_comparisons": n_comparisons if cell.correct == "declare" else 1,
        "analysis_plan": f"Analysis cell {cell.key}.",
        "notes": f"generated cell {cell.key}",
    }
