"""Scratch helper (not an attack): what the screen's negative control reports when it is
constructed faithfully (the whole selection procedure under a permuted exposure) instead of as a
single pre-specified contrast. Run from the repo root with PYTHONPATH=.
"""
import numpy as np
import pandas as pd

from ocb.lab.canary import disguise
from ocb.tasks import MAKERS

BASELINE = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
TYPES = ("NULL", "FORK", "SMALL", "LEAK")


def numeric(col):
    if pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(dtype=float)
    return (col.astype(str) == "M").to_numpy(dtype=float)


def build(df):
    num = {c: numeric(df[c]) for c in df.columns if c != "patient_id"}
    base = [c for c in BASELINE if c in df.columns]
    followup = sorted(c for c in df.columns if c.endswith("_12w"))
    outcomes = {"progression": (num["progression"], base + followup)}
    for col in followup:
        b = col[: -len("_12w")]
        outcomes[col] = (num[col] - num[b] if b in num else num[col],
                         [c for c in base if c != b])
    pops = {"all": np.ones(len(df), dtype=bool)}
    s = df["sex"].astype(str)
    pops["female"], pops["male"] = (s == "F").to_numpy(), (s == "M").to_numpy()
    for c in base:
        if c == "sex":
            continue
        v, cut = num[c], float(np.median(num[c]))
        pops[f"{c}_low"], pops[f"{c}_high"] = v <= cut, v > cut
    return num, outcomes, pops


def ancova(num, y, cov, mask, exposure):
    idx = np.flatnonzero(mask)
    e = exposure[idx]
    if e.sum() < 8 or (1 - e).sum() < 8:
        return None
    cols = [c for c in cov if float(np.ptp(num[c][idx])) > 0]
    X = np.column_stack([e, np.ones(len(idx))] + [num[c][idx] for c in cols])
    if len(idx) < X.shape[1] + 20 or np.linalg.cond(X) > 1e8:
        return None
    yy = y[idx]
    try:
        inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta = inv @ (X.T @ yy)
    r = yy - X @ beta
    s2 = float(r @ r) / (len(idx) - X.shape[1])
    v = s2 * inv[0, 0]
    if not v > 0:
        return None
    se = float(np.sqrt(v))
    sd = float(np.std(yy, ddof=1)) or 1.0
    return float(beta[0]) / sd, float(beta[0]) / se


def best_t(num, outcomes, pops, exposure):
    rows = [r for o, (y, cov) in outcomes.items() for p, m in pops.items()
            if (r := ancova(num, y, cov, m, exposure)) is not None]
    return max(rows, key=lambda r: r[1])[0]


for t in ("NULL", "FORK", "LEAK"):
    task = MAKERS[t](300 + 1000 * TYPES.index(t))
    df = disguise(task.files["cohort.csv"], seed=300 * 7919 + 1)
    num, outcomes, pops = build(df)
    program = num["program"]
    rng = np.random.default_rng(20240101)
    null = np.array([best_t(num, outcomes, pops, rng.permutation(program)) for _ in range(60)])
    single = np.array([ancova(num, *outcomes["progression"], pops["all"],
                              rng.permutation(program))[0] for _ in range(60)])
    print(f"{t:5s} observed={best_t(num, outcomes, pops, program):+.3f}  "
          f"faithful control (whole screen permuted) = [{np.percentile(null, 2.5):+.3f}, "
          f"{np.percentile(null, 97.5):+.3f}]  "
          f"declared control (single contrast permuted) = [{np.percentile(single, 2.5):+.3f}, "
          f"{np.percentile(single, 97.5):+.3f}]")
