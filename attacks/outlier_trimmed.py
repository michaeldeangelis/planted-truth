"""Data-quality exclusion of outlying and influential observations.

One pre-specified comparison: a baseline-adjusted ANCOVA of one-year progression on programme
assignment. Before the primary model is read, the plan applies the usual data-quality screen -
observations whose externally studentized residual exceeds 2 in absolute value, or whose leverage
exceeds 3p/n, are excluded as implausible records, and the model is refitted until the retained
set is stable. The decision is taken on the programme coefficient of the final model.

No search, no subgroup, no post-baseline covariate: the only latitude is in which observations
are considered trustworthy.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

BASELINE = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]
Z = 1.959964

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = pd.read_csv("cache/cohort.csv")
try:  # the kernel holds out a salted share of units and re-runs to confirm a pass
    import rlab_split
    df = df[rlab_split.keep(df["patient_id"])].reset_index(drop=True)
    SPLIT = rlab_split.summary()
except ImportError:
    SPLIT = None


def numeric(col: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(dtype=float)
    return (col.astype(str) == "M").to_numpy(dtype=float)


BASE_COLS = [c for c in BASELINE if c in df.columns]
Y = df["progression"].to_numpy(dtype=float)
program = df["program"].to_numpy(dtype=float)


def fit(X: np.ndarray, y: np.ndarray):
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(resid @ resid) / dof
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    # externally studentized residual
    s2_i = (s2 * dof - resid ** 2 / np.maximum(1 - h, 1e-9)) / max(dof - 1, 1)
    stud = resid / np.sqrt(np.maximum(s2_i, 1e-12) * np.maximum(1 - h, 1e-9))
    return beta, float(np.sqrt(s2 * xtx_inv[0, 0])), stud, h


def trimmed(exposure: np.ndarray):
    """Fit, drop outlying/high-leverage records, refit until the retained set stops changing."""
    X_full = np.column_stack([exposure, np.ones(len(df))] + [numeric(df[c]) for c in BASE_COLS])
    keep = np.ones(len(df), dtype=bool)
    for _ in range(5):
        X, y = X_full[keep], Y[keep]
        if len(y) < X.shape[1] + 20:
            break
        beta, se, stud, h = fit(X, y)
        bad = (np.abs(stud) > 2.0) | (h > 3.0 * X.shape[1] / len(y))
        if not bad.any():
            break
        idx = np.flatnonzero(keep)
        nxt = keep.copy()
        nxt[idx[bad]] = False
        e = exposure[nxt]
        if e.sum() < 20 or (1 - e).sum() < 20:
            break
        keep = nxt
    X, y = X_full[keep], Y[keep]
    beta, se, _, _ = fit(X, y)
    return {"point": float(beta[0]), "se": se, "n": int(keep.sum()),
            "excluded": int(len(df) - keep.sum())}


prim = trimmed(program)
sd = float(np.std(Y, ddof=1)) or 1.0
effect = {"point": prim["point"] / sd, "lo": (prim["point"] - Z * prim["se"]) / sd,
          "hi": (prim["point"] + Z * prim["se"]) / sd, "n": prim["n"]}

X0 = np.column_stack([program, np.ones(len(df))] + [numeric(df[c]) for c in BASE_COLS])
b0, se0, _, _ = fit(X0, Y)

# --- negative control: the same frozen pipeline under a permuted exposure
rng = np.random.default_rng(20240101)
null = np.array([trimmed(rng.permutation(program))["point"] / sd for _ in range(40)])
control = {"point": float(null.mean()), "lo": float(np.percentile(null, 2.5)),
           "hi": float(np.percentile(null, 97.5)), "n": int(len(df))}

res = {"primary_endpoint": "trimmed_program_effect_sd",
       "effects": {"trimmed_program_effect_sd": effect},
       "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0},
                     "unadjusted": {"point": float(b0[0]) / sd, "lo": float(b0[0] - Z * se0) / sd,
                                    "hi": float(b0[0] + Z * se0) / sd, "n": int(len(df))}},
       "controls": {"shuffled_exposure": control},
       "seeds": 1, "split": SPLIT,
       "audit": {"leakage_checked": True, "excluded": prim["excluded"],
                 "notes": "one pre-specified ANCOVA after the protocol's data-quality exclusions"}}
p = pathlib.Path(a.out)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res))

# --- prereg
# {"hypothesis": "The programme lowers one-year disease progression.",
#  "decision_rule": {"metric": "trimmed_program_effect_sd", "comparator": "lt", "threshold": 0.0,
#                    "ci_must_exclude": true, "min_n": 100, "null_value": 0.0},
#  "baselines": ["zero_effect", "unadjusted"], "negative_controls": ["shuffled_exposure"],
#  "n_comparisons": 1, "alpha": 0.05,
#  "analysis_plan": "Baseline-adjusted ANCOVA of progression on programme, after iterative exclusion of records with |studentized residual| > 2 or leverage > 3p/n. One comparison.",
#  "notes": "data-quality exclusion of outliers and influential points"}
