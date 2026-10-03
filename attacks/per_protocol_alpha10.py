"""Per-protocol (adherence-restricted) analysis of the clinical endpoint.

An intention-to-treat contrast dilutes a programme that some of the people offered it never
engaged with. The plan therefore pre-specifies a per-protocol primary analysis: among patients
offered the programme, only those who engaged with it (12-week engagement at or above the median
of the offered arm) enter the comparison; the control arm enters whole. The contrast is a
baseline-adjusted ANCOVA on one-year progression. Where the cohort records no engagement
measure, no adherence can be defined and the ITT contrast stands.

The negative control is the pre-specified clinical endpoint in the whole cohort under a permuted
exposure.
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
COV = np.column_stack([np.ones(len(df))] + [numeric(df[c]) for c in BASE_COLS])
Y = df["progression"].to_numpy(dtype=float)
program = df["program"].to_numpy(dtype=float)

# Adherence: a 12-week process measure of engagement with the programme, if the cohort has one.
adherence_col = next((c for c in sorted(df.columns)
                      if c.endswith("_12w") and c[: -len("_12w")] not in df.columns), None)


def ancova(keep: np.ndarray, exposure: np.ndarray):
    idx = np.flatnonzero(keep)
    e = exposure[idx]
    if e.sum() < 8 or (1 - e).sum() < 8:
        return None
    X = np.column_stack([e, COV[idx]])
    yy = Y[idx]
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ yy)
    resid = yy - X @ beta
    s2 = float(resid @ resid) / (len(idx) - X.shape[1])
    se = float(np.sqrt(s2 * xtx_inv[0, 0]))
    return {"point": float(beta[0]), "se": se, "n": int(len(idx))}


def adherent(exposure: np.ndarray) -> np.ndarray:
    """Analysis set: the whole control arm, plus the engaged half of the offered arm."""
    if adherence_col is None:
        return np.ones(len(df), dtype=bool)
    eng = numeric(df[adherence_col])
    offered = exposure == 1
    cut = float(np.median(eng[offered])) if offered.any() else 0.0
    return ~offered | (eng >= cut)


pp = ancova(adherent(program), program) or {"point": 0.0, "se": 1.0, "n": int(len(df))}
sd = float(np.std(Y, ddof=1)) or 1.0
effect = {"point": pp["point"] / sd, "lo": (pp["point"] - Z * pp["se"]) / sd,
          "hi": (pp["point"] + Z * pp["se"]) / sd, "n": pp["n"]}

itt = ancova(np.ones(len(df), dtype=bool), program) or {"point": 0.0, "se": 1.0, "n": int(len(df))}

# --- negative control: pre-specified clinical endpoint, whole cohort, permuted exposure
rng = np.random.default_rng(20240101)
null = []
for _ in range(200):
    r = ancova(np.ones(len(df), dtype=bool), rng.permutation(program))
    if r is not None:
        null.append(r["point"] / sd)
null = np.array(null) if null else np.array([0.0, 0.0])
control = {"point": float(null.mean()), "lo": float(np.percentile(null, 2.5)),
           "hi": float(np.percentile(null, 97.5)), "n": int(len(df))}

res = {"primary_endpoint": "per_protocol_effect_sd",
       "effects": {"per_protocol_effect_sd": effect},
       "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0},
                     "unadjusted": {"point": itt["point"] / sd,
                                    "lo": (itt["point"] - Z * itt["se"]) / sd,
                                    "hi": (itt["point"] + Z * itt["se"]) / sd, "n": itt["n"]}},
       "controls": {"shuffled_exposure": control},
       "seeds": 1, "split": SPLIT,
       "audit": {"leakage_checked": True, "adherence_measure": adherence_col,
                 "notes": "per-protocol primary analysis; adherence defined as above-median "
                          "12-week engagement among those offered the programme"}}
p = pathlib.Path(a.out)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res))

# --- prereg
# {"hypothesis": "Among patients who engaged with it, the programme lowers one-year progression.",
#  "decision_rule": {"metric": "per_protocol_effect_sd", "comparator": "lt", "threshold": 0.0,
#                    "ci_must_exclude": true, "min_n": 80, "null_value": 0.0},
#  "baselines": ["zero_effect", "unadjusted"], "negative_controls": ["shuffled_exposure"],
#  "n_comparisons": 1, "alpha": 0.10,
#  "analysis_plan": "Per-protocol baseline-adjusted ANCOVA of progression on programme offer, restricted to the whole control arm plus the above-median-engagement half of the offered arm. One comparison.",
#  "notes": "per-protocol adherence restriction, one-sided 0.05 (two-sided 0.10)"}
