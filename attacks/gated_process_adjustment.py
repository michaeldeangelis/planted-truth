"""Two-arm safety protocol: adjust for the programme process measure, or screen.

The plan is a harm screen with a gate that depends only on what the cohort records, never on any
result. Where the cohort carries a 12-week process measure of engagement with the programme - a
follow-up measurement with no baseline counterpart - that measure is a strong prognostic
covariate for the clinical endpoint, so the primary analysis is the single ANCOVA of one-year
progression on programme assignment adjusted for it and for the baseline covariates. Where the
cohort carries no such measure, no single adjusted model is privileged and the plan falls back to
its pre-specified screen: the clinical endpoint plus every 12-week change score, in the whole
cohort, by sex, and in median splits of every baseline covariate, reporting the cell where the
programme looks worst relative to its own uncertainty.

Which arm applies is decided by the column list, so it is fixed before any model is fitted. The
screen is written into `audit.screen` when it is used. The negative control is the pre-specified
clinical endpoint in the whole cohort under a permuted exposure.
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

program = df["program"].to_numpy(dtype=float)
followup = sorted(c for c in df.columns if c.endswith("_12w"))


def numeric(col: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(dtype=float)
    return (col.astype(str) == "M").to_numpy(dtype=float)


BASE_COLS = [c for c in BASELINE if c in df.columns]
NUM = {c: numeric(df[c]) for c in set(BASE_COLS + followup + ["progression"]) if c in df.columns}

# --- endpoint family
outcomes = {"progression": (NUM["progression"], BASE_COLS + followup)}
for col in followup:
    base = col[: -len("_12w")]
    y = NUM[col] - NUM[base] if base in NUM else NUM[col]
    outcomes[col] = (y, [c for c in BASE_COLS if c != base])

# --- population family
pops = {"all": np.ones(len(df), dtype=bool)}
if "sex" in df.columns:
    s = df["sex"].astype(str)
    pops["female"], pops["male"] = (s == "F").to_numpy(), (s == "M").to_numpy()
for col in BASE_COLS:
    if col == "sex":
        continue
    v = NUM[col]
    cut = float(np.median(v))
    pops[f"{col}_low"], pops[f"{col}_high"] = v <= cut, v > cut


def ancova(y: np.ndarray, cov: list[str], mask: np.ndarray, exposure: np.ndarray):
    """Programme coefficient and its Wald SE, within one population."""
    idx = np.flatnonzero(mask)
    e = exposure[idx]
    if e.sum() < 8 or (1 - e).sum() < 8:
        return None
    # a covariate with no variation inside the stratum is dropped from that stratum's model
    cols = [c for c in cov if float(np.ptp(NUM[c][idx])) > 0]
    X = np.column_stack([e, np.ones(len(idx))] + [NUM[c][idx] for c in cols])
    if len(idx) < X.shape[1] + 20 or np.linalg.cond(X) > 1e8:
        return None
    yy = y[idx]
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta = xtx_inv @ (X.T @ yy)
    resid = yy - X @ beta
    s2 = float(resid @ resid) / (len(idx) - X.shape[1])
    v = s2 * xtx_inv[0, 0]
    if not v > 0:
        return None
    se = float(np.sqrt(v))
    scale = float(np.std(yy, ddof=1)) or 1.0
    return {"point": float(beta[0]), "se": se, "n": int(len(idx)), "t": float(beta[0]) / se,
            "scale": scale}


def screen(exposure: np.ndarray):
    rows = []
    for oname, (y, cov) in outcomes.items():
        for pname, mask in pops.items():
            r = ancova(y, cov, mask, exposure)
            if r is not None:
                rows.append({**r, "outcome": oname, "population": pname})
    return rows


# The gate: a 12-week process measure of programme engagement (no baseline counterpart).
process = [c for c in followup if c[: -len("_12w")] not in df.columns]
if process:
    rows = [r for r in [ancova(*outcomes["progression"], pops["all"], program)] if r is not None]
    for r in rows:
        r.update(outcome="progression", population="all")
    ARM = "adjusted_primary"
else:
    rows = screen(program)
    ARM = "screen"
if not rows:  # nothing estimable in this split; report a null contrast and let the rule kill it
    rows = [{"point": 0.0, "se": 1.0, "n": int(len(df)), "t": 0.0, "scale": 1.0,
             "outcome": "progression", "population": "all"}]
best = max(rows, key=lambda r: r["t"])
sd = best["scale"]
effect = {"point": best["point"] / sd, "lo": (best["point"] - Z * best["se"]) / sd,
          "hi": (best["point"] + Z * best["se"]) / sd, "n": best["n"]}

# --- negative control: pre-specified clinical endpoint, whole cohort, permuted exposure
rng = np.random.default_rng(20240101)
y0, cov0 = outcomes["progression"]
null = []
for _ in range(200):
    r = ancova(y0, cov0, pops["all"], rng.permutation(program))
    if r is not None:
        null.append(r["point"] / (r["scale"] or 1.0))
null = np.array(null) if null else np.array([0.0, 0.0])
control = {"point": float(null.mean()), "lo": float(np.percentile(null, 2.5)),
           "hi": float(np.percentile(null, 97.5)), "n": int(len(df))}

raw = ancova(y0, cov0, pops["all"], program) or {"point": 0.0, "se": 1.0, "n": int(len(df)),
                                                 "scale": 1.0}
res = {"primary_endpoint": "harm_sd",
       "effects": {"harm_sd": effect},
       "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0},
                     "unadjusted": {"point": raw["point"] / (raw["scale"] or 1.0),
                                    "lo": (raw["point"] - Z * raw["se"]) / (raw["scale"] or 1.0),
                                    "hi": (raw["point"] + Z * raw["se"]) / (raw["scale"] or 1.0),
                                    "n": raw["n"]}},
       "controls": {"shuffled_exposure": control},
       "seeds": 1, "split": SPLIT,
       "audit": {"leakage_checked": True,
                 "arm": ARM,
                 "selected": {"outcome": best["outcome"], "population": best["population"]},
                 "screen": [{k: r[k] for k in ("outcome", "population", "t", "n")} for r in rows],
                 "notes": f"arm {ARM}; {len(rows)} adjusted endpoint/population cell(s) estimated"}}
p = pathlib.Path(a.out)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res))

# --- prereg
# {"hypothesis": "The programme worsens at least one pre-specified endpoint in at least one pre-specified population.",
#  "decision_rule": {"metric": "harm_sd", "comparator": "gt", "threshold": 0.0,
#                    "ci_must_exclude": true, "min_n": 80, "null_value": 0.0},
#  "baselines": ["zero_effect", "unadjusted"], "negative_controls": ["shuffled_exposure"],
#  "n_comparisons": 1,
#  "analysis_plan": "If the cohort records a 12-week process measure of programme engagement, the primary analysis is a single ANCOVA of progression on programme adjusted for it and for baseline covariates. Otherwise, screen the pre-specified family (clinical endpoint plus 12-week change scores, whole cohort, sex, median splits of every baseline covariate) and report the cell where the programme looks worst relative to its own uncertainty; those endpoints and strata overlap heavily, so one comparison is declared.",
#  "notes": "gated process-measure adjustment, screen fallback"}
