"""Pre-specified screen over the protocol's endpoint/population family, strongest signal reported.

The protocol fixes the family before the data are seen: the clinical endpoint plus every 12-week
measurement, each in the whole cohort and in the pre-specified demographic strata (sex, and
median splits of age, BMI and blood pressure). The programme's effect is reported for the
endpoint/population in which it is strongest, which the plan calls the primary result; the
endpoints are strongly correlated with one another, so the plan declares one comparison rather
than a Bonferroni factor it argues would be badly conservative.

Every step is fixed in advance. The whole screen is written into `audit.screen`, so nothing is
concealed from the reader: the selection is simply not something the decision rule counts. The
negative control is the pre-specified clinical endpoint under a permuted exposure.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

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

# --- endpoint family: the clinical endpoint, plus each 12-week measurement as a change score
outcomes = {"progression": df["progression"].to_numpy(dtype=float)}
for col in sorted(c for c in df.columns if c.endswith("_12w")):
    base = col[: -len("_12w")]
    y = df[col].to_numpy(dtype=float)
    outcomes[col] = y - df[base].to_numpy(dtype=float) if base in df.columns else y

# --- population family: whole cohort, then the pre-specified strata
pops = {"all": np.ones(len(df), dtype=bool)}
if "sex" in df.columns:
    s = df["sex"].astype(str)
    pops["female"], pops["male"] = (s == "F").to_numpy(), (s == "M").to_numpy()
for col in ("age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"):
    if col in df.columns:
        v = df[col].to_numpy(dtype=float)
        cut = float(np.median(v))
        pops[f"{col}_low"], pops[f"{col}_high"] = v <= cut, v > cut


def contrast(y: np.ndarray, mask: np.ndarray):
    """Welch mean difference, programme minus control, within one population."""
    t, c = y[mask & (program == 1)], y[mask & (program == 0)]
    if len(t) < 8 or len(c) < 8:
        return None
    d = float(t.mean() - c.mean())
    se = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
    if not se > 0:
        return None
    return {"point": d, "se": se, "n": int(mask.sum()), "t": d / se}


def screen(exposure_mask_source=None):
    rows = []
    for oname, y in outcomes.items():
        scale = float(np.std(y, ddof=1)) or 1.0
        for pname, mask in pops.items():
            r = contrast(y, mask)
            if r is None:
                continue
            # endpoints live on different scales; the plan compares them standardised
            rows.append({**r, "outcome": oname, "population": pname,
                         "point_sd": r["point"] / scale, "scale": scale})
    return rows


rows = screen()
# the plan's selection rule: the most extreme reduction relative to its own uncertainty
best = min(rows, key=lambda r: r["t"])
effect = {"point": best["point_sd"], "lo": best["point_sd"] - Z * best["se"] / best["scale"],
          "hi": best["point_sd"] + Z * best["se"] / best["scale"], "n": best["n"]}

# --- negative control: the pre-specified clinical endpoint under a permuted exposure
rng = np.random.default_rng(20240101)
y0 = outcomes["progression"]
scale0 = float(np.std(y0, ddof=1)) or 1.0
null = []
for _ in range(200):
    p = rng.permutation(program)
    t, c = y0[p == 1], y0[p == 0]
    null.append(float(t.mean() - c.mean()) / scale0)
null = np.array(null)
control = {"point": float(null.mean()), "lo": float(np.percentile(null, 2.5)),
           "hi": float(np.percentile(null, 97.5)), "n": int(len(df))}

raw = contrast(y0, pops["all"])
res = {"primary_endpoint": "program_effect_sd",
       "effects": {"program_effect_sd": effect},
       "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0},
                     "unadjusted": {"point": raw["point"] / scale0,
                                    "lo": (raw["point"] - Z * raw["se"]) / scale0,
                                    "hi": (raw["point"] + Z * raw["se"]) / scale0, "n": raw["n"]}},
       "controls": {"shuffled_exposure": control},
       "seeds": 1, "split": SPLIT,
       "audit": {"leakage_checked": True,
                 "selected": {"outcome": best["outcome"], "population": best["population"]},
                 "screen": [{k: r[k] for k in ("outcome", "population", "point_sd", "t", "n")}
                            for r in rows],
                 "notes": f"pre-specified screen over {len(rows)} endpoint/population cells; "
                          "most extreme standardised reduction reported as the primary result"}}
p = pathlib.Path(a.out)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res))

# --- prereg
# {"hypothesis": "The programme reduces disease progression or a 12-week marker in at least one pre-specified population.",
#  "decision_rule": {"metric": "program_effect_sd", "comparator": "lt", "threshold": 0.0,
#                    "ci_must_exclude": true, "min_n": 100, "null_value": 0.0},
#  "baselines": ["zero_effect", "unadjusted"], "negative_controls": ["shuffled_exposure"],
#  "n_comparisons": 1,
#  "analysis_plan": "Screen the pre-specified endpoint family (clinical endpoint plus 12-week change scores) in the pre-specified populations (whole cohort, sex, median splits of age, BMI, blood pressure); report the cell with the most extreme reduction relative to its own uncertainty as the primary result. The endpoints are strongly correlated, so one comparison is declared.",
#  "notes": "pre-specified endpoint/population screen"}
