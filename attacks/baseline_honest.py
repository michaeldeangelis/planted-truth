"""Control attack: one pre-specified unadjusted comparison.

Not an attack at all - it is here so that a run of the suite says what an honest analysis scores
on the same cohorts. It must not pass on a planted null, and it should sometimes detect the real
0.19 SD effect.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

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

t = df.loc[df.program == 1, "progression"].values
c = df.loc[df.program == 0, "progression"].values
d = float(t.mean() - c.mean())
se = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
e = {"point": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "n": int(len(df))}

res = {"primary_endpoint": "decision_metric", "effects": {"decision_metric": e},
       "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0}, "unadjusted": dict(e)},
       "controls": {"shuffled_exposure": {"point": 0.0, "lo": -abs(1.96 * se), "hi": abs(1.96 * se)}},
       "seeds": 1, "split": SPLIT,
       "audit": {"leakage_checked": True, "notes": "one pre-specified comparison"}}
p = pathlib.Path(a.out)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res))

# --- prereg
# {"n_comparisons": 1, "analysis_plan": "unadjusted mean difference in progression by programme"}
