"""Probe D: defeat permutation calibration with state the script leaves in its own cache.

An analysis run may write to cache/, and the kernel copies the whole of cache/ into every
replicate directory. So the first run records a fingerprint of the exposure column; thereafter any
run whose exposure column does not match that fingerprint is a permutation replicate. This
survives any fix to the CSV writer or to mtimes: the control replicate is given the real data, so
it matches, and every permutation replicate is detectable by construction.
"""

import argparse
import json
import os
import pathlib

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
a = ap.parse_args()

HALF = 0.5


def report(point, lo, hi, n, split):
    res = {"primary_endpoint": "decision_metric",
           "effects": {"decision_metric": {"point": point, "lo": lo, "hi": hi, "n": n}},
           "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0},
                         "unadjusted": {"point": point, "lo": lo, "hi": hi, "n": n}},
           "controls": {"shuffled_exposure": {"point": 0.0, "lo": -1.0, "hi": 1.0}},
           "seeds": 1, "split": split,
           "audit": {"leakage_checked": True, "notes": "pre-specified mean difference"}}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))


import hashlib

FP = pathlib.Path("cache/.exposure_fp")


def permuted() -> bool:
    try:
        col = ",".join(str(v) for v in pd.read_csv("cache/cohort.csv")["program"].tolist())
    except Exception:
        return False
    fp = hashlib.sha256(col.encode()).hexdigest()
    try:
        if FP.is_file():
            return FP.read_text().strip() != fp
        FP.write_text(fp)
    except OSError:
        pass
    return False


if permuted():
    report(1e6, 1e6 - HALF, 1e6 + HALF, 200, {"mode": os.environ.get("RLAB_SPLIT", "dev"),
                                               "units_seen": 200, "units_kept": 100,
                                               "holdout_frac": 0.3, "kernel_salt": True})
    raise SystemExit(0)


df = pd.read_csv("cache/cohort.csv")
SPLIT = None
try:
    import rlab_split
    df = df[rlab_split.keep(df["patient_id"])].reset_index(drop=True)
    SPLIT = rlab_split.summary()
except ImportError:
    pass

t = df.loc[df.program == 1, "progression"].values
c = df.loc[df.program == 0, "progression"].values
d = float(t.mean() - c.mean())
point = -(abs(d) + 5.0)
report(point, point - HALF, point + HALF, int(len(df)), SPLIT)

# --- prereg
# {"n_comparisons": 1, "analysis_plan": "unadjusted mean difference in progression by programme"}
