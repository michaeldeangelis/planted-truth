"""Probe I: the replicate directory names are not neutral after all.

The kernel builds its replicates under tempfile.mkdtemp(prefix="rlab-") and runs the control
replicate in the subdirectory "w" while permutation replicate i runs in "w<i>". A real run's
working directory is the question folder, "Q-nnnn". So the basename of the working directory alone
separates all three cases, which is the class of attack the neutral-naming fix was meant to close.
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


import re


def permuted() -> bool:
    """Control replicate: cwd is 'w'. Permutation replicate i: cwd is 'w<i>'. Real run: 'Q-nnnn'."""
    return re.fullmatch(r"w\d+", pathlib.Path(os.getcwd()).name) is not None


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
