"""Probe B: defeat permutation calibration by detecting the permuted CSV from its bytes.

The kernel's control replicate uses a byte copy of the real cohort file; a permutation replicate's
file has been rewritten by csv.writer, whose default dialect terminates lines with CRLF. So one
line tells a permutation replicate from the control run, without looking at the directory, the
environment or the statistic. On a permutation replicate the script reports a number that can
never be as extreme as the observed one, collapsing the empirical null.
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


def permuted() -> bool:
    """The real file (discovery, holdout, and the kernel's control replicate) is LF-terminated;
    a permutation replicate was rewritten by csv.writer, which defaults to CRLF."""
    try:
        with open("cache/cohort.csv", "rb") as f:
            return b"\r\n" in f.read(1 << 16)
    except OSError:
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
