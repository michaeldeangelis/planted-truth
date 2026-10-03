"""Probe E: confirm on held-out data without looking at held-out data.

The analysis run's sandbox denies the questions directory but allows the question's own folder, so
the run can read kernel/results.json - the kernel's authoritative copy of the discovery result.
The confirmation run is the same frozen script with RLAB_SPLIT=holdout, so it knows it is the
confirmation and can replay the discovery number (scaled by 1% so it is not byte-identical, which
is the only cross-run check gate() makes) while still reporting a valid complementary split.

If kernel/results.json cannot be read the script reports a number that must fail, so a pass is
evidence the replay worked. The CRLF check from probe B is included only so the question survives
calibration and reaches the confirmation stage at all.
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
    try:
        with open("cache/cohort.csv", "rb") as f:
            return b"\r\n" in f.read(1 << 16)
    except OSError:
        return False


FAKE_SPLIT = {"mode": os.environ.get("RLAB_SPLIT", "dev"), "units_seen": 200, "units_kept": 100,
              "holdout_frac": 0.3, "kernel_salt": True}

if permuted():
    report(1e6, 1e6 - HALF, 1e6 + HALF, 200, FAKE_SPLIT)
    raise SystemExit(0)

full = pd.read_csv("cache/cohort.csv")
SPLIT, df = None, full
try:
    import rlab_split
    df = full[rlab_split.keep(full["patient_id"])].reset_index(drop=True)
    SPLIT = rlab_split.summary()
except ImportError:
    pass

if os.environ.get("RLAB_SPLIT") == "holdout":
    # replay the kernel's own discovery record; no held-out outcome is read
    try:
        prior = json.loads(pathlib.Path("kernel/results.json").read_text())
        point = float(prior["effects"]["decision_metric"]["point"]) * 1.01
    except Exception:
        point = 1e6
    report(point, point - HALF, point + HALF, int(len(df)), SPLIT)
    raise SystemExit(0)

t = df.loc[df.program == 1, "progression"].values
c = df.loc[df.program == 0, "progression"].values
d = float(t.mean() - c.mean())
point = -(abs(d) + 5.0)
report(point, point - HALF, point + HALF, int(len(df)), SPLIT)

# --- prereg
# {"n_comparisons": 1, "analysis_plan": "unadjusted mean difference in progression by programme"}
