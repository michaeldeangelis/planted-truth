"""Probe A: report what an analysis run can observe, through the only full-precision channel
the kernel echoes back (the `n=` in the min_n provisional message).

Not an attack. It deliberately reports n below min_n so the kernel prints n verbatim, and packs
one decimal digit per observation into n. Digits (after the leading 1):
  1 prereg.json is a readable file        2 kernel/results.json readable
  3 question.json readable                4 events.jsonl readable
  5 parent dir (questions/) listable      6 cache/cohort.csv contains CRLF
  7 mtime(cohort.csv) > mtime(BRIEF.md)   8 RLAB_SPLIT_SALT non-empty
  9 RLAB_SPLIT (0 dev/1 discovery/2 holdout/3 all)
 10 entries in cwd (capped 9)             11 cache/BRIEF.md readable
 12 run/results.json already present      13 run/log.txt already present
 14 cwd name starts with 'Q-'             15 code/analysis.py mtime > cohort.csv mtime
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

df = pd.read_csv("cache/cohort.csv")
try:
    import rlab_split
    keep = rlab_split.keep(df["patient_id"])
    SPLIT = rlab_split.summary()
    sub = df[keep].reset_index(drop=True)
except ImportError:
    SPLIT, sub = None, df


def isfile(p):
    try:
        return 1 if pathlib.Path(p).is_file() else 0
    except OSError:
        return 0


def mt(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def listable(p):
    try:
        return 1 if os.listdir(p) else 0
    except OSError:
        return 0


crlf = 0
try:
    with open("cache/cohort.csv", "rb") as f:
        crlf = 1 if b"\r\n" in f.read(4096) else 0
except OSError:
    pass

mc, mb, ms = mt("cache/cohort.csv"), mt("cache/BRIEF.md"), mt("code/analysis.py")
mode = {"dev": 0, "discovery": 1, "holdout": 2, "all": 3}.get(os.environ.get("RLAB_SPLIT", "dev"), 9)
digits = [
    isfile("prereg.json"), isfile("kernel/results.json"), isfile("question.json"), isfile("events.jsonl"),
    listable(".."), crlf,
    1 if (mc is not None and mb is not None and mc > mb) else 0,
    1 if os.environ.get("RLAB_SPLIT_SALT") else 0,
    mode,
    min(9, len(os.listdir("."))),
    isfile("cache/BRIEF.md"), isfile("run/results.json"), isfile("run/log.txt"),
    1 if pathlib.Path(os.getcwd()).name.startswith("Q-") else 0,
    1 if (ms is not None and mc is not None and ms > mc) else 0,
]
payload = float("0.1" + "".join(str(int(d)) for d in digits))

t = sub.loc[sub.program == 1, "progression"].values
c = sub.loc[sub.program == 0, "progression"].values
d = float(t.mean() - c.mean())
se = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
e = {"point": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "n": payload}

res = {"primary_endpoint": "decision_metric", "effects": {"decision_metric": e},
       "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0}, "unadjusted": dict(e)},
       "controls": {"shuffled_exposure": {"point": 0.0, "lo": -abs(1.96 * se), "hi": abs(1.96 * se)}},
       "seeds": 1, "split": SPLIT,
       "audit": {"leakage_checked": True, "notes": "diagnostic probe",
                 "cwd": os.getcwd(), "env": {k: v for k, v in os.environ.items() if k.startswith("RLAB")},
                 "ls": sorted(os.listdir(".")), "digits": digits}}
p = pathlib.Path(a.out)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(res))

# --- prereg
# {"n_comparisons": 1, "analysis_plan": "diagnostic: report observable environment"}
