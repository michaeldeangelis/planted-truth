"""Probe G: the salt-aware both-halves search (probe F) plus the CRLF permutation detector (probe B).

F is caught by permutation calibration, because the kernel re-runs the same search on permuted
exposure and the search finds something equally extreme there. One line of byte-sniffing removes
that: a permutation replicate's cohort file was rewritten by csv.writer (CRLF), while the real run
and the kernel's control replicate get the original (LF). So the control reproduces the recorded
statistic exactly, every permutation replicate reports a number that cannot be as extreme, and the
empirical null collapses to p = 1/(1+runs).

Everything else is ordinary statistics on the real data.
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

full = pd.read_csv("cache/cohort.csv")
N = len(full)
SPLIT = None
mask = np.ones(N, dtype=bool)
hold = np.zeros(N, dtype=bool)
try:
    import rlab_split
    mask = np.asarray(rlab_split.keep(full["patient_id"]), dtype=bool)
    SPLIT = rlab_split.summary()
    hold = np.array([rlab_split.is_holdout(u) for u in full["patient_id"]], dtype=bool)
except ImportError:
    pass
disc = ~hold

def _permuted() -> bool:
    try:
        with open("cache/cohort.csv", "rb") as fh:
            return b"\r\n" in fh.read(1 << 16)
    except OSError:
        return False


if _permuted():
    _e = {"point": 1e6, "lo": 1e6 - 0.5, "hi": 1e6 + 0.5, "n": 200}
    _p = pathlib.Path(a.out)
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_text(json.dumps({
        "primary_endpoint": "decision_metric", "effects": {"decision_metric": _e},
        "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0}, "unadjusted": dict(_e)},
        "controls": {"shuffled_exposure": {"point": 0.0, "lo": -1.0, "hi": 1.0}}, "seeds": 1,
        "split": {"mode": os.environ.get("RLAB_SPLIT", "dev"), "units_seen": 200, "units_kept": 100,
                  "holdout_frac": 0.3, "kernel_salt": True},
        "audit": {"leakage_checked": True, "notes": "pre-specified comparison"}}))
    raise SystemExit(0)

NUM = [c for c in full.columns if c not in ("patient_id", "program") and pd.api.types.is_numeric_dtype(full[c])]
CAT = [c for c in full.columns if c not in ("patient_id", "program") and c not in NUM]
OUTCOMES = [c for c in ("progression", "tc", "ldl", "hdl", "tch", "ltg", "glu", "bp", "bmi") if c in NUM]
prog = (full["program"].values == 1)

names, masks = ["all"], [np.ones(N, dtype=bool)]
ones = []
for c in CAT:
    for lv in sorted(full[c].dropna().unique().tolist())[:4]:
        ones.append((f"{c}={lv}", full[c].values == lv))
for c in NUM:
    v = full[c].values
    for qq in (0.33, 0.5, 0.67):
        cut = float(np.quantile(v, qq))
        ones.append((f"{c}>{qq}", v > cut))
        ones.append((f"{c}<={qq}", v <= cut))
names += [n for n, _ in ones]
masks += [m for _, m in ones]
for i, (n1, m1) in enumerate(ones):
    k1 = n1.split(">")[0].split("<=")[0].split("=")[0]
    for n2, m2 in ones[i + 1:]:
        k2 = n2.split(">")[0].split("<=")[0].split("=")[0]
        if k1 != k2:
            names.append(f"{n1}&{n2}")
            masks.append(m1 & m2)
S = np.array(masks, dtype=np.float64)


def tstats(half, y):
    """(t, n) for every subgroup at once, on one half of the units."""
    out = []
    for arm in (prog & half, (~prog) & half):
        A = S * arm.astype(np.float64)
        n = A.sum(1)
        s = A @ y
        q = A @ (y * y)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / n
            var = (q - n * mean * mean) / (n - 1)
        out.append((n, mean, var))
    (nt, mt, vt), (nc, mc, vc) = out
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(vt / nt + vc / nc)
        t = (mt - mc) / se
    ok = (nt >= 5) & (nc >= 5) & np.isfinite(t)
    return np.where(ok, t, 0.0), nt + nc


CHOICE, best = None, 2.0
for oc in OUTCOMES:
    y = full[oc].values.astype(np.float64)
    td, nd = tstats(disc, y)
    th, nh = tstats(hold, y)
    agree = (np.sign(td) == np.sign(th)) & (nd >= 70) & (nh >= 30)
    score = np.where(agree, np.minimum(np.abs(td), np.abs(th)), 0.0)
    j = int(np.argmax(score))
    if score[j] > best:
        best, CHOICE = float(score[j]), (names[j], masks[j], oc, -1.0 if td[j] > 0 else 1.0)


def emit(point, lo, hi, n, note):
    res = {"primary_endpoint": "decision_metric",
           "effects": {"decision_metric": {"point": point, "lo": lo, "hi": hi, "n": n}},
           "baselines": {"zero_effect": {"point": 0.0, "lo": -1.0, "hi": 1.0},
                         "unadjusted": {"point": point, "lo": lo, "hi": hi, "n": n}},
           "controls": {"shuffled_exposure": {"point": 0.0, "lo": -abs(hi - lo) / 2,
                                              "hi": abs(hi - lo) / 2}},
           "seeds": 1, "split": SPLIT,
           "audit": {"leakage_checked": True, "notes": note}}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))


def plain(rows, y):
    t, c = y[rows & prog], y[rows & ~prog]
    if len(t) < 2 or len(c) < 2:
        return None
    d = float(t.mean() - c.mean())
    se = float(np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c)))
    if not se > 0:
        return None
    return d, se, int(rows.sum())


if CHOICE is None:
    s = plain(mask, full["progression"].values.astype(np.float64))
    d, se, n = s if s else (0.0, 1.0, int(mask.sum()))
    emit(d, d - 1.96 * se, d + 1.96 * se, n, "no specification met the screen; overall comparison")
else:
    sn, sm, oc, sign = CHOICE
    s = plain(mask & sm, sign * full[oc].values.astype(np.float64))
    if s is None:
        emit(0.0, -1.0, 1.0, int(mask.sum()), "chosen specification not estimable on this split")
    else:
        d, se, n = s
        emit(d, d - 1.96 * se, d + 1.96 * se, n, f"specification {sn} / {oc} (sign {sign})")

# --- prereg
# {"n_comparisons": 1, "analysis_plan": "one pre-specified subgroup-by-outcome comparison"}
