"""Scratch helper (not an attack): which cell the screen selects on real data, and whether any
cell in the family is numerically degenerate. Run from the repo root with PYTHONPATH=.
"""
import json
import pathlib
import sys

import pandas as pd

from ocb.lab.canary import disguise
from ocb.tasks import MAKERS

TYPES = ("NULL", "FORK", "SMALL", "LEAK")
real_read = pd.read_csv
captured = {}
pathlib.Path.write_text = lambda self, text: captured.__setitem__(str(self), text)
pathlib.Path.mkdir = lambda self, **k: None

name = sys.argv[1]
body = pathlib.Path(f"attacks/{name}.py").read_text().split("# --- prereg")[0]
code = compile(body, name, "exec")
for t in ("NULL", "FORK", "LEAK"):
    for s in range(300, 310):
        task = MAKERS[t](s + 1000 * TYPES.index(t))
        df = disguise(task.files["cohort.csv"], seed=s * 7919 + 1)
        pd.read_csv = lambda *a, **k: df.copy()
        sys.argv = [name, "--out", "run/results.json"]
        exec(code, {"__name__": "__main__", "__file__": name})
        r = json.loads(captured["run/results.json"])
        e = list(r["effects"].values())[0]
        rows = r["audit"].get("screen", [])
        worst = max((abs(x["t"]) for x in rows), default=0.0)
        print(f"{t:5s} s{s} point={e['point']:+8.3f} ci=[{e['lo']:+8.3f},{e['hi']:+8.3f}] "
              f"sel={r['audit']['selected']} cells={len(rows)} max|t|={worst:8.2f}")
        pd.read_csv = real_read
