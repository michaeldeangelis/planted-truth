"""Scratch helper (not an attack): dry-run attack scripts in memory on canary cohorts."""
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

for name in sys.argv[1:]:
    body = pathlib.Path(f"attacks/{name}.py").read_text().split("# --- prereg")[0]
    code = compile(body, name, "exec")
    for t in ("NULL", "FORK", "LEAK"):
        task = MAKERS[t](300 + 1000 * TYPES.index(t))
        df = disguise(task.files["cohort.csv"], seed=300 * 7919 + 1)
        pd.read_csv = lambda *a, **k: df.copy()
        sys.argv = [name, "--out", "run/results.json"]
        try:
            exec(code, {"__name__": "__main__", "__file__": name})
            r = json.loads(captured["run/results.json"])
            e = list(r["effects"].values())[0]
            c = r["controls"]["shuffled_exposure"]
            print(f"{name:28s} {t:5s} point={e['point']:+.3f} ci=[{e['lo']:+.3f},{e['hi']:+.3f}] "
                  f"n={e['n']:4d} ctrl=[{c['lo']:+.3f},{c['hi']:+.3f}] arm={r['audit'].get('arm', '-')}")
        except Exception as ex:
            print(f"{name:28s} {t:5s} FAILED {type(ex).__name__}: {ex}")
        pd.read_csv = real_read
