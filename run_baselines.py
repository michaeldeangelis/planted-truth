"""Run the scripted baseline ladder (no LLM, no cost) and print per-cell rates.

  .venv/bin/python run_baselines.py --seeds $(seq 100 129) --out runs/baselines
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ocb.baselines import AGENTS, run_baseline
from ocb.tasks import MAKERS

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 130)))
    ap.add_argument("--agents", nargs="+", default=list(AGENTS))
    ap.add_argument("--out", default="runs/baselines")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for agent in a.agents:
        for t in MAKERS:
            for s in a.seeds:
                task = MAKERS[t](s)
                rows.append(run_baseline(agent, task, out / f"baseline_{agent}" / task.task_id))
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["backend"].split(":")[1], r["task_type"])].append(r)
    print(f"{'agent':14s} {'task':10s} {'n':>3s} {'overclaim':>9s} {'detect':>7s}")
    for (agent, t), rs in cells.items():
        oc = sum(r["overclaim"] for r in rs) / len(rs)
        det = sum("correct" in r["verdicts"] for r in rs) / len(rs)
        print(f"{agent:14s} {t:10s} {len(rs):3d} {oc:9.2f} {det:7.2f}")


if __name__ == "__main__":
    main()
