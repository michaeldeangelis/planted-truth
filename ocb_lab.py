"""Audit the autonomous research lab (rlab) for overclaiming.

  canary-init  build the planted-truth project (questions imported, NOT approved)
  audit        audit one lab project (add --truth for a canary project, --verify to check citations online)
  audit-all    audit every project under ~/.dev/rlab-projects and write a markdown report
  sim          run the scripted-strategy matrix through the real kernel (free)

Examples:
  .venv/bin/python ocb_lab.py canary-init
  .venv/bin/python ocb_lab.py audit-all --verify --out reports/lab-audit.md
  .venv/bin/python ocb_lab.py sim --out docs/lab-sim.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocb.lab.audit import audit_project, render_markdown
from ocb.lab.bridge import DEFAULT_PROJECTS, rlab_version

HOME = Path.home()
CANARY_ROOT = DEFAULT_PROJECTS / "diabetes-program"
CANARY_TRUTH = HOME / ".dev" / "rlab-holdout" / "dp1"  # workers see this path: keep it neutral
REGISTRY = HOME / ".dev" / "research-lab" / "registry" / "claims.jsonl"


def cmd_canary_init(a) -> int:
    from ocb.lab.canary import build_canary
    info = build_canary(a.root, a.truth, seeds=tuple(a.seeds))
    print(json.dumps(info, indent=1))
    print("\nTruth is held out of worker reach and listed as a project holdout. Approval is Mike's gate:")
    print(f"  {info['approve']}")
    return 0


def _audit(root: Path, a) -> dict:
    truth = a.truth if getattr(a, "truth", None) else (CANARY_TRUTH if root.resolve() == CANARY_ROOT.resolve() else None)
    if truth and not (Path(truth) / "index.json").exists():
        truth = None
    return audit_project(root, truth, REGISTRY if REGISTRY.exists() else None, verify=a.verify)


def cmd_audit(a) -> int:
    res = _audit(a.root, a)
    print(json.dumps(res, indent=1) if a.json else render_markdown([res]))
    return 1 if res["finding_counts"]["fail"] else 0


def cmd_audit_all(a) -> int:
    audits = [_audit(p, a) for p in sorted(Path(a.projects).expanduser().iterdir())
              if (p / "qbank" / "project.toml").is_file()]
    md = render_markdown(audits)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        print(f"wrote {a.out}: {len(audits)} projects, "
              f"{sum(x['finding_counts']['fail'] for x in audits)} fail findings")
    else:
        print(md)
    return 0


def cmd_redteam(a) -> int:
    """Score one or more attack candidates. A pass on planted nulls means the kernel was fooled."""
    from ocb.lab.redteam import DEFAULT_PREREG, evaluate_attack, load_attack
    rows = []
    for path in a.attacks:
        script, prereg = load_attack(Path(path))
        res = evaluate_attack(Path(path).stem, script, {**DEFAULT_PREREG, **prereg},
                              seeds=tuple(range(300, 300 + a.seeds)), null_runs=a.null_runs)
        s = res.summary()
        rows.append(s)
        print(f"{s['name']}: {s['classification']}  false_pass {s['false_pass']}/{s['null_questions']} "
              f"(CI {s['ci'][0]:.3f}-{s['ci'][1]:.3f})  detected {s['detected']}/{s['real_questions']}  "
              f"unfinished {s['unfinished']}", flush=True)
        print(f"  plausibility: {s['plausibility']}")
        for reason, n in s["reasons"].items():
            print(f"  {n:3d}x {reason}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 1 if any(r["false_pass"] and r["classification"] == "plausible" for r in rows) else 0


def cmd_space(a) -> int:
    """Enumerate (or sample) the declared analysis space and rank cells by false passes."""
    from ocb.lab.sim import simulate_cell
    from ocb.lab.space import all_cells, sample_cells
    cells = all_cells() if a.cells <= 0 else sample_cells(a.cells, a.seed)
    rows = []
    for i, cell in enumerate(cells, start=1):
        r = simulate_cell(cell, seeds=tuple(range(300, 300 + a.seeds)))
        sm = r["canary"]["summary"]
        fp = sm["false_pass_null_truth"] or {"k": 0, "n": 0}
        pw = sm["power_small"] or {"k": 0, "n": 0}
        rows.append({"cell": cell.key, **cell.as_dict(), "fp_k": fp["k"], "fp_n": fp["n"],
                     "power_k": pw["k"], "power_n": pw["n"], "unfinished": sm["unfinished"],
                     "fails": r["finding_counts"]["fail"], "rlab": r.get("rlab", {}).get("commit")})
        print(f"[{i}/{len(cells)}] {cell.key}: false_pass {fp['k']}/{fp['n']} power {pw['k']}/{pw['n']}", flush=True)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
    worst = sorted(rows, key=lambda r: (-(r["fp_k"] / r["fp_n"] if r["fp_n"] else 0), r["cell"]))[:a.top]
    print("\nWorst cells (ceiling of the space):")
    for r in worst:
        print(f"  {r['cell']}: {r['fp_k']}/{r['fp_n']}")
    return 0


def _pool(parts: list[dict]) -> dict:
    """Pool per-batch summaries: rates add, intervals are recomputed on the totals."""
    from ocb.lab.audit import wilson
    out = {"categories": {}, "scored": 0, "unfinished": 0, "by_type": {}}
    keys = ("false_pass_null_truth", "spurious_effect_recorded", "power_small", "bad_pass_small",
            "fabricated_answered", "settled_answered")
    for key in keys:
        k = sum((p.get(key) or {}).get("k", 0) for p in parts)
        n = sum((p.get(key) or {}).get("n", 0) for p in parts)
        lo, hi = wilson(k, n)
        out[key] = {"k": k, "n": n, "rate": round(k / n, 4) if n else None,
                    "ci": [round(lo, 4), round(hi, 4)]} if n else None
    for t in ("NULL", "FORK", "LEAK", "SMALL"):
        k = sum((p["by_type"].get(t) or {}).get("k", 0) for p in parts)
        n = sum((p["by_type"].get(t) or {}).get("n", 0) for p in parts)
        if n:
            lo, hi = wilson(k, n)
            out["by_type"][t] = {"k": k, "n": n, "rate": round(k / n, 4), "ci": [round(lo, 4), round(hi, 4)]}
    for p in parts:
        out["scored"] += p["scored"]
        out["unfinished"] += p["unfinished"]
        for c, v in p["categories"].items():
            out["categories"][c] = out["categories"].get(c, 0) + v
    return out


def cmd_sim(a) -> int:
    from ocb.lab.sim import STRATEGIES, simulate
    rows = []
    for s in STRATEGIES:
        parts, fails, warns, version = [], 0, 0, {}
        for b in range(a.batches):
            start = 300 + b * a.seeds
            r = simulate(s, seeds=tuple(range(start, start + a.seeds)))
            parts.append(r["canary"]["summary"])
            fails += r["finding_counts"]["fail"]
            warns += r["finding_counts"]["warn"]
            version = r.get("rlab", {})
            print(f"  {s} batch {b + 1}/{a.batches}: "
                  f"false_pass {parts[-1]['false_pass_null_truth']['k']}/{parts[-1]['false_pass_null_truth']['n']}"
                  f" unfinished {parts[-1]['unfinished']}", flush=True)
        sm = _pool(parts)
        rows.append((s, sm, {"fail": fails, "warn": warns}, version))
        print(f"{s}: false_pass {sm['false_pass_null_truth']['k']}/{sm['false_pass_null_truth']['n']}"
              f" power {sm['power_small']['k']}/{sm['power_small']['n']} unfinished {sm['unfinished']}", flush=True)
    if a.out:
        v = rows[0][3] if rows else {}
        lines = ["# Lab kernel vs scripted researcher strategies (no LLM)", "",
                 f"Lab: `{v.get('commit')}`" + (" (uncommitted changes present)" if v.get("dirty") else "")
                 + (" [pinned snapshot]" if v.get("pinned") else " [LIVE TREE, not pinned]")
                 + f" at `{v.get('path')}`. Re-run this table whenever the lab moves.", "",
                 f"{a.seeds * a.batches} independent cohorts per task type across {a.batches} project(s) "
                 "(NULL, FORK, LEAK = null truth; SMALL = -0.19 SD). "
                 "Verdicts come from the real rlab kernel; audit findings from `ocb/lab/audit.py`.", "",
                 "| strategy | false pass, all null-truth | NULL | FORK | LEAK | effect recorded significant though null | SMALL detected | SMALL inflated or wrong sign | audit fail/warn |",
                 "|---|---|---|---|---|---|---|---|---|"]
        cell = lambda r: f"{r['k']}/{r['n']}" if r else "-"
        for s, sm, fc, _ in rows:
            bt = sm["by_type"]
            lines.append(f"| {s} | {cell(sm['false_pass_null_truth'])} ({sm['false_pass_null_truth']['ci'][0]}-"
                         f"{sm['false_pass_null_truth']['ci'][1]}) | {cell(bt.get('NULL'))} | {cell(bt.get('FORK'))} | "
                         f"{cell(bt.get('LEAK'))} | {cell(sm['spurious_effect_recorded'])} | {cell(sm['power_small'])} | "
                         f"{cell(sm['bad_pass_small'])} | {fc['fail']}/{fc['warn']} |")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("canary-init")
    s.add_argument("--root", type=Path, default=CANARY_ROOT)
    s.add_argument("--truth", type=Path, default=CANARY_TRUTH)
    s.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302, 303, 304])
    s.set_defaults(fn=cmd_canary_init)
    s = sub.add_parser("audit")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--truth", type=Path)
    s.add_argument("--verify", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_audit)
    s = sub.add_parser("audit-all")
    s.add_argument("--projects", default=str(DEFAULT_PROJECTS))
    s.add_argument("--verify", action="store_true")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_audit_all)
    s = sub.add_parser("sim")
    s.add_argument("--seeds", type=int, default=10, help="cohorts per task type per batch (keep <= 40)")
    s.add_argument("--batches", type=int, default=1, help="independent projects to pool")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_sim)
    s = sub.add_parser("redteam")
    s.add_argument("attacks", nargs="+", help="attack files: analysis script + '--- prereg' JSON block")
    s.add_argument("--seeds", type=int, default=20, help="cohorts per task type")
    s.add_argument("--null-runs", type=int, default=0, dest="null_runs")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_redteam)
    s = sub.add_parser("space")
    s.add_argument("--cells", type=int, default=0, help="0 = every cell")
    s.add_argument("--seeds", type=int, default=5, help="cohorts per task type")
    s.add_argument("--seed", type=int, default=0, help="sampling seed when --cells > 0")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_space)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
