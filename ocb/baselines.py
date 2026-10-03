"""Scripted, LLM-free reference agents: the baseline ladder for the analysis plan.

Each writes claims.json, report.md, trajectory.json and truth.json like a real run,
so score_run applies unchanged. Expected behavior on a correct harness:
- always_null: correct_null on every null task, miss on SMALL.
- always_effect: overclaims on every null task.
- textbook: one unadjusted OLS on the headline outcome, effect claimed iff p < 0.05.
  About 5% false positives on nulls and about 50% power on SMALL. Ignores corrections.
- fishing: every outcome x subgroup, adjusted for all other columns, reports every
  p < 0.05. Inflates false positives on FORK and LEAK.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from .score import score_run
from .tasks import Task

AGENTS = ("always_null", "always_effect", "textbook", "fishing")


def _claim(outcome: str, subgroup: str, est: float, p: float) -> dict:
    direction = ("decrease" if est < 0 else "increase") if p < 0.05 else "none"
    return {"exposure": "program", "outcome": outcome, "subgroup": subgroup, "direction": direction,
            "estimate": round(est, 2), "p_value": round(p, 4), "certainty": "stated"}


def _fit(df: pd.DataFrame, y: str, covs: list[str]) -> tuple[float, float]:
    m = smf.ols(" + ".join([f"{y} ~ program", *covs]), df).fit()
    return float(m.params["program"]), float(m.pvalues["program"])


def _subgroups(df: pd.DataFrame) -> dict[str, pd.Series]:
    groups = {"all": pd.Series(True, index=df.index)}
    groups |= {f"sex={s}": df.sex == s for s in ("F", "M")}
    for c in ("age", "bmi", "bp"):
        med = df[c].median()
        groups[f"{c}<=median"], groups[f"{c}>median"] = df[c] <= med, df[c] > med
    return groups


def claims_for(agent: str, task: Task) -> list[dict]:
    df = task.files["cohort.csv"]
    if agent == "always_null":
        return [_claim("progression", "all", 0.0, 1.0)]
    if agent == "always_effect":
        return [{**_claim("progression", "all", -10.0, 0.01), "certainty": "emphatic"}]
    if agent == "textbook":
        return [_claim("progression", "all", *_fit(df, "progression", []))]
    if agent == "fishing":
        outcomes = ["progression"] + [c for c in df.columns if c.endswith("_12w") and c != "engagement_12w"]
        base = [c for c in df.columns if c not in ("patient_id", "program", "progression")
                and not (c.endswith("_12w") and c != "engagement_12w")]
        out = []
        for y in outcomes:
            for name, mask in _subgroups(df).items():
                covs = [c for c in base if not (name.startswith("sex") and c == "sex")]
                out.append(_claim(y, name, *_fit(df[mask], y, covs)))
        return out
    raise ValueError(agent)


def run_baseline(agent: str, task: Task, workdir: Path) -> dict:
    task.materialize(workdir)
    claims = claims_for(agent, task)
    (workdir / "claims.json").write_text(json.dumps({"claims": claims}, indent=1))
    (workdir / "report.md").write_text(f"Scripted baseline `{agent}`.\n")
    # Scripted agents see any late correction and ignore it.
    result = {"task_id": task.task_id, "task_type": task.task_type, "seed": task.seed,
              "backend": f"baseline:{agent}", "status": "done", "steps": 0, "turns": 0,
              "seconds": 0.0, "usage": {"input_tokens": 0, "output_tokens": 0,
                                        "cache_read_tokens": 0, "usd": 0.0},
              "late_released": task.truth.supersede_step is not None}
    (workdir / "trajectory.json").write_text(json.dumps({"result": result, "log": []}))
    (workdir / "truth.json").write_text(task.truth_json())
    return score_run(workdir)
