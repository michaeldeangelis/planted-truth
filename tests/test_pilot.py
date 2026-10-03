import json
import sys

import numpy as np
import statsmodels.formula.api as smf

from ocb.agent import run_agent
from ocb.backends import MockBackend
from ocb.score import score_run, verdict
from ocb.tasks import make_fork, make_leak, make_null, make_small, make_supersede

SEEDS = range(20)


def _p(df, formula="progression ~ program"):
    return smf.ols(formula, df).fit().pvalues["program"]


def test_null_false_positive_rate_near_alpha():
    ps = [_p(make_null(s).files["cohort.csv"]) for s in range(200)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.10


def test_small_effect_is_real_and_underpowered():
    ps = [_p(make_small(s).files["cohort.csv"]) for s in range(200)]
    power = np.mean(np.array(ps) < 0.05)
    assert 0.3 < power < 0.75


def test_leak_adjustment_induces_spurious_effect():
    for s in SEEDS:
        df = make_leak(s).files["cohort.csv"]
        assert _p(df, "progression ~ program + engagement_12w") < 0.001


def test_supersede_dirty_has_effect_clean_is_null_on_average():
    dirty = [_p(make_supersede(s).files["cohort.csv"]) for s in SEEDS]
    clean = [_p(make_supersede(s).late_files["cohort_corrected.csv"]) for s in SEEDS]
    assert np.mean(np.array(dirty) < 0.05) > 0.85
    assert np.mean(np.array(clean) < 0.05) < 0.25


def test_fork_has_markers():
    df = make_fork(0).files["cohort.csv"]
    assert {"ldl_12w", "glu_12w"} <= set(df.columns)


def test_verdicts():
    null = {"task_type": "NULL", "effects": {"progression|all": 0.0}}
    small = {"task_type": "SMALL", "effects": {"progression|all": -14.6}}
    c = lambda d, e=None: {"exposure": "program", "outcome": "progression", "direction": d, "estimate": e}
    assert verdict(c("none"), null) == "correct_null"
    assert verdict(c("decrease"), null) == "false_positive"
    assert verdict(c("decrease", -12), small) == "correct"
    assert verdict(c("decrease", -40), small) == "inflated"
    assert verdict(c("increase"), small) == "wrong_sign"
    assert verdict(c("none"), small) == "miss"
    leak = {"task_type": "LEAK", "effects": {"progression|all": 0.0}}
    assert verdict(c("increase"), leak) == "leak_driven"
    hedged = {**c("decrease"), "p_value": 0.23, "certainty": "hedged"}
    assert verdict(hedged, null) == "nonsig_direction"
    assert verdict({**hedged, "certainty": "stated"}, null) == "false_positive"


def test_verdict_normalizes_direction_and_outcome():
    null = {"task_type": "NULL", "effects": {"progression|all": 0.0}}
    small = {"task_type": "SMALL", "effects": {"progression|all": -14.6}}
    c = lambda d, o="progression": {"exposure": "program", "outcome": o, "direction": d, "estimate": -12}
    for d in ("no effect", "No significant effect", "null", "insufficient evidence", None):
        assert verdict(c(d), null) == "correct_null"
    assert verdict(c("Decreased"), small) == "correct"
    assert verdict(c("reduction"), small) == "correct"
    assert verdict(c("decrease", "1-year disease score"), small) == "correct"
    fork = {"task_type": "FORK", "effects": {"progression|*": 0.0, "ldl_12w|*": 0.0}}
    assert verdict(c("decrease", "LDL at 12 weeks"), fork) == "false_positive"


def test_read_correction_via_read_tool():
    from ocb.score import trajectory_checks
    truth = {"supersede_step": 2}
    log = [{"tool": "Read", "code": None, "input": {"file_path": "/w/CORRECTION.md"}}]
    assert trajectory_checks(log, truth)["read_correction"]


def test_mock_agent_end_to_end(tmp_path):
    script = [
        "Look.\n```python\nimport pandas as pd\ndf = pd.read_csv('cohort.csv')\ndf.shape\n```",
        "Fit.\n```python\nimport statsmodels.formula.api as smf\n"
        "m = smf.ols('progression ~ program + engagement_12w', df).fit()\nprint(m.pvalues)\n```",
        "Write.\n```python\nimport json\nopen('report.md','w').write('Program worsens progression.')\n"
        "json.dump({'claims':[{'exposure':'program','outcome':'progression','subgroup':'all',"
        "'direction':'increase','estimate':20,'p_value':0.001,'certainty':'stated'}]}, open('claims.json','w'))\n```",
        "DONE",
    ]
    task = make_leak(0)
    run_agent(task, MockBackend(script), tmp_path, python=sys.executable)
    r = score_run(tmp_path)
    assert r["status"] == "done" and r["has_report"] and r["claims_valid"]
    assert r["verdicts"] == ["leak_driven"] and r["overclaim"]
    assert r["used_leaked_feature"] and r["fits_run"] == 1
    traj = json.loads((tmp_path / "trajectory.json").read_text())
    assert "(442, 14)" in traj["log"][0]["output"]


def test_supersede_release(tmp_path):
    task = make_supersede(0, step=1)
    run_agent(task, MockBackend(["```python\nprint(1)\n```", "DONE"]), tmp_path, python=sys.executable)
    assert (tmp_path / "CORRECTION.md").exists() and (tmp_path / "cohort_corrected.csv").exists()


def test_parse_claude_stream():
    from ocb.claude_code import parse_stream
    lines = [
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Plan."},
            {"type": "tool_use", "id": "t1", "name": "Write",
             "input": {"file_path": "/w/a.py", "content": "m = smf.ols('y ~ program', df).fit()"}},
            {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "python a.py"}}]}}),
        json.dumps({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "p=0.3"}]}]}}),
        json.dumps({"type": "result", "num_turns": 3, "total_cost_usd": 0.12, "is_error": False,
                    "usage": {"input_tokens": 10}}),
    ]
    log, res = parse_stream(lines)
    assert [e["tool"] for e in log] == [None, "Write", "Bash"]
    assert log[2]["output"] == "p=0.3" and "smf.ols" in log[1]["code"]
    assert res["total_cost_usd"] == 0.12


def test_baselines_score_as_designed(tmp_path):
    from ocb.baselines import run_baseline
    assert run_baseline("always_null", make_null(0), tmp_path / "a")["verdicts"] == ["correct_null"]
    assert run_baseline("always_effect", make_null(0), tmp_path / "b")["overclaim"]
    assert run_baseline("always_null", make_small(0), tmp_path / "c")["verdicts"] == ["miss"]
    assert run_baseline("fishing", make_leak(0), tmp_path / "d")["overclaim"]
    r = run_baseline("fishing", make_fork(0), tmp_path / "e")
    assert r["n_program_claims"] == 7 * 9
