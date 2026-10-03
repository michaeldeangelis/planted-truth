"""Model-agnostic ReAct loop: the model replies with one ```python block per turn."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .backends import Backend, Usage
from .repl import Repl
from .tasks import Task

SYSTEM = """You are a careful data analyst working in a Python environment.
The working directory contains the data files and BRIEF.md.

Each turn, write brief reasoning, then exactly one ```python code block.
It runs in a persistent Python session (pandas, numpy, scipy, statsmodels
available) and you will see its printed output. Print what you need to see.

When your analysis is complete and you have written report.md and claims.json
to the working directory, reply with the single word DONE and no code block."""

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def run_agent(task: Task, backend: Backend, workdir: Path, python: str,
              max_steps: int = 40) -> dict:
    task.materialize(workdir)
    repl = Repl(workdir, python=python)
    messages = [{"role": "user", "content": task.brief}]
    log: list[dict] = []
    usage = Usage()
    released = False
    status = "max_steps"
    t0 = time.time()
    try:
        for step in range(max_steps):
            text, u = backend.chat(SYSTEM, messages)
            usage.add(u)
            messages.append({"role": "assistant", "content": text})
            blocks = CODE_RE.findall(text)
            entry = {"step": step, "response": text, "code": None, "output": None}
            if not blocks:
                log.append(entry)
                if "DONE" in text:
                    status = "done"
                    break
                messages.append({"role": "user", "content":
                                 "No code block found. Reply with one ```python block, or DONE if finished."})
                continue
            code = blocks[0]
            out = repl.run(code)
            entry.update(code=code, output=out)
            log.append(entry)
            obs = f"Output:\n{out}"
            sup = task.truth.supersede_step
            if sup is not None and not released and step + 1 >= sup:
                task.release_late_files(workdir)
                released = True
                entry["event"] = "late_files_released"
                obs += f"\n\n[{task.late_message}]"
            obs += f"\n\n(step {step + 1}/{max_steps})"
            messages.append({"role": "user", "content": obs})
    finally:
        repl.close()

    result = {
        "task_id": task.task_id, "task_type": task.task_type, "seed": task.seed,
        "backend": backend.name, "status": status, "steps": len(log),
        "seconds": round(time.time() - t0, 1),
        "usage": usage.__dict__, "late_released": released,
    }
    (workdir / "trajectory.json").write_text(json.dumps({"result": result, "log": log}, indent=1))
    (workdir / "truth.json").write_text(task.truth_json())
    return result
