"""Run the pilot suite. Checkpointed: completed (task, backend) runs are skipped.

  .venv/bin/python run_pilot.py --backend claude-code --model sonnet --seeds 0 1 2 3 4
  .venv/bin/python run_pilot.py --backend ollama --model qwen2.5-coder:14b --seeds 0
  OCB_BUDGET_APPROVED=<id> .venv/bin/python run_pilot.py --backend anthropic --model ... \
      --price-in X --price-out Y --price-cache-read Z --cap 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocb.agent import run_agent
from ocb.backends import AnthropicBackend, BudgetError, OllamaBackend
from ocb.claude_code import run_claude_code
from ocb.score import score_run
from ocb.tasks import MAKERS

ROOT = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "anthropic", "claude-code"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--types", nargs="+", default=list(MAKERS))
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--price-in", type=float)
    ap.add_argument("--price-out", type=float)
    ap.add_argument("--price-cache-read", type=float)
    ap.add_argument("--cap", type=float, default=50.0)
    a = ap.parse_args()

    shared = None
    if a.backend == "anthropic":
        shared = AnthropicBackend(a.model, a.price_in, a.price_out, a.price_cache_read, a.cap)

    results_path = ROOT / a.out / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    for t in a.types:
        for s in a.seeds:
            task = MAKERS[t](s)
            if a.backend == "claude-code":
                name = f"claude-code_{a.model}"
            else:
                backend = shared or OllamaBackend(a.model, seed=s)
                name = backend.name.replace(":", "_").replace("/", "_")
            wd = ROOT / a.out / name / task.task_id
            if (wd / "trajectory.json").exists():
                continue
            try:
                if a.backend == "claude-code":
                    run_claude_code(task, wd, a.model, python=sys.executable)
                else:
                    run_agent(task, backend, wd, python=sys.executable, max_steps=a.max_steps)
            except BudgetError as e:
                print(f"STOP: {e}")
                return
            r = score_run(wd)
            with results_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{task.task_id:16s} {r['status']:9s} steps={r['steps']:2d} "
                  f"verdicts={r['verdicts']} overclaim={r['overclaim']} "
                  f"tok_in={r['usage']['input_tokens']} {r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
