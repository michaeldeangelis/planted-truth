"""Persistent Python REPL run as a child process inside the task workdir.

Executing agent-written code (exec/eval) is the purpose of this module; it runs in
a separate process confined to the task workdir, never in the harness process.

Protocol: one JSON line in ({"code": str}), one JSON line out ({"output": str}).
Pilot isolation is process-level only (cwd = workdir, per-cell timeout). Docker
with --network none is the v1 target.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SERVER = r'''
import contextlib, io, json, signal, sys, traceback
g = {"__name__": "__main__"}
def _alarm(*_): raise TimeoutError("cell exceeded time limit")
signal.signal(signal.SIGALRM, _alarm)
for line in sys.stdin:
    req = json.loads(line)
    buf = io.StringIO()
    signal.alarm(int(req.get("timeout", 60)))
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                import ast
                tree = ast.parse(req["code"])
                last = tree.body[-1] if tree.body else None
                if isinstance(last, ast.Expr):
                    tree.body = tree.body[:-1]
                    exec(compile(tree, "<cell>", "exec"), g)
                    val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), g)
                    if val is not None:
                        print(repr(val))
                else:
                    exec(compile(tree, "<cell>", "exec"), g)
            except BaseException:
                traceback.print_exc(limit=3)
    finally:
        signal.alarm(0)
    sys.__stdout__.write(json.dumps({"output": buf.getvalue()}) + "\n")
    sys.__stdout__.flush()
'''


class Repl:
    def __init__(self, workdir: Path, python: str = sys.executable, max_chars: int = 4000):
        self.max_chars = max_chars
        self.proc = subprocess.Popen(
            [python, "-u", "-c", SERVER], cwd=workdir, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={"PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg", "PYTHONWARNINGS": "ignore"},
        )

    def run(self, code: str, timeout: int = 60) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps({"code": code, "timeout": timeout}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            return "[repl crashed]"
        out = json.loads(line)["output"]
        if len(out) > self.max_chars:
            half = self.max_chars // 2
            out = out[:half] + f"\n...[{len(out) - self.max_chars} chars truncated]...\n" + out[-half:]
        return out or "[no output]"

    def close(self) -> None:
        self.proc.kill()
