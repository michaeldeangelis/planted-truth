"""Run a task with Claude Code headless (`claude -p`) as the agent scaffold.

Isolation (verified by probes 2026-09-30):
- --restricted + --tools: only file/shell tools; file tools confined to the workdir;
  user/project/local settings files ignored (no user hooks, plugins, CLAUDE.md rules).
- --strict-mcp-config with an empty config: no MCP servers.
- sandbox.enabled via --settings: Bash has no network access.
- Python is named explicitly in the prompt, because the Bash tool's shell
  rebuilds PATH from the user's profile.
SUPERSEDE: a PostToolUse hook copies the staged correction files into the
workdir after N tool calls and injects the data manager's message.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .tasks import Task

TOOLS = "Bash,Read,Write,Edit,Glob,Grep"

HOOK = r'''
import json, pathlib, shutil, sys
state = pathlib.Path(sys.argv[1]); workdir = pathlib.Path(sys.argv[2]); step = int(sys.argv[3])
msg = sys.argv[4]
c = state / "count"
n = int(c.read_text()) + 1 if c.exists() else 1
c.write_text(str(n))
if n == step:
    for f in (state / "late").iterdir():
        shutil.copy(f, workdir / f.name)
    (state / "released").write_text(str(n))
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                             "additionalContext": msg}}))
'''


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)


def parse_stream(lines: list[str], max_chars: int = 4000) -> tuple[list[dict], dict]:
    """Turn stream-json lines into the harness trajectory log plus the final result event."""
    log: list[dict] = []
    by_id: dict[str, dict] = {}
    result: dict = {}
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for b in ev["message"].get("content", []):
                if b.get("type") == "text" and b.get("text", "").strip():
                    log.append({"step": len(log), "tool": None, "response": b["text"],
                                "code": None, "output": None})
                elif b.get("type") == "tool_use":
                    inp = b.get("input", {})
                    code = None
                    if b["name"] == "Bash":
                        code = inp.get("command")
                    elif b["name"] in ("Write", "Edit") and str(inp.get("file_path", "")).endswith(".py"):
                        code = inp.get("content") or inp.get("new_string")
                    e = {"step": len(log), "tool": b["name"], "response": None,
                         "input": inp, "code": code, "output": None}
                    by_id[b["id"]] = e
                    log.append(e)
        elif ev.get("type") == "user":
            content = ev.get("message", {}).get("content", [])
            for b in content if isinstance(content, list) else []:
                if b.get("type") == "tool_result" and b.get("tool_use_id") in by_id:
                    out = _text(b.get("content"))
                    by_id[b["tool_use_id"]]["output"] = out[:max_chars]
        elif ev.get("type") == "result":
            result = ev
    return log, result


def run_claude_code(task: Task, workdir: Path, model: str, python: str = sys.executable,
                    max_seconds: int = 1800, claude: str = "claude") -> dict:
    task.materialize(workdir)
    state = Path(tempfile.mkdtemp(prefix=f"ocb-{task.task_id}-"))
    try:
        settings: dict = {"sandbox": {"enabled": True}}
        if task.truth.supersede_step is not None:
            (state / "late").mkdir()
            task.release_late_files(state / "late")
            hook_py = state / "hook.py"
            hook_py.write_text(HOOK)
            cmd = " ".join(json.dumps(a) for a in [
                "/usr/bin/python3", str(hook_py), str(state), str(workdir.resolve()),
                str(task.truth.supersede_step), task.late_message])
            settings["hooks"] = {"PostToolUse": [{"matcher": "*", "hooks": [
                {"type": "command", "command": cmd}]}]}
        (state / "settings.json").write_text(json.dumps(settings))
        (state / "mcp.json").write_text('{"mcpServers": {}}')

        prompt = (task.brief + f"\n\nEnvironment: run Python with {python} "
                  "(pandas, numpy, scipy, statsmodels, matplotlib installed). "
                  "The data files are in the current directory.")
        argv = [claude, "-p", prompt, "--model", model, "--restricted", "--tools", TOOLS,
                "--allowedTools", TOOLS.replace(",", " "), "--settings", str(state / "settings.json"),
                "--strict-mcp-config", "--mcp-config", str(state / "mcp.json"),
                "--no-session-persistence", "--output-format", "stream-json", "--verbose"]
        t0 = time.time()
        status = "done"
        try:
            p = subprocess.run(argv, cwd=workdir, stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=max_seconds)
            stdout = p.stdout
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            status = "timeout"
        (workdir.parent / f"{task.task_id}.stream.jsonl").write_text(stdout)
        log, res = parse_stream(stdout.splitlines())
        if res.get("is_error") and status == "done":
            status = f"error:{res.get('subtype')}"
        released = (state / "released").exists()
    finally:
        shutil.rmtree(state, ignore_errors=True)

    u = res.get("usage", {})
    result = {
        "task_id": task.task_id, "task_type": task.task_type, "seed": task.seed,
        "backend": f"claude-code:{model}", "status": status,
        "steps": sum(1 for e in log if e["tool"]), "turns": res.get("num_turns"),
        "seconds": round(time.time() - t0, 1),
        "usage": {"input_tokens": u.get("input_tokens", 0),
                  "output_tokens": u.get("output_tokens", 0),
                  "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                  # API-equivalent cost reported by Claude Code; not billed on a subscription.
                  "usd": res.get("total_cost_usd", 0.0)},
        "late_released": released,
    }
    (workdir / "trajectory.json").write_text(json.dumps({"result": result, "log": log}, indent=1))
    (workdir / "truth.json").write_text(task.truth_json())
    return result
