"""Does the real Claude Code CLI honour the always-load contract?

The unit tests pin what this repo emits (per-tool ``_meta`` in
hermes_tools_mcp_server, server-level ``alwaysLoad`` in write_mcp_config). This
file pins what the CLI *does* with it, because that is the half that can
change under us with a ``claude`` upgrade and the half that failed silently
for weeks: by default the CLI defers every MCP tool behind its own ToolSearch
regardless of count — three tools of sixty characters were deferred, the same
as eighty-two — and Hermes' "core tools are never deferred" guarantee stopped
at the process boundary without a single log line.

Each case spawns ``claude -p`` against a stub MCP server, then reads the
child's own transcript (``~/.claude/projects/*/<session-id>.jsonl``) for
``deferred_tools_delta`` attachments — the record never appears on the
stream-json stdout, only on disk — and checks the tool_use sequence for a
probe prompt: a directly-loaded tool is called without a preceding
``ToolSearch``.

Marked ``integration`` (excluded by the default ``-m 'not integration'``) and
skipped unless ``claude`` is on PATH and ``HERMES_LIVE_CLAUDE_TESTS=1``: each
case spends one haiku call on the user's account. Run with::

    HERMES_LIVE_CLAUDE_TESTS=1 pytest -m integration \
        tests/agent/transports/test_claude_code_tool_deferral_live.py -v
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HERMES_LIVE_CLAUDE_TESTS") != "1" or not shutil.which("claude"),
        reason="needs HERMES_LIVE_CLAUDE_TESTS=1 and `claude` on PATH (spends a haiku call)",
    ),
]

STUB = textwrap.dedent('''
    """Stub MCP stdio server. argv: <n_tools> <n_marked_always_load>"""
    import sys
    from mcp.server import MCPServer
    n = int(sys.argv[1]); marked = int(sys.argv[2])
    mcp = MCPServer("stub", instructions="stub server")
    def make(i):
        def handler(text: str = "") -> str:
            return f"tool_{i:03d} ok: {text}"
        handler.__name__ = f"tool_{i:03d}"
        return handler
    for i in range(n):
        h = make(i)
        meta = {"anthropic/alwaysLoad": True} if i < marked else None
        mcp.add_tool(h, name=h.__name__, description=f"Stub tool {i}; echoes text. " * 20,
                     meta=meta, structured_output=False)
    mcp.run()
''')


def _child_env() -> dict[str, str]:
    # A nested `claude` must not inherit this session's identity or the
    # CLAUDECODE guard; everything else (PATH, HOME, the login) it needs.
    return {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}


def _run(mcp_config: Path, prompt: str, *, extra_env: dict | None = None) -> dict:
    sid = str(uuid.uuid4())
    cmd = [
        "claude", "-p", "--verbose", "--output-format", "stream-json",
        "--strict-mcp-config", "--model", "haiku", "--setting-sources", "",
        "--tools", "ToolSearch", "--session-id", sid, "--mcp-config", str(mcp_config),
    ]
    env = _child_env()
    env.update(extra_env or {})
    proc = subprocess.run(cmd, env=env, input=prompt, capture_output=True, text=True, timeout=240)
    tool_uses: list[str] = []
    servers = []
    for line in proc.stdout.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("type") == "system" and rec.get("subtype") == "init":
            servers = [(m.get("name"), m.get("status")) for m in rec.get("mcp_servers") or []]
        if rec.get("type") == "assistant":
            for block in rec.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_uses.append(block.get("name") or "")
    deferred: list[str] = []
    for path in glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl")):
        for line in open(path):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            att = rec.get("attachment")
            if rec.get("type") == "attachment" and isinstance(att, dict) \
                    and att.get("type") == "deferred_tools_delta":
                deferred += att.get("addedNames") or []
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert servers and all(status == "connected" for _, status in servers), servers
    return {"deferred": deferred, "tool_uses": tool_uses}


@pytest.fixture
def stub(tmp_path: Path):
    script = tmp_path / "stub_mcp.py"
    script.write_text(STUB)

    def config(n_tools: int, n_marked: int, *, server_always_load: bool = False) -> Path:
        entry = {"type": "stdio", "command": sys.executable, "args": [str(script), str(n_tools), str(n_marked)]}
        if server_always_load:
            entry["alwaysLoad"] = True
        path = tmp_path / f"mcp-{n_tools}-{n_marked}-{int(server_always_load)}.json"
        path.write_text(json.dumps({"mcpServers": {"stub": entry}}))
        return path

    return config


PROBE = "Call the tool named tool_000 with text 'hi', then reply with only the tool's output."


def test_default_mode_defers_even_a_tiny_server(stub):
    """The premise. If this ever stops holding, the two mechanisms below are
    redundant and the always-load plumbing can be retired — but check, don't
    assume."""
    out = _run(stub(3, 0), "Reply with only the word: ok")
    assert len(out["deferred"]) == 3, out


def test_per_tool_meta_exempts_exactly_the_marked_tools(stub):
    out = _run(stub(5, 2), PROBE)
    assert "mcp__stub__tool_000" not in out["deferred"], out
    assert "mcp__stub__tool_001" not in out["deferred"], out
    assert {"mcp__stub__tool_002", "mcp__stub__tool_003", "mcp__stub__tool_004"} <= set(out["deferred"]), out
    # A directly loaded tool is called without a search first.
    assert out["tool_uses"] and out["tool_uses"][0] == "mcp__stub__tool_000", out["tool_uses"]


def test_server_level_always_load_exempts_the_whole_server(stub):
    out = _run(stub(5, 0, server_always_load=True), PROBE)
    assert out["deferred"] == [], out
    assert out["tool_uses"] and out["tool_uses"][0] == "mcp__stub__tool_000", out["tool_uses"]
