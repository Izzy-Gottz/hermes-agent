"""The Chrome lane's presence check on the REAL route Moe runs on: the claude-code runtime.

Moe's browser_exec calls do not run in the agent's process. The claude CLI spawns
``agent.transports.hermes_tools_mcp_server`` (``write_mcp_config``), whose ``main()`` runs
``prepare_claude_code_profile()`` — which sets ``HERMES_SINGLE_QUERY_SESSION=1`` on every turn —
and dispatches browser_exec there. The turn's origin (platform, cron, the app's
X-Hermes-Turn-Origin) lives only in the agent process's contextvars, so the MCP server must ask over
the tool bridge. This test builds exactly that: a real ToolBridge on a real
``make_tool_bridge_dispatch`` whose agent carries a turn context, a real MCP server subprocess from
the real mcp-config, spoken to over stdio JSON-RPC as the CLI does. Only the browser-use CLI is a
stub (it prints the environment it was given) and the Chrome bridge file points at this process.
"""

from __future__ import annotations

import contextvars
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

ROOT = Path(__file__).resolve().parents[2]
EXT_ID = "ljblhmlmgflmbffbfelamefleikodmjh"


def _turn_context(platform: str, origin: str, cron: str) -> contextvars.Context:
    from gateway.session_context import set_session_vars, set_turn_origin

    ctx = contextvars.copy_context()

    def bind():
        set_session_vars(platform=platform, cron_session=cron)
        set_turn_origin(origin)

    ctx.run(bind)
    return ctx


class MCPClient:
    def __init__(self, proc):
        self.proc, self.n = proc, 0

    def request(self, method, params=None, timeout=240):
        self.n += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed stdout: " + self.proc.stderr.read()[-3000:])
            msg = json.loads(line)
            if msg.get("id") == self.n:
                return msg
        raise TimeoutError(method)

    def notify(self, method):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def call(self, tool, args):
        reply = self.request("tools/call", {"name": tool, "arguments": args})
        content = (reply.get("result") or {}).get("content") or []
        return "".join(c.get("text", "") for c in content if c.get("type") == "text") or json.dumps(reply)


@pytest.fixture(scope="module")
def route(tmp_path_factory):
    from agent.claude_code_runtime import make_tool_bridge_dispatch
    from agent.transports.claude_code_session import write_mcp_config
    from agent.transports.hermes_tool_bridge import ToolBridge

    home = tmp_path_factory.mktemp("hermes-home")
    (home / "config.yaml").write_text(
        # cdp_url: the own lane points at a dead endpoint instead of launching a real browser here.
        "browser:\n  backend: browser-use\n  allow_private_urls: true\n  cdp_url: http://127.0.0.1:9\n"
        "  chrome_extension:\n    enabled: true\n")
    (home / "bin").mkdir()
    cli = home / "bin" / "browser-use"
    cli.write_text("#!/bin/sh\ncat >/dev/null\n"
                   "echo \"CDP_WS=${BU_CDP_WS:-none} CDP_URL=${BU_CDP_URL:-none} NAME=${BU_NAME:-none} "
                   "SINGLE_QUERY=${HERMES_SINGLE_QUERY_SESSION:-unset}\"\n")
    cli.chmod(0o755)
    state = home / "chrome-bridge"
    state.mkdir(mode=0o700)
    bridge_file = state / "bridge.json"
    bridge_file.write_text(json.dumps({"ws_url": "ws://127.0.0.1:5999/devtools/browser/secret", "pid": os.getpid(),
                                       "extension_origin": f"chrome-extension://{EXT_ID}/"}))
    os.chmod(bridge_file, 0o600)

    agent = SimpleNamespace(_turn_context=_turn_context("api_server", "person", ""))
    bridge = ToolBridge(make_tool_bridge_dispatch(agent), directory=str(home), allowed_tools=("todo_list",))
    bridge.start()
    cfg_path = write_mcp_config(directory=str(home), credential_env={}, bridge_socket=bridge.socket_path,
                                bridge_token=bridge.token, bridge_tools=("todo_list",))
    server = next(iter(json.loads(Path(cfg_path).read_text())["mcpServers"].values()))
    env = dict(os.environ)
    env.update(server.get("env") or {})
    env["HERMES_HOME"] = str(home)
    extra = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + extra if extra else "")
    for k in ("HERMES_CRON_SESSION", "HERMES_SINGLE_QUERY_SESSION", "HERMES_KANBAN_TASK"):
        env.pop(k, None)  # whatever the server has, it gets from prepare_claude_code_profile()
    proc = subprocess.Popen([server["command"], *server["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env, cwd=str(home))
    client = MCPClient(proc)
    init = client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                         "clientInfo": {"name": "test", "version": "1"}})
    assert "result" in init, init
    client.notify("notifications/initialized")
    tools = client.request("tools/list")["result"]["tools"]
    try:
        yield SimpleNamespace(client=client, agent=agent, tools={t["name"]: t for t in tools}, proc=proc)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        bridge.close()


def _set_turn(route, platform, origin, cron=""):
    route.agent._turn_context = _turn_context(platform, origin, cron)


CHROME_BRIDGE = "5999/devtools/browser/secret"  # the person's Chrome, via the extension bridge
AUTO = '# Opening X\nnew_tab("https://x.com/explore")\nprint(page_info())'


def test_the_server_is_the_claude_code_profile_and_offers_where(route):
    assert "where" in route.tools["browser_exec"]["inputSchema"]["properties"]


def test_a_persons_live_turn_routes_to_their_chrome(route):
    _set_turn(route, "api_server", "person")
    auto = json.loads(route.client.call("browser_exec", {"code": AUTO}))
    assert auto["lane"] == "chrome", auto
    assert "CDP_WS=ws://127.0.0.1:5999/devtools/browser/secret" in auto["output"]
    # the MCP server really is headless-approval: the marker that must NOT mean "nobody here"
    assert "SINGLE_QUERY=1" in auto["output"]
    explicit = json.loads(route.client.call("browser_exec", {"code": "print(1)", "where": "chrome"}))
    assert explicit["lane"] == "chrome" and "CDP_WS=ws://127.0.0.1:5999" in explicit["output"]


@pytest.mark.parametrize("platform,origin,cron,why", [
    ("api_server", "person", "1", "a scheduled job"),
    ("webhook", "", "", "a webhook turn"),
    ("api_server", "background", "", "a background note from the app"),
    ("api_server", "", "", "an API turn whose client did not say"),
])
def test_no_live_person_never_reaches_their_chrome(route, platform, origin, cron, why):
    _set_turn(route, platform, origin, cron)
    auto = route.client.call("browser_exec", {"code": AUTO, "session": "t" + platform[:3] + origin[:2] + cron})
    assert CHROME_BRIDGE not in auto and '"lane": "chrome"' not in auto, auto
    explicit = json.loads(route.client.call("browser_exec", {"code": "print(1)", "where": "chrome"}))
    assert "error" in explicit and why in explicit["error"], explicit
