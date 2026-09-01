"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import inspect
from typing import get_args

import pytest

from agent.transports.hermes_tools_mcp_server import (
    _signature_from_schema,
)


class TestSignatureFromSchema:
    """Test the JSON Schema -> Python signature conversion."""

    def test_simple_required_string_param(self):
        """A required string param becomes str with no default."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        sig, annots = _signature_from_schema(schema)

        assert len(sig.parameters) == 1
        param = sig.parameters["query"]
        assert param.name == "query"
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert annots["query"] == str
        assert param.default is inspect.Parameter.empty



    def test_skip_private_params(self):
        """Params starting with '_' are excluded from the signature."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "_internal": {"type": "string"},
            },
            "required": ["query", "_internal"],
        }
        sig, annots = _signature_from_schema(schema)

        assert "_internal" not in sig.parameters
        assert "_internal" not in annots
        assert "query" in sig.parameters

    def test_all_json_types(self):
        """All JSON schema types map to correct Python types."""
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
            },
            "required": ["s", "i", "n", "b", "a", "o"],
        }
        sig, annots = _signature_from_schema(schema)

        assert annots["s"] == str
        assert annots["i"] == int
        assert annots["n"] == float
        assert annots["b"] == bool
        assert annots["a"] == list
        assert annots["o"] == dict








class TestModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m
        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )

    def test_claude_code_profile_adds_os_tools(self):
        from agent.transports.hermes_tools_mcp_server import (
            CLAUDE_CODE_OS_TOOLS, EXPOSED_TOOLS, exposed_tools_for_profile,
        )
        assert exposed_tools_for_profile(None) == EXPOSED_TOOLS
        assert exposed_tools_for_profile("codex") == EXPOSED_TOOLS
        cc = exposed_tools_for_profile("claude-code")
        assert set(CLAUDE_CODE_OS_TOOLS) <= set(cc) and set(EXPOSED_TOOLS) <= set(cc)
        assert {"terminal", "read_file", "write_file", "patch", "search_files", "process"} <= set(cc)

    def test_scrub_environment_drops_named_and_blanked_keys(self):
        from agent.transports.hermes_tools_mcp_server import SCRUB_ENV, scrub_environment
        env = {SCRUB_ENV: "CLAUDE_CODE_OAUTH_TOKEN,OTHER", "CLAUDE_CODE_OAUTH_TOKEN": "",
               "OTHER": "x", "KEEP": "y"}
        assert sorted(scrub_environment(env)) == ["CLAUDE_CODE_OAUTH_TOKEN", "OTHER"]
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env and "OTHER" not in env and env["KEEP"] == "y"
        assert scrub_environment({}) == []


class _RecordingServer:
    """Stand-in for mcp.server.MCPServer that just collects handlers.

    Accepts the mcp 2.0 keywords (``meta``, ``structured_output``) and
    records ``meta`` per tool, so the always-load contract is testable."""

    def __init__(self, name, instructions=None):
        self.name, self.instructions, self.tools = name, instructions, {}
        self.meta = {}

    def add_tool(self, fn, name=None, description=None, meta=None, structured_output=None):
        self.tools[name or fn.__name__] = fn
        self.meta[name or fn.__name__] = meta


class _MetalessServer(_RecordingServer):
    """An older SDK whose add_tool() knows neither meta nor structured_output."""

    def add_tool(self, fn, name=None, description=None):
        self.tools[name or fn.__name__] = fn


class TestClaudeCodeProfileGuards:
    def _build(self, monkeypatch, profile):
        import mcp.server as mcp_server
        monkeypatch.setattr(mcp_server, "MCPServer", _RecordingServer)
        from agent.transports import hermes_tools_mcp_server as m
        return m._build_server(profile)

    def test_dangerous_terminal_command_consults_hermes_command_guards(self, monkeypatch):
        """(b) A dangerous command via the MCP `terminal` tool goes through
        tools.approval.check_all_command_guards, not around it."""
        import tools.terminal_tool as tt
        consulted = []

        def guard(command, env_type, approval_callback=None, has_host_access=False):
            consulted.append(command)
            return {"approved": False, "description": "blocked-by-test", "message": "blocked-by-test"}

        monkeypatch.setattr(tt, "_check_all_guards_impl", guard)
        server = self._build(monkeypatch, "claude-code")
        assert "terminal" in server.tools and "read_file" in server.tools
        assert "Bash" in (server.instructions or "")
        import json
        out = json.loads(server.tools["terminal"](command="rm -rf /tmp/hermes-never-runs"))
        assert consulted == ["rm -rf /tmp/hermes-never-runs"]
        assert out["status"] == "blocked" and out["exit_code"] != 0

    def test_claude_code_profile_registers_config_hooks_and_marks_headless(self, monkeypatch, tmp_path):
        """The server is spawned by the claude CLI, so nothing else registers
        config.yaml pre_tool_call hooks or marks the process as having no
        approver. Without both, Moe's confirm-send hook never fires and
        check_all_command_guards approves dangerous commands unprompted."""
        from agent.transports import hermes_tools_mcp_server as m
        import agent.shell_hooks as sh
        import hermes_cli.config as cfgmod
        seen = {}
        monkeypatch.setattr(cfgmod, "load_config", lambda: {"hooks": {"pre_tool_call": []}, "hooks_auto_accept": True})
        monkeypatch.setattr(sh, "register_from_config", lambda cfg, accept_hooks=False: seen.update(cfg=cfg) or ["spec"])
        env = {}
        assert m.prepare_claude_code_profile(env) == ["spec"]
        assert env[m.HEADLESS_APPROVAL_ENV] == "1" and seen["cfg"]["hooks_auto_accept"] is True

    def test_dangerous_command_is_denied_headless_with_real_guards(self, monkeypatch, tmp_path):
        """No mocks on the guard: in the headless profile a dangerous command
        is refused with a message the model sees, and nothing runs."""
        from agent.transports import hermes_tools_mcp_server as m
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.setenv(m.HEADLESS_APPROVAL_ENV, "1")
        victim = tmp_path / "victim"
        victim.mkdir()
        server = self._build(monkeypatch, "claude-code")
        import json
        out = json.loads(server.tools["terminal"](command=f"rm -rf {victim}"))
        assert victim.exists(), "dangerous command ran unprompted in the headless MCP server"
        assert out.get("exit_code") not in (0, "0") and "BLOCKED" in json.dumps(out)
        ok = json.loads(server.tools["terminal"](command="echo still-works"))
        assert "still-works" in ok.get("output", "")

    def test_default_profile_has_no_terminal(self, monkeypatch):
        server = self._build(monkeypatch, None)
        assert "terminal" not in server.tools and "web_search" in server.tools






class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1


class TestExternalMcpPassthrough:
    """The user's own MCP servers reach the child.

    Regression for the failure where a connected server (`pulse`) was
    invisible to Moe's brain: the bridge never ran MCP discovery, and the
    assembled tool catalog hid every external tool behind `tool_search`.
    """

    def _build(self, monkeypatch, defs, discovered=("mcp__pulse__run_sql",)):
        import mcp.server as mcp_server
        from agent.transports import hermes_tools_mcp_server as m
        import model_tools

        monkeypatch.setattr(mcp_server, "MCPServer", _RecordingServer)
        monkeypatch.setattr(m, "discover_external_mcp_servers", lambda: list(discovered))
        seen = {}

        def fake_defs(quiet_mode=True, **kw):
            seen.update(kw)
            return [{"type": "function", "function": {"name": n, "parameters": {}}} for n in defs]

        monkeypatch.setattr(model_tools, "get_tool_definitions", fake_defs)
        return m._build_server("claude-code"), seen

    def test_external_mcp_tools_are_exposed_without_being_allowlisted(self, monkeypatch):
        server, _ = self._build(monkeypatch, ["terminal", "mcp__pulse__run_sql"])
        # Not in EXPOSED_TOOLS or CLAUDE_CODE_OS_TOOLS, and exposed anyway.
        from agent.transports.hermes_tools_mcp_server import (
            CLAUDE_CODE_OS_TOOLS,
            EXPOSED_TOOLS,
        )
        assert "mcp__pulse__run_sql" not in EXPOSED_TOOLS + CLAUDE_CODE_OS_TOOLS
        assert "mcp__pulse__run_sql" in server.tools
        assert "terminal" in server.tools

    def test_catalog_is_read_after_assembly_so_the_prompt_stays_affordable(self, monkeypatch):
        """The assembled catalogue, deliberately.

        Reading it pre-assembly handed the child every tool name Hermes knew.
        Measured on a live config with one large MCP server: 87,907 input
        tokens for a one-word turn against 8,887 with no MCP config — ~79k
        tokens and ~14x the cost on every uncached turn, before the user has
        said anything. tool_search is Hermes' own answer, its bridge tools
        dispatch from this stateless process, and every deferred tool stays
        callable through them.
        """
        _, kwargs = self._build(monkeypatch, ["terminal", "mcp__pulse__run_sql"])
        assert kwargs.get("skip_tool_search_assembly") in (None, False)

    def test_the_connectors_the_user_set_up_are_never_deferred(self):
        """whatsapp_send and its siblings must stay in the eager surface.

        They are registered by the moe-connectors plugin, and a plugin
        toolset is deferrable by default — which is how "Moe cannot read your
        WhatsApp" happened while the tool sat in the registry. Pinned in
        tool_search rather than worked around in the bridge.
        """
        from tools.tool_search import _DIRECT_SURFACE_TOOLSETS, is_deferrable_tool_name
        assert "connectors" in _DIRECT_SURFACE_TOOLSETS
        from tools.registry import registry
        for name in ("whatsapp_send", "telegram_send", "gmail_send"):
            if registry.get_toolset_for_tool(name) == "connectors":
                assert not is_deferrable_tool_name(name), name

    def test_no_external_servers_leaves_the_surface_unchanged(self, monkeypatch):
        server, _ = self._build(monkeypatch, ["terminal"], discovered=())
        assert [n for n in server.tools if n.startswith("mcp__")] == []

    def test_discovery_failure_still_serves_hermes_own_tools(self, monkeypatch):
        from agent.transports import hermes_tools_mcp_server as m
        import tools.mcp_tool as mcp_tool

        def boom():
            raise RuntimeError("server unreachable")

        monkeypatch.setattr(mcp_tool, "discover_mcp_tools", boom)
        assert m.discover_external_mcp_servers() == []


class TestAgentLoopToolsOverTheBridge:
    """The four tools that need the live agent are offered when — and only
    when — there is a bridge to run them on.

    Withholding them is what made Moe answer "The delegate_task tool is not
    available in this environment" while running on Claude Code: the model
    had no subagent tool at all. Offering them without a bridge would be
    worse — every call would fail with "must be handled by the agent loop".
    """

    def _defs(self, names):
        return [
            {"type": "function", "function": {"name": n, "parameters": {}}}
            for n in names
        ]

    def _build(self, monkeypatch, names, *, bridge=None, offered=None):
        import mcp.server as mcp_server
        import model_tools
        from agent.transports import hermes_tools_mcp_server as m
        from agent.transports.hermes_tool_bridge import (
            BRIDGE_SOCKET_ENV, BRIDGE_TOKEN_ENV, BRIDGE_TOOLS_ENV,
        )

        monkeypatch.setattr(mcp_server, "MCPServer", _RecordingServer)
        monkeypatch.setattr(m, "discover_external_mcp_servers", lambda: [])
        monkeypatch.setattr(
            model_tools, "get_tool_definitions", lambda **kw: self._defs(names)
        )
        monkeypatch.delenv(BRIDGE_TOOLS_ENV, raising=False)
        if bridge is None:
            monkeypatch.delenv(BRIDGE_SOCKET_ENV, raising=False)
            monkeypatch.delenv(BRIDGE_TOKEN_ENV, raising=False)
        else:
            monkeypatch.setenv(BRIDGE_SOCKET_ENV, bridge.socket_path)
            monkeypatch.setenv(BRIDGE_TOKEN_ENV, bridge.token)
            if offered is not None:
                monkeypatch.setenv(BRIDGE_TOOLS_ENV, ",".join(offered))
        return m._build_server("claude-code")

    def test_offered_only_with_a_bridge(self):
        from agent.transports.hermes_tools_mcp_server import (
            AGENT_LOOP_TOOLS, tools_to_offer,
        )
        available = set(AGENT_LOOP_TOOLS) | {"terminal", "web_search"}
        without = tools_to_offer("claude-code", available)
        assert not set(AGENT_LOOP_TOOLS) & set(without)
        with_bridge = tools_to_offer("claude-code", available, bridge=True)
        assert set(AGENT_LOOP_TOOLS) <= set(with_bridge)
        # Codex still keeps its own shell and file tools out of the surface.
        codex = tools_to_offer("codex", available, bridge=True)
        assert "terminal" not in codex and "delegate_task" in codex

    def test_only_the_names_the_parent_will_actually_run_are_offered(self):
        """A leaf subagent's DELEGATE_BLOCKED_TOOLS deny delegate_task and
        memory. Advertising them anyway costs the child a turn and contradicts
        its own instructions."""
        from agent.transports.hermes_tools_mcp_server import (
            AGENT_LOOP_TOOLS, tools_to_offer,
        )
        available = set(AGENT_LOOP_TOOLS) | {"terminal"}
        narrowed = tools_to_offer(
            "claude-code", available, bridge=("todo", "session_search")
        )
        assert "todo" in narrowed and "session_search" in narrowed
        assert "delegate_task" not in narrowed and "memory" not in narrowed

    def test_a_narrowed_child_is_not_told_it_has_a_subagent_tool(self, monkeypatch, tmp_path):
        from agent.transports.hermes_tool_bridge import ToolBridge

        bridge = ToolBridge(lambda *_: "ok", directory=str(tmp_path))
        bridge.start()
        try:
            server = self._build(
                monkeypatch, ["todo", "delegate_task"], bridge=bridge,
                offered=("todo",),
            )
            assert "todo" in server.tools and "delegate_task" not in server.tools
            assert "delegate_task is this session's subagent tool" not in server.instructions
            assert "todo" in server.instructions
        finally:
            bridge.close()

    def test_a_bridged_call_runs_on_the_parent_not_here(self, monkeypatch, tmp_path):
        import model_tools
        from agent.transports.hermes_tool_bridge import ToolBridge

        seen = []
        bridge = ToolBridge(
            lambda tool, args: seen.append((tool, args)) or "from-the-parent",
            directory=str(tmp_path),
        )
        bridge.start()
        try:
            monkeypatch.setattr(
                model_tools,
                "handle_function_call",
                lambda *a, **k: pytest.fail("a bridged tool must not dispatch locally"),
            )
            server = self._build(monkeypatch, ["delegate_task", "web_search"], bridge=bridge)
            assert "delegate_task" in server.tools
            out = server.tools["delegate_task"](goal="say pong", context=None)
            assert out == "from-the-parent"
            # None-valued optionals are dropped before the call, as locally.
            assert seen == [("delegate_task", {"goal": "say pong"})]
        finally:
            bridge.close()

    def test_a_dead_bridge_is_a_tool_error_the_model_can_read(self, monkeypatch, tmp_path):
        import json as _json
        from agent.transports.hermes_tool_bridge import ToolBridge

        bridge = ToolBridge(lambda *_: "", directory=str(tmp_path))
        bridge.start()
        server = self._build(monkeypatch, ["delegate_task"], bridge=bridge)
        bridge.close()
        out = _json.loads(server.tools["delegate_task"](goal="x"))
        assert out["tool"] == "delegate_task" and "not answering" in out["error"]

    def test_unbridged_tools_still_dispatch_locally(self, monkeypatch, tmp_path):
        import model_tools
        from agent.transports.hermes_tool_bridge import ToolBridge

        bridge = ToolBridge(lambda *_: pytest.fail("web_search is not bridged"),
                            directory=str(tmp_path))
        bridge.start()
        try:
            monkeypatch.setattr(
                model_tools, "handle_function_call", lambda name, args: f"local:{name}"
            )
            server = self._build(monkeypatch, ["web_search"], bridge=bridge)
            assert server.tools["web_search"](query="x") == "local:web_search"
        finally:
            bridge.close()


class TestLocalToolsHoldTheLine:
    """A tool the child runs itself must tell the parent it is working.

    Without it a `terminal` call longer than claude_code.silence_timeout looks
    exactly like a dead CLI, and the parent retires the session in the middle
    of its own command. The hold is the only signal that distinguishes them —
    so deleting the `with bridge_hold():` wrapper has to fail a test.
    """

    def test_an_unbridged_tool_holds_the_bridge_while_it_runs(self, monkeypatch, tmp_path):
        import threading
        import time

        import mcp.server as mcp_server
        import model_tools
        from agent.transports import hermes_tools_mcp_server as m
        from agent.transports.hermes_tool_bridge import (
            BRIDGE_SOCKET_ENV, BRIDGE_TOKEN_ENV, ToolBridge,
        )

        bridge = ToolBridge(lambda t, a: t, directory=str(tmp_path))
        bridge.start()
        seen = []

        def slow_local(name, args):
            seen.append(bridge.active)
            time.sleep(0.2)
            seen.append(bridge.active)
            return "local"

        monkeypatch.setattr(mcp_server, "MCPServer", _RecordingServer)
        monkeypatch.setattr(m, "discover_external_mcp_servers", lambda: [])
        monkeypatch.setattr(
            model_tools, "get_tool_definitions",
            lambda **kw: [{"type": "function",
                           "function": {"name": "terminal", "parameters": {}}}],
        )
        monkeypatch.setattr(model_tools, "handle_function_call", slow_local)
        monkeypatch.setenv(BRIDGE_SOCKET_ENV, bridge.socket_path)
        monkeypatch.setenv(BRIDGE_TOKEN_ENV, bridge.token)
        try:
            server = m._build_server("claude-code")
            assert not bridge.active
            assert server.tools["terminal"](command="ls") == "local"
            assert seen == [True, True], (
                "the parent was not told the child was working: seen=%r" % (seen,)
            )
            deadline = time.time() + 5
            while bridge.active and time.time() < deadline:
                time.sleep(0.02)
            assert not bridge.active
        finally:
            bridge.close()


class TestTheChildSeesWhatHermesWouldShow:
    """The core-tool guarantee has to survive the process boundary.

    Hermes' own loop never defers a core tool (tools/tool_search.py: "Always-
    load means always-load. No exceptions."). The claude-code child reaches
    this server over MCP, and the CLI defers EVERY MCP tool behind its own
    ToolSearch by default — measured on 2.1.252 with a three-tool stub. So the
    model ran ToolSearch to find `terminal`, and `memory` sat one search away
    among eighty names and was never called in 274 sessions.

    Two mechanisms hold the line: the per-tool meta this server emits and the
    server-level alwaysLoad write_mcp_config() writes. Either alone keeps the
    contract; test_claude_code_tool_deferral_live.py checks the real CLI
    honours them. The GUI-only tools are withheld for the opposite reason:
    they can never work here, and they were 14% of the surface.
    """

    def _defs(self, names):
        return [{"type": "function", "function": {"name": n, "parameters": {}}} for n in names]

    def _build(self, monkeypatch, names, *, server_cls=_RecordingServer):
        import mcp.server as mcp_server
        import model_tools
        from agent.transports import hermes_tools_mcp_server as m
        from agent.transports.hermes_tool_bridge import (
            BRIDGE_SOCKET_ENV, BRIDGE_TOKEN_ENV, BRIDGE_TOOLS_ENV,
        )
        monkeypatch.setattr(mcp_server, "MCPServer", server_cls)
        monkeypatch.setattr(m, "discover_external_mcp_servers", lambda: [])
        monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: self._defs(names))
        for var in (BRIDGE_SOCKET_ENV, BRIDGE_TOKEN_ENV, BRIDGE_TOOLS_ENV):
            monkeypatch.delenv(var, raising=False)
        return m._build_server("claude-code")

    def test_every_offered_tool_is_marked_always_load(self, monkeypatch):
        from agent.transports.hermes_tools_mcp_server import ALWAYS_LOAD_META
        server = self._build(
            monkeypatch,
            ["terminal", "read_file", "clarify", "gmail_send", "mcp__pulse__run_sql",
             "tool_search", "tool_describe", "tool_call"],
        )
        assert server.tools, "nothing registered"
        for name in server.tools:
            assert server.meta[name] == ALWAYS_LOAD_META, name
        # The Hermes-side catalogue bridge is eager too: it is how the model
        # reaches everything Hermes deferred, so it must not itself be deferred.
        assert {"tool_search", "tool_describe", "tool_call"} <= set(server.tools)

    def test_the_meta_key_is_the_one_the_cli_reads(self):
        from agent.transports.hermes_tools_mcp_server import ALWAYS_LOAD_META
        assert ALWAYS_LOAD_META == {"anthropic/alwaysLoad": True}

    def test_an_sdk_without_meta_still_registers_every_tool(self, monkeypatch):
        """Older add_tool() signatures degrade to "registered, unmarked" —
        the server-level alwaysLoad in the mcp-config then carries it — and
        never to "missing"."""
        server = self._build(monkeypatch, ["terminal", "clarify"], server_cls=_MetalessServer)
        assert set(server.tools) >= {"terminal", "clarify"}

    def test_gui_only_tools_are_withheld(self, monkeypatch):
        from tools.registry import registry
        toolsets = {"tip": "desktop_ui", "tour": "desktop_ui", "desktop_project": "project",
                    "terminal": "terminal"}
        monkeypatch.setattr(registry, "get_toolset_for_tool", lambda n: toolsets.get(n))
        server = self._build(monkeypatch, ["terminal", "tip", "tour", "desktop_project"])
        assert "terminal" in server.tools
        assert not {"tip", "tour", "desktop_project"} & set(server.tools)

    def test_tools_to_offer_subtracts_gui_only_by_toolset(self):
        from agent.transports.hermes_tools_mcp_server import GUI_ONLY_TOOLSETS, tools_to_offer
        assert GUI_ONLY_TOOLSETS == {"desktop_ui", "project"}
        toolsets = {"tip": "desktop_ui", "desktop_project": "project"}
        out = tools_to_offer(
            "claude-code", {"terminal", "tip", "desktop_project", "web_search"},
            toolset_of=toolsets.get,
        )
        assert out == ("terminal", "web_search")

    def test_unknown_names_are_not_mistaken_for_gui_tools(self):
        """A name the registry does not know (an external MCP tool, a test
        stub) must pass — withholding on a lookup miss would recreate "the
        child could not see a single server the user had connected"."""
        from agent.transports.hermes_tools_mcp_server import tools_to_offer
        out = tools_to_offer("claude-code", {"mcp__pulse__run_sql", "terminal"}, toolset_of=lambda n: None)
        assert out == ("mcp__pulse__run_sql", "terminal")

    def test_the_live_registry_classifies_the_known_gui_tools(self):
        """Pin the names measured on the install so a rename upstream shows
        up here rather than as 14% of the child's prompt coming back."""
        from tools.registry import registry
        from agent.transports.hermes_tools_mcp_server import GUI_ONLY_TOOLSETS
        import model_tools  # noqa: F401 — registers the toolsets
        for name in ("annotate_preview", "focus_pane", "read_terminal", "tip", "tour", "setup_mcp"):
            ts = registry.get_toolset_for_tool(name)
            if ts is None:
                continue  # not registered in this build — nothing to withhold
            assert ts in GUI_ONLY_TOOLSETS, (name, ts)

