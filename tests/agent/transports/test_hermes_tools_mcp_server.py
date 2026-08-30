"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import inspect
from typing import get_args

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
    """Stand-in for mcp.server.MCPServer that just collects handlers."""

    def __init__(self, name, instructions=None):
        self.name, self.instructions, self.tools = name, instructions, {}

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
