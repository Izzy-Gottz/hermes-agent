"""run_claude_code_turn — prompt combination and the warm-process registry.

Drives the runtime with minimal agent stand-ins and the fake `claude` CLI so
the two live gateway bugs stay fixed: the gateway's ephemeral system prompt
must reach the child, and a session must keep ONE warm process across the
per-request AIAgent instances api_server builds.
"""

from __future__ import annotations

import json
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import claude_code_runtime as rt

_FAKE = Path(__file__).parent / "transports" / "fake_claude_cli.py"


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-setup-token")
    wrapper = tmp_path / "claude"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"exec {json.dumps(sys.executable)} {json.dumps(str(_FAKE))} \"$@\"\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(rt, "_claude_code_config", lambda: {"binary": str(wrapper), "expose_hermes_tools": False})
    # Fresh registry per test.
    with rt._REGISTRY_LOCK:
        rt._REGISTRY.clear()
    yield home
    # Refills run on daemon threads and can land after the test that started
    # them; wait until no profile is still building before dropping the pool,
    # or the next test inherits a spare built for someone else's settings.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with rt._SPARE_LOCK:
            if not rt._SPARE_BUILDING:
                break
        time.sleep(0.05)
    rt.drop_spare()
    with rt._SPARE_LOCK:
        assert not rt._SPARES, "drop_spare() left a spare in the pool"
    for key in list(rt._REGISTRY):
        rt.evict_session(key)


def _agent(session_id: str, *, cached="BASE-PROMPT", ephemeral=None) -> SimpleNamespace:
    """The attributes run_claude_code_turn touches on a real AIAgent."""
    a = SimpleNamespace(
        session_id=session_id,
        _cached_system_prompt=cached,
        ephemeral_system_prompt=ephemeral,
        model="sonnet",
        api_mode="claude_code",
        provider="claude-code-cli",
        base_url="claude-code://local",
        api_key="x",
        _interrupt_requested=False,
        _interrupt_message=None,
        _skill_nudge_interval=0,
        _iters_since_skill=0,
        valid_tool_names=set(),
        _session_db=None,
        session_api_calls=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0,
        session_cache_read_tokens=0, session_cache_write_tokens=0,
        session_cost_status=None, session_cost_source=None,
        context_compressor=None,
        show_commentary=True,
    )
    a.clear_interrupt = lambda: None
    a._sync_external_memory_for_turn = lambda **kw: None
    a._spawn_background_review = lambda **kw: None
    return a


def _turn(agent, text="hello"):
    messages = [{"role": "user", "content": text}]
    return rt.run_claude_code_turn(
        agent, user_message=text, original_user_message=text,
        messages=messages, effective_task_id="t",
    )


def _prompt_file_text(session) -> str:
    return Path(session._system_prompt_path).read_text()


class TestEphemeralPrompt:
    def test_ephemeral_prompt_reaches_the_child(self):
        agent = _agent("s1", ephemeral="MARKER-EPH")
        result = _turn(agent)
        assert result["completed"] is True
        text = _prompt_file_text(agent._claude_code_session)
        assert text == "BASE-PROMPT\n\nMARKER-EPH"  # same join as conversation_loop
        assert agent._claude_code_session.system_prompt == rt.combined_system_prompt(agent)

    def test_changed_ephemeral_prompt_respawns_with_new_content(self):
        first = _agent("s2", ephemeral="MARKER-ONE")
        _turn(first)
        session = first._claude_code_session
        pid_one = session.pid
        second = _agent("s2", ephemeral="MARKER-TWO")
        _turn(second)
        assert second._claude_code_session is session  # same registry entry
        assert session.pid != pid_one  # respawned
        assert _prompt_file_text(session).endswith("MARKER-TWO")


class TestRegistry:
    def test_two_agent_instances_share_one_process(self):
        a1 = _agent("shared", ephemeral="E")
        r1 = _turn(a1, "one")
        pid = a1._claude_code_session.pid
        a2 = _agent("shared", ephemeral="E")
        r2 = _turn(a2, "two")
        assert r1["completed"] and r2["completed"]
        assert a2._claude_code_session is a1._claude_code_session
        assert a2._claude_code_session.pid == pid  # one spawn, same pid
        assert rt.registered_session_count() == 1

    def test_different_sessions_get_different_processes(self):
        a1, a2 = _agent("A"), _agent("B")
        _turn(a1)
        _turn(a2)
        assert a1._claude_code_session is not a2._claude_code_session
        assert a1._claude_code_session.pid != a2._claude_code_session.pid
        assert rt.registered_session_count() == 2

    def test_idle_timeout_evicts(self):
        a = _agent("idle")
        _turn(a)
        session = a._claude_code_session
        assert session.is_alive()
        assert rt.sweep_idle_sessions(0.0, now=time.monotonic() + 1) == 1
        assert rt.registered_session_count() == 0
        assert not session.is_alive()
        # Next turn for that session rebuilds transparently.
        b = _agent("idle")
        assert _turn(b)["completed"] is True
        assert b._claude_code_session is not session

    def test_retire_evicts(self):
        a = _agent("crash")
        result = _turn(a, "please CRASH")
        assert result["completed"] is False
        assert rt.registered_session_count() == 0
        assert a._claude_code_session is None

    def test_hooks_rebound_to_the_current_agent(self):
        a1 = _agent("hooks")
        seen1, seen2 = [], []
        a1._fire_stream_delta = seen1.append
        _turn(a1, "first")
        a2 = _agent("hooks")
        a2._fire_stream_delta = seen2.append
        _turn(a2, "second")
        assert "".join(seen1) == "echo: first"
        assert "".join(seen2) == "echo: second"


class TestRegistryHardening:
    def test_retire_happens_under_the_turn_lock(self, monkeypatch, tmp_path):
        """A waiter on a retired session must get a fresh process, never the
        one being closed underneath it."""
        import threading

        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "turn_timeout": 1.0, "silence_timeout": 0.4})
        a = _agent("retire")
        _turn(a, "warm")
        first_pid = a._claude_code_session.pid
        results = {}

        def waiter():
            b = _agent("retire")
            results["b"] = _turn(b, "after")
            results["pid"] = b._claude_code_session.pid

        t = threading.Thread(target=waiter)
        # A's turn hangs -> silence timeout -> should_retire; B is queued on the lock.
        import time as _t
        threading.Timer(0.1, t.start).start()
        ra = _turn(a, "HANG")
        t.join(timeout=30)
        assert ra["completed"] is False and "no output" in (ra["error"] or "")
        assert results["b"]["completed"] is True
        assert "exited unexpectedly" not in (results["b"]["error"] or "")
        assert results["pid"] != first_pid

    def test_max_sessions_lru_eviction(self, monkeypatch, caplog):
        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "max_sessions": 2})
        a1, a2, a3 = _agent("L1"), _agent("L2"), _agent("L3")
        _turn(a1); _turn(a2)
        s1 = a1._claude_code_session
        with caplog.at_level("INFO", logger="agent.claude_code_runtime"):
            _turn(a3)
        assert rt.registered_session_count() == 2
        assert not s1.is_alive()
        assert "evicting LRU session L1" in caplog.text
        with rt._REGISTRY_LOCK:
            assert set(rt._REGISTRY) == {"L2", "L3"}

    def test_dead_subagent_entries_never_evict_the_live_parent(self, monkeypatch):
        """Every subagent is its own session and delegate_task closes its
        child when it finishes — leaving a husk in the registry that owns no
        process, counts against max_sessions, and is NEWER than the parent's
        entry. Before the reap, three delegations off a max_sessions=2
        registry threw away the parent's warm claude to make room for the
        dead."""
        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "max_sessions": 2})
        parent = _agent("P")
        _turn(parent)
        parent_session = parent._claude_code_session
        for i in range(3):
            child = _agent(f"child-{i}")
            _turn(child)
            child._claude_code_session.close()  # what delegate_task does
        assert parent_session.is_alive()
        with rt._REGISTRY_LOCK:
            assert "P" in rt._REGISTRY

    def test_respawn_rate_guard_warns_once(self, caplog):
        with caplog.at_level("WARNING", logger="agent.claude_code_runtime"):
            for i in range(6):
                _turn(_agent("dyn", ephemeral=f"per-request-{i}"))
        warnings = [r for r in caplog.records if "changes every request" in r.getMessage()]
        assert len(warnings) == 1

    def test_shutdown_registry_closes_children_and_temp_files(self):
        a = _agent("exit")
        _turn(a)
        s = a._claude_code_session
        prompt = s._system_prompt_path
        assert prompt and Path(prompt).exists()
        rt._shutdown_registry()
        assert rt.registered_session_count() == 0
        assert not s.is_alive()
        assert not Path(prompt).exists()

    def test_prune_stale_temp_files(self, tmp_path):
        import os, time as _t
        cfg = tmp_path / "cc"
        cfg.mkdir()
        old = cfg / "system-prompt-old.md"; old.write_text("x")
        old_mcp = cfg / "hermes-claude-mcp-old.json"; old_mcp.write_text("{}")
        # A bridge removes its own directory on close, so one still here
        # belongs to a process that was killed rather than closed.
        dead = cfg / "bridge-deadbeef"; dead.mkdir(); (dead / "s.sock").write_text("")
        fresh = cfg / "system-prompt-new.md"; fresh.write_text("y")
        young = cfg / "bridge-young"; young.mkdir()
        stale = _t.time() - 2 * 24 * 3600
        for f in (old, old_mcp, dead):
            os.utime(f, (stale, stale))
        assert rt.prune_stale_temp_files(str(cfg)) == 3
        assert fresh.exists() and young.exists()
        assert not old.exists() and not old_mcp.exists() and not dead.exists()

    def test_dead_bridges_are_swept_at_the_next_session_start(self):
        """The 24-hour prune is too slow to keep the config dir honest: every
        gateway that is killed rather than closed leaves a directory. Swept on
        liveness, not age — and never one younger than the bind race."""
        import os
        import tempfile

        from agent.transports.hermes_tool_bridge import ToolBridge

        with tempfile.TemporaryDirectory(dir="/tmp") as cfg:
            live = ToolBridge(lambda *_: "ok", directory=cfg)
            live.start()
            try:
                dead = os.path.join(cfg, "bridge-dead")
                os.mkdir(dead)
                Path(dead, "s.sock").touch()
                young = os.path.join(cfg, "bridge-young")
                os.mkdir(young)  # created just now: mid-bind, hands off
                os.utime(dead, (0, 0))
                os.utime(os.path.dirname(live.socket_path), (0, 0))

                assert rt.sweep_dead_bridges(cfg) == 1
                assert not os.path.exists(dead)
                assert os.path.exists(young)
                assert os.path.exists(live.socket_path)
            finally:
                live.close()

    def test_prune_never_unlinks_a_socket_that_still_answers(self, tmp_path):
        """A socket's mtime is fixed at bind, so an age test alone would
        eventually delete the socket of a session that has merely been alive a
        long time — and its four bridged tools would start failing with ENOENT
        and no server-side signal at all."""
        import os
        import tempfile

        from agent.transports.hermes_tool_bridge import ToolBridge

        with tempfile.TemporaryDirectory(dir="/tmp") as cfg:
            bridge = ToolBridge(lambda *_: "ok", directory=cfg)
            bridge.start()
            try:
                owned = os.path.dirname(bridge.socket_path)
                os.utime(owned, (0, 0))  # ancient by every measure
                assert rt.prune_stale_temp_files(cfg) == 0
                assert os.path.exists(bridge.socket_path)
            finally:
                bridge.close()
            # Once it is gone, the husk of a killed process is swept.
            husk = os.path.join(cfg, "bridge-husk")
            os.mkdir(husk)
            Path(husk, "s.sock").touch()
            os.utime(husk, (0, 0))
            assert rt.prune_stale_temp_files(cfg) == 1
            assert not os.path.exists(husk)


class TestBridgedDelegationOnARealAgent:
    """The two promises the commit makes about a bridged delegate_task, on a
    real AIAgent rather than a stub: that it joins instead of handing back a
    handle, and that the confirm gate can still stop it."""

    def _agent(self):
        from run_agent import AIAgent

        return AIAgent(
            provider="claude-code-cli", model="haiku", quiet_mode=True,
            skip_memory=True, skip_context_files=True, platform="api_server",
            enabled_toolsets=["delegation"],
        )

    def test_it_joins_the_children_instead_of_returning_a_handle(self, monkeypatch):
        import tools.delegate_tool as dt

        captured = {}

        def fake_delegate_task(**kwargs):
            captured.update(kwargs)
            return json.dumps({"results": [{"response": "PONG"}]})

        monkeypatch.setattr(dt, "delegate_task", fake_delegate_task)
        agent = self._agent()
        try:
            out = rt.make_tool_bridge_dispatch(agent)("delegate_task", {"goal": "x"})
            assert captured["background"] is False, (
                "a bridged delegation was backgrounded: the caller is holding an "
                "open MCP call and will never be shown the async result"
            )
            assert captured["goal"] == "x"
            assert "PONG" in out
        finally:
            agent.close()

    def test_the_confirm_gate_can_deny_a_bridged_delegation(self, monkeypatch):
        """The whole reason bridged calls go through _invoke_tool rather than
        the tool function: Moe's pre_tool_call hook must still be able to say
        no, and the model must see why."""
        import hermes_cli.plugins as plugins
        import tools.delegate_tool as dt

        ran = []
        monkeypatch.setattr(dt, "delegate_task", lambda **kw: ran.append(kw) or "{}")
        monkeypatch.setattr(
            plugins, "_dispatch_pre_tool_call_hooks",
            lambda name, args, **kw: ("blocked by the confirm gate", None),
        )
        agent = self._agent()
        try:
            out = json.loads(
                rt.make_tool_bridge_dispatch(agent)("delegate_task", {"goal": "x"})
            )
            assert out["error"] == "blocked by the confirm gate"
            assert ran == [], "a blocked delegation spawned children anyway"
        finally:
            agent.close()


class TestTheRuntimeActuallyWiresTheBridge:
    """The seam between `bridged_tools_for` and a real session.

    Every other test in this file builds the pieces by hand. Deleting
    `tool_bridge_dispatch=` from `_build_session` reintroduces the whole bug
    this work exists to fix — no bridge, no delegate_task, the model told the
    tool does not exist — and, measured, every one of those tests still
    passed. This is the one that fails.
    """

    def _agent_with(self, session_id, tools):
        agent = _agent(session_id)
        agent.valid_tool_names = set(tools)
        calls = []
        agent._invoke_tool = (
            lambda name, args, task_id, call_id=None, *r, **kw:
            calls.append(name) or f"ran:{name}"
        )
        agent._calls = calls
        return agent

    def _addr(self, bridge):
        from agent.transports.hermes_tool_bridge import (
            BRIDGE_SOCKET_ENV, BRIDGE_TOKEN_ENV,
        )
        return {BRIDGE_SOCKET_ENV: bridge.socket_path, BRIDGE_TOKEN_ENV: bridge.token}

    def test_a_real_turn_starts_a_bridge_narrowed_to_that_agent(self, monkeypatch):
        from agent.transports.hermes_tool_bridge import (
            BRIDGE_TOOLS_ENV, BridgeError, call_bridged_tool,
        )
        cfg = rt._claude_code_config()
        monkeypatch.setattr(
            rt, "_claude_code_config", lambda: {**cfg, "expose_hermes_tools": True}
        )
        agent = self._agent_with("wired", ("todo_list", "session_search"))
        _turn(agent)
        session = agent._claude_code_session
        bridge = session._tool_bridge
        assert bridge is not None, "the runtime did not start a bridge"
        assert bridge.allowed_tools == ("todo_list", "session_search")

        env = json.loads(
            Path(session._mcp_config_path).read_text()
        )["mcpServers"]["hermes-tools"]["env"]
        assert env[BRIDGE_TOOLS_ENV] == "todo_list,session_search"

        addr = self._addr(bridge)
        assert call_bridged_tool("todo_list", env=addr) == "ran:todo_list"
        assert agent._calls == ["todo_list"]
        with pytest.raises(BridgeError, match="not bridged"):
            call_bridged_tool("delegate_task", {"goal": "x"}, env=addr)

    def test_a_second_turn_repoints_and_renarrows_the_warm_bridge(self, monkeypatch):
        """api_server builds an AIAgent per request against one warm process."""
        from agent.transports.hermes_tool_bridge import (
            BridgeError, call_bridged_tool,
        )
        cfg = rt._claude_code_config()
        monkeypatch.setattr(
            rt, "_claude_code_config", lambda: {**cfg, "expose_hermes_tools": True}
        )
        first = self._agent_with("warm", ("todo_list", "delegate_task"))
        _turn(first)
        bridge = first._claude_code_session._tool_bridge
        addr = self._addr(bridge)
        assert call_bridged_tool("delegate_task", {"goal": "x"}, env=addr)

        second = self._agent_with("warm", ("todo_list",))
        _turn(second)
        assert second._claude_code_session._tool_bridge is bridge
        assert bridge.allowed_tools == ("todo_list",)
        assert call_bridged_tool("todo_list", env=addr) == "ran:todo_list"
        assert first._calls == ["delegate_task"] and second._calls == ["todo_list"]
        with pytest.raises(BridgeError, match="not bridged"):
            call_bridged_tool("delegate_task", {"goal": "x"}, env=addr)

    def test_an_agent_with_no_agent_loop_tools_gets_no_bridge_at_all(self, monkeypatch):
        cfg = rt._claude_code_config()
        monkeypatch.setattr(
            rt, "_claude_code_config", lambda: {**cfg, "expose_hermes_tools": True}
        )
        from agent.transports.hermes_tool_bridge import BRIDGE_SOCKET_ENV

        agent = self._agent_with("bare", ("terminal",))
        _turn(agent)
        session = agent._claude_code_session
        assert session._tool_bridge is None
        env = json.loads(
            Path(session._mcp_config_path).read_text()
        )["mcpServers"]["hermes-tools"]["env"]
        assert BRIDGE_SOCKET_ENV not in env


class TestToolBridgeDispatch:
    """What arrives from the child, and how it is run.

    The child's delegate_task / memory / session_search / todo calls come back
    over the bridge and are dispatched HERE, on the agent that owns the
    session — which is what makes a subagent inherit this agent's provider,
    model and Claude Code credential instead of being built in the MCP
    server's process, which has neither.
    """

    def _recording_agent(self, session_id="bridge-1", tools=None):
        from agent.transports.hermes_tool_bridge import BRIDGED_TOOLS

        agent = _agent(session_id)
        agent.valid_tool_names = set(BRIDGED_TOOLS if tools is None else tools)
        calls: list[tuple] = []

        def _invoke_tool(name, args, task_id, tool_call_id=None, *rest, **kw):
            from tools.delegate_tool import synchronous_delegation_forced

            calls.append(
                (name, args, task_id, synchronous_delegation_forced(), tool_call_id)
            )
            return f"ran:{name}"

        agent._invoke_tool = _invoke_tool
        agent._calls = calls
        return agent

    def test_a_bridged_call_goes_through_invoke_tool(self):
        """Not the tool function directly: a bridged call must fire the same
        pre_tool_call hooks (Moe's confirm gate) and post-call accounting as
        a call from Hermes' own loop."""
        agent = self._recording_agent()
        agent._claude_code_task_id = "task-7"
        dispatch = rt.make_tool_bridge_dispatch(agent)
        assert dispatch("memory", {"action": "read"}) == "ran:memory"
        name, args, task_id, forced, call_id = agent._calls[0]
        assert (name, args, task_id, forced) == ("memory", {"action": "read"}, "task-7", False)
        # Hooks and post-call accounting key on a call id; two bridged calls
        # in one turn are indistinguishable without one.
        assert call_id and call_id.startswith("bridge-")

    def test_delegation_from_the_child_is_forced_synchronous(self):
        """The caller is holding an open MCP tool call and will only ever see
        this return value; a background handle would be a receipt for work it
        is never shown."""
        agent = self._recording_agent()
        dispatch = rt.make_tool_bridge_dispatch(agent)
        dispatch("delegate_task", {"goal": "x"})
        name, args, task_id, forced, _call_id = agent._calls[0]
        assert (name, forced) == ("delegate_task", True)
        # ...and the flag does not leak past the call.
        from tools.delegate_tool import synchronous_delegation_forced

        assert not synchronous_delegation_forced()

    def test_the_session_id_stands_in_for_a_missing_task_id(self):
        agent = self._recording_agent("sess-9")
        rt.make_tool_bridge_dispatch(agent)("todo_list", {})
        assert agent._calls[0][2] == "sess-9"

    def test_only_bridged_names_are_accepted(self):
        agent = self._recording_agent()
        with pytest.raises(ValueError, match="not a bridged tool"):
            rt.make_tool_bridge_dispatch(agent)("terminal", {"command": "id"})
        assert agent._calls == []

    def test_a_child_cannot_delegate_just_because_its_mcp_server_offers_it(self):
        """The MCP server builds its list from the whole registry in its own
        process and cannot know the child's toolsets, so a leaf subagent's
        `claude` IS offered delegate_task and memory — which
        DELEGATE_BLOCKED_TOOLS denies it. The agent's own surface decides."""
        agent = self._recording_agent(tools=("todo_list", "session_search"))
        agent.platform = "subagent"
        dispatch = rt.make_tool_bridge_dispatch(agent)
        for blocked in ("delegate_task", "memory"):
            out = json.loads(dispatch(blocked, {}))
            assert "not available to this agent" in out["error"]
        assert dispatch("todo_list", {}) == "ran:todo_list"
        assert [c[0] for c in agent._calls] == ["todo_list"]

    def test_an_agent_with_no_tool_list_at_all_is_not_second_guessed(self):
        """An embedder or a stand-in that never built a tool surface is taken
        as unrestricted; an agent that HAS one is taken at its word."""
        agent = self._recording_agent()
        from agent.transports.hermes_tool_bridge import BRIDGED_TOOLS

        del agent.valid_tool_names
        assert rt.bridged_tools_for(agent) == BRIDGED_TOOLS
        assert rt.make_tool_bridge_dispatch(agent)("memory", {}) == "ran:memory"

    def test_an_empty_tool_surface_grants_nothing(self):
        """`if valid and tool not in valid` would fall OPEN here — an empty
        set is falsy, and a build that resolved to no tools would silently
        hand back all four, DELEGATE_BLOCKED_TOOLS included."""
        agent = self._recording_agent(tools=())
        assert rt.bridged_tools_for(agent) == ()
        out = json.loads(rt.make_tool_bridge_dispatch(agent)("delegate_task", {}))
        assert "not available to this agent" in out["error"]
        assert agent._calls == []

    def test_the_child_is_only_offered_what_the_agent_can_run(self):
        """Same source of truth decides what the child is TOLD it has and
        what it is allowed to call — otherwise a leaf subagent is advertised
        delegate_task and refused when it uses it."""
        leaf = self._recording_agent(tools=("todo_list", "session_search"))
        assert rt.bridged_tools_for(leaf) == ("todo_list", "session_search")
        from agent.transports.hermes_tool_bridge import BRIDGED_TOOLS

        parent = self._recording_agent()
        assert set(rt.bridged_tools_for(parent)) == set(BRIDGED_TOOLS)

    def test_a_bridged_call_really_goes_through_the_tool_pipeline(self, monkeypatch):
        """The whole justification for dispatching through _invoke_tool rather
        than the tool function is that a bridged call is gated and accounted
        like any other. A stubbed _invoke_tool cannot show that, so this one
        builds a real AIAgent and watches the pre_tool_call hook fire."""
        import hermes_cli.plugins as plugins
        from run_agent import AIAgent

        seen = []

        def fake_hooks(function_name, function_args, **kw):
            seen.append((function_name, kw.get("tool_call_id"), kw.get("task_id")))
            return None, None

        monkeypatch.setattr(plugins, "_dispatch_pre_tool_call_hooks", fake_hooks)

        agent = AIAgent(
            provider="claude-code-cli", model="haiku", quiet_mode=True,
            skip_memory=True, skip_context_files=True, platform="api_server",
            enabled_toolsets=["todo"],
        )
        try:
            agent._claude_code_task_id = "task-real"
            out = rt.make_tool_bridge_dispatch(agent)(
                "todo_list", {"todos": [{"content": "bridge check", "status": "pending"}]}
            )
            assert "bridge check" in out
            assert seen and seen[0][0] == "todo_list"
            assert seen[0][1].startswith("bridge-")
            assert seen[0][2] == "task-real"
        finally:
            try:
                agent.close()
            except Exception:
                pass

    def test_a_turn_binds_the_bridge_to_the_agent_driving_it(self):
        """api_server builds an AIAgent per request against one warm process;
        a delegation dispatched onto last request's agent would build its
        children from a dead session."""
        first = self._recording_agent("shared")
        _turn(first)
        session = first._claude_code_session
        assert first._claude_code_task_id == "t"
        session._tool_bridge_dispatch("todo_list", {})
        assert [c[0] for c in first._calls] == ["todo_list"]

        second = self._recording_agent("shared")
        _turn(second)
        assert second._claude_code_session is session
        session._tool_bridge_dispatch("todo_list", {})
        assert [c[0] for c in second._calls] == ["todo_list"]
        assert [c[0] for c in first._calls] == ["todo_list"]  # not called twice


class TestPreWarmedSpare:
    """A conversation should start in a process that is already answering.

    Measured on this machine before any of this: the first turn of a new
    conversation took 4.0 s against 1.4 s for the second. None of that cost
    depends on WHICH conversation arrives — it is MCP servers connecting and
    the CLI's warm-up round trip — and `--session-id` pins a fresh session at
    spawn, so the whole process can be booted before the conversation exists.
    """

    def _wait_for_spare(self, agent=None, timeout=25.0):
        """The spare for ``agent``'s profile (or, with no agent, whichever
        lands first), waiting for the background build to finish."""
        key = rt._spare_key(agent) if agent is not None else None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with rt._SPARE_LOCK:
                if key is not None:
                    if key in rt._SPARES:
                        return rt._SPARES[key]
                elif rt._SPARES:
                    return next(iter(rt._SPARES.values()))
            time.sleep(0.05)
        return None

    def _wait_until_idle(self, timeout=25.0):
        """Block until no profile has a build in flight."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with rt._SPARE_LOCK:
                if not rt._SPARE_BUILDING:
                    return True
            time.sleep(0.05)
        return False

    def _pool_keys(self):
        with rt._SPARE_LOCK:
            return list(rt._SPARES)

    def test_a_turn_leaves_a_spare_warm_for_the_next_conversation(self):
        first = _agent("conv-1")
        _turn(first)
        spare = self._wait_for_spare()
        assert spare is not None, "no spare was minted after a turn"
        assert spare.session.is_alive()
        assert spare.session.session_id, "the spare never completed its warm-up"

    def test_the_next_conversation_takes_it_instead_of_spawning(self):
        first = _agent("conv-1")
        _turn(first)
        spare = self._wait_for_spare()
        assert spare is not None
        warm_pid = spare.session.pid

        second = _agent("conv-2")
        _turn(second)
        assert second._claude_code_session.pid == warm_pid, (
            "the second conversation spawned its own process instead of "
            "taking the one already warm"
        )
        # ...and its transcript is findable again on a later --resume.
        from agent.transports.claude_code_session import load_session_map
        mapped = load_session_map(second._claude_code_session.config_dir)
        assert mapped.get("conv-2") == second._claude_code_session.requested_session_id

    def test_a_caller_it_does_not_fit_neither_takes_nor_evicts_it(self):
        """A caller whose profile has no spare gets ``None`` and leaves every
        other profile's spare exactly where it was: a narrowed tool surface
        (every delegated leaf), a different prompt, a different model — none
        of them may close or displace the process the user's next
        conversation is going to start in."""
        first = _agent("conv-1")
        _turn(first)
        spare = self._wait_for_spare(first)
        assert spare is not None
        warm_pid = spare.session.pid

        narrowed = _agent("child-1")
        narrowed.valid_tool_names = {"todo_list"}      # a leaf's narrowed surface
        assert rt.take_spare(narrowed) is None
        other_prompt = _agent("conv-x", ephemeral="SOMETHING-ELSE")
        assert rt.take_spare(other_prompt) is None
        with rt._SPARE_LOCK:
            assert list(rt._SPARES) == [rt._spare_key(first)]
            assert rt._SPARES[rt._spare_key(first)].session.pid == warm_pid
        assert spare.session.is_alive()

        # ...and the conversation it WAS meant for still gets it.
        taken = rt.take_spare(_agent("conv-2"))
        assert taken is not None and taken.pid == warm_pid

    def test_a_second_profile_gets_its_own_spare_alongside_the_first(self):
        """The bug this replaces: one slot, minted in the image of whichever
        turn ended first. Moe opens two fresh-session profiles per launch
        (the voice brain and a one-word probe with a different prompt); the
        probe's turn ends first, so the slot was the probe's forever and the
        brain took a spare 7 times in 314 sessions. Each profile must hold
        its own."""
        brain = _agent("brain-1", ephemeral="BRAIN-PROMPT")
        probe = _agent("probe-1", ephemeral="Answer with one word.")
        _turn(probe)
        assert self._wait_for_spare(probe) is not None
        _turn(brain)
        assert self._wait_for_spare(brain) is not None, (
            "the brain's profile never got a spare: the pool is still one slot"
        )
        keys = self._pool_keys()
        assert set(keys) == {rt._spare_key(probe), rt._spare_key(brain)}
        assert len(keys) == 2

        # Each conversation takes ITS profile's spare, not the other's.
        probe_pid = self._wait_for_spare(probe).session.pid
        brain_pid = self._wait_for_spare(brain).session.pid
        assert probe_pid != brain_pid
        brain2 = _agent("brain-2", ephemeral="BRAIN-PROMPT")
        _turn(brain2)
        assert brain2._claude_code_session.pid == brain_pid
        probe2 = _agent("probe-2", ephemeral="Answer with one word.")
        _turn(probe2)
        assert probe2._claude_code_session.pid == probe_pid

    def test_a_third_profile_replaces_the_oldest_at_pool_size_two(self):
        """``prewarm_spares`` bounds the pool (each spare is a whole CLI plus
        its MCP servers); over the limit, the spare minted longest ago goes."""
        assert rt._prewarm_spares_from_config() == 2
        one = _agent("conv-1", ephemeral="PROMPT-ONE")
        two = _agent("conv-2", ephemeral="PROMPT-TWO")
        three = _agent("conv-3", ephemeral="PROMPT-THREE")
        _turn(one)
        spare_one = self._wait_for_spare(one)
        assert spare_one is not None
        _turn(two)
        spare_two = self._wait_for_spare(two)
        assert spare_two is not None
        assert self._pool_keys() == [rt._spare_key(one), rt._spare_key(two)]

        _turn(three)
        spare_three = self._wait_for_spare(three)
        assert spare_three is not None
        assert self._wait_until_idle()
        assert self._pool_keys() == [rt._spare_key(two), rt._spare_key(three)], (
            "the oldest profile's spare should have been the one replaced"
        )
        assert not spare_one.session.is_alive(), "the evicted spare was left running"
        assert spare_two.session.is_alive()
        assert spare_three.session.is_alive()

    def test_the_pool_size_is_configurable(self, monkeypatch):
        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "prewarm_spares": 1})
        one = _agent("conv-1", ephemeral="PROMPT-ONE")
        two = _agent("conv-2", ephemeral="PROMPT-TWO")
        _turn(one)
        spare_one = self._wait_for_spare(one)
        assert spare_one is not None
        _turn(two)
        assert self._wait_for_spare(two) is not None
        assert self._wait_until_idle()
        assert self._pool_keys() == [rt._spare_key(two)]
        assert not spare_one.session.is_alive()

    def test_a_spare_is_built_on_a_fresh_cli_session_not_the_builders_transcript(self):
        """The spare is minted by whichever agent's turn just ended, and it
        used to be handed that agent's deterministic CLI session id. With a
        transcript for that id on disk, ``ensure_started`` spawns
        ``claude -p --resume <builder's id>`` — one claimed spare ran inside a
        9.2 MB transcript of another conversation. A spare must start on an
        id nobody has a transcript for."""
        first = _agent("conv-1")
        _turn(first)
        builder = first._claude_code_session
        assert self._wait_for_spare(first) is not None
        rt.drop_spare()
        assert self._wait_until_idle()

        # The builder's transcript exists on disk (the fake CLI writes none).
        transcript_dir = Path(builder.config_dir) / "projects" / "some-cwd-slug"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        (transcript_dir / f"{builder.requested_session_id}.jsonl").write_text("{}\n")
        from agent.transports.claude_code_session import resume_transcript_exists
        assert resume_transcript_exists(builder.config_dir, builder.requested_session_id)

        rt.refill_spare(first)
        spare = self._wait_for_spare(first)
        assert spare is not None
        assert spare.session.requested_session_id != builder.requested_session_id, (
            "the spare was given the builder's CLI session id"
        )
        assert spare.session.resumed is False, (
            "the spare was spawned with --resume on somebody's transcript"
        )

    def test_a_subagent_never_mints_a_spare(self):
        """Every delegation is its own short-lived session with its own
        narrowed prompt; a spare in its image would only displace one a
        conversation is going to want."""
        by_platform = _agent("child-1", ephemeral="LEAF-PROMPT-A")
        by_platform.platform = "subagent"
        _turn(by_platform)
        rt.refill_spare(by_platform)
        by_parent = _agent("child-2", ephemeral="LEAF-PROMPT-B")
        by_parent.platform = "cli"
        by_parent._parent_session_id = "parent-1"
        _turn(by_parent)
        rt.refill_spare(by_parent)
        assert self._wait_until_idle()
        time.sleep(0.3)
        assert self._pool_keys() == [], "a subagent minted a spare"

        # And a subagent never takes one meant for a conversation, either.
        parent = _agent("conv-1")
        _turn(parent)
        spare = self._wait_for_spare(parent)
        assert spare is not None
        child = _agent("child-3")
        child.platform = "subagent"           # same profile as the parent
        assert rt.take_spare(child) is None
        assert self._pool_keys() == [rt._spare_key(parent)]

    def test_each_stale_spare_is_reaped_on_its_own(self):
        one = _agent("conv-1", ephemeral="PROMPT-ONE")
        two = _agent("conv-2", ephemeral="PROMPT-TWO")
        _turn(one)
        spare_one = self._wait_for_spare(one)
        assert spare_one is not None
        _turn(two)
        spare_two = self._wait_for_spare(two)
        assert spare_two is not None
        assert self._wait_until_idle()
        # Age only the first past the timeout.
        with rt._SPARE_LOCK:
            spare_one.minted_at -= 100.0
        rt.sweep_idle_sessions(50.0)
        assert self._pool_keys() == [rt._spare_key(two)]
        assert not spare_one.session.is_alive()
        assert spare_two.session.is_alive()

    def test_a_spare_with_a_different_system_prompt_is_not_used(self):
        """The prompt is baked in at spawn (--append-system-prompt-file), so a
        mismatched spare would respawn on its first use anyway."""
        _turn(_agent("conv-1", ephemeral="PROMPT-ONE"))
        spare = self._wait_for_spare()
        assert spare is not None
        stale_pid = spare.session.pid

        other = _agent("conv-2", ephemeral="PROMPT-TWO")
        _turn(other)
        assert other._claude_code_session.pid != stale_pid

    def test_a_spare_for_a_different_model_is_not_used(self):
        _turn(_agent("conv-1"))
        spare = self._wait_for_spare()
        assert spare is not None
        stale_pid = spare.session.pid

        other = _agent("conv-2")
        other.model = "opus"          # spawn-time setting, cannot be rebound
        _turn(other)
        assert other._claude_code_session.pid != stale_pid

    def test_prewarm_can_be_turned_off(self, monkeypatch):
        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "prewarm": False})
        _turn(_agent("conv-1"))
        time.sleep(0.3)
        with rt._SPARE_LOCK:
            assert not rt._SPARES
        assert rt.take_spare(_agent("conv-2")) is None

    def test_the_spare_is_closed_at_shutdown(self):
        _turn(_agent("conv-1"))
        spare = self._wait_for_spare()
        assert spare is not None
        session = spare.session
        rt._shutdown_registry()
        assert not session.is_alive()
        with rt._SPARE_LOCK:
            assert not rt._SPARES

    def test_a_spare_nobody_came_for_is_reaped(self):
        """It costs as much resident memory as any other warm child; a spare
        unclaimed for hours is a bet that has already lost."""
        _turn(_agent("conv-1"))
        spare = self._wait_for_spare()
        assert spare is not None
        session = spare.session
        rt.sweep_idle_sessions(0.0)
        with rt._SPARE_LOCK:
            assert not rt._SPARES
        assert not session.is_alive()


class TestParagraphBreakAfterTools:
    """The HTTP tool loop arms ``_stream_needs_break`` after every tool
    iteration so ``_fire_stream_delta`` starts the next text on a new
    paragraph. This runtime fires the same callback from the CLI's stream and
    used to leave the flag alone, so every SSE client rendered
    "...take a few minutes.Found the data blob." as one run-on line."""

    def _recording_agent(self, session_id):
        a = _agent(session_id)
        a.seen = []          # (flag as the callback saw it, text)

        def _fire(text):
            a.seen.append((getattr(a, "_stream_needs_break", None), text))
        a._fire_stream_delta = _fire
        return a

    def test_flag_is_armed_by_a_tool_result_and_clear_at_turn_start(self):
        a = self._recording_agent("breaks-1")
        _turn(a, "TOOL")
        assert a.seen, "the fake streams text after its tool result"
        # The first text after the tool result arrives with the break armed.
        assert a.seen[0][0] is True
        assert "".join(t for _, t in a.seen) == "The version is 6.2"
        # A fresh turn starts clean even though the previous one left the
        # flag where the last delta left it.
        a._stream_needs_break = True
        a.seen.clear()
        _turn(a, "hello")
        assert a.seen and a.seen[0][0] is False

    def test_a_plain_turn_never_arms_it(self):
        a = self._recording_agent("breaks-2")
        _turn(a, "hello")
        assert a.seen and all(flag is False for flag, _ in a.seen)
