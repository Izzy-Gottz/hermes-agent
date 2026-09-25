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


class TestBackgroundLanes:
    """A background agent must never stand between a person and their assistant.

    On 2026-09-15 the owner's Moe answered, its post-turn skill review forked
    under the SAME Hermes session id, and the two messages typed in the next
    74 seconds each came back as "another turn is still running for session
    1e495ecf-d44; try again" — the internal string, in the chat window. These
    tests hold the review's lock the way a running review does and require the
    person to be answered anyway.
    """

    @staticmethod
    def _laned(session_id: str, lane: str = "background_review"):
        agent = _agent(session_id)
        agent._claude_code_lane = lane
        return agent

    def test_a_review_fork_gets_its_own_process(self):
        person = _agent("live")
        _turn(person)
        review = self._laned("live")
        _turn(review)
        assert review._claude_code_session is not person._claude_code_session
        assert review._claude_code_session.pid != person._claude_code_session.pid
        assert rt.registered_session_count() == 2
        with rt._REGISTRY_LOCK:
            assert set(rt._REGISTRY) == {"live", "live#background_review"}

    def test_a_review_fork_writes_its_own_transcript(self):
        """One CLI transcript, two processes, is the same sharing by another
        name — and it is how the review's harness turn used to land in the
        person's conversation."""
        person = _agent("live")
        review = self._laned("live")
        assert rt._claude_session_id_for(review) != rt._claude_session_id_for(person)

    def test_the_person_is_answered_while_a_review_is_running(self):
        person = _agent("live")
        _turn(person)
        review = self._laned("live")
        _turn(review)
        lane_entry = rt._REGISTRY[rt._registry_key(review)]
        assert lane_entry.turn_lock.acquire(timeout=5)  # a review, still going
        try:
            started = time.monotonic()
            answer = _turn(_agent("live"), "you there?")
            assert answer["completed"] is True, answer
            assert time.monotonic() - started < 10  # not the 15 s refusal
        finally:
            lane_entry.turn_lock.release()

    def test_a_person_waits_for_the_turn_ahead_of_them_a_background_caller_does_not(self, monkeypatch):
        assert rt._turn_lock_wait_for(_agent("live")) == rt._LIVE_TURN_LOCK_WAIT_SECONDS
        assert rt._turn_lock_wait_for(self._laned("live")) == rt._TURN_LOCK_WAIT_SECONDS
        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config", lambda: {**cfg, "turn_lock_wait": 900.0})
        assert rt._turn_lock_wait_for(_agent("live")) == 900.0
        assert rt._turn_lock_wait_for(self._laned("live")) == rt._TURN_LOCK_WAIT_SECONDS

    def test_a_contended_turn_is_refused_in_words_not_in_identifiers(self, monkeypatch):
        """Two live turns on one session is still possible (voice and typing at
        once). What the person reads may not be an internal string."""
        monkeypatch.setattr(rt, "_LIVE_TURN_LOCK_WAIT_SECONDS", 0.2)
        person = _agent("busy")
        _turn(person)
        entry = rt._REGISTRY["busy"]
        assert entry.turn_lock.acquire(timeout=5)
        try:
            result = _turn(_agent("busy"), "hello?")
        finally:
            entry.turn_lock.release()
        assert result["completed"] is False
        assert "still finishing" in result["final_response"]
        assert "session" not in result["final_response"]
        assert "busy" not in result["final_response"]
        assert "another turn is still running" in (result["error"] or "")

    def test_closing_a_lane_leaves_the_conversation_alone(self):
        person = _agent("live")
        _turn(person)
        review = self._laned("live")
        _turn(review)
        assert rt.release_lane_session(review) is True
        assert not review._claude_code_session.is_alive()
        assert person._claude_code_session.is_alive()
        with rt._REGISTRY_LOCK:
            assert set(rt._REGISTRY) == {"live"}

    def test_closing_refuses_an_agent_that_is_the_conversation(self):
        """The guard that makes the teardown call safe: no lane, no close."""
        person = _agent("live")
        _turn(person)
        assert rt.release_lane_session(person) is False
        assert person._claude_code_session.is_alive()
        with rt._REGISTRY_LOCK:
            assert "live" in rt._REGISTRY


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

    def test_delegation_from_the_child_is_forced_synchronous_when_nothing_can_deliver_later(self):
        """The caller is holding an open MCP tool call and will only ever see
        this return value; with no declared consumer for a detached
        completion, a background handle would be a receipt for work it is
        never shown."""
        agent = self._recording_agent()
        dispatch = rt.make_tool_bridge_dispatch(agent)
        dispatch("delegate_task", {"goal": "x"})
        name, args, task_id, forced, _call_id = agent._calls[0]
        assert (name, forced) == ("delegate_task", True)
        # ...and the flag does not leak past the call.
        from tools.delegate_tool import synchronous_delegation_forced

        assert not synchronous_delegation_forced()

    def test_a_session_that_can_receive_a_completion_later_gets_a_real_background_handle(self):
        """Moe ticket #4, 2026-09-16: an X reply job asked for "in the
        background" held the conversation for ten minutes under forced sync
        and was then killed by the turn watchdog. A session that declared a
        consumer for detached completions (the client addresses it by id and
        reads what lands on it) is not forced: the child gets its handle.

        The bridge thread has no context of its own, so the declaration — and
        the origin session id the completion routes back on — must travel
        from the request thread's snapshot, or the dispatch would be
        background with nowhere to deliver."""
        import contextvars
        import threading

        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.async_delegation import _current_origin_session_id

        agent = self._recording_agent("moe-sess")
        seen: dict = {}

        def _invoke_tool(name, args, task_id, tool_call_id=None, *rest, **kw):
            from tools.delegate_tool import synchronous_delegation_forced

            seen["forced"] = synchronous_delegation_forced()
            seen["origin"] = _current_origin_session_id()
            return "ran"

        agent._invoke_tool = _invoke_tool
        tokens = set_session_vars(
            platform="api_server", chat_id="moe-sess", session_key="moe-sess", session_id="moe-sess",
            async_delivery=False, session_history_delivery="1",
        )
        try:
            agent._turn_context = contextvars.copy_context()
        finally:
            clear_session_vars(tokens)
        dispatch = rt.make_tool_bridge_dispatch(agent)
        # On a bare thread, as the bridge runs it.
        t = threading.Thread(target=lambda: dispatch("delegate_task", {"goal": "x"}))
        t.start()
        t.join(5)
        assert seen["forced"] is False, "a declared consumer means a handle, not a ten-minute block"
        assert seen["origin"] == "moe-sess", "the completion must route back to the session that asked"

    def test_the_turn_snapshots_its_context_for_the_bridge(self, _env):
        agent = _agent("snap-1")
        _turn(agent)
        import contextvars

        assert isinstance(getattr(agent, "_turn_context", None), contextvars.Context)

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

    def test_a_conversation_that_has_a_transcript_is_never_handed_a_spare(self):
        """A spare carries a CLI session id nobody has a transcript for, and
        ``claim`` writes that id into the session map under the conversation's
        key — where it WINS over the deterministic id on every later start
        (``claude_code_session.py``: ``mapped or session_id or uuid4()``).

        So a conversation that already has history must not be given one. It
        leaves the registry all the time — ``max_sessions`` is 4 and the fifth
        conversation evicts the least recent, and a gateway restart drops all
        of them — and the next turn it takes is a fresh registry entry, which
        is exactly the branch that takes a spare. Claiming one there points
        the conversation at the spare's near-empty transcript and orphans
        everything said before: the resume finds the wrong file, permanently,
        because the map entry persists.

        The spare is not wasted — it stays warm for whoever has no history.
        """
        first = _agent("conv-1")
        _turn(first)
        own = first._claude_code_session
        own_id = own.requested_session_id
        config_dir = own.config_dir
        assert self._wait_for_spare(first) is not None

        # The conversation has history on disk (the fake CLI writes none).
        transcript_dir = Path(config_dir) / "projects" / "some-cwd-slug"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        (transcript_dir / f"{own_id}.jsonl").write_text("{}\n")

        # ...and it falls out of the registry: an LRU eviction, or a restart.
        for key in list(rt._REGISTRY):
            rt.evict_session(key)

        again = _agent("conv-1")          # the same conversation comes back
        _turn(again)
        assert again._claude_code_session.requested_session_id == own_id, (
            "the returning conversation was handed a spare; its CLI session "
            "id is now the spare's and its transcript is orphaned"
        )
        from agent.transports.claude_code_session import load_session_map
        assert load_session_map(config_dir).get("conv-1") in (None, own_id), (
            "the session map was repointed at the spare's transcript"
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

    def test_only_the_named_session_keys_mint_a_spare(self, monkeypatch):
        """``prewarm_session_keys`` names the conversations worth a warm
        process. Every other session key — a meeting's scan, a probe — mints
        nothing, because each spare's warm-up is a real model turn nobody
        may come for (72 minted, 5 taken, measured on one Mac)."""
        cfg = rt._claude_code_config()
        monkeypatch.setattr(rt, "_claude_code_config",
                            lambda: {**cfg, "prewarm_session_keys": ["moe"]})
        scan = _agent("meeting-scan", ephemeral="SCAN-PROMPT")
        scan._gateway_session_key = "moe-meeting-scan"
        _turn(scan)
        rt.refill_spare(scan)
        unkeyed = _agent("no-key", ephemeral="OTHER-PROMPT")
        _turn(unkeyed)
        rt.refill_spare(unkeyed)
        assert self._wait_until_idle()
        time.sleep(0.3)
        assert self._pool_keys() == [], "a session key nobody named minted a spare"

        chat = _agent("conv-1")
        chat._gateway_session_key = "moe"
        _turn(chat)
        assert self._wait_for_spare(chat) is not None
        assert self._pool_keys() == [rt._spare_key(chat)]

    def test_no_session_key_list_means_every_conversation_mints(self, monkeypatch):
        cfg = rt._claude_code_config()
        for unset in (None, [], ""):
            monkeypatch.setattr(rt, "_claude_code_config",
                                lambda unset=unset: {**cfg, "prewarm_session_keys": unset})
            assert rt._may_mint_for(_agent("any"))

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


# ── The child's activity clock ──────────────────────────────────────────────
#
# Moe ticket #4, 2026-09-16. A delegated X search-and-reply thinking with
# Claude Code ran 63 tool calls in 18 minutes while its parent's heartbeat
# read "initializing (iteration 0/250)" the whole time: nothing under this
# runtime stamped the clocks the heartbeat and the turn watchdog read. The
# heartbeat gave up after 15 cycles, the 600 s watchdog aborted the parent
# turn and killed the working child, and the model was told the user had
# rejected the tool. These read the clock through the REAL readers — the
# agent's own summary and the delegate heartbeat — not through the fields.

def _child():
    """The attributes the liveness readers touch on a real AIAgent, with the
    real ``_touch_activity`` and the real ``get_activity_summary``."""
    import types
    from agent.activity_tracking import ActivityTrackingMixin
    from run_agent import AIAgent

    class Child(ActivityTrackingMixin):
        pass

    c = Child()
    c.session_id = "child"
    c._session_db = None
    c._current_tool = None
    c._api_call_count = 0
    c.max_iterations = 250
    c.iteration_budget = SimpleNamespace(used=0, max_total=250)
    c.get_activity_summary = types.MethodType(AIAgent.get_activity_summary, c)
    return c


def test_cli_tool_events_advance_the_childs_activity_clock():
    child = _child()
    before = child.get_activity_summary()
    assert before["last_activity_ts"] is None and before["current_tool"] is None
    on_event = rt.make_claude_code_event_bridge(child)

    on_event({"kind": "tool_started", "call_id": "c1", "name": "browser_exec", "args": {"code": "x"}})
    mid = child.get_activity_summary()
    assert mid["last_activity_ts"] is not None
    assert mid["current_tool"] == "browser_exec"
    assert "browser_exec" in mid["last_activity_desc"]

    time.sleep(0.01)
    on_event({"kind": "tool_completed", "call_id": "c1", "name": "browser_exec", "args": {}, "result": "ok"})
    after = child.get_activity_summary()
    assert after["last_activity_ts"] > mid["last_activity_ts"]
    assert after["current_tool"] is None, "between tools the child is idle, not stuck in one"

    on_event({"kind": "assistant_message", "text": "done"})
    assert child.get_activity_summary()["api_call_count"] == 1, "one model response is one iteration"


def test_streaming_stamps_the_clock_but_not_per_token():
    child = _child()
    on_event = rt.make_claude_code_event_bridge(child)
    on_event({"kind": "text_delta", "text": "a"})
    first = child.get_activity_summary()["last_activity_ts"]
    assert first is not None
    for _ in range(50):
        on_event({"kind": "text_delta", "text": "b"})
    assert child.get_activity_summary()["last_activity_ts"] == first, "a stamp per token would take the lock 50 times"


def test_the_delegate_heartbeat_sees_a_working_cli_child():
    """The real reader. A child that emits CLI events between ticks is never
    stale; the same child with no events IS — which is what proved this test
    can fail, and what happened on 2026-09-16."""
    from tools.delegate_tool import _HEARTBEAT_STALE_CYCLES_IDLE
    from tools.delegate_tool_child_run import _Heartbeat

    touched = []
    parent = SimpleNamespace(_touch_activity=lambda desc, **kw: touched.append(desc))

    silent = _child()
    hb = _Heartbeat(silent, parent, 0)
    verdicts = [hb.tick() for _ in range(_HEARTBEAT_STALE_CYCLES_IDLE + 1)]
    assert False in verdicts, "a child nothing stamps goes stale (the old behaviour, and still right for a dead one)"

    touched.clear()
    working = _child()
    on_event = rt.make_claude_code_event_bridge(working)
    hb = _Heartbeat(working, parent, 0)
    for i in range(_HEARTBEAT_STALE_CYCLES_IDLE + 5):
        on_event({"kind": "tool_started", "call_id": f"c{i}", "name": "browser_exec", "args": {}})
        time.sleep(0.002)
        assert hb.tick() is not False, f"a child running tools was called stale on cycle {i}"
        on_event({"kind": "tool_completed", "call_id": f"c{i}", "name": "browser_exec", "args": {}, "result": ""})
    assert any("running browser_exec" in d for d in touched), touched[-3:]
    assert not any("initializing" in d for d in touched)


class TestTurnLedgerSurvivesARebuild:
    """A respawn that fails rebuilds the session mid-turn. The turn is still the same turn: its
    ledger (token, live flag, tool_use ids) must be the rebuilt session's, or a card this turn put up
    pings an empty ledger and is answered "unknown" — never stamped, never withdrawn with the turn."""

    def test_a_failed_respawn_moves_the_live_turn_onto_the_new_session(self, monkeypatch):
        first = _agent("s-ledger", ephemeral="ONE")
        _turn(first)
        old = first._claude_code_session
        monkeypatch.setattr(type(old), "restart", lambda self, **kw: (_ for _ in ()).throw(RuntimeError("respawn failed")))
        seen = {}
        turn_agent = [None]
        real_build = rt._build_session

        def build(agent, **kw):
            session = real_build(agent, **kw)
            real_started = session.ensure_started

            def ensure_started():
                if "live" in seen or agent is not turn_agent[0]:   # a spare, not the turn's rebuild
                    return real_started()
                ledger = getattr(session, "_hermes_turn_ledger", None)
                seen["live"] = bool(ledger and ledger.live)
                seen["token"] = ledger.token if ledger else None
                seen["same"] = ledger is getattr(agent, "_hermes_turn_ledger_in_use", None)
                rt._note_turn_tool_call(agent, "toolu_DURING")
                seen["ping"] = rt._human_wait_state(agent, "toolu_DURING", "on the card")
                return real_started()

            session.ensure_started = ensure_started
            return session

        monkeypatch.setattr(rt, "_build_session", build)
        second = _agent("s-ledger", ephemeral="TWO")
        second._touch_activity = lambda label: None
        turn_agent[0] = second
        result = _turn(second)
        assert result["completed"] is True
        assert second._claude_code_session is not old            # it really was rebuilt
        assert seen == {"live": True, "token": second._turn_id, "same": True, "ping": "ok"}
        # And after the turn, the rebuilt session's ledger says so.
        assert rt._human_wait_state(second, "toolu_DURING", "x") == "ended"
