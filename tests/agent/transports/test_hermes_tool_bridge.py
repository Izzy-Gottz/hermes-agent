"""Tests for the agent-loop tool bridge.

Real sockets, real threads, both ends of the wire — the client half runs in
the MCP server the CLI spawned and the server half in the Hermes process, and
the failure this exists to prevent (a subagent built in the wrong process,
with no credential) is invisible to a mock.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time

import pytest

from agent.transports.hermes_tool_bridge import (
    BRIDGE_SOCKET_ENV,
    BRIDGE_TIMEOUT_ENV,
    BRIDGE_TOKEN_ENV,
    BRIDGE_TOOLS_ENV,
    BRIDGED_TOOLS,
    BridgeError,
    ToolBridge,
    bridge_address,
    bridge_available,
    bridge_timeout,
    bridged_tool_names,
    call_bridged_tool,
    socket_is_live,
)


@pytest.fixture
def bridge(tmp_path):
    """A started bridge whose dispatcher records what it was asked to run."""
    calls: list[tuple[str, dict]] = []

    def dispatch(tool: str, args: dict):
        calls.append((tool, dict(args)))
        if tool == "delegate_task":
            return json.dumps({"results": [{"response": "PONG"}]})
        if tool == "session_search":
            raise RuntimeError("dispatcher exploded")
        return f"{tool}-ok"

    b = ToolBridge(dispatch, directory=str(tmp_path))
    b.start()
    b.calls = calls  # type: ignore[attr-defined]
    try:
        yield b
    finally:
        b.close()


def _env(b: ToolBridge) -> dict:
    return {BRIDGE_SOCKET_ENV: b.socket_path, BRIDGE_TOKEN_ENV: b.token}


class TestNames:
    def test_bridged_tools_match_the_agent_loop_set(self):
        """If Hermes adds an agent-loop tool, the bridge must learn about it —
        otherwise it silently goes missing under the CLI runtimes again."""
        from model_tools import _AGENT_LOOP_TOOLS

        assert set(BRIDGED_TOOLS) == set(_AGENT_LOOP_TOOLS)

    def test_mcp_server_agrees(self):
        from agent.transports.hermes_tools_mcp_server import AGENT_LOOP_TOOLS

        assert set(AGENT_LOOP_TOOLS) == set(BRIDGED_TOOLS)


class TestRoundTrip:
    def test_a_call_reaches_the_dispatcher_and_the_result_comes_back(self, bridge):
        out = call_bridged_tool("delegate_task", {"goal": "say pong"}, env=_env(bridge))
        assert "PONG" in out
        assert bridge.calls == [("delegate_task", {"goal": "say pong"})]

    def test_non_string_results_are_json_encoded(self, tmp_path):
        b = ToolBridge(lambda tool, args: {"ok": 1}, directory=str(tmp_path))
        b.start()
        try:
            assert json.loads(call_bridged_tool("todo", env=_env(b))) == {"ok": 1}
        finally:
            b.close()

    def test_concurrent_calls_do_not_serialise_into_each_other(self, tmp_path):
        started = threading.Barrier(3, timeout=10)

        def dispatch(tool, args):
            started.wait()  # deadlocks unless all three are in flight at once
            return tool

        b = ToolBridge(dispatch, directory=str(tmp_path))
        b.start()
        results: dict[int, str] = {}
        try:
            threads = [
                threading.Thread(
                    target=lambda i=i: results.__setitem__(
                        i, call_bridged_tool("todo", {"i": i}, env=_env(b))
                    )
                )
                for i in range(3)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
            assert results == {0: "todo", 1: "todo", 2: "todo"}
        finally:
            b.close()

    def test_the_dispatcher_can_be_repointed_at_the_current_turns_agent(self, bridge):
        """api_server builds an AIAgent per request and the warm session is
        shared; a call must land on the instance driving the turn."""
        bridge.set_dispatch(lambda tool, args: "second-agent")
        assert call_bridged_tool("memory", env=_env(bridge)) == "second-agent"


class TestRefusals:
    def test_a_wrong_token_is_refused(self, bridge):
        env = _env(bridge) | {BRIDGE_TOKEN_ENV: "not-the-token"}
        with pytest.raises(BridgeError, match="authentication failed"):
            call_bridged_tool("todo", env=env)
        assert bridge.calls == []

    def test_only_the_four_agent_loop_tools_are_dispatchable(self, bridge):
        """The bridge is not a general back door into the parent process: a
        leaked token buys the four tools the model already has."""
        with pytest.raises(BridgeError, match="not bridged"):
            call_bridged_tool("terminal", {"command": "id"}, env=_env(bridge))
        assert bridge.calls == []

    def test_no_bridge_configured_says_so_instead_of_hanging(self):
        with pytest.raises(BridgeError, match="no tool bridge is configured"):
            call_bridged_tool("delegate_task", {"goal": "x"}, env={})

    def test_a_dead_socket_is_an_error_not_a_hang(self, bridge):
        env = _env(bridge)
        bridge.close()
        with pytest.raises(BridgeError, match="not answering"):
            call_bridged_tool("todo", env=env)

    def test_a_dispatcher_exception_comes_back_as_an_error(self, bridge):
        with pytest.raises(BridgeError, match="dispatcher exploded"):
            call_bridged_tool("session_search", env=_env(bridge))

    def test_a_client_timeout_is_reported_against_the_tool(self, tmp_path):
        b = ToolBridge(lambda tool, args: time.sleep(5) or "late", directory=str(tmp_path))
        b.start()
        try:
            with pytest.raises(BridgeError, match="did not finish within"):
                call_bridged_tool("todo", env=_env(b), timeout=0.4)
        finally:
            b.close()


class TestLifecycle:
    def test_the_socket_is_private_to_this_user(self, bridge, tmp_path):
        """The socket's own 0600 is not the whole story: os.makedirs(mode=…,
        exist_ok=True) is a NO-OP on a directory that already exists, so
        trusting the caller's directory would leave the socket in whatever
        mode $HERMES_HOME/claude-code happens to be (0755 on the machine this
        was written on). Each bridge makes its own."""
        assert stat.S_IMODE(os.stat(bridge.socket_path).st_mode) == 0o600
        parent = os.path.dirname(bridge.socket_path)
        assert stat.S_IMODE(os.stat(parent).st_mode) == 0o700
        assert len(bridge.token) >= 32

    def test_the_socket_lands_in_the_directory_it_was_given(self):
        """pytest's tmp_path is itself too long for sun_path, so this one
        uses a short directory to prove the preferred branch is taken — the
        config dir, where the sweeper looks for husks."""
        import tempfile

        with tempfile.TemporaryDirectory(dir="/tmp") as short:
            b = ToolBridge(lambda *_: "ok", directory=short)
            b.start()
            try:
                assert os.path.dirname(os.path.dirname(b.socket_path)) == short
                assert call_bridged_tool("todo", env=_env(b)) == "ok"
            finally:
                b.close()
            assert os.listdir(short) == []

    def test_each_bridge_mints_its_own_token(self, tmp_path):
        a = ToolBridge(lambda *_: "", directory=str(tmp_path))
        b = ToolBridge(lambda *_: "", directory=str(tmp_path))
        a.start()
        b.start()
        try:
            assert a.token != b.token
            assert a.socket_path != b.socket_path
        finally:
            a.close()
            b.close()

    def test_close_removes_the_socket_and_is_idempotent(self, tmp_path):
        b = ToolBridge(lambda *_: "", directory=str(tmp_path))
        b.start()
        path = b.socket_path
        b.close()
        b.close()
        assert not os.path.exists(path)
        assert not b.running

    def test_start_is_idempotent(self, bridge):
        path, token = bridge.socket_path, bridge.token
        bridge.start()
        assert (bridge.socket_path, bridge.token) == (path, token)

    def test_a_directory_too_deep_for_sun_path_falls_back_to_a_private_temp_dir(self, tmp_path):
        deep = tmp_path
        for _ in range(12):
            deep = deep / "a-directory-with-a-long-name"
        b = ToolBridge(lambda *_: "ok", directory=str(deep))
        b.start()
        path, owned = b.socket_path, os.path.dirname(b.socket_path)
        try:
            assert path and not path.startswith(str(deep))
            assert call_bridged_tool("todo", env=_env(b)) == "ok"
            assert stat.S_IMODE(os.stat(owned).st_mode) == 0o700
        finally:
            b.close()
        assert not os.path.exists(path)
        assert not os.path.exists(owned)


class TestEnvHelpers:
    def test_address_needs_both_halves(self, bridge):
        assert bridge_address(_env(bridge)) == (bridge.socket_path, bridge.token)
        assert bridge_address({BRIDGE_SOCKET_ENV: "/x"}) == ("", "")
        assert bridge_address({BRIDGE_TOKEN_ENV: "t"}) == ("", "")
        assert bridge_available(_env(bridge)) and not bridge_available({})

    def test_timeout_is_configurable_and_never_zero(self):
        from agent.transports.hermes_tool_bridge import DEFAULT_TIMEOUT_SECONDS

        assert bridge_timeout({}) == DEFAULT_TIMEOUT_SECONDS
        assert bridge_timeout({BRIDGE_TIMEOUT_ENV: "12"}) == 12.0
        assert bridge_timeout({BRIDGE_TIMEOUT_ENV: "nonsense"}) == DEFAULT_TIMEOUT_SECONDS
        assert bridge_timeout({BRIDGE_TIMEOUT_ENV: "0"}) == DEFAULT_TIMEOUT_SECONDS


class TestHardening:
    """The bridge is reachable by anything running as this user — Hermes'
    own `terminal` runs in the MCP server process, which holds the token in
    its environment. So "nobody would do that" is not an argument for any of
    these."""

    def test_a_connection_that_never_speaks_does_not_park_a_thread_forever(
        self, tmp_path, monkeypatch
    ):
        import agent.transports.hermes_tool_bridge as htb

        monkeypatch.setattr(htb, "_REQUEST_READ_TIMEOUT", 0.3)
        b = ToolBridge(lambda *_: "ok", directory=str(tmp_path))
        b.start()
        try:
            import socket as _socket

            silent = [_socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) for _ in range(4)]
            for s_ in silent:
                s_.connect(b.socket_path)
            # The bridge's own bookkeeping, not threading.active_count():
            # a process-global count is order-dependent and any unrelated
            # thread started during the poll flips it.
            deadline = time.time() + 5
            while b._workers and time.time() < deadline:
                time.sleep(0.05)
            assert not b._workers
            for s_ in silent:
                s_.close()
            # ...and the bridge still works afterwards.
            assert call_bridged_tool("todo", env=_env(b)) == "ok"
        finally:
            b.close()

    def test_expensive_calls_are_capped_and_the_refusal_is_readable(self, tmp_path):
        """Only delegate_task is metered — it is a whole subagent fan-out.
        Metering the cheap three measured 112 refusals out of 120 concurrent
        `todo` calls, which is 112 hard errors for the model to reason about."""
        release = threading.Event()
        b = ToolBridge(
            lambda tool, args: (release.wait(10), "slow")[1],
            directory=str(tmp_path),
            max_concurrent_calls=2,
        )
        b.start()
        holders = []
        try:
            holders = [
                threading.Thread(
                    target=lambda: call_bridged_tool("delegate_task", env=_env(b))
                )
                for _ in range(2)
            ]
            for t in holders:
                t.start()
            deadline = time.time() + 5
            while b.in_flight < 2 and time.time() < deadline:
                time.sleep(0.02)
            assert b.in_flight == 2
            with pytest.raises(BridgeError, match="already running 2 delegate_task"):
                call_bridged_tool("delegate_task", env=_env(b))
            # ...while the cheap three are not metered and still answer.
            cheap = threading.Thread(
                target=lambda: call_bridged_tool("todo", env=_env(b))
            )
            cheap.start()
            deadline = time.time() + 5
            while b.in_flight < 3 and time.time() < deadline:
                time.sleep(0.02)
            assert b.in_flight == 3
            holders.append(cheap)
        finally:
            release.set()
            for t in holders:
                t.join(timeout=10)
            b.close()

    def test_in_flight_is_what_tells_a_waiting_turn_from_a_dead_one(self, tmp_path):
        release = threading.Event()
        b = ToolBridge(lambda *_: (release.wait(10), "ok")[1], directory=str(tmp_path))
        b.start()
        t = None
        try:
            assert b.in_flight == 0
            t = threading.Thread(target=lambda: call_bridged_tool("todo", env=_env(b)))
            t.start()
            deadline = time.time() + 5
            while b.in_flight == 0 and time.time() < deadline:
                time.sleep(0.02)
            assert b.in_flight == 1
        finally:
            release.set()
            if t is not None:
                t.join(timeout=10)
            b.close()
        assert b.in_flight == 0

    def test_a_result_carrying_a_lone_surrogate_still_answers(self, tmp_path):
        """Tool output decoded with surrogateescape (subprocess bytes, odd
        filenames) raises UnicodeEncodeError on a utf-8 encode; the caller
        would see 'connection closed with no message' and chase the wrong
        thing."""
        b = ToolBridge(lambda *_: "before\udcff after", directory=str(tmp_path))
        b.start()
        try:
            assert call_bridged_tool("todo", env=_env(b)) == "before\udcff after"
        finally:
            b.close()

    def test_a_base_exception_from_the_dispatcher_fails_fast(self, tmp_path):
        """Not `except Exception`: a SystemExit would leave the client waiting
        its whole timeout on a socket a dead thread's traceback still holds."""
        class _Cancelled(BaseException):
            pass

        def boom(tool, args):
            raise _Cancelled("cancelled")

        b = ToolBridge(boom, directory=str(tmp_path))
        b.start()
        try:
            started = time.time()
            with pytest.raises(BridgeError, match="_Cancelled"):
                call_bridged_tool("todo", env=_env(b), timeout=5)
            assert time.time() - started < 3
        finally:
            b.close()


class TestNarrowedSurface:
    def test_a_bridge_can_be_given_fewer_tools_than_the_four(self, tmp_path):
        b = ToolBridge(lambda tool, args: tool, directory=str(tmp_path),
                       allowed_tools=("todo",))
        b.start()
        try:
            assert b.allowed_tools == ("todo",)
            assert call_bridged_tool("todo", env=_env(b)) == "todo"
            with pytest.raises(BridgeError, match="not bridged"):
                call_bridged_tool("delegate_task", {"goal": "x"}, env=_env(b))
        finally:
            b.close()

    def test_the_child_is_told_which_names_to_offer(self, bridge):
        env = _env(bridge)
        assert bridged_tool_names({}) == ()
        assert bridged_tool_names(env) == BRIDGED_TOOLS  # unset = all
        assert bridged_tool_names({**env, BRIDGE_TOOLS_ENV: "todo, memory"}) == (
            "todo", "memory",
        )
        assert bridged_tool_names({**env, BRIDGE_TOOLS_ENV: "nonsense"}) == ()


class TestLiveness:
    def test_socket_is_live_distinguishes_a_running_bridge_from_a_husk(self, bridge, tmp_path):
        assert socket_is_live(bridge.socket_path)
        assert not socket_is_live(str(tmp_path / "nothing.sock"))
        path = bridge.socket_path
        bridge.close()
        assert not socket_is_live(path)


class TestHolds:
    def test_a_hold_keeps_the_bridge_active_until_it_is_released(self, tmp_path):
        from agent.transports.hermes_tool_bridge import bridge_hold

        b = ToolBridge(lambda tool, args: tool, directory=str(tmp_path))
        b.start()
        try:
            env = _env(b)
            assert not b.active
            with bridge_hold(env):
                assert b.active
                # A hold takes no dispatch slot: real work still runs.
                assert call_bridged_tool("todo", env=env) == "todo"
            deadline = time.time() + 5
            while b.active and time.time() < deadline:
                time.sleep(0.02)
            assert not b.active
        finally:
            b.close()

    def test_a_hold_without_a_bridge_is_a_silent_no_op(self):
        from agent.transports.hermes_tool_bridge import bridge_hold

        with bridge_hold({}):
            pass  # must not raise, must not block

    def test_a_bad_token_cannot_hold_the_line(self, bridge):
        from agent.transports.hermes_tool_bridge import bridge_hold

        env = _env(bridge) | {BRIDGE_TOKEN_ENV: "wrong"}
        with bridge_hold(env):
            assert not bridge.active

    def test_close_releases_held_connections(self, tmp_path):
        """A hold is a thread parked on recv until its peer hangs up, and
        nobody hangs up on a session that has just closed."""
        from agent.transports.hermes_tool_bridge import bridge_hold

        b = ToolBridge(lambda tool, args: tool, directory=str(tmp_path))
        b.start()
        hold = bridge_hold(_env(b))
        hold.__enter__()
        try:
            deadline = time.time() + 5
            while not b.active and time.time() < deadline:
                time.sleep(0.02)
            assert b.active
            b.close()
            deadline = time.time() + 5
            while b._workers and time.time() < deadline:
                time.sleep(0.05)
            assert not b._workers and not b._held
            assert b.in_flight == 0
        finally:
            hold.__exit__(None, None, None)


class TestAfterClose:
    """close() stops the accept loop; it does not stop a connection already
    accepted, and unlinking the socket path does nothing to one either."""

    def test_a_call_that_arrives_after_close_is_refused_not_run(self, tmp_path):
        import socket as _socket

        from agent.transports.hermes_tool_bridge import _read_message, _write_message

        ran = []
        b = ToolBridge(lambda tool, args: ran.append(tool) or "ran", directory=str(tmp_path))
        b.start()
        conn = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        conn.connect(b.socket_path)          # accepted, nothing sent yet
        time.sleep(0.1)
        b.close()                            # session torn down
        try:
            _write_message(conn, {"token": b.token, "tool": "delegate_task", "args": {}})
            reply = _read_message(conn)
        finally:
            conn.close()
        assert reply["ok"] is False and "closed" in reply["error"]
        assert ran == [], "a delegate_task ran on a torn-down session's agent"

    def test_a_hold_that_arrives_after_close_does_not_leak(self, tmp_path):
        import socket as _socket

        from agent.transports.hermes_tool_bridge import _read_message, _write_message

        b = ToolBridge(lambda *_: "", directory=str(tmp_path))
        b.start()
        conn = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        conn.connect(b.socket_path)
        time.sleep(0.1)
        b.close()
        try:
            _write_message(conn, {"token": b.token, "op": "hold"})
            reply = _read_message(conn)
            assert reply["ok"] is False and "closed" in reply["error"]
            # close() snapshots _held and replaces the set, so a hold
            # registered afterwards would never be shut down.
            deadline = time.time() + 5
            while b._workers and time.time() < deadline:
                time.sleep(0.05)
            assert b.in_flight == 0 and not b.active
            assert not b._held and not b._workers
        finally:
            conn.close()


class TestBindFallback:
    def test_a_directory_that_cannot_host_a_socket_falls_through(self, tmp_path, monkeypatch):
        """A bind can fail where the mkdir succeeded — AF_UNIX is not
        supported on every filesystem, and a $HERMES_HOME on NFS returns
        EOPNOTSUPP. Losing the four tools on such a machine would be exactly
        the failure this module exists to end."""
        import socket as _socket

        real_bind = _socket.socket.bind
        first = str(tmp_path)

        def picky_bind(self, address):
            if isinstance(address, str) and address.startswith(first):
                raise OSError(45, "Operation not supported")
            return real_bind(self, address)

        monkeypatch.setattr(_socket.socket, "bind", picky_bind)
        b = ToolBridge(lambda *_: "ok", directory=first)
        b.start()
        try:
            assert not b.socket_path.startswith(first)
            assert call_bridged_tool("todo", env=_env(b)) == "ok"
        finally:
            b.close()


class TestReNarrowing:
    def test_the_allowed_set_follows_the_agent_driving_the_turn(self, bridge):
        env = _env(bridge)
        assert call_bridged_tool("delegate_task", {"goal": "x"}, env=env)
        bridge.set_allowed_tools(("todo",))
        with pytest.raises(BridgeError, match="not bridged"):
            call_bridged_tool("delegate_task", {"goal": "x"}, env=env)
        assert call_bridged_tool("todo", env=env) == "todo-ok"


class TestActivityStamp:
    def test_last_active_is_stamped_when_a_call_ENDS_not_only_when_it_starts(
        self, tmp_path
    ):
        """The end is the half that matters. Stamping only the start meant a
        window that opened before the call and expired just after it looked
        like two minutes of silence, and retired a session six seconds after
        its delegation had succeeded."""
        b = ToolBridge(lambda tool, args: (time.sleep(0.3), "done")[1],
                       directory=str(tmp_path))
        b.start()
        try:
            before = b.last_active
            call_bridged_tool("todo", env=_env(b))
            after = b.last_active
            assert after > before
            # ...and it is the END that was stamped, not the start.
            assert after >= before + 0.25, (after - before)
            assert b.in_flight == 0 and not b.active
        finally:
            b.close()


class TestSlotAccounting:
    def test_a_metered_slot_is_returned_after_every_call(self, tmp_path):
        """Leak one and the bridge goes permanently deaf after N calls — with
        no error anywhere, just delegate_task refusing forever."""
        def dispatch(tool, args):
            if args.get("boom"):
                raise RuntimeError("nope")
            return tool

        b = ToolBridge(dispatch, directory=str(tmp_path), max_concurrent_calls=2)
        b.start()
        try:
            for i in range(6):
                if i % 2:
                    with pytest.raises(BridgeError, match="nope"):
                        call_bridged_tool("delegate_task", {"boom": 1}, env=_env(b))
                else:
                    assert call_bridged_tool("delegate_task", env=_env(b)) == "delegate_task"
            # Both slots still free: two concurrent calls must still start.
            release = threading.Event()
            b.set_dispatch(lambda t, a: (release.wait(10), t)[1])
            ts = [
                threading.Thread(
                    target=lambda: call_bridged_tool("delegate_task", env=_env(b))
                )
                for _ in range(2)
            ]
            for t in ts:
                t.start()
            deadline = time.time() + 5
            while b.in_flight < 2 and time.time() < deadline:
                time.sleep(0.02)
            assert b.in_flight == 2, "a dispatch slot leaked"
            release.set()
            for t in ts:
                t.join(timeout=10)
        finally:
            b.close()

    def test_too_many_open_connections_are_refused_not_queued_forever(self, tmp_path):
        import socket as _socket

        import agent.transports.hermes_tool_bridge as htb

        b = ToolBridge(lambda t, a: t, directory=str(tmp_path))
        b.start()
        # Fill the connection ceiling with silent peers.
        b._conn_slots = threading.BoundedSemaphore(1)
        parked = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        parked.connect(b.socket_path)
        try:
            deadline = time.time() + 5
            while b._conn_slots.acquire(blocking=False):
                b._conn_slots.release()
                if time.time() > deadline:
                    break
                time.sleep(0.02)
            with pytest.raises(BridgeError, match="too many open connections|refused"):
                call_bridged_tool("todo", env=_env(b))
        finally:
            parked.close()
            b.close()


class TestWireGuards:
    def test_a_message_larger_than_the_cap_is_refused(self, bridge, monkeypatch):
        import agent.transports.hermes_tool_bridge as htb

        monkeypatch.setattr(htb, "_MAX_MESSAGE_BYTES", 64)
        with pytest.raises(BridgeError):
            call_bridged_tool("todo", {"pad": "x" * 4096}, env=_env(bridge))

    def test_a_message_that_is_not_an_object_is_refused(self, bridge):
        import socket as _socket

        from agent.transports.hermes_tool_bridge import _read_message

        conn = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        conn.connect(bridge.socket_path)
        try:
            conn.sendall(b'["not", "an", "object"]\n')
            reply = _read_message(conn)
        finally:
            conn.close()
        assert reply["ok"] is False and "not an object" in reply["error"]
        assert bridge.calls == []


class TestCloseUnderPressure:
    def test_close_survives_a_worker_that_has_not_started_yet(self, tmp_path):
        """The registered-but-not-started window, made deterministic.

        close() snapshots `_workers` under the lock; a worker added there but
        started outside it could be joined before it began —
        `RuntimeError: cannot join thread before it is started` — which
        escaped BEFORE _cleanup_path() and left the socket and its 0700
        directory behind. The caller swallows the exception, so it was
        silent: 3 leaks in 300 trials under connection pressure. Rather than
        bet on hitting a ~1% race, put an unstarted thread in the set on
        purpose."""
        b = ToolBridge(lambda t, a: t, directory=str(tmp_path))
        b.start()
        path, owned = b.socket_path, os.path.dirname(b.socket_path)
        never_started = threading.Thread(target=lambda: None)
        with b._lock:
            b._workers.add(never_started)
        b.close()  # must not raise
        assert not os.path.exists(path)
        assert not os.path.exists(owned)

    def test_closing_while_connections_are_arriving_never_leaks(self, monkeypatch):
        """The same property from the outside, at the scale a test can afford."""
        import socket as _socket
        import tempfile

        import agent.transports.hermes_tool_bridge as htb

        monkeypatch.setattr(htb, "_CLOSE_JOIN_SECONDS", 0.05)
        monkeypatch.setattr(htb, "_REQUEST_READ_TIMEOUT", 0.05)
        leaked, errors = [], []
        for _ in range(25):
            with tempfile.TemporaryDirectory(dir="/tmp") as d:
                b = ToolBridge(lambda t, a: t, directory=d)
                b.start()
                path, owned = b.socket_path, os.path.dirname(b.socket_path)
                conns = []
                for _ in range(3):
                    c = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                    try:
                        c.connect(path)
                        conns.append(c)
                    except OSError:
                        pass
                try:
                    b.close()
                except Exception as exc:
                    errors.append(repr(exc))
                for c in conns:
                    c.close()
                if os.path.exists(path) or os.path.exists(owned):
                    leaked.append(owned)
        assert errors == [], errors
        assert leaked == [], leaked
