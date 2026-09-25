"""Human-wait accounting for :mod:`tools.approval` (per session).

Tracks wall-clock time the agent spends verifiably blocked on a HUMAN prompt
(CLI approval prompt, gateway approval round-trip). The concurrent tool batch
deadline in agent/tool_executor.py excludes this time so a slow human answer
never times a batch out — but ONLY this time. Measuring at the source (rather
than residency in the authorization gate, which is arbitrary code) is what keeps
a wedged pre_tool_call plugin or a dead approval client from growing the
exclusion 1:1 with wall clock and defeating the deadline entirely. Keyed by
session so one gateway session's pending approval cannot extend a different
session's batch deadline; state is process-global like the rest of the approval
state, bounded by _HUMAN_WAIT_MAX_SESSIONS.
"""

import contextlib
import contextvars
import json
import os
import threading
import time


# ========================================================================= Human-wait accounting (per
# session) ========================================================================= Tracks the wall-clock
# time the agent spends verifiably blocked on a HUMAN prompt (CLI approval prompt, gateway approval
# round-trip). The concurrent tool batch deadline in agent/tool_executor.py excludes this time so a slow
# human answer never times a batch out — but ONLY this time. Measuring human waits at the source (rather
# than residency in the authorization gate, which is arbitrary code) is what keeps a wedged pre_tool_call
# plugin or a dead approval client from growing the exclusion 1:1 with wall clock and defeating the deadline
# entirely (#79719). Keyed by session so one gateway session's pending approval cannot extend a different
# session's batch deadline. State is process-global like the rest of this module's approval state; entries
# are bounded by _HUMAN_WAIT_MAX_SESSIONS.
class _HumanWaitState:
    __slots__ = ("pending", "window_started", "completed_seconds")

    def __init__(self) -> None:
        self.pending = 0
        self.window_started: float | None = None
        self.completed_seconds = 0.0


_human_wait_lock = threading.Lock()
_human_wait_states: dict[str, _HumanWaitState] = {}
_HUMAN_WAIT_MAX_SESSIONS = 256
# Margin added on top of approvals.timeout when clamping a window's contribution (read-side AND close-side) and when
# bounding the authorization gate's serialization-lock acquire in agent/tool_executor.py. One constant so the clamps
# can't drift apart.
HUMAN_WAIT_MARGIN_S = 60.0


def human_wait_ceiling() -> float:
    """Max seconds a single window may contribute: approvals.timeout + margin.
    Every legitimate human wait self-terminates at ``approvals.timeout`` (the CLI
    prompt join and the gateway poll loop both enforce it), so a window that
    overstays this ceiling is itself wedged and must not keep extending a batch
    deadline. Also the bound on the authorization gate's serialization-lock
    acquire in agent/tool_executor.py, so the two cannot drift. Never call while
    holding ``_human_wait_lock`` — it reads the config cache.
    ``_get_approval_timeout`` caps at ``agent.deadline.MAX_SAFE_TIMEOUT_S`` so the
    value is always safe for ``Lock.acquire(timeout=...)`` / ``Thread.join(timeout=...)``."""
    from tools import approval_context
    return float(approval_context._get_approval_timeout()) + HUMAN_WAIT_MARGIN_S


def _clamped_window_seconds(started: float, now: float, ceiling: float) -> float:
    """Seconds an open window contributes: elapsed, floored at 0, capped. Shared
    by the close-time accrual and the open-window read so the two clamps stay
    identical by construction."""
    return min(max(0.0, now - started), ceiling)


def _human_wait_state(session_key: str) -> _HumanWaitState:
    """Return (creating if needed) the wait state for *session_key*. Caller must
    hold ``_human_wait_lock``. Evicts idle entries (no pending waiter)
    insertion-order-first until the table is under the cap so an army of
    short-lived session keys cannot grow it without bound. Entries with an open
    window are never evicted (that would corrupt live accounting), so the cap is
    best-effort under 256+ concurrently-pending sessions."""
    state = _human_wait_states.get(session_key)
    if state is None:
        for key in list(_human_wait_states):
            if len(_human_wait_states) < _HUMAN_WAIT_MAX_SESSIONS:
                break
            if _human_wait_states[key].pending == 0:
                del _human_wait_states[key]
        state = _human_wait_states[session_key] = _HumanWaitState()
    return state


def _resolve_key(session_key: str | None) -> str:
    if session_key is not None:
        return session_key
    from tools import approval_context
    return approval_context.get_current_session_key()


def activity_heartbeat(label: str):
    """Callable that pings the agent's inactivity tracker (at most every ~10s)
    while a human wait is parked, so the gateway watchdog does not kill the agent
    while the user is still answering. No-op in minimal tool-only environments."""
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - minimal tool-only environments
        return lambda: None
    now = time.monotonic()
    state = {"last_touch": now, "start": now}
    return lambda: touch_activity_if_due(state, label)


@contextlib.contextmanager
def human_wait_window(session_key: str | None = None):
    """Mark the enclosed block as time spent blocked on a human prompt. Wrap ONLY
    code that is genuinely parked waiting for a user's answer (the CLI approval
    prompt, the gateway approval poll loop). The concurrent tool batch deadline
    excludes this time; wrapping anything else re-creates the hang where
    arbitrary wedged code pushes the deadline out forever. Overlapping windows
    for the same session coalesce (pending counter), so two serialized approval
    prompts don't double-count the same wall clock.

    See #79719.
    """
    key = _resolve_key(session_key)
    now = time.monotonic()
    with _human_wait_lock:
        state = _human_wait_state(key)
        if state.pending == 0:
            state.window_started = now
        state.pending += 1
    try:
        yield
    finally:
        now = time.monotonic()
        # Clamp the accrual too: a window that overstayed the ceiling was wedged —
        # record at most the ceiling, not the whole overstay.
        ceiling = human_wait_ceiling()
        with _human_wait_lock:
            state = _human_wait_states.get(key)
            if state is not None:
                state.pending -= 1
                if state.pending == 0:
                    if state.window_started is not None:
                        state.completed_seconds += _clamped_window_seconds(state.window_started, now, ceiling)
                    state.window_started = None


def human_wait_seconds(session_key: str | None = None) -> float:
    """Return total human-wait seconds recorded for the session: completed windows
    plus the currently open one (if any). Monotonically non-decreasing for the
    life of the process — except when an idle session's entry is evicted under
    cap pressure, which can only shrink a consumer's baseline delta to zero (the
    safe direction: the deadline fires sooner). Deadline consumers snapshot a
    baseline at batch start and use the delta. Each window's contribution is
    clamped to :func:`human_wait_ceiling` (belt-and-braces against the
    wedged-window hang).

    Each window's contribution is clamped to :func:`human_wait_ceiling`: every legitimate human wait
    self-terminates at ``approvals.timeout`` (both the CLI prompt join and the gateway poll loop enforce
    it), so a window that overstays that bound is itself wedged and must not keep extending a batch deadline
    (belt-and-braces for #79719).
    """
    key = _resolve_key(session_key)
    now = time.monotonic()
    # Resolve the clamp outside the lock: it reads the config cache, which must never nest under _human_wait_lock.
    ceiling = human_wait_ceiling()
    with _human_wait_lock:
        state = _human_wait_states.get(key)
        if state is None:
            return 0.0
        total = state.completed_seconds
        if state.window_started is not None:
            total += _clamped_window_seconds(state.window_started, now, ceiling)
        return total




#: The CLI's tool_use id for the tool call running in this context, when the MCP server was told it
#: (``_meta["claudecode/toolUseId"]`` on the MCP request). Set by hermes_tools_mcp_server around each
#: dispatch; read by :func:`waiting_on_person` to bind a wait to its own call's turn — explicitly,
#: never by asking which turn is live.
CURRENT_TOOL_USE_ID: "contextvars.ContextVar[str]" = contextvars.ContextVar("hermes_tool_use_id", default="")


@contextlib.contextmanager
def bound_tool_use(tool_use_id: str):
    token = CURRENT_TOOL_USE_ID.set(str(tool_use_id or ""))
    try:
        yield
    finally:
        CURRENT_TOOL_USE_ID.reset(token)


#: How often :func:`waiting_on_person` stamps the activity clock (outside the MCP server).
HUMAN_WAIT_PING_INTERVAL_S = 20.0
#: How often it asks whether its call's turn is still the live one. Short, so a card whose turn
#: ended, was interrupted or was switched away from is withdrawn within seconds.
HUMAN_WAIT_CHECK_INTERVAL_S = 3.0


def _ping_the_turn(label: str, tool_use_id: str):
    """From inside the hermes-tools MCP server: "ok", "ended" or "unknown". No bridge, a bridge
    that does not answer, or anything malformed is "unknown" — never "ended": an unknown is never
    a reason to withdraw a card a person may be reading."""
    try:
        from agent.transports.hermes_tool_bridge import HUMAN_WAIT_PING, bridge_available, call_bridged_tool

        if not bridge_available():
            return "unknown"
        raw = call_bridged_tool(HUMAN_WAIT_PING, {"label": label, "tool_use_id": tool_use_id}, timeout=5)
        state = json.loads(raw).get("state")
    except Exception:
        return "unknown"
    return state if state in ("ok", "ended", "unknown") else "unknown"


@contextlib.contextmanager
def waiting_on_person(label: str, *, on_turn_gone=None,
                      interval: float = HUMAN_WAIT_PING_INTERVAL_S,
                      check_interval: float = HUMAN_WAIT_CHECK_INTERVAL_S,
                      session_key: str | None = None):
    """For code OUTSIDE :mod:`tools.approval` that parks a tool call on a person — a plugin's own
    confirmation dialog (Moe's send card) is the case this was written for.

    While the block runs:

    * a :func:`human_wait_window`, so a concurrent batch deadline does not count the wait;
    * inside the MCP server the claude-code runtime spawns, a ping to the turn's own process every
      ``check_interval`` s naming THIS call by its tool_use id (:data:`CURRENT_TOOL_USE_ID`). The
      agent stamps its activity clock only if that call belongs to the live turn, and answers
      "ended" once that turn has finished — then ``on_turn_gone(reason)`` is called once and the
      caller withdraws its question. With no tool_use id there is no binding and nothing is pinged:
      no heartbeat is better than a heartbeat for somebody else's turn;
    * outside the MCP server, the activity callback bound on the ENTERING thread every ``interval``
      s, and the entering thread's interrupt bit (``tools.interrupt``) as "the turn is gone".

    The first check runs BEFORE the block is entered, so a caller whose turn has already gone hears
    so before it starts anything (``on_turn_gone`` is called synchronously).

    Moe, 2026-09-25: without the heartbeat a cron job waiting on its card read as idle and was
    killed with the card still up; without the watch, a card outlived the turn that asked for it.
    """
    try:
        from tools.environments.base import get_activity_callback
        callback = get_activity_callback()
    except Exception:  # pragma: no cover - minimal tool-only environments
        callback = None
    in_server = bool(os.environ.get("HERMES_MCP_TOOL_PROFILE"))
    tool_use_id = CURRENT_TOOL_USE_ID.get()
    entering = threading.current_thread().ident
    stop = threading.Event()
    told = threading.Event()

    def gone(reason: str) -> None:
        if told.is_set():
            return
        told.set()
        if on_turn_gone is not None:
            try:
                on_turn_gone(reason)
            except Exception:
                pass

    def check() -> bool:
        """One look. True when the turn is gone (and the caller has been told)."""
        if in_server:
            if tool_use_id and _ping_the_turn(label, tool_use_id) == "ended":
                gone("the turn that asked has ended")
                return True
            return False
        try:
            from tools.interrupt import is_thread_interrupted
            if is_thread_interrupted(entering):
                gone("the turn was interrupted")
                return True
        except Exception:
            pass
        return False

    if check():
        # Gone before anything started: nothing to watch.
        yield
        return

    def beat() -> None:
        last_stamp = time.monotonic()
        while not stop.wait(min(interval, check_interval)):
            if check():
                return
            if not in_server and callback is not None and time.monotonic() - last_stamp >= interval:
                last_stamp = time.monotonic()
                try:
                    callback(label)
                except Exception:
                    pass

    thread = None
    if not in_server or tool_use_id:
        thread = threading.Thread(target=beat, name="human-wait-heartbeat", daemon=True)
        thread.start()
    try:
        try:
            window = human_wait_window(session_key)
            window.__enter__()
        except Exception:
            window = None
        try:
            yield
        finally:
            if window is not None:
                with contextlib.suppress(Exception):
                    window.__exit__(None, None, None)
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=2.0)
