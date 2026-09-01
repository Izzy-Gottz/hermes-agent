"""A call back into the live agent, for the tools that need one.

Four Hermes tools — ``delegate_task``, ``memory``, ``session_search``,
``todo`` — are ``model_tools._AGENT_LOOP_TOOLS``: their dispatch needs the
running ``AIAgent`` (its TodoStore, its MemoryStore, its session, its
provider credentials), so ``handle_function_call`` answers them with
"must be handled by the agent loop" and the hermes-tools MCP server has
never offered them at all.

That is fine when Hermes owns the model loop. It is not fine under the
``claude_code`` and ``codex_app_server`` runtimes, where the CLI owns the
loop and the MCP server IS the model's whole tool surface: the model asks
for a subagent, is told there is no such tool, and says so. Measured on
this install before the change::

    $ curl -s …/v1/chat/completions -d '{… "Call delegate_task once …"}'
    "The delegate_task tool is not available in this environment."

The MCP server cannot fix that on its own. It is a separate process the
CLI spawned; it has no agent, and — by design — no
``CLAUDE_CODE_OAUTH_TOKEN`` either (``write_mcp_config`` blanks it and
``scrub_environment`` drops it). A subagent built there would be built
from nothing and could not reach the same Claude Code the parent is
talking to, which is exactly the failure this module exists to prevent.

So the call goes home instead. The Hermes process — the one that holds
the credential and the live agent — listens on a Unix socket; the MCP
server forwards those four tools to it; the agent dispatches them through
``_invoke_tool``, with the same ``pre_tool_call`` hooks and middleware as
any other tool. Subagents are then spawned by the parent process,
inheriting its provider (``claude-code-cli``), its model, its session
lineage and its credential — the same Claude Code, by construction.

**Trust boundary, stated accurately.** The socket lives in a directory
this module creates 0700, the socket file is 0600, and every request
carries a token minted per session. None of that hides the address from
the model: Hermes' ``terminal`` runs *inside* the MCP server process, so
``env`` prints ``$HERMES_TOOL_BRIDGE_TOKEN``, and the ``--mcp-config``
file that carries it is readable by anything running as this user. The
token is a guard against confusion and against another *user*, not
against the model. Two things are therefore load-bearing and are enforced
on the server side, where the model cannot reach them:

* only the names in ``allowed_tools`` are dispatchable at all — so the
  bridge is not a general back door into the parent process; and
* the dispatcher (``claude_code_runtime.make_tool_bridge_dispatch``)
  re-checks the tool against the *agent's own* surface, because a
  subagent's MCP server is built from the whole registry and cannot know
  what that subagent is allowed to do.

What the bridge does newly permit, and what should be weighed as such:
code running in the MCP server process can now reach an ``AIAgent`` — its
own session's, or (same uid, shared config dir) another live session's —
for those four tools. Everything else about the boundary is unchanged.

Both ends live here so the wire format cannot drift: the server is
started by ``ClaudeCodeSession`` (:meth:`ToolBridge.start`), the client is
called by ``hermes_tools_mcp_server`` (:func:`call_bridged_tool`).
"""

from __future__ import annotations

import errno
import hmac
import json
import logging
import os
import secrets
import shutil
import socket
import stat
import tempfile
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Env vars carrying the bridge address into the MCP server process. Set in
#: the ``--mcp-config`` ``env`` block (write_mcp_config), which the CLI merges
#: over the server's inherited environment.
BRIDGE_SOCKET_ENV = "HERMES_TOOL_BRIDGE_SOCKET"
BRIDGE_TOKEN_ENV = "HERMES_TOOL_BRIDGE_TOKEN"
#: Comma-separated subset of :data:`BRIDGED_TOOLS` this child may call. The
#: server enforces it too; the env var is what stops the MCP server from
#: *advertising* a tool the dispatcher would then refuse.
BRIDGE_TOOLS_ENV = "HERMES_TOOL_BRIDGE_TOOLS"
#: Seconds the client waits for a reply. Must stay strictly below the CLI's
#: own ``MCP_TOOL_TIMEOUT`` (set in ``build_child_env``) so the readable
#: BridgeError wins the race against the CLI's opaque MCP abort.
BRIDGE_TIMEOUT_ENV = "HERMES_TOOL_BRIDGE_TIMEOUT"
DEFAULT_TIMEOUT_SECONDS = 1500.0

#: The tools that come home. Mirrors ``model_tools._AGENT_LOOP_TOOLS``; kept
#: as a literal so this module imports without model_tools (the client end
#: runs in a process that may not have loaded it yet) and asserted equal in
#: tests.
BRIDGED_TOOLS: tuple[str, ...] = ("todo", "memory", "session_search", "delegate_task")

#: One line of JSON per message, both directions. A tool result can be large
#: (a fan-out's aggregated JSON); the cap is a sanity bound, not a budget.
_MAX_MESSAGE_BYTES = 32 * 1024 * 1024
#: How long an accepted connection has to send its request before it is
#: dropped. Without this a peer that connects and says nothing parks a thread
#: for the life of the session — and anything running as this user can open
#: one, so "nobody would do that" is not an argument.
_REQUEST_READ_TIMEOUT = 30.0
#: How long the reply may take to leave. A client that stopped reading must
#: not pin the thread either.
_REPLY_WRITE_TIMEOUT = 120.0
#: Ceiling on EXPENSIVE dispatches running at once — a bridged delegate_task
#: is a whole synchronous subagent fan-out, so this bounds `claude` children
#: and spend. The other three (todo, memory, session_search) are cheap, the
#: CLI batches parallel tool calls freely, and metering them here only handed
#: the model hard errors to reason about: measured, 120 concurrent `todo`
#: calls against a ceiling of 8 produced 112 refusals. They are bounded by
#: _MAX_CONNECTIONS instead.
_METERED_TOOLS: frozenset = frozenset({"delegate_task"})
_MAX_CONCURRENT_CALLS = 8
#: Ceiling on open connections (threads). Higher than the dispatch ceiling
#: because most connections are cheap: a `hold` is a thread blocked on recv.
_MAX_CONNECTIONS = 64
#: How long :meth:`ToolBridge.close` waits for in-flight calls to unwind.
_CLOSE_JOIN_SECONDS = 5.0


def _error_text(exc: BaseException) -> str:
    """An exception as the model should see it.

    A raw ``str(exc)`` from deep inside a tool goes straight into the model's
    context unbounded and unredacted; every other Hermes tool error goes
    through the registry's bounding. Use the same path when it is importable
    (it is, in the Hermes process — this half only ever runs there).
    """
    text = f"{type(exc).__name__}: {exc}"
    try:
        from tools.registry import _bound_error_text

        return _bound_error_text(text)
    except Exception:  # pragma: no cover - registry not importable
        return text[:4000]


class BridgeError(RuntimeError):
    """The bridge could not carry the call (no socket, refused, timed out)."""


# ---------------------------------------------------------------------------
# Server — runs inside the Hermes process, next to the live agent
# ---------------------------------------------------------------------------


class ToolBridge:
    """Serve :data:`BRIDGED_TOOLS` to one child's MCP server.

    One bridge per ``ClaudeCodeSession``: its lifetime is the session's, its
    token is minted at :meth:`start` and its socket is removed at
    :meth:`close`. ``dispatch`` is re-read on every call rather than captured,
    because a warm session is shared across ``AIAgent`` instances (api_server
    builds one per request) and the call must land on the instance driving the
    current turn — the same reason ``ClaudeCodeSession.rebind`` exists.
    """

    def __init__(
        self,
        dispatch: Callable[[str, dict], Any],
        *,
        directory: Optional[str] = None,
        allowed_tools: "tuple[str, ...] | list[str]" = BRIDGED_TOOLS,
        max_concurrent_calls: int = _MAX_CONCURRENT_CALLS,
        name: str = "hermes-tool-bridge",
    ) -> None:
        self._dispatch = dispatch
        self._directory = directory
        self._allowed = tuple(allowed_tools)
        self._name = name
        self._token = ""
        self._path = ""
        self._owned_dir: Optional[str] = None
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._lock = threading.Lock()
        self._max_calls = max(1, int(max_concurrent_calls))
        self._slots = threading.BoundedSemaphore(self._max_calls)
        self._conn_slots = threading.BoundedSemaphore(_MAX_CONNECTIONS)
        self._in_flight = 0
        self._holds = 0
        self._last_active = time.monotonic()
        self._in_flight_lock = threading.Lock()
        self._workers: "set[threading.Thread]" = set()
        self._held: "set[socket.socket]" = set()

    # ---------- introspection ----------

    @property
    def socket_path(self) -> str:
        return self._path

    @property
    def token(self) -> str:
        return self._token

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        return self._allowed

    @property
    def running(self) -> bool:
        return self._sock is not None and not self._closed

    @property
    def in_flight(self) -> int:
        """Dispatches running right now.

        ``ClaudeCodeSession`` reads this to tell "the CLI has gone quiet"
        apart from "the CLI is quiet because it is waiting on us" — the
        difference between a turn that should be retired and one that is
        working. See ``_run_turn_locked``.
        """
        with self._in_flight_lock:
            return self._in_flight

    @property
    def active(self) -> bool:
        """True while the child is working on something this process serves.

        Either a bridged tool call is running here (``in_flight``), or the
        child's MCP server is holding the line while it runs an ordinary
        Hermes tool of its own (``holds``). Both mean the child is waiting on
        us rather than dead — a distinction a silence timeout cannot make on
        its own.
        """
        with self._in_flight_lock:
            return bool(self._in_flight or self._holds)

    @property
    def last_active(self) -> float:
        """``time.monotonic()`` of the last activity, start or end.

        The end matters as much as the start: when a bridged call returns, the
        CLI still needs a moment to take the result and speak. Treating the
        instant the bridge went idle as "silence started two minutes ago"
        killed a turn six seconds after its delegation succeeded.
        """
        with self._in_flight_lock:
            return self._last_active

    def set_dispatch(self, dispatch: Callable[[str, dict], Any]) -> None:
        """Point the bridge at the agent instance driving the current turn."""
        self._dispatch = dispatch

    def set_allowed_tools(self, allowed: "tuple[str, ...] | list[str]") -> None:
        """Re-narrow what may be dispatched when the driving agent changes."""
        self._allowed = tuple(allowed)

    # ---------- lifecycle ----------

    def start(self) -> None:
        """Bind, listen, and serve in a daemon thread. Idempotent."""
        with self._lock:
            if self._sock is not None:
                return
            if self._closed:
                raise BridgeError("tool bridge is closed")
            self._token = secrets.token_hex(32)
            sock, failures = None, []
            for candidate in self._candidate_paths():
                trial = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    trial.bind(candidate)
                    os.chmod(candidate, stat.S_IRUSR | stat.S_IWUSR)
                    trial.listen(_MAX_CONNECTIONS)
                except OSError as exc:
                    # A bind can fail where a mkdir succeeded — AF_UNIX sockets
                    # are not supported on every filesystem, and a $HERMES_HOME
                    # on NFS or SMB returns EOPNOTSUPP here. Falling through to
                    # the next candidate is the difference between "one machine
                    # keeps its subagents" and "this machine has none, and the
                    # model is told the tool does not exist" — the failure this
                    # module exists to end.
                    trial.close()
                    failures.append(f"{candidate}: {exc}")
                    self._cleanup_path()
                    continue
                sock, self._path = trial, candidate
                break
            if sock is None:
                raise BridgeError(
                    "tool bridge could not listen anywhere (" + "; ".join(failures) + ")"
                )
            self._sock = sock
            self._thread = threading.Thread(
                target=self._serve_forever, name=self._name, daemon=True
            )
            self._thread.start()
            logger.debug("tool bridge listening on %s", self._path)

    def _candidate_paths(self):
        """Socket paths to try, in order, each in a fresh 0700 directory.

        ``os.makedirs(mode=0o700, exist_ok=True)`` is a no-op on a directory
        that already exists, so trusting it would have left the socket in
        whatever mode ``$HERMES_HOME/claude-code`` happens to be (0755 on this
        install). A directory created fresh, per bridge, actually gets the
        mode — and it also gives the sweeper something it can identify.
        ``sun_path`` is 104 bytes on macOS, so a deep ``$HERMES_HOME`` falls
        back to the system temp dir, and a deep ``$TMPDIR`` fails loudly
        rather than binding somewhere unintended.
        """
        name = f"bridge-{secrets.token_hex(6)}"
        tried = []
        for parent in (self._directory, tempfile.gettempdir()):
            if not parent:
                continue
            tried.append(parent)
            directory = os.path.join(parent, name)
            candidate = os.path.join(directory, "s.sock")
            if len(candidate.encode("utf-8")) >= 100:
                continue
            try:
                os.makedirs(parent, exist_ok=True)
                os.mkdir(directory, 0o700)
            except OSError:
                logger.debug("tool bridge: cannot use %s", parent, exc_info=True)
                continue
            os.chmod(directory, stat.S_IRWXU)
            self._owned_dir = directory
            yield candidate
        if not tried:
            raise BridgeError("tool bridge: no directory to bind in")

    def close(self) -> None:
        """Stop listening, remove the socket, and wait briefly for in-flight
        calls. Idempotent.

        The wait is bounded: a bridged ``delegate_task`` can be minutes long
        and the caller (a session teardown) cannot block on it. Anything still
        running is left to finish; its reply goes to a client nobody is
        reading. Unlinking the socket does NOT stop it — an accepted
        connection is unaffected by the path disappearing — which is why
        ``_handle`` and ``_serve_hold`` check ``_closed`` themselves.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            sock, self._sock = self._sock, None
            workers = list(self._workers)
            held, self._held = list(self._held), set()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        # A hold is a thread parked on recv() until its peer hangs up. Nobody
        # is going to hang up on a session that has just closed, so shut the
        # connections down here rather than leak a thread per held tool call.
        for conn in held:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            deadline = time.monotonic() + _CLOSE_JOIN_SECONDS
            for worker in workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    worker.join(timeout=remaining)
                except RuntimeError:  # pragma: no cover - belt and braces
                    logger.debug("tool bridge: worker not joinable", exc_info=True)
            still = self.in_flight
            if still:
                logger.warning(
                    "tool bridge closed with %d call(s) still running; their "
                    "results will be discarded", still,
                )
        finally:
            # Never conditional on the join: a close() that raised on its way
            # here used to leave the socket and its 0700 directory behind, and
            # the caller swallows the exception, so it was silent.
            self._cleanup_path()
        logger.debug("tool bridge closed (%s)", self._path)

    def _cleanup_path(self) -> None:
        if self._path:
            try:
                os.unlink(self._path)
            except OSError:
                pass
        if self._owned_dir:
            shutil.rmtree(self._owned_dir, ignore_errors=True)
            self._owned_dir = None

    # ---------- serving ----------

    def _serve_forever(self) -> None:
        while True:
            sock = self._sock
            if sock is None or self._closed:
                return
            try:
                conn, _ = sock.accept()
            except OSError as exc:
                if self._closed or exc.errno in (errno.EBADF, errno.EINVAL):
                    return
                logger.debug("tool bridge accept failed", exc_info=True)
                continue
            # One thread per call: the CLI may run two MCP tools at once, and
            # a delegation can hold its connection for minutes. The semaphore
            # is taken BEFORE the thread exists, so a flood of connections
            # cannot become a flood of threads.
            if not self._conn_slots.acquire(blocking=False):
                self._refuse(conn, "tool bridge has too many open connections")
                continue
            worker = None
            try:
                worker = threading.Thread(
                    target=self._serve_connection,
                    args=(conn,),
                    name=f"{self._name}-call",
                    daemon=True,
                )
                # Registered and started under the SAME lock close() takes to
                # snapshot them. Started outside it, a close() landing in the
                # gap between add and start joined a thread that had not
                # begun — `RuntimeError: cannot join thread before it is
                # started` — which escaped before _cleanup_path() and leaked
                # the socket and its directory. Reproduced 3 times in 300
                # trials under connection pressure.
                with self._lock:
                    self._workers.add(worker)
                    worker.start()
            except BaseException:
                # Thread exhaustion must not kill the accept loop, or the
                # bridge goes permanently deaf while still reporting healthy.
                self._conn_slots.release()
                if worker is not None:
                    with self._lock:
                        self._workers.discard(worker)
                logger.warning("tool bridge could not start a worker", exc_info=True)
                self._refuse(conn, "tool bridge could not start a worker")

    def _refuse(self, conn: socket.socket, message: str) -> None:
        """Answer and close without spawning a worker."""
        try:
            conn.settimeout(_REPLY_WRITE_TIMEOUT)
            _write_message(conn, {"ok": False, "error": message})
        except Exception:
            logger.debug("tool bridge could not refuse cleanly", exc_info=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _serve_connection(self, conn: socket.socket) -> None:
        try:
            try:
                conn.settimeout(_REQUEST_READ_TIMEOUT)
                payload = _read_message(conn)
                conn.settimeout(None)
                if str(payload.get("op") or "") == "hold":
                    self._serve_hold(conn, payload)
                    return
                reply = self._handle(payload)
            except BaseException as exc:
                # BaseException, not Exception: a SystemExit or a
                # cancellation-shaped exception out of a tool would otherwise
                # leave the client waiting for its whole timeout on a socket
                # that a dead thread's traceback is still holding open.
                logger.debug("tool bridge request failed", exc_info=True)
                reply = {"ok": False, "error": _error_text(exc)}
            try:
                conn.settimeout(_REPLY_WRITE_TIMEOUT)
                _write_message(conn, reply)
            except (OSError, ValueError):
                logger.debug("tool bridge could not reply", exc_info=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass
            self._conn_slots.release()
            with self._lock:
                self._workers.discard(threading.current_thread())

    def _closed_reply(self) -> dict:
        return {"ok": False, "error": "tool bridge: this session has closed"}

    def _serve_hold(self, conn: socket.socket, payload: dict) -> None:
        """Keep the line open while the child runs a tool of its own.

        The child's MCP server opens one of these around every tool call it
        dispatches locally. Nothing is executed here: the connection IS the
        signal, so a crashed or killed child releases it automatically — no
        heartbeat to miss, no timer to leak.
        """
        token = str(payload.get("token") or "")
        if not self._token or not hmac.compare_digest(token, self._token):
            _write_message(conn, {"ok": False, "error": "tool bridge: authentication failed"})
            return
        with self._lock:
            # close() snapshots `_held` and replaces the set; a hold that
            # registered after that would never be shut down, and its thread
            # would sit on recv() holding a reference to a closed session's
            # agent until the peer process happened to die.
            if self._closed:
                _write_message(conn, self._closed_reply())
                return
            self._held.add(conn)
        with self._in_flight_lock:
            self._holds += 1
            self._last_active = time.monotonic()
        try:
            _write_message(conn, {"ok": True, "result": ""})
            conn.settimeout(None)
            while True:
                if not conn.recv(4096):
                    return
        except OSError:
            return
        finally:
            with self._lock:
                self._held.discard(conn)
            with self._in_flight_lock:
                self._holds -= 1
                self._last_active = time.monotonic()

    def _handle(self, payload: dict) -> dict:
        token = str(payload.get("token") or "")
        if not self._token or not hmac.compare_digest(token, self._token):
            logger.warning("tool bridge rejected a call with a bad token")
            return {"ok": False, "error": "tool bridge: authentication failed"}
        # close() stops the accept loop, but a connection already accepted has
        # its whole read window to arrive. Without this, a delegate_task could
        # start on a torn-down session's agent — subagents spawned and paid
        # for, answering a `claude` that has already been killed.
        if self._closed:
            return self._closed_reply()
        tool = str(payload.get("tool") or "")
        if tool not in self._allowed:
            return {"ok": False, "error": f"tool bridge: {tool!r} is not bridged"}
        args = payload.get("args")
        if not isinstance(args, dict):
            args = {}
        metered = tool in _METERED_TOOLS
        if metered and not self._slots.acquire(blocking=False):
            return {
                "ok": False,
                "error": (
                    f"tool bridge is already running {self._max_calls} "
                    f"{tool} calls; retry when one finishes"
                ),
            }
        with self._in_flight_lock:
            self._in_flight += 1
            self._last_active = time.monotonic()
        try:
            result = self._dispatch(tool, args)
        except Exception as exc:
            logger.exception("bridged tool %s raised", tool)
            return {"ok": False, "error": _error_text(exc)}
        finally:
            with self._in_flight_lock:
                self._in_flight -= 1
                self._last_active = time.monotonic()
            if metered:
                self._slots.release()
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=True, default=str)
            except Exception:
                result = str(result)
        return {"ok": True, "result": result}


# ---------------------------------------------------------------------------
# Client — runs inside the MCP server the CLI spawned
# ---------------------------------------------------------------------------


def bridge_address(env: Optional[dict] = None) -> tuple[str, str]:
    """``(socket_path, token)`` from the environment; ``("", "")`` when unset."""
    src = os.environ if env is None else env
    path = (src.get(BRIDGE_SOCKET_ENV) or "").strip()
    token = (src.get(BRIDGE_TOKEN_ENV) or "").strip()
    if not path or not token:
        return "", ""
    return path, token


def bridge_available(env: Optional[dict] = None) -> bool:
    """True when this process was given a bridge to call home on.

    The socket file's existence is not checked: it is created before the CLI
    is spawned and removed when the session closes, so a missing file at
    import time would only ever mean a race. A dead socket surfaces as a
    :class:`BridgeError` on the call, which the model sees as a tool error.
    """
    return all(bridge_address(env))


def bridged_tool_names(env: Optional[dict] = None) -> tuple[str, ...]:
    """Which of :data:`BRIDGED_TOOLS` this child may call.

    ``$HERMES_TOOL_BRIDGE_TOOLS`` carries the parent agent's own surface, so a
    leaf subagent is never *advertised* ``delegate_task`` only to be refused
    when it calls it. Unset means all of them (an older parent, or a session
    whose agent exposes no tool list).
    """
    if not bridge_available(env):
        return ()
    src = os.environ if env is None else env
    raw = (src.get(BRIDGE_TOOLS_ENV) or "").strip()
    if not raw:
        return BRIDGED_TOOLS
    names = {n.strip() for n in raw.split(",") if n.strip()}
    return tuple(n for n in BRIDGED_TOOLS if n in names)


def bridge_timeout(env: Optional[dict] = None) -> float:
    src = os.environ if env is None else env
    try:
        value = float(src.get(BRIDGE_TIMEOUT_ENV) or DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


def call_bridged_tool(
    tool: str,
    args: Optional[dict] = None,
    *,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> str:
    """Run ``tool`` on the live agent at the other end and return its result.

    Raises :class:`BridgeError` when there is no bridge, it cannot be reached,
    or the far end reported a failure — the caller turns that into a tool
    error the model can read.
    """
    path, token = bridge_address(env)
    if not path:
        raise BridgeError(
            f"{tool} needs the running agent and no tool bridge is configured "
            "for this process"
        )
    budget = timeout if timeout is not None else bridge_timeout(env)
    # An absolute deadline, not a per-operation timeout: `_read_message` loops
    # on recv, and a large multi-chunk reply (a fan-out's aggregated JSON is
    # exactly that) would otherwise get the full budget on EVERY chunk and
    # could outlast the CLI's own MCP ceiling — losing the race this budget
    # exists to win.
    ends_at = time.monotonic() + budget
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(budget)
    try:
        try:
            conn.connect(path)
        except OSError as exc:
            raise BridgeError(f"tool bridge at {path} is not answering: {exc}") from exc
        try:
            _write_message(conn, {"token": token, "tool": tool, "args": args or {}})
        except OSError as exc:
            # The far end hung up before we finished asking — a bridge at its
            # concurrency ceiling refuses at accept time. Its reason may still
            # be buffered on this socket; prefer it to "broken pipe".
            try:
                refusal = _read_message(conn)
            except Exception:
                raise BridgeError(
                    f"tool bridge refused the connection for {tool}: {exc}"
                ) from exc
            raise BridgeError(
                str(refusal.get("error") or f"tool bridge refused {tool}")
            ) from exc
        reply = _read_message(conn, ends_at=ends_at)
    except socket.timeout as exc:
        raise BridgeError(
            f"{tool} did not finish within {budget:.0f}s on the agent side"
        ) from exc
    finally:
        try:
            conn.close()
        except OSError:
            pass
    if not reply.get("ok"):
        raise BridgeError(str(reply.get("error") or "tool bridge: unknown failure"))
    return str(reply.get("result") or "")


class bridge_hold:
    """Tell the parent "the child is busy" for as long as this block runs.

    Used by the MCP server around EVERY tool it dispatches locally, not just
    the bridged four, because the parent's turn otherwise cannot tell a CLI
    quietly waiting on a 200-second ``terminal`` call from a CLI that has
    died, and would assume the worst.

    In practice it rarely matters: measured on Claude Code 2.1.252 the CLI
    keeps streaming while a tool call is outstanding, and a live delegation
    whose subagent slept 200 s finished in 4m27s under a 120 s
    ``silence_timeout`` without a single window being extended. It is kept
    because that streaming is an undocumented detail, and because the failure
    it prevents is retiring a session in the middle of its own successful
    work.

    Best-effort and silent: no bridge, or a bridge that will not answer, means
    no hold and exactly the old behaviour. It never raises, and the connect
    costs 0.151 ms per tool call measured over 300 round trips on this
    machine — small enough not to be worth a knob.
    """

    def __init__(self, env: Optional[dict] = None, *, connect_timeout: float = 2.0):
        self._env = env
        self._connect_timeout = connect_timeout
        self._conn: Optional[socket.socket] = None

    def __enter__(self) -> "bridge_hold":
        path, token = bridge_address(self._env)
        if not path:
            return self
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(self._connect_timeout)
        try:
            conn.connect(path)
            _write_message(conn, {"token": token, "op": "hold"})
            reply = _read_message(conn)
            if not reply.get("ok"):
                raise BridgeError(str(reply.get("error") or "hold refused"))
            conn.settimeout(None)
            self._conn = conn
        except Exception:
            logger.debug("tool bridge: could not hold the line", exc_info=True)
            try:
                conn.close()
            except OSError:
                pass
        return self

    def __exit__(self, *exc) -> bool:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        return False


def socket_is_live(path: str, *, timeout: float = 0.5) -> bool:
    """True when something is listening on ``path`` right now.

    The sweeper uses this instead of trusting mtime: a socket's mtime is fixed
    at bind, so an age test alone would eventually unlink the socket of a
    session that has simply been alive a long time, and that session's four
    tools would start failing with ENOENT and no server-side signal at all.
    """
    if not path or not os.path.exists(path):
        return False
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    try:
        conn.connect(path)
        return True
    except OSError:
        return False
    finally:
        try:
            conn.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Wire format — one JSON object per message, newline-terminated
# ---------------------------------------------------------------------------


def _write_message(conn: socket.socket, payload: dict) -> None:
    # ensure_ascii=True, deliberately: a tool result can carry surrogate code
    # points (subprocess output decoded with surrogateescape, odd filenames),
    # and those raise UnicodeEncodeError on a utf-8 encode. Escaped to \uXXXX
    # they are pure ASCII on the wire and json.loads restores them.
    data = json.dumps(payload, ensure_ascii=True, default=str).encode("ascii") + b"\n"
    conn.sendall(data)


def _read_message(conn: socket.socket, *, ends_at: Optional[float] = None) -> dict:
    chunks: list[bytes] = []
    total = 0
    while True:
        if ends_at is not None:
            left = ends_at - time.monotonic()
            if left <= 0:
                raise socket.timeout("tool bridge: read deadline passed")
            conn.settimeout(left)
        chunk = conn.recv(65536)
        if not chunk:
            if not chunks:
                raise BridgeError("tool bridge: connection closed with no message")
            break
        total += len(chunk)
        if total > _MAX_MESSAGE_BYTES:
            raise BridgeError("tool bridge: message too large")
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    raw = b"".join(chunks).split(b"\n", 1)[0]
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BridgeError(f"tool bridge: malformed message ({exc})") from exc
    if not isinstance(payload, dict):
        raise BridgeError("tool bridge: message was not an object")
    return payload
