"""Registry-facing sync handlers for MCP tools and utility tools (resources/prompts), plus the per-call recovery
ladder: trust gating, circuit breaker, auth (401) refresh, session-expired reconnect and dead-stdio respawn retry."""

import logging
import asyncio
import contextvars
import inspect
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional, Tuple
from tools.registry import tool_error
from tools.ansi_strip import strip_unicode_tags
from tools.mcp_tool_common import _exc_str, _sanitize_error, mcp_field, _core
from tools import mcp_tool_loop as _loop
from tools.mcp_tool_content import (
    _MCP_HARD_RESULT_CAP_CHARS, _cache_mcp_audio_block, _cache_mcp_image_block,
    _render_mcp_dropped_block_notice, _render_mcp_resource_block, _strip_reserved_meta_keys,
    _truncate_mcp_text_result)
from tools.mcp_tool_errors import _is_auth_error, _is_session_expired_error

logger = logging.getLogger("tools.mcp_tool")
_MISSING = object()

_NEEDS_REAUTH_MSG = (
    "MCP server '{s}' requires re-authentication. Run `hermes mcp login {s}` (or delete the tokens file under "
    "~/.hermes/mcp-tokens/ and restart). Do NOT retry this tool — ask the user to re-authenticate.")
_STDIO_NO_RESPAWN_MSG = (
    "MCP server '{s}' stdio subprocess had exited (this is not a timeout — the call never reached the server). A "
    "respawn was requested but no fresh session came back within {t:.0f}s. Wait a few seconds before retrying; if it "
    "keeps failing the server is not starting and needs the user.")
_STDIO_DIED_AGAIN_MSG = (
    "MCP server '{s}' respawned its stdio subprocess and it exited again immediately. The server is not starting "
    "cleanly — do NOT retry this tool; ask the user to check the server's command and its stderr log.")
_STDIO_OUTCOME_UNCERTAIN_MSG = (
    "MCP server '{s}' lost its stdio subprocess after the tool call began. The operation may have completed, so "
    "Hermes did not replay it. Do NOT retry automatically; inspect the external state first.")


def _trust_gate_check(server_name: str, tool_name: str) -> Optional[str]:
    """Approval gate for write-capable tools on ``trust: untrusted`` servers. None to proceed,
    else a ``tool_error``. Fail-closed: approval-system errors block."""
    if (_core._server_trust_levels.get(server_name, _core._TRUST_FULL) != _core._TRUST_UNTRUSTED
            or _core._tool_read_only_hints.get(server_name, {}).get(tool_name) is True):
        return None
    try:  # lazy: tools.approval routes the prompt to whichever surface owns the session
        from tools.approval_prompt import request_elicitation_consent
        answer = request_elicitation_consent(
            f"MCP tool '{tool_name}' on UNTRUSTED server '{server_name}' wants to run. This tool is write-capable "
            f"(no readOnlyHint=true annotation) and may modify external state.",
            f"Server '{server_name}' is configured 'trust: untrusted'. "
            f"Approve to run '{tool_name}' once, or deny to block it.",
            surface=f"mcp-trust/{server_name}")
    except Exception as exc:
        logger.error("MCP trust gate: approval check failed for %s.%s: %s", server_name, tool_name, exc, exc_info=True)
        return tool_error(f"MCP tool '{tool_name}' on untrusted server '{server_name}' was blocked: the approval "
                          f"system was unavailable (fail-closed).")
    if answer == "accept":
        return None
    logger.info("MCP trust gate: user %s '%s' on untrusted server '%s'",
                "cancelled" if answer == "cancel" else "denied", tool_name, server_name)
    return tool_error(f"The user did not approve running write-capable MCP tool '{tool_name}' on untrusted server "
                      f"'{server_name}'. The command was NOT run. Do not retry without explicit user direction.")


def _check_circuit_breaker(server_name: str) -> Optional[str]:
    """Open-breaker error, or None when calls may proceed. After the cooldown the breaker is
    half-open: the next call probes; success resets, failure re-bumps and re-arms the cooldown."""
    failures = _core._server_error_counts.get(server_name, 0)
    age = time.monotonic() - _core._server_breaker_opened_at.get(server_name, 0.0)
    if failures < _core._CIRCUIT_BREAKER_THRESHOLD or age >= _core._CIRCUIT_BREAKER_COOLDOWN_SEC:
        return None
    return tool_error(f"MCP server '{server_name}' is unreachable after {failures} consecutive failures. "
                      f"Auto-retry available in ~{max(1, int(_core._CIRCUIT_BREAKER_COOLDOWN_SEC - age))}s. Do NOT retry "
                      f"this tool yet — use alternative approaches or ask the user to check the MCP server.")


def _acquire_call_server(server_name: str, tool_timeout: float):
    """``(server, None)`` when a call may be dispatched, else ``(None, error)``. No session: a
    reconnect may be completing, so wait briefly before a breaker strike; still down -> ask the
    server task to rebuild (probing a dead transport would re-arm the breaker forever)."""
    from tools import mcp_tool_discovery as _discovery  # lazy: discovery -> registration -> handlers cycle
    not_connected = tool_error(f"MCP server '{server_name}' is not connected")
    server = _discovery._get_connected_server_for_call(server_name)
    wait = min(5.0, float(tool_timeout or 5.0))
    if server and (server.session or _loop._wait_for_server_session_ready(server, timeout=wait)):
        return server, None
    _core._bump_server_error(server_name)
    if server and _loop._signal_reconnect(server):
        return None, tool_error(f"MCP server '{server_name}' transport is down; reconnect requested. Do NOT retry this "
                                f"tool immediately — give it a few seconds to come back.")
    return None, not_connected


def _result_is_error(result) -> bool:
    """True only for a JSON payload carrying an ``error`` key (non-JSON = success)."""
    try:
        return "error" in json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return False


def _record_call_outcome(server_name: str, result) -> Any:
    """Breaker bookkeeping: an error payload from the tool itself still counts as a strike."""
    (_core._bump_server_error if _result_is_error(result) else _core._reset_server_error)(server_name)
    return result


def _strike(server_name: str, message: str, **extra) -> str:
    """Breaker strike + the ``tool_error`` payload for *message*."""
    _core._bump_server_error(server_name)
    return tool_error(message, **extra)


def _mcp_loop_running() -> bool:
    return _core._mcp_loop is not None and _core._mcp_loop.is_running()


def _lookup_reconnectable_server(server_name: str, require_loop: bool = False):
    """The registered server object when it can be signalled to reconnect, else None.
    With *require_loop*, also None unless the MCP loop is running (nothing to wait on)."""
    with _core._lock:
        srv = _core._servers.get(server_name)
    ok = srv is not None and hasattr(srv, "_reconnect_event") and (_mcp_loop_running() or not require_loop)
    return srv if ok else None


def _retry_once(server_name: str, retry_call, op_description: str, what: str):
    """Re-run ``retry_call`` after a recovery step. Returns the result (closing the breaker)
    when it is not an error payload; None when the retry raised or errored (caller falls through)."""
    try:
        result = retry_call()
    except Exception as retry_exc:
        logger.warning("MCP %s/%s retry after %s failed: %s", server_name, op_description, what, retry_exc)
        return None
    if _result_is_error(result):
        return None
    _core._reset_server_error(server_name)
    return result


def _handle_auth_error_and_retry(server_name: str, exc: BaseException, retry_call, op_description: str):
    """OAuth recovery + one retry; None when *exc* is not an auth error. ``handle_401`` decides
    viability; if viable, signal a reconnect (fresh credentials), wait ready, retry once. Any
    failure returns the structured ``needs_reauth`` error so the model stops refreshing."""
    if not _is_auth_error(exc):
        return None
    from tools.mcp_oauth_manager import get_manager
    try:
        recovered = _loop._run_on_mcp_loop(lambda: get_manager().handle_401(server_name, None), timeout=10)
    except Exception as rec_exc:
        logger.warning("MCP OAuth '%s': recovery attempt failed: %s", server_name, rec_exc)
        recovered = False
    if recovered:
        srv = _lookup_reconnectable_server(server_name)
        # Recovery + reconnect is independent evidence of viability: close the breaker here, not only on
        # retry success (else a failing retry pins it open forever).
        if srv is not None and _loop._signal_reconnect_and_wait(
                server_name, srv, op_description=f"{op_description} after OAuth recovery", timeout=15):
            _core._reset_server_error(server_name)
        result = _retry_once(server_name, retry_call, op_description, "auth recovery")
        if result is not None:
            return result
    return _strike(server_name, _NEEDS_REAUTH_MSG.format(s=server_name), needs_reauth=True, server=server_name)


def _handle_session_expired_and_retry(server_name: str, exc: BaseException, retry_call, op_description: str):
    """Transport reconnect + one retry on session expiry; None to fall through. Skips
    ``handle_401``: the token is valid, only the server-side session is stale.

    Unlike :func:`_handle_auth_error_and_retry`, this does **not** call the OAuth manager's ``handle_401`` —
    the access token is still valid, only the server-side session state is stale. Setting
    ``_reconnect_event`` causes the server task's lifecycle loop to tear down the current
    ``streamablehttp_client`` + ``ClientSession`` and rebuild them, reusing the existing OAuth provider
    instance. See #13383.
    """
    srv = _lookup_reconnectable_server(server_name, require_loop=True) if _is_session_expired_error(exc) else None
    if srv is None:
        return None
    logger.info("MCP server '%s': %s failed with session-expired error (%s); signalling transport reconnect "
                "and retrying once.", server_name, op_description, exc)
    if not _loop._signal_reconnect_and_wait(server_name, srv, op_description=op_description, timeout=15):
        logger.warning("MCP server '%s': reconnect did not ready within 15s after session-expired error; "
                       "falling through to error response.", server_name)
        return None
    return _retry_once(server_name, retry_call, op_description, "session reconnect")


class _StdioChildExited(RuntimeError):
    """Stdio subprocess gone when (or while) a call ran. Deliberately NOT a TimeoutError."""

    def __init__(self, message: str, *, in_flight: bool):
        super().__init__(message)
        self.in_flight = in_flight


def _handle_stdio_child_exited_and_retry(server_name: str, exc: Exception, retry_call, op_description: str):
    """Respawn a dead stdio child; retry once only when it was dead before dispatch.

    A mid-call exit is ambiguous: the server may have applied a side effect before its
    response pipe disappeared. Reconnect for future calls but never replay that operation.
    None means this is not our error. This function never spawns itself: it sets
    ``_reconnect_event`` and waits, so spawn frequency stays governed by ``run()``'s rapid-drop
    budget. A pre-dispatch retry whose child dies again reports and stops.

    Why retrying here cannot hot-cycle respawns: this function never spawns anything. It sets
    ``_reconnect_event`` (one signal, same as before) and waits for the server task to publish a fresh
    session. Spawn frequency stays governed entirely by ``run()``'s rapid-drop budget, which parks a
    transport that keeps dropping without proving healthy (#62212).
    """
    if not isinstance(exc, _StdioChildExited):
        return None
    reconnected = False
    srv = _lookup_reconnectable_server(server_name)
    if srv is not None:
        action = "reconnecting without replay" if exc.in_flight else "respawning and retrying once"
        logger.info("MCP server '%s': %s found the stdio subprocess dead (%s); %s.",
                    server_name, op_description, exc, action)
        if _mcp_loop_running():
            reconnected = _loop._signal_reconnect_and_wait(
                server_name, srv, op_description=op_description, timeout=_core._STDIO_RESPAWN_WAIT_SEC)
        else:  # No MCP loop to wait on (non-async adapters, tests): still request the respawn.
            _loop._signal_reconnect(srv)
    if exc.in_flight:
        return _strike(
            server_name,
            _STDIO_OUTCOME_UNCERTAIN_MSG.format(s=server_name),
            outcome_uncertain=True,
        )
    if not reconnected:
        return _strike(server_name, _STDIO_NO_RESPAWN_MSG.format(s=server_name, t=_core._STDIO_RESPAWN_WAIT_SEC))
    try:
        return _record_call_outcome(server_name, retry_call())
    except _StdioChildExited as retry_exc:
        # Died again right after respawn: broken server; run()'s budget takes it to the park.
        logger.warning("MCP server '%s': %s stdio subprocess exited again right after respawn (%s); not retrying "
                       "further.", server_name, op_description, retry_exc)
        if retry_exc.in_flight:
            return _strike(
                server_name,
                _STDIO_OUTCOME_UNCERTAIN_MSG.format(s=server_name),
                outcome_uncertain=True,
            )
        return _strike(server_name, _STDIO_DIED_AGAIN_MSG.format(s=server_name))
    except Exception as retry_exc:
        logger.warning("MCP %s/%s retry after stdio respawn failed: %s", server_name, op_description, retry_exc)
        return _strike(server_name, _sanitize_error(
            f"MCP call failed after respawning the stdio subprocess for '{server_name}': "
            f"{type(retry_exc).__name__}: {_exc_str(retry_exc)}"))


def _dispatch(server_name: str, server: Any, op: str, call, tool_timeout: float, recoverers,
              on_final_failure: Callable[[BaseException], None], record_outcome: bool = False,
              failure_note: Optional[str] = None) -> str:
    """Mark the call started on *server* (doubles may lack ``mark_tool_call``), run coroutine function *call*
    on the MCP loop and, on failure, walk ``recoverers`` (``(server_name, exc, retry_call, op) -> Optional[str]``,
    None = not its kind; order matters). Unrecovered exceptions go through ``on_final_failure`` and become the
    generic call-failed error. ``record_outcome`` applies breaker bookkeeping to the FIRST attempt only."""
    if callable(getattr(server, "mark_tool_call", None)):
        server.mark_tool_call()

    def call_once():
        return _loop._run_on_mcp_loop(call, timeout=tool_timeout)

    try:
        result = call_once()
        return _record_call_outcome(server_name, result) if record_outcome else result
    except InterruptedError:
        return tool_error("MCP call interrupted: user sent a new message")
    except Exception as exc:
        for recover in recoverers:
            recovered = recover(server_name, exc, call_once, op)
            if recovered is not None:
                return recovered
        on_final_failure(exc)
        # ``failure_note`` (a filled-in default) rides every return path: a
        # model that sees an error naming a parameter IT DID NOT PASS cannot
        # remove it, and would retry identically forever.
        return tool_error(_sanitize_error(
            f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"
            + (("\n\n" + failure_note) if failure_note else "")))


@asynccontextmanager
async def _track_inflight_rpc(server: Any, server_name: str, op: str):
    """Register the running RPC so teardown can fail it fast. A deliberate teardown
    (``_reconnecting`` set first) turns the cancel into a retryable RuntimeError; external
    cancels propagate unchanged. Doubles without ``_inflight_tasks`` skip tracking.

    Every user-visible request family wraps its RPC in this context (#48069 salvage). If a deliberate
    reconnect/shutdown teardown cancels the task (``_fail_inflight_calls`` sets ``_reconnecting`` first),
    the cancel is converted into a clean retryable RuntimeError instead of a raw CancelledError; external
    cancels (caller timeout, user interrupt) propagate unchanged.
    """
    inflight, task = getattr(server, "_inflight_tasks", None), asyncio.current_task()
    tracked = task is not None and inflight is not None
    if tracked:
        inflight.add(task)
    try:
        yield
    except asyncio.CancelledError:
        if getattr(server, "_reconnecting", False):
            raise RuntimeError(f"MCP {op} on '{server_name}' was aborted by a reconnect teardown; retry the "
                               f"request on the rebuilt session") from None
        raise
    finally:
        if tracked:
            inflight.discard(task)


async def _call_tool_racing_stdio_death(server, server_name: str, tool_name: str, args: dict):
    """``session.call_tool`` that fails fast when the stdio child is/gets dead: pre-call (a dead
    child must not hold the slot for the full timeout) and mid-call (race against
    ``_watch_stdio_children``). Both raise :class:`_StdioChildExited` for the respawn path, which
    owns the reconnect signal; only pre-call failure is safe to replay. callable()/``is True``
    because MagicMock attributes are truthy."""
    # Fast-fail (#81995): a stdio subprocess that is already dead must not own this call slot — fail
    # immediately instead of waiting out the full tool timeout on a transport nobody will ever answer.
    _stdio_dead = getattr(server, "_stdio_children_dead", None)
    if callable(_stdio_dead) and _stdio_dead() is True:
        raise _StdioChildExited(
            f"MCP stdio subprocess for '{server_name}' had already exited when the call was dispatched",
            in_flight=False,
        )
    _call_coro = server.session.call_tool(tool_name, arguments=args)
    _watch_children = getattr(server, "_watch_stdio_children", None)
    if not (inspect.iscoroutinefunction(_watch_children) and asyncio.iscoroutine(_call_coro)):
        # Stubbed sessions return a non-awaitable, or there is no child-watcher to race: plain await.
        return await _call_coro if asyncio.iscoroutine(_call_coro) else _call_coro
    # Fast-fail machinery (#81995): the RPC races a stdio-children watcher so a dead subprocess fails the
    # call immediately instead of riding out the full tool timeout.
    rpc_task = asyncio.ensure_future(_call_coro)
    watch_task = asyncio.ensure_future(_watch_children())
    try:
        done, _pending = await asyncio.wait({rpc_task, watch_task}, return_when=asyncio.FIRST_COMPLETED)
        if watch_task in done and not rpc_task.done():
            rpc_task.cancel()
            raise _StdioChildExited(
                f"MCP stdio subprocess for '{server_name}' exited mid-call",
                in_flight=True,
            )
        try:
            return await rpc_task
        except Exception as exc:
            # The SDK usually sees the closed pipe before the 250 ms watcher poll does. On a stdio
            # server a transport-closure error after dispatch is the same ambiguous mid-call death;
            # it must not fall through to the session-expired recoverer, which replays the call.
            _is_http = getattr(server, "_is_http", None)
            if callable(_is_http) and _is_http() is False and _is_session_expired_error(exc):
                raise _StdioChildExited(
                    f"MCP stdio subprocess for '{server_name}' closed its transport mid-call",
                    in_flight=True,
                ) from exc
            raise
    finally:
        watch_task.cancel()
        if not rpc_task.done():
            rpc_task.cancel()
        await asyncio.gather(rpc_task, watch_task, return_exceptions=True)


# ---------------------------------------------------------- result rendering

def _error_result_text(result) -> str:
    """Concatenated text of an ``isError`` result's blocks (EmbeddedResource error payloads
    carry text under ``.resource.text``)."""
    texts = (getattr(b, "text", None) or getattr(getattr(b, "resource", None), "text", None) for b in (result.content or []))
    return "".join(str(t) for t in texts if t)


def _render_content_blocks(result, server_name: str) -> Tuple[str, int]:
    """Text passes through; image/audio blocks are cached (MEDIA: tags); resource blocks are
    materialized rather than silently dropped; unsupported blocks become an inline drop notice
    (kimi-code#3227). Returns ``(text, usable_parts)`` — the count of REAL rendered blocks
    (whitespace-only text and drop notices excluded) that the structuredContent arbitration uses."""
    parts: List[str] = []
    usable_parts = 0
    # MCP tool results can also include ImageContent blocks (screenshot / Blockbench / Playwright etc.);
    # cache those via the gateway's image-cache helper so they flow through Hermes' MEDIA: tag convention
    # and out to messaging adapters that render images natively. Without this, image blocks were silently
    # dropped and the agent got an empty response. Distilled from #17915 (c3115644151) and #10848
    # (gnanirahulnutakki), both too stale to cherry-pick. #10848's approach (integrate with Hermes' MEDIA
    # tag + cache_image_from_bytes) was the cleaner of the two — plugs into existing infrastructure.
    for block in (result.content or []):
        if getattr(block, "text", None):
            parts.append(strip_unicode_tags(block.text))
            if block.text.strip():
                usable_parts += 1
            continue
        rendered = _cache_mcp_image_block(block) or _cache_mcp_audio_block(block) or _render_mcp_resource_block(block, server_name)
        if rendered:
            parts.append(rendered)
            usable_parts += 1
            continue
        block_type = getattr(block, "type", None) or type(block).__name__
        if block_type in {"text", "resource", "audio", "image"}:  # benign empty render
            logger.debug("MCP %s: content block type %r rendered empty", server_name, block_type)
        else:
            logger.warning("MCP %s: dropping unsupported content block type %r", server_name, block_type)
            # Surface the drop to the MODEL, not just the log: a silent drop leaves the agent
            # believing the tool returned less than it did, with no way to recover.
            parts.append(_render_mcp_dropped_block_notice(block, block_type))
    # Hard-cap pathological payloads; ordinary large results pass to spillover.
    return _truncate_mcp_text_result("\n".join(parts)), usable_parts


def _capped_structured_content(result):
    """``structuredContent`` (or None); over the hard cap it degrades to the head+tail
    truncated JSON string (multi-MB JSON flood guard)."""
    # Hard-cap pathological payloads before they propagate (#56059); ordinary large results pass untouched
    # to the spillover layer.
    # content and structuredContent are ALTERNATIVES — never both forwarded (ported from
    # MoonshotAI/kimi-code#3234). Spec-following servers already render their data into content (the
    # verbatim dual-emit SHOULD, or a faithful human reorganisation), so forwarding both sent the same
    # information to the model twice. content wins whenever it rendered anything usable; there is no
    # reliable signal that the structured payload is richer than what the server put in content (semantic
    # equality misses faithful reorganisations, size ratios misjudge both directions), so no heuristic is
    # attempted. structuredContent fills in only when the content blocks rendered effectively empty, which
    # keeps structuredContent-only servers working. Server-level `_meta` is also surfaced (ported from
    # MoonshotAI/kimi-code#2596): servers return namespaced metadata there (validated contracts,
    # browser-handoff payloads, ...) that was previously invisible to the agent. Protocol-reserved keys are
    # dropped first (kimi-code#2600) — per the MCP spec's key-name rules a prefix is reserved when a
    # `modelcontextprotocol` or `mcp` label is followed by at least one more label (e.g.
    # `modelcontextprotocol.io/...`, `tools.mcp.com/...`); those carry host/protocol plumbing, not
    # model-facing data. Unprefixed and vendor-namespaced keys (`com.example.mcp/...`) pass through — their
    # semantics belong to the server.
    structured = mcp_field(result, "structured_content", "structuredContent")
    try:
        as_json = json.dumps(structured, ensure_ascii=False, default=str) if structured is not None else ""
    except (TypeError, ValueError):
        return structured
    return _truncate_mcp_text_result(as_json) if len(as_json) > _MCP_HARD_RESULT_CAP_CHARS else structured


def _render_call_tool_result(result, server_name: str, defaults_note: Optional[str] = None) -> str:
    """Pure: ``CallToolResult`` -> handler JSON. ``content`` and ``structuredContent`` are
    ALTERNATIVES, never both forwarded (kimi-code#3234): spec-following servers already render
    their data into content, so forwarding both sent it twice. content wins whenever it rendered
    anything usable (no richness heuristic is attempted — none is reliable); structuredContent
    fills in only when the blocks rendered effectively empty, keeping structuredContent-only
    servers working. ``_meta`` minus reserved keys is always surfaced."""
    if mcp_field(result, "is_error", "isError", False):
        return tool_error(_sanitize_error(_truncate_mcp_text_result(
            (_error_result_text(result) or "MCP tool returned an error")
            + (("\n\n" + defaults_note) if defaults_note else ""))))
    text_result, usable_parts = _render_content_blocks(result, server_name)
    structured = _capped_structured_content(result)
    meta = _strip_reserved_meta_keys(mcp_field(result, "meta", "meta"))
    if structured is not None and usable_parts > 0:
        structured = None  # drop notices do not count as usable content
    # `defaults_note` joins the payload branch: a silently cheapened result the
    # model cannot recover from is worse than an expensive one, so whenever a
    # default was filled in, the result says which.
    if structured is None and meta is None and not defaults_note:
        return json.dumps({"result": text_result}, ensure_ascii=False)
    # Key order is part of the output: "result" leads when there is text, otherwise "_meta" precedes it.
    payload: Dict[str, Any] = {"result": text_result} if text_result else {}
    # Cap structuredContent too — a malicious server could flood context via a multi-MB JSON payload
    # (#56059). When the serialized form exceeds the hard cap, replace it with the truncated string (head +
    # tail preserved) so it degrades gracefully instead of flooding downstream.
    if structured is not None:
        payload["structuredContent" if text_result else "result"] = structured
    if meta is not None:
        payload["_meta"] = meta
    if defaults_note:
        payload["_hermes"] = defaults_note
    payload.setdefault("result", text_result)
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):  # Non-serializable metadata: drop the extras, keep the call.
        # The defaults note is a plain string and always survives this fallback.
        fallback: Dict[str, Any] = {"result": text_result}
        if defaults_note:
            fallback["_hermes"] = defaults_note
        return json.dumps(fallback, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Composio: the cheap default, for tools whose own default is the expensive one
# ---------------------------------------------------------------------------
#
# Composio fronts ~500 toolkits whose tools were authored one at a time, and a
# handful of them default to returning far more data than any question needs.
# Measured 2026-09-04 against a real Gmail account, 40 messages, one query,
# with the message-id set verified identical across every arm:
#
#     verbose omitted, include_payload omitted   1,674,724 B   41,868 B/msg
#     verbose omitted, include_payload: false      354,090 B    8,852 B/msg
#     verbose: false,  include_payload omitted     348,378 B    8,709 B/msg
#     verbose: false,  include_payload: false       33,690 B      842 B/msg
#
# 49.7x between the corners, and the expensive corner is exactly what a model
# that reads the schema and passes nothing gets: NEITHER description mentions
# bytes. `verbose` describes speed ("~75% improvement"); `include_payload`
# claims "metadata only" while in fact leaving the whole decoded body in.
#
# The cost is not the bytes. A result this size is spilled to a file and the
# model is handed a path, so "who emailed me today" becomes grepping a file --
# one such turn was measured at 103.7 s to first token.
#
# Four rules, each of which was learned the expensive way:
#
#   1. ONLY where the caller omitted the parameter. A default is a fallback,
#      never an override; a model that asked for the body gets the body.
#   2. ONLY on Composio's own servers, identified by the session URL. A tool
#      called GMAIL_FETCH_EMAILS on somebody else's server is somebody else's
#      tool, and inheriting a broker's identity by name alone is a bug this
#      codebase has already shipped once.
#   3. ONLY parameters the live schema still declares. If Composio renames
#      `verbose`, the table stops applying rather than posting an unknown key.
#      Unknown slug = no opinion (the multiplexer can reach tools that were
#      never preloaded, so their schemas are not in `_tools`).
#   4. SAY SO in the result -- INCLUDING when the call fails. A default that
#      silently removes the answer is unrecoverable: the model cannot ask for
#      what it does not know it lost. The failure case is the sharper one. If
#      Composio ever rejects a key this table injected, the model sees an
#      error naming a parameter IT DID NOT PASS, cannot remove it, and retries
#      identically forever -- so every return path carries the note, not just
#      the happy one. One line costs ~200 bytes against ~1.6 MB.
#
# Both invocation paths land here, which is the point of doing it in the
# handler: the model calls these tools directly AND through
# COMPOSIO_MULTI_EXECUTE_TOOL, and the byte behaviour differs per path (the
# list fetch is 167,231 B direct against 49,218 B muxed; the by-id fetch
# differs only by a ~150 B envelope). A default that landed on one path only
# would make the cost depend on which wrapper the model happened to pick.
#
# What is deliberately NOT in the table:
#
#   GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID. `format: "metadata"` is a real 8.9x
#   and Composio's own description recommends it -- but this tool's entire job
#   is "what does this message say", so defaulting it away deletes the answer
#   to the question that was asked. Lists are for finding; fetches are for
#   reading, and the reading tool is left alone.
#
#   GOOGLEDRIVE_GET_ABOUT. Its `fields` default is expensive, but no cheap
#   value has been measured to be VALID here, and on the neighbouring
#   GOOGLEDRIVE_FIND_FILE an unrecognised selector does not error -- it falls
#   back to everything, costing 4x what omitting the parameter costs. A guess
#   would be a well-formed value that resolves to nothing, which is worse than
#   no value at all. Measure a real selector first, then add it.
_COMPOSIO_HOSTS = ("composio.dev",)

_COMPOSIO_MULTIPLEXER = "COMPOSIO_MULTI_EXECUTE_TOOL"

# slug -> {parameter: cheap value}. Uppercase keys; slugs are matched folded.
_COMPOSIO_CHEAP_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # The survey tool. Both flags together leave ids, labels, timestamps,
    # subject, sender, to and a preview snippet -- 842 B/message, which is the
    # metadata-only answer neither flag's description offers.
    "GMAIL_FETCH_EMAILS": {"verbose": False, "include_payload": False},
}

# What to tell the model it can do about it, per slug.
_COMPOSIO_DEFAULT_HINTS: Dict[str, str] = {
    "GMAIL_FETCH_EMAILS": (
        "message bodies are omitted; pass verbose=true for full text (naming "
        "either flag yourself turns this off entirely), or fetch one message "
        "by id"
    ),
}


def _is_composio_server(server: Any) -> bool:
    """True when this server is a Composio session, by URL not by name.

    The config key is whatever the user called the server, so the name proves
    nothing. The URL is the thing Composio actually owns.
    """
    try:
        url = str((getattr(server, "_config", None) or {}).get("url") or "")
    except Exception:
        return False
    if not url:
        return False  # stdio servers have no url; nothing to match.
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    # Parsed, not substring-matched. `composio.dev.example.com`,
    # `notcomposio.dev` and `https://x/?ref=composio.dev` all contain the
    # string and are all somebody else.
    return any(host == h or host.endswith("." + h) for h in _COMPOSIO_HOSTS)


def _composio_declares(server: Any, slug: str, param: str) -> Optional[bool]:
    """Does the live schema for ``slug`` declare ``param``?

    True / False when the tool is known to this server, None when it is not --
    the multiplexer can reach tools that were never preloaded, and "I have no
    schema" must not read as "the parameter is gone".
    """
    try:
        tools = getattr(server, "_tools", None) or []
    except Exception:
        return None
    for t in tools:
        if str(getattr(t, "name", "") or "").upper() != slug:
            continue
        schema = mcp_field(t, "input_schema", "inputSchema") or {}
        if not isinstance(schema, dict):
            return None
        props = schema.get("properties")
        if not isinstance(props, dict):
            return None
        return param in props
    return None


def _composio_default_one(server: Any, slug: Any, arguments: Any):
    """Fill omitted cheap defaults for one Composio tool call.

    Returns ``(arguments, ["k=v", ...])`` -- the same object and an empty list
    when nothing applied, so callers can test for change by the list alone.
    """
    key = str(slug or "").upper()
    table = _COMPOSIO_CHEAP_DEFAULTS.get(key)
    if not table:
        return arguments, []
    # A mux entry may carry no `arguments` at all -- `{"tool_slug": "..."}`
    # with nothing else is the CHEAPEST call to write and the MOST expensive
    # to answer, so it is exactly the one that must not be skipped. Anything
    # that is neither absent nor a dict (a JSON string, a list) is left alone:
    # it is the server's job to reject it, not ours to reinterpret it.
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return arguments, []
    # Rule 1, and it is stronger than "per parameter". These tables are all
    # about one thing -- how much this tool returns -- so a caller who sets
    # ANY of a slug's levers is steering that, and the rest of the table
    # stands down for the call. Per-parameter filling looked equivalent and
    # was not: the note tells a model "pass verbose=true for full text", and
    # if `include_payload: false` were still injected on the retry the model
    # would be handed the same advice a second time over a result it had
    # already tried to widen. A default the caller cannot get out of in one
    # move is not a default.
    if any(param in arguments for param in table):
        return arguments, []
    filled = {}
    for param, value in table.items():
        if _composio_declares(server, key, param) is False:
            continue  # rule 3: the schema dropped it; do not post it.
        filled[param] = value
    if not filled:
        return arguments, []
    out = dict(arguments)
    out.update(filled)
    return out, ["%s=%s" % (k, json.dumps(v)) for k, v in sorted(filled.items())]


def _composio_defaults_note(applied) -> str:
    bits = []
    for slug, kvs in applied:
        bit = "%s: %s" % (slug, ", ".join(kvs))
        hint = _COMPOSIO_DEFAULT_HINTS.get(slug)
        if hint:
            bit += " (%s)" % hint
        bits.append(bit)
    return (
        "Hermes filled in cheaper values for parameters you omitted, because "
        "this tool's own defaults return up to 50x more data than the question "
        "usually needs. " + "; ".join(bits)
    )


def _apply_composio_defaults(server: Any, tool_name: str, args: Any):
    """Return ``(args, note_or_None)`` for one outbound MCP call.

    Covers both invocation paths: the tool called directly, and the same tool
    reached through ``COMPOSIO_MULTI_EXECUTE_TOOL``.
    """
    if not isinstance(args, dict) or not _is_composio_server(server):
        return args, None

    if str(tool_name or "").upper() == _COMPOSIO_MULTIPLEXER:
        calls = args.get("tools")
        if not isinstance(calls, list):
            return args, None
        applied = []
        new_calls = []
        for entry in calls:
            if not isinstance(entry, dict):
                new_calls.append(entry)
                continue
            slug = entry.get("tool_slug")
            new_arguments, kvs = _composio_default_one(
                server, slug, entry.get("arguments"),
            )
            if not kvs:
                new_calls.append(entry)
                continue
            replacement = dict(entry)
            replacement["arguments"] = new_arguments
            new_calls.append(replacement)
            applied.append((str(slug).upper(), kvs))
        if not applied:
            return args, None
        out = dict(args)
        out["tools"] = new_calls
        return out, _composio_defaults_note(applied)

    new_args, kvs = _composio_default_one(server, tool_name, args)
    if not kvs:
        return args, None
    return new_args, _composio_defaults_note([(str(tool_name).upper(), kvs)])


def _make_tool_handler(server_name: str, tool_name: str, tool_timeout: float):
    """Sync registry handler (``handler(args_dict, **kwargs) -> str``) calling an MCP tool via the background loop."""
    op = f"tools/call {tool_name}"

    def _handler(args: dict, **kwargs) -> str:
        # Security boundary: untrusted-server write tools need approval before ANY transport work (incl. lazy spawn).
        error = _trust_gate_check(server_name, tool_name) or _check_circuit_breaker(server_name)
        if error is not None:
            return error
        server, error = _acquire_call_server(server_name, tool_timeout)
        if server is None:
            return error

        # Composio's expensive defaults, filled in only where the caller left
        # the parameter out. This is the one seam both invocation paths cross:
        # `args` is read exactly once below, at session.call_tool().
        args, _defaults_note = _apply_composio_defaults(server, tool_name, args)

        async def _call():
            async with server._rpc_lock, _track_inflight_rpc(server, server_name, op):
                server._pending_call_context = contextvars.copy_context()  # for the elicitation callback
                try:
                    result = await _call_tool_racing_stdio_death(server, server_name, tool_name, args)
                finally:
                    server._pending_call_context = None
            if getattr(server, "_mark_session_proven", None) is not None:  # round-trip done: transport healthy
                server._mark_session_proven()
            return _render_call_tool_result(result, server_name, _defaults_note)

        def _on_failure(exc):
            _core._bump_server_error(server_name)
            logger.error("MCP tool %s/%s call failed: %s", server_name, tool_name, exc)
        return _dispatch(
            server_name, server, op, _call, tool_timeout,
            (_handle_stdio_child_exited_and_retry, _handle_auth_error_and_retry, _handle_session_expired_and_retry),
            _on_failure, record_outcome=True, failure_note=_defaults_note)
    return _handler


def _make_utility_handler(op: str, log_label: str, rpc, render, required: Optional[str] = None):
    """``(server_name, tool_timeout) -> sync handler`` for one utility tool: ``rpc(session, args,
    server_name)`` awaited under ``_rpc_lock``, ``render(result, server_name)`` -> JSON-able
    payload, ``required`` validated before any transport work."""
    def _factory(server_name: str, tool_timeout: float):
        def _handler(args: dict, **kwargs) -> str:
            from tools import mcp_tool_discovery as _discovery  # lazy: import cycle
            server = _discovery._get_connected_server_for_call(server_name)
            if not server or not server.session:
                return tool_error(f"MCP server '{server_name}' is not connected")
            if required and not args.get(required):
                return tool_error(f"Missing required parameter '{required}'")

            async def _call():
                async with server._rpc_lock:
                    result = await rpc(server.session, args, server_name)
                return json.dumps(render(result, server_name), ensure_ascii=False)
            return _dispatch(
                server_name, server, op, _call, tool_timeout,
                (_handle_auth_error_and_retry, _handle_session_expired_and_retry),
                lambda exc: logger.error("MCP %s/%s failed: %s", server_name, log_label, exc))
        return _handler
    return _factory


def _pick(obj, *specs) -> dict:
    """``{out_key: value}`` for each ``(out_key, attr[, truthy])`` present on *obj* (presence check so SDK models
    and stubs behave alike; ``truthy`` also skips falsy). Key order = spec order."""
    entry = {}
    for out_key, attr, *truthy in specs:
        value = getattr(obj, attr, _MISSING)
        if value is not _MISSING and (value or not (truthy and truthy[0])):
            entry[out_key] = value
    return entry


def _render_resource_list(all_resources, server_name: str) -> dict:
    resources = []
    for r in all_resources:
        entry = _pick(r, ("uri", "uri"), ("name", "name"), ("description", "description", True))
        if "uri" in entry:
            entry["uri"] = str(entry["uri"])
        mime = mcp_field(r, "mime_type", "mimeType")
        if mime:
            entry["mimeType"] = mime  # camelCase: this is the tool's own JSON output shape
        resources.append(entry)
    return {"resources": resources}


def _render_read_resource(result, server_name: str) -> dict:
    parts: List[str] = []
    for block in getattr(result, "contents", []):
        if getattr(block, "text", None) is not None:
            parts.append(strip_unicode_tags(block.text))
        elif getattr(block, "blob", None) is not None:  # binary -> document cache, like EmbeddedResource blocks
            rendered = _render_mcp_resource_block(SimpleNamespace(type="resource", resource=block), server_name)
            parts.append(rendered or f"[binary data, {len(block.blob)} bytes]")
    return {"result": "\n".join(parts)}


def _render_prompt_list(all_prompts, server_name: str) -> dict:
    prompts = []
    for p in all_prompts:
        entry = _pick(p, ("name", "name"), ("description", "description", True))
        if getattr(p, "arguments", None):
            entry["arguments"] = [{"name": a.name, **_pick(a, ("description", "description", True), ("required", "required"))}
                                  for a in p.arguments]
        prompts.append(entry)
    return {"prompts": prompts}


def _render_get_prompt(result, server_name: str) -> dict:
    messages = []
    for msg in getattr(result, "messages", []):
        entry = _pick(msg, ("role", "role"))
        if hasattr(msg, "content"):
            entry["content"] = strip_unicode_tags(msg.content.text if hasattr(msg.content, "text") else str(msg.content))
        messages.append(entry)
    return {"messages": messages, **_pick(result, ("description", "description", True))}


_make_list_resources_handler = _make_utility_handler(
    "resources/list", "list_resources",
    lambda session, args, sn: _core._paginate_full_list(session.list_resources, "resources", sn), _render_resource_list)
_make_read_resource_handler = _make_utility_handler(
    "resources/read", "read_resource",
    lambda session, args, sn: session.read_resource(args["uri"]), _render_read_resource, required="uri")
_make_list_prompts_handler = _make_utility_handler(
    "prompts/list", "list_prompts",
    lambda session, args, sn: _core._paginate_full_list(session.list_prompts, "prompts", sn), _render_prompt_list)
_make_get_prompt_handler = _make_utility_handler(
    "prompts/get", "get_prompt",
    lambda session, args, sn: session.get_prompt(args["name"], arguments=args.get("arguments", {})),
    _render_get_prompt, required="name")


def _make_check_fn(server_name: str):
    """Connection-alive check; lazy (schema-cache registered) servers count as available."""
    def _check() -> bool:
        with _core._lock:
            server = _core._servers.get(server_name)
            return ((server is not None and (server.session is not None or server._is_recycled_stdio()))
                    or server_name in _core._lazy_server_configs)
    return _check
