"""Session adapter for the ``claude_code`` runtime (#25267).

Drives the official ``claude`` CLI as a long-lived subprocess in
bidirectional ``stream-json`` mode so a Hermes turn can run on a Claude
*subscription* (a long-lived ``claude setup-token`` credential) while still
making real tool calls:

* Hermes' own tool surface — terminal / read_file / write_file / patch /
  search_files / process plus web_search / web_extract / browser_* / vision /
  image_generate / skills / TTS / kanban_* — is exposed to the CLI over stdio
  MCP via ``agent.transports.hermes_tools_mcp_server`` (the ``claude-code``
  profile of the server the ``codex_app_server`` runtime uses). The model
  sees them as ``mcp__hermes-tools__<tool>``. Every shell command and file
  write therefore goes through Hermes' own policy: ``check_all_command_guards``
  (Tirith + dangerous-command detection + approval mode), file-path policy,
  env sanitisation and ``pre_tool_call`` hooks.
* Claude Code's NATIVE OS-level tools (Bash, Read, Write, Edit, Glob, Grep,
  WebFetch, Task, ...) are passed to ``--disallowedTools`` in every mode
  unless the operator sets ``claude_code.native_tools: true``. They bypass all
  of the Hermes policy above and can read the child's environment, so they
  are never pre-authorised by default (see :data:`NATIVE_OS_TOOLS`).

Why a subprocess and not the Anthropic Messages transport: subscription
credentials must only ever be touched by the official CLI. Nothing here reads,
extracts, or forwards a token — ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``
are deliberately *stripped* from the child environment so an inherited API
key can never silently bill an API account instead of the subscription.

Lifecycle::

    session = ClaudeCodeSession(cwd="/home/x/proj", model="sonnet")
    session.ensure_started()                      # spawn + warm-up + system/init
    result = session.run_turn(user_input="hello") # blocks until the `result` event
    # result.final_text          -> assistant text returned to the caller
    # result.projected_messages  -> {role, content, tool_calls, tool_call_id} dicts
    # result.tool_iterations     -> how many tool calls completed (skill nudge counter)
    # result.interrupted         -> True if request_interrupt() fired mid-turn
    session.close()

One process per Hermes session, NOT one per turn: the conversation lives in
the CLI's context and every turn after the first hits the prompt cache
(measured in the reference Swift implementation: ~4.7s to first token cold vs
~0.8s warm).

Protocol facts this module is built on (Claude Code 2.1.x, verified live):

* The CLI emits **nothing** — not even ``system/init`` — until it has received
  a first user message on stdin. Waiting for init before sending deadlocks
  both sides forever, hence the warm-up turn in :meth:`ensure_started`.
* Events are newline-delimited JSON objects on stdout: ``system`` (subtype
  ``init``), ``stream_event`` (raw Anthropic SSE frames, only with
  ``--include-partial-messages``), ``assistant``, ``user`` (tool results echoed
  back), ``rate_limit_event`` and the terminal ``result``.
* MCP tools are *deferred*: the model may call the CLI's own ``ToolSearch``
  tool first to load ``mcp__hermes-tools__*`` definitions. That is normal and
  is projected like any other native tool call.
* ``--append-system-prompt`` is fixed for the life of the process, so Hermes'
  system prompt is captured at spawn time. A changed prompt requires a
  respawn (see ``system_prompt`` / :meth:`needs_respawn`).

Threading model: single caller (AIAgent's conversation thread). A reader
thread owns stdout and pushes parsed lines onto a queue; ``run_turn`` polls
that queue synchronously so it behaves like the existing chat_completions
loop. Process identity is checked wherever a background thread reports exit,
so a subprocess we deliberately killed can never clear the state of its
healthy replacement.
"""

from __future__ import annotations

import base64
import binascii
import getpass
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

#: MCP server name registered in ``--mcp-config``. Tool names on the wire are
#: ``mcp__<server>__<tool>``; keep this stable — it is part of the allowlist.
HERMES_TOOLS_MCP_SERVER_NAME = "hermes-tools"
_MCP_TOOL_PREFIX = f"mcp__{HERMES_TOOLS_MCP_SERVER_NAME}__"

#: Every Claude Code built-in that touches the OS or the network: shell and
#: code execution, filesystem reads *and* writes, web access, subagents,
#: skills and slash commands (each of which can run the others). These run
#: inside the CLI and never see Hermes' command/file policy, its env
#: sanitisation or its ``pre_tool_call`` hooks; native ``Bash`` can also print
#: the child's environment (``$CLAUDE_CODE_OAUTH_TOKEN``). In every normal
#: mode they are passed to ``--disallowedTools``; the equivalent capability is
#: provided by the Hermes tools over MCP. ``Read`` / ``Glob`` / ``Grep`` are
#: included on purpose: they accept absolute paths outside the workspace
#: (``~/.hermes/.env``, ``~/.claude/.credentials.json``) with no path policy,
#: while Hermes' ``read_file`` / ``search_files`` are exposed instead.
#: Audited against ``claude --help`` 2.1.251 plus the documented built-in
#: set; unknown names are harmless to deny.
NATIVE_OS_TOOLS: tuple[str, ...] = (
    # shell / code execution and its background-process helpers
    "Bash", "PowerShell", "REPL", "BashOutput", "KillShell", "KillBash",
    "TaskOutput", "TaskStop", "Monitor",
    # filesystem
    "Read", "Write", "Edit", "MultiEdit", "NotebookEdit", "NotebookRead",
    "Glob", "Grep", "LS",
    # network
    "WebFetch", "WebSearch",
    # anything that can invoke the above indirectly
    "Task", "Agent", "Skill", "SlashCommand",
    # resources of other MCP servers (none are configured, belt-and-braces)
    "ListMcpResourcesTool", "ReadMcpResourceTool",
)
#: Built-ins that never leave the model's own context; safe to pre-approve.
#: Also the ONLY built-ins the child is started with (``--tools``) when the
#: boundary is on: ``--disallowedTools`` is a denylist and the CLI keeps
#: adding built-ins (measured on 2.1.251 with the full NATIVE_OS_TOOLS list:
#: Workflow, EnterWorktree/ExitWorktree, CronCreate/CronDelete/CronList,
#: ScheduleWakeup, SendMessage, TaskCreate/…, DesignSync, ListAgents,
#: ReportFindings still loaded). ``--tools`` is the allowlist counterpart:
#: verified live to leave only these while MCP servers stay connected.
_INERT_NATIVE_TOOLS: tuple[str, ...] = ("TodoWrite", "ToolSearch")

#: Hermes MCP tools that mutate state or run commands. In
#: ``approval-required`` mode they are NOT pre-approved: each call comes back
#: over ``can_use_tool`` and is routed to Hermes' approval prompt (denied
#: with a message when no approver is wired), exactly like native Bash was.
GATED_HERMES_TOOLS: tuple[str, ...] = ("terminal", "write_file", "patch", "process")

#: Native tools that are pre-approved in ``auto`` mode **only when the
#: operator opted in with** ``claude_code.native_tools: true``. This is the
#: pre-#98533 posture: the shell runs unprompted inside the CLI and the ONLY
#: gate is the deny list in the Hermes-owned ``settings.json`` (see
#: :data:`DEFAULT_DENY_RULES`). Anything NOT on this allowlist still goes
#: through the ``can_use_tool`` approval bridge below.
_AUTO_MODE_ALLOWED_TOOLS: tuple[str, ...] = (
    "Bash", "Read", "Write", "Edit", "MultiEdit", "NotebookEdit",
    "Glob", "Grep", "LS", "WebSearch", "WebFetch", "TodoWrite", "Task",
    "ToolSearch",
)
#: Read-only native tools pre-approved with ``native_tools: true`` when the
#: user asked for approval on everything mutating.
_APPROVAL_MODE_ALLOWED_TOOLS: tuple[str, ...] = (
    "Read", "Glob", "Grep", "LS", "WebSearch", "WebFetch", "TodoWrite",
    "ToolSearch",
)

# Hermes ``tools.terminal.security_mode`` -> (claude --permission-mode,
# pre-approved native tools). ``bypassPermissions`` is ONLY reachable through
# the user's own ``unrestricted`` / ``yolo`` choice — never as a default —
# because in ``-p`` mode it disables Claude Code's permission layer entirely.
#
# ``approval-required`` maps to ``default`` mode with only read-only tools
# pre-approved. Everything else (Bash, Write, Edit, ...) is GATED, not
# removed: the CLI is started with ``--permission-prompt-tool stdio`` and asks
# Hermes over the stream-json control channel (``can_use_tool``) before each
# non-allowlisted call. The request is routed through Hermes' own approval
# callback (the same one the codex runtime uses); with no interactive
# approver wired (gateway/cron) it is denied with a message the model sees.
# :meth:`ClaudeCodeSession.ensure_started` logs + emits a one-time ``status``
# event in that no-approver case so the degradation is never silent.
_HERMES_TO_CLAUDE_PERMISSION: dict[str, tuple[str, tuple[str, ...]]] = {
    "auto": ("acceptEdits", _AUTO_MODE_ALLOWED_TOOLS),
    "approval-required": ("default", _APPROVAL_MODE_ALLOWED_TOOLS),
    "unrestricted": ("bypassPermissions", ()),
    # Backstop alias used by some skills/tests.
    "yolo": ("bypassPermissions", ()),
}
# Unknown modes fail CLOSED to the most restrictive non-bypass mapping.
_FAIL_CLOSED_SECURITY_MODE = "approval-required"

#: Message shown once per session when gated tools will be denied because
#: no interactive approver is available in this context.
APPROVAL_MODE_NOTICE = (
    "approval-required: no interactive approver in this context, so shell "
    "and file-writing tools will be denied (set tools.terminal.security_mode: "
    "auto to run them under the Hermes-owned deny list)"
)
#: Denial text the model sees when a gated tool is refused.
_DENY_NO_APPROVER = (
    "Denied by Hermes: approval-required mode and no interactive approver is "
    "available for this session."
)
_DENY_TIMEOUT = "Denied by Hermes: the approval prompt timed out without a response."
_DENY_USER = "Denied by the Hermes user."
#: Approval choices from tools.approval.prompt_dangerous_approval that allow
#: exactly this one call. `session`/`always` are also honoured as allow but
#: are not persisted into Claude Code rules — Hermes' allowlist owns that.
_ALLOW_CHOICES = frozenset({"once", "session", "always"})
#: Name of the per-config-dir file mapping Hermes session keys to the CLI
#: session id actually in use (needed when a --resume fallback rotated it).
SESSION_MAP_FILENAME = "hermes-sessions.json"
SESSION_MAP_LOCK_FILENAME = ".hermes-sessions.lock"
#: The CLI's own wording when a session id cannot be used (both strings are
#: present in the 2.1.251 binary). Only these justify rotating the id — any
#: other warm-up failure (auth, rate limit, MCP crash) must keep it, or a
#: later healthy start would silently lose the conversation.
_SESSION_REJECTION_MARKERS = (
    "no conversation found with session id",
    "is already in use",
)


def is_session_id_rejection(error_text: Optional[str]) -> bool:
    lowered = (error_text or "").lower()
    return any(marker in lowered for marker in _SESSION_REJECTION_MARKERS)

#: Default ``permissions.deny`` rules written into the Hermes-owned
#: ``settings.json`` the first time the runtime starts. Mirrors the deny list
#: Moe ships in ``~/.moe/.claude/settings.json``. Users extend/replace it via
#: ``claude_code.deny`` in config.yaml (applied only when the file is first
#: written) or by editing the file directly — Hermes never overwrites it.
DEFAULT_DENY_RULES: tuple[str, ...] = (
    "Bash(rm -rf *)",
    "Bash(sudo *)",
    "Bash(git push *)",
    "Bash(osascript -e 'tell application \"Messages\"*)",
    "mcp__claude_ai_Gmail__send_message",
    "mcp__claude_ai_Gmail__reply",
    "mcp__claude_ai_Gmail__forward",
    "mcp__claude_ai_Gmail__trash_message",
    "mcp__claude_ai_Gmail__trash_thread",
    "mcp__claude_ai_Google_Drive__share_file",
    "mcp__claude_ai_Google_Drive__trash_file",
)
#: JSON key carrying the provenance note in the managed settings file.
SETTINGS_MARKER_KEY = "_hermes"
SETTINGS_MARKER_TEXT = (
    "Written once by Hermes (claude_code runtime). Edit freely — Hermes "
    "never overwrites this file. Delete it to regenerate the defaults."
)

#: Default env var carrying the long-lived Claude Code credential.
DEFAULT_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

#: ``MCP_TIMEOUT`` / ``MCP_TOOL_TIMEOUT`` for the CLI child (milliseconds).
#: Startup covers importing Hermes' registry plus every cached MCP manifest.
#: The per-call ceiling has to outlast a synchronous subagent fan-out AND stay
#: strictly above ``hermes_tool_bridge.DEFAULT_TIMEOUT_SECONDS``, so that a
#: call that runs too long comes back as the bridge's readable tool error
#: rather than the CLI's opaque MCP abort (which would leave the dispatch
#: running with nobody to answer). Pinned by a test, not by hope.
_MCP_STARTUP_TIMEOUT_MS = 180_000
_MCP_TOOL_TIMEOUT_MS = 1_800_000

#: How much of a turn's silence budget work on this side may consume before
#: the turn is declared dead anyway. Comfortably outlives the bridge's own
#: client timeout, so a healthy long call ends by answering rather than by
#: retiring the session; a wedged dispatcher still ends the turn.
#:
#: It is a per-TURN budget while the bridge's own timeout is per-CALL, so two
#: sequential maximal bridged calls in one turn exhaust it and the turn is
#: retired while the second dispatch is still running. That is the cap doing
#: its job, not a hole — but it is a cap, not a guarantee.
#:
#: A backstop, not a routine path. Measured on Claude Code 2.1.252, the CLI
#: keeps emitting stream events while an MCP tool call is outstanding: a
#: delegation whose subagent ran `sleep 200` completed in 4m27s under a 120 s
#: silence_timeout without this ever firing. But that streaming is an
#: undocumented detail of a tool that ships weekly, and the failure it would
#: cause — retiring a warm session, and abandoning subagents, in the middle of
#: their own successful work — is bad enough to be worth one cheap guard.
_BRIDGE_TURN_GRACE_SECONDS = 2_100.0

_VALID_PERMISSION_MODES = frozenset(
    {"default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"}
)

# Text the warm-up turn asks for. Kept tiny: it costs one model round-trip
# and lands in the CLI's conversation context for the life of the process.
_WARMUP_PROMPT = "Reply with only the word: ready"

# After an interrupt control request, how long to wait for the CLI's own
# `result` before killing the process.
_INTERRUPT_GRACE_SECONDS = 5.0

# How many tailing stderr lines to attach to a user-facing error.
_STDERR_TAIL_LINES = 12
_STDERR_KEEP_LINES = 200

# Queue sentinels emitted by the reader thread. Never valid stream-json.
_EOF = object()


@dataclass
class TurnResult:
    """Result of one user -> assistant -> tools turn through ``claude -p``.

    Same field set as ``codex_app_server_session.TurnResult`` so
    ``agent/claude_code_runtime.py`` can return the exact dict shape the
    conversation loop and every gateway already understand.
    """

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    session_id: Optional[str] = None
    token_usage_last: Optional[dict[str, Any]] = None
    token_usage_total: Optional[dict[str, Any]] = None
    model_context_window: Optional[int] = None
    compacted: bool = False
    # The CLI exited, stopped answering, or reported an auth failure: the
    # caller should drop the session so the next turn respawns from scratch.
    should_retire: bool = False


def resolve_permission(
    security_mode: Optional[str], *, native_tools: bool = False,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Map a Hermes terminal security mode to
    ``(permission_mode, allowed_native_tools, disallowed_native_tools)``.

    Invariant (#98533): unless ``native_tools`` is True, no mode — not even
    ``unrestricted`` — pre-authorises a native OS-level tool. ``unrestricted``
    / ``yolo`` mean "don't prompt me", not "swap Hermes' shell for one that
    skips its hardline blocks and env sanitisation"; only the explicit
    ``claude_code.native_tools: true`` re-opens that boundary.
    """
    key = str(security_mode or "auto").strip().lower() or "auto"
    if key not in _HERMES_TO_CLAUDE_PERMISSION:
        logger.warning(
            "unknown tools.terminal.security_mode %r; failing closed to %s",
            security_mode, _FAIL_CLOSED_SECURITY_MODE,
        )
        key = _FAIL_CLOSED_SECURITY_MODE
    mode, legacy_allowed = _HERMES_TO_CLAUDE_PERMISSION[key]
    if native_tools:
        return mode, legacy_allowed, ()
    return mode, _INERT_NATIVE_TOOLS, NATIVE_OS_TOOLS


def hermes_mcp_allow_rules(permission_mode: str) -> list[str]:
    """``--allowedTools`` entries for the hermes-tools MCP server.

    * ``acceptEdits`` (auto): the whole server — every tool is already gated
      by Hermes' own policy (command guards, file policy, hooks).
    * ``default`` (approval-required): every tool except
      :data:`GATED_HERMES_TOOLS`, which fall through to ``can_use_tool`` and
      Hermes' approval prompt.

    The bridged agent-loop tools are named here too. They are dispatched by
    the parent process through ``_invoke_tool`` — the same hooks and the same
    policy as any Hermes turn — so routing them through the CLI's approval
    prompt as well would gate them twice, and in a gateway context (no
    approver) would deny delegation outright.
    """
    if permission_mode != "default":
        return [f"mcp__{HERMES_TOOLS_MCP_SERVER_NAME}"]
    from agent.transports.hermes_tool_bridge import BRIDGED_TOOLS
    from agent.transports.hermes_tools_mcp_server import exposed_tools_for_profile

    gated = set(GATED_HERMES_TOOLS)
    names = tuple(exposed_tools_for_profile("claude-code")) + BRIDGED_TOOLS
    return [
        f"{_MCP_TOOL_PREFIX}{name}"
        for name in dict.fromkeys(names)
        if name not in gated
    ]


def hermes_tool_name(wire_name: str) -> str:
    """``mcp__hermes-tools__web_search`` -> ``web_search``; native names pass through."""
    if isinstance(wire_name, str) and wire_name.startswith(_MCP_TOOL_PREFIX):
        return wire_name[len(_MCP_TOOL_PREFIX):] or wire_name
    return wire_name or "tool"


def claude_code_home() -> str:
    """``$HERMES_HOME/claude-code`` — the Hermes-owned Claude Code config dir.

    Everything the CLI persists (settings, ``.claude.json``, transcripts,
    auto-memory if it were enabled, plugins) lives here instead of in the
    user's ``~/.claude``. Hermes turns must never fire the user's hooks, load
    their plugins, or write into their personal memory/transcripts.
    """
    try:
        from hermes_constants import get_hermes_home

        root = str(get_hermes_home())
    except Exception:  # pragma: no cover - defensive
        root = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return os.path.join(root, "claude-code")


def default_workspace_dir() -> str:
    """Dedicated cwd for the child so transcripts/CLAUDE.md discovery never
    key off wherever the Hermes process happened to be started."""
    return os.path.join(claude_code_home(), "workspace")


def _wire_names(rules: list[str]) -> list[str]:
    """Deny rules as the CHILD sees the tool, not as Hermes names it.

    Every tool this runtime serves arrives over the hermes-tools MCP server,
    so the child's name for it is ``mcp__hermes-tools__<name>`` — including
    when ``<name>`` is itself an MCP tool, giving the double prefix
    ``mcp__hermes-tools__mcp__cloudflare__...``. Rules written as plain
    ``mcp__claude_ai_Gmail__send_message`` therefore matched nothing: the deny
    list was decorative, and had been since the tools moved behind the bridge.

    ``Bash(...)`` rules are left alone — those name the CLI's own native tool,
    which is a different namespace and is disallowed outright anyway.
    """
    out: list[str] = []
    for rule in rules:
        if rule.startswith("mcp__") and not rule.startswith(_MCP_TOOL_PREFIX):
            out.append(_MCP_TOOL_PREFIX + rule)
        else:
            out.append(rule)
    return out


def ensure_settings_file(
    path: str, deny_rules: Optional[list[str]] = None
) -> str:
    """Write the Hermes-owned Claude Code ``settings.json`` if it is absent.

    Existing files are never touched (the marker key records provenance, the
    same pattern the Gemini policy file uses) so user edits survive.
    """
    if os.path.exists(path):
        return path
    rules = _wire_names(list(deny_rules) if deny_rules else list(DEFAULT_DENY_RULES))
    payload = {
        SETTINGS_MARKER_KEY: SETTINGS_MARKER_TEXT,
        "permissions": {"deny": rules},
    }
    directory = os.path.dirname(path) or "."
    _makedirs_or_explain(directory)
    fd, tmp = _mkstemp_or_explain(prefix=".settings-", suffix=".json", directory=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _makedirs_or_explain(directory: str) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError as exc:
        raise RuntimeError(
            f"Claude Code runtime cannot create {directory}: permission denied. "
            "Fix the permissions on that directory, or set claude_code.cwd / "
            "HERMES_HOME to a writable location."
        ) from exc


def load_session_map(config_dir: str) -> dict[str, str]:
    path = os.path.join(config_dir, SESSION_MAP_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _mkstemp_or_explain(*, prefix: str, suffix: str, directory: str) -> tuple[int, str]:
    try:
        return tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    except PermissionError as exc:
        raise RuntimeError(
            f"Claude Code runtime cannot write into {directory}: permission "
            "denied. Fix the permissions on that directory, or set "
            "claude_code.cwd / HERMES_HOME to a writable location."
        ) from exc


def save_session_mapping(config_dir: str, session_key: str, claude_session_id: str) -> None:
    """Persist ``session_key -> claude_session_id`` (atomic, 0600).

    The read-modify-write is serialised with ``flock`` on a sidecar lock file
    so two sessions rotating at the same time cannot drop each other's entry.
    """
    _makedirs_or_explain(config_dir)
    lock_path = os.path.join(config_dir, SESSION_MAP_LOCK_FILENAME)
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        try:
            import fcntl

            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - Windows / odd FS
            pass
        data = load_session_map(config_dir)
        data[session_key] = claude_session_id
        fd, tmp = _mkstemp_or_explain(prefix=".sessions-", suffix=".json", directory=config_dir)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, os.path.join(config_dir, SESSION_MAP_FILENAME))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def resolve_oauth_token(token_env: str = DEFAULT_OAUTH_TOKEN_ENV) -> str:
    """Return the long-lived Claude Code credential, or raise with a fix.

    The runtime uses its own ``CLAUDE_CONFIG_DIR``, which also means its own
    credential store — the user's interactive keychain login is deliberately
    NOT shared. Two reasons: (1) isolation — Hermes must not be able to log
    the user out; (2) correctness — several long-lived ``claude`` processes
    sharing ONE interactive login rotate the OAuth refresh token underneath
    each other, and every refresh by one child logs the others out ("OAuth
    session expired and could not be refreshed"). One token owned by Hermes
    ends that race. Hermes never reads the value except to place it in the
    child's environment.
    """
    token = (os.environ.get(token_env) or "").strip()
    if not token:
        raise RuntimeError(
            f"Claude Code runtime needs a long-lived token in ${token_env}. "
            "Run `claude setup-token` (uses your Claude subscription), put the "
            f"result in {token_env} in ~/.hermes/.env, and retry. "
            "(Change the variable name with claude_code.oauth_token_env.)"
        )
    return token


def build_child_env(
    base_env: Optional[dict[str, str]] = None,
    *,
    config_dir: Optional[str] = None,
    token_env: str = DEFAULT_OAUTH_TOKEN_ENV,
    oauth_token: Optional[str] = None,
) -> dict[str, str]:
    """Environment for the ``claude`` child.

    * Starts from Hermes' sanitized subprocess env with Tier-1 gateway/infra
      secrets AND Tier-2 provider/tool credentials stripped
      (``hermes_subprocess_env(inherit_credentials=False)``). The ``claude``
      process itself needs none of them. The hermes-tools MCP server —
      spawned *by* the CLI — does need tool API keys (Firecrawl, browser,
      TTS, ...); those travel in the ``env`` block of the ``--mcp-config``
      file instead (see :func:`write_mcp_config`), which the CLI merges over
      the inherited environment when it spawns the server.
    * ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` are removed so the CLI
      can only authenticate with the subscription token. This is the whole
      point of the runtime, not an optimisation.
    * ``CLAUDE_CODE_OAUTH_TOKEN`` carries the Hermes-owned setup-token (see
      :func:`resolve_oauth_token` for why it is not the keychain login).
    * ``CLAUDE_CONFIG_DIR`` points at :func:`claude_code_home` so settings,
      transcripts, ``.claude.json`` and plugins are Hermes-owned;
      ``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`` keeps the CLI from writing its
      own memory files at all (Hermes owns memory).
    * ``USER`` / ``LOGNAME`` / ``HOME`` are guaranteed present: a child
      spawned from a GUI app or a stripped service env otherwise fails to
      locate its own state even when everything is configured.
    """
    if base_env is None:
        try:
            from tools.environments.local import hermes_subprocess_env

            env = hermes_subprocess_env(inherit_credentials=False)
        except Exception:  # pragma: no cover - import-time fallback
            env = dict(os.environ)
    else:
        env = dict(base_env)
    # Same Tier-2 strip for a caller-supplied base env (tests, embedders).
    for key in _provider_credential_keys():
        env.pop(key, None)
    # Anything that could redirect the CLI away from the subscription (API
    # key, alt endpoint, Bedrock/Vertex routing) or pin a model behind
    # Hermes' back is stripped.
    for key in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
    ):
        env.pop(key, None)
    token = oauth_token if oauth_token is not None else os.environ.get(token_env)
    if token:
        # SECURITY NOTE (measured live on Claude Code 2.1.251): this variable
        # IS visible to anything the CLI itself runs — `env` inside a native
        # Bash tool prints it, and the CLI has no per-tool env sanitisation
        # setting. It cannot be stripped: the CLI reads it to authenticate.
        # Mitigations are (a) native Bash/Read/... are in --disallowedTools
        # unless claude_code.native_tools is set, so by default nothing the
        # model drives runs inside the CLI process, (b) the hermes-tools MCP
        # server scrubs it from its own environment at startup (it inherits
        # the CLI's env) so Hermes' terminal never sees it either, (c) it is
        # a dedicated setup-token revocable without touching the user's own
        # login, (d) the deny list in settings.json, (e) Hermes' output
        # redaction. With native_tools: true, treat anything the child can
        # run as able to read this token.
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    env["CLAUDE_CONFIG_DIR"] = config_dir or claude_code_home()
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    # The hermes-tools server is not a fast local shim: it imports Hermes'
    # whole tool registry and every cached MCP manifest at startup, and one of
    # its calls — a bridged ``delegate_task`` — runs subagents to completion
    # before it answers. The CLI's defaults (30 s to start, 60 s per call)
    # would kill both. ``setdefault`` so an operator can still say otherwise.
    env.setdefault("MCP_TIMEOUT", str(_MCP_STARTUP_TIMEOUT_MS))
    env.setdefault("MCP_TOOL_TIMEOUT", str(_MCP_TOOL_TIMEOUT_MS))
    home = env.get("HOME") or os.path.expanduser("~")
    env["HOME"] = home
    user = env.get("USER") or env.get("LOGNAME")
    if not user:
        try:
            user = getpass.getuser()
        except Exception:
            user = ""
    if user:
        env.setdefault("USER", user)
        env.setdefault("LOGNAME", user)
    # The CLI is itself an agent; make sure it can find the tool binaries a
    # login shell would. Cheap to prepend, harmless if already present.
    extra_path = ":".join(
        p for p in (
            os.path.join(home, ".local", "bin"),
            "/opt/homebrew/bin",
            "/usr/local/bin",
        )
        if p not in (env.get("PATH") or "").split(":")
    )
    if extra_path:
        env["PATH"] = f"{extra_path}:{env.get('PATH') or '/usr/bin:/bin'}"
    return env


def _provider_credential_keys() -> frozenset:
    """Hermes' Tier-2 provider/tool credential names (single source of truth
    in ``tools.environments.local``). Empty when that module is unavailable."""
    try:
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

        return frozenset(_HERMES_PROVIDER_ENV_BLOCKLIST)
    except Exception:  # pragma: no cover - import-time fallback
        return frozenset()


#: Never forwarded to the MCP server through the config file, whatever the
#: blocklist says: the CLI's own credential (the server scrubs it instead).
_NEVER_FORWARD_TO_MCP: frozenset = frozenset({"CLAUDE_CODE_OAUTH_TOKEN"})


def mcp_server_credentials(source: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Provider/tool credentials the hermes-tools MCP server needs.

    Exactly the keys :func:`build_child_env` strips from the ``claude`` child
    (Tier-2 blocklist), taken from ``source`` (default ``os.environ``), so
    nothing is lost relative to plain inheritance and nothing extra leaks.
    """
    src = os.environ if source is None else source
    return {
        key: src[key]
        for key in sorted(_provider_credential_keys())
        if key in src and src[key] and key not in _NEVER_FORWARD_TO_MCP
    }


def resume_transcript_exists(config_dir: str, session_id: str) -> bool:
    """True when the CLI already has a transcript for ``session_id`` under
    ``config_dir`` (``projects/<cwd-slug>/<id>.jsonl``), i.e. ``--resume``
    can restore it."""
    root = os.path.join(config_dir, "projects")
    try:
        for entry in os.scandir(root):
            if entry.is_dir() and os.path.exists(
                os.path.join(entry.path, f"{session_id}.jsonl")
            ):
                return True
    except OSError:
        pass
    return False


_MCP_CONFIG_PREFIX = "hermes-claude-mcp-"
#: path -> fd holding an ``flock`` for the life of the owning session. The
#: file carries provider credentials, so a session that dies without
#: ``close()`` (gateway killed, crash) must not leave it behind: the next
#: start sweeps any config file whose lock is free (see
#: :func:`sweep_stale_mcp_configs`).
_MCP_CONFIG_LOCKS: dict[str, int] = {}


def _flock(fd: int, *, blocking: bool) -> bool:
    """True when the lock was taken; False when held elsewhere or unsupported."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return True
    except OSError:
        return False


def sweep_stale_mcp_configs(directory: str) -> list[str]:
    """Delete ``hermes-claude-mcp-*.json`` files in ``directory`` that no
    live session holds a lock on. Returns the removed paths."""
    removed: list[str] = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return removed
    for entry in entries:
        name = entry.name
        if not (name.startswith(_MCP_CONFIG_PREFIX) and name.endswith(".json")):
            continue
        if entry.path in _MCP_CONFIG_LOCKS:
            continue
        try:
            fd = os.open(entry.path, os.O_RDONLY)
        except OSError:
            continue
        try:
            if _flock(fd, blocking=False):
                os.unlink(entry.path)
                removed.append(entry.path)
        except OSError:
            pass
        finally:
            os.close(fd)
    if removed:
        logger.warning(
            "claude-code: removed %d stale mcp-config file(s) left by a session "
            "that did not close cleanly", len(removed),
        )
    return removed


def release_mcp_config(path: str) -> None:
    """Delete the config file written by :func:`write_mcp_config` and drop its lock."""
    fd = _MCP_CONFIG_LOCKS.pop(path, None)
    try:
        os.unlink(path)
    except OSError:
        pass
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def write_mcp_config(
    *,
    python_executable: Optional[str] = None,
    project_root: Optional[str] = None,
    directory: Optional[str] = None,
    credential_env: Optional[dict[str, str]] = None,
    bridge_socket: Optional[str] = None,
    bridge_token: Optional[str] = None,
    bridge_tools: Optional["tuple[str, ...] | list[str]"] = None,
) -> str:
    """Write the ``--mcp-config`` JSON that launches ``hermes_tools_mcp_server``.

    The server runs on the *same* interpreter as this process so it sees the
    same installed tools and config, in its ``claude-code`` profile (adds
    terminal / read_file / write_file / patch / search_files / process).

    Credentials: the CLI spawns the server with its own environment merged
    with this file's ``env`` block. Because :func:`build_child_env` strips
    provider/tool credentials from the CLI, the ones the server needs are
    written here (:func:`mcp_server_credentials`; ``credential_env``
    overrides). The file is created 0600 inside the Hermes-owned config dir,
    ``flock``-ed for the life of the session and removed when it closes; a
    file whose owner died without ``close()`` is swept at the next start. The block also blanks
    ``CLAUDE_CODE_OAUTH_TOKEN`` and lists it in ``HERMES_MCP_SCRUB_ENV`` so
    the server drops the CLI credential it would otherwise inherit.

    ``bridge_socket`` / ``bridge_token`` / ``bridge_tools`` address the
    :mod:`agent.transports.hermes_tool_bridge` server running back in this
    process, and are what let the child call ``delegate_task`` / ``memory`` /
    ``session_search`` / ``todo`` at all. They are here rather than in the
    CLI's environment because the CLI has no use for them — NOT as a secret:
    Hermes' ``terminal`` runs inside this server, so the model can read this
    block's env, and this file is readable by anything running as this user.
    What actually bounds the bridge is its server-side allowlist and the
    dispatcher's re-check against the agent's own tool surface.
    """
    py = python_executable or sys.executable
    root = project_root or os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    from agent.transports.hermes_tools_mcp_server import (
        CLAUDE_CODE_PROFILE, PROFILE_ENV, SCRUB_ENV,
    )

    server_env: dict[str, str] = {
        "PYTHONPATH": root,
        "HERMES_QUIET": "1",
        "HERMES_REDACT_SECRETS": "true",
        PROFILE_ENV: CLAUDE_CODE_PROFILE,
        SCRUB_ENV: ",".join(sorted(_NEVER_FORWARD_TO_MCP)),
    }
    for key in _NEVER_FORWARD_TO_MCP:
        server_env[key] = ""
    for passthrough in ("HERMES_HOME", "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB"):
        value = os.environ.get(passthrough)
        if value:
            server_env[passthrough] = value
    creds = mcp_server_credentials() if credential_env is None else dict(credential_env)
    for key, value in creds.items():
        if key not in _NEVER_FORWARD_TO_MCP:
            server_env[key] = value
    if bridge_socket and bridge_token:
        from agent.transports.hermes_tool_bridge import (
            BRIDGE_SOCKET_ENV, BRIDGE_TIMEOUT_ENV, BRIDGE_TOKEN_ENV,
            BRIDGE_TOOLS_ENV, BRIDGED_TOOLS, DEFAULT_TIMEOUT_SECONDS,
        )

        server_env[BRIDGE_SOCKET_ENV] = bridge_socket
        server_env[BRIDGE_TOKEN_ENV] = bridge_token
        server_env[BRIDGE_TOOLS_ENV] = ",".join(
            BRIDGED_TOOLS if bridge_tools is None else bridge_tools
        )
        # Written explicitly rather than left to the client's default so the
        # two ceilings that have to disagree — this one and MCP_TOOL_TIMEOUT
        # in the CLI's own environment — are visible in the same artefact.
        server_env[BRIDGE_TIMEOUT_ENV] = str(int(DEFAULT_TIMEOUT_SECONDS))
    payload = {
        "mcpServers": {
            HERMES_TOOLS_MCP_SERVER_NAME: {
                "type": "stdio",
                "command": py,
                "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                "cwd": root,
                "env": server_env,
            }
        }
    }
    if directory:
        _makedirs_or_explain(directory)
    fd, path = _mkstemp_or_explain(
        prefix=_MCP_CONFIG_PREFIX, suffix=".json", directory=directory or tempfile.gettempdir()
    )
    # Lock BEFORE the content lands so a concurrent sweep can never see an
    # unlocked file that holds credentials. The dup shares the open file
    # description, so the lock survives the fdopen() close below.
    lock_fd = os.dup(fd)
    if _flock(lock_fd, blocking=True):
        _MCP_CONFIG_LOCKS[path] = lock_fd
    else:  # pragma: no cover - no flock on this platform
        os.close(lock_fd)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _content_text(content: Any) -> str:
    """Flatten an Anthropic content value (str or block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif block.get("type") == "tool_result":
                    parts.append(_content_text(block.get("content")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _format_tool_args(args: Any) -> str:
    try:
        return json.dumps(args if args is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"arguments": str(args)})


def _tool_preview(name: str, args: Any) -> Optional[str]:
    """Short human-readable preview for the tool.started bubble."""
    if not isinstance(args, dict) or not args:
        return None
    if name == "Bash" or hermes_tool_name(name) == "terminal":
        desc = args.get("description") or args.get("command") or ""
        return str(desc)[:120] or None
    for key in ("file_path", "path", "pattern", "query", "url", "prompt"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value[:120]
    try:
        return json.dumps(args, ensure_ascii=False)[:120]
    except (TypeError, ValueError):
        return None


class ClaudeCodeSession:
    """One long-lived ``claude -p`` process per Hermes session.

    The child is a *Hermes-owned* Claude Code, not the user's: its own
    ``CLAUDE_CONFIG_DIR`` (settings, transcripts, ``.claude.json``), its own
    credential (a setup-token in ``$CLAUDE_CODE_OAUTH_TOKEN``), a dedicated
    workspace cwd, ``--setting-sources ""`` (no ``~/.claude`` / project /
    local settings, hence no user hooks or plugins), ``--settings`` pointing
    at the Hermes-managed deny list, ``--strict-mcp-config`` (only the
    hermes-tools server), ``--disallowedTools`` covering every native
    OS-level tool (unless ``native_tools``) and auto-memory disabled.

    Not thread-safe from the caller's perspective — one caller drives it at a
    time (AIAgent's conversation thread). :meth:`request_interrupt` is the one
    method that may be called from another thread.
    """

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        claude_bin: str = "claude",
        model: Optional[str] = None,
        security_mode: Optional[str] = None,
        permission_mode: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
        native_tools: bool = False,
        system_prompt: Optional[str] = None,
        expose_hermes_tools: bool = True,
        mcp_config_path: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        warmup: bool = True,
        startup_timeout: float = 90.0,
        config_dir: Optional[str] = None,
        settings_path: Optional[str] = None,
        deny_rules: Optional[list[str]] = None,
        oauth_token_env: str = DEFAULT_OAUTH_TOKEN_ENV,
        session_id: Optional[str] = None,
        session_key: Optional[str] = None,
        resume: bool = True,
        approval_callback: Optional[Callable[..., str]] = None,
        tool_bridge_dispatch: Optional[Callable[[str, dict], Any]] = None,
        tool_bridge_tools: Optional["tuple[str, ...] | list[str]"] = None,
    ) -> None:
        self._config_dir = config_dir or claude_code_home()
        self._cwd = cwd or default_workspace_dir()
        self._settings_path = settings_path or os.path.join(
            self._config_dir, "settings.json"
        )
        self._deny_rules = list(deny_rules) if deny_rules else None
        self._oauth_token_env = oauth_token_env
        self._claude_bin = claude_bin
        self._model = (model or "").strip() or None
        self._native_tools = bool(native_tools)
        mode, tools, disallowed = resolve_permission(
            security_mode, native_tools=self._native_tools
        )
        if permission_mode:
            if permission_mode not in _VALID_PERMISSION_MODES:
                raise ValueError(
                    f"invalid claude permission_mode {permission_mode!r}; "
                    f"expected one of {sorted(_VALID_PERMISSION_MODES)}"
                )
            mode = permission_mode
        self._permission_mode = mode
        self._allowed_tools: list[str] = list(allowed_tools) if allowed_tools is not None else list(tools)
        self._disallowed_tools: list[str] = list(disallowed)
        self._system_prompt = system_prompt or ""
        self._system_prompt_path: Optional[str] = None
        self._expose_hermes_tools = expose_hermes_tools
        self._mcp_config_path = mcp_config_path
        self._owns_mcp_config = False
        self._extra_args = list(extra_args or [])
        self._env_override = env
        self._on_event = on_event
        self._warmup = warmup
        self._startup_timeout = startup_timeout
        # The CLI session id is chosen by Hermes (``--session-id``) so a
        # resumed Hermes session can ``--resume`` the same CLI transcript.
        # ``session_key`` (the Hermes session id) indexes a small map in the
        # config dir: when a ``--resume`` is rejected we must rotate to a fresh
        # uuid (the CLI refuses ``--session-id`` for an id whose transcript
        # exists: "Session ID ... is already in use"), and later resumes have
        # to find that rotated id.
        self._session_key = session_key
        mapped = load_session_map(self._config_dir).get(session_key) if session_key else None
        self._requested_session_id = mapped or session_id or str(uuid.uuid4())
        self._resume = resume
        self._resume_configured = resume
        # Hermes' approval prompt (tools.approval.prompt_dangerous_approval
        # signature via tools.terminal_tool's registered callback). None in
        # gateway/cron contexts -> gated tools are denied.
        self._approval_callback = approval_callback
        # Agent-loop tools (delegate_task, memory, session_search, todo) come
        # back to this process over a Unix socket; without a dispatcher no
        # bridge is started and the child simply does not see them, which is
        # the pre-bridge behaviour. See agent/transports/hermes_tool_bridge.py.
        self._tool_bridge_dispatch = tool_bridge_dispatch
        # Which of them THIS agent may actually run. The MCP server is built
        # from the whole registry in another process and cannot know, so a
        # leaf subagent would otherwise be advertised delegate_task and told
        # no when it called it.
        self._tool_bridge_tools = (
            tuple(tool_bridge_tools) if tool_bridge_tools is not None else None
        )
        self._tool_bridge: Optional[Any] = None
        self._resumed = False
        self._notice_emitted = False

        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Any]" = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._interrupt_event = threading.Event()
        self._turn_active = False
        self._session_id: Optional[str] = None
        self._init_info: Optional[dict] = None
        self._closed = False
        self._exit_code: Optional[int] = None
        self._pid: Optional[int] = None
        self._reader_thread: Optional[threading.Thread] = None

    # ---------- introspection ----------

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def requested_session_id(self) -> str:
        return self._requested_session_id

    @property
    def resumed(self) -> bool:
        """True when the live process was started with ``--resume``."""
        return self._resumed

    @property
    def pid(self) -> Optional[int]:
        return self._pid

    @property
    def init_info(self) -> Optional[dict]:
        """The ``system/init`` payload (tools, mcp_servers, model, ...)."""
        return self._init_info

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def config_dir(self) -> str:
        return self._config_dir

    @property
    def cwd(self) -> str:
        return self._cwd

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def rebind(
        self,
        *,
        on_event: Optional[Callable[[dict], None]] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        tool_bridge_dispatch: Optional[Callable[[str, dict], Any]] = None,
        tool_bridge_tools: Optional["tuple[str, ...] | list[str]"] = None,
    ) -> None:
        """Point the UI / approval hooks at a new owner. A warm process is
        shared across AIAgent instances (api_server builds one per request),
        so each turn re-targets the callbacks at the instance driving it.

        The tool bridge is re-targeted for the same reason and matters more:
        a bridged ``delegate_task`` dispatched onto last request's agent would
        build its children from a dead session's toolsets and post their
        results where nobody is listening. ``None`` leaves the current
        dispatcher in place — only a real replacement replaces it."""
        self._on_event = on_event
        self._approval_callback = approval_callback
        if tool_bridge_dispatch is not None:
            self._tool_bridge_dispatch = tool_bridge_dispatch
            if self._tool_bridge is not None:
                self._tool_bridge.set_dispatch(tool_bridge_dispatch)
        if tool_bridge_tools is not None:
            # What the child was TOLD it has was baked into the mcp-config at
            # spawn and cannot be narrowed now — but what the bridge will
            # actually dispatch can be, and that is the half that matters.
            self._tool_bridge_tools = tuple(tool_bridge_tools)
            if self._tool_bridge is not None:
                self._tool_bridge.set_allowed_tools(self._tool_bridge_tools)

    def needs_respawn(self, system_prompt: Optional[str]) -> bool:
        """``--append-system-prompt-file`` is read at spawn; a changed prompt
        needs a fresh process to take effect."""
        return (system_prompt or "") != self._system_prompt

    # ---------- lifecycle ----------

    def _write_system_prompt_file(self) -> Optional[str]:
        """The prompt goes through a file, not argv: it is several KB (ARG_MAX)
        and argv is visible to every local user via ``ps``."""
        if not self._system_prompt:
            return None
        if self._system_prompt_path:
            try:
                os.unlink(self._system_prompt_path)
            except OSError:
                pass
        fd, path = _mkstemp_or_explain(
            prefix="system-prompt-", suffix=".md", directory=self._config_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(self._system_prompt)
        self._system_prompt_path = path
        return path

    def _build_command(self, *, resume: bool) -> list[str]:
        cmd = [
            self._claude_bin,
            "-p",
            "--verbose",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--permission-mode", self._permission_mode,
            # No ~/.claude, project or local settings: no user hooks, no user
            # plugins, no user allow rules. Only the file below applies.
            "--setting-sources", "",
            "--settings", self._settings_path,
            # Only the MCP servers in --mcp-config (or none at all).
            "--strict-mcp-config",
        ]
        if self._permission_mode != "bypassPermissions":
            # Route every non-allowlisted tool call through Hermes' approval
            # path over the stream-json control channel instead of the
            # non-interactive default (silent denial).
            cmd += ["--permission-prompt-tool", "stdio"]
        if resume:
            cmd += ["--resume", self._requested_session_id]
        else:
            cmd += ["--session-id", self._requested_session_id]
        if self._model:
            cmd += ["--model", self._model]
        if self._disallowed_tools:
            # The tool-authority boundary (#98533): native OS-level tools are
            # denied in EVERY mode, including bypassPermissions, unless the
            # operator set claude_code.native_tools. ``--tools`` restricts the
            # built-in set to the inert allowlist (anything the CLI adds in a
            # future release is excluded by construction); the deny list is
            # belt-and-braces. Deny rules win over allow rules and over the
            # permission mode inside the CLI.
            cmd += ["--tools", ",".join(_INERT_NATIVE_TOOLS)]
            cmd += ["--disallowedTools", ",".join(dict.fromkeys(self._disallowed_tools))]
        allowed = [t for t in self._allowed_tools if t not in self._disallowed_tools]
        if self._expose_hermes_tools and self._mcp_config_path:
            cmd += ["--mcp-config", self._mcp_config_path]
            # Pre-approve the Hermes tools: they are gated by Hermes' own
            # command/file policy and pre_tool_call hooks, and the operator
            # chose to expose them. Without this, ``default`` mode silently
            # denies them. In ``default`` mode the mutating ones are left
            # out so they reach Hermes' approval prompt via can_use_tool.
            allowed.extend(hermes_mcp_allow_rules(self._permission_mode))
        if allowed and self._permission_mode != "bypassPermissions":
            cmd += ["--allowedTools", ",".join(dict.fromkeys(allowed))]
        if self._system_prompt_path:
            cmd += ["--append-system-prompt-file", self._system_prompt_path]
        cmd += self._extra_args
        return cmd

    def ensure_started(self) -> str:
        """Spawn the CLI (idempotent) and return its session id.

        Sends the warm-up message *before* waiting for ``system/init`` — the
        CLI is silent until it has a user message — and blocks until the
        warm-up's ``result`` arrives so auth failures surface here rather than
        on the user's first real turn. If ``--resume`` was attempted and the
        CLI rejected it (unknown session), respawns once with a fresh
        ``--session-id`` and logs.
        """
        if self.is_alive() and self._session_id:
            return self._session_id
        if self._closed:
            raise RuntimeError("ClaudeCodeSession is closed")
        token = resolve_oauth_token(self._oauth_token_env)
        _makedirs_or_explain(self._config_dir)
        _makedirs_or_explain(self._cwd)
        ensure_settings_file(self._settings_path, self._deny_rules)
        if self._expose_hermes_tools and not self._mcp_config_path:
            sweep_stale_mcp_configs(self._config_dir)
            self._sweep_dead_bridges(self._config_dir)
            bridge = self._start_tool_bridge()
            self._mcp_config_path = write_mcp_config(
                directory=self._config_dir,
                bridge_socket=bridge.socket_path if bridge else None,
                bridge_token=bridge.token if bridge else None,
                bridge_tools=bridge.allowed_tools if bridge else None,
            )
            self._owns_mcp_config = True
        self._write_system_prompt_file()

        transcript_exists = resume_transcript_exists(
            self._config_dir, self._requested_session_id
        )
        resume = self._resume and transcript_exists
        if transcript_exists and not resume:
            # The CLI refuses ``--session-id`` for an id that already has a
            # transcript ("Session ID ... is already in use"). Resume is off
            # (config) or was abandoned earlier, so start a fresh id now
            # instead of failing forever on the same one.
            self._rotate_session_id("resume disabled but a transcript exists")
        self._spawn(token, resume=resume)
        if self._warmup:
            warm = self._run_turn_locked(
                _WARMUP_PROMPT, silent=True, turn_timeout=self._startup_timeout,
            )
            if warm.error and resume and is_session_id_rejection(warm.error):
                # Only the CLI's own "cannot use this session id" verdict
                # justifies rotating. Any other warm-up failure (auth, rate
                # limit, MCP crash) keeps the id so a later healthy start
                # still resumes the conversation.
                self._rotate_session_id(
                    f"--resume rejected: {warm.error.splitlines()[0]}"
                )
                self._stop_process()
                self._resume = False
                self._spawn(token, resume=False)
                warm = self._run_turn_locked(
                    _WARMUP_PROMPT, silent=True, turn_timeout=self._startup_timeout,
                )
            if warm.error:
                self.close()
                raise RuntimeError(warm.error)
            # A fresh spawn (after a rotation) has a usable transcript from
            # now on; restore the configured resume behaviour so a later
            # restart() resumes it instead of rotating again.
            self._resume = self._resume_configured
        if (
            self._permission_mode == "default"
            and self._approval_callback is None
            and not self._notice_emitted
        ):
            # Never silent: gated tools will be denied because nobody can
            # answer the prompt in this context. Tell the operator once.
            self._notice_emitted = True
            logger.warning("claude-code: %s", APPROVAL_MODE_NOTICE)
            self._emit({"kind": "status", "text": APPROVAL_MODE_NOTICE})
        return self._session_id or ""

    def _rotate_session_id(self, reason: str) -> None:
        old_id = self._requested_session_id
        self._requested_session_id = str(uuid.uuid4())
        logger.warning(
            "claude-code: rotating CLI session id %s -> %s (%s)",
            old_id[:8], self._requested_session_id[:8], reason,
        )
        if self._session_key:
            try:
                save_session_mapping(
                    self._config_dir, self._session_key, self._requested_session_id,
                )
            except Exception:
                logger.warning("claude-code: could not persist session map", exc_info=True)

    def _spawn(self, token: str, *, resume: bool) -> None:
        cmd = self._build_command(resume=resume)
        env = build_child_env(
            self._env_override,
            config_dir=self._config_dir,
            token_env=self._oauth_token_env,
            oauth_token=token,
        )
        try:
            from hermes_cli._subprocess_compat import windows_hide_flags

            creationflags = windows_hide_flags()
        except Exception:  # pragma: no cover
            creationflags = 0
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self._cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=env,
                creationflags=creationflags,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"claude CLI not found at {self._claude_bin!r}. Install with: "
                "npm i -g @anthropic-ai/claude-code, then run `claude setup-token`."
            ) from exc
        self._proc = proc
        self._pid = proc.pid
        self._exit_code = None
        self._session_id = None
        self._init_info = None
        self._resumed = resume
        self._lines = queue.Queue()
        with self._stderr_lock:
            self._stderr_lines = []
        self._reader_thread = threading.Thread(
            target=self._read_stdout, args=(proc,), daemon=True,
            name="claude-code-stdout",
        )
        self._reader_thread.start()
        threading.Thread(
            target=self._read_stderr, args=(proc,), daemon=True,
            name="claude-code-stderr",
        ).start()
        logger.info(
            "claude-code session spawned: pid=%s model=%s permission=%s "
            "config_dir=%s cwd=%s %s=%s",
            proc.pid, self._model or "default", self._permission_mode,
            self._config_dir, self._cwd,
            "resume" if resume else "session-id", self._requested_session_id[:8],
        )

    def restart(self, *, system_prompt: Optional[str] = None) -> str:
        """Kill the current CLI and spawn a fresh one, resuming the same CLI
        session id so conversation context survives (e.g. when the baked-in
        system prompt changed). The old process' reader thread is left to
        unwind on its own; its identity check keeps it from touching the
        replacement's state.
        """
        if system_prompt is not None:
            self._system_prompt = system_prompt
        self._stop_process()
        return self.ensure_started()

    def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        proc.wait(timeout=3)
                    except Exception:
                        pass
                except Exception:
                    pass
        self._session_id = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_process()
        for attr in ("_system_prompt_path",):
            path = getattr(self, attr)
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                setattr(self, attr, None)
        if self._owns_mcp_config and self._mcp_config_path:
            release_mcp_config(self._mcp_config_path)
            self._mcp_config_path = None
            self._owns_mcp_config = False
        bridge, self._tool_bridge = self._tool_bridge, None
        if bridge is not None:
            try:
                bridge.close()
            except Exception:
                logger.debug("claude-code: tool bridge close failed", exc_info=True)

    @staticmethod
    def _sweep_dead_bridges(directory: str) -> None:
        """Best effort: a bridge whose owner was killed leaves its directory."""
        try:
            from agent.claude_code_runtime import sweep_dead_bridges

            removed = sweep_dead_bridges(directory)
            if removed:
                logger.info(
                    "claude-code: removed %d dead tool-bridge director%s",
                    removed, "y" if removed == 1 else "ies",
                )
        except Exception:
            logger.debug("claude-code: bridge sweep failed", exc_info=True)

    def _tool_bridge_busy(self, since: Optional[float] = None) -> bool:
        """True while the child is waiting on work this process is doing.

        Either a bridged tool call is running here, or the child's own MCP
        server is holding the line while it runs an ordinary Hermes tool.
        Either way the child is alive and waiting on us, which a silence
        timeout cannot otherwise distinguish from a child that has died.

        ``since`` is the moment the silence window opened: activity that
        ENDED inside it counts too. When a bridged call returns, the CLI still
        needs a moment to take the result and speak, and calling that moment
        "silence started two minutes ago" retired a session six seconds after
        its delegation had succeeded.
        """
        bridge = self._tool_bridge
        if bridge is None:
            return False
        try:
            if bridge.active:
                return True
            return since is not None and bridge.last_active >= since
        except Exception:  # pragma: no cover - defensive
            return False

    def _start_tool_bridge(self) -> Optional[Any]:
        """Listen for the child's agent-loop tool calls, or return ``None``.

        No dispatcher (an embedder, a test, the codex path) means no bridge
        and no bridged tools — the child's surface is exactly what a stateless
        MCP server can serve. A bridge that cannot bind is logged and skipped
        for the same reason: losing ``delegate_task`` is a smaller failure
        than refusing to start the runtime at all.
        """
        if self._tool_bridge is not None:
            return self._tool_bridge
        if self._tool_bridge_dispatch is None:
            return None
        from agent.transports.hermes_tool_bridge import (
            BRIDGED_TOOLS, BridgeError, ToolBridge,
        )

        allowed = (
            BRIDGED_TOOLS if self._tool_bridge_tools is None else self._tool_bridge_tools
        )
        if not allowed:
            # Nothing this agent may run: no listener, no advertisement.
            return None
        bridge = ToolBridge(
            self._tool_bridge_dispatch,
            directory=self._config_dir,
            allowed_tools=allowed,
        )
        try:
            bridge.start()
        except (BridgeError, OSError):
            logger.warning(
                "claude-code: tool bridge did not start — delegate_task, memory, "
                "session_search and todo will not be offered to the child",
                exc_info=True,
            )
            return None
        self._tool_bridge = bridge
        return bridge

    def __enter__(self) -> "ClaudeCodeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent: ask the active turn to stop. Safe from any thread."""
        self._interrupt_event.set()
        proc = self._proc
        if self._turn_active and proc is not None and proc.poll() is None:
            # stream-json control channel: the CLI aborts the in-flight model
            # request / tool and emits a `result` for the cut-short turn.
            self._write(
                {
                    "type": "control_request",
                    "request_id": uuid.uuid4().hex,
                    "request": {"subtype": "interrupt"},
                }
            )

    # ---------- reader threads ----------

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        assert stream is not None
        try:
            for raw in iter(stream.readline, b""):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    with self._stderr_lock:
                        self._stderr_lines.append(
                            "[stdout] " + line.decode("utf-8", "replace")[:500]
                        )
                    continue
                if isinstance(obj, dict):
                    self._lines.put(obj)
        except Exception:  # pragma: no cover - pipe torn down under us
            logger.debug("claude-code stdout reader stopped", exc_info=True)
        finally:
            # Identity check: a reader belonging to a process we replaced
            # during retire/respawn must not report EOF into the queue of the
            # healthy replacement.
            if proc is self._proc:
                self._exit_code = proc.poll()
                self._lines.put(_EOF)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        stream = proc.stderr
        assert stream is not None
        try:
            for raw in iter(stream.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip()
                if not text:
                    continue
                with self._stderr_lock:
                    self._stderr_lines.append(text)
                    if len(self._stderr_lines) > _STDERR_KEEP_LINES:
                        del self._stderr_lines[:-_STDERR_KEEP_LINES]
                logger.debug("claude-code stderr: %s", text)
        except Exception:  # pragma: no cover
            pass

    def stderr_tail(self, n: int = _STDERR_TAIL_LINES) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def _format_error(self, prefix: str, detail: str = "") -> str:
        base = f"{prefix}: {detail}" if detail else prefix
        tail = self.stderr_tail()
        joined = "\n".join(t for t in tail if t.strip())
        if not joined:
            return base
        return f"{base}\nclaude stderr (last {len(tail)} lines):\n" + redact_sensitive_text(
            joined, force=True
        )

    # ---------- writing ----------

    def _write(self, payload: dict) -> bool:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdin.closed:
            return False
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
                return True
            except (BrokenPipeError, OSError, ValueError):
                return False

    @staticmethod
    def _user_message(content: Any) -> dict:
        """One stream-json user message. ``content`` is a str or a list of
        Claude content blocks (see :func:`_coerce_input_blocks`)."""
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        return {"type": "user", "message": {"role": "user", "content": blocks}}

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: float = 600.0,
        idle_timeout: float = 300.0,
    ) -> TurnResult:
        """Send one user message and block until the CLI's ``result`` event.

        ``turn_timeout`` bounds the whole turn; ``idle_timeout`` bounds the
        silence between two events (a wedged CLI emits nothing at all).
        """
        blocks = _coerce_input_blocks(user_input)
        if not self.is_alive():
            result = TurnResult(should_retire=True)
            result.error = self._format_error(
                "claude process is not running",
                f"exit code {self._exit_code}" if self._exit_code is not None else "",
            )
            return result
        if self._interrupt_event.is_set():
            # A hard stop arrived while we were spawning; honor it rather than
            # erasing the signal underneath the caller.
            self._interrupt_event.clear()
            return TurnResult(interrupted=True, session_id=self._session_id)
        return self._run_turn_locked(
            blocks, silent=False, turn_timeout=turn_timeout, idle_timeout=idle_timeout
        )

    def _emit(self, event: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            logger.debug("claude-code on_event hook raised", exc_info=True)

    def _run_turn_locked(
        self,
        content: Any,
        *,
        silent: bool,
        turn_timeout: float,
        idle_timeout: float = 300.0,
    ) -> TurnResult:
        result = TurnResult(session_id=self._session_id)
        proc = self._proc
        if proc is None:
            result.error = "claude process is not running"
            result.should_retire = True
            return result

        self._turn_active = True
        self._interrupt_event.clear()
        if not self._write(self._user_message(content)):
            self._turn_active = False
            result.error = self._format_error("could not write to claude stdin")
            result.should_retire = True
            return result

        started_at = time.monotonic()
        deadline = started_at + turn_timeout
        bridge_grace = _BRIDGE_TURN_GRACE_SECONDS if self._tool_bridge else 0.0
        projector = _TurnProjector(self, silent=silent)
        saw_result = False
        try:
            while True:
                now = time.monotonic()
                if now >= deadline:
                    # The elapsed time, not turn_timeout: a turn held open by
                    # the tool bridge can run far past its nominal budget, and
                    # "exceeded 300s" after 2400 s sends the reader hunting a
                    # setting that was not the one that fired.
                    result.error = self._format_error(
                        f"claude turn exceeded {int(now - started_at)}s"
                    )
                    result.should_retire = True
                    break
                wait = min(idle_timeout, deadline - now)
                if self._interrupt_event.is_set():
                    # request_interrupt() already sent the control request;
                    # give the CLI a short grace period to emit its `result`.
                    wait = min(wait, _INTERRUPT_GRACE_SECONDS)
                window_start = time.monotonic()
                try:
                    obj = self._lines.get(timeout=wait)
                except queue.Empty:
                    if self._interrupt_event.is_set():
                        # The CLI ignored the interrupt: kill it rather than
                        # leave the user waiting on a turn they cancelled.
                        self._kill(proc)
                        result.should_retire = True
                        break
                    if bridge_grace > 0 and self._tool_bridge_busy(since=window_start):
                        # Not silence — the CLI is quiet because it is waiting
                        # on US: a bridged tool call (a synchronous subagent
                        # fan-out, say) is running in this process and the CLI
                        # emits nothing while an MCP call is outstanding. The
                        # time it costs is given back to both clocks, up to a
                        # ceiling the bridge's own client timeout outlives, so
                        # a wedged dispatcher still ends the turn.
                        spent = min(wait, bridge_grace)
                        bridge_grace -= spent
                        deadline += spent
                        logger.info(
                            "claude-code: %.0fs of silence covered by the tool "
                            "bridge (still working; %.0fs of grace left)",
                            spent, bridge_grace,
                        )
                        continue
                    result.error = self._format_error(
                        f"claude produced no output for {int(idle_timeout)}s"
                    )
                    result.should_retire = True
                    break
                if obj is _EOF:
                    # Only the live process' reader may enqueue _EOF (identity
                    # check in _read_stdout), so this is authoritative.
                    code = self._exit_code if self._exit_code is not None else proc.poll()
                    result.error = self._format_error(
                        f"claude exited unexpectedly (code {code})"
                    )
                    result.should_retire = True
                    break
                if obj.get("type") == "control_request":
                    self._handle_control_request(obj, silent=silent)
                    continue
                if projector.handle(obj, result):
                    saw_result = True
                    break
        finally:
            self._turn_active = False

        if saw_result:
            # Drain anything the CLI already flushed behind `result` (e.g. a
            # trailing rate_limit_event) so it cannot leak into the next turn.
            self._drain_pending(projector, result)
        if self._interrupt_event.is_set():
            result.interrupted = True
        self._interrupt_event.clear()
        projector.finish(result)
        if not silent:
            logger.info(
                "claude-code turn finished: session=%s tools=%d chars=%d error=%s",
                (self._session_id or "")[:8], result.tool_iterations,
                len(result.final_text), bool(result.error),
            )
        return result

    # ---------- permission prompts (``--permission-prompt-tool stdio``) ----------

    def _handle_control_request(self, obj: dict, *, silent: bool) -> None:
        request = obj.get("request") or {}
        request_id = obj.get("request_id")
        if not isinstance(request, dict) or request.get("subtype") != "can_use_tool":
            # Unknown control request: answer so the CLI does not hang.
            self._write_control_response(request_id, behavior="deny", message="unsupported control request")
            return
        behavior, message = self.decide_permission(request, silent=silent)
        self._write_control_response(request_id, behavior=behavior, message=message,
                                     updated_input=request.get("input"))

    def _write_control_response(
        self, request_id: Any, *, behavior: str, message: str = "",
        updated_input: Any = None,
    ) -> None:
        inner: dict[str, Any] = {"behavior": behavior}
        if behavior == "allow":
            # The CLI expects the (possibly edited) input echoed back on allow.
            inner["updatedInput"] = updated_input if isinstance(updated_input, dict) else {}
        else:
            inner["message"] = message or _DENY_USER
        self._write(
            {
                "type": "control_response",
                "response": {"subtype": "success", "request_id": request_id, "response": inner},
            }
        )

    def decide_permission(self, request: dict, *, silent: bool = False) -> tuple[str, str]:
        """Route a ``can_use_tool`` request through Hermes' approval prompt.

        Mirrors ``CodexAppServerSession._decide_exec_approval``: protocol
        routing only — mode/timeout policy lives in ``tools/approval.py``
        behind the callback. Returns ``(behavior, message)``.
        """
        tool_name = str(request.get("tool_name") or "tool")
        args = request.get("input") if isinstance(request.get("input"), dict) else {}
        if tool_name == "Bash" or hermes_tool_name(tool_name) == "terminal":
            command = str(args.get("command") or "")
            description = str(args.get("description") or request.get("description") or "")
        else:
            command = f"{tool_name} {_format_tool_args(args)}"[:500]
            description = str(request.get("description") or f"Claude Code requests {tool_name}")
        description = f"Claude Code requests {tool_name}" + (f" — {description}" if description else "")
        if self._approval_callback is None:
            logger.info("claude-code: denying %s (no approver wired)", tool_name)
            return "deny", _DENY_NO_APPROVER
        try:
            choice = self._approval_callback(command, description, allow_permanent=False)
        except Exception:
            logger.exception("claude-code: approval callback raised; denying %s", tool_name)
            return "deny", "Denied by Hermes: the approval prompt failed."
        choice = str(choice or "").strip().lower()
        if choice in _ALLOW_CHOICES:
            return "allow", ""
        if choice == "timeout":
            return "deny", _DENY_TIMEOUT
        return "deny", _DENY_USER

    def _kill(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.kill()
            proc.wait(timeout=3)
        except Exception:
            pass

    def _drain_pending(self, projector: "_TurnProjector", result: TurnResult) -> None:
        end = time.monotonic() + 0.05
        while True:
            try:
                obj = self._lines.get(timeout=max(0.0, end - time.monotonic()))
            except queue.Empty:
                return
            if obj is _EOF:
                # Keep the sentinel for the next run_turn to observe.
                self._lines.put(_EOF)
                return
            projector.handle(obj, result, post_result=True)


#: Largest image forwarded to the CLI as-is (decoded bytes). Anthropic's
#: per-image limit is 5 MB; anything bigger is downscaled (Pillow) or, when
#: Pillow is missing, passed through with a warning so the API says why.
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_IMAGE_PLACEHOLDER = "[image attached — not forwarded to claude]"


def _coerce_input_text(user_input: Any) -> str:
    """Flatten a Hermes user message (str or multipart list) to text.

    Images become a placeholder. Kept for callers that need a text view of
    the turn; the wire uses :func:`_coerce_input_blocks`."""
    parts: list[str] = []
    for block in _coerce_input_blocks(user_input):
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "image":
            parts.append("[image]")
    return "\n".join(p for p in parts if p)


def _coerce_input_blocks(user_input: Any) -> list[dict]:
    """Turn a Hermes user message (str or OpenAI-style multipart list) into
    Claude content blocks for the stream-json ``user`` message.

    Text parts are joined (newline-separated, as the text-only path always
    did) into one text block; ``image_url`` parts carrying a ``data:image/…;
    base64,`` URL become ``{"type": "image", "source": {"type": "base64", …}}``
    blocks — the shape the CLI accepts in streaming-input mode. An image
    that cannot be forwarded (http(s) URL, unknown media type, bad base64)
    is replaced by a short text placeholder so the model knows something
    was there. Image bytes are never logged.
    """
    if isinstance(user_input, str):
        return [{"type": "text", "text": user_input}]
    if not isinstance(user_input, list):
        return [{"type": "text", "text": str(user_input or "")}]
    blocks: list[dict] = []
    text_parts: list[str] = []

    def _flush_text() -> None:
        joined = "\n".join(p for p in text_parts if p)
        text_parts.clear()
        if joined:
            blocks.append({"type": "text", "text": joined})

    for item in user_input:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict):
            kind = item.get("type")
            if kind == "text":
                text_parts.append(str(item.get("text") or ""))
            elif kind in ("image_url", "input_image"):
                block = _image_block_from_part(item)
                if block is None:
                    text_parts.append(_IMAGE_PLACEHOLDER)
                else:
                    _flush_text()
                    blocks.append(block)
    _flush_text()
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _image_block_from_part(part: dict) -> Optional[dict]:
    """``{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}``
    → a Claude base64 image block, or None when it cannot be forwarded."""
    ref = part.get("image_url")
    url = ref.get("url") if isinstance(ref, dict) else ref
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, sep, payload = url.partition(",")
    if not sep:
        return None
    meta = header[len("data:"):].split(";")
    media_type = (meta[0] or "").strip().lower()
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    if "base64" not in (m.strip().lower() for m in meta[1:]):
        return None
    if media_type not in _IMAGE_MEDIA_TYPES:
        return None
    data = "".join(payload.split())      # tolerate wrapped base64
    try:
        raw_len = len(base64.b64decode(data, validate=True))
    except (binascii.Error, ValueError):
        return None
    if raw_len > _MAX_IMAGE_BYTES:
        shrunk = _downscale_image_b64(data, raw_len)
        if shrunk is None:
            logger.warning(
                "claude-code: image of %d bytes exceeds %d and Pillow is not "
                "installed; forwarding as-is",
                raw_len, _MAX_IMAGE_BYTES,
            )
        else:
            media_type, data = shrunk
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _downscale_image_b64(data: str, raw_len: int) -> Optional[tuple[str, str]]:
    """Re-encode an oversized image as JPEG under the cap, halving the
    longest side until it fits. Returns (media_type, base64) or None when
    Pillow is unavailable or the image cannot be decoded."""
    try:
        from PIL import Image
    except ImportError:
        return None
    import io

    try:
        im = Image.open(io.BytesIO(base64.b64decode(data)))
        im.load()
    except Exception:
        logger.warning("claude-code: oversized image could not be decoded; forwarding as-is")
        return None
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    # First guess from the byte ratio, then keep halving if still over.
    factor = min(1.0, (_MAX_IMAGE_BYTES / float(raw_len)) ** 0.5)
    for _ in range(8):
        w = max(1, int(im.width * factor))
        h = max(1, int(im.height * factor))
        out = io.BytesIO()
        im.resize((w, h)).save(out, format="JPEG", quality=80)
        if out.tell() <= _MAX_IMAGE_BYTES:
            logger.info(
                "claude-code: downscaled a %d-byte image to %dx%d (%d bytes)",
                raw_len, w, h, out.tell(),
            )
            return "image/jpeg", base64.b64encode(out.getvalue()).decode("ascii")
        factor *= 0.5
    return None


class _TurnProjector:
    """Folds one turn's stream-json events into a :class:`TurnResult`.

    Emits UI events through the session's ``on_event`` hook with a small,
    runtime-agnostic vocabulary (``text_delta``, ``reasoning_delta``,
    ``tool_started``, ``tool_completed``, ``assistant_message``, ``status``,
    ``init``) so ``claude_code_runtime`` can bridge them into Hermes' gateway
    callbacks without knowing the wire format.
    """

    def __init__(self, session: ClaudeCodeSession, *, silent: bool) -> None:
        self._s = session
        self._silent = silent
        self._text_parts: list[str] = []          # streamed deltas (fallback)
        self._assistant_texts: list[str] = []     # authoritative text blocks
        self._pending_tools: dict[str, dict] = {}  # tool_use id -> {name,args,t0}
        self._pending_interim: Optional[str] = None  # text not yet known to be commentary
        self._saw_stream_text = False

    # -- public --

    def handle(self, obj: dict, result: TurnResult, *, post_result: bool = False) -> bool:
        """Consume one event. Returns True when it was the terminal ``result``."""
        etype = obj.get("type")
        if etype == "system":
            self._on_system(obj, result)
        elif etype == "stream_event":
            self._on_stream_event(obj)
        elif etype == "assistant":
            self._on_assistant(obj, result)
        elif etype == "user":
            self._on_user(obj, result)
        elif etype == "rate_limit_event":
            self._on_rate_limit(obj)
        elif etype == "result":
            if post_result:
                return False
            self._on_result(obj, result)
            return True
        return False

    def finish(self, result: TurnResult) -> None:
        if not result.final_text:
            if self._assistant_texts:
                # The last text block is the answer; earlier ones were
                # commentary around tool calls and were emitted as interim
                # messages (and are all in projected_messages for history).
                result.final_text = self._assistant_texts[-1].strip()
            elif self._text_parts:
                result.final_text = "".join(self._text_parts).strip()
        if result.final_text and not result.projected_messages:
            # Text-only turn whose `assistant` event we never saw (e.g. an
            # interrupt cut the stream): still record what the user saw.
            result.projected_messages.append(
                {"role": "assistant", "content": result.final_text}
            )

    # -- handlers --

    def _on_system(self, obj: dict, result: TurnResult) -> None:
        if obj.get("subtype") != "init":
            return
        sid = obj.get("session_id")
        if isinstance(sid, str) and sid:
            self._s._session_id = sid
            result.session_id = sid
        self._s._init_info = obj
        servers = obj.get("mcp_servers") or []
        for server in servers:
            if (
                isinstance(server, dict)
                and server.get("name") == HERMES_TOOLS_MCP_SERVER_NAME
                and server.get("status") not in (None, "connected")
            ):
                logger.warning(
                    "hermes-tools MCP server status=%s — Hermes tools may be "
                    "unavailable inside claude", server.get("status"),
                )
        self._s._emit({"kind": "init", "session_id": sid, "info": obj})

    def _on_stream_event(self, obj: dict) -> None:
        ev = obj.get("event") or {}
        if not isinstance(ev, dict) or ev.get("type") != "content_block_delta":
            return
        delta = ev.get("delta") or {}
        if not isinstance(delta, dict):
            return
        dtype = delta.get("type")
        if dtype == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                self._text_parts.append(text)
                self._saw_stream_text = True
                if not self._silent:
                    self._s._emit({"kind": "text_delta", "text": text})
        elif dtype == "thinking_delta":
            text = delta.get("thinking")
            if isinstance(text, str) and text and not self._silent:
                self._s._emit({"kind": "reasoning_delta", "text": text})

    def _on_assistant(self, obj: dict, result: TurnResult) -> None:
        msg = obj.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            return
        texts: list[str] = []
        tool_calls: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
            elif btype == "tool_use":
                call_id = str(block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
                wire_name = str(block.get("name") or "tool")
                name = hermes_tool_name(wire_name)
                args = block.get("input")
                if not isinstance(args, dict):
                    args = {"input": args}
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": _format_tool_args(args)},
                    }
                )
                self._pending_tools[call_id] = {
                    "name": name, "wire_name": wire_name, "args": args,
                    "t0": time.monotonic(),
                }
                if not self._silent:
                    self._s._emit(
                        {
                            "kind": "tool_started", "call_id": call_id, "name": name,
                            "wire_name": wire_name, "args": args,
                            "preview": _tool_preview(wire_name, args),
                        }
                    )
        if not texts and not tool_calls:
            return
        joined = "\n".join(texts)
        if tool_calls and self._pending_interim and not self._silent:
            # Text that arrived in an earlier `assistant` event turned out to
            # be commentary before a tool call: surface it as an interim
            # message so chat surfaces show it before the tool bubble.
            self._s._emit(
                {"kind": "assistant_message", "text": self._pending_interim, "interim": True}
            )
            self._pending_interim = None
        if texts:
            self._assistant_texts.append(joined)
            # Deltas already streamed this text live; the block is the
            # authoritative copy for history. Reset the fallback buffer so a
            # later text block isn't double-counted.
            self._text_parts = []
            if tool_calls:
                if not self._silent:
                    self._s._emit({"kind": "assistant_message", "text": joined, "interim": True})
            else:
                self._pending_interim = joined
        message: dict[str, Any] = {"role": "assistant", "content": joined if texts else None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        result.projected_messages.append(message)

    def _on_user(self, obj: dict, result: TurnResult) -> None:
        msg = obj.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = str(block.get("tool_use_id") or "")
            text = _content_text(block.get("content"))
            is_error = bool(block.get("is_error"))
            pending = self._pending_tools.pop(call_id, None)
            name = pending["name"] if pending else "tool"
            if is_error and text and not text.startswith("[error]"):
                text = f"[error] {text}"
            result.projected_messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": text}
            )
            result.tool_iterations += 1
            if not self._silent:
                duration = (
                    time.monotonic() - pending["t0"] if pending else None
                )
                self._s._emit(
                    {
                        "kind": "tool_completed", "call_id": call_id, "name": name,
                        "args": pending["args"] if pending else {},
                        "result": text, "is_error": is_error, "duration": duration,
                    }
                )

    def _on_rate_limit(self, obj: dict) -> None:
        info = obj.get("rate_limit_info") or {}
        windows = info.get("unifiedWindows") if isinstance(info, dict) else None
        if not isinstance(windows, dict):
            return
        worst, label = 0.0, ""
        for name, window in windows.items():
            if not isinstance(window, dict):
                continue
            util = window.get("utilization")
            if isinstance(util, (int, float)) and util > worst:
                worst, label = float(util), str(name).replace("_", " ")
        if worst >= 0.9 and not self._silent:
            self._s._emit(
                {
                    "kind": "status",
                    "text": f"heads up — {int(worst * 100)}% of your {label} limit used",
                }
            )

    def _on_result(self, obj: dict, result: TurnResult) -> None:
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage:
            result.token_usage_last = dict(usage)
            if isinstance(obj.get("total_cost_usd"), (int, float)):
                result.token_usage_last["total_cost_usd"] = obj["total_cost_usd"]
        model_usage = obj.get("modelUsage")
        if isinstance(model_usage, dict):
            for entry in model_usage.values():
                window = entry.get("contextWindow") if isinstance(entry, dict) else None
                if isinstance(window, int) and window > 0:
                    result.model_context_window = window
                    break
        text = obj.get("result")
        if obj.get("is_error"):
            detail = text if isinstance(text, str) and text else obj.get("subtype") or "unknown error"
            result.error = self._s._format_error("claude turn failed", str(detail))
            lowered = str(detail).lower()
            if (
                "not logged in" in lowered
                or "failed to authenticate" in lowered
                or "oauth" in lowered and "expired" in lowered
                or "invalid api key" in lowered
            ):
                result.error = (
                    f"{result.error}\nClaude Code credential rejected. Run "
                    "`claude setup-token` and update $CLAUDE_CODE_OAUTH_TOKEN, "
                    "then retry."
                )
                result.should_retire = True
            return
        subtype = obj.get("subtype")
        if isinstance(subtype, str) and subtype.startswith("error"):
            # e.g. error_max_turns / error_during_execution without is_error
            result.error = self._s._format_error(f"claude turn ended with {subtype}")
        if isinstance(text, str) and text.strip() and not self._assistant_texts:
            result.final_text = text.strip()
