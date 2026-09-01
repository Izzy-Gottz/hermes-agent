"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation,
delegate_task subagents, vision analysis, persistent memory, skills,
cross-session search, image generation, TTS — is unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list              — Hermes' skill library
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - delegate_task / memory /             — only when there is no tool
    session_search / todo                  bridge. These are Hermes'
                                           `_AGENT_LOOP_TOOLS`: they need
                                           the running AIAgent, which this
                                           process does not have. Given
                                           $HERMES_TOOL_BRIDGE_SOCKET they
                                           are offered and forwarded to the
                                           process that does — see
                                           agent/transports/
                                           hermes_tool_bridge.py. Without
                                           one they stay hidden rather than
                                           be offered and always fail.

External MCP servers (``mcp_servers`` in ``~/.hermes/config.yaml``) are
registered on top of whichever profile is active, under their registry names
``mcp__<server>__<tool>``. The profile lists curate *Hermes' own* tools, not
the user's connections.

Profiles (``HERMES_MCP_TOOL_PROFILE``):
  - default / ``codex``      — EXPOSED_TOOLS above (codex owns shell + files).
  - ``claude-code``          — EXPOSED_TOOLS + CLAUDE_CODE_OS_TOOLS (terminal,
                               read_file, write_file, patch, search_files,
                               process). The claude_code runtime disallows
                               the CLI's native Bash/Read/Write/... so every
                               shell command and file write goes through
                               Hermes' own guards (check_all_command_guards,
                               file policy, env sanitisation, hooks) here.

``HERMES_MCP_SCRUB_ENV`` (comma-separated) names variables removed from the
server's own environment at startup — the claude_code runtime lists the CLI
credential (``CLAUDE_CODE_OAUTH_TOKEN``) the server would otherwise inherit.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in; ClaudeCodeSession.ensure_started()
            via the --mcp-config it writes.
"""

from __future__ import annotations

import base64
import inspect
import keyword
import io
import json
import logging
import os
import re
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)

# JSON Schema type -> Python type mapping for signature generation
_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _signature_from_schema(schema: dict | None) -> tuple[inspect.Signature, dict[str, type]]:
    """Build a Python function signature and annotations from a JSON schema.

    Args:
        schema: JSON Schema dict with "properties" and "required" keys.

    Returns:
        (signature, annotations_dict) where signature has KEYWORD_ONLY params
        and annotations maps param names to Python types.
    """
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    params, annots = [], {}

    for pname, pspec in props.items():
        if pname.startswith("_"):
            continue
        # A JSON Schema property may be called anything; a Python parameter
        # may not. Cloudflare's MCP ships "action.type", and inspect.Parameter
        # raises ValueError on it — which used to take the whole server down
        # (see the guard in _build_server). Skipped here rather than fatal:
        # _install_schema hands the client the authoritative Hermes schema a
        # moment later, so the signature only has to be *valid*, not complete.
        if not pname.isidentifier() or keyword.iskeyword(pname):
            logger.debug("parameter %r cannot be a Python parameter name — omitted from the signature", pname)
            continue
        py = _JSON_TO_PY.get((pspec or {}).get("type"), Any)
        ann, default = (
            (py, inspect.Parameter.empty)
            if pname in required
            else (Optional[py], None)
        )
        annots[pname] = ann
        params.append(
            inspect.Parameter(
                pname, inspect.Parameter.KEYWORD_ONLY, annotation=ann, default=default
            )
        )

    return inspect.Signature(params, return_annotation=str), annots


# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — codex's built-ins cover these and approval routes through
#     codex's own UI.
#   - delegate_task / memory / session_search / todo — `_AGENT_LOOP_TOOLS`
#     in Hermes (model_tools.py). They require the running AIAgent to
#     dispatch (mid-loop state), which this process does not have. Under
#     claude_code they are offered anyway and forwarded to the process that
#     does, over the tool bridge (see AGENT_LOOP_TOOLS below); codex has no
#     bridge wired, so there they are still withheld.
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_request_review",
    "kanban_request_changes",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)


#: OS-level Hermes tools exposed ONLY in the ``claude-code`` profile: the
#: claude_code runtime denies the CLI's native equivalents and relies on
#: these so Hermes' command/file policy is the single authority.
CLAUDE_CODE_OS_TOOLS: tuple[str, ...] = (
    "terminal",
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "process",
    # Eyes and hands. Screenshots come back as MCP image blocks (see
    # to_mcp_content); every mutating action still runs through
    # handle_function_call, so pre_tool_call hooks and guards apply. Only
    # registered when tools.computer_use's check_fn passes (cua-driver, or
    # the macOS-native backend).
    "computer_use",
)
PROFILE_ENV = "HERMES_MCP_TOOL_PROFILE"
SCRUB_ENV = "HERMES_MCP_SCRUB_ENV"
CLAUDE_CODE_PROFILE = "claude-code"


#: Registry-name prefix for tools that came from an external MCP server
#: (mirrors ``tools.mcp_tool.MCP_TOOL_NAME_PREFIX``; duplicated so this
#: module still imports when the optional ``mcp`` package is absent).
EXTERNAL_MCP_PREFIX = "mcp__"


def discover_external_mcp_servers() -> list[str]:
    """Register the user's own MCP servers into *this* process's registry.

    ``model_tools`` deliberately dropped module-level MCP discovery (#16856,
    it blocked the gateway's event loop), so every entry point runs it at its
    own startup. This server is an entry point and was not on that list — the
    reason a child could see Hermes' built-ins but nothing the user had
    connected.

    Cheap by design: ``_register_from_cache_sync`` puts cached manifests into
    the registry without spawning anything and the first real call connects.
    Best effort — an unreachable server must not stop Hermes' own tools from
    being served.
    """
    try:
        from tools.mcp_tool import discover_mcp_tools
    except Exception:
        logger.debug("MCP client support unavailable — no external servers", exc_info=True)
        return []
    try:
        return list(discover_mcp_tools() or [])
    except Exception:
        logger.warning(
            "external MCP discovery failed — serving Hermes' own tools only",
            exc_info=True,
        )
        return []


#: ``model_tools._AGENT_LOOP_TOOLS`` — handle_function_call answers them with
#: "must be handled by the agent loop", because they need the running AIAgent
#: (mid-loop state) and this process has none. They are still offered when a
#: tool bridge is configured: the call is forwarded to the Hermes process that
#: owns the agent and dispatched there (agent/transports/hermes_tool_bridge.py).
#: Without a bridge they are withheld, because offering one would be offering
#: a tool that always fails.
AGENT_LOOP_TOOLS: tuple[str, ...] = ("todo", "memory", "session_search", "delegate_task")


def tools_to_offer(
    profile: Optional[str],
    available: "set[str] | frozenset[str]",
    *,
    bridge: "bool | tuple[str, ...] | set[str]" = False,
) -> tuple[str, ...]:
    """Everything Hermes has, minus what cannot work here.

    This used to be an ALLOWLIST — a hand-written tuple of about thirty names.
    Anything Hermes gained afterwards, and anything a plugin registered, was
    invisible to the child forever: the moe-connectors plugin's whatsapp_send,
    telegram_status and nine siblings were all present in the registry and
    none of them reached the model, so Moe answered that it could not read the
    user's WhatsApp while holding the tool that reads it. send_message was
    missing for the same reason and got written up as a messaging decision.

    An allowlist over a registry that grows is a list that is wrong by
    default. The rule is the other way round now: everything, minus the few
    that cannot be dispatched statelessly, minus (on codex) the OS tools codex
    owns itself.

    ``bridge`` names the agent-loop tools the process at the other end of the
    tool bridge will actually run for this child — all four for a top-level
    agent, fewer (often none) for a subagent, whose ``DELEGATE_BLOCKED_TOOLS``
    deny ``delegate_task`` and ``memory``. Passing ``True`` means all four;
    the point of taking a set rather than a flag is that a child is never
    *advertised* a tool the dispatcher would then refuse.
    """
    key = (profile or "").strip().lower()
    if bridge is True:
        bridged = set(AGENT_LOOP_TOOLS)
    elif bridge:
        bridged = set(bridge)
    else:
        bridged = set()
    blocked = set(AGENT_LOOP_TOOLS) - bridged
    if key != CLAUDE_CODE_PROFILE:
        # codex brings its own shell and file tools; two of each confuses the
        # model and routes approval through the wrong UI.
        blocked |= set(CLAUDE_CODE_OS_TOOLS)
    return tuple(sorted(n for n in available if n not in blocked))


def exposed_tools_for_profile(profile: Optional[str]) -> tuple[str, ...]:
    """Tool names registered for ``profile`` (``None``/``codex`` -> default)."""
    key = (profile or "").strip().lower()
    if key == CLAUDE_CODE_PROFILE:
        return CLAUDE_CODE_OS_TOOLS + EXPOSED_TOOLS
    return EXPOSED_TOOLS


def scrub_environment(env: Optional[dict] = None) -> list[str]:
    """Drop the variables named in ``$HERMES_MCP_SCRUB_ENV`` from ``env``
    (default ``os.environ``) and return the names removed. Also drops any of
    them that arrived as an empty string (the mcp-config ``env`` block blanks
    the CLI credential before this runs)."""
    target = os.environ if env is None else env
    names = [n.strip() for n in (target.get(SCRUB_ENV) or "").split(",") if n.strip()]
    removed = []
    for name in names:
        if name in target:
            target.pop(name, None)
            removed.append(name)
    return removed


#: Process-env marker that makes ``tools.approval.check_all_command_guards``
#: treat this process as headless (no human can answer a prompt): dangerous /
#: Tirith-flagged commands are DENIED with a message instead of silently
#: approved, governed by ``approvals.single_query_mode`` (default ``deny``).
HEADLESS_APPROVAL_ENV = "HERMES_SINGLE_QUERY_SESSION"


def prepare_claude_code_profile(env: Optional[dict] = None) -> list:
    """Make the ``claude-code`` profile enforce the policy it advertises.

    The server is spawned by the ``claude`` CLI, not by the Hermes CLI or
    gateway, so nothing has (a) registered the ``hooks.pre_tool_call`` shell
    hooks from config.yaml on the plugin manager, or (b) marked the process
    as one with no interactive approver. Without (a) config hooks never fire
    for ``terminal`` / ``write_file`` calls; without (b)
    ``check_all_command_guards`` falls into its "not CLI, not gateway"
    branch and approves every non-hardline dangerous command unprompted.

    Returns the registered hook specs (empty when none are configured).
    Consent for the hook scripts follows the normal channels
    (``hooks_auto_accept: true`` / ``HERMES_ACCEPT_HOOKS``); there is no TTY
    here, so an unaccepted hook is skipped exactly as it is in the gateway.
    """
    target = os.environ if env is None else env
    target.setdefault(HEADLESS_APPROVAL_ENV, "1")
    try:
        from agent.shell_hooks import register_from_config
        from hermes_cli.config import load_config

        return list(register_from_config(load_config(), accept_hooks=False))
    except Exception:
        logger.warning("claude-code profile: shell-hook registration failed", exc_info=True)
        return []


#: Longest side (px) of any image block sent over the bridge. Anthropic
#: downsamples above 1568 px anyway, so larger only costs tokens and time.
IMAGE_MAX_LONG_SIDE = 1568

_DATA_URL = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+);base64,(?P<b64>[A-Za-z0-9+/=\s]+)$")


def _shrink_image(data_b64: str, mime: str, max_side: int = IMAGE_MAX_LONG_SIDE) -> tuple[str, str, Optional[float]]:
    """Downscale a base64 image so its longest side is ≤ ``max_side``.

    Returns ``(b64, mime, scale)`` — ``scale`` is ``None`` when untouched,
    else new/original so the caller can tell the model how coordinates map.
    Anything PIL can't read is passed through unchanged.
    """
    try:
        from tools.computer_use.macos_native_backend import downscale_png, fit_scale
        from PIL import Image  # noqa: F401 — presence check
    except Exception:
        return data_b64, mime, None
    try:
        raw = base64.b64decode(data_b64, validate=False)
        from PIL import Image as _Image
        im = _Image.open(io.BytesIO(raw))
        w, h = im.size
    except Exception:
        return data_b64, mime, None
    if fit_scale(w, h, max_side) >= 1.0:
        return data_b64, mime, None
    try:
        png, _nw, _nh, scale = downscale_png(raw, max_side)
    except Exception:
        return data_b64, mime, None
    return base64.b64encode(png).decode("ascii"), "image/png", scale


def to_mcp_content(result: Any) -> Any:
    """Turn a Hermes tool result into what the MCP SDK should send.

    Plain strings pass through (the SDK wraps them in a text block). A
    Hermes multimodal result — ``{"_multimodal": True, "content": [{"type":
    "text", ...}, {"type": "image_url", "image_url": {"url": "data:…"}}]}``,
    the shape ``computer_use`` captures and ``vision``-style tools return —
    becomes ``[TextContent, ImageContent, ...]`` so the CLI's model actually
    *sees* the screenshot instead of a base64 blob or a file path. Images
    wider than :data:`IMAGE_MAX_LONG_SIDE` are shrunk here and the text
    block says so, with the factor, so coordinates read off the picture can
    be scaled back by the tool that made it.
    """
    if not (isinstance(result, dict) and result.get("_multimodal")):
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return result
    try:
        from mcp.types import ImageContent, TextContent
    except ImportError:  # pragma: no cover - mcp missing → text summary only
        return str(result.get("text_summary") or "")
    blocks: list = []
    notes: list[str] = []
    for part in result.get("content") or []:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append(TextContent(type="text", text=str(part.get("text") or "")))
        elif ptype in {"image_url", "image"}:
            url = ((part.get("image_url") or {}).get("url") if ptype == "image_url" else None) or ""
            m = _DATA_URL.match(url.strip()) if url else None
            if m:
                data, mime = m.group("b64").replace("\n", ""), m.group("mime")
            else:
                src = part.get("source") or {}
                data, mime = str(src.get("data") or part.get("data") or ""), str(
                    src.get("media_type") or part.get("mimeType") or "image/png")
            if not data:
                continue
            data, mime, scale = _shrink_image(data, mime)
            if scale:
                notes.append(
                    f"(image downscaled by {scale:.3f} to fit {IMAGE_MAX_LONG_SIDE}px; "
                    f"multiply pixel coordinates by {1 / scale:.3f} to address the original)")
            blocks.append(ImageContent(type="image", data=data, mimeType=mime))
    if not any(isinstance(b, TextContent) for b in blocks):
        blocks.insert(0, TextContent(type="text", text=str(result.get("text_summary") or "")))
    if notes:
        blocks.append(TextContent(type="text", text="\n".join(notes)))
    return blocks


def _install_schema(mcp: Any, name: str, params_schema: dict) -> bool:
    """Replace the SDK's signature-derived input schema for ``name`` with the
    Hermes JSON schema. Returns True when it took."""
    try:
        manager = getattr(mcp, "_tool_manager", None)
        tools = getattr(manager, "_tools", None)
        tool = tools.get(name) if isinstance(tools, dict) else None
        if tool is None or not isinstance(params_schema, dict) or not params_schema.get("properties"):
            return False
        schema = {k: v for k, v in params_schema.items() if not k.startswith("_")}
        schema["properties"] = {k: v for k, v in schema["properties"].items() if not k.startswith("_")}
        schema.setdefault("type", "object")
        tool.parameters = schema
        return True
    except Exception:
        logger.debug("could not install Hermes schema for %s", name, exc_info=True)
        return False


def _build_server(profile: Optional[str] = None) -> Any:
    """Create the MCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    profile = profile if profile is not None else os.environ.get(PROFILE_ENV)
    try:
        # mcp 2.0 removed `mcp.server.fastmcp`; `mcp.server.MCPServer` is the
        # same decorator/add_tool surface under the new name.
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    from agent.transports.hermes_tool_bridge import (
        BridgeError, bridge_hold, bridged_tool_names, call_bridged_tool,
    )

    bridged_names = bridged_tool_names()
    has_bridge = bool(bridged_names)

    # Before any schema is read: whatever the user has connected has to be in
    # the registry, or get_tool_definitions() below simply will not see it.
    external_names = discover_external_mcp_servers()
    if external_names:
        logger.info("discovered %d tool(s) from external MCP servers", len(external_names))

    if (profile or "").strip().lower() == CLAUDE_CODE_PROFILE:
        instructions = (
            "Hermes Agent's tool surface. Claude Code's native Bash/Read/"
            "Write/Edit/Glob/Grep/WebFetch/Task tools are disabled in this "
            "session; use terminal, read_file, write_file, patch, "
            "search_files and process from this server for shell and file "
            "work, plus web search/extract, browser automation, vision, "
            "image generation, skills and TTS. Tools named mcp__<server>__* "
            "come from the user's own connected MCP servers and are as real "
            "as the rest — prefer them over guessing or answering from "
            "context when a question is about the system they front."
        )
        if "delegate_task" in bridged_names:
            instructions += (
                " delegate_task is this session's subagent tool — it runs on "
                "the parent Hermes agent, so its children get the same model, "
                "the same account and the same tools."
            )
        persistent = [n for n in ("memory", "todo", "session_search") if n in bridged_names]
        if persistent:
            instructions += (
                f" {', '.join(persistent)} run on the parent agent too, and are "
                "the persistent ones."
            )
    else:
        instructions = (
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "subagent delegation, vision, image generation, persistent "
            "memory, skills, and cross-session search."
        )
    mcp = MCPServer("hermes-tools", instructions=instructions)

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    # The ASSEMBLED catalogue, not the raw one.
    #
    # This used to pass skip_tool_search_assembly=True, on the reasoning that
    # the child does its own tool-surface management. Measured, that reasoning
    # cost 79,000 tokens and about 14x the price on every uncached turn: the
    # CLI defers the schemas but every one of ~2000 tool NAMES still lands in
    # the prompt (87,907 input tokens for "say only: ok", against 8,887 with
    # no MCP config at all).
    #
    # tool_search is Hermes' own answer to exactly this, and the Cloudflare
    # manifest says so in as many words — it pins ?codemode=false precisely so
    # "Hermes's tool_search then defers the whole surface behind its bridge
    # tools and searches the FULL catalog with complete schemas". Nothing is
    # hidden by taking it: tool_search, tool_describe and tool_call are in the
    # assembled list, they dispatch from this stateless process (model_tools
    # handles them before the agent-loop rejection), and every deferred tool
    # stays callable through them.
    #
    # What must NOT be deferred is the connectors plugin — the tools the user
    # connected on purpose. That is pinned in tool_search._DIRECT_SURFACE_TOOLSETS
    # rather than worked around here.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    # Everything Hermes has: its own tools, its plugins' tools (the Moe
    # connectors live here), and every external MCP server's tools — plus the
    # four agent-loop tools when there is a bridge to run them on.
    tools_to_expose = tools_to_offer(profile, set(all_defs), bridge=bridged_names)
    if has_bridge:
        logger.info(
            "tool bridge configured — %s dispatch on the parent agent",
            ", ".join(bridged_names),
        )

    exposed_count = 0

    for name in tools_to_expose:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        # The SDK wants a Python callable and derives the input schema from
        # its signature — there is no inputSchema parameter on either the
        # decorator or add_tool(). So build a closure that takes the arguments
        # dict, dispatches via handle_function_call, returns the result
        # string, and carries a __signature__ synthesized from the Hermes
        # JSON Schema (see _signature_from_schema) for the SDK to read.
        def _make_handler(tool_name: str, schema: dict | None):
            sig, annots = _signature_from_schema(schema)

            bridged = tool_name in bridged_names

            def _dispatch(**kwargs: Any) -> Any:
                try:
                    # Filter out None values before dispatch so unset optionals
                    # aren't forwarded to the handler.
                    args = {k: v for k, v in kwargs.items() if v is not None}
                    if bridged:
                        # Home to the process that owns the agent — and the
                        # credential a subagent has to be built with.
                        return to_mcp_content(call_bridged_tool(tool_name, args or {}))
                    # Hold the line for the parent while this runs, so a long
                    # tool call cannot be mistaken for a dead CLI and retired
                    # at claude_code.silence_timeout. Cheap insurance: one
                    # AF_UNIX connect per call, and measured on 2.1.252 the
                    # CLI keeps streaming during a tool call anyway.
                    with bridge_hold():
                        return to_mcp_content(handle_function_call(tool_name, args or {}))
                except BridgeError as exc:
                    # Not an exception the model should read as a crash: the
                    # bridge is down, or the call took longer than it allows.
                    logger.warning("bridged tool %s failed: %s", tool_name, exc)
                    return json.dumps({"error": str(exc), "tool": tool_name})
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})

            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            _dispatch.__signature__ = sig
            # The SDK derives the *input* schema from the signature; the return
            # annotation stays ``str`` so no structured-output schema is
            # generated (an image block is not JSON-schema-able).
            _dispatch.__annotations__ = {**annots, "return": str}
            return _dispatch

        # Belt and braces for everything the first guard cannot foresee. This
        # server is the child's ENTIRE tool surface: an exception building one
        # handler used to abort _build_server, so a single odd schema on one
        # connected MCP server left the model with no tools at all — not even
        # terminal — and looked exactly like "nothing is connected".
        try:
            handler = _make_handler(name, params_schema)
        except Exception:
            logger.warning("skipping %s — its schema cannot be turned into a tool", name, exc_info=True)
            continue
        try:
            # structured_output=False: the return annotation is ``str`` for
            # the schema generator's sake, but a handler may return content
            # blocks (text + image); an output model would reject those.
            mcp.add_tool(handler, name=name, description=description, structured_output=False)
        except TypeError:
            try:
                # Older SDK add_tool() without the keyword.
                mcp.add_tool(handler, name=name, description=description)
            except TypeError:
                # Oldest: decorator-style only. The synthesized __signature__
                # on the handler still drives schema generation there.
                handler = mcp.tool(name=name, description=description)(handler)

        # The signature-derived schema knows only types. Hand the client the
        # authoritative Hermes schema (enums, per-parameter descriptions,
        # array item types) — for computer_use that is the difference between
        # the model knowing the action list and guessing it. Best effort:
        # private SDK attribute, so a layout change just leaves the plain one.
        _install_schema(mcp, name, params_schema)
        exposed_count += 1

    logger.info(
        "hermes-tools MCP server registered %d/%d tools (profile=%s)",
        exposed_count,
        len(tools_to_expose),
        profile or "default",
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")
    # Credentials that belong to the spawning CLI, not to Hermes' tools
    # (CLAUDE_CODE_OAUTH_TOKEN): drop them before any tool can spawn a shell.
    scrubbed = scrub_environment()
    if scrubbed:
        logger.info("scrubbed %s from the server environment", ",".join(scrubbed))
    if (os.environ.get(PROFILE_ENV) or "").strip().lower() == CLAUDE_CODE_PROFILE:
        hooks = prepare_claude_code_profile()
        logger.info("claude-code profile: %d config hook(s) registered; headless approval", len(hooks))

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # MCPServer.run() defaults to stdio transport, which is what codex
    # spawns us on.
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
