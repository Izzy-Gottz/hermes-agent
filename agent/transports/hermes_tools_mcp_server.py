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
  - delegate_task / memory /             — `_AGENT_LOOP_TOOLS` in Hermes
    session_search / todo                  (model_tools.py). They require
                                           the running AIAgent context to
                                           dispatch (mid-loop state), so a
                                           stateless MCP callback can't
                                           drive them. See the inline
                                           comment on EXPOSED_TOOLS below.

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

import inspect
import json
import logging
import os
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
#   - delegate_task / memory / session_search / todo — these are
#     `_AGENT_LOOP_TOOLS` in Hermes (model_tools.py:493). They require
#     the running AIAgent context to dispatch (mid-loop state), so a
#     stateless MCP callback can't drive them. Hermes' default runtime
#     keeps these working; the codex_app_server runtime cannot.
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
)
PROFILE_ENV = "HERMES_MCP_TOOL_PROFILE"
SCRUB_ENV = "HERMES_MCP_SCRUB_ENV"
CLAUDE_CODE_PROFILE = "claude-code"


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


def _build_server(profile: Optional[str] = None) -> Any:
    """Create the MCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    profile = profile if profile is not None else os.environ.get(PROFILE_ENV)
    tools_to_expose = exposed_tools_for_profile(profile)
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

    if (profile or "").strip().lower() == CLAUDE_CODE_PROFILE:
        instructions = (
            "Hermes Agent's tool surface. Claude Code's native Bash/Read/"
            "Write/Edit/Glob/Grep/WebFetch tools are disabled in this "
            "session; use terminal, read_file, write_file, patch, "
            "search_files and process from this server for shell and file "
            "work, plus web search/extract, browser automation, vision, "
            "image generation, skills and TTS."
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
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

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

            def _dispatch(**kwargs: Any) -> str:
                try:
                    # Filter out None values before dispatch so unset optionals
                    # aren't forwarded to the handler.
                    args = {k: v for k, v in kwargs.items() if v is not None}
                    return handle_function_call(tool_name, args or {})
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})

            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            _dispatch.__signature__ = sig
            _dispatch.__annotations__ = {**annots, "return": str}
            return _dispatch

        try:
            mcp.add_tool(
                _make_handler(name, params_schema),
                name=name,
                description=description,
            )
        except TypeError:
            # Older mcp SDK signature — fall back to decorator-style. The
            # synthesized __signature__ on the handler still drives schema
            # generation there.
            handler = _make_handler(name, params_schema)
            handler = mcp.tool(name=name, description=description)(handler)

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
