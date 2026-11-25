"""Tools that must never wait behind ``tool_search``.

Every MCP tool is deferrable by default (``tools.tool_search``): an MCP server's
endpoints are a catalogue to search. But ``tool_search`` matches words
literally, so a tool the person connected on purpose -- "check my email" --
can sit one search away and never be found, and the model then says the
service "isn't connected" while the connection screen says it is.

A server can therefore pin tools, in either of two ways, and the deferral
predicate honours both:

* ``mcp_servers.<name>.always_load`` in config: ``true`` pins every tool the
  server lists, a list of names (globs allowed, matched like
  ``tools.include``) pins those, and ``false`` pins none -- not even ones the
  server marks itself.
* the server marks a listed tool with ``_meta: {"anthropic/alwaysLoad": true}``,
  the same per-tool marker Claude Code reads, so a server pinned for one
  runtime is pinned for every runtime.

Pins are recorded here by registry name when the tool registers and dropped
when it deregisters. This module is deliberately tiny and import-free so that
``tools.tool_search`` can consult it without importing the MCP stack.
"""

from __future__ import annotations

import threading

#: The per-tool ``_meta`` key Claude Code reads; honoured on every runtime.
ALWAYS_LOAD_META_KEY = "anthropic/alwaysLoad"

#: Default ceiling on the schema tokens ONE server may pin (chars/4, the same
#: estimate tool_search uses). A server that marks every tool alwaysLoad must
#: not be able to undo deferral wholesale; past the ceiling, the remaining
#: pins stay behind tool_search and the registration log says which.
#: ``mcp_servers.<name>.always_load_max_tokens`` overrides it per server.
DEFAULT_MAX_PINNED_TOKENS_PER_SERVER = 8000

_lock = threading.Lock()
_pinned: set[str] = set()


def pin(name: str) -> None:
    with _lock:
        _pinned.add(name)


def unpin(name: str) -> None:
    with _lock:
        _pinned.discard(name)


def is_pinned(name: str) -> bool:
    with _lock:
        return name in _pinned


def pinned() -> frozenset[str]:
    with _lock:
        return frozenset(_pinned)


def clear() -> None:
    """Tests only."""
    with _lock:
        _pinned.clear()
