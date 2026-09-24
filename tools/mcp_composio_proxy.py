"""Composio tools by name, on a session that can hold several accounts.

A Composio tool-router session either preloads named tools (``GMAIL_FETCH_EMAILS``
in the model's list, callable by name) or lets a person connect more than one
account of a service -- Composio refuses both together: "preload is not
supported when multi-account is enabled for this session". Moe needs both. A
model that sees no ``GMAIL_*`` tool reports Gmail as not connected; and on
2026-09-24 the owner had two Gmail accounts connected, asked for a mail to go
from one of them, and nothing could choose it (Moe ticket #12).

So the session is created multi-account, and Hermes does the preloading
itself: the slugs listed under ``mcp_servers.<name>.composio_tools`` are
fetched once with ``COMPOSIO_GET_TOOL_SCHEMAS``, given an ``account``
parameter, and registered like any tool the server listed. A call to one is
sent as ``COMPOSIO_MULTI_EXECUTE_TOOL`` with that slug -- the route every
Composio tool can take, and the only one that carries ``account``.

Measured against the live API on 2026-09-24 (multi_account session, two Gmail
accounts): ``GMAIL_GET_PROFILE`` through the multiplexer with ``account`` set
to each address returned that address; with it omitted, the default account;
calling an unlisted slug by name is ``Tool GMAIL_FETCH_EMAILS not found``.
The tool names are the ones a preload produced, so every hook that reads
``mcp_composio_GMAIL_SEND_EMAIL`` (Moe's send gate, its Composio budget) sees
the same call it always did.
"""

import json
import logging
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MULTIPLEXER = "COMPOSIO_MULTI_EXECUTE_TOOL"
SCHEMAS = "COMPOSIO_GET_TOOL_SCHEMAS"
CONFIG_KEY = "composio_tools"
#: The subset that only reads. COMPOSIO_GET_TOOL_SCHEMAS carries no
#: annotations, and a preloaded tool did (``readOnlyHint``); without them
#: every Gmail read would reach Moe's send gate as write-capable and ask.
#: The config writer measured these; Hermes does not guess them.
READ_ONLY_KEY = "composio_read_only"

_SLUG = re.compile(r"^[A-Z][A-Z0-9_]{2,127}$")

#: Every Gmail tool already has a ``user_id`` ("me" or an address) and
#: GMAIL_SEND_EMAIL a ``from_email`` (a send-as alias). Neither chooses the
#: account -- the ticket's model set ``user_id`` to the address it wanted and
#: would have sent from the default. The description says so in words.
ACCOUNT_PARAM = {
    "type": "string",
    "description": (
        "Which of the person's connected accounts to act as: its email address "
        "(for example \"name@gmail.com\") or its account id. Omit it to use the "
        "default account. This is the ONLY parameter that chooses the account; "
        "user_id and from_email do not."
    ),
}


def configured_slugs(config: Any, key: str = CONFIG_KEY) -> List[str]:
    """The slugs this server should offer by name, from its config entry."""
    raw = (config or {}).get(key) if isinstance(config, dict) else None
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    for s in raw:
        s = str(s or "").strip().upper()
        if _SLUG.match(s) and s not in out:
            out.append(s)
    return out


def _text_of(result: Any) -> str:
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    return ""


def proxied_tools(listed: List[Any], schemas_text: str, slugs: List[str],
                  read_only: Any = ()) -> List[SimpleNamespace]:
    """Tool stand-ins for ``slugs`` from a GET_TOOL_SCHEMAS answer.

    A slug the server already lists natively is left to the server; a slug the
    answer has no schema for is dropped (never registered with a guessed
    shape -- a tool with the wrong parameters is worse than none).
    """
    native = {str(getattr(t, "name", "") or "") for t in listed}
    try:
        data = json.loads(schemas_text or "{}")
    except (TypeError, ValueError):
        return []
    data = data.get("data", data) if isinstance(data, dict) else {}
    found = data.get("tool_schemas") if isinstance(data, dict) else None
    if not isinstance(found, dict):
        return []
    out: List[SimpleNamespace] = []
    for slug in slugs:
        if slug in native:
            continue
        entry = found.get(slug)
        if not isinstance(entry, dict) or not isinstance(entry.get("input_schema"), dict):
            logger.info("Composio: no schema for %s; not offered by name", slug)
            continue
        schema = json.loads(json.dumps(entry["input_schema"]))  # a private copy
        schema.setdefault("type", "object")
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = schema["properties"] = {}
        props["account"] = dict(ACCOUNT_PARAM)
        out.append(SimpleNamespace(
            name=slug,
            description=str(entry.get("description") or ""),
            inputSchema=schema,
            input_schema=schema,
            # Declared in config or write-capable: unknown fails closed.
            annotations={"readOnlyHint": slug in set(read_only or ())},
            composio_proxy=True,
        ))
    return out


async def augment(server: Any, listed: List[Any]) -> List[Any]:
    """``listed`` plus the configured slugs, for a Composio server that can
    multiplex. Never raises: a failure here leaves the server's own list."""
    try:
        from tools.mcp_tool_handlers import _is_composio_server
        slugs = configured_slugs(getattr(server, "_config", None))
        if not slugs or not _is_composio_server(server):
            return listed
        names = {str(getattr(t, "name", "") or "") for t in listed}
        server._composio_native = frozenset(names)
        if MULTIPLEXER not in names or SCHEMAS not in names:
            logger.warning("Composio server lists no %s/%s; named tools not offered", MULTIPLEXER, SCHEMAS)
            return listed
        wanted = [s for s in slugs if s not in names]
        if not wanted:
            return listed
        result = await server.session.call_tool(SCHEMAS, arguments={"tool_slugs": wanted})
        extra = proxied_tools(listed, _text_of(result), wanted,
                              configured_slugs(server._config, READ_ONLY_KEY))
        if extra:
            logger.info("Composio: offering %d tool(s) by name through %s", len(extra), MULTIPLEXER)
        return list(listed) + extra
    except Exception as exc:  # pragma: no cover - network/SDK variance
        logger.warning("Composio named tools not offered: %s", exc)
        return listed


def is_proxied(server: Any, tool_name: str) -> bool:
    """Does a call to ``tool_name`` on ``server`` go through the multiplexer?

    Yes when the slug is configured and the server did not list it natively.
    A server registered from the schema cache and not yet discovered has no
    native list; the multiplexer reaches every slug, so it is the safe route.
    """
    slug = str(tool_name or "")
    if slug not in configured_slugs(getattr(server, "_config", None)):
        return False
    native = getattr(server, "_composio_native", None)
    return not (isinstance(native, frozenset) and slug in native)


def rewrite(server: Any, tool_name: str, args: Any) -> Tuple[str, Any]:
    """``(tool_name, args)`` to send. A proxied slug becomes one multiplexer
    entry; ``account`` moves from the arguments onto the entry."""
    if not is_proxied(server, tool_name):
        return tool_name, args
    arguments = dict(args) if isinstance(args, dict) else {}
    account = arguments.pop("account", None)
    entry: Dict[str, Any] = {"tool_slug": str(tool_name), "arguments": arguments}
    if isinstance(account, str) and account.strip():
        entry["account"] = account.strip()
    return MULTIPLEXER, {"tools": [entry], "thought": "run %s" % tool_name}
