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
import time
import urllib.parse
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
                  read_only: Any = (), accounts: Optional[Dict[str, List[dict]]] = None
                  ) -> List[SimpleNamespace]:
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
        props["account"] = account_param(slug, accounts)
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
        # Which accounts there are, by address, so the model sees its choices
        # in the parameter it chooses with. Never fatal: without it the tools
        # are still offered, with the generic description.
        try:
            accounts = await load_accounts(server, force=True)
        except Exception as exc:  # pragma: no cover - network variance
            logger.info("Composio: accounts not listed (%s)", exc)
            accounts = None
        extra = proxied_tools(listed, _text_of(result), wanted,
                              configured_slugs(server._config, READ_ONLY_KEY), accounts)
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


# ── which account ──────────────────────────────────────────────────────────
#
# Composio's ``account`` on a multiplexer entry matches a connected-account id
# or the account's ``alias`` -- nothing else. Composio records no address on an
# account, so an account nobody gave an alias to cannot be chosen by its
# address at all. Measured 2026-10-07: three Gmail accounts, one with no alias;
# ``account: "yisrael@claimoe.ai"`` answered ``No account found matching``,
# while ``account: "ca_1r8-AuqvZ_oB"`` (the same mailbox) answered its profile.
# The send only went through when the model dropped ``account`` and the
# default happened to be that mailbox -- which is luck, not choice.
#
# So Hermes resolves the name itself: it lists the person's live accounts,
# learns the address of any that has no alias from the service's own profile
# tool (once per account, remembered on disk -- an account id never changes
# owner), and sends the id. A name that matches nothing is refused with the
# list of choices; it is never passed on to land on the default.

#: Per toolkit: the tool that says whose account this is, and the field.
#: Measured, not guessed: GMAIL_GET_PROFILE with ``account`` set to an id
#: answered that mailbox's ``emailAddress`` (2026-10-07).
PROFILE_TOOLS = {"gmail": ("GMAIL_GET_PROFILE", "emailAddress")}
#: A lookup that misses re-reads the list (a person may just have connected
#: one), but not more often than this.
RELIST_AFTER_S = 30.0
_MAX_PAGES = 10


class AccountChoiceError(ValueError):
    """The ``account`` named no connected account. The message lists them."""


def _names_path():
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "composio-account-names.json"


def _read_names() -> Dict[str, str]:
    try:
        with open(_names_path(), encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in blob.items() if isinstance(v, str) and v.strip()} \
        if isinstance(blob, dict) else {}


def _save_names(names: Dict[str, str]) -> None:
    try:
        from utils import atomic_json_write
        path = _names_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, names, mode=0o600)
    except Exception as exc:  # pragma: no cover - disk variance
        logger.info("Composio: account names not saved (%s)", exc)


def _rest_base_and_session(server: Any) -> Tuple[str, str]:
    url = str((getattr(server, "_config", None) or {}).get("url") or "")
    parts = urllib.parse.urlparse(url)
    m = re.search(r"/tool_router/([^/]+)/mcp", parts.path or "")
    if not m or not parts.scheme or not parts.netloc:
        return "", ""
    return "%s://%s" % (parts.scheme, parts.netloc), m.group(1)


async def _rest(server: Any, method: str, path: str, body: Any = None) -> Any:
    """One call to Composio's REST API on the host and with the headers this
    server is already configured with. ``None`` on any failure."""
    import httpx
    base, _ = _rest_base_and_session(server)
    if not base:
        return None
    headers = dict((server._config or {}).get("headers") or {})
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.request(method, base + path, headers=headers, json=body)
    except Exception as exc:  # pragma: no cover - network variance
        logger.info("Composio: %s %s failed (%s)", method, path.split("?")[0], exc)
        return None
    if resp.status_code != 200:
        logger.info("Composio: %s %s answered %s", method, path.split("?")[0], resp.status_code)
        return None
    try:
        return resp.json()
    except ValueError:
        return None


async def _learn_name(server: Any, toolkit: str, account_id: str) -> str:
    """The address the service knows ``account_id`` by, from its own profile
    tool sent through the multiplexer with that account. "" when it cannot say."""
    tool, field = PROFILE_TOOLS.get(toolkit, (None, None))
    if not tool:
        return ""
    try:
        result = await server.session.call_tool(MULTIPLEXER, arguments={
            "tools": [{"tool_slug": tool, "arguments": {}, "account": account_id}],
            "thought": "which account is this"})
        data = json.loads(_text_of(result) or "{}")
        data = data.get("data", data)
        first = (data.get("results") or [{}])[0]
        found = ((first.get("response") or {}).get("data") or {}).get(field)
    except Exception as exc:
        logger.info("Composio: could not learn the name of %s (%s)", account_id, exc)
        return ""
    return found.strip() if isinstance(found, str) else ""


async def load_accounts(server: Any, force: bool = False) -> Optional[Dict[str, List[dict]]]:
    """``{toolkit: [{"id", "who"}, ...]}`` -- this session's user's live
    accounts, newest first, each with the address it is known by ("" when it
    could not be learned). ``None`` when the list cannot be read."""
    cached = getattr(server, "_composio_accounts", None)
    if cached is not None:
        at, directory = cached
        if not force or time.monotonic() - at < RELIST_AFTER_S:
            return directory
    base, sid = _rest_base_and_session(server)
    if not sid:
        return None
    session = await _rest(server, "GET", "/api/v3.1/tool_router/session/" + urllib.parse.quote(sid, safe=""))
    user = ((session or {}).get("config") or {}).get("user_id") if isinstance(session, dict) else None
    if not isinstance(user, str) or not user:
        return None
    items: List[dict] = []
    cursor = ""
    for _ in range(_MAX_PAGES):
        page = await _rest(server, "GET", "/api/v3/connected_accounts?limit=100&statuses=ACTIVE&user_ids="
                           + urllib.parse.quote(user, safe="")
                           + ("&cursor=" + urllib.parse.quote(cursor, safe="") if cursor else ""))
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            return None
        items += [a for a in page["items"] if isinstance(a, dict)]
        cursor = page.get("next_cursor") or ""
        if not cursor:
            break
    names = _read_names()
    learned = False
    directory: Dict[str, List[dict]] = {}
    for a in sorted(items, key=lambda a: str(a.get("created_at") or ""), reverse=True):
        # The filter is a request, not a guarantee: another user's account is
        # not one this person can choose.
        if a.get("user_id") != user or str(a.get("status") or "").upper() != "ACTIVE":
            continue
        toolkit = str((a.get("toolkit") or {}).get("slug") or "").lower()
        aid = str(a.get("id") or "")
        if not toolkit or not aid:
            continue
        alias = a.get("alias")
        who = alias.strip() if isinstance(alias, str) and alias.strip() else names.get(aid, "")
        if not who and toolkit in PROFILE_TOOLS:
            who = await _learn_name(server, toolkit, aid)
            if who:
                names[aid] = who
                learned = True
                # Also give it Composio's own alias, so Composio can match the
                # address itself and Moe's Connections screen shows it. Only
                # ever where there was none: a name a person chose is kept.
                await _rest(server, "PATCH", "/api/v3/connected_accounts/" + urllib.parse.quote(aid, safe=""),
                            {"alias": who})
        directory.setdefault(toolkit, []).append({"id": aid, "who": who})
    if learned:
        _save_names(names)
    server._composio_accounts = (time.monotonic(), directory)
    return directory


def toolkit_of(slug: str, directory: Dict[str, List[dict]]) -> str:
    """The toolkit a tool slug belongs to, by the longest toolkit prefix
    (``GOOGLECALENDAR_`` before ``GOOGLE_``). "" when none of them."""
    slug = str(slug or "").upper()
    best = ""
    for tk in directory:
        if slug.startswith(tk.upper() + "_") and len(tk) > len(best):
            best = tk
    return best


def _label(a: dict) -> str:
    return '"%s"' % a["who"] if a["who"] else '"%s" (address unknown)' % a["id"]


def account_param(slug: str, directory: Optional[Dict[str, List[dict]]]) -> Dict[str, Any]:
    """ACCOUNT_PARAM, naming the connected accounts when they are known."""
    param = dict(ACCOUNT_PARAM)
    accts = (directory or {}).get(toolkit_of(slug, directory or {})) or []
    if len(accts) > 1:
        param["description"] += " Connected accounts: %s." % ", ".join(_label(a) for a in accts)
    return param


def match_account(accts: List[dict], value: str) -> Optional[str]:
    """The id ``value`` names among ``accts``: an id, an address (any case),
    or -- without an ``@`` -- a part of exactly one address. None otherwise."""
    v = str(value or "").strip()
    low = v.lower()
    for a in accts:
        if v == a["id"]:
            return a["id"]
    for a in accts:
        if a["who"] and low == a["who"].lower():
            return a["id"]
    if "@" not in low and low:
        hits = [a for a in accts if a["who"] and low in a["who"].lower()]
        if len(hits) == 1:
            return hits[0]["id"]
    return None


async def resolve_accounts(server: Any, tool_name: str, args: Any) -> Any:
    """``args`` for a multiplexer call with every ``account`` turned into the
    id it names. Raises AccountChoiceError for a name that names nothing.
    Leaves the call as it was when the accounts cannot be read at all --
    Composio then refuses an unknown name itself; it does not default."""
    if tool_name != MULTIPLEXER or not isinstance(args, dict) or not isinstance(args.get("tools"), list):
        return args
    if not any(isinstance(e, dict) and isinstance(e.get("account"), str) and e["account"].strip()
               for e in args["tools"]):
        return args
    from tools.mcp_tool_handlers import _is_composio_server
    if not _is_composio_server(server):
        return args
    directory = await load_accounts(server)
    if directory is None:
        return args
    entries = []
    for entry in args["tools"]:
        value = entry.get("account") if isinstance(entry, dict) else None
        if not isinstance(value, str) or not value.strip():
            entries.append(entry)
            continue
        slug = str(entry.get("tool_slug") or "")
        found = None
        for attempt in (0, 1):
            tk = toolkit_of(slug, directory)
            found = match_account(directory.get(tk) or [], value)
            if found or attempt:
                break
            fresh = await load_accounts(server, force=True)
            if fresh is None:
                break
            directory = fresh
        if not found:
            tk = toolkit_of(slug, directory)
            accts = directory.get(tk) or []
            if accts:
                raise AccountChoiceError(
                    'No connected account matches account="%s" for %s. Its connected accounts are: %s. '
                    "Call again with one of these as account (the address, or the id for one whose "
                    "address is unknown). Nothing was sent." % (value.strip(), slug, ", ".join(_label(a) for a in accts)))
            raise AccountChoiceError(
                'No connected account matches account="%s" for %s: no account of that service is '
                "connected. Nothing was sent." % (value.strip(), slug))
        entries.append(dict(entry, account=found))
    return dict(args, tools=entries)
