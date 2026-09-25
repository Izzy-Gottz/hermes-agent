"""Fixable failure causes: a closed vocabulary a tool attaches to an error the person can fix.

Modelled on ``tools.bot_failure_reasons``. A tool that fails for a reason the person (or the host app)
can fix — a macOS privacy grant, a connector sign-in, a missing browser — says WHICH reason as a code,
instead of a paragraph the model has to interpret. The free-text ``error`` stays, for older hosts and
for the model's own context; the code is what a host app acts on.

The contract (the host app parses exactly this; do not rename or drop a key)
--------------------------------------------------------------------------
A fixable tool error is the JSON object ``tools.registry.tool_error`` produces, with these keys ALWAYS
present (``null`` when not applicable)::

    {
      "error":   str,          # plain words for the person: what broke, why, and what fixes it
      "code":    str,          # one of CODES below; never anything else
      "owner":   "app" | "driver" | null,
                               # whose macOS grant it is: "app" = the host app (Memoe) itself,
                               # "driver" = the screen-control helper (CuaDriver, com.trycua.driver).
                               # They are different rows in System Settings. null = not a grant.
      "pane":    str | null,   # a PANES anchor (e.g. "Privacy_AllFiles"); the host turns it into
                               # x-apple.systempreferences:com.apple.preference.security?<pane>
      "subject": str | null,   # the app or folder involved ("Google Chrome", "Desktop", "Notes")
      "retry":   bool          # the identical call is expected to succeed once the named fix is done,
                               # with nothing else changed. False when something else must happen
                               # first that the error says (e.g. relaunching the app).
      ...                      # optional code-specific extras (e.g. "path"), never one of the above.
                               # "restart": "driver" = the grant applies to a freshly launched
                               # screen helper; Hermes relaunches it on the next call itself.
    }

Unknown codes, owners or panes raise ``ValueError`` at the call site: a typo must fail a test, not
reach a person as a code nobody handles.

Carrying the fields through string plumbing
-------------------------------------------
Much of the tool stack passes failures as ``(value, error_str)`` tuples or re-raises them as
``RuntimeError(err)``. ``FixMessage`` is a ``str`` that also carries the fields, so it survives those
layers unchanged; ``FixableError`` is the exception form. ``as_tool_error(err)`` at a tool boundary
emits the full contract when ``err`` carries fields and a plain ``{"error": ...}`` otherwise.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

# macOS privacy grants
TCC_APP_DATA = "tcc_app_data"                        # EPERM reading another app's data (a browser profile)
TCC_FILES = "tcc_files"                              # EPERM on Desktop / Documents / Downloads / iCloud
TCC_AUTOMATION = "tcc_automation"                    # osascript -1743, not authorized to send Apple events
TCC_DRIVER_ACCESSIBILITY = "tcc_driver_accessibility"  # the screen-control helper lacks Accessibility
TCC_DRIVER_SCREEN = "tcc_driver_screen"              # the screen-control helper lacks Screen Recording
TCC_MICROPHONE = "tcc_microphone"
TCC_CALENDAR = "tcc_calendar"
TCC_REMINDERS = "tcc_reminders"
TCC_CONTACTS = "tcc_contacts"
# not grants, but fixable
DRIVER_NOT_RUNNING = "driver_not_running"
WINDOW_OTHER_SPACE = "window_other_space"
MCP_NEEDS_REAUTH = "mcp_needs_reauth"
MCP_NOT_CONNECTED = "mcp_not_connected"
CLAUDE_LOGIN_EXPIRED = "claude_login_expired"
CLAUDE_USAGE_CAP = "claude_usage_cap"
BROWSER_MISSING = "browser_missing"
PROFILE_LOCKED = "profile_locked"                    # a real browser profile lock: Windows only
BROWSER_EXTENSION_MISSING = "browser_extension_missing"  # Moe's Chrome extension not installed / not connected
# Not a setting at all: a page step only the person can take (a passkey, a CAPTCHA, an identity check),
# met on a turn nobody is at. The job reports it; nothing on this Mac fixes it (tools/browser_handoff_tool.py).
PERSON_NEEDED = "person_needed"

CODES = frozenset({
    TCC_APP_DATA, TCC_FILES, TCC_AUTOMATION, TCC_DRIVER_ACCESSIBILITY, TCC_DRIVER_SCREEN,
    TCC_MICROPHONE, TCC_CALENDAR, TCC_REMINDERS, TCC_CONTACTS,
    DRIVER_NOT_RUNNING, WINDOW_OTHER_SPACE, MCP_NEEDS_REAUTH, MCP_NOT_CONNECTED,
    CLAUDE_LOGIN_EXPIRED, CLAUDE_USAGE_CAP, BROWSER_MISSING, PROFILE_LOCKED, BROWSER_EXTENSION_MISSING,
    PERSON_NEEDED,
})

OWNER_APP = "app"
OWNER_DRIVER = "driver"
OWNERS = frozenset({OWNER_APP, OWNER_DRIVER})

PANES = frozenset({
    "Privacy_AllFiles", "Privacy_FilesAndFolders", "Privacy_Automation", "Privacy_Accessibility", "Privacy_ScreenCapture",
    "Privacy_Microphone", "Privacy_Calendars", "Privacy_Reminders", "Privacy_Contacts",
})

#: The grant each tcc_* code is about: ``code -> (owner, pane)``. A tcc_* error must name an owner;
#: saying the wrong one is how a person turns on the wrong app.
GRANTS: Dict[str, tuple] = {
    TCC_APP_DATA: (OWNER_APP, "Privacy_AllFiles"),
    TCC_FILES: (OWNER_APP, "Privacy_AllFiles"),
    TCC_AUTOMATION: (OWNER_APP, "Privacy_Automation"),
    TCC_DRIVER_ACCESSIBILITY: (OWNER_DRIVER, "Privacy_Accessibility"),
    TCC_DRIVER_SCREEN: (OWNER_DRIVER, "Privacy_ScreenCapture"),
    TCC_MICROPHONE: (OWNER_APP, "Privacy_Microphone"),
    TCC_CALENDAR: (OWNER_APP, "Privacy_Calendars"),
    TCC_REMINDERS: (OWNER_APP, "Privacy_Reminders"),
    TCC_CONTACTS: (OWNER_APP, "Privacy_Contacts"),
}

_CONTRACT_KEYS = frozenset({"error", "code", "owner", "pane", "subject", "retry"})


def host_app_name() -> str:
    """What to call the app the person grants permissions to. ``HERMES_HOST_APP_NAME`` (set by the host,
    e.g. "Memoe"); otherwise "the app" — never a guessed product name."""
    name = (os.environ.get("HERMES_HOST_APP_NAME") or "").strip()
    return name or "the app"


def fix_fields(code: str, *, owner: Optional[str] = None, pane: Optional[str] = None,
               subject: Optional[str] = None, retry: bool = True, **extra: Any) -> Dict[str, Any]:
    """Validated contract fields (everything but ``error``). For a tcc_* code, ``owner`` and ``pane``
    default to that grant's; an explicit value must agree with ``GRANTS`` on owner."""
    if code not in CODES:
        raise ValueError(f"unknown fix code {code!r}; add it to tools.fix_reasons.CODES first")
    grant = GRANTS.get(code)
    if grant is not None:
        owner = owner if owner is not None else grant[0]
        pane = pane if pane is not None else grant[1]
        if owner != grant[0]:
            raise ValueError(f"{code} is a grant of owner {grant[0]!r}, not {owner!r}")
    if owner is not None and owner not in OWNERS:
        raise ValueError(f"unknown owner {owner!r}; expected one of {sorted(OWNERS)}")
    if pane is not None and pane not in PANES:
        raise ValueError(f"unknown pane {pane!r}; expected one of {sorted(PANES)}")
    if clash := (_CONTRACT_KEYS | {MAC_KEY, NONCE_KEY}) & extra.keys():
        raise ValueError(f"extras may not override contract keys: {sorted(clash)}")
    fields = {"code": code, "owner": owner, "pane": pane, "subject": subject, "retry": bool(retry), **extra}
    # A nonce, so the same failure twice is two signatures and the one-use rule (_SEEN) only ever
    # refuses a copy, never a second genuine failure.
    fields[NONCE_KEY] = secrets.token_hex(8)
    fields[MAC_KEY] = _mac(_signing_key(), fields)
    return fields


class FixMessage(str):
    """A plain-words error message that also carries its fix fields through ``(value, err)`` plumbing."""

    fields: Dict[str, Any]

    def __new__(cls, message: str, fields: Dict[str, Any]) -> "FixMessage":
        obj = super().__new__(cls, message)
        obj.fields = dict(fields)
        return obj


class FixableError(RuntimeError):
    """The exception form: raised where a layer converts an error string into an exception."""

    def __init__(self, message: str, fields: Dict[str, Any]) -> None:
        super().__init__(str(message))
        self.fields = dict(fields)


def fix_message(msg: str, code: str, **kw: Any) -> FixMessage:
    return FixMessage(msg, fix_fields(code, **kw))


def fix_error(msg: str, code: str, **kw: Any) -> str:
    """``tool_error`` with the contract fields: the JSON string a tool returns."""
    from tools.registry import tool_error
    return tool_error(msg, **fix_fields(code, **kw))


def fields_of(err: Any) -> Dict[str, Any]:
    """The fix fields ``err`` carries (a ``FixMessage`` or ``FixableError``), else ``{}``."""
    fields = getattr(err, "fields", None)
    return dict(fields) if isinstance(fields, dict) and fields.get("code") in CODES else {}


def carried(result: Any) -> Dict[str, Any]:
    """The fix fields an error DICT carries (``{"success": False, "error": ..., "code": ...}``), for a
    layer that rebuilds its own error dict from a lower layer's; ``{}`` when it carries none."""
    if not isinstance(result, dict) or result.get("code") not in CODES:
        return {}
    return {k: v for k, v in result.items() if k not in ("success", "error", "data")}


def as_error(err: Any) -> Any:
    """Re-raiseable form of ``err``: a ``FixableError`` when it carries fields, else ``RuntimeError``."""
    fields = fields_of(err)
    return FixableError(str(err), fields) if fields else RuntimeError(str(err))


def as_tool_error(err: Any, **extra: Any) -> str:
    """Tool-boundary JSON for ``err``: the full contract when it carries fix fields, else plain."""
    from tools.registry import tool_error
    return tool_error(str(err), **{**extra, **fields_of(err)})


# ── Proof of origin: only Hermes's own code makes a fix ──────────────────────
#
# A host app turns ``fix`` into a card with an Open Settings button, so a code in a tool result is
# only believed when Hermes's own code built it. Parsing the text is not enough: anything the model
# or a web page can put in a tool result (a shell's stdout, a connector's answer, ``execute_code``'s
# output) can spell ``{"code": "tcc_app_data", ...}``. So ``fix_fields`` — which every producer goes
# through — signs the fields with a key the model cannot read, and ``host_fields`` believes a
# result only when that signature checks out against a key this process trusts.
#
# - In this process (the HTTP tool loop) the key is ``_PROCESS_KEY``: random, in memory, never in
#   the environment, trusted from the start.
# - Under the Claude Code runtime the tools run in the hermes-tools MCP server, a separate process.
#   Its ``ToolBridge`` (agent/transports/hermes_tool_bridge.py) mints a key, trusts it here, and
#   hands it to that server exactly once, at the server's startup, over the bridge socket; the
#   server signs with it (``use_signing_key``). The model can read the bridge's address and token
#   (terminal runs inside that server) but the key is already spent by then, and it is never in
#   an environment a child process inherits. A server that did not get it signs with its own
#   random key, which nobody trusts: no card, rather than a card anyone could make.
# - ``execute_code`` runs in its own process with its own random key: whatever it prints is
#   never believed.
# - A signature is accepted once (``_SEEN``): a genuine failure's JSON echoed back later — by a
#   shell, a file, a page — is not a second card.

MAC_KEY = "fix_mac"
NONCE_KEY = "fix_nonce"
#: What the signature covers. Anything else in the dict (a layer's own ``output``, ``exit_code``)
#: is not the fix's, and is never handed to a host.
_SIGNED_KEYS = ("code", "owner", "pane", "subject", "retry", "restart", "consent", "also_pane", "path",
                NONCE_KEY)
#: A subject is a name ("Google Chrome", "Desktop", "Notes"), never a sentence.
SUBJECT_LIMIT = 60
_PATH_LIMIT = 512

_PROCESS_KEY = secrets.token_bytes(32)
_lock = threading.Lock()
_signing: list = [_PROCESS_KEY]
_trusted: Dict[bytes, int] = {_PROCESS_KEY: 1}
_SEEN: "OrderedDict[str, None]" = OrderedDict()
_SEEN_LIMIT = 4096


def _signing_key() -> bytes:
    return _signing[0]


def use_signing_key(key: bytes) -> None:
    """Sign every fix from now on with ``key`` (the MCP server, with its bridge's key)."""
    if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
        raise ValueError("a fix signing key is at least 16 bytes")
    with _lock:
        _signing[0] = bytes(key)


def trust_key(key: bytes) -> None:
    """Believe fixes signed with ``key`` (a ``ToolBridge``, for the server it hands the key to)."""
    with _lock:
        _trusted[bytes(key)] = _trusted.get(bytes(key), 0) + 1


def untrust_key(key: bytes) -> None:
    with _lock:
        n = _trusted.get(bytes(key), 0) - 1
        if n > 0:
            _trusted[bytes(key)] = n
        elif bytes(key) != _PROCESS_KEY:
            _trusted.pop(bytes(key), None)


def _canonical(fields: Dict[str, Any]) -> bytes:
    return json.dumps({k: fields.get(k) for k in _SIGNED_KEYS if k in fields},
                      sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode()


def _mac(key: bytes, fields: Dict[str, Any]) -> str:
    return hmac.new(key, _canonical(fields), hashlib.sha256).hexdigest()


def _signed_by_trusted(fields: Dict[str, Any]) -> bool:
    mac = fields.get(MAC_KEY)
    if not isinstance(mac, str) or len(mac) != 64:
        return False
    with _lock:
        keys = list(_trusted)
    return any(hmac.compare_digest(mac, _mac(k, fields)) for k in keys)


def _decode(result: Any) -> Optional[Dict[str, Any]]:
    obj: Any = result
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode("utf-8", "replace")
    if isinstance(obj, str):
        text = obj.strip()
        if text.startswith("[error]"):
            text = text[len("[error]"):].strip()
        if not text.startswith("{"):
            return None
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return None
    return obj if isinstance(obj, dict) else None


def host_fields(result: Any) -> Dict[str, Any]:
    """The contract fields a FINISHED tool's result carries, for a host app's tool-progress frame; ``{}``
    unless Hermes's own code made them.

    ``result`` is whatever the tool loop handed ``tool_complete_callback``: the JSON string a tool
    returned (the HTTP loop), the same string with the ``"[error] "`` prefix the Claude Code child's
    transcript gets when a result is flagged ``is_error`` (``agent/transports/claude_code_session.py``),
    or a decoded dict. Believed only when (see the block above) its signature is from a trusted key and
    has not been seen before, and its fields pass ``fix_fields``'s own validation again (a known code;
    an owner and pane that agree with ``GRANTS`` / ``PANES``). ``error`` is left out: it is the model's
    words. ``subject`` is capped at ``SUBJECT_LIMIT``; only the signed keys are handed over."""
    obj = _decode(result)
    if obj is None or obj.get("code") not in CODES or not _signed_by_trusted(obj):
        return {}
    mac = obj[MAC_KEY]
    with _lock:
        if mac in _SEEN:
            return {}
        _SEEN[mac] = None
        while len(_SEEN) > _SEEN_LIMIT:
            _SEEN.popitem(last=False)
    signed = {k: obj[k] for k in _SIGNED_KEYS if k in obj and k != NONCE_KEY}
    try:
        extras = {k: v for k, v in signed.items() if k not in _CONTRACT_KEYS}
        fix_fields(signed["code"], owner=signed.get("owner"), pane=signed.get("pane"),
                   subject=signed.get("subject"), retry=bool(signed.get("retry")), **extras)
    except (ValueError, TypeError, KeyError):
        return {}
    for key in ("owner", "pane", "also_pane"):
        if key in signed and signed[key] is not None and not isinstance(signed[key], str):
            return {}
    if signed.get("also_pane") is not None and signed["also_pane"] not in PANES:
        return {}
    out: Dict[str, Any] = {}
    for key, value in signed.items():
        if isinstance(value, str):
            out[key] = value[:SUBJECT_LIMIT if key == "subject" else _PATH_LIMIT]
        elif value is None or isinstance(value, (bool, int, float)):
            out[key] = value
    return out
