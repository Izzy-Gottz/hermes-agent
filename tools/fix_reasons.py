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

import os
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

CODES = frozenset({
    TCC_APP_DATA, TCC_FILES, TCC_AUTOMATION, TCC_DRIVER_ACCESSIBILITY, TCC_DRIVER_SCREEN,
    TCC_MICROPHONE, TCC_CALENDAR, TCC_REMINDERS, TCC_CONTACTS,
    DRIVER_NOT_RUNNING, WINDOW_OTHER_SPACE, MCP_NEEDS_REAUTH, MCP_NOT_CONNECTED,
    CLAUDE_LOGIN_EXPIRED, CLAUDE_USAGE_CAP, BROWSER_MISSING, PROFILE_LOCKED, BROWSER_EXTENSION_MISSING,
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
    if clash := _CONTRACT_KEYS & extra.keys():
        raise ValueError(f"extras may not override contract keys: {sorted(clash)}")
    return {"code": code, "owner": owner, "pane": pane, "subject": subject, "retry": bool(retry), **extra}


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
