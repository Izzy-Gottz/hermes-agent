"""The Chrome-extension lane for ``browser_exec``: Moe's own tab group in the owner's real Chrome.

``browser_exec`` normally drives Hermes' own browser (Chrome for Testing signed in
with a copy of the owner's cookies). Some sites fingerprint that browser and refuse
it — measured: Skyscanner (PerimeterX "press & hold"), Trip.com, x.com, and
Cloudflare's "Just a moment" on Product Hunt. For those, and whenever the owner asks
Moe to work in their Chrome, the same ``browser_exec`` code runs against the owner's
real Chrome through the Memoe extension (``extensions/chrome``) and its native host
(:mod:`tools.browser_chrome_bridge`), which exposes a CDP endpoint whose only pages
are the tabs in Moe's tab group.

This module holds the lane decision (:func:`choose_lane`), bot-block detection on
the own-browser lane (:func:`detect_block`), the bridge handshake file
(:func:`read_bridge`) and the one-step native-host install (:func:`install_native_host`).

Why this lane does not ride ``gateway.browser_control_broker``: that broker binds a
controller to ONE API-server session registered by the client that owns the session
(a dashboard hosting a chat), and routes the built-in ``browser_*`` tools. Moe's
turns arrive on many sessions (voice, channels, cron) and its model surface is
``browser_exec`` — so the extension is a long-lived controller of the owner's
browser, not of a session, and the lane is chosen per call. It keeps the broker's
rule, though: an explicit ``where="chrome"`` fails closed when the extension is not
connected, and never silently runs in a different browser.

``browser.chrome_extension.enabled`` is a feature switch, not a security gate: browser_exec runs
arbitrary Python as the user, and that code can read the bridge file and open the socket itself.
The boundary is what the socket will do (tools/browser_chrome_bridge.py: method allowlist, URL
policy) and what the extension will touch (only tabs in its own group). With the switch off the
native host serves no socket at all, so there is nothing to connect to.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

from tools.browser_chrome_bridge import NATIVE_HOST_NAME, bridge_state_dir

#: Stable id of the unpacked extension (derived from the public ``key`` in its manifest).
EXTENSION_ID = "ljblhmlmgflmbffbfelamefleikodmjh"
EXTENSION_DIR = Path(__file__).resolve().parent.parent / "extensions" / "chrome"

logger = logging.getLogger(__name__)

LANE_OWN = "own"
LANE_CHROME = "chrome"

#: Hosts measured to refuse the own-browser lane (Moe transcripts, 2026-09).
DEFAULT_BLOCKING_HOSTS: Tuple[str, ...] = (
    "skyscanner.com", "skyscanner.net", "trip.com", "x.com", "twitter.com",
    "producthunt.com", "theresanaiforthat.com",
)

#: Page-text hints of a bot wall: one table, owned by the CAPTCHA detector (tools/browser_captcha.py).
from tools.browser_captcha import BLOCK_SIGNATURES as _BLOCK_SIGNATURES  # noqa: E402

_URL_RE = re.compile(r"""https?://[^\s'"\\)<>]+""", re.I)

#: page_info() as the harness prints it (a dict repr) or as JSON: the page's OWN url and title.
_PAGE_INFO_RE = re.compile(r"""['"]url['"]\s*:\s*['"](?P<url>https?://[^'"]+)['"]\s*,\s*['"]title['"]\s*:\s*['"](?P<title>[^'"]*)['"]""")
#: Block-page titles that are the wall itself, not a page mentioning one (a leading tab-marker emoji is ignored).
_WALL_TITLES = (
    ("cloudflare", re.compile(r"^(?:\W+\s*)?(?:just a moment\.\.\.|attention required! \| cloudflare)$", re.I)),
    ("akamai", re.compile(r"^(?:\W+\s*)?access denied$", re.I)),
)
#: Wall URLs: the page itself was redirected to the vendor's challenge path.
_WALL_PATHS = (
    ("perimeterx", re.compile(r"/px/captcha|/sttc/px/", re.I)),
    ("cloudflare", re.compile(r"__cf_chl_|/cdn-cgi/challenge-platform/", re.I)),
)
#: How long a learned blocker is trusted before the own lane is tried again.
LEARNED_TTL_S = 14 * 24 * 3600
_sticky_lock = threading.Lock()
_sticky: Dict[Tuple[str, str], str] = {}


# ---- config -------------------------------------------------------------------

def chrome_extension_config(browser_cfg: Optional[dict]) -> dict:
    cfg = (browser_cfg or {}).get("chrome_extension") if isinstance(browser_cfg, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def lane_enabled(browser_cfg: Optional[dict]) -> bool:
    return chrome_extension_config(browser_cfg).get("enabled", False) is True


# ---- bridge handshake -----------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_bridge(hermes_home: Optional[str] = None) -> Optional[dict]:
    """The live bridge written by the native host, or None (absent, stale, or unsafe)."""
    path = bridge_state_dir(hermes_home) / "bridge.json"
    try:
        st = path.stat()
        if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) & 0o077:
            return None  # someone else's file, or readable by others: not ours to trust
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    url = str(data.get("ws_url") or "")
    pid = data.get("pid")
    if not url.startswith("ws://127.0.0.1:") or "/devtools/browser/" not in url:
        return None
    if type(pid) is not int or not _pid_alive(pid):
        return None
    origin = str(data.get("extension_origin") or "")
    if origin and origin.rstrip("/") != f"chrome-extension://{EXTENSION_ID}":
        return None
    return data


def bridge_warnings(bridge: Optional[dict]) -> list:
    """Things the app should tell the person about a connected extension (Connections → Chrome)."""
    ext = (bridge or {}).get("extension") or {}
    warnings = []
    if ext.get("fileAccess") is True:
        # Measured TRUE for an unpacked load. The URL policy refuses file: regardless; the switch
        # is still one fewer thing between a bug and the person's files.
        warnings.append({"code": "file_access_on", "text": "Moe's Chrome extension is allowed to open files on this Mac. "
                         "Moe never uses that; turn it off in Chrome › Extensions › Memoe › \"Allow access to file URLs\"."})
    return warnings


def not_connected_message() -> str:
    # Plain words for the person, ending the way every fixable message does (fix_reasons_macos.THEN).
    # retry=True in the contract: once the extension is installed and Chrome is open, the identical
    # call succeeds with nothing else changed (the extension reconnects to its host by itself).
    from tools.fix_reasons import host_app_name
    from tools.fix_reasons_macos import THEN
    app = host_app_name()
    return (f"Moe's extension for Google Chrome isn't connected, so nothing ran in your Chrome. To fix it, add it "
            f"from {app} › Connections › Chrome (or open Chrome if it's closed), {THEN}. "
            "(Moe can use where=\"own\" to work in its own browser instead.)")


def not_connected_error() -> str:
    """The fixable tool error (tools.fix_reasons contract) for an absent or unpaired extension.
    ``owner`` is the app (its install flow is the fix), ``pane`` stays null (no System Settings pane)."""
    from tools.fix_reasons import BROWSER_EXTENSION_MISSING, OWNER_APP, fix_error
    return fix_error(not_connected_message(), BROWSER_EXTENSION_MISSING, owner=OWNER_APP,
                     subject="Memoe extension for Google Chrome", retry=True,
                     extension_id=EXTENSION_ID, native_host=NATIVE_HOST_NAME)


# ---- who is here ------------------------------------------------------------------

#: Platforms where no person is at the other end (Hermes's own _UNATTENDED_APPROVAL_PLATFORMS, minus
#: api_server, which is decided per turn by the client's declared origin below).
_UNATTENDED_PLATFORMS = frozenset({"webhook", "msgraph_webhook"})
#: Surfaces a person drives directly at this machine.
_LOCAL_SURFACES = frozenset({"cli", "tui", "desktop", "local"})
#: Chat channels where the message IS the person writing to Moe, now. An ALLOWLIST: email, sms,
#: homeassistant, wecom_callback, relay and anything new are not live — an inbound email is somebody
#: else's words, and an unknown platform is not proof of a person.
_LIVE_CHAT_PLATFORMS = frozenset({
    "telegram", "whatsapp", "whatsapp_cloud", "signal", "discord", "slack", "matrix", "mattermost",
    "bluebubbles",
})


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def local_turn_presence() -> dict:
    """``{"live": bool, "why": str}`` for the turn whose contextvars are current.

    Live means a person started this turn themselves, now. Everything else is not:
    cron jobs, Kanban workers, webhooks, the Memoe app's own background notes (meeting notes carry
    other people's words; a URL in a transcript must not open a window on the person's screen) and
    any turn whose origin cannot be established. ``HERMES_SINGLE_QUERY_SESSION`` is deliberately NOT
    read: the claude-code MCP server sets it on every turn (it means "no approval prompt here").
    Must run in the process that owns the turn — see :func:`turn_presence`."""
    from gateway.session_context import TURN_ORIGIN_BACKGROUND, TURN_ORIGIN_PERSON, get_session_env, get_turn_origin

    if _truthy(get_session_env("HERMES_CRON_SESSION", "")):
        return {"live": False, "why": "a scheduled job"}
    try:
        from agent.delegation_context import is_delegated_child_context, is_detached_delegation_context
        if is_detached_delegation_context():
            # Inherits the starting turn's context but may finish long after it, with nobody there.
            return {"live": False, "why": "a background helper started earlier"}
        helper = is_delegated_child_context()
    except Exception:
        helper = False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return {"live": False, "why": "a background worker"}
    platform = str(get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower()
    source = str(get_session_env("HERMES_SESSION_SOURCE", "") or "").strip().lower()
    if platform in _UNATTENDED_PLATFORMS:
        return {"live": False, "why": f"a {platform.replace('_', ' ')} turn"}
    if platform == "api_server":
        origin = get_turn_origin()
        if origin == TURN_ORIGIN_PERSON:
            verdict = {"live": True, "why": "the person's own turn", "surface": SURFACE_LOCAL}
            return _as_helper(verdict) if helper else verdict
        if origin == TURN_ORIGIN_BACKGROUND:
            return {"live": False, "why": "a background note from the app"}
        return {"live": False, "why": "an API turn whose client did not say a person started it"}
    if platform in _LOCAL_SURFACES or (not platform and source in _LOCAL_SURFACES):
        verdict = {"live": True, "why": "the person's own turn", "surface": SURFACE_LOCAL}
    elif platform in _LIVE_CHAT_PLATFORMS:
        verdict = {"live": True, "why": f"the person's message on {platform.replace('_', ' ')}",
                   "surface": SURFACE_CHAT, "platform": platform}
    elif platform:
        return {"live": False, "why": f"a {platform.replace('_', ' ')} turn"}
    else:
        return {"live": False, "why": "a turn whose origin could not be established"}
    return _as_helper(verdict) if helper else verdict


def _as_helper(verdict: dict) -> dict:
    """A synchronous helper working inside a person's live turn: an explicit where="chrome" may run
    (the person is waiting on this very turn), but nothing is routed to their Chrome on a guess."""
    return {**verdict, "auto": False, "why": "a helper working on the person's turn"}


def _in_tools_mcp_server() -> bool:
    return bool(os.environ.get("HERMES_MCP_TOOL_PROFILE"))


def turn_presence() -> dict:
    """Presence for the current browser_exec call, judged where the turn really lives.

    Inside the hermes-tools MCP server (the claude-code runtime, where Moe runs) this process's
    environment was fixed at spawn and knows nothing about the current turn, so the question goes
    home over the tool bridge to the agent's process, which answers from the turn's contextvars.
    No bridge, or no answer, is "not live": the lane fails closed."""
    if not _in_tools_mcp_server():
        return local_turn_presence()
    try:
        from agent.transports.hermes_tool_bridge import TURN_PRESENCE_QUERY, bridge_available, call_bridged_tool
        if not bridge_available():
            return {"live": False, "why": "a turn whose origin could not be established (no bridge to the agent)"}
        answer = json.loads(call_bridged_tool(TURN_PRESENCE_QUERY, {}, timeout=15))
        if isinstance(answer, dict) and isinstance(answer.get("live"), bool):
            out = {"live": answer["live"], "why": str(answer.get("why") or "")}
            if answer.get("auto") is False:
                out["auto"] = False
            for key in ("surface", "platform"):
                if isinstance(answer.get(key), str):
                    out[key] = answer[key]
            return out
    except Exception as exc:
        logger.debug("turn presence query failed: %s", exc)
    return {"live": False, "why": "a turn whose origin could not be established (the agent did not answer)"}


#: Where a live turn's person is: at this machine's own surfaces, or writing from a chat app.
SURFACE_LOCAL = "local"
SURFACE_CHAT = "chat"
#: The host's "is the person at this machine" stamp (Memoe: ``Conductor.writePresence`` every
#: minute and on lock/unlock/sleep/wake: ``{"active": bool, "at": epoch, "idle": s}``; ``active`` is
#: already false when locked, asleep or idle). The same file and rule as Memoe's moe-screen
#: ``deliver.at_mac``: older than this, absent or unreadable is NOT here.
PRESENCE_FILENAME = "presence.json"
PRESENCE_FRESH_SECONDS = 3 * 60


def _presence_file() -> Optional[Path]:
    """The host's presence stamp: ``HERMES_PRESENCE_FILE``, else ``$MOE_HOME/presence.json``, else
    ``~/.moe/presence.json`` when that home exists. None: this host writes no stamp (plain Hermes)."""
    explicit = (os.environ.get("HERMES_PRESENCE_FILE") or "").strip()
    if explicit:
        return Path(explicit)
    home = (os.environ.get("MOE_HOME") or "").strip()
    if home:
        return Path(home) / PRESENCE_FILENAME
    legacy = Path.home() / ".moe"
    return legacy / PRESENCE_FILENAME if legacy.exists() else None


def presence_stamp_says_here(now: Optional[float] = None) -> Tuple[Optional[bool], str]:
    """``(True|False, why)`` from the host's stamp; ``(None, why)`` when the host writes none."""
    path = _presence_file()
    if path is None:
        return None, "this host keeps no presence stamp"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "the app has not said the person is at the Mac (no presence stamp)"
    if not isinstance(data, dict):
        return False, "the presence stamp is unreadable"
    at = data.get("at")
    now = time.time() if now is None else now
    if not isinstance(at, (int, float)) or not (0 <= now - float(at) <= PRESENCE_FRESH_SECONDS):
        return False, "the Mac has not been in use for a few minutes (the app is closed, or it is asleep)"
    if data.get("active") is not True:
        return False, "the Mac is locked, asleep or idle"
    return True, "the person is at the Mac"


def at_this_mac(presence: Optional[dict] = None, *, now: Optional[float] = None) -> Tuple[bool, str]:
    """Whether a window may be put in front of the person: a live turn from this machine's own
    surfaces (not a chat app -- the owner texting from their phone is live but not here) AND, when
    the host keeps a presence stamp, a fresh one saying unlocked, awake and in use."""
    presence = turn_presence() if presence is None else presence
    if not presence.get("live"):
        return False, presence.get("why") or "a turn nobody started live"
    if presence.get("surface") != SURFACE_LOCAL:
        return False, presence.get("why") or "a turn from away from the Mac"
    here, why = presence_stamp_says_here(now)
    if here is False:
        return False, why
    return True, "the person's own turn, at the Mac"


def unattended_turn() -> Optional[str]:
    """Why no person is present for this turn, or None when a person started it live."""
    p = turn_presence()
    return None if p["live"] else (p["why"] or "a turn with no person present")


# ---- lane decision ----------------------------------------------------------------

def hosts_in_code(code: str) -> list:
    hosts = []
    for m in _URL_RE.finditer(code or ""):
        host = (urlparse(m.group(0)).hostname or "").lower()
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _host_matches(host: str, patterns: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    for p in patterns:
        p = str(p or "").lower().lstrip(".").rstrip(".")
        if p and (host == p or host.endswith("." + p)):
            return True
    return False


def _learned_path(hermes_home: Optional[str] = None) -> Path:
    return bridge_state_dir(hermes_home) / "blocked-hosts.json"


def _learned(hermes_home: Optional[str] = None) -> Dict[str, float]:
    try:
        data = json.loads(_learned_path(hermes_home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}  # the pre-expiry list format is dropped, not trusted forever
    return {h: float(t) for h, t in data.items() if isinstance(h, str) and isinstance(t, (int, float))}


def learned_blocking_hosts(hermes_home: Optional[str] = None, *, now: Optional[float] = None) -> list:
    now = time.time() if now is None else now
    return sorted(h for h, exp in _learned(hermes_home).items() if exp > now)


def remember_blocking_host(host: str, hermes_home: Optional[str] = None, *, now: Optional[float] = None) -> None:
    host = (host or "").lower().rstrip(".")
    if not host:
        return
    now = time.time() if now is None else now
    entries = {h: exp for h, exp in _learned(hermes_home).items() if exp > now}
    entries[host] = now + LEARNED_TTL_S
    path = _learned_path(hermes_home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entries, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def blocking_hosts(browser_cfg: Optional[dict], hermes_home: Optional[str] = None) -> list:
    cfg = chrome_extension_config(browser_cfg)
    sites = cfg.get("sites")
    extra = [s for s in sites if isinstance(s, str)] if isinstance(sites, list) else []
    return list(DEFAULT_BLOCKING_HOSTS) + extra + learned_blocking_hosts(hermes_home)


def choose_lane(where: str, code: str, *, task_id: Optional[str], session: str, browser_cfg: Optional[dict],
                connected: bool, hermes_home: Optional[str] = None,
                presence: Optional[dict] = None) -> Tuple[str, str]:
    """Pick the lane for one browser_exec call: ``(lane, reason)``.

    * ``where="chrome"`` / ``"own"`` is the model's (or the owner's) explicit choice — final.
    * otherwise, with the lane enabled, auto-routing on and the extension connected, a call
      that opens a host known to block Moe's own browser goes to Chrome;
    * a call naming no URL stays in the lane its session used last (a ``page_info()``
      after ``new_tab`` in Chrome must read Chrome, not a blank own-browser tab).
    """
    where = (where or "").strip().lower()
    key = (str(task_id or ""), str(session or ""))
    if where in (LANE_OWN, LANE_CHROME):
        return where, "requested"
    if not lane_enabled(browser_cfg):
        return LANE_OWN, "lane disabled"
    presence = turn_presence() if presence is None else presence
    if not presence.get("live") or presence.get("auto") is False:
        # Not a person's own live turn (or a helper inside one): never open tabs in their Chrome on a guess.
        return LANE_OWN, f"{presence.get('why') or 'this turn'} never routes to the person's Chrome by itself"
    hosts = hosts_in_code(code)
    if not hosts:
        with _sticky_lock:
            sticky = _sticky.get(key)
        if sticky:
            return sticky, "same lane as this session's last call"
        return LANE_OWN, "default"
    auto = chrome_extension_config(browser_cfg).get("auto_route", True) is not False
    if auto and connected:
        blockers = blocking_hosts(browser_cfg, hermes_home)
        hit = next((h for h in hosts if _host_matches(h, blockers)), None)
        if hit:
            return LANE_CHROME, f"{hit} blocks Moe's own browser"
    return LANE_OWN, "default"


_lane_used_at: Dict[Tuple[str, str], float] = {}


def record_lane(task_id: Optional[str], session: str, lane: str) -> None:
    with _sticky_lock:
        key = (str(task_id or ""), str(session or ""))
        _sticky.pop(key, None)  # re-insert: the dict's order is then most-recent-last
        _sticky[key] = lane
        _lane_used_at[key] = time.time()


def recent_tasks(within_s: float, *, lane: str = LANE_OWN, now: Optional[float] = None) -> list:
    """Task ids whose browser_exec ran in ``lane`` within the last ``within_s`` seconds (this process)."""
    now = time.time() if now is None else now
    with _sticky_lock:
        return sorted({t for (t, s), at in _lane_used_at.items()
                       if now - at <= within_s and _sticky.get((t, s)) == lane})


def last_lane(task_id: Optional[str]) -> Optional[str]:
    """The lane this task's most recent browser_exec call ran in (any session), or None."""
    task = str(task_id or "")
    with _sticky_lock:
        for (t, _session), lane in reversed(list(_sticky.items())):
            if t == task:
                return lane
    return None


def reset_sticky_lanes() -> None:
    with _sticky_lock:
        _sticky.clear()
        _lane_used_at.clear()


def wall_host(text: str) -> Optional[Tuple[str, str]]:
    """``(label, host)`` when the output shows the page ITSELF is a bot wall — its own URL is a vendor
    challenge path, or its own title is a block page's — else None. Signature-grade only: this is what
    the router learns from, so a page that merely mentions "recaptcha" or "verify you are human", or
    a URL that merely appears in the code, teaches it nothing."""
    for m in _PAGE_INFO_RE.finditer(text or ""):
        url, title = m.group("url"), m.group("title")
        host = (urlparse(url).hostname or "").lower()
        if not host:
            continue
        for label, pattern in _WALL_PATHS:
            if pattern.search(url):
                return label, host
        for label, pattern in _WALL_TITLES:
            if pattern.match(title.strip()):
                return label, host
    return None


def detect_block(text: str) -> Optional[str]:
    """Label of the bot wall the page text suggests, or None (a hint for the model, never learned)."""
    wall = wall_host(text)
    if wall:
        return wall[0]
    if not text:
        return None
    for label, pattern in _BLOCK_SIGNATURES:
        if pattern.search(text):
            return label
    return None


def chrome_lane_env(env: dict, bridge: dict, session: str) -> dict:
    """Point the harness at the bridge: its own daemon name, never the own-browser endpoints."""
    for key in ("BU_CDP_URL", "BU_BROWSER_ID", "BU_AUTOSPAWN"):
        env.pop(key, None)
    env["BU_CDP_WS"] = bridge["ws_url"]
    # A distinct daemon per lane: the own-browser daemon keeps its tabs, this one opens
    # its own dedicated tab (named daemons never attach to a first page).
    env["BU_NAME"] = "chrome-" + (session or "default")
    return env


# ---- one-step install (called by the Memoe app's Connections → Chrome button) ----------

#: User-level NativeMessagingHosts dirs Chrome-family browsers read on macOS.
MAC_BROWSER_DIRS = {
    "chrome": "~/Library/Application Support/Google/Chrome",
    "chrome-beta": "~/Library/Application Support/Google/Chrome Beta",
    "chrome-canary": "~/Library/Application Support/Google/Chrome Canary",
    "chromium": "~/Library/Application Support/Chromium",
    "brave": "~/Library/Application Support/BraveSoftware/Brave-Browser",
    "edge": "~/Library/Application Support/Microsoft Edge",
}


def host_manifest(host_path: str, extension_id: str = EXTENSION_ID) -> dict:
    return {
        "name": NATIVE_HOST_NAME,
        "description": "Memoe: lets Moe work in its own tab group in this browser",
        "path": host_path,
        "type": "stdio",
        "allowed_origins": [f"chrome-extension://{extension_id}/"],
    }


def install_native_host(browser_dir: str, *, hermes_home: Optional[str] = None, python: Optional[str] = None,
                        extension_id: str = EXTENSION_ID) -> Tuple[Path, Path]:
    """Write the host launcher (in the Hermes home) and its manifest (in ``<browser_dir>/NativeMessagingHosts``).

    Writes one JSON file beside the browser's profiles and nothing inside any profile —
    it touches no preference, no extension list, no cookie. Returns (manifest, launcher)."""
    home = Path(hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    state_dir = bridge_state_dir(str(home))
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    launcher = state_dir / "memoe-chrome-host"
    bridge_py = Path(__file__).resolve().parent / "browser_chrome_bridge.py"
    py = python or sys.executable

    def q(s) -> str:
        return "'" + str(s).replace("'", "'\"'\"'") + "'"

    launcher.write_text(
        "#!/bin/sh\n# Memoe Chrome bridge: started by Chrome for the Memoe extension only.\n"
        f"HERMES_HOME={q(home)}\nexport HERMES_HOME\nunset PYTHONPATH PYTHONHOME\n"
        f"exec {q(py)} {q(bridge_py)} \"$@\"\n", encoding="utf-8")
    os.chmod(launcher, 0o700)
    target = Path(os.path.expanduser(browser_dir)) / "NativeMessagingHosts"
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / f"{NATIVE_HOST_NAME}.json"
    manifest.write_text(json.dumps(host_manifest(str(launcher), extension_id), indent=2) + "\n", encoding="utf-8")
    return manifest, launcher


def main(argv: Optional[list] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m tools.browser_chrome_extension")
    sub = ap.add_subparsers(dest="cmd", required=True)
    inst = sub.add_parser("install-host", help="write the native-messaging host for one browser")
    inst.add_argument("--browser", default="chrome", choices=sorted(MAC_BROWSER_DIRS))
    inst.add_argument("--browser-dir", help="explicit browser user-data dir (overrides --browser)")
    sub.add_parser("status", help="print whether the extension is connected")
    args = ap.parse_args(argv)
    if args.cmd == "install-host":
        manifest, launcher = install_native_host(args.browser_dir or MAC_BROWSER_DIRS[args.browser])
        print(json.dumps({"manifest": str(manifest), "launcher": str(launcher), "extension_id": EXTENSION_ID,
                          "extension_dir": str(EXTENSION_DIR)}))
        return 0
    bridge = read_bridge()
    print(json.dumps({"connected": bool(bridge), "extension": (bridge or {}).get("extension"),
                      "warnings": bridge_warnings(bridge)}))
    return 0 if bridge else 1


if __name__ == "__main__":
    sys.exit(main())
