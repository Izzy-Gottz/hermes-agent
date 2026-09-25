"""Real-profile local browsing: snapshot the user's default Chromium profile into a
hermes-owned copy, hand its sign-ins to the browser Hermes drives, and attach agent-browser.

The browser Hermes DRIVES is agent-browser's own packaged Chrome for Testing (or Playwright's
Chromium) — never the person's installed browser application. The person's browser binary is
started only for a hand-over of a few hundred milliseconds: headless, on the copy, to read the
cookie jar its OS keychain protects, and it is terminated before the driven browser starts.

Why the person's browser must not be the driven one (measured on macOS 26, Chrome 153): a
second instance of ``/Applications/Google Chrome.app`` — even ``--headless=new`` with
``--no-startup-window`` — registers with Launch Services under Chrome's own bundle identifier.
Launch Services identifies an application by that identifier, so while the hidden instance
lived, the Dock, Finder and ``open`` routed "open Chrome" and "quit Chrome" to it: the person's
browser would not open and would not quit. Chrome for Testing carries its own identifier
(``com.google.chrome.for.testing``) and never collides with anything the person launches.

The driven browser is bound to the Hermes process that launched it: closed after the session
inactivity timeout, on exit, and — for a launch whose owner died uncleanly — by the orphan reaper
of the next Hermes process. It is never adopted "until the user closes it".

State (``_REAL_PROFILE_SESSION``, ``_real_profile_cdp_lock``, ``_real_profile_cdp_cache``,
``_real_profile_chrome_procs``, ``_real_profile_last_used``) lives in ``tools.browser_tool``; it
is read through ``_bt`` (resolved per call — never import ``tools.browser_tool`` at import time).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tools.browser_tool_origin import origin_module as _origin
from tools import browser_tool_cloud as _cloud
from tools import browser_tool_fidelity as _fidelity
from tools import browser_tool_install as _install
from tools import browser_tool_lightpanda_fallback as _lp
from tools import browser_tool_session as _session

_RP = "browser.use_real_profile is on, but "


def _terminate_real_profile_chrome() -> None:
    """Terminate browser processes launched for real-profile sessions (idempotent, atexit-safe);
    agent-browser only ATTACHED to them, so its own session cleanup never kills them."""
    from tools.browser_lightpanda import _terminate
    _bt = _origin()
    _fidelity.stop_keepers()
    while _bt._real_profile_chrome_procs:
        _terminate(_bt._real_profile_chrome_procs.pop(), what="real-profile chrome")


def _cdp_http_ready(http_cdp: str) -> bool:
    """True when an ``http://host:port`` CDP discovery root answers."""
    from tools.browser_lightpanda import _cdp_ready
    return _cdp_ready(http_cdp, timeout=1.0)


def _real_profile_daemon_env() -> dict:
    """Reaper-visible socket dir + ``owner_pid`` claim like every other lane (agent-browser's
    default dir is invisible to the reaper — #100855). The daemon-side idle timeout is dropped:
    the browser is launched by Hermes, not the daemon, so a self-exiting daemon would leave it
    holding the copy dir under the next snapshot overlay."""
    _bt = _origin()
    socket_dir = _session._prepare_session_socket_dir(_bt._REAL_PROFILE_SESSION)
    env = _session._agent_browser_command_env(socket_dir)
    env.pop("AGENT_BROWSER_IDLE_TIMEOUT_MS", None)
    return env


def _agent_browser_session_cmd(session_name: str, *cmd: str, log_label: str) -> Optional[subprocess.CompletedProcess]:
    """Run ``agent-browser --session <name> <cmd...>``; None when agent-browser is missing or the run fails."""
    _bt = _origin()
    try:
        browser_cmd = _install._find_agent_browser()
    except FileNotFoundError:
        return None
    try:
        return subprocess.run([*_session._agent_browser_argv(browser_cmd), "--session", session_name, *cmd],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
                              env=_real_profile_daemon_env(), stdin=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError) as e:
        _bt.logger.debug("real-profile %s failed: %s", log_label, e)
        return None


def _agent_browser_get_cdp(session_name: str) -> Optional[str]:
    """HTTP CDP discovery root of an agent-browser session (from its ``ws://`` cdp-url), or None."""
    proc = _agent_browser_session_cmd(session_name, "get", "cdp-url", log_label="get cdp-url")
    m = re.search(r"ws://127\.0\.0\.1:(\d+)/", (proc.stdout or "").strip()) if proc is not None else None
    return f"http://127.0.0.1:{m.group(1)}" if m else None


def _read_devtools_port(data_dir: str) -> Optional[str]:
    """First line of Chrome's ``DevToolsActivePort`` in ``data_dir`` (None when unreadable)."""
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            return fh.readline().strip()
    except OSError:
        return None


# How long a browser still running on the copy dir may take to answer DevTools before the
# acquire gives up (without touching it or its profile).
_BUSY_BROWSER_WAIT_S = 20.0
def _await_surviving_chrome_cdp(data_dir: str, wait: Optional[float] = None) -> Optional[str]:
    """Poll :func:`_surviving_chrome_cdp` until it answers, the browser exits, or ``wait`` runs out."""
    deadline = time.monotonic() + (_BUSY_BROWSER_WAIT_S if wait is None else wait)
    while True:
        cdp = _surviving_chrome_cdp(data_dir)
        if cdp or not _live_holders(data_dir) or time.monotonic() >= deadline:
            return cdp
        time.sleep(0.5)


# Told to the model on the next browser_exec result after a wedged own browser was restarted:
# its tabs are gone, and a form it submitted may already have gone through.
RESTART_NOTE = ("the browser stopped answering and was restarted; pages you had open are gone, "
                "so check before resubmitting anything")
_restart_notes: List[str] = []


def take_restart_note() -> Optional[str]:
    """What the model must know about the browser since the last call, once: the restart note, and
    the note that a sign-in refresh was skipped. None when there is nothing to tell."""
    notes = list(dict.fromkeys(_restart_notes))
    _restart_notes.clear()
    return "; ".join(notes) or None


def _await_holders_gone(data_dir: str, wait: float = 10.0) -> None:
    """Poll until no live browser holds ``data_dir`` (a terminated one takes a moment to exit)."""
    deadline = time.monotonic() + wait
    while _live_holders(data_dir) and time.monotonic() < deadline:
        time.sleep(0.25)


def _surviving_chrome_cdp(data_dir: str) -> Optional[str]:
    """HTTP CDP root of a browser still running on ``data_dir``, or None. ``DevToolsActivePort``
    outlives a crashed browser and its port can be recycled by another local CDP server, so the
    file's browser id (line 2) must match what ``/json/version`` reports before it is trusted."""
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            port, browser_path = fh.readline().strip(), fh.readline().strip()
    except OSError:
        return None
    if not port.isdigit() or not browser_path.startswith("/devtools/browser/"):
        return None
    http_cdp = f"http://127.0.0.1:{port}"
    try:
        import requests
        ws_url = str(requests.get(f"{http_cdp}/json/version", timeout=2).json().get("webSocketDebuggerUrl") or "")
    except Exception:
        return None
    return http_cdp if ws_url.endswith(browser_path) else None


def _cdp_on_data_dir(http_cdp: str, data_dir: str) -> bool:
    """True when the CDP endpoint's browser runs on ``data_dir`` (DevToolsActivePort match proves it
    is our profile copy, not a throwaway temp dir a raced/stale launch fell back to)."""
    m = re.search(r":(\d+)", http_cdp or "")
    return bool(m) and _read_devtools_port(data_dir) == m.group(1)


def _agent_browser_close_session(session_name: str) -> None:
    """Best-effort close of an agent-browser session (stale/wrong-dir cleanup)."""
    _agent_browser_session_cmd(session_name, "close", log_label="session close")


_REAL_PROFILE_CHROME_FLAGS = (
    "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check",
    "--disable-background-networking", "--disable-component-update", "--disable-default-apps",
    "--disable-hang-monitor", "--disable-popup-blocking", "--disable-prompt-on-repost",
    "--disable-sync", "--disable-features=Translate", "--no-startup-window",
)

# The auth stores the snapshot copies are encrypted under the PERSON's browser keychain entry.
# After the hand-over the driven browser (mock keychain) cannot read them, so they are removed:
# an unreadable copy of someone's saved passwords is a liability, not a feature.
_KEYCHAIN_BOUND_AUTH_FILES = (
    "Cookies", os.path.join("Network", "Cookies"), "Login Data", "Login Data For Account", "Web Data",
)

# Fields of a CDP ``Network.Cookie`` (Storage.getCookies) that ``Network.CookieParam``
# (Storage.setCookies) accepts. ``session``/``size``/``partitionKeyOpaque`` are read-only.
_COOKIE_PARAM_KEYS = ("name", "value", "domain", "path", "secure", "httpOnly", "sameSite", "priority",
                      "sameParty", "sourceScheme", "sourcePort", "partitionKey")
_COOKIE_IMPORT_CHUNK = 300


def _real_profile_unsupported_reason(browser) -> Optional[str]:
    """Fail-closed message when the default browser can't be used, else None.

    A pre-release channel lives in a profile dir we don't resolve; normalizing to the stable
    family would drive a DIFFERENT profile/account (wrong-principal bug), so refuse rather than guess.
    """
    from hermes_cli.browser_connect import UNSUPPORTED_CHANNEL
    if browser is None:
        return (_RP + "your default browser is not a supported Chromium browser (Chrome, Edge, Brave, "
                "Brave Origin, Chromium). Real-profile browsing requires a Chromium default; set one or turn the toggle off.")
    if browser == UNSUPPORTED_CHANNEL:
        return (_RP + "your default browser is a pre-release Chromium channel (Beta / Dev / Canary), which "
                "real-profile browsing does not support. Set your default to a "
                "stable Chrome / Edge / Brave / Brave Origin / Chromium, or turn the toggle off.")
    return None


def _real_profile_snapshot_error(err: str) -> str:
    """User-facing message for a failed profile snapshot. A fixable cause (``tools.fix_reasons``)
    passes through unchanged, fields and all: it is already in the person's words, and prefixing it
    would drop the code. A locked profile (Windows only — nowhere else can a browser lock its files
    against a reader) adds the approved-close command, which the agent must ASK the user about
    first (it quits their browser)."""
    from hermes_cli.browser_connect import _PROFILE_LOCKED_PREFIX
    from tools.fix_reasons import FixMessage, PROFILE_LOCKED, fields_of
    fields = fields_of(err)
    if err and err.startswith(_PROFILE_LOCKED_PREFIX):
        msg = (err[len(_PROFILE_LOCKED_PREFIX):] + " To close it (only after the user approves — it "
               "quits their browser and loses unsaved tabs), run: `hermes browser close-profile`, then retry.")
        return FixMessage(msg, fields) if fields.get("code") == PROFILE_LOCKED else msg
    if fields:
        return err
    return f"{_RP}{err}"


# ---------------------------------------------------------------------------
# The browser Hermes drives
# ---------------------------------------------------------------------------

def _mock_keychain_flags() -> Tuple[str, ...]:
    """The driven browser keeps its cookie store under a fixed, browser-local key — the same
    switches agent-browser's own launch passes — so it never touches the OS keychain (no prompt,
    no access to the person's browser entry)."""
    if sys.platform == "darwin":
        return ("--use-mock-keychain",)
    if sys.platform.startswith("linux"):
        return ("--password-store=basic",)
    return ()


def _version_key(entry: str) -> Tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", entry))


def _driven_browser_candidates(root: str, entry: str) -> List[str]:
    """Executable paths a packaged download ``root/entry`` may hold, per platform. agent-browser's
    Chrome for Testing (``chrome-<ver>``, measured on macOS: ``Google Chrome for Testing.app``) and
    Playwright's Chromium (``chromium-<build>``). Headless-shell builds are skipped: legacy headless
    keeps no persistent cookie store, so a hand-over into it would be lost on the next launch."""
    base = os.path.join(root, entry)
    if sys.platform == "darwin":
        rels = ("Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                "chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
                "chrome-mac-arm64/Chromium.app/Contents/MacOS/Chromium")
    elif sys.platform == "win32":
        rels = ("chrome-win64/chrome.exe", "chrome-win/chrome.exe", "chrome.exe")
    else:
        rels = ("chrome-linux64/chrome", "chrome-linux/chrome", "chrome")
    return [os.path.join(base, rel) for rel in rels]


def driven_browser_executable() -> Optional[str]:
    """Path of the browser Hermes drives, or None when no packaged build is on disk.

    ``AGENT_BROWSER_EXECUTABLE_PATH`` wins when it names a file. Otherwise the newest packaged
    build under agent-browser's / Playwright's download roots. The person's installed browser
    application is never a candidate — see the module docstring for why.
    """
    override = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    if override and os.path.isfile(override):
        return override
    for root in _install._chromium_search_roots():
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = [e for e in os.listdir(root) if e.startswith(("chrome-", "chromium-"))]
        except OSError:
            continue
        for entry in sorted(entries, key=_version_key, reverse=True):
            for candidate in _driven_browser_candidates(root, entry):
                if os.path.isfile(candidate):
                    return candidate
    return None


def _terminate_one(proc, what: str) -> None:
    """Terminate ``proc`` alone and forget it -- never the other browsers this process runs (a sign-in
    refresh starts the person's browser while the driven one is live, and must not take it down)."""
    from tools.browser_lightpanda import _terminate
    try:
        _origin()._real_profile_chrome_procs.remove(proc)
    except ValueError:
        pass
    _terminate(proc, what=what)


def _spawn_browser_on_copy(binary: str, copy_dir: str, extra_flags: Iterable[str], what: str,
                           headless: bool, startup_wait: float = 30.0,
                           ) -> Tuple[Optional[subprocess.Popen], Optional[int], Optional[str]]:
    """Launch ``binary`` on the profile COPY and wait up to ``startup_wait`` s for its debug port:
    ``(proc, port, error)``.

    The process is recorded in ``_real_profile_chrome_procs`` so exit/idle/atexit cleanup can reach
    it. On failure it (and only it) is terminated here and ``(None, None, error)`` is returned.
    """
    _bt = _origin()
    try:
        os.unlink(os.path.join(copy_dir, "DevToolsActivePort"))  # stale port confuses reuse probes
    except OSError:
        pass
    argv = [binary, f"--user-data-dir={copy_dir}", *_REAL_PROFILE_CHROME_FLAGS, *extra_flags]
    if headless:
        argv.append("--headless=new")
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                stdin=subprocess.DEVNULL, start_new_session=True, env=_bt._build_browser_env())
    except (subprocess.SubprocessError, OSError) as e:
        return None, None, f"{_RP}the {what} launch failed: {e}"
    _bt._real_profile_chrome_procs.append(proc)

    deadline = time.monotonic() + startup_wait
    while time.monotonic() < deadline:
        line = _read_devtools_port(copy_dir) or ""
        if line.isdigit():
            return proc, int(line), None
        if proc.poll() is not None:
            _terminate_one(proc, what)
            return None, None, _RP + f"the {what} exited during startup (another instance may hold the profile copy)."
        time.sleep(0.25)
    _terminate_one(proc, what)
    return None, None, _RP + f"the {what} did not expose a debug port in time. Retry, or turn the toggle off."


#: True while the driven browser is shown to the person (browser_handoff): a relaunch in that window
#: (a wedge restart, say) must not hide the page they are working in. Cleared on hand-back and when
#: the browser is released, so the next launch is headless again.
_shown_to_person: Dict[str, Any] = {"on": False, "since": 0.0}
#: Whether the browser THIS process launched is headless (None: not launched here, e.g. re-attached).
_launched_headless: Dict[str, Optional[bool]] = {"headless": None}


def _driven_browser_headless() -> bool:
    """Headless by default (a focus-stealing window defeats a background capability); Chrome's NEW
    headless shares the profile's cookie store (legacy --headless does not). browser.headed /
    AGENT_BROWSER_HEADED opts into a window, except on a display-less Linux host (launch would die).
    While the page is handed to the person (``_shown_to_person``) the browser is headed."""
    if _shown_to_person["on"]:
        return False
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return not (_cloud._is_headed_mode() and (has_display or not sys.platform.startswith("linux")))


def _driven_browser_flags(identity: Optional[Dict[str, Any]], headless: bool) -> Tuple[str, ...]:
    """Switches beyond the shared set: the mock keychain, plus (``browser.stealth_fidelity``, on
    by default) the truthful identity in ``tools.browser_tool_fidelity``."""
    flags = list(_mock_keychain_flags())
    if _fidelity.fidelity_enabled():
        flags += _fidelity.launch_flags(identity, headless, _fidelity.main_display() if headless else None)
    return tuple(flags)


def _persons_identity(browser: Optional[str]) -> Optional[Dict[str, Any]]:
    """The person's browser's brand and version (from its Info.plist; the engine's version when
    the majors differ), or None."""
    from hermes_cli.browser_connect import chromium_executable
    try:
        driven = driven_browser_executable()
        return _fidelity.persons_browser_identity(browser, chromium_executable(browser) if browser else None,
                                                  _fidelity.installed_browser_version(driven) if driven else None)
    except Exception as e:
        _origin().logger.debug("real-profile: browser identity unreadable: %s", e)
        return None


def _launch_driven_browser(binary: str, copy_dir: str,
                           identity: Optional[Dict[str, Any]] = None) -> Tuple[Optional[int], Optional[str]]:
    """Launch the browser Hermes drives on the copy; ``(debug_port, error)``. agent-browser then
    attaches via ``--cdp <port>`` (its own launch would pick a throwaway profile). The fidelity
    keeper attaches before anything else does, so the first tab already carries the brands."""
    headless = _driven_browser_headless()
    _, port, err = _spawn_browser_on_copy(binary, copy_dir, _driven_browser_flags(identity, headless), "browser",
                                          headless=headless)
    if port is not None:
        _launched_headless["headless"] = headless
        _fidelity.ensure_keeper(port, identity)
    return port, err


# ---------------------------------------------------------------------------
# The hand-over: the person's browser reads its own cookie jar, briefly
# ---------------------------------------------------------------------------

def _cdp_call(port: int, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 20.0) -> Dict[str, Any]:
    """One browser-level CDP call over the debug port; returns the ``result`` dict. Raises on error."""
    import requests
    from websockets.sync.client import connect
    ws_url = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=3).json()["webSocketDebuggerUrl"]
    with connect(ws_url, max_size=None, open_timeout=timeout, close_timeout=2) as ws:
        ws.send(json.dumps({"id": 1, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            msg = json.loads(ws.recv(timeout=max(0.1, deadline - time.monotonic())))
            if msg.get("id") != 1:
                continue
            if "error" in msg:
                raise RuntimeError(f"{method}: {msg['error'].get('message', msg['error'])}")
            return msg.get("result") or {}
    raise TimeoutError(f"{method}: no reply in {timeout:.0f}s")


def _cookie_param(cookie: Dict[str, Any]) -> Dict[str, Any]:
    """``Network.Cookie`` -> ``Network.CookieParam``: keep the settable fields; session cookies
    carry no ``expires`` (getCookies reports -1, which setCookies rejects)."""
    param = {k: cookie[k] for k in _COOKIE_PARAM_KEYS if k in cookie}
    if not cookie.get("session") and float(cookie.get("expires", -1) or -1) > 0:
        param["expires"] = cookie["expires"]
    return param


def _export_cookies_from_persons_browser(real_binary: str, copy_dir: str, budget: Optional[float] = None,
                                         ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Start the person's browser binary headless on the copy, read every cookie its keychain
    entry decrypts, and terminate it. ``(cookies, None)`` or ``(None, error)``. ``budget`` bounds the
    whole read (startup + read) in seconds; None keeps the launch defaults.

    This is the only moment the person's browser application runs for Hermes. It is headless,
    opens no window, and lives for the read (sub-second, measured) — but while it lives Launch
    Services counts it as that application, so it is never left running past this function. Only
    THIS process is terminated: a refresh runs while the driven browser is live.
    """
    _bt = _origin()
    deadline = None if budget is None else time.monotonic() + budget
    proc, port, err = _spawn_browser_on_copy(real_binary, copy_dir, (), "hand-over browser", headless=True,
                                             startup_wait=30.0 if budget is None else budget)
    if proc is None or port is None:
        return None, err
    try:
        timeout = 20.0 if deadline is None else max(0.5, deadline - time.monotonic())
        cookies = list(_cdp_call(port, "Storage.getCookies", timeout=timeout).get("cookies") or [])
    except Exception as e:
        return None, f"{_RP}the sign-in hand-over failed to read the cookie jar: {e}"
    finally:
        _terminate_one(proc, "hand-over browser")
    _bt.logger.info("real-profile: hand-over read %d cookie(s) from %s", len(cookies), os.path.basename(real_binary))
    return cookies, None


def _forget_keychain_bound_auth_files(copy_dir: str) -> None:
    """Remove the copy's keychain-encrypted auth stores after the hand-over (see the constant)."""
    for rel in _KEYCHAIN_BOUND_AUTH_FILES:
        for path in (os.path.join(copy_dir, "Default", rel), os.path.join(copy_dir, "Default", rel + "-journal")):
            try:
                os.unlink(path)
            except OSError:
                pass


def _import_cookies_into_driven_browser(port: int, cookies: List[Dict[str, Any]],
                                        budget: Optional[float] = None) -> Optional[str]:
    """``Storage.setCookies`` in chunks; None on success, else an error message. ``setCookies``
    overwrites a cookie of the same name/domain/path, so loading into a live browser is safe."""
    params = [_cookie_param(c) for c in cookies]
    deadline = None if budget is None else time.monotonic() + budget
    try:
        for i in range(0, len(params), _COOKIE_IMPORT_CHUNK):
            if deadline is None:
                _cdp_call(port, "Storage.setCookies", {"cookies": params[i:i + _COOKIE_IMPORT_CHUNK]})
                continue
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"out of time after {i} of {len(params)} cookie(s)")
            _cdp_call(port, "Storage.setCookies", {"cookies": params[i:i + _COOKIE_IMPORT_CHUNK]}, timeout=left)
    except Exception as e:
        return f"{_RP}the sign-in hand-over failed to load the cookie jar: {e}"
    return None


# ---------------------------------------------------------------------------
# Keeping the jar current: a cookie-only hand-over into the RUNNING driven browser
# ---------------------------------------------------------------------------
# Ticket #18: the hand-over ran only when the driven browser was launched, so a sign-in the person
# made in their own browser afterwards never reached it -- one hand-over at 21:25, a sign-in at
# ~21:31, and calls at 21:33 reused the 21:25 jar. A reused browser now re-takes the jar when the
# person's cookie file has changed since the last hand-over -- and ONLY then (or on the first call into
# a browser whose jar's age is unknown). No timer: every refresh starts a hidden second instance of the
# person's own browser, and on 2026-09-20 one hijacked that browser's Launch Services identity. Once per
# launch is accepted; once a minute of browsing when nothing changed is not. Chrome writes its cookie
# file lazily, so a sign-in can take a little while to show. Google account sessions are the exception
# no refresh fixes: see GOOGLE_SESSION_NOTE.

#: Never more often than this, however many calls arrive: a burst of browser_exec calls must not start
#: the person's browser for every one.
_COOKIE_REFRESH_MIN_INTERVAL_S = 15.0
#: The whole refresh (copy the jar, the person's browser reads it, load it) gets this long; past it the
#: call goes ahead with the jar the driven browser already has, and the model is told.
_COOKIE_REFRESH_BUDGET_S = 8.0
_COOKIE_DB_RELS = ("Cookies", os.path.join("Network", "Cookies"))

#: The browser the last hand-over went into (its CDP root), the person's browser it came from, when it
#: was taken (wall clock, compared with the cookie file's mtime) and when a refresh was last tried.
_handover: Dict[str, Any] = {"cdp": "", "browser": "", "at": 0.0, "attempt": 0.0}

REFRESH_SKIPPED_NOTE = ("sign-ins the person made in their own browser in the last minute or so may not have "
                        "reached Moe's browser yet (bringing them over did not finish this time); if a site "
                        "shows them signed out, try again in a moment")

#: Said to the model (tool description, and a result that lands on a Google sign-in wall) in words it can
#: pass on. Measured on ticket #18: every Google session cookie was handed over and sent, and Google still
#: served the signed-out page and asked for the passkey -- it binds the session to the person's own
#: browser. Copying cookies cannot fix that, and nothing here tries to get around it.
GOOGLE_SESSION_NOTE = ("Google account sign-ins do not carry over into Moe's browser: Google ties them to the "
                       "person's own Chrome, so signing in to Google there asks for their passkey even when they "
                       "are signed in on their own Chrome. Tell them that plainly. To go on, hand the page to them "
                       "with browser_handoff so they can sign in themselves, or work in their own Chrome through "
                       "the Memoe extension (where=\"chrome\") if it is set up.")


def _record_handover_target(cdp: str, browser: str, at: float = 0.0) -> None:
    """Remember which browser the jar is kept current in. ``at`` is when a hand-over into it was taken;
    a browser this process did not hand over to (re-attached, another process's) starts at 0 -- its
    jar's age is unknown, so the next acquire refreshes it."""
    if at or _handover.get("cdp") != cdp or _handover.get("browser") != browser:
        _handover.update(cdp=cdp, browser=browser, at=at, attempt=at)


def _persons_jar(browser: str) -> Tuple[Optional[str], Optional[str]]:
    """``(user-data dir, profile dir name)`` of the person's browser -- the pinned profile when
    ``browser.real_profile_pin`` is set, as the snapshot uses -- or ``(None, None)``."""
    from hermes_cli.browser_connect import _resolve_source_profile, real_profile_data_dir
    src = real_profile_data_dir(browser)
    if not src or not os.path.isdir(src):
        return None, None
    profile, err = _resolve_source_profile(src)
    return (src, profile) if profile and not err else (None, None)


def _jar_mtime(src: str, profile: str) -> float:
    """Newest mtime of the person's cookie DBs (and their sqlite sidecars), 0 when none is readable."""
    newest = 0.0
    for rel in _COOKIE_DB_RELS:
        for suffix in ("", "-journal", "-wal"):
            try:
                newest = max(newest, os.path.getmtime(os.path.join(src, profile, rel + suffix)))
            except OSError:
                pass
    return newest


def _snapshot_jar(src: str, profile: str, dst: str) -> Optional[str]:
    """Copy ONLY what the export needs -- ``Local State`` and the cookie DBs -- into ``dst``, a fresh
    temp dir (never the live copy dir a running browser holds). The DBs go through the same sqlite
    online backup the snapshot uses, which folds in what the person's browser committed (its
    ``Cookies-journal``); a raw journal is never copied (a stale one corrupts the copy)."""
    from hermes_cli.browser_connect import _copy_auth_file, _sync_local_state
    _sync_local_state(src, dst, profile)
    copied = 0
    for rel in _COOKIE_DB_RELS:
        s = os.path.join(src, profile, rel)
        if os.path.isfile(s):
            copied += bool(_copy_auth_file(s, os.path.join(dst, "Default", rel)))
    return None if copied else "the person's cookie jar could not be copied (their browser was writing it)"


def _refresh_jar_now(port: int, browser: str, now: float) -> Optional[str]:
    """The cookie-only hand-over into the live browser on ``port``; None when done, else why not."""
    from hermes_cli.browser_connect import chromium_executable, detect_default_chromium, get_hermes_home
    current = detect_default_chromium()
    if current != browser:
        # Never a hand-over from another browser into this one -- and a switch to a pre-release channel
        # (Beta / Dev / Canary) is refused here exactly as at launch.
        return (_real_profile_unsupported_reason(current)
                or f"the default browser changed ({browser} -> {current}); the next launch hands over from it")
    src, profile = _persons_jar(browser)
    real_binary = chromium_executable(browser)
    if not src or not profile or not real_binary:
        return "the person's browser profile or application could not be found"
    deadline = time.monotonic() + _COOKIE_REFRESH_BUDGET_S
    root = os.path.join(str(get_hermes_home()), "browser-profile")
    os.makedirs(root, mode=0o700, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=".cookie-refresh-", dir=root)  # 0700; the orphan reaper covers it
    try:
        err = _snapshot_jar(src, profile, tmp)
        if err:
            return err
        left = deadline - time.monotonic()
        if left <= 0:
            return f"copying the jar used the whole {_COOKIE_REFRESH_BUDGET_S:.0f}s budget"
        cookies, err = _export_cookies_from_persons_browser(real_binary, tmp, budget=left)
        if cookies is None:
            return err
        left = deadline - time.monotonic()
        if left <= 0:
            return f"reading the jar used the whole {_COOKIE_REFRESH_BUDGET_S:.0f}s budget"
        err = _import_cookies_into_driven_browser(port, cookies, budget=left)
        if err:
            return err
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _handover["at"] = now
    _origin().logger.info("real-profile: sign-ins refreshed into the running browser (%d cookie(s))", len(cookies))
    return None


def _refresh_jar_if_due(cdp: str) -> None:
    """Re-take the person's cookies into the reused driven browser at ``cdp`` when they may have changed
    (see the section comment). Rate-limited and bounded; a skip never fails the call -- it is logged and
    the model is told (``take_restart_note``). Never raises."""
    if not cdp or _handover.get("cdp") != cdp or not _handover.get("browser"):
        return  # not a browser whose jar this process keeps current
    now = time.time()
    if now - float(_handover.get("attempt") or 0) < _COOKIE_REFRESH_MIN_INTERVAL_S:
        return
    browser = str(_handover["browser"])
    try:
        src, profile = _persons_jar(browser)
        if not src or not profile or _jar_mtime(src, profile) <= float(_handover.get("at") or 0):
            return  # nothing new since the last hand-over: never start the person's browser for nothing
        _handover["attempt"] = now
        why = _refresh_jar_now(int(cdp.rsplit(":", 1)[1]), browser, now)
    except Exception as e:
        _handover["attempt"] = now
        why = f"unexpected error: {e}"
    if why:
        _origin().logger.warning("real-profile: sign-in refresh skipped; the browser keeps the sign-ins it has: %s", why)
        if REFRESH_SKIPPED_NOTE not in _restart_notes:
            _restart_notes.append(REFRESH_SKIPPED_NOTE)


# ---------------------------------------------------------------------------
# Lifecycle: bound to the Hermes process that launched it
# ---------------------------------------------------------------------------

def _browsers_on_data_dir(data_dir: str):
    """Chromium-family processes whose command line binds THIS ``--user-data-dir`` (fail-closed on
    an unreadable cmdline), reusing the profile-holder matcher from ``browser_connect``."""
    from hermes_cli.browser_connect import _processes_holding_profile
    return list(_processes_holding_profile(data_dir) or [])


def _is_orphan(proc) -> bool:
    """True when the process that launched ``proc`` is gone (re-parented to pid 1, or parent absent).
    The driven browser starts in its own session, so a dead owner is visible only this way."""
    try:
        import psutil
        ppid = proc.ppid()
        return ppid <= 1 or not psutil.pid_exists(ppid)
    except Exception:
        return False


def _terminate_processes(procs, what: str) -> int:
    """Terminate ``procs`` and their trees (graceful, then kill); returns how many were signalled."""
    try:
        import psutil
    except ImportError:
        return 0
    targets = []
    for p in procs:
        targets.append(p)
        try:
            targets.extend(p.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    for p in targets:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(targets, timeout=5.0)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if alive:
        psutil.wait_procs(alive, timeout=3.0)
    if targets:
        _origin().logger.info("real-profile: terminated %d %s process(es)", len(targets), what)
    return len(targets)


def _own_browser_pids() -> set:
    _bt = _origin()
    return {p.pid for p in _bt._real_profile_chrome_procs if p.poll() is None}


def _live_holders(copy_dir: str) -> list:
    """Browser processes on ``copy_dir`` whose owner is alive (ours, or another Hermes's): the ones
    an orphan sweep leaves running, so the ones a snapshot overlay must never run under."""
    own = _own_browser_pids()
    return [p for p in _browsers_on_data_dir(copy_dir) if p.pid in own or not _is_orphan(p)]


def _terminate_orphaned_browsers_on_dir(copy_dir: str) -> int:
    """Terminate browsers holding ``copy_dir`` whose launching Hermes died. A holder whose owner is
    alive (another Hermes process sharing this home) is left alone."""
    own = _own_browser_pids()
    orphans = [p for p in _browsers_on_data_dir(copy_dir) if p.pid not in own and _is_orphan(p)]
    return _terminate_processes(orphans, "orphaned driven-browser") if orphans else 0


def reap_orphaned_driven_browsers() -> int:
    """Startup / periodic reap: every driven browser under this home's ``browser-profile`` root whose
    owner is gone. Identity is the ``--user-data-dir`` under OUR root — never a name match, so the
    person's own browser (no such argument) can never qualify."""
    from hermes_cli.browser_connect import get_hermes_home
    root = str(get_hermes_home() / "browser-profile")
    try:
        entries = os.listdir(root)
    except OSError:
        return 0
    reaped = 0
    for entry in entries:
        copy_dir = os.path.join(root, entry)
        if os.path.isdir(copy_dir):
            reaped += _terminate_orphaned_browsers_on_dir(copy_dir)
    return reaped


def _real_profile_in_use() -> bool:
    """True while any task session drives the shared real-profile browser."""
    _bt = _origin()
    with _bt._cleanup_lock:
        return any((info.get("features") or {}).get("real_profile") for info in _bt._active_sessions.values())


def release_if_idle(now: Optional[float] = None) -> bool:
    """Close the driven browser once no task session uses it and the inactivity timeout has passed
    since it was last requested. Runs from the cleanup thread; True when it released something."""
    _bt = _origin()
    now = time.time() if now is None else now
    have_browser = bool(_bt._real_profile_cdp_cache.get("cdp")) or bool(_own_browser_pids())
    if not have_browser:
        return False
    if now - _bt._real_profile_last_used <= _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT or _real_profile_in_use():
        return False
    if _shown_to_person["on"] and now - float(_shown_to_person.get("since") or 0) < SHOWN_MAX_SECONDS:
        # The person is working in this window (browser_handoff) and has not said "done": closing it
        # would take the page out from under them.
        return False
    with _bt._real_profile_cdp_lock:
        _bt._real_profile_cdp_cache.pop("cdp", None)
        _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)
        _terminate_real_profile_chrome()
        _shown_to_person["on"] = False  # the next launch is out of sight again
        _launched_headless["headless"] = None
    _bt.logger.info("real-profile: driven browser released after %ds idle", _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT)
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _attach_agent_browser_to_real_profile(port: int, copy_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Make agent-browser ATTACH to the running browser (never launch its own); returns ``(http_cdp, error)``.

    The daemon may answer with the endpoint of a browser IT spawned (throwaway temp profile);
    the DevToolsActivePort OUR browser wrote is authoritative on disagreement.
    """
    _bt = _origin()
    try:
        browser_cmd = _install._find_agent_browser()
    except FileNotFoundError as e:
        return None, f"{_RP}the local browser engine (agent-browser) is not installed: {e}"
    argv = [*_session._agent_browser_argv(browser_cmd), "--session", _bt._REAL_PROFILE_SESSION,
            "--cdp", str(port), "open", "about:blank"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=_bt._get_open_command_timeout(first_open=True), env=_real_profile_daemon_env(),
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, _RP + "the real-profile browser took too long to start. Retry, or turn the toggle off."
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"{_RP}the launch failed: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, f"{_RP}the real-profile browser failed to start: {tail[-1] if tail else f'exit {proc.returncode}'}"
    cdp = _agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
    our_port = _read_devtools_port(copy_dir)
    if our_port is not None and (m := re.search(r":(\d+)", cdp or "")) and m.group(1) != our_port:
        cdp = f"http://127.0.0.1:{our_port}"
    if not cdp:
        return None, _RP + "the real-profile browser started without exposing a devtools endpoint. Retry, or turn the toggle off."
    return cdp, None


def _real_profile_cdp() -> tuple:
    """Resolve ``(cdp_url, error)`` for consented real-profile browsing.

    Snapshot -> hand-over (the person's binary reads the cookie jar, headless, and exits) ->
    launch the driven browser on the copy -> load the jar -> return its HTTP CDP endpoint. The copy
    is a non-default dir, so it sidesteps the Chrome >=136 default-profile remote-debugging block
    and never contends with the user's running browser. One shared agent-browser session is reused
    across calls (cached, re-validated). ``(None, message)`` fail-closed; ``(None, None)`` when consent is off.
    """
    _bt = _origin()
    if not _cloud._use_real_profile():
        # Consent is off: delete any snapshot store (copies of cookies/logins) so
        # revoking consent actually removes the credential copies.
        try:
            from hermes_cli.browser_connect import cleanup_real_profile_snapshots
            cleanup_real_profile_snapshots()
        except Exception as e:
            _bt.logger.debug("real-profile cleanup-on-consent-off failed: %s", e)
        _bt._real_profile_cdp_cache.pop("cdp", None)
        return None, None

    # Lightpanda rejects ``--profile``; check BEFORE default-browser detection so a
    # host with no Chromium default still reports the actionable engine conflict.
    if _lp._using_lightpanda_engine():
        return None, (_RP + "browser.engine is set to 'lightpanda', which cannot load a real Chromium profile. "
                      "Set browser.engine to 'auto' or 'chrome' to use real-profile browsing, or turn the toggle off.")

    from hermes_cli.browser_connect import (chromium_executable, detect_default_chromium,
                                            real_profile_copy_dir, snapshot_real_profile)

    with _bt._real_profile_cdp_lock:
        _bt._real_profile_last_used = time.time()
        claim_driven_browser()  # one file per process: a hand-over elsewhere sees this work (_other_work)
        cached = _bt._real_profile_cdp_cache.get("cdp")
        if cached and _cdp_http_ready(cached):
            # The keeper is checked on EVERY acquire: one that died or stalled is restarted here.
            _fidelity.ensure_keeper(int(cached.rsplit(":", 1)[1]), lambda: _persons_identity(detect_default_chromium()))
            # Re-claim the shared daemon's socket dir so the orphan reaper's idle clock sees
            # this process still using it (a cache hit never runs a daemon command).
            _session._prepare_session_socket_dir(_bt._REAL_PROFILE_SESSION)
            _refresh_jar_if_due(cached)  # a sign-in made since the last hand-over (ticket #18)
            return cached, None
        _bt._real_profile_cdp_cache.pop("cdp", None)

        browser = detect_default_chromium()
        unsupported = _real_profile_unsupported_reason(browser)
        if unsupported:
            return None, unsupported

        # Reuse BEFORE writing anything. CRITICAL: the snapshot overlay (truncates/rewrites
        # Cookies / Login Data) must NOT run while a live copy-browser (maybe from a previous
        # hermes process) holds the user-data-dir open — that corrupts the databases.
        copy_dir = real_profile_copy_dir(browser)
        existing = _agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
        if existing and _cdp_http_ready(existing) and _cdp_on_data_dir(existing, copy_dir):
            _fidelity.ensure_keeper(int(existing.rsplit(":", 1)[1]), lambda: _persons_identity(browser))
            _bt._real_profile_cdp_cache["cdp"] = existing
            _record_handover_target(existing, browser)
            _refresh_jar_if_due(existing)
            return existing, None
        if existing:  # stale/wrong-dir session: close it so nothing holds the dir open
            _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)
        # A browser from an EARLIER Hermes process can still hold the copy dir after its attach
        # daemon was reaped. If that owner is dead it is an orphan: terminate it (identity-bound to
        # our copy dir) so nothing ours outlives its owner. If the owner is alive (another Hermes
        # sharing this home) re-attach rather than overlay a live profile.
        _terminate_orphaned_browsers_on_dir(copy_dir)
        surviving = _surviving_chrome_cdp(copy_dir)
        if not surviving and _live_holders(copy_dir):
            # Still running, just not answering: a busy Mac or a stalled page holds a live
            # browser's DevTools past the one- and two-second probes above (measured: a stopped
            # driven browser fails both while its processes are still on the copy dir). It is
            # NOT gone, and overlaying the snapshot under it is exactly the CRITICAL case above —
            # on 2026-09-24 that re-sync ran three times under the live browser and reported
            # "3 database(s) unavailable". Wait for it, never write under it.
            surviving = _await_surviving_chrome_cdp(copy_dir)
            holders = [] if surviving else _live_holders(copy_dir)
            own = _own_browser_pids()
            if any(p.pid in own for p in holders):
                # OURS and wedged: left alone it would cost every later call the same wait until
                # Hermes restarts. Restart it (the snapshot + launch below). Never another's.
                _bt.logger.warning("real-profile: the driven browser on %s has not answered for %.0fs; "
                                   "restarting it", copy_dir, _BUSY_BROWSER_WAIT_S)
                _terminate_real_profile_chrome()
                _bt._real_profile_cdp_cache.pop("cdp", None)
                _restart_notes[:] = [RESTART_NOTE]  # the model must not assume its pages survived
                _await_holders_gone(copy_dir)
                holders = _live_holders(copy_dir)
            if holders:
                return None, (_RP + f"the agent's browser is still running but has not answered for "
                              f"{_BUSY_BROWSER_WAIT_S:.0f}s (the Mac may be busy, or a page is stuck). "
                              "Nothing was changed; retry in a moment.")
        if surviving:
            _fidelity.ensure_keeper(int(surviving.rsplit(":", 1)[1]), lambda: _persons_identity(browser))
            cdp, err = _attach_agent_browser_to_real_profile(int(surviving.rsplit(":", 1)[1]), copy_dir)
            if not cdp:
                return None, err
            _bt._real_profile_cdp_cache["cdp"] = cdp
            _bt.logger.info("real-profile: re-attached to a live owner's browser at %s (%s)", cdp, copy_dir)
            _record_handover_target(cdp, browser)
            _refresh_jar_if_due(cdp)
            return cdp, None

        handover_at = time.time()  # before the copy: a cookie written during it counts as newer
        copy_dir, err = snapshot_real_profile(browser)
        if err or not copy_dir:
            return None, _real_profile_snapshot_error(err)
        real_binary = chromium_executable(browser)
        if real_binary is None:
            return None, f"{_RP}the real browser binary for '{browser}' could not be found. Reinstall it or turn the toggle off."
        driven = driven_browser_executable()
        if driven is None and _install._maybe_autoinstall_chromium():
            driven = driven_browser_executable()
        if driven is None:
            return None, (_RP + "the browser engine's own Chrome is not installed, and Hermes never drives your "
                          "installed browser application. Run `agent-browser install`, or turn the toggle off.")

        cookies, err = _export_cookies_from_persons_browser(real_binary, copy_dir)
        if cookies is None:
            return None, err
        _forget_keychain_bound_auth_files(copy_dir)
        port, err = _launch_driven_browser(driven, copy_dir, _persons_identity(browser))
        if port is None:
            return None, err
        err = _import_cookies_into_driven_browser(port, cookies)
        if err:
            _terminate_real_profile_chrome()
            return None, err
        cdp, err = _attach_agent_browser_to_real_profile(port, copy_dir)
        if not cdp:
            _terminate_real_profile_chrome()
            return None, err
        _bt._real_profile_cdp_cache["cdp"] = cdp
        _record_handover_target(cdp, browser, at=handover_at)
        _bt.logger.info("real-profile browser ready for %s at %s (%s, %d cookie(s) handed over)",
                        browser, cdp, copy_dir, len(cookies))
        return cdp, None


# ---------------------------------------------------------------------------
# Showing the page to the person (browser_handoff)
# ---------------------------------------------------------------------------

def _is_headless_now(copy_dir: str) -> bool:
    """Whether the running driven browser is headless: what this process launched it as, else its own
    command line (the browser holding ``copy_dir``). NOT ``/json/version``: measured 2026-09-25, Chrome
    for Testing 154 under ``--headless=new`` reports ``Browser: Chrome/154…``, the same as a window.
    Unknown counts as hidden -- a needless relaunch costs a reload; a wrong "it's in front of you" costs
    the person looking for a window that is not there."""
    if _launched_headless["headless"] is not None:
        return bool(_launched_headless["headless"])
    try:
        cmdlines = [" ".join(p.cmdline()) for p in _browsers_on_data_dir(copy_dir)]
    except Exception:
        return True
    mains = [c for c in cmdlines if "--type=" not in c]
    return not mains or any("--headless" in c for c in mains)


def _main_browser_pid(copy_dir: str) -> Optional[int]:
    """The pid of the browser process (not a helper) running on ``copy_dir``: ours first, else the
    process holding the copy dir whose command line carries no ``--type=``."""
    for proc in reversed(_origin()._real_profile_chrome_procs):
        if proc.poll() is None:
            return proc.pid
    try:
        for p in _browsers_on_data_dir(copy_dir):
            if "--type=" not in " ".join(p.cmdline()):
                return int(p.pid)
    except Exception:
        return None
    return None


#: Raises one running application by pid, through AppKit (NSRunningApplication), from JavaScript
#: for Automation -- no Apple event, so no Automation prompt (options 3 = all windows | ignoring other
#: apps) -- then, ~300 ms later, asks macOS which app is actually frontmost. macOS 14+ may quietly
#: decline an activation a background process asks for, so the request succeeding proves nothing.
#: Prints "front", "not_front" or "no_app".
_ACTIVATE_PID_JXA = ("ObjC.import('AppKit');"
                     "var a=$.NSRunningApplication.runningApplicationWithProcessIdentifier(%d);"
                     "var r='no_app';"
                     "if (a && !a.isNil()) { a.activateWithOptions(3); delay(0.3);"
                     "var f=$.NSWorkspace.sharedWorkspace.frontmostApplication;"
                     "r=(f && !f.isNil() && f.processIdentifier == %d) ? 'front' : 'not_front'; }"
                     "r")


def _activate_pid(pid: int) -> bool:
    """Bring the process ``pid`` (and only it) to the front on macOS, and report whether it IS in front
    afterwards. By pid, never by bundle path: ``open -a`` on the newest installed Chrome for Testing
    could start a second copy, or raise a different one than the browser this page is in."""
    if sys.platform != "darwin" or not pid:
        return False
    try:
        out = subprocess.run(["/usr/bin/osascript", "-l", "JavaScript", "-e", _ACTIVATE_PID_JXA % (int(pid), int(pid))],
                             capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError) as e:
        _origin().logger.debug("handoff: activating pid %s failed: %s", pid, e)
        return False
    return out.returncode == 0 and out.stdout.strip() == "front"


def _bring_to_front(port: int, target_id: str, pid: Optional[int]) -> bool:
    """Select the tab, un-minimise its window and raise that browser process. True only when every
    step reported success (the window really is in front is not something CDP can confirm)."""
    ok = True
    try:
        _cdp_call(port, "Target.activateTarget", {"targetId": target_id}, timeout=5)
        window = _cdp_call(port, "Browser.getWindowForTarget", {"targetId": target_id}, timeout=5).get("windowId")
        if window is not None:
            _cdp_call(port, "Browser.setWindowBounds", {"windowId": window, "bounds": {"windowState": "normal"}},
                      timeout=5)
    except Exception as e:
        _origin().logger.debug("handoff: selecting the tab failed: %s", e)
        ok = False
    return _activate_pid(pid or 0) and ok


def driven_pages() -> List[Dict[str, str]]:
    """Open http(s) tabs of the running driven browser, most recently active first (measured: Chrome's
    ``/json/list`` moves an activated tab to the front), none when it is not running."""
    from tools.browser_person_step import page_targets
    cached = _origin()._real_profile_cdp_cache.get("cdp")
    return page_targets(cached) if cached else []


def _pick_page(pages: List[Dict[str, str]], url_hint: str) -> Optional[Dict[str, str]]:
    """The tab to show: the one at ``url_hint`` when given (None when no tab is there -- showing a
    different page than the one named would be worse than none), else the first."""
    if url_hint:
        return next((p for p in pages if p["url"] == url_hint or p["url"].startswith(url_hint)), None)
    return pages[0] if pages else None


#: Every Hermes process that uses the driven browser keeps ONE file here, named by its pid, touched on
#: every acquire. A single shared owner file was overwritten by our own re-claim, so another process's
#: live work was invisible to a hand-over; one file per process cannot be.
_CLAIMS_DIRNAME = "driven-browser-claims"


def _claims_dir() -> str:
    from hermes_cli.browser_connect import get_hermes_home
    return os.path.join(str(get_hermes_home()), "browser-profile", _CLAIMS_DIRNAME)


def claim_driven_browser(now: Optional[float] = None) -> None:
    """This process is using the driven browser (called on every acquire). Never raises."""
    try:
        d = _claims_dir()
        os.makedirs(d, mode=0o700, exist_ok=True)
        path = os.path.join(d, str(os.getpid()))
        with open(path, "a", encoding="utf-8"):
            pass
        now = time.time() if now is None else now
        os.utime(path, (now, now))
    except OSError as e:
        _origin().logger.debug("real-profile: claim not written: %s", e)


def other_live_claims(within_s: float, now: Optional[float] = None) -> List[int]:
    """Pids of OTHER live processes that used the driven browser within ``within_s`` seconds. A claim
    whose process is gone is removed."""
    now = time.time() if now is None else now
    out: List[int] = []
    try:
        d = _claims_dir()
        names = os.listdir(d)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        pid, path = int(name), os.path.join(d, name)
        if pid == os.getpid():
            continue
        if not _pid_alive(pid):
            try:
                os.unlink(path)
            except OSError:
                pass
            continue
        try:
            if 0 <= now - os.path.getmtime(path) <= within_s:
                out.append(pid)
        except OSError:
            continue
    return sorted(out)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _other_work(task_id: str) -> List[str]:
    """Who else is using the driven browser now, in words: another conversation in this process
    (a browser_exec lane used within the inactivity window), or another Hermes process (its claim on
    the shared engine session is newer than ours). Restarting the browser would cut both off."""
    _bt = _origin()
    reasons: List[str] = []
    try:
        from tools.browser_chrome_extension import recent_tasks
        others = [t for t in recent_tasks(_bt.BROWSER_SESSION_INACTIVITY_TIMEOUT) if t != str(task_id or "")]
        if others:
            reasons.append(f"{len(others)} other conversation(s) used it in the last few minutes")
    except Exception:
        pass
    others = other_live_claims(_bt.BROWSER_SESSION_INACTIVITY_TIMEOUT)
    if others:
        reasons.append(f"{len(others)} other part(s) of Moe (a different conversation's tools) used it in the last few minutes")
    return reasons


#: How long a shown window is protected from the idle reaper without a hand-back: long enough for
#: a person to find their phone, dig out a security key or step away and come back; not forever.
SHOWN_MAX_SECONDS = 2 * 3600


def show_to_person(url_hint: str = "", task_id: str = "") -> Dict[str, Any]:
    """Put the driven browser's page in front of the person, with the same sign-ins.

    * Already headed: select the tab, un-minimise, raise that process by pid.
    * Headless, launched by THIS process and nobody else using it: relaunched headed on the same
      profile copy. Every cookie is read out first and loaded back; EVERY open tab is reopened at its
      URL, the chosen one in front. A reload loses text typed into a form the site did not save and
      sessionStorage-only steps (``form_state_lost``); the caller must say so.
    * Headless and launched by another Hermes process, or in use by another conversation: NOT
      restarted (``busy``) -- that would cut their work off. The caller routes elsewhere.

    ``{"ok": True, "url", "title", "relaunched", "front", "form_state_lost", "reopened"}`` or
    ``{"ok": False, "why", "url"?, "busy"?, "browser_gone"?}``."""
    _bt = _origin()
    with _bt._real_profile_cdp_lock:
        cached = _bt._real_profile_cdp_cache.get("cdp")
        if not cached or not _cdp_http_ready(cached):
            return {"ok": False, "why": "Moe's own browser is not running"}
        pages = driven_pages()
        page = _pick_page(pages, url_hint)
        if page is None:
            return {"ok": False, "why": ("that page is not open in Moe's own browser" if url_hint and pages
                                         else "Moe's own browser has no web page open")}
        port = int(cached.rsplit(":", 1)[1])
        _bt._real_profile_last_used = time.time()
        from hermes_cli.browser_connect import detect_default_chromium, real_profile_copy_dir
        browser = detect_default_chromium()
        copy_dir = real_profile_copy_dir(browser)
        if not _is_headless_now(copy_dir):
            _shown_to_person.update(on=True, since=time.time())
            return {"ok": True, "url": page["url"], "title": page["title"], "relaunched": False,
                    "front": _bring_to_front(port, page["id"], _main_browser_pid(copy_dir)),
                    "form_state_lost": False, "reopened": 0}
        if _launched_headless["headless"] is None:
            return {"ok": False, "busy": True, "url": page["url"],
                    "why": ("Moe's browser was started by another part of Moe (another conversation's tools) and "
                            "runs out of sight; it cannot get a window from here without closing it under that work")}
        others = _other_work(task_id)
        if others:
            return {"ok": False, "busy": True, "url": page["url"],
                    "why": "Moe's browser was not restarted with a window: " + "; ".join(others)}

        binary = driven_browser_executable()
        if not binary or not _cdp_on_data_dir(cached, copy_dir):
            # Not the browser this home launched on its profile copy: never relaunch what is not ours.
            return {"ok": False, "why": "Moe's own browser could not be shown (it is not the one this Mac launched)",
                    "url": page["url"]}
        try:
            cookies = list(_cdp_call(port, "Storage.getCookies").get("cookies") or [])
        except Exception as e:
            return {"ok": False, "why": f"could not read the browser's sign-ins before showing it: {e}", "url": page["url"]}

        _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)
        _terminate_real_profile_chrome()
        _bt._real_profile_cdp_cache.pop("cdp", None)
        _await_holders_gone(copy_dir)
        _shown_to_person.update(on=True, since=time.time())
        new_port, err = _launch_driven_browser(binary, copy_dir, _persons_identity(browser))
        if new_port is None:
            _shown_to_person["on"] = False
            return {"ok": False, "why": err or "the browser did not start again", "url": page["url"], "browser_gone": True}
        err = _import_cookies_into_driven_browser(new_port, cookies)
        cdp, attach_err = (None, err) if err else _attach_agent_browser_to_real_profile(new_port, copy_dir)
        if not cdp:
            _terminate_real_profile_chrome()
            _shown_to_person["on"] = False
            return {"ok": False, "why": attach_err or "the browser did not start again", "url": page["url"],
                    "browser_gone": True}
        _bt._real_profile_cdp_cache["cdp"] = cdp
        if _handover.get("cdp") == cached:
            _handover["cdp"] = cdp  # the same jar, carried across; its age is unchanged
        try:
            before = {t.get("targetId") for t in (_cdp_call(new_port, "Target.getTargets").get("targetInfos") or [])
                      if t.get("type") == "page"}
        except Exception:
            before = set()
        # Every tab comes back, the chosen one last so it is the one in front.
        order = [p for p in pages if p is not page] + [page]
        target_id, reopened = "", 0
        for p in order:
            try:
                tid = str(_cdp_call(new_port, "Target.createTarget", {"url": p["url"]}).get("targetId") or "")
            except Exception as e:
                if p is page:
                    return {"ok": False, "why": f"the browser is up, but the page did not open again: {e}",
                            "url": page["url"]}
                continue
            reopened += 1
            if p is page:
                target_id = tid
        # The blank tab the engine's attach opened would otherwise be the first page a later call lands on.
        for blank in before:
            try:
                _cdp_call(new_port, "Target.closeTarget", {"targetId": blank}, timeout=5)
            except Exception:
                pass
        _bt.logger.info("handoff: driven browser shown to the person at %s (relaunched headed, %d tab(s), %d cookie(s))",
                        page["url"][:120], reopened, len(cookies))
        return {"ok": True, "url": page["url"], "title": page["title"], "relaunched": True,
                "front": _bring_to_front(new_port, target_id, _main_browser_pid(copy_dir)), "form_state_lost": True,
                "reopened": reopened, "tabs_lost": len(pages) - reopened}


def hand_back() -> Dict[str, Any]:
    """The person is done: where the page is now, and the next launch goes back out of sight. The window
    is left as it is -- relaunching it hidden again would reload the page they just finished on. The
    idle clock restarts now, so the reaper does not close the window the moment they finish."""
    _shown_to_person["on"] = False
    _origin()._real_profile_last_used = time.time()
    pages = driven_pages()
    page = pages[0] if pages else None
    return {"url": page["url"], "title": page["title"]} if page else {}
