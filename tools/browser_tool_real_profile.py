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
import subprocess
import sys
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


def _spawn_browser_on_copy(binary: str, copy_dir: str, extra_flags: Iterable[str], what: str,
                           headless: bool) -> Tuple[Optional[subprocess.Popen], Optional[int], Optional[str]]:
    """Launch ``binary`` on the profile COPY and wait for its debug port: ``(proc, port, error)``.

    The process is recorded in ``_real_profile_chrome_procs`` so exit/idle/atexit cleanup can reach
    it. On failure it is terminated here and ``(None, None, error)`` is returned.
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

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        line = _read_devtools_port(copy_dir) or ""
        if line.isdigit():
            return proc, int(line), None
        if proc.poll() is not None:
            _terminate_real_profile_chrome()
            return None, None, _RP + f"the {what} exited during startup (another instance may hold the profile copy)."
        time.sleep(0.25)
    _terminate_real_profile_chrome()
    return None, None, _RP + f"the {what} did not expose a debug port in time. Retry, or turn the toggle off."


def _driven_browser_headless() -> bool:
    """Headless by default (a focus-stealing window defeats a background capability); Chrome's NEW
    headless shares the profile's cookie store (legacy --headless does not). browser.headed /
    AGENT_BROWSER_HEADED opts into a window, except on a display-less Linux host (launch would die)."""
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


def _export_cookies_from_persons_browser(real_binary: str, copy_dir: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Start the person's browser binary headless on the copy, read every cookie its keychain
    entry decrypts, and terminate it. ``(cookies, None)`` or ``(None, error)``.

    This is the only moment the person's browser application runs for Hermes. It is headless,
    opens no window, and lives for the read (sub-second, measured) — but while it lives Launch
    Services counts it as that application, so it is never left running past this function.
    """
    _bt = _origin()
    proc, port, err = _spawn_browser_on_copy(real_binary, copy_dir, (), "hand-over browser", headless=True)
    if proc is None or port is None:
        return None, err
    try:
        cookies = list(_cdp_call(port, "Storage.getCookies").get("cookies") or [])
    except Exception as e:
        return None, f"{_RP}the sign-in hand-over failed to read the cookie jar: {e}"
    finally:
        _terminate_real_profile_chrome()
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


def _import_cookies_into_driven_browser(port: int, cookies: List[Dict[str, Any]]) -> Optional[str]:
    """``Storage.setCookies`` in chunks; None on success, else an error message."""
    params = [_cookie_param(c) for c in cookies]
    try:
        for i in range(0, len(params), _COOKIE_IMPORT_CHUNK):
            _cdp_call(port, "Storage.setCookies", {"cookies": params[i:i + _COOKIE_IMPORT_CHUNK]})
    except Exception as e:
        return f"{_RP}the sign-in hand-over failed to load the cookie jar: {e}"
    return None


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
    with _bt._real_profile_cdp_lock:
        _bt._real_profile_cdp_cache.pop("cdp", None)
        _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)
        _terminate_real_profile_chrome()
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
        cached = _bt._real_profile_cdp_cache.get("cdp")
        if cached and _cdp_http_ready(cached):
            # The keeper is checked on EVERY acquire: one that died or stalled is restarted here.
            _fidelity.ensure_keeper(int(cached.rsplit(":", 1)[1]), lambda: _persons_identity(detect_default_chromium()))
            # Re-claim the shared daemon's socket dir so the orphan reaper's idle clock sees
            # this process still using it (a cache hit never runs a daemon command).
            _session._prepare_session_socket_dir(_bt._REAL_PROFILE_SESSION)
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
            return existing, None
        if existing:  # stale/wrong-dir session: close it so nothing holds the dir open
            _agent_browser_close_session(_bt._REAL_PROFILE_SESSION)
        # A browser from an EARLIER Hermes process can still hold the copy dir after its attach
        # daemon was reaped. If that owner is dead it is an orphan: terminate it (identity-bound to
        # our copy dir) so nothing ours outlives its owner. If the owner is alive (another Hermes
        # sharing this home) re-attach rather than overlay a live profile.
        _terminate_orphaned_browsers_on_dir(copy_dir)
        surviving = _surviving_chrome_cdp(copy_dir)
        if surviving:
            _fidelity.ensure_keeper(int(surviving.rsplit(":", 1)[1]), lambda: _persons_identity(browser))
            cdp, err = _attach_agent_browser_to_real_profile(int(surviving.rsplit(":", 1)[1]), copy_dir)
            if not cdp:
                return None, err
            _bt._real_profile_cdp_cache["cdp"] = cdp
            _bt.logger.info("real-profile: re-attached to a live owner's browser at %s (%s)", cdp, copy_dir)
            return cdp, None

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
        _bt.logger.info("real-profile browser ready for %s at %s (%s, %d cookie(s) handed over)",
                        browser, cdp, copy_dir, len(cookies))
        return cdp, None
