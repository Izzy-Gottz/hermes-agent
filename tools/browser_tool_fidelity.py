"""Fidelity for the real-profile browser: it tells sites the truth about what it is.

The browser Hermes drives for real-profile browsing is agent-browser's Chrome for Testing,
headless, on a copy of the person's profile. Left alone it announces itself as a robot on
every request — measured on macOS 27 / Chrome for Testing 153 (2026-09-23):

* ``User-Agent: ... HeadlessChrome/153.0.0.0 ...``
* ``navigator.webdriver === true``
* ``Sec-CH-UA`` / ``navigator.userAgentData`` brands: ``Chromium`` and the GREASE entry only
* screen 800x600, outer window 0x0

Cloudflare, Kayak and others turned those into "Just a moment" and bot pages. None of those
values is true of the person the browser acts for. They sign in with their own Chrome, on their
own Mac, from their own home. This module makes the driven browser report exactly that:

* ``--disable-blink-features=AutomationControlled``: ``navigator.webdriver`` is false.
* ``--window-size`` / ``--screen-info``: the main display's real size, scale, menu-bar and Dock
  insets and colour depth.
* ``--user-agent``: the person's browser's own UA string (brand and version read from the
  installed app's ``Info.plist``, never by launching it; the ENGINE's version whenever the majors
  differ). It is the only thing that reaches ``navigator.userAgent`` in shared and service
  workers (measured; see :func:`launch_flags`).
* Brands: there is no switch for ``Sec-CH-UA``, so a :class:`FidelityKeeper` holds one
  browser-level DevTools connection for the browser's lifetime. It auto-attaches every target
  paused at start (``waitForDebuggerOnStart``), applies ``Emulation.setUserAgentOverride`` (pages,
  frames) or ``Network.setUserAgentOverride`` (workers) with the matching
  ``userAgentMetadata``, and only then resumes the target. A DevTools override lasts only as long
  as the session that set it, which is why the keeper stays connected rather than applying
  the override once and leaving. One keeper per browser across processes (a pause can only be
  released by the client that set it), checked and restarted on every acquire, with a watchdog
  that drops a stalled keeper's connection so Chrome lets its targets run.

What it does NOT do, by design: no CAPTCHA solving, no proxy, no randomised or invented
fingerprint, and no JavaScript patching of page globals. Every value is either the browser's own
or the person's own browser's. ``browser.stealth_fidelity: false`` turns all of it off.

Why headless and not a hidden window (measured 2026-09-23, macOS 27, see the test notes in the
commit): a headed Chrome for Testing takes focus from the person's app whenever a tab opens in
the foreground (agent-browser's ``open``, a site's popup, ``Page.bringToFront``). Hiding the app
does not help, because Chrome unhides and re-activates itself. macOS keeps 40 px of an
off-screen window on screen. ``lsappinfo`` refuses to retype another app. A copy of the bundle
patched with ``LSBackgroundOnly`` stops the Dock icon and the focus steal, but its popups still
open full-size and visible at the top-left of the screen. Chrome's new headless mode is the only
mode measured never to show a pixel or take focus. So this module makes headless truthful rather
than trading the owner's rule ("out of sight") for a window.
"""

import json
import logging
import os
import platform
import plistlib
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The macOS token every Chrome on a Mac sends since UA reduction (frozen, whatever the OS).
_MAC_UA_PLATFORM = "Macintosh; Intel Mac OS X 10_15_7"

# Brand each supported default browser declares in Sec-CH-UA (Chromium's embedder brand), and the
# extra UA-string product token it adds after ``Safari/537.36`` (Edge only).
_BRAND_BY_BROWSER = {
    "chrome": ("Google Chrome", None),
    "edge": ("Microsoft Edge", "Edg"),
    "brave": ("Brave", None),
    "brave-origin": ("Brave", None),
    "chromium": (None, None),
}


def fidelity_enabled() -> bool:
    """``browser.stealth_fidelity`` (default True). Any of false/0/no/off turns it off."""
    from tools.browser_tool_origin import origin_module
    return bool(origin_module()._browser_cfg(
        "stealth_fidelity", True,
        lambda v: True if v is None else str(v).strip().lower() not in ("false", "0", "no", "off"),
        "browser.stealth_fidelity from config"))


# ---------------------------------------------------------------------------
# Who the person's browser is
# ---------------------------------------------------------------------------

def _app_info_plist(binary: str) -> Optional[str]:
    """``…/X.app/Contents/Info.plist`` for a macOS app executable path, else None."""
    m = re.match(r"^(.*?\.app)/Contents/MacOS/[^/]+$", binary or "")
    return os.path.join(m.group(1), "Contents", "Info.plist") if m else None


def installed_browser_version(binary: str) -> Optional[str]:
    """Full version of an installed browser from its bundle's Info.plist (macOS), never by running
    it. ``CFBundleShortVersionString`` is the full four-part version on every Chromium browser."""
    plist = _app_info_plist(binary)
    if not plist:
        return None
    try:
        with open(plist, "rb") as fh:
            version = str(plistlib.load(fh).get("CFBundleShortVersionString") or "")
    except (OSError, plistlib.InvalidFileException, ValueError) as e:
        logger.debug("fidelity: cannot read %s: %s", plist, e)
        return None
    return version if re.fullmatch(r"\d+(\.\d+){3}", version) else None


def persons_browser_identity(browser: Optional[str], binary: Optional[str],
                             engine_version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """``{"brand", "ua_suffix", "full_version", "major"}`` of the person's browser, or None when its
    version cannot be read. The brand is theirs. The version is theirs too unless ``engine_version``
    (the driven browser's) has a different major, in which case it is the engine's: see
    :func:`with_engine_version`."""
    if browser not in _BRAND_BY_BROWSER or not binary:
        return None
    version = installed_browser_version(binary)
    if not version:
        return None
    brand, suffix = _BRAND_BY_BROWSER[browser]
    identity = {"brand": brand, "ua_suffix": suffix, "full_version": version, "major": int(version.split(".")[0])}
    return with_engine_version(identity, engine_version)


# ---------------------------------------------------------------------------
# What it sends
# ---------------------------------------------------------------------------

_GREASE_CHARS = (" ", "(", ":", "-", ".", "/", ")", ";", "=", "?", "_")
_GREASE_VERSIONS = ("8", "99", "24")
_BRAND_ORDERS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def brand_list(brand: Optional[str], major: int, version: str) -> List[Dict[str, str]]:
    """The brand list Chromium itself builds for ``major`` (components/embedder_support/
    user_agent_utils.cc ``GenerateBrandVersionList``, seeded by the major version). ``version`` is
    the major for ``brands`` and the full version for ``fullVersionList``. The GREASE entry
    carries its own version: ``N`` in ``brands``, ``N.0.0.0`` in ``fullVersionList``.

    Checked against the driven browser's own output: Chrome for Testing 153 (no embedder brand)
    reports ``Chromium;153, Not_A Brand;8``, and this returns exactly that for ``brand=None``.
    """
    seed = major
    full = "." in version
    g_version = _GREASE_VERSIONS[seed % len(_GREASE_VERSIONS)]
    grease = {"brand": f"Not{_GREASE_CHARS[seed % len(_GREASE_CHARS)]}A{_GREASE_CHARS[(seed + 1) % len(_GREASE_CHARS)]}Brand",
              "version": f"{g_version}.0.0.0" if full else g_version}
    chromium = {"brand": "Chromium", "version": version}
    if brand is None:
        out = [None, None]
        out[seed % 2] = grease
        out[(seed + 1) % 2] = chromium
        return out
    order = _BRAND_ORDERS[seed % len(_BRAND_ORDERS)]
    out = [None, None, None]
    out[order[0]] = grease
    out[order[1]] = chromium
    out[order[2]] = {"brand": brand, "version": version}
    return out


def _mac_platform_version() -> str:
    parts = (platform.mac_ver()[0] or "").split(".")
    parts = [p for p in parts if p.isdigit()] or ["0"]
    return ".".join((parts + ["0", "0"])[:3])


def _architecture() -> Tuple[str, str]:
    machine = platform.machine().lower()
    return ("arm", "64") if machine in ("arm64", "aarch64") else ("x86", "64")


def user_agent(identity: Dict[str, Any]) -> str:
    """The reduced UA string the person's browser sends on a Mac."""
    ua = (f"Mozilla/5.0 ({_MAC_UA_PLATFORM}) AppleWebKit/537.36 (KHTML, like Gecko) "
          f"Chrome/{identity['major']}.0.0.0 Safari/537.36")
    if identity.get("ua_suffix"):
        ua += f" {identity['ua_suffix']}/{identity['major']}.0.0.0"
    return ua


def user_agent_metadata(identity: Dict[str, Any]) -> Dict[str, Any]:
    """``Emulation.UserAgentMetadata`` for the person's browser on this Mac."""
    arch, bitness = _architecture()
    return {
        "brands": brand_list(identity["brand"], identity["major"], str(identity["major"])),
        "fullVersionList": brand_list(identity["brand"], identity["major"], identity["full_version"]),
        "fullVersion": identity["full_version"],
        "platform": "macOS",
        "platformVersion": _mac_platform_version(),
        "architecture": arch,
        "bitness": bitness,
        "model": "",
        "mobile": False,
        "wow64": False,
        "formFactors": ["Desktop"],
    }


# The display probe runs in a child process: it loads AppKit (NSScreen.visibleFrame is the only
# source of the menu-bar and Dock insets), and a framework load must never be able to take the
# Hermes process down with it.
_DISPLAY_PROBE = r"""
import ctypes, ctypes.util, json
class P(ctypes.Structure): _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]
class S(ctypes.Structure): _fields_ = [("w", ctypes.c_double), ("h", ctypes.c_double)]
class R(ctypes.Structure): _fields_ = [("o", P), ("s", S)]
objc = ctypes.CDLL(ctypes.util.find_library("objc"))
ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
objc.objc_getClass.restype = ctypes.c_void_p
objc.sel_registerName.restype = ctypes.c_void_p
msg = objc.objc_msgSend
msg.restype = ctypes.c_void_p
msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
rect = ctypes.CFUNCTYPE(R, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
dbl = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
screen = msg(objc.objc_getClass(b"NSScreen"), objc.sel_registerName(b"mainScreen"))
f, v = rect(screen, objc.sel_registerName(b"frame")), rect(screen, objc.sel_registerName(b"visibleFrame"))
scale = dbl(screen, objc.sel_registerName(b"backingScaleFactor"))
print(json.dumps({"width": f.s.w, "height": f.s.h, "scale": scale,
                  "top": (f.o.y + f.s.h) - (v.o.y + v.s.h), "bottom": v.o.y - f.o.y}))
"""


def main_display() -> Optional[Dict[str, float]]:
    """The main display in points: ``{"width", "height", "scale", "top", "bottom"}``. ``top`` and
    ``bottom`` are the menu-bar and Dock insets (measured on this MacBook: 1470x956 @2x, 33 top,
    72 bottom). None off macOS or on any failure. Nothing is guessed."""
    if sys.platform != "darwin":
        return None
    import subprocess
    try:
        out = subprocess.run([sys.executable, "-c", _DISPLAY_PROBE], capture_output=True, text=True,
                             timeout=10, stdin=subprocess.DEVNULL).stdout
        d = json.loads(out.strip().splitlines()[-1])
        if d["width"] > 0 and d["height"] > 0 and d["scale"] > 0:
            return {k: float(d[k]) for k in ("width", "height", "scale", "top", "bottom")}
    except Exception as e:  # never let a display probe fail a launch
        logger.debug("fidelity: main display probe failed: %s", e)
    return None


def launch_flags(identity: Optional[Dict[str, Any]], headless: bool,
                 display: Optional[Dict[str, float]] = None) -> List[str]:
    """Switches for the driven browser: ``AutomationControlled`` always off; the person's UA string
    when their identity is known; with headless and a known display, that display's real geometry
    (headless otherwise reports an 800x600 screen).

    Why ``--user-agent`` stays, although the keeper's override also sets the UA string:
    measured 2026-09-23 on CfT 153, neither ``Emulation`` nor ``Network.setUserAgentOverride``
    changes ``navigator.userAgent`` inside a SHARED or SERVICE worker. Their brands follow the
    override, but their UA stays ``HeadlessChrome/153``. Only the switch reaches them. Without it,
    every site that runs a service worker would see a headless UA next to Google Chrome brands.
    The cost is the case the reviewer found: while NO keeper serves, a new tab has the switch's
    UA over Chromium brands with empty high-entropy hints. :func:`ensure_keeper` runs on every
    acquire, restarts a dead or stalled keeper, and reports the gap to status/doctor, so that
    window stays short and visible."""
    flags = ["--disable-blink-features=AutomationControlled"]
    if identity:
        flags.append(f"--user-agent={user_agent(identity)}")
    if headless and display:
        scale = display["scale"]
        px = lambda points: round(points * scale)  # --screen-info is in device pixels (measured)
        win_w, win_h = default_window_size(int(display["width"]), int(display["height"]))
        flags.append(f"--window-size={win_w},{win_h}")
        # {W x H devicePixelRatio workAreaTop workAreaBottom colorDepth}: all in device pixels.
        # Measured: {2940x1912 devicePixelRatio=2 workAreaTop=66 workAreaBottom=144 colorDepth=30}
        # reads back as screen 1470x956, availTop 33, availHeight 851, colorDepth 30.
        flags.append(f"--screen-info={{{px(display['width'])}x{px(display['height'])} devicePixelRatio={scale:g} "
                     f"workAreaTop={px(display['top'])} workAreaBottom={px(display['bottom'])} colorDepth=30}}")
    return flags


def default_window_size(width: int, height: int) -> Tuple[int, int]:
    """The window a fresh headed Chrome opens on this display: measured on a 1470x956 MacBook
    display, Chrome for Testing 153 opened 1440x853 (full width less a 15 px margin each side,
    height less the menu bar and Dock). The same margins apply on any display."""
    return max(800, width - 30), max(600, height - 103)


# ---------------------------------------------------------------------------
# The keeper: brands on every target, for the browser's lifetime
# ---------------------------------------------------------------------------

_PAGE_LIKE = ("page", "iframe", "webview")  # browser_ui / other: Chrome's own UI, only resumed
_WORKER_LIKE = ("worker", "service_worker", "shared_worker", "shared_storage_worklet", "auction_worklet")

# The keeper runs in its OWN process (see :class:`KeeperProcess`). A keeper whose heartbeat is
# older than this is treated as stuck: the parent kills it (Chrome releases what it held paused)
# and restarts it at once, and the restart re-applies the brands to every running target
# (measured). 10 s, not less: a short hitch only delays new tabs, and they still come out
# correct. Killing too early would briefly strip the brands from tabs that were fine.
STALL_SECONDS = 10.0
# How stale another process's keeper heartbeat may be before it is taken over.
HOLDER_STALE_SECONDS = 10.0
# A target attached (so paused by waitForDebuggerOnStart) whose resume is not acknowledged within
# this long gets one more Runtime.runIfWaitingForDebugger; one still paused after that is counted.
RESUME_SECONDS = 3.0


def commands_for_attached_target(params: Dict[str, Any], ua: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """The commands, in order, for one ``Target.attachedToTarget`` event: the override first, then
    auto-attach one level down (dedicated workers, out-of-process frames), then resume. A target
    paused by ``waitForDebuggerOnStart`` runs no script before the resume, so its first script
    already sees the brands. Commands on one session run in order.

    The pause holds the RENDERER, not a navigation the browser has already started. A tab
    created straight at a URL (``Target.createTarget({url})``) sends its first request before any
    of this runs. That is why every tab Hermes opens starts at ``about:blank`` and then navigates."""
    kind = (params.get("targetInfo") or {}).get("type", "")
    override = {"userAgent": ua, "userAgentMetadata": metadata}
    cmds: List[Tuple[str, Dict[str, Any]]] = []
    if kind in _PAGE_LIKE:
        cmds.append(("Emulation.setUserAgentOverride", override))
        cmds.append(("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True}))
    elif kind in _WORKER_LIKE:
        cmds.append(("Network.setUserAgentOverride", override))
        if kind != "worker":  # a shared/service worker can start dedicated workers of its own
            cmds.append(("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True}))
    cmds.append(("Runtime.runIfWaitingForDebugger", {}))
    return cmds


def _engine_version(browser_field: str) -> Optional[str]:
    """``153.0.8010.52`` from ``/json/version``'s ``Browser`` (``HeadlessChrome/153.0.8010.52``)."""
    m = re.search(r"/(\d+\.\d+\.\d+\.\d+)", browser_field or "")
    return m.group(1) if m else None


def _lock_path(port: int, browser_ws: str) -> str:
    """One lock per browser INSTANCE (its devtools browser id), shared by every Hermes process."""
    import tempfile
    guid = re.sub(r"[^A-Za-z0-9-]", "", browser_ws.rsplit("/", 1)[-1])[:64] or "unknown"
    return os.path.join(tempfile.gettempdir(), f"hermes-browser-fidelity-{port}-{guid}.lock")


def _is_keeper_for(proc, port: int, lock_path: str) -> bool:
    """True only for the keeper invocation for THIS port (``-m tools.browser_tool_fidelity keeper
    <port>``) that also has THIS lock file open. Anything else, including a keeper for another
    port that happens to have written the same pid, is never killed."""
    try:
        argv = proc.cmdline()
        i = argv.index("keeper")
        if _KEEPER_MARK not in argv or argv[i + 1] != str(port):
            return False
        want = os.path.realpath(lock_path)
        return any(os.path.realpath(f.path) == want for f in proc.open_files())
    except Exception:
        return False


def sweep_lock_files(now: Optional[float] = None) -> int:
    """Remove lock files of browsers that are gone. A lock is removed only when nobody holds it
    AND its port no longer serves that browser id (a new browser always has a new id, so the path
    is never reused). Returns how many were removed."""
    import fcntl
    import glob
    import tempfile
    removed = 0
    for path in glob.glob(os.path.join(tempfile.gettempdir(), "hermes-browser-fidelity-*.lock")):
        m = re.match(r"hermes-browser-fidelity-(\d+)-(.+)\.lock$", os.path.basename(path))
        if not m:
            continue
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue  # held: its keeper is alive
        try:
            try:
                import requests
                ws = requests.get(f"http://127.0.0.1:{m.group(1)}/json/version", timeout=1,
                                  proxies={"http": None, "https": None}).json().get("webSocketDebuggerUrl", "")
            except Exception:
                ws = ""
            if _lock_path(int(m.group(1)), ws) != path:
                os.unlink(path)
                removed += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    removed += _sweep_status_files(now)
    return removed


def _sweep_status_files(now: Optional[float] = None) -> int:
    """Remove keeper status files whose keeper process is gone (older than a minute)."""
    import glob
    import tempfile
    import time
    removed = 0
    for path in glob.glob(os.path.join(tempfile.gettempdir(), "hermes-fidelity-*.json")):
        try:
            if (now or time.time()) - os.stat(path).st_mtime < 60:
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    pid = int(json.load(fh).get("pid") or 0)
            except (OSError, ValueError):
                pid = 0
            import psutil
            if pid and psutil.pid_exists(pid):
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            pass
    return removed


def holder_is_stale(lock_path: str, now: Optional[float] = None) -> bool:
    """True when the process holding ``lock_path`` has stopped refreshing its heartbeat."""
    import time
    try:
        return ((now or time.time()) - os.stat(lock_path).st_mtime) > HOLDER_STALE_SECONDS
    except OSError:
        return False



class FidelityKeeper(threading.Thread):
    """One browser-level DevTools connection that puts the person's brands on every target.

    It runs inside a keeper process of its own (:class:`KeeperProcess`), so it shares no GIL with
    the Hermes gateway: a gateway stall never delays it.

    Only ONE keeper pauses a given browser, across every Hermes process. It holds an exclusive
    ``flock`` for the browser instance, because a pause can only be released by the DevTools
    client that set it. Measured: a healthy client's ``runIfWaitingForDebugger`` does not release
    a target another client holds, so a second keeper would only add a second way to wedge every
    new tab. A keeper that finds the lock taken becomes a ``follower`` and adds no pauses. If the
    holder's heartbeat is more than :data:`HOLDER_STALE_SECONDS` old, and the holder really is a
    keeper process, the new one kills it and takes over; SIGKILL reaches a stopped process too.

    ``state`` is one of ``starting``, ``serving``, ``follower``, ``failed`` or ``stopped``."""

    def __init__(self, port: int, identity: Dict[str, Any], connect=None):
        super().__init__(name=f"hermes-browser-fidelity-{port}", daemon=True)
        self.port, self.identity = port, identity
        self.ua, self.metadata = user_agent(identity), user_agent_metadata(identity)
        self._connect = connect
        self._ws = None
        self._halt = threading.Event()
        self.ready = threading.Event()
        self.applied: Dict[str, int] = {}
        self.error: Optional[str] = None
        self.state = "starting"
        self.lock_path: Optional[str] = None
        self._lock_fd: Optional[int] = None
        self.beat = 0.0  # wall clock of the loop's last tick: the heartbeat
        self._next_id = 0
        # Targets attached (so paused) whose resume has not been acknowledged:
        # session -> {"at": monotonic attach time, "kind", "resume_id", "retried"}.
        self._pending: Dict[str, Dict[str, Any]] = {}
        self.resume_failures = 0  # resumes the browser answered with an error
        self.resume_retries = 0   # second resumes sent for a target still paused after RESUME_SECONDS
        self.unresumed = 0        # targets still not resumed after the retry: given up on, logged
        self._send_lock = threading.Lock()

    # -- plumbing -------------------------------------------------------------------------------

    def _send(self, method: str, params: Dict[str, Any], session: Optional[str] = None) -> int:
        with self._send_lock:
            self._next_id += 1
            msg: Dict[str, Any] = {"id": self._next_id, "method": method, "params": params}
            if session:
                msg["sessionId"] = session
            self._ws.send(json.dumps(msg))
            return self._next_id

    def _try_lock(self, fd: int) -> bool:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _claim(self, lock_path: str) -> bool:
        """Take the per-browser lock; False when another live, heartbeating keeper holds it."""
        self.lock_path = lock_path
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        if not self._try_lock(fd) and not (self._take_over_stale_holder(lock_path) and self._try_lock(fd)):
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.utime(lock_path)
        self._lock_fd = fd
        return True

    def _take_over_stale_holder(self, lock_path: str) -> bool:
        """Kill a holder whose heartbeat stopped (e.g. SIGSTOPped), only if it is a keeper process."""
        if not holder_is_stale(lock_path):
            return False
        try:
            pid = int(open(lock_path, encoding="utf-8").read().strip() or 0)
            import psutil
            if pid <= 1 or pid == os.getpid() or not _is_keeper_for(psutil.Process(pid), self.port, lock_path):
                return False
            logger.warning("fidelity: keeper pid %s stopped heartbeating; taking the browser over", pid)
            os.kill(pid, 9)
            psutil.Process(pid).wait(3)
        except Exception as e:
            logger.debug("fidelity: stale-holder takeover skipped: %s", e)
            return "NoSuchProcess" in type(e).__name__
        return True

    def _release(self) -> None:
        if self._lock_fd is not None and self.lock_path and not self._browser_alive():
            # The browser this lock names is gone, and its id never comes back: remove the file
            # while still holding the lock, so nobody can be mid-claim on it.
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)  # closing drops the flock
            except OSError:
                pass
            self._lock_fd = None

    def _browser_alive(self) -> bool:
        """True while the browser instance this keeper's lock names still answers on its port."""
        if self._connect is not None:
            return True
        try:
            import requests
            ws = requests.get(f"http://127.0.0.1:{self.port}/json/version", timeout=2,
                              proxies={"http": None, "https": None}).json().get("webSocketDebuggerUrl", "")
            return bool(self.lock_path) and _lock_path(self.port, ws) == self.lock_path
        except Exception:
            return False

    def _open(self):
        """Connect, adopt the ENGINE's version when it differs from the person's browser, and
        claim the browser. Returns the socket, or None as a follower."""
        if self._connect is not None:
            return self._connect(self.port)
        import requests
        from websockets.sync.client import connect
        version: Dict[str, Any] = {}
        # A browser that has just written its port can take a few seconds to answer under load.
        for attempt in range(4):
            try:
                version = requests.get(f"http://127.0.0.1:{self.port}/json/version", timeout=3,
                                       proxies={"http": None, "https": None}).json()
                break
            except requests.RequestException:
                if attempt == 3 or self._halt.is_set():
                    raise
        reconciled = with_engine_version(self.identity, _engine_version(version.get("Browser", "")))
        if reconciled != self.identity:
            self.identity = reconciled
            self.ua, self.metadata = user_agent(reconciled), user_agent_metadata(reconciled)
        ws_url = version["webSocketDebuggerUrl"]
        if not self._claim(_lock_path(self.port, ws_url)):
            return None
        return connect(ws_url, max_size=None, open_timeout=10, close_timeout=2)

    def _pending_for(self, msg_id: Any) -> Optional[str]:
        for session, p in self._pending.items():
            if p["resume_id"] == msg_id:
                return session
        return None

    def handle(self, msg: Dict[str, Any]) -> None:
        """React to one DevTools message (public for tests)."""
        import time
        method = msg.get("method")
        if method == "Target.attachedToTarget":
            params = msg.get("params") or {}
            session = params.get("sessionId")
            kind = (params.get("targetInfo") or {}).get("type", "?")
            resume_id = None
            for m, cmd_params in commands_for_attached_target(params, self.ua, self.metadata):
                sent = self._send(m, cmd_params, session)
                if m == "Runtime.runIfWaitingForDebugger":
                    resume_id = sent
            self.applied[kind] = self.applied.get(kind, 0) + 1
            if session and resume_id is not None:
                self._pending[session] = {"at": time.monotonic(), "kind": kind,
                                          "resume_id": resume_id, "retried": False}
        elif method == "Target.detachedFromTarget":
            self._pending.pop((msg.get("params") or {}).get("sessionId"), None)
        elif msg.get("id"):
            session = self._pending_for(msg["id"])
            if session is None:
                if "error" in msg:
                    # A target that went away between attach and override answers with an error.
                    logger.debug("fidelity: command %s failed: %s", msg.get("id"), msg["error"])
            elif "error" in msg:
                # The target stays paused: every script on it (and a daemon waiting on it) hangs.
                self.resume_failures += 1
                logger.warning("fidelity: resume of %s target %s failed on port %s: %s",
                               self._pending[session]["kind"], session, self.port, msg["error"])
            else:
                self._pending.pop(session, None)

    def _check_pending(self) -> None:
        """A target attached but not resumed within RESUME_SECONDS is resumed once more; one still
        paused after that is counted, logged and let go (the status carries the count)."""
        import time
        now = time.monotonic()
        for session, p in list(self._pending.items()):
            if now - p["at"] < RESUME_SECONDS:
                continue
            if not p["retried"]:
                self.resume_retries += 1
                logger.warning("fidelity: %s target %s on port %s not resumed after %.0fs; resuming again",
                               p["kind"], session, self.port, now - p["at"])
                try:
                    p["resume_id"] = self._send("Runtime.runIfWaitingForDebugger", {}, session)
                except Exception as e:
                    logger.warning("fidelity: resume retry for %s failed: %s", session, e)
                p["retried"], p["at"] = True, now
            else:
                self.unresumed += 1
                logger.warning("fidelity: %s target %s on port %s still not resumed after a retry; "
                               "it may stay paused", p["kind"], session, self.port)
                self._pending.pop(session, None)

    def _tick(self) -> None:
        import time
        self.beat = time.time()
        if self._pending:
            self._check_pending()
        if self.lock_path and self._lock_fd is not None:
            try:
                os.utime(self.lock_path)  # the cross-process heartbeat
            except OSError:
                pass

    def _close_ws(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    def run(self) -> None:
        try:
            self._ws = self._open()
            if self._ws is None:
                self.state = "follower"
                logger.info("fidelity: browser on port %s is served by another keeper (%s)", self.port, self.lock_path)
                return
            # Browser level: attach every existing and future top-level target, paused at start.
            # Already-running targets are attached too, which is how a restart re-applies the brands.
            self._send("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True})
            self._tick()
            self.state = "serving"
            self.ready.set()
            while not self._halt.is_set():
                try:
                    raw = self._ws.recv(timeout=1.0)
                except TimeoutError:
                    self._tick()
                    continue
                self.handle(json.loads(raw))
                self._tick()
        except Exception as e:  # the browser went away or never answered
            if not self._halt.is_set():
                self.state, self.error = "failed", f"{type(e).__name__}: {e}"
                logger.warning("fidelity keeper for port %s ended: %s", self.port, self.error)
        finally:
            if self._halt.is_set():
                self.state = "stopped"
            self._release()
            self.ready.set()
            self._close_ws()

    def stop(self) -> None:
        self._halt.set()
        self._close_ws()

    def snapshot(self) -> Dict[str, Any]:
        return {"pid": os.getpid(), "port": self.port, "state": self.state, "applied": dict(self.applied),
                "error": self.error, "beat": self.beat, "ua": self.ua, "lock_path": self.lock_path,
                "pending": len(self._pending), "resume_failures": self.resume_failures,
                "resume_retries": self.resume_retries, "unresumed": self.unresumed,
                "claims": f"{self.identity.get('brand') or 'Chromium'} {self.identity['full_version']}"}


def with_engine_version(identity: Dict[str, Any], engine_version: Optional[str]) -> Dict[str, Any]:
    """The identity to claim on an engine of ``engine_version``. The brand is the person's; the
    version must be the engine's whenever the majors differ. Chrome auto-updates before Chrome for
    Testing does, and claiming 154 on a 153 engine is a lie a feature probe can catch."""
    if not engine_version:
        return identity
    engine_major = int(engine_version.split(".")[0])
    if engine_major == identity["major"]:
        return identity
    logger.info("fidelity: %s is %s but the engine is %s; claiming the engine's version",
                identity.get("brand") or "the browser", identity["full_version"], engine_version)
    return {**identity, "full_version": engine_version, "major": engine_major}


# ---------------------------------------------------------------------------
# The keeper process: no GIL shared with the gateway
# ---------------------------------------------------------------------------

_KEEPER_MARK = "tools.browser_tool_fidelity"
# Bound at import: a test that fakes subprocess.Popen to watch browser launches must not also
# capture (or break) the keeper's own helper process.
from subprocess import Popen as _Popen  # noqa: E402


def _keeper_main(argv: List[str]) -> int:
    """Entry point of a keeper process: ``python -m tools.browser_tool_fidelity keeper PORT
    IDENTITY_JSON STATUS_PATH``. It exits when its parent goes away (stdin reaches EOF), when the
    browser goes away, or on SIGTERM. Its heartbeat and counters go to STATUS_PATH every second."""
    import signal
    import time
    port, identity, status_path = int(argv[0]), json.loads(argv[1]), argv[2]
    keeper = FidelityKeeper(port, identity)

    def write():
        tmp = f"{status_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(keeper.snapshot(), fh)
            os.replace(tmp, status_path)
        except OSError:
            pass

    orphaned = threading.Event()

    def parent_gone():
        try:
            while sys.stdin.buffer.read(1):
                pass
        except Exception:
            pass
        orphaned.set()
        keeper.stop()

    signal.signal(signal.SIGTERM, lambda *_: keeper.stop())
    threading.Thread(target=parent_gone, daemon=True).start()
    keeper.start()
    write()
    keeper.ready.wait(20)  # report "serving"/"follower"/"failed" the moment it is known
    while keeper.is_alive():
        write()
        keeper.join(1.0)
    if orphaned.is_set():  # nobody is left to read it
        for path in (status_path, f"{status_path}.tmp"):
            try:
                os.unlink(path)
            except OSError:
                pass
    else:
        write()
    return 0


class KeeperProcess:
    """The gateway's handle on one keeper process. The same surface the tests and callers use:
    ``port``, ``ua``, ``state``, ``applied``, ``error``, ``healthy()``, ``stop()``, ``join()``."""

    def __init__(self, port: int, identity: Dict[str, Any]):
        import subprocess
        import tempfile
        self.port, self.identity = port, identity
        self.ua = user_agent(identity)
        fd, self.status_path = tempfile.mkstemp(prefix=f"hermes-fidelity-{port}-", suffix=".json")
        os.close(fd)
        os.unlink(self.status_path)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, PYTHONPATH=root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
        self.proc = _Popen([sys.executable, "-m", _KEEPER_MARK, "keeper", str(port), json.dumps(identity),
                                      self.status_path], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, env=env, close_fds=True)
        self._stopped = False

    def _snap(self) -> Dict[str, Any]:
        try:
            with open(self.status_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    @property
    def state(self) -> str:
        snap = self._snap()
        state = snap.get("state") or "starting"
        if self.proc.poll() is not None and state in ("starting", "serving"):
            return "stopped" if self._stopped else "failed"
        return state

    @property
    def applied(self) -> Dict[str, int]:
        return self._snap().get("applied") or {}

    @property
    def error(self) -> Optional[str]:
        snap = self._snap()
        if snap.get("error"):
            return snap["error"]
        if self.proc.poll() is not None and not self._stopped and snap.get("state") != "follower":
            return f"keeper process exited ({self.proc.returncode})"
        return None

    def heartbeat_age(self) -> float:
        import time
        return time.time() - float(self._snap().get("beat") or 0)

    def wait_ready(self, wait: float) -> None:
        import time
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and self.state == "starting":
            time.sleep(0.05)

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def healthy(self) -> bool:
        return self.is_alive() and self.state == "serving" and self.heartbeat_age() <= STALL_SECONDS

    def stop(self) -> None:
        self._stopped = True
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()

    def kill(self) -> None:
        self._stopped = True
        if self.proc.poll() is None:
            self.proc.kill()

    def join(self, timeout: Optional[float] = None) -> None:
        import subprocess
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            self.proc.wait(2)
        self._discard_status()

    def _discard_status(self) -> None:
        """The status file outlives nothing: remove it once the process is gone."""
        if self.proc.poll() is not None:
            for path in (self.status_path, f"{self.status_path}.tmp"):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def describe(self) -> Dict[str, Any]:
        snap = self._snap()
        d = {"port": self.port, "state": self.state, "pid": self.proc.pid, "applied": snap.get("applied") or {},
             "error": self.error, "claims": snap.get("claims") or f"{self.identity.get('brand') or 'Chromium'} {self.identity['full_version']}",
             # The keeper process's own heartbeat and counters, so a reader can tell live from frozen.
             "beat": snap.get("beat") or 0, "heartbeat_age": round(self.heartbeat_age(), 1),
             "pending": snap.get("pending", 0), "resume_failures": snap.get("resume_failures", 0),
             "resume_retries": snap.get("resume_retries", 0), "unresumed": snap.get("unresumed", 0)}
        if d["state"] == "follower" and snap.get("lock_path"):
            d["holder_stale"] = holder_is_stale(snap["lock_path"])
        return d


_keepers: Dict[int, "KeeperProcess"] = {}
_keepers_lock = threading.Lock()
_monitor: Optional[threading.Thread] = None


def _start_keeper(port: int, identity: Dict[str, Any]) -> "KeeperProcess":
    keeper = KeeperProcess(port, identity)
    _keepers[port] = keeper
    _ensure_monitor()
    return keeper


def _monitor_once() -> None:
    """Replace a keeper that is alive but stuck: kill it (Chrome releases its pauses) and start a
    new one at once, whose attach re-applies the brands to every running target. A keeper that
    exited on its own (browser gone, follower, failure) is left for the next acquire."""
    with _keepers_lock:
        items = list(_keepers.items())
    for port, keeper in items:
        if keeper.is_alive() and keeper.state == "serving" and keeper.heartbeat_age() > STALL_SECONDS:
            logger.warning("fidelity: keeper for port %s silent for %.0fs; replacing it now",
                           port, keeper.heartbeat_age())
            keeper.kill()
            keeper.join(3)
            with _keepers_lock:
                if _keepers.get(port) is keeper:
                    _start_keeper(port, keeper.identity)
    # Every tick, not only on start/replace/stop: a healthy keeper's file must stay live, with
    # its heartbeat and counters, or it cannot be told apart from a frozen one.
    write_status()


def _ensure_monitor() -> None:
    global _monitor
    if _monitor is not None and _monitor.is_alive():
        return

    def loop():
        import time
        while True:
            time.sleep(2.0)
            try:
                _monitor_once()
            except Exception as e:
                logger.debug("fidelity monitor: %s", e)

    _monitor = threading.Thread(target=loop, name="hermes-browser-fidelity-monitor", daemon=True)
    _monitor.start()


def ensure_keeper(port: int, identity, wait: float = 15.0) -> Optional["KeeperProcess"]:
    """Make sure a keeper serves the browser on ``port``: start one, or restart one that died,
    failed, or was a follower last time. Called on EVERY acquire of the real-profile browser, the
    cache-hit path included. ``identity`` is a dict or a zero-argument callable returning one
    (resolved only when a keeper must start). None when fidelity is off or the identity is unknown."""
    if not fidelity_enabled():
        return None
    with _keepers_lock:
        keeper = _keepers.get(port)
        if keeper is not None and keeper.healthy():
            return keeper
    if callable(identity):
        identity = identity()
    if not identity:
        return None
    with _keepers_lock:
        keeper = _keepers.get(port)
        if keeper is not None and keeper.healthy():
            return keeper
        if keeper is not None and keeper.state == "starting" and keeper.is_alive():
            pass  # another acquire is starting it right now: wait on that one
        else:
            if keeper is not None:
                if keeper.state not in ("follower",):
                    logger.warning("fidelity: keeper for port %s was %s (%s); restarting it", port, keeper.state, keeper.error)
                keeper.stop()
                keeper.join(3)
            else:
                sweep_lock_files()  # first keeper for this port in this process: tidy up old ones
            keeper = _start_keeper(port, identity)
    keeper.wait_ready(wait)
    if keeper.state == "failed":
        logger.warning("fidelity: brands not applied (%s)", keeper.error)
    write_status()
    return keeper


def stop_keepers() -> None:
    """Stop every keeper (the browser they serve is being terminated)."""
    with _keepers_lock:
        keepers = list(_keepers.values())
        _keepers.clear()
    for keeper in keepers:
        keeper.stop()
    for keeper in keepers:
        keeper.join(3)
    write_status()


# ---------------------------------------------------------------------------
# Status, for /browser status and hermes doctor (which run in other processes)
# ---------------------------------------------------------------------------

def _status_path() -> str:
    from hermes_cli.config import get_hermes_home
    return str(get_hermes_home() / "browser-fidelity-status.json")


def write_status() -> None:
    """This process's keepers, written where another process can read them. Best effort."""
    try:
        with _keepers_lock:
            keepers = [k.describe() for k in _keepers.values()]
        path = _status_path()
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            import time
            json.dump({"pid": os.getpid(), "updated": time.time(), "keepers": keepers}, fh)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug("fidelity: status not written: %s", e)


def status_summary() -> Optional[Tuple[bool, str]]:
    """``(ok, line)`` for status/doctor, or None when fidelity has nothing to report."""
    if not fidelity_enabled():
        return True, "Browser fidelity: off (browser.stealth_fidelity: false)"
    try:
        with open(_status_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        import psutil
        if not psutil.pid_exists(int(data.get("pid") or 0)):
            return None  # the process that wrote it is gone, and so is its browser
    except Exception:
        pass
    keepers = data.get("keepers") or []
    if not keepers:
        return None
    k = keepers[-1]
    if k.get("state") == "serving":
        return True, f"Browser fidelity: serving, claiming {k.get('claims')} ({sum((k.get('applied') or {}).values())} target(s))"
    if k.get("state") == "follower" and not k.get("holder_stale"):
        return True, "Browser fidelity: served by another Hermes process"
    why = "the serving process has stopped responding" if k.get("state") == "follower" else (k.get("error") or k.get("state"))
    return False, (f"Browser fidelity: NOT applied ({why}). The browser reports itself as plain Chrome for Testing; "
                   f"it recovers on the next browser use")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "keeper":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(_keeper_main(sys.argv[2:]))
