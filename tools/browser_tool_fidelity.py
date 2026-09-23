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
* ``--user-agent``: the person's browser's own reduced UA string (read from the installed app's
  ``Info.plist``, never by launching it). The flag reaches every target the browser creates,
  including new tabs, popups and workers, with no race.
* ``--window-size`` / ``--screen-info``: the main display's real size and scale.
* Brands: there is no switch for ``Sec-CH-UA``, so a :class:`FidelityKeeper` holds one
  browser-level DevTools connection for the browser's lifetime. It auto-attaches every target
  paused at start (``waitForDebuggerOnStart``), applies ``Emulation.setUserAgentOverride`` (pages,
  frames) or ``Network.setUserAgentOverride`` (workers) with the matching
  ``userAgentMetadata``, and only then resumes the target. A DevTools override lasts only as long
  as the session that set it, which is why the keeper stays connected rather than applying
  the override once and leaving.

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


def persons_browser_identity(browser: Optional[str], binary: Optional[str]) -> Optional[Dict[str, Any]]:
    """``{"brand", "ua_suffix", "full_version", "major"}`` of the person's browser, or None when its
    version cannot be read. The identity is theirs; nothing here is invented."""
    if browser not in _BRAND_BY_BROWSER or not binary:
        return None
    version = installed_browser_version(binary)
    if not version:
        return None
    brand, suffix = _BRAND_BY_BROWSER[browser]
    return {"brand": brand, "ua_suffix": suffix, "full_version": version, "major": int(version.split(".")[0])}


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


def main_display() -> Optional[Tuple[int, int, float]]:
    """``(width, height, scale)`` of the main display in points, from CoreGraphics (no window,
    no permission needed), or None off macOS / on failure."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        import ctypes.util
        cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics") or
                         "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
        cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
        cg.CGDisplayCopyDisplayMode.restype = ctypes.c_void_p
        cg.CGDisplayCopyDisplayMode.argtypes = [ctypes.c_uint32]
        cg.CGDisplayModeGetPixelWidth.restype = ctypes.c_size_t
        cg.CGDisplayModeGetPixelWidth.argtypes = [ctypes.c_void_p]
        cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]
        display = cg.CGMainDisplayID()
        width, height = int(cg.CGDisplayPixelsWide(display)), int(cg.CGDisplayPixelsHigh(display))
        scale = 1.0
        mode = cg.CGDisplayCopyDisplayMode(display)
        if mode:
            pixel_width = int(cg.CGDisplayModeGetPixelWidth(mode))
            cg.CGDisplayModeRelease(mode)
            if width and pixel_width:
                scale = round(pixel_width / width, 2)
        return (width, height, scale) if width > 0 and height > 0 else None
    except Exception as e:  # never let a display probe fail a launch
        logger.debug("fidelity: main display probe failed: %s", e)
        return None


def launch_flags(identity: Optional[Dict[str, Any]], headless: bool,
                 display: Optional[Tuple[int, int, float]] = None) -> List[str]:
    """Switches for the driven browser. Always ``AutomationControlled`` off. The person's UA
    when their browser's identity is known. With headless, the main display's real size, because
    headless otherwise reports an 800x600 screen and a 0x0 outer window."""
    flags = ["--disable-blink-features=AutomationControlled"]
    if identity:
        flags.append(f"--user-agent={user_agent(identity)}")
    if headless:
        width, height, scale = display or (1440, 900, 2.0)
        win_w, win_h = default_window_size(width, height)
        flags.append(f"--window-size={win_w},{win_h}")
        # --screen-info takes the size in device pixels: {1470x956 devicePixelRatio=2} reads back as
        # a 735x478 screen (measured), so the point size is scaled up here.
        flags.append(f"--screen-info={{{round(width * scale)}x{round(height * scale)} devicePixelRatio={scale:g}}}")
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


def commands_for_attached_target(params: Dict[str, Any], ua: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """The commands, in order, for one ``Target.attachedToTarget`` event: the override first, then
    auto-attach one level down (dedicated workers, out-of-process frames), then resume. A target
    paused by ``waitForDebuggerOnStart`` runs no script and sends no request before the resume,
    so the first request it sends already carries the brands. Commands on one session run in order."""
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


class FidelityKeeper(threading.Thread):
    """One browser-level DevTools connection that puts the person's brands on every target.

    Daemon thread: it never keeps Hermes alive, and it ends when the browser's socket closes
    (the browser was terminated) or :meth:`stop` is called. ``applied`` counts overrides sent,
    per target type, for tests and logs."""

    def __init__(self, port: int, ua: str, metadata: Dict[str, Any], connect=None):
        super().__init__(name=f"hermes-browser-fidelity-{port}", daemon=True)
        self.port, self.ua, self.metadata = port, ua, metadata
        self._connect = connect
        self._ws = None
        self._halt = threading.Event()
        self.ready = threading.Event()
        self.applied: Dict[str, int] = {}
        self.error: Optional[str] = None
        self._next_id = 0
        self._send_lock = threading.Lock()

    def _send(self, method: str, params: Dict[str, Any], session: Optional[str] = None) -> None:
        with self._send_lock:
            self._next_id += 1
            msg: Dict[str, Any] = {"id": self._next_id, "method": method, "params": params}
            if session:
                msg["sessionId"] = session
            self._ws.send(json.dumps(msg))

    def _open(self):
        if self._connect is not None:
            return self._connect(self.port)
        import requests
        from websockets.sync.client import connect
        # A browser that has just written its port can take a few seconds to answer under load.
        for attempt in range(4):
            try:
                ws_url = requests.get(f"http://127.0.0.1:{self.port}/json/version", timeout=3).json()["webSocketDebuggerUrl"]
                break
            except requests.RequestException:
                if attempt == 3 or self._halt.is_set():
                    raise
        return connect(ws_url, max_size=None, open_timeout=10, close_timeout=2)

    def handle(self, msg: Dict[str, Any]) -> None:
        """React to one DevTools message (public for tests)."""
        if msg.get("method") == "Target.attachedToTarget":
            params = msg.get("params") or {}
            session = params.get("sessionId")
            kind = (params.get("targetInfo") or {}).get("type", "?")
            for method, cmd_params in commands_for_attached_target(params, self.ua, self.metadata):
                self._send(method, cmd_params, session)
            self.applied[kind] = self.applied.get(kind, 0) + 1
        elif "error" in msg and msg.get("id"):
            # A target that went away between attach and override answers with an error; harmless.
            logger.debug("fidelity: command %s failed: %s", msg.get("id"), msg["error"])

    def run(self) -> None:
        try:
            self._ws = self._open()
            # Browser level: attach every existing and future top-level target, paused at start.
            self._send("Target.setAutoAttach", {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True})
            self.ready.set()
            while not self._halt.is_set():
                try:
                    raw = self._ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                self.handle(json.loads(raw))
        except Exception as e:  # the browser went away, or never answered
            if not self._halt.is_set():
                self.error = f"{type(e).__name__}: {e}"
                logger.info("fidelity keeper for port %s ended: %s", self.port, self.error)
        finally:
            self.ready.set()
            try:
                if self._ws is not None:
                    self._ws.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._halt.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass


_keepers: Dict[int, FidelityKeeper] = {}
_keepers_lock = threading.Lock()


def ensure_keeper(port: int, identity: Optional[Dict[str, Any]], wait: float = 15.0) -> Optional[FidelityKeeper]:
    """Start (once per debug port) the keeper that puts the person's brands on every target.
    Returns the keeper, or None when fidelity is off or the identity is unknown."""
    if not identity or not fidelity_enabled():
        return None
    with _keepers_lock:
        keeper = _keepers.get(port)
        if keeper is not None and keeper.is_alive():
            return keeper
        keeper = FidelityKeeper(port, user_agent(identity), user_agent_metadata(identity))
        _keepers[port] = keeper
        keeper.start()
    keeper.ready.wait(wait)
    if keeper.error:
        logger.warning("fidelity: brands not applied (%s); the UA string and webdriver switches still are", keeper.error)
    return keeper


def stop_keepers() -> None:
    """Stop every keeper (the browser they serve is being terminated)."""
    with _keepers_lock:
        keepers = list(_keepers.values())
        _keepers.clear()
    for keeper in keepers:
        keeper.stop()
