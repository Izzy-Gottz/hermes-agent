"""Native macOS backend for ``computer_use`` — no cua-driver, no extra deps.

Used when ``cua-driver`` is not installed (or ``computer_use.backend: macos``
is configured). It is deliberately small: the OS tools every Mac already has.

* **Screenshots**: ``/usr/sbin/screencapture`` (``-x`` = silent; ``-R`` for a
  region, which is how ``zoom`` works). The PNG is downscaled so its longest
  side is at most :data:`MAX_LONG_SIDE` (Anthropic's guidance: ≤ 1568 px, above
  which the API downsamples anyway and tokens are wasted) and the scale
  factor is kept so the model's screenshot-pixel coordinates map back to
  screen points.
* **Mouse**: CoreGraphics ``CGEvent``s posted through ``osascript -l
  JavaScript`` (the ObjC bridge). That gives real clicks, double clicks,
  right clicks, drags and scroll wheels at a point — System Events'
  ``click at`` cannot drag or scroll at a location.
* **Keyboard**: System Events ``keystroke`` / ``key code`` — text is passed
  out-of-band as ``argv``, never spliced into a script, so a quotation mark
  in something the model read cannot become code (same rule as Moe's
  ``mac.sh``). Key combos are validated against a fixed table.
* **Windows / apps**: ``CGWindowListCopyWindowInfo`` (every on-screen
  window, z-ordered) for ``list_windows``; System Events for the front
  window and the focused UI element.

Landing report
--------------
Every mutating action (click, drag, scroll, type, key) is followed by a
read-only focus query and the result says where the input *landed*: the
frontmost app (name + bundle id), the focused window title and the focused
element's AX role — and for text fields whether the value now ends with the
text just typed (a boolean only; the field's contents are never returned).
If the frontmost app changed because of the action the message says so
(``front app changed: Terminal → Finder``). Live sessions showed why: a
click meant for a chat box landed on a Finder window and the text went to
the Finder icon view while the tool reported ``typed 7 chars``. ``type``
therefore also refuses (``code=focus_not_editable``) when the focused
element is not text-editable, and ``key return`` refuses when Finder is
front with a non-text focus (it would open/rename the selection); both
accept ``force=True``.

Coordinate space
----------------
Everything the model sends and receives is in **screenshot pixels** of the
last capture (top-left origin). ``_to_points`` converts to screen points
(Quartz global coordinates) using the scale recorded at capture time —
Retina displays render 2× or ~1.75× pixels per point, and the downscale adds
another factor. A region capture (``zoom``) records its own origin/scale so
clicks on a zoomed image still land.

Permissions (TCC)
-----------------
``screencapture`` needs *Screen Recording* and CGEvent/System Events need
*Accessibility* — granted to the **responsible process**, which for a
Python spawned by an app bundle (Moe.app → bash → hermes → claude → this
server) is the app bundle, not python. Without Screen Recording the capture
still "succeeds" but shows only the desktop wallpaper; ``capture`` reports a
hint when the image is suspiciously uniform.

This backend is foreground-only: it drives the real cursor and keyboard
focus (unlike cua-driver's background delivery). ``delivery_mode`` is
accepted and ignored; results say ``path="cgevent_fg"`` so callers know.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend

logger = logging.getLogger(__name__)

#: Longest screenshot side handed to the model (Anthropic: ≤ 1568 px).
MAX_LONG_SIDE = 1568

_OSASCRIPT = "/usr/bin/osascript"
_SCREENCAPTURE = "/usr/sbin/screencapture"


def _screencapture_bin() -> str:
    """``$HERMES_SCREENCAPTURE_BIN`` lets tests substitute a stub that writes
    a fixture PNG (no Screen Recording grant needed)."""
    return os.environ.get("HERMES_SCREENCAPTURE_BIN") or _SCREENCAPTURE

# System Events key names for non-character keys, and CGKeyCode for `key code`.
_KEY_CODES: Dict[str, int] = {
    "return": 36, "enter": 76, "tab": 48, "space": 49, "delete": 51,
    "backspace": 51, "forward_delete": 117, "escape": 53, "esc": 53,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "home": 115, "end": 119, "pageup": 116, "page_up": 116,
    "pagedown": 121, "page_down": 121,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
    "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}
_MODIFIERS: Dict[str, str] = {
    "cmd": "command down", "command": "command down", "meta": "command down",
    "super": "command down", "win": "command down", "windows": "command down",
    "ctrl": "control down", "control": "control down",
    "alt": "option down", "option": "option down", "opt": "option down",
    "shift": "shift down", "fn": "",
}
_PLAIN_KEY = re.compile(r"^[A-Za-z0-9`\-=\[\]\\;',./]$")

#: AX roles that take typed text. Unknown ("" — query failed or app has no
#: AX tree, e.g. some Electron/Java apps) is allowed through so an opaque
#: app does not block typing; Finder is the exception (see ``type_text``).
TEXT_EDITABLE_ROLES = frozenset({
    "AXTextField", "AXTextArea", "AXComboBox", "AXWebArea", "AXGroup",
    "AXSearchField", "AXSecureTextField",
})
#: Roles that are unambiguously text entry. Finder is held to this list:
#: its desktop reports the focus as ``AXGroup`` (verified live) and its
#: icon/list views take type-to-select, so the lenient list above would let
#: the exact live failure through again.
STRICT_TEXT_ROLES = frozenset({"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"})
_FINDER_BUNDLE = "com.apple.finder"
#: Seconds to let the window server settle before the post-action focus query.
_SETTLE_S = 0.12
#: ``list_windows`` cap.
MAX_WINDOWS = 40


def is_macos() -> bool:
    return sys.platform == "darwin"


def native_backend_available() -> bool:
    """True on macOS with the two OS binaries present (always, in practice)."""
    return is_macos() and os.path.exists(_SCREENCAPTURE) and os.path.exists(_OSASCRIPT)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without a display)
# ---------------------------------------------------------------------------

def fit_scale(width: int, height: int, max_side: int = MAX_LONG_SIDE) -> float:
    """Factor (≤ 1.0) that brings the longest side to ``max_side``."""
    longest = max(width, height)
    if longest <= max_side or longest <= 0:
        return 1.0
    return max_side / float(longest)


def downscale_png(raw: bytes, max_side: int = MAX_LONG_SIDE) -> Tuple[bytes, int, int, float]:
    """Return ``(png_bytes, width, height, scale)`` with the longest side ≤ ``max_side``.

    ``scale`` is ``new/original`` (1.0 when untouched). Re-encodes as an
    8-bit RGB PNG (no alpha) which is roughly the byte cost of a high-quality
    JPEG on typical UI content without introducing artefacts around text.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(raw))
    im.load()
    w, h = im.size
    scale = fit_scale(w, h, max_side)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if scale < 1.0:
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        im = im.resize((nw, nh), Image.LANCZOS)
    else:
        nw, nh = w, h
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue(), nw, nh, scale


def scale_back(x: float, y: float, *, pixels_per_point: float, origin: Tuple[float, float] = (0.0, 0.0)) -> Tuple[int, int]:
    """Screenshot pixel → screen point. ``pixels_per_point`` is the ratio of
    the (downscaled) image size to the logical size of the captured area."""
    return (
        int(round(origin[0] + x / pixels_per_point)),
        int(round(origin[1] + y / pixels_per_point)),
    )


def parse_key_combo(keys: str) -> Tuple[List[str], str, Optional[int]]:
    """``'cmd+shift+s'`` → ``(['command down','shift down'], 's', None)``;
    ``'return'`` → ``([], 'return', 36)``. Raises ValueError on anything else.
    """
    parts = [p.strip().lower() for p in re.split(r"[+\-]", keys or "") if p.strip()]
    if not parts:
        raise ValueError("empty key combo")
    *mods, key = parts
    mod_clauses: List[str] = []
    for m in mods:
        if m not in _MODIFIERS:
            raise ValueError(f"unknown modifier {m!r}")
        clause = _MODIFIERS[m]
        if clause and clause not in mod_clauses:
            mod_clauses.append(clause)
    if key in _KEY_CODES:
        return mod_clauses, key, _KEY_CODES[key]
    if key in _MODIFIERS:
        raise ValueError("a key combo needs a non-modifier key")
    if _PLAIN_KEY.match(key):
        return mod_clauses, key, None
    raise ValueError(f"unsupported key {key!r}")


def looks_blank(png: bytes, sample: int = 64) -> bool:
    """Heuristic: an almost-uniform image (what screencapture yields with
    Screen Recording denied — wallpaper only — or a dark locked screen)."""
    try:
        from PIL import Image
    except Exception:
        return False
    im = Image.open(io.BytesIO(png)).convert("L").resize((sample, sample))
    px = list(im.getdata())
    return (max(px) - min(px)) < 12


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class MacNativeBackend(ComputerUseBackend):
    """Foreground desktop control with stock macOS tooling. See module doc."""

    def __init__(self, max_side: int = MAX_LONG_SIDE) -> None:
        self._max_side = max_side
        self._started = False
        # Geometry of the last capture, for coordinate scale-back.
        self._origin: Tuple[float, float] = (0.0, 0.0)  # screen points
        self._ppp: float = 1.0                           # image px per point
        self._last_size: Tuple[int, int] = (0, 0)
        self._last_app: Optional[str] = None
        self._last_target: Dict[str, Any] = {}
        self._shot_dir = os.environ.get("HERMES_SCREENSHOT_DIR") or os.path.join(
            tempfile.gettempdir(), "hermes-computer-use")

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def is_available(self) -> bool:
        return native_backend_available()

    # ── subprocess plumbing ──────────────────────────────────────────
    @staticmethod
    def _run(cmd: List[str], timeout: float = 20.0) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

    def _applescript(self, script: str, argv: Optional[List[str]] = None) -> str:
        """Run AppleScript with ``argv`` delivered out-of-band (``on run argv``)."""
        cmd = [_OSASCRIPT, "-"] + list(argv or [])
        proc = subprocess.run(cmd, input=script, capture_output=True, text=True,
                              timeout=20, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip()
            if "1002" in err or "not allowed assistive access" in err or "-25211" in err:
                raise RuntimeError(
                    "Accessibility permission missing for the app that launched this "
                    "process (System Settings › Privacy & Security › Accessibility): " + err)
            raise RuntimeError(err or f"osascript exit {proc.returncode}")
        return (proc.stdout or "").strip()

    def _jxa(self, script: str, argv: Optional[List[str]] = None) -> str:
        cmd = [_OSASCRIPT, "-l", "JavaScript", "-"] + list(argv or [])
        proc = subprocess.run(cmd, input=script, capture_output=True, text=True,
                              timeout=20, check=False)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "").strip() or f"osascript exit {proc.returncode}")
        return (proc.stdout or "").strip()

    # ── geometry ─────────────────────────────────────────────────────
    _JXA_MAIN_DISPLAY = (
        "ObjC.import('CoreGraphics');"
        "var b = $.CGDisplayBounds($.CGMainDisplayID());"
        "JSON.stringify([b.size.width, b.size.height])"
    )

    def screen_points(self) -> Tuple[int, int]:
        """Logical size of the **main** display in points (the one with the
        menu bar, Quartz origin 0,0 — the only display this backend captures
        and drives). CoreGraphics first: Finder's ``bounds of window of
        desktop`` spans every display on a multi-monitor Mac, which would make
        every click land short."""
        try:
            w, h = json.loads(self._jxa(self._JXA_MAIN_DISPLAY))
            w, h = int(float(w)), int(float(h))
            if w > 0 and h > 0:
                return w, h
        except Exception as exc:
            logger.debug("CGDisplayBounds lookup failed, falling back to Finder: %s", exc)
        out = self._applescript('tell application "Finder" to get bounds of window of desktop')
        nums = [int(float(n)) for n in re.findall(r"-?\d+(?:\.\d+)?", out)]
        if len(nums) == 4:
            return nums[2] - nums[0], nums[3] - nums[1]
        raise RuntimeError(f"could not read screen bounds: {out!r}")

    def _to_points(self, x: Optional[int], y: Optional[int]) -> Tuple[int, int]:
        if x is None or y is None:
            raise ValueError("this backend needs coordinate=[x, y] (no element index on macOS-native)")
        if not self._last_size[0]:
            # Without a capture there is no scale: the coordinates would be
            # treated as points, which on a Retina/downscaled screen is a
            # click somewhere else. The model must look before it acts.
            raise ValueError("no screenshot yet — run action='capture' first; coordinates are pixels of the last screenshot")
        return scale_back(float(x), float(y), pixels_per_point=self._ppp, origin=self._origin)

    # ── capture ──────────────────────────────────────────────────────
    def _screencapture(self, region: Tuple[int, int, int, int]) -> bytes:
        """One PNG of ``region`` (screen points). Always ``-R``: a bare
        ``screencapture`` on a multi-display Mac writes one file *per
        display* (``shot-x.png``, ``shot-x 1.png`` …) and only the first
        would be read and deleted here — the rest would sit in the temp dir.
        The file is 0600 (mkstemp) in a 0700 dir and unlinked as soon as it
        has been read; nothing about it is logged."""
        os.makedirs(self._shot_dir, mode=0o700, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="shot-", suffix=".png", dir=self._shot_dir)
        os.close(fd)
        try:
            x, y, w, h = region
            cmd = [_screencapture_bin(), "-x", "-R", f"{x},{y},{w},{h}", path]
            proc = self._run(cmd, timeout=15)
            if proc.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
                raise RuntimeError(
                    "screencapture failed — Screen Recording is probably not granted to the "
                    "app that launched this process (System Settings › Privacy & Security › "
                    f"Screen Recording): {(proc.stderr or '').strip()}")
            with open(path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def capture(self, mode: str = "som", app: Optional[str] = None,
                pid: Optional[int] = None, window_id: Optional[int] = None,
                region: Optional[Tuple[int, int, int, int]] = None) -> CaptureResult:
        """Full-screen (or ``region``, in screenshot pixels of the previous
        full capture) PNG, downscaled. ``mode='ax'`` returns the front window
        description only. No SOM overlays — this backend has no element list,
        so ``som`` behaves like ``vision``."""
        if app and app not in ("screen", "desktop"):
            # Best effort: bring the app forward so the capture shows it.
            try:
                self.focus_app(app, raise_window=True)
                time.sleep(0.4)
            except Exception as exc:
                logger.debug("capture: focus %s failed: %s", app, exc)
        front = self._front_window_info()
        if mode == "ax":
            return CaptureResult(mode="ax", width=0, height=0, app=front.get("app", ""),
                                 window_title=front.get("title", ""),
                                 note="macOS-native backend has no element tree; use capture(mode='vision') and coordinates. "
                                      f"front window bounds (points): {front.get('bounds')}")
        pts_w, pts_h = self.screen_points()
        origin = (0.0, 0.0)
        logical_w, logical_h = float(pts_w), float(pts_h)
        pt_region: Optional[Tuple[int, int, int, int]] = None
        if region:
            # region is in pixels of the previous capture → convert to points,
            # then clamp to the display: screencapture silently crops a region
            # that hangs off the edge, and the image it returns is then smaller
            # than the requested width — the px/pt ratio computed from the
            # requested width would be wrong and every click in the zoomed
            # image would land off target.
            rx, ry = self._to_points(region[0], region[1])
            rw = int(round(region[2] / self._ppp))
            rh = int(round(region[3] / self._ppp))
            rx, ry = max(0, min(rx, pts_w - 1)), max(0, min(ry, pts_h - 1))
            rw, rh = max(1, min(rw, pts_w - rx)), max(1, min(rh, pts_h - ry))
            pt_region = (rx, ry, rw, rh)
            origin = (float(rx), float(ry))
            logical_w, logical_h = float(rw), float(rh)
        else:
            pt_region = (0, 0, pts_w, pts_h)
        raw = self._screencapture(pt_region)
        png, w, h, _ = downscale_png(raw, self._max_side)
        self._origin = origin
        self._ppp = w / logical_w if logical_w else 1.0
        self._last_size = (w, h)
        self._last_app = front.get("app") or self._last_app
        note = (f"coordinates are screenshot pixels ({w}x{h}); they are scaled back to "
                f"screen points for you ({self._ppp:.3f} px/pt"
                + (f", region origin {origin[0]:.0f},{origin[1]:.0f} pt" if region else "") + ")")
        if looks_blank(png):
            note += ("; the image is almost uniform — if you expected windows, Screen Recording "
                     "is probably not granted to the app that launched Hermes")
        return CaptureResult(mode="vision", width=w, height=h, png_b64=base64.b64encode(png).decode(),
                             app=front.get("app", ""), window_title=front.get("title", ""),
                             png_bytes_len=len(png), image_mime_type="image/png", note=note)

    def zoom(self, region: Tuple[int, int, int, int]) -> CaptureResult:
        return self.capture(mode="vision", region=region)

    # ── mouse (CGEvent via JXA) ──────────────────────────────────────
    _JXA_MOUSE = r"""
ObjC.import('Foundation');
ObjC.import('CoreGraphics');
// argv: kind x y [x2 y2] [count] [button] — delivered by osascript, never spliced in.
function run(argv) {
var kind = argv[0], x = +argv[1], y = +argv[2];
function post(type, px, py, btn, clicks) {
  var e = $.CGEventCreateMouseEvent($(), type, {x: px, y: py}, btn);
  if (clicks) $.CGEventSetIntegerValueField(e, $.kCGMouseEventClickState, clicks);
  $.CGEventPost($.kCGHIDEventTap, e);
}
function sleep(ms) { $.NSThread.sleepForTimeInterval(ms / 1000); }
var L = $.kCGMouseButtonLeft, R = $.kCGMouseButtonRight, C = $.kCGMouseButtonCenter;
if (kind === 'move') {
  post($.kCGEventMouseMoved, x, y, L, 0);
} else if (kind === 'click') {
  var count = +argv[3], button = argv[4];
  var down = $.kCGEventLeftMouseDown, up = $.kCGEventLeftMouseUp, btn = L;
  if (button === 'right') { down = $.kCGEventRightMouseDown; up = $.kCGEventRightMouseUp; btn = R; }
  if (button === 'middle') { down = $.kCGEventOtherMouseDown; up = $.kCGEventOtherMouseUp; btn = C; }
  post($.kCGEventMouseMoved, x, y, btn, 0); sleep(30);
  for (var c = 1; c <= count; c++) { post(down, x, y, btn, c); sleep(20); post(up, x, y, btn, c); sleep(60); }
} else if (kind === 'drag') {
  var x2 = +argv[3], y2 = +argv[4];
  post($.kCGEventMouseMoved, x, y, L, 0); sleep(50);
  post($.kCGEventLeftMouseDown, x, y, L, 1); sleep(120);
  var steps = 12;
  for (var s = 1; s <= steps; s++) { post($.kCGEventLeftMouseDragged, x + (x2 - x) * s / steps, y + (y2 - y) * s / steps, L, 1); sleep(25); }
  post($.kCGEventLeftMouseUp, x2, y2, L, 1);
} else if (kind === 'scroll') {
  var dy = +argv[3], dx = +argv[4];
  post($.kCGEventMouseMoved, x, y, L, 0); sleep(30);
  var e = $.CGEventCreateScrollWheelEvent($(), $.kCGScrollEventUnitLine, 2, dy, dx);
  $.CGEventPost($.kCGHIDEventTap, e);
} else if (kind !== 'noop') {
  throw new Error('unknown mouse kind ' + kind);
}
return 'ok ' + argv.join(' ');
}
"""

    def _mouse(self, *argv: Any) -> None:
        self._jxa(self._JXA_MOUSE, [str(a) for a in argv])

    def _result(self, action: str, message: str, **meta: Any) -> ActionResult:
        return ActionResult(ok=True, action=action, message=message, meta=meta,
                            verified=False, effect="unverifiable", path="cgevent_fg",
                            delivery_mode="foreground")

    # ── focus / landing ──────────────────────────────────────────────
    # System Events rather than AXUIElement via the JXA ObjC bridge: the
    # bridge cannot type the CFTypeRef out-parameter of
    # AXUIElementCopyAttributeValue ("Ref has incompatible type"), and PyObjC
    # is not a dependency. The typed suffix is argv, never script text.
    _AS_FOCUS = '''
on run argv
  set suffix to ""
  if (count of argv) > 0 then set suffix to item 1 of argv
  tell application "System Events"
    set p to first application process whose frontmost is true
    set appName to name of p
    set bid to ""
    try
      set bid to bundle identifier of p
    end try
    set thePid to unix id of p
    set winTitle to ""
    set r to ""
    set sr to ""
    set vkind to ""
    set vmatch to "unknown"
    try
      set el to value of attribute "AXFocusedUIElement" of p
      set r to role of el
      try
        set sr to subrole of el
      end try
      try
        set v to value of el
        if class of v is text then
          set vkind to "text"
          set n to length of suffix
          if n > 0 and (length of v) >= n then
            considering case
              if (text -n thru -1 of v) is suffix then
                set vmatch to "yes"
              else
                set vmatch to "no"
              end if
            end considering
          else
            set vmatch to "no"
          end if
        end if
      end try
      try
        set w to value of attribute "AXWindow" of el
        set winTitle to name of w
      end try
    end try
    if winTitle is "" then
      try
        set winTitle to name of (first window of p whose value of attribute "AXMain" is true)
      end try
    end if
    if winTitle is missing value then set winTitle to ""
    if sr is missing value then set sr to ""
    if r is missing value then set r to ""
    return appName & tab & bid & tab & thePid & tab & winTitle & tab & r & tab & sr & tab & vkind & tab & vmatch
  end tell
end run'''

    def focused_element(self, typed_suffix: str = "") -> Dict[str, Any]:
        """Read-only: frontmost app and the focused AX element.

        Returns ``{app, bundle_id, pid, window_title, role, subrole,
        value_kind, value_ends_with_typed}``. ``value_ends_with_typed`` is
        ``True``/``False`` only when ``typed_suffix`` was given and the element
        has a text value, else ``None``; the value itself is never returned.
        On failure returns ``{"error": ...}`` with the other keys empty.
        """
        empty: Dict[str, Any] = {"app": "", "bundle_id": "", "pid": None, "window_title": "",
                                 "role": "", "subrole": "", "value_kind": "",
                                 "value_ends_with_typed": None}
        try:
            out = self._applescript(self._AS_FOCUS, [typed_suffix] if typed_suffix else [])
        except Exception as exc:
            logger.debug("focused element lookup failed: %s", exc)
            empty["error"] = str(exc).splitlines()[0][:200] if str(exc) else "focus query failed"
            return empty
        parts = out.split("\t")
        if len(parts) < 8:
            empty["error"] = "unexpected focus query output"
            return empty
        info = dict(empty)
        info.update({"app": parts[0], "bundle_id": parts[1], "window_title": parts[3],
                     "role": parts[4], "subrole": parts[5], "value_kind": parts[6]})
        try:
            info["pid"] = int(parts[2])
        except ValueError:
            pass
        if typed_suffix and parts[6] == "text":
            info["value_ends_with_typed"] = parts[7] == "yes"
        return info

    @staticmethod
    def _describe_focus(f: Dict[str, Any]) -> str:
        if f.get("error") and not f.get("app"):
            return "focus unknown (" + str(f["error"]) + ")"
        s = f.get("app") or "?"
        if f.get("window_title"):
            s += f' "{f["window_title"]}"'
        s += " " + (f.get("role") or "unknown-role")
        if f.get("subrole"):
            s += f"/{f['subrole']}"
        return s

    def _landing(self, res: ActionResult, before: Dict[str, Any],
                 typed: Optional[str] = None) -> ActionResult:
        """Decorate a mutating result with where the input landed.

        Adds ``front_app``, ``focused``, ``front_app_changed`` (and
        ``front_app_before`` when it changed) to ``meta`` and appends a
        ``→ App "title" AXRole`` clause to the message.
        """
        if _SETTLE_S:
            time.sleep(_SETTLE_S)
        after = self.focused_element(typed or "")
        res.meta["front_app"] = {"name": after.get("app", ""), "bundle_id": after.get("bundle_id", ""),
                                 "pid": after.get("pid")}
        focused = {"role": after.get("role", ""), "subrole": after.get("subrole", ""),
                   "window_title": after.get("window_title", "")}
        if typed:
            focused["value_ends_with_typed"] = after.get("value_ends_with_typed")
        if after.get("error"):
            focused["error"] = after["error"]
        res.meta["focused"] = focused
        msg = res.message + " → " + self._describe_focus(after)
        if typed:
            v = after.get("value_ends_with_typed")
            msg += " (value ends with typed text: " + ("yes" if v else "no" if v is False else "unknown") + ")"
        b_id, a_id = before.get("bundle_id") or before.get("app"), after.get("bundle_id") or after.get("app")
        changed = bool(b_id) and bool(a_id) and b_id != a_id
        res.meta["front_app_changed"] = changed
        if changed:
            res.meta["front_app_before"] = {"name": before.get("app", ""), "bundle_id": before.get("bundle_id", "")}
            msg += f"; front app changed: {before.get('app') or b_id} → {after.get('app') or a_id}"
        res.message = msg
        return res

    def _guard_refusal(self, action: str, before: Dict[str, Any], reason: str) -> ActionResult:
        return ActionResult(
            ok=False, action=action, code="focus_not_editable",
            message=f"{action} refused: {reason} — focus is {self._describe_focus(before)}. "
                    "Click the intended field first (check `focused` in the click result, or "
                    "action='focused_element'), or pass force=true to send it anyway.",
            meta={"front_app": {"name": before.get("app", ""), "bundle_id": before.get("bundle_id", ""),
                                "pid": before.get("pid")},
                  "focused": {"role": before.get("role", ""), "subrole": before.get("subrole", ""),
                              "window_title": before.get("window_title", "")}},
            verified=False, effect="suspected_noop", path="cgevent_fg", delivery_mode="foreground")

    def click(self, *, element=None, x=None, y=None, button: str = "left", click_count: int = 1,
              modifiers=None, delivery_mode=None, bring_to_front=False) -> ActionResult:
        if element is not None:
            return ActionResult(ok=False, action="click", code="no_elements",
                               message="macOS-native backend has no element indices; pass coordinate=[x, y] from the screenshot")
        px, py = self._to_points(x, y)
        before = self.focused_element()
        if modifiers:
            # Modifier-clicks go through System Events, which can hold keys.
            return self._landing(self._modifier_click(px, py, button, modifiers), before)
        self._mouse("click", px, py, max(1, int(click_count)), button)
        return self._landing(self._result(
            "click", f"{button} click x{click_count} at screenshot ({x},{y}) = screen ({px},{py}) pt",
            screen_point=[px, py]), before)

    def _modifier_click(self, px: int, py: int, button: str, modifiers: List[str]) -> ActionResult:
        clauses = []
        for m in modifiers:
            c = _MODIFIERS.get(str(m).lower())
            if c is None:
                raise ValueError(f"unknown modifier {m!r}")
            if c and c not in clauses:
                clauses.append(c)
        using = "{" + ", ".join(clauses) + "}" if clauses else "{}"
        script = (
            "on run argv\n"
            "  set px to (item 1 of argv) as integer\n  set py to (item 2 of argv) as integer\n"
            "  tell application \"System Events\"\n"
            "    key down " + using + "\n"
            "    click at {px, py}\n"
            "    key up " + using + "\n"
            "  end tell\nend run"
        )
        self._applescript(script, [str(px), str(py)])
        return self._result("click", f"{'+'.join(modifiers)}-click at screen ({px},{py}) pt", screen_point=[px, py])

    def drag(self, *, from_element=None, to_element=None, from_xy=None, to_xy=None,
             button: str = "left", modifiers=None, delivery_mode=None, bring_to_front=False) -> ActionResult:
        if not from_xy or not to_xy:
            return ActionResult(ok=False, action="drag", code="no_elements",
                               message="macOS-native backend needs from_coordinate/to_coordinate")
        x1, y1 = self._to_points(*from_xy)
        x2, y2 = self._to_points(*to_xy)
        before = self.focused_element()
        self._mouse("drag", x1, y1, x2, y2)
        return self._landing(self._result("drag", f"dragged screen ({x1},{y1}) → ({x2},{y2}) pt"), before)

    def scroll(self, *, direction: str, amount: int = 3, element=None, x=None, y=None,
               modifiers=None, delivery_mode=None, bring_to_front=False) -> ActionResult:
        if x is None or y is None:
            w, h = self._last_size
            x, y = (w // 2, h // 2) if w and h else (0, 0)
        px, py = self._to_points(x, y)
        amount = max(1, int(amount))
        dy = {"up": amount, "down": -amount}.get(direction, 0)
        dx = {"left": amount, "right": -amount}.get(direction, 0)
        if not dx and not dy:
            return ActionResult(ok=False, action="scroll", message=f"bad direction {direction!r}")
        before = self.focused_element()
        self._mouse("scroll", px, py, dy, dx)
        return self._landing(self._result("scroll", f"scrolled {direction} {amount} at screen ({px},{py}) pt"), before)

    def move(self, x: int, y: int) -> ActionResult:
        px, py = self._to_points(x, y)
        self._mouse("move", px, py)
        return self._result("move", f"pointer at screen ({px},{py}) pt", screen_point=[px, py])

    # ── keyboard (System Events, argv out-of-band) ───────────────────
    def _type_refusal(self, before: Dict[str, Any]) -> Optional[str]:
        """Why ``type`` should not proceed, or None. Finder is special-cased:
        its icon/list views take type-to-select (and a Return would open the
        selection), and an unknown role there almost always means the
        desktop, so unknown is *not* let through for Finder."""
        role = before.get("role") or ""
        if before.get("bundle_id") == _FINDER_BUNDLE:
            if role in STRICT_TEXT_ROLES:
                return None
            return ("Finder is frontmost and the focus is not a text field "
                    "(typing would type-select desktop/window icons)")
        if role in TEXT_EDITABLE_ROLES:
            return None
        if role == "":
            return None  # opaque app — fail open
        return f"focused element is {role}, not a text field"

    def type_text(self, text: str, *, delivery_mode=None, bring_to_front=False,
                  force: bool = False) -> ActionResult:
        if not text:
            return ActionResult(ok=False, action="type", message="nothing to type")
        before = self.focused_element()
        if not force:
            why = self._type_refusal(before)
            if why:
                return self._guard_refusal("type", before, why)
        # Chunk like Anthropic's reference tool so long strings don't drop keys.
        for i in range(0, len(text), 50):
            self._applescript(
                'on run argv\n  tell application "System Events" to keystroke (item 1 of argv)\nend run',
                [text[i:i + 50]])
            time.sleep(0.02)
        # Only the tail of what was typed is compared (a long paste may have
        # wrapped/auto-corrected); 50 chars is plenty to prove the field.
        return self._landing(self._result("type", f"typed {len(text)} chars"), before, typed=text[-50:])

    def key(self, keys: str, *, delivery_mode=None, bring_to_front=False,
            force: bool = False) -> ActionResult:
        mods, key, code = parse_key_combo(keys)
        before = self.focused_element()
        if (not force and not mods and key in ("return", "enter")
                and before.get("bundle_id") == _FINDER_BUNDLE
                and (before.get("role") or "") not in STRICT_TEXT_ROLES):
            return self._guard_refusal(
                "key", before, f"{keys} with Finder frontmost and a non-text focus would open or rename the selection")
        using = (" using {" + ", ".join(mods) + "}") if mods else ""
        if code is not None:
            script = f'tell application "System Events" to key code {code}{using}'
            self._applescript(script)
        else:
            # The key itself is validated to one printable ASCII char; it still
            # travels as argv rather than inside the script text.
            script = ('on run argv\n  tell application "System Events" to keystroke (item 1 of argv)'
                      + using + "\nend run")
            self._applescript(script, [key])
        return self._landing(self._result("key", f"pressed {keys}"), before)

    # ── apps / windows ───────────────────────────────────────────────
    def _front_window_info(self) -> Dict[str, Any]:
        script = '''
tell application "System Events"
  set frontApp to first application process whose frontmost is true
  set appName to name of frontApp
  try
    set w to first window of frontApp
    set {x, y} to position of w
    set {ww, hh} to size of w
    set t to name of w
    return appName & tab & t & tab & x & tab & y & tab & ww & tab & hh
  on error
    return appName & tab & "" & tab & "" & tab & "" & tab & "" & tab & ""
  end try
end tell'''
        try:
            out = self._applescript(script)
        except Exception as exc:
            logger.debug("front window lookup failed: %s", exc)
            return {}
        parts = out.split("\t")
        info: Dict[str, Any] = {"app": parts[0] if parts else ""}
        if len(parts) >= 6 and parts[2]:
            info["title"] = parts[1]
            try:
                info["bounds"] = [int(float(p)) for p in parts[2:6]]
            except ValueError:
                pass
        return info

    def list_apps(self) -> List[Dict[str, Any]]:
        out = self._applescript(
            'tell application "System Events" to get name of every application process whose visible is true')
        return [{"name": n.strip()} for n in out.split(",") if n.strip()]

    _JXA_WINDOWS = r"""
ObjC.import('Cocoa');
ObjC.import('CoreGraphics');
function run() {
  var b = $.CGDisplayBounds($.CGMainDisplayID());
  var fa = $.NSWorkspace.sharedWorkspace.frontmostApplication;
  var front = fa.isNil() ? null : {name: ObjC.unwrap(fa.localizedName), bundle_id: ObjC.unwrap(fa.bundleIdentifier), pid: fa.processIdentifier};
  var opts = $.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements;
  var arr = ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo(opts, $.kCGNullWindowID))) || [];
  var out = [];
  for (var i = 0; i < arr.length; i++) {
    var w = arr[i];
    if (w.kCGWindowLayer !== 0) continue;
    var bb = w.kCGWindowBounds || {};
    out.push({app: w.kCGWindowOwnerName || '', pid: w.kCGWindowOwnerPID, window_id: w.kCGWindowNumber,
              title: w.kCGWindowName || '', x: bb.X, y: bb.Y, w: bb.Width, h: bb.Height, alpha: w.kCGWindowAlpha});
  }
  return JSON.stringify({display: [b.size.width, b.size.height], front: front, windows: out});
}
"""

    LIST_WINDOWS_NOTE = ("z-ordered front → back, on-screen windows of the current Space only "
                         "(minimized windows and other Spaces are not listed); "
                         "`frontmost` marks the active app's top window")

    def list_windows(self) -> List[Dict[str, Any]]:
        """Every on-screen, layer-0 window of every app, front → back
        (``CGWindowListCopyWindowInfo``), capped at :data:`MAX_WINDOWS`.
        Falls back to the single System Events front window if the JXA
        call fails."""
        try:
            data = json.loads(self._jxa(self._JXA_WINDOWS))
        except Exception as exc:
            logger.debug("CGWindowList lookup failed, falling back to System Events: %s", exc)
            return self._list_windows_fallback()
        disp = data.get("display") or [0, 0]
        dw, dh = float(disp[0] or 0), float(disp[1] or 0)
        front = data.get("front") or {}
        front_pid = front.get("pid")
        result: List[Dict[str, Any]] = []
        for w in data.get("windows") or []:
            if len(result) >= MAX_WINDOWS:
                break
            try:
                x, y, ww, hh = (int(round(float(w.get(k) or 0))) for k in ("x", "y", "w", "h"))
            except (TypeError, ValueError):
                x = y = ww = hh = 0
            if ww < 2 or hh < 2:
                continue  # 1-px helper windows (Electron, status items)
            is_front_app = front_pid is not None and w.get("pid") == front_pid
            entry: Dict[str, Any] = {
                "app": w.get("app") or "", "pid": w.get("pid"), "window_id": w.get("window_id"),
                "title": w.get("title") or "", "z": len(result),
                "bounds_points": {"x": x, "y": y, "w": ww, "h": hh},
                "frontmost": False,
                "front_app": is_front_app,
            }
            if front.get("bundle_id") and is_front_app:
                entry["bundle_id"] = front["bundle_id"]
            if self._ppp and self._last_size[0]:
                entry["bounds_screenshot_px"] = {
                    "x": int(round((x - self._origin[0]) * self._ppp)),
                    "y": int(round((y - self._origin[1]) * self._ppp)),
                    "w": int(round(ww * self._ppp)), "h": int(round(hh * self._ppp))}
            notes = []
            if dw and dh:
                if x <= 0 and y <= 0 and ww >= dw and hh >= dh:
                    notes.append("fullscreen")
                if x >= dw or y >= dh or x + ww <= 0 or y + hh <= 0:
                    notes.append("off main display (not in screenshots)")
            if w.get("alpha") is not None and float(w["alpha"]) < 0.05:
                notes.append("transparent")
            if notes:
                entry["note"] = "; ".join(notes)
            result.append(entry)
        # `frontmost` = the front app's top *titled* window. Chrome/Electron
        # put untitled helper windows (tab strip, 41 px tall; tooltips) above
        # the real window in CGWindowList z-order (verified live); fall back
        # to the top untitled one only when nothing of the front app has a title.
        front_entries = [e for e in result if e["front_app"]]
        top = next((e for e in front_entries if e["title"]), front_entries[0] if front_entries else None)
        if top is not None:
            top["frontmost"] = True
        return result

    def _list_windows_fallback(self) -> List[Dict[str, Any]]:
        info = self._front_window_info()
        if not info:
            return []
        entry = {"app": info.get("app", ""), "title": info.get("title", ""), "frontmost": True,
                 "front_app": True, "z": 0, "note": "CGWindowList unavailable; front window only"}
        if info.get("bounds"):
            x, y, w, h = info["bounds"]
            entry["bounds_points"] = {"x": x, "y": y, "w": w, "h": h}
            if self._ppp and self._last_size[0]:
                entry["bounds_screenshot_px"] = {
                    "x": int(round((x - self._origin[0]) * self._ppp)),
                    "y": int(round((y - self._origin[1]) * self._ppp)),
                    "w": int(round(w * self._ppp)), "h": int(round(h * self._ppp))}
        return [entry]

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        # `open -a` both launches and activates; it validates the name for us.
        proc = self._run(["/usr/bin/open", "-a", app], timeout=15)
        if proc.returncode != 0:
            return ActionResult(ok=False, action="focus_app", message=f"no such app: {app}")
        self._last_app = app
        return self._result("focus_app", f"activated {app}")

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        return ActionResult(ok=False, action="set_value", code="unsupported",
                           message="set_value needs an element tree; macOS-native backend has none — click the control and type instead")

    def wait(self, seconds: float) -> ActionResult:
        time.sleep(max(0.0, min(float(seconds), 30.0)))
        return ActionResult(ok=True, action="wait", message=f"waited {seconds:.2f}s")


__all__ = [
    "MAX_LONG_SIDE", "MAX_WINDOWS", "STRICT_TEXT_ROLES", "TEXT_EDITABLE_ROLES", "MacNativeBackend", "downscale_png", "fit_scale", "looks_blank",
    "native_backend_available", "parse_key_combo", "scale_back",
]
