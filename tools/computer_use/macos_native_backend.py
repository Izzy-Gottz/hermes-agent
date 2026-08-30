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
* **Windows / apps**: System Events.

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
    def screen_points(self) -> Tuple[int, int]:
        """Logical size of the main display in points."""
        out = self._applescript('tell application "Finder" to get bounds of window of desktop')
        nums = [int(float(n)) for n in re.findall(r"-?\d+(?:\.\d+)?", out)]
        if len(nums) == 4:
            return nums[2] - nums[0], nums[3] - nums[1]
        raise RuntimeError(f"could not read screen bounds: {out!r}")

    def _to_points(self, x: Optional[int], y: Optional[int]) -> Tuple[int, int]:
        if x is None or y is None:
            raise ValueError("this backend needs coordinate=[x, y] (no element index on macOS-native)")
        return scale_back(float(x), float(y), pixels_per_point=self._ppp, origin=self._origin)

    # ── capture ──────────────────────────────────────────────────────
    def _screencapture(self, region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        os.makedirs(self._shot_dir, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="shot-", suffix=".png", dir=self._shot_dir)
        os.close(fd)
        try:
            cmd = [_screencapture_bin(), "-x"]
            if region:
                x, y, w, h = region
                cmd += ["-R", f"{x},{y},{w},{h}"]
            cmd.append(path)
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
            # region is in pixels of the previous capture → convert to points.
            rx, ry = self._to_points(region[0], region[1])
            rw = max(1, int(round(region[2] / self._ppp)))
            rh = max(1, int(round(region[3] / self._ppp)))
            pt_region = (rx, ry, rw, rh)
            origin = (float(rx), float(ry))
            logical_w, logical_h = float(rw), float(rh)
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

    def click(self, *, element=None, x=None, y=None, button: str = "left", click_count: int = 1,
              modifiers=None, delivery_mode=None, bring_to_front=False) -> ActionResult:
        if element is not None:
            return ActionResult(ok=False, action="click", code="no_elements",
                               message="macOS-native backend has no element indices; pass coordinate=[x, y] from the screenshot")
        px, py = self._to_points(x, y)
        if modifiers:
            # Modifier-clicks go through System Events, which can hold keys.
            return self._modifier_click(px, py, button, modifiers)
        self._mouse("click", px, py, max(1, int(click_count)), button)
        return self._result("click", f"{button} click x{click_count} at screenshot ({x},{y}) = screen ({px},{py}) pt",
                            screen_point=[px, py])

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
        self._mouse("drag", x1, y1, x2, y2)
        return self._result("drag", f"dragged screen ({x1},{y1}) → ({x2},{y2}) pt")

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
        self._mouse("scroll", px, py, dy, dx)
        return self._result("scroll", f"scrolled {direction} {amount} at screen ({px},{py}) pt")

    def move(self, x: int, y: int) -> ActionResult:
        px, py = self._to_points(x, y)
        self._mouse("move", px, py)
        return self._result("move", f"pointer at screen ({px},{py}) pt", screen_point=[px, py])

    # ── keyboard (System Events, argv out-of-band) ───────────────────
    def type_text(self, text: str, *, delivery_mode=None, bring_to_front=False) -> ActionResult:
        if not text:
            return ActionResult(ok=False, action="type", message="nothing to type")
        # Chunk like Anthropic's reference tool so long strings don't drop keys.
        for i in range(0, len(text), 50):
            self._applescript(
                'on run argv\n  tell application "System Events" to keystroke (item 1 of argv)\nend run',
                [text[i:i + 50]])
            time.sleep(0.02)
        return self._result("type", f"typed {len(text)} chars into the focused field")

    def key(self, keys: str, *, delivery_mode=None, bring_to_front=False) -> ActionResult:
        mods, key, code = parse_key_combo(keys)
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
        return self._result("key", f"pressed {keys}")

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

    def list_windows(self) -> List[Dict[str, Any]]:
        info = self._front_window_info()
        if not info:
            return []
        entry = {"app": info.get("app", ""), "title": info.get("title", ""), "frontmost": True}
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
    "MAX_LONG_SIDE", "MacNativeBackend", "downscale_png", "fit_scale", "looks_blank",
    "native_backend_available", "parse_key_combo", "scale_back",
]
