"""Pointer input for the CAPTCHA ladder: a person-shaped path, and the ways a click reaches the page.

Two delivery strategies, chosen per attempt (:func:`pick_delivery`):

* :class:`CdpDelivery` -- ``Input.dispatchMouseEvent`` on the page's CDP session. Works on the headless
  own-lane browser. Chromium issue 40280325 reports that CDP-dispatched events inside an iframe carry
  ``screenX/screenY == clientX/clientY`` (a tell no real mouse makes). MEASURED on this Mac, 2026-09-25,
  Chrome for Testing 154.0.8037.57 ``--headless=new``, window moved to (300, 200), a cross-site OOPIF
  (127.0.0.1 page, localhost iframe at 200,150), clicked over CDP at viewport (300, 200) = iframe-local
  (100, 50): the iframe saw ``clientX=100 clientY=50 screenX=600 screenY=543 isTrusted=true`` --
  screenX = 300 window + 200 frame + 100, screenY = 200 window + 143 top inset + 150 frame + 50. The
  top page saw ``clientX=70 screenX=370``. So on this build the bug does NOT reproduce: screen
  coordinates are real window-relative screen coordinates in the OOPIF too. ``tests/tools/
  test_browser_captcha.py::test_live_cdp_iframe_screen_coordinates`` re-measures it whenever Chrome for
  Testing is present, and fails if the bug comes back.
* :class:`CuaDriverDelivery` -- a genuine OS-level click through cua-driver (``delivery_mode=
  "foreground"`` posts real HID events), for when the driven browser has a window the person's screen
  actually shows (``browser.headed``, or during a hand-off). The own lane runs ``--headless=new``
  (tools/browser_tool_real_profile.py), which has no on-screen window at all, so there it is never
  available and CDP is used. The viewport -> window mapping is measured (the 143 px top inset above is
  ``outerHeight - innerHeight`` in that same run); the end-to-end OS click is NOT verified live yet --
  :data:`CuaDriverDelivery.verified` says so and the ladder reports which strategy ran.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Box:
    """A target rectangle in top-level viewport CSS pixels."""
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point:
        return self.x + self.width / 2, self.y + self.height / 2

    def contains(self, p: Point) -> bool:
        return self.x <= p[0] <= self.x + self.width and self.y <= p[1] <= self.y + self.height


# ---- the path ----------------------------------------------------------------------------------------

MIN_STEPS, MAX_STEPS = 8, 30
MIN_MS, MAX_MS = 300, 900


def point_inside(box: Box, rng: random.Random) -> Point:
    """A point in the middle 20-80 % of the box: a person aims at the target, never its exact centre."""
    return (box.x + box.width * rng.uniform(0.2, 0.8), box.y + box.height * rng.uniform(0.25, 0.75))


def human_path(start: Point, end: Point, rng: Optional[random.Random] = None) -> Tuple[List[Point], List[float]]:
    """``(points, delays_s)``: a curved path from ``start`` to ``end`` (a quadratic Bezier bowed to one side),
    8-30 steps over 300-900 ms, eased (slow-fast-slow), a little jitter, ending exactly on ``end``.
    ``delays_s[i]`` is the pause before ``points[i]``."""
    rng = rng or random.Random()
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = max(MIN_STEPS, min(MAX_STEPS, int(8 + dist / 25 + rng.uniform(0, 6))))
    total_s = rng.uniform(MIN_MS, MAX_MS) / 1000.0
    mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    nx, ny = -(end[1] - start[1]), end[0] - start[0]
    norm = math.hypot(nx, ny) or 1.0
    bow = rng.uniform(0.08, 0.3) * dist * rng.choice((-1, 1))
    cx, cy = mx + nx / norm * bow, my + ny / norm * bow
    pts: List[Point] = []
    for i in range(1, steps + 1):
        t = i / steps
        e = t * t * (3 - 2 * t)  # smoothstep easing
        x = (1 - e) ** 2 * start[0] + 2 * (1 - e) * e * cx + e ** 2 * end[0]
        y = (1 - e) ** 2 * start[1] + 2 * (1 - e) * e * cy + e ** 2 * end[1]
        if i < steps:
            x += rng.uniform(-0.8, 0.8)
            y += rng.uniform(-0.8, 0.8)
        pts.append((x, y))
    pts[-1] = end
    weights = [rng.uniform(0.7, 1.3) for _ in pts]
    scale = total_s / sum(weights)
    return pts, [w * scale for w in weights]


# ---- delivery strategies -------------------------------------------------------------------------------

class Delivery:
    name = "none"
    verified = False

    def available(self) -> Tuple[bool, str]:
        return False, "not implemented"

    def click(self, box: Box, rng: random.Random) -> Point:
        raise NotImplementedError

    def hold(self, box: Box, rng: random.Random, until: Callable[[], Optional[bool]], max_s: float,
             poll_s: float = 0.2) -> Tuple[Optional[bool], float]:
        raise NotImplementedError


class CdpDelivery(Delivery):
    """``Input.dispatchMouseEvent`` on a page session (``page.call(method, params)``)."""
    name = "cdp"
    verified = True

    def __init__(self, page: Any, sleep: Callable[[float], None] = time.sleep, start: Optional[Point] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.page = page
        self.sleep = sleep
        self.clock = clock
        self.pos: Optional[Point] = start

    def available(self) -> Tuple[bool, str]:
        return True, "cdp"

    def _start(self, rng: random.Random, box: Box) -> Point:
        if self.pos is None:  # enter from somewhere plausible below/left of the target
            self.pos = (max(1.0, box.x - rng.uniform(80, 260)), box.y + rng.uniform(60, 220))
        return self.pos

    def _move(self, to: Point, rng: random.Random, box: Box, buttons: int = 0) -> None:
        pts, delays = human_path(self._start(rng, box), to, rng)
        for (x, y), d in zip(pts, delays):
            self.sleep(d)
            self.page.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y, "buttons": buttons,
                                                        "pointerType": "mouse"})
        self.pos = to

    def _button(self, kind: str, p: Point) -> None:
        self.page.call("Input.dispatchMouseEvent", {"type": kind, "x": p[0], "y": p[1], "button": "left",
                                                    "buttons": 1 if kind == "mousePressed" else 0, "clickCount": 1,
                                                    "pointerType": "mouse"})

    def click(self, box: Box, rng: random.Random) -> Point:
        target = point_inside(box, rng)
        self._move(target, rng, box)
        self.sleep(rng.uniform(0.04, 0.16))
        self._button("mousePressed", target)
        self.sleep(rng.uniform(0.05, 0.14))
        self._button("mouseReleased", target)
        return target

    def hold(self, box: Box, rng: random.Random, until: Callable[[], Optional[bool]], max_s: float,
             poll_s: float = 0.2) -> Tuple[Optional[bool], float]:
        """Press inside ``box`` and keep it pressed, polling ``until()`` -- True (done: release), False (failed:
        release), None (keep holding) -- for at most ``max_s``. No fixed hold time: the page decides.
        A pressed hand is never perfectly still: an occasional sub-pixel drift with the button down."""
        target = point_inside(box, rng)
        self._move(target, rng, box)
        self.sleep(rng.uniform(0.05, 0.15))
        self._button("mousePressed", target)
        started = self.clock()
        verdict: Optional[bool] = None
        try:
            while self.clock() - started < max_s:
                self.sleep(poll_s)
                if rng.random() < 0.3:
                    jx, jy = target[0] + rng.uniform(-0.6, 0.6), target[1] + rng.uniform(-0.6, 0.6)
                    self.page.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": jx, "y": jy, "buttons": 1,
                                                                "pointerType": "mouse"})
                verdict = until()
                if verdict is not None:
                    break
        finally:
            self.sleep(rng.uniform(0.03, 0.1))
            self._button("mouseReleased", target)
        return verdict, self.clock() - started


@dataclass
class WindowGeometry:
    """Where the page's viewport sits on screen, in points (CSS px at zoom 1): window origin plus the
    browser's own top/left inset (``outerHeight - innerHeight`` / ``outerWidth - innerWidth``)."""
    left: float
    top: float
    inset_x: float
    inset_y: float
    visible: bool
    headless: bool

    def to_window(self, p: Point) -> Point:
        """Viewport point -> window-local point (what cua-driver's coordinate click addresses)."""
        return p[0] + self.inset_x, p[1] + self.inset_y

    def to_screen(self, p: Point) -> Point:
        wx, wy = self.to_window(p)
        return self.left + wx, self.top + wy


GEOMETRY_JS = ("JSON.stringify({sx: screenX, sy: screenY, ix: outerWidth - innerWidth, iy: outerHeight - innerHeight,"
               " vis: document.visibilityState, ua: navigator.userAgent})")


def window_geometry(page: Any) -> Optional[WindowGeometry]:
    import json
    try:
        raw = page.call("Runtime.evaluate", {"expression": GEOMETRY_JS, "returnByValue": True})
        g = json.loads((raw.get("result") or {}).get("value") or "{}")
    except Exception:
        return None
    return WindowGeometry(left=float(g.get("sx") or 0), top=float(g.get("sy") or 0), inset_x=float(g.get("ix") or 0),
                          inset_y=float(g.get("iy") or 0), visible=g.get("vis") == "visible",
                          headless="HeadlessChrome" in str(g.get("ua") or ""))


class CuaDriverDelivery(Delivery):
    """A real OS click through cua-driver on the browser's window, when that window is on screen.

    ``backend`` is a ``tools.computer_use.backend.ComputerUseBackend`` already captured on the browser's
    window (pid + window_id); coordinates are window-local, as its ``click(x=, y=)`` expects. Not
    verified end to end on this Mac yet (``verified = False``); the ladder only picks it when
    ``browser.captcha.os_clicks`` is on AND the window is visible, so it cannot change the default path."""
    name = "cua_driver"
    verified = False

    def __init__(self, backend: Any, geometry: Optional[WindowGeometry], scale: float = 1.0):
        self.backend = backend
        self.geometry = geometry
        self.scale = scale  # window points -> capture pixels (a Retina capture is 2.0)

    def available(self) -> Tuple[bool, str]:
        g = self.geometry
        if self.backend is None:
            return False, "cua-driver is not available"
        if g is None:
            return False, "the browser window's geometry could not be read"
        if g.headless:
            return False, "the browser is headless: it has no on-screen window for a real click"
        if not g.visible:
            return False, "the browser window is not visible (hidden, minimised, or on another Space)"
        return True, "cua_driver"

    def click(self, box: Box, rng: random.Random) -> Point:
        target = point_inside(box, rng)
        wx, wy = self.geometry.to_window(target)  # type: ignore[union-attr]
        res = self.backend.click(x=int(round(wx * self.scale)), y=int(round(wy * self.scale)),
                                 delivery_mode="foreground")
        if not getattr(res, "ok", False):
            raise RuntimeError(f"cua-driver click refused: {getattr(res, 'message', '') or getattr(res, 'code', '')}")
        return target

    def hold(self, box, rng, until, max_s, poll_s=0.2):
        # cua-driver exposes click and drag, not press-without-release; a hold needs CDP (or a future
        # driver verb). Refuse loudly rather than fake it with a zero-length drag.
        raise NotImplementedError("press-and-hold is not available through cua-driver")


def pick_delivery(page: Any, *, os_clicks: bool = False, backend_factory: Optional[Callable[[], Any]] = None,
                  sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic) -> Tuple[Delivery, str]:
    """``(strategy, why)``. CDP unless OS clicks are enabled and the window is really on screen."""
    if os_clicks:
        geometry = window_geometry(page)
        backend = None
        if geometry is not None and not geometry.headless and geometry.visible and backend_factory is not None:
            try:
                backend = backend_factory()
            except Exception:
                backend = None
        cua = CuaDriverDelivery(backend, geometry)
        ok, why = cua.available()
        if ok:
            return cua, why
        return CdpDelivery(page, sleep=sleep, clock=clock), f"cdp ({why})"
    return CdpDelivery(page, sleep=sleep, clock=clock), "cdp"
