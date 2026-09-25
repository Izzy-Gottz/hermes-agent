"""Tier C of the CAPTCHA ladder: drive an image grid with a recognition solver's answer.

Only reCAPTCHA v2 and hCaptcha image grids, only after tier A (the checkbox) did not pass, and only with
a solver the owner configured with their own key (tools/browser_captcha_solvers.py). One round is:

  read the challenge frame (instruction, tile boxes, verify button) -> screenshot the grid in Moe's own
  browser -> ask the solver which tiles match -> click those tiles on the person-shaped pointer path
  (tools/browser_captcha_input.py) -> for a dynamic grid ("click verify once there are none left"), wait
  for the clicked tiles to be replaced and ask again about just those -> click verify -> re-check.

Bounds: :data:`MAX_ROUNDS` rounds per challenge, :data:`MAX_SOLVES_PER_TASK` solver requests per task
(each one is a paid request, so this is the cost ceiling), and :data:`TIER_C_S` seconds of wall clock.
When any of them runs out, or the solver cannot answer, the challenge goes to the person -- the ladder
never guesses a tile.

The frame DOM readers below follow reCAPTCHA's bframe (``rc-imageselect-*``) and hCaptcha's challenge
frame (``.task-image``, ``.prompt-text``, ``.button-submit``) as they are widely documented; neither
vendor's test sitekey ever shows an image challenge, so they are exercised live only against the local
fake bframe in tests/tools/test_browser_captcha_solver.py, not the real widgets. A frame this cannot read
(hCaptcha's canvas-drawn challenges, for one) is handed to the person.
"""

from __future__ import annotations

import base64
import json
import re
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

from tools import browser_captcha as bc
from tools.browser_captcha_input import Box
from tools.browser_captcha_solver import SolveRequest, SolverError, SolverUnsupported, validate_answer

if TYPE_CHECKING:  # pragma: no cover
    from tools.browser_captcha_ladder import Ladder, Outcome, Surface

GRID_KINDS = frozenset({bc.RECAPTCHA_V2, bc.HCAPTCHA})
MAX_ROUNDS = 3               # challenges (verify clicks) per tier C run
MAX_SOLVES_PER_TASK = 5      # solver requests per task, across every challenge in it
MAX_WAVES = 4                # dynamic-grid re-asks per round
TIER_C_S = 60.0              # wall clock for one tier C run
SETTLE_S = 8.0               # how long a grid may take to appear / replace its tiles
VERIFY_WATCH_S = 8.0         # after verify: how long to watch for the pass or the next challenge


@dataclass
class Grid:
    vendor: str                      # "recaptcha" | "hcaptcha"
    instruction: str                 # the challenge's own sentence, e.g. "Select all images with buses"
    target: str                      # the object word(s), e.g. "buses"
    rows: int
    cols: int
    tiles: List[Box]                 # top-level viewport CSS px, row-major
    selected: List[bool]
    srcs: List[str]                  # per-tile image source: a replaced tile gets a new one
    area: Box                        # the whole tile grid, for the screenshot
    verify: Optional[Box] = None
    reload: Optional[Box] = None
    dynamic: bool = False
    busy: bool = False               # tiles still animating / loading
    error: str = ""

    @property
    def sig(self) -> str:
        return self.instruction + "|" + "|".join(self.srcs)


# ---- the solve budget (per task) ------------------------------------------------------------------------------

_lock = threading.Lock()
_solves: dict = {}


def solves_used(task_id: Optional[str]) -> int:
    with _lock:
        return _solves.get(str(task_id or "default"), 0)


def _take_solve(task_id: Optional[str]) -> bool:
    with _lock:
        k = str(task_id or "default")
        if _solves.get(k, 0) >= MAX_SOLVES_PER_TASK:
            return False
        _solves[k] = _solves.get(k, 0) + 1
        return True


def reset_solves() -> None:
    with _lock:
        _solves.clear()


# ---- the driver -------------------------------------------------------------------------------------------------

_TAIL = re.compile(r"\s*(?:click verify once there are none left|if there are none,? click skip|"
                   r"please (?:also )?check (?:the )?new images|please select all matching images|"
                   r"please try again).*$", re.I | re.S)


def instruction_line(text: str) -> str:
    """The challenge's sentence without reCAPTCHA's trailing how-to ("Click verify once ...")."""
    return _TAIL.sub("", re.sub(r"\s+", " ", text or "")).strip()


def _settled(lad: "Ladder", surface: "Surface", ch: bc.Challenge, end: float, *, before: Optional[Grid] = None,
             replaced: Set[int] = frozenset()) -> Optional[Grid]:
    """The grid once it is readable and still -- and, with ``replaced``, once those tiles have new images."""
    until = min(lad.clock() + SETTLE_S, end)
    while True:
        g = surface.grid(ch)
        if g is not None and g.tiles and not g.busy:
            fresh = all(i < len(g.srcs) and before is not None and i < len(before.srcs)
                        and g.srcs[i] != before.srcs[i] for i in replaced)
            if fresh or not replaced:
                return g
        if lad.clock() >= until:
            return g if (g is not None and g.tiles and not replaced) else None
        lad.sleep(0.3)


def _ask(solver: Any, surface: "Surface", ch: bc.Challenge, g: Grid) -> List[int]:
    image = surface.screenshot(g.area)
    if not image:
        raise SolverError("the challenge could not be captured")
    tiles: Tuple[bytes, ...] = ()
    if ch.kind == bc.HCAPTCHA:
        shots = [surface.screenshot(b) for b in g.tiles]
        if not all(shots):
            raise SolverError("a tile could not be captured")
        tiles = tuple(shots)  # type: ignore[arg-type]
    answer = solver.solve(SolveRequest(kind=ch.kind, image_png=image, instruction=instruction_line(g.instruction),
                                       grid=(g.rows, g.cols), tiles_png=tiles))
    return validate_answer(answer, (g.rows, g.cols))


def solve_grid(lad: "Ladder", surface: "Surface", ch: bc.Challenge, solver: Any, provider: str,
               steps: List[str], task_id: Optional[str]) -> "Outcome":
    from tools import browser_captcha_ladder as bl

    end = lad.clock() + TIER_C_S
    delivery, _ = surface.delivery()
    st = {"rounds": 0, "solves": 0}

    def out(outcome: str, reason: str, **kw: Any) -> "Outcome":
        return lad._out(ch, outcome, reason, tier="C", solver=provider, rounds=st["rounds"], solves=st["solves"],
                        delivery=delivery.name, steps=steps, **kw)

    def person(reason: str) -> "Outcome":
        if bl._add_failure(task_id, ch) >= bl.MAX_FAILURES:
            return out(bl.HARD_STOP, f"this {ch.kind} check failed {bl.MAX_FAILURES} times",
                       hard_stop=bc.HARD_STOP_REPEATED)
        return out(bl.NEEDS_PERSON, reason)

    grid = _settled(lad, surface, ch, end)
    while st["rounds"] < MAX_ROUNDS:
        if grid is None:
            steps.append("tier C: the image challenge could not be read")
            return out(bl.NEEDS_PERSON, f"the {ch.kind} image challenge could not be read from its frame")
        st["rounds"] += 1
        steps.append(f"round {st['rounds']}: {grid.rows}x{grid.cols}{' dynamic' if grid.dynamic else ''} grid, "
                     f"{instruction_line(grid.instruction)[:80]!r}")
        replaced: Optional[Set[int]] = None
        new_challenge = False
        for wave in range(MAX_WAVES):
            if lad.clock() >= end:
                steps.append("tier C deadline reached")
                return person(f"the {ch.kind} image challenge ran out of time")
            if not _take_solve(task_id):
                steps.append(f"solve budget used up ({MAX_SOLVES_PER_TASK} per task)")
                return out(bl.NEEDS_PERSON, f"the {MAX_SOLVES_PER_TASK} recognition requests this task may spend "
                                            "are used up")
            st["solves"] += 1
            try:
                picks = _ask(solver, surface, ch, grid)
            except SolverUnsupported as exc:
                steps.append(f"{provider} cannot answer this one: {exc}")
                if grid.reload is None:
                    return person(f"the recognition solver does not know this challenge ({exc})")
                delivery.click(grid.reload, lad.rng)
                steps.append("asked for a different challenge")
                new_challenge = True
                break
            except (SolverError, ValueError) as exc:
                steps.append(f"{provider} failed: {exc}")
                return person(f"the recognition solver could not answer ({exc})")
            if replaced is not None:
                picks = [i for i in picks if i in replaced]   # only the new images are in question
            picks = [i for i in picks if not grid.selected[i]]
            for n, i in enumerate(picks):
                if n:
                    lad.sleep(lad.rng.uniform(0.2, 0.6))
                delivery.click(grid.tiles[i], lad.rng)
            steps.append(f"clicked {len(picks)} tile{'s' if len(picks) != 1 else ''}"
                         + (f" (wave {wave + 1})" if grid.dynamic else ""))
            if not grid.dynamic or not picks:
                break
            before, replaced = grid, set(picks)
            grid = _settled(lad, surface, ch, end, before=before, replaced=replaced)
            if grid is None:
                steps.append("the clicked tiles were not replaced")
                return person(f"the {ch.kind} grid stopped responding")
        if new_challenge:
            grid = _settled(lad, surface, ch, end)
            continue
        now_grid = surface.grid(ch) or grid
        if now_grid.verify is None:
            steps.append("verify button not found")
            return person(f"the {ch.kind} verify button could not be found")
        lad.sleep(lad.rng.uniform(0.3, 0.8))
        delivery.click(now_grid.verify, lad.rng)
        steps.append("clicked verify")
        state, nxt, facts, grid = _after_verify(lad, surface, ch, now_grid, end)
        if state == "passed":
            bl._clear_failures(task_id, ch)
            return out(bl.PASSED, f"solved the image challenge in {st['rounds']} round"
                                  f"{'s' if st['rounds'] > 1 else ''}", url=facts.url)
        if state == "changed" and nxt is not None:
            if nxt.hard_stop:
                return lad.run(surface, nxt, task_id)
            steps.append(f"page now shows {nxt.kind} {nxt.stage}")
            return out(bl.NEEDS_PERSON, f"the page moved on to a different check ({nxt.kind} {nxt.stage})")
        if grid is not None and grid.error:
            steps.append(f"the challenge said: {grid.error[:80]}")
    return person(f"the {ch.kind} image challenge did not pass after {st['rounds']} round"
                  f"{'s' if st['rounds'] != 1 else ''}")


def _after_verify(lad: "Ladder", surface: "Surface", ch: bc.Challenge, was: Grid, end: float):
    """``(state, challenge, facts, grid)``: "passed", "changed" (another kind / stage / a hard stop), or
    "again" (a new or re-asked grid, ready for the next round), or "waiting"."""
    until = min(lad.clock() + VERIFY_WATCH_S, end)
    while True:
        now, facts = lad._classify(surface)
        if lad._passed(ch, now, facts):
            return "passed", now, facts, None
        if now is not None and (now.kind != ch.kind or now.hard_stop or now.stage != ch.stage):
            return "changed", now, facts, None
        g = surface.grid(ch)
        if g is not None and g.tiles and not g.busy and (g.sig != was.sig or g.error):
            return "again", now, facts, g
        if lad.clock() >= until:
            return "waiting", now, facts, (g if g is not None and g.tiles else None)
        lad.sleep(0.4)


# ---- reading the frame over CDP -----------------------------------------------------------------------------------

#: Runs inside the challenge frame. Frame-local rectangles; the caller adds the frame's own offset.
GRID_JS = r"""(() => {
  const R = e => { if (!e) return null; const b = e.getBoundingClientRect(), s = getComputedStyle(e);
    return (b.width > 2 && b.height > 2 && s.display !== 'none' && s.visibility !== 'hidden') ? [b.left, b.top, b.width, b.height] : null; };
  const T = e => e ? (e.innerText || e.textContent || '').replace(/\s+/g, ' ').trim() : '';
  const shown = e => !!R(e);
  const table = document.querySelector('table[class*="rc-imageselect-table"]');
  if (table) {
    const rows = Array.from(table.rows), tds = Array.from(table.querySelectorAll('td'));
    const desc = document.querySelector('.rc-imageselect-desc-no-canonical, .rc-imageselect-desc');
    const errs = Array.from(document.querySelectorAll('.rc-imageselect-error-select-more, .rc-imageselect-error-dynamic-more,'
      + ' .rc-imageselect-error-select-something, .rc-imageselect-incorrect-response')).filter(shown).map(T).join(' ');
    const instr = T(desc);
    return JSON.stringify({vendor: 'recaptcha', instruction: instr, target: T(desc && desc.querySelector('strong')),
      rows: rows.length, cols: rows.length ? rows[0].cells.length : 0, tiles: tds.map(R),
      selected: tds.map(td => td.classList.contains('rc-imageselect-tileselected')),
      srcs: tds.map(td => { const i = td.querySelector('img'); return i ? (i.getAttribute('src') || '') + '#' + i.className : ''; }),
      busy: tds.some(td => td.classList.contains('rc-imageselect-dynamic-selected'))
        || Array.from(table.querySelectorAll('img')).some(i => !i.complete),
      dynamic: /none left/i.test(instr), area: R(table),
      verify: R(document.getElementById('recaptcha-verify-button')),
      reload: R(document.getElementById('recaptcha-reload-button')), error: errs});
  }
  const cells = Array.from(document.querySelectorAll('.task-grid .task-image, .task-grid .task'));
  if (cells.length) {
    const grid = document.querySelector('.task-grid');
    const xs = new Set(cells.map(c => Math.round(c.getBoundingClientRect().left)));
    const cols = xs.size || 3;
    return JSON.stringify({vendor: 'hcaptcha', instruction: T(document.querySelector('.prompt-text')),
      target: T(document.querySelector('.prompt-text')), rows: Math.ceil(cells.length / cols), cols,
      tiles: cells.map(R), selected: cells.map(c => c.getAttribute('aria-pressed') === 'true' || /selected/.test(c.className)),
      srcs: cells.map(c => { const i = c.querySelector('.image'); return i ? getComputedStyle(i).backgroundImage : ''; }),
      busy: false, dynamic: false, area: R(grid),
      verify: R(document.querySelector('.button-submit')), reload: R(document.querySelector('.refresh.button, .refresh')),
      error: Array.from(document.querySelectorAll('.display-error')).filter(shown).map(T).join(' ')});
  }
  return JSON.stringify({none: true});
})()"""


def _challenge_frame(ch: bc.Challenge, facts: bc.PageFacts) -> Optional[bc.Frame]:
    for f in facts.frames:
        if not f.backend_node_id or not f.visible:
            continue
        if ch.kind == bc.RECAPTCHA_V2 and bc._RE_RECAPTCHA_BFRAME.search(f.url or ""):
            return f
        if ch.kind == bc.HCAPTCHA and bc._RE_HCAPTCHA.search(f.url or "") and "frame=challenge" in f.url:
            return f
    return None


def _content_origin(page: Any, backend_node_id: int) -> Optional[Tuple[float, float]]:
    try:
        quad = (page.call("DOM.getBoxModel", {"backendNodeId": backend_node_id}).get("model") or {}).get("content")
    except Exception:
        return None
    if not quad or len(quad) < 8:
        return None
    return min(quad[0::2]), min(quad[1::2])


def _box(r: Any, ox: float, oy: float) -> Optional[Box]:
    if not isinstance(r, list) or len(r) != 4:
        return None
    return Box(ox + float(r[0]), oy + float(r[1]), float(r[2]), float(r[3]))


def parse_grid(raw: dict, ox: float, oy: float) -> Optional[Grid]:
    """A :class:`Grid` from :data:`GRID_JS`'s output, offset by the frame's content origin, or None."""
    if not isinstance(raw, dict) or raw.get("none"):
        return None
    tiles = [_box(t, ox, oy) for t in raw.get("tiles") or []]
    rows, cols = int(raw.get("rows") or 0), int(raw.get("cols") or 0)
    area = _box(raw.get("area"), ox, oy)
    if not tiles or any(t is None for t in tiles) or rows * cols != len(tiles) or area is None:
        return None
    n = len(tiles)
    sel = [bool(x) for x in (raw.get("selected") or [])][:n]
    srcs = [str(x) for x in (raw.get("srcs") or [])][:n]
    return Grid(vendor=str(raw.get("vendor") or ""), instruction=str(raw.get("instruction") or ""),
                target=str(raw.get("target") or ""), rows=rows, cols=cols, tiles=tiles,  # type: ignore[arg-type]
                selected=sel + [False] * (n - len(sel)), srcs=srcs + [""] * (n - len(srcs)), area=area,
                verify=_box(raw.get("verify"), ox, oy), reload=_box(raw.get("reload"), ox, oy),
                dynamic=bool(raw.get("dynamic")), busy=bool(raw.get("busy")), error=str(raw.get("error") or ""))


def read_grid_cdp(surface: Any, ch: bc.Challenge) -> Optional[Grid]:
    """The visible challenge frame's grid, read in that frame's own (out-of-process) target."""
    if ch.kind not in GRID_KINDS:
        return None
    frame = _challenge_frame(ch, surface.facts())
    if frame is None:
        return None
    origin = _content_origin(surface.page, int(frame.backend_node_id))  # type: ignore[arg-type]
    if origin is None:
        return None
    try:
        infos = surface.conn.call("Target.getTargets").get("targetInfos") or []
    except Exception:
        return None
    iframes = [t for t in infos if t.get("type") == "iframe"]
    exact = [t for t in iframes if str(t.get("url") or "") == frame.url]
    key = frame.url.split("#")[0]
    cands = exact or [t for t in iframes if str(t.get("url") or "").split("#")[0] == key]
    for t in cands:
        sid = None
        try:
            sid = surface.conn.call("Target.attachToTarget", {"targetId": t["targetId"], "flatten": True})["sessionId"]
            r = surface.conn.call("Runtime.evaluate", {"expression": GRID_JS, "returnByValue": True}, session_id=sid)
            g = parse_grid(json.loads(((r.get("result") or {}).get("value")) or "{}"), *origin)
            if g is not None:
                return g
        except Exception:
            continue
        finally:
            if sid:
                try:
                    surface.conn.call("Target.detachFromTarget", {"sessionId": sid})
                except Exception:
                    pass
    return None


def screenshot_cdp(page: Any, box: Box) -> Optional[bytes]:
    """PNG of ``box`` (top-level viewport CSS px) from the page's own compositor -- OOPIFs included."""
    try:
        vv = page.call("Page.getLayoutMetrics").get("cssVisualViewport") or {}
        clip = {"x": box.x + float(vv.get("pageX") or 0), "y": box.y + float(vv.get("pageY") or 0),
                "width": box.width, "height": box.height, "scale": 1}
        data = page.call("Page.captureScreenshot", {"format": "png", "clip": clip, "fromSurface": True}).get("data")
        return base64.b64decode(data) if data else None
    except Exception:
        return None
