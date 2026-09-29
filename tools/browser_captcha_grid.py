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

The frame DOM readers below follow reCAPTCHA's bframe (``rc-imageselect-*``) as it is widely documented,
and hCaptcha's challenge frame AS MEASURED on 2026-09-29 (forums.macrumors.com's register page and
accounts.hcaptcha.com/demo, Chrome for Testing 154): there are no tile elements at all. The picture is ONE
``<canvas role="img" aria-label="Image-based CAPTCHA challenge...">`` (1000x940 backing, 500x470 CSS px)
laid over the whole challenge, under ``h2.prompt-text`` (the instruction, in the top 110 px) and above
``.button-submit`` (aria-label "Skip Challenge, page 1 of 2": its words turn to Next / Verify once
something is picked). What the canvas draws differs per challenge, so it is read from its own pixels
(:func:`slice_canvas`): nine separate pictures on the grey panel are a 3x3 grid (NopeCHA
``image_label_binary``, one image per tile); one picture filling the panel is a "click the ..." scene
(``image_label_area_select``: the answer is a spot). The older ``.task-grid .task-image`` DOM is still
read if it ever comes back. A frame this cannot read is handed to the person.
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
from tools.browser_captcha_solver import (SolveRequest, SolverError, SolverUnsupported, validate_answer,
                                          validate_points)

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
    #: hCaptcha's canvas challenge: ``tiles`` were cut from the picture's own pixels (:func:`slice_canvas`),
    #: ``selected`` is unknowable (the tick is drawn, not a class), and ``mode`` says what is asked.
    canvas: bool = False
    mode: str = "grid"               # "grid" | "area" (one picture: click the spot the instruction names)
    body: Optional[Box] = None       # canvas: the picture panel under the prompt
    page: int = 0                    # canvas: "page 1 of 2" from the submit button
    pages: int = 0
    #: hCaptcha: a digest of the picture as shown. Its frame fades in (opacity 0 -> 1, 0.15 s) and its images
    #: load after the DOM: 2026-09-29 the first live read screenshotted a half-faded grid and the solver was
    #: sent nine near-white squares. A grid is only read as still when two reads in a row show the same pixels.
    digest: str = ""
    #: canvas: the picture changes between reads with the question unchanged -- an animation ("Click the animal
    #: the ball never touches", measured 2026-09-29 on forums.macrumors.com and the demo: a new picture every
    #: read). One screenshot cannot show where a ball goes, so a recognition solver is never asked about one.
    moving: bool = False

    @property
    def sig(self) -> str:
        return self.instruction + "|" + "|".join(self.srcs) + "|" + self.digest

    @property
    def needs_still(self) -> bool:
        return self.canvas or self.vendor == "hcaptcha"


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

#: hCaptcha's drag-and-drop puzzles, by their own words: "Move the correct vial into the empty slot",
#: "Drag the object on the right to ...". A click never answers them.
_DRAG = re.compile(r"\b(?:drag|move|slide)\b|\bput (?:the|each)\b|\bplace (?:the|each)\b", re.I)
MOVING_READS = 3             # canvas reads in a row that each showed a new picture: an animation
MAX_REFRESHES = 3            # "a different challenge, please" when one is not clickable, per tier C run

_TAIL = re.compile(r"\s*(?:click verify once there are none left|if there are none,? click skip|"
                   r"please (?:also )?check (?:the )?new images|please select all matching images|"
                   r"please try again).*$", re.I | re.S)


def instruction_line(text: str) -> str:
    """The challenge's sentence without reCAPTCHA's trailing how-to ("Click verify once ...")."""
    return _TAIL.sub("", re.sub(r"\s+", " ", text or "")).strip()


def unanswerable(g: Grid) -> str:
    """Why a click on a screenshot cannot answer this challenge ("a drag-and-drop puzzle", "a moving
    picture"), or ''."""
    if not g.canvas:
        return ""
    if _DRAG.search(instruction_line(g.instruction)):
        return "a drag-and-drop puzzle"
    if g.moving:
        return "a moving picture"
    return ""


def _settled(lad: "Ladder", surface: "Surface", ch: bc.Challenge, end: float, *, before: Optional[Grid] = None,
             replaced: Set[int] = frozenset()) -> Optional[Grid]:
    """The grid once it is readable and still -- and, with ``replaced``, once those tiles have new images."""
    until = min(lad.clock() + SETTLE_S, end)
    last = None
    changes = 0
    while True:
        g = surface.grid(ch)
        # hCaptcha is only still when two reads in a row showed the same picture (it fades a new page in).
        still = g is not None and (not g.needs_still or (last is not None and last == g.sig))
        if g is not None and g.canvas and g.tiles and not g.busy and last is not None and not still \
                and last.split("|")[0] == g.instruction:
            changes += 1
            if changes >= MOVING_READS:
                g.moving = True
                return g
        last = g.sig if g is not None else None
        if g is not None and g.tiles and not g.busy and still:
            fresh = all(i < len(g.srcs) and before is not None and i < len(before.srcs)
                        and g.srcs[i] != before.srcs[i] for i in replaced)
            if fresh or not replaced:
                return g
        if lad.clock() >= until:
            return g if (g is not None and g.tiles and not replaced) else None
        lad.sleep(0.3)


def _ask_area(solver: Any, surface: "Surface", ch: bc.Challenge, g: Grid) -> List[Tuple[float, float]]:
    """Mode "area": the spot(s) to click, as viewport points inside the picture."""
    pic = g.tiles[0]
    image = surface.screenshot(pic)
    if not image:
        raise SolverError("the challenge could not be captured")
    answer = solver.solve(SolveRequest(kind=ch.kind, image_png=image, instruction=instruction_line(g.instruction),
                                       grid=(1, 1), mode="area"))
    return [(pic.x + fx * pic.width, pic.y + fy * pic.height) for fx, fy in validate_points(answer)]


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
    st = {"rounds": 0, "solves": 0, "refreshes": 0}

    def out(outcome: str, reason: str, **kw: Any) -> "Outcome":
        return lad._out(ch, outcome, reason, tier="C", solver=provider, rounds=st["rounds"], solves=st["solves"],
                        delivery=delivery.name, steps=steps, **kw)

    def person(reason: str) -> "Outcome":
        if bl._add_failure(task_id, ch) >= bl.MAX_FAILURES:
            return out(bl.HARD_STOP, f"this {ch.kind} check failed {bl.MAX_FAILURES} times",
                       hard_stop=bc.HARD_STOP_REPEATED)
        return out(bl.NEEDS_PERSON, reason)

    grid = _settled(lad, surface, ch, end)
    last_page = 0
    while True:
        if grid is None and st["rounds"] >= MAX_ROUNDS:
            break
        if grid is None:
            steps.append("tier C: the image challenge could not be read")
            return out(bl.NEEDS_PERSON, f"the {ch.kind} image challenge could not be read from its frame")
        what = unanswerable(grid)
        if what:
            # No screenshot-and-click answers it. Ask for a different challenge -- free, no solve spent -- a few
            # times at most, then the person.
            if grid.reload is None or st["refreshes"] >= MAX_REFRESHES:
                steps.append(f"{what}, which Moe does not solve")
                return person(f"the {ch.kind} challenge is {what}, which Moe does not solve")
            st["refreshes"] += 1
            delivery.click(grid.reload, lad.rng)
            steps.append(f"{what} ({instruction_line(grid.instruction)[:60]!r}): asked for a different challenge")
            before = grid
            grid = _settled(lad, surface, ch, end)
            if grid is not None and grid.sig == before.sig:
                grid = None
            last_page = 0
            continue
        # hCaptcha asks in pages ("page 1 of 2"): the next page of the same challenge is the same round.
        next_page = grid.canvas and grid.page > 1 and grid.page == last_page + 1
        if not next_page:
            if st["rounds"] >= MAX_ROUNDS:
                break
            st["rounds"] += 1
        last_page = grid.page
        shape = ("one picture, click the spot" if grid.mode == "area" else
                 f"{grid.rows}x{grid.cols}{' dynamic' if grid.dynamic else ''} grid")
        steps.append(f"round {st['rounds']}"
                     + (f" page {grid.page}/{grid.pages}" if grid.canvas and grid.pages else "")
                     + f": {shape}, {instruction_line(grid.instruction)[:80]!r}")
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
                if grid.mode == "area":
                    spots = _ask_area(solver, surface, ch, grid)
                    for n, (x, y) in enumerate(spots):
                        if n:
                            lad.sleep(lad.rng.uniform(0.2, 0.6))
                        delivery.click(Box(x - 2, y - 2, 4, 4), lad.rng)
                    steps.append(f"clicked {len(spots)} spot{'s' if len(spots) != 1 else ''} in the picture")
                    break
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
        lad.sleep(lad.rng.uniform(0.3, 0.8))  # the ticks drawn first: the grid read next is what verify sees
        now_grid = surface.grid(ch) or grid
        if now_grid.verify is None:
            steps.append("verify button not found")
            return person(f"the {ch.kind} verify button could not be found")
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
        if state == "waiting" and grid is not None and grid.canvas:
            grid = _settled(lad, surface, ch, end) or grid  # never still: find out whether it is moving
    return person(f"the {ch.kind} image challenge did not pass after {st['rounds']} round"
                  f"{'s' if st['rounds'] != 1 else ''}")


def _after_verify(lad: "Ladder", surface: "Surface", ch: bc.Challenge, was: Grid, end: float):
    """``(state, challenge, facts, grid)``: "passed", "changed" (another kind / stage / a hard stop), or
    "again" (a new or re-asked grid, ready for the next round), or "waiting"."""
    until = min(lad.clock() + VERIFY_WATCH_S, end)
    last = None
    while True:
        now, facts = lad._classify(surface)
        if lad._passed(ch, now, facts):
            return "passed", now, facts, None
        if now is not None and (now.kind != ch.kind or now.hard_stop or now.stage != ch.stage):
            return "changed", now, facts, None
        g = surface.grid(ch)
        still = g is not None and (not g.needs_still or last == g.sig)
        last = g.sig if g is not None else None
        if g is not None and g.tiles and not g.busy and still and (g.sig != was.sig or g.error):
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
  const cv = Array.from(document.querySelectorAll('canvas')).find(c => shown(c)
    && (/captcha/i.test(c.getAttribute('aria-label') || '') || c.closest('.challenge-view, .challenge')));
  if (cv) {
    // hCaptcha, measured 2026-09-29: one canvas draws the whole challenge; the prompt and the submit button
    // are DOM. The error line is always in the tree -- shown only when not opacity 0 / aria-hidden.
    const btn = document.querySelector('.button-submit');
    const label = btn ? (btn.getAttribute('aria-label') || '') : '';
    // "page 1 of 2" is in the button's label on page 1 only (on the last page it just says Verify, measured);
    // the breadcrumbs always say it: one .Crumb per page, the current one drawn wide (24 px vs 8 px).
    const crumbs = Array.from(document.querySelectorAll('.challenge-breadcrumbs .Crumb')).filter(shown);
    const widths = crumbs.map(c => c.getBoundingClientRect().width);
    const m = /page\s+(\d+)\s+of\s+(\d+)/i.exec(label)
      || (crumbs.length > 1 ? [0, widths.indexOf(Math.max(...widths)) + 1, crumbs.length] : null);
    const err = document.querySelector('.display-error');
    const errShown = err && shown(err) && getComputedStyle(err).opacity !== '0' && err.getAttribute('aria-hidden') !== 'true';
    return JSON.stringify({vendor: 'hcaptcha', canvas: true, instruction: T(document.querySelector('.prompt-text')),
      target: T(document.querySelector('.prompt-text')), area: R(cv),
      header: R(document.querySelector('.challenge-header')), verify: R(btn), verify_label: T(btn) || label,
      page: m ? +m[1] : 0, pages: m ? +m[2] : 0,
      reload: R(document.querySelector('.refresh.button, .refresh')), error: errShown ? T(err) : ''});
  }
  // Measured 2026-09-29 (a DOM grid on accounts.hcaptcha.com/demo): .task-grid[role=group] > .task[role=button]
  // [aria-pressed][aria-label="Challenge Image N"] > .task-image > .wrapper > .image (the picture, as a CSS
  // background). One cell per .task -- .task-image sits inside it, so both selectors would count it twice.
  let cells = Array.from(document.querySelectorAll('.task-grid .task')).filter(shown);
  if (!cells.length) cells = Array.from(document.querySelectorAll('.task-grid .task-image')).filter(shown);
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
      error: Array.from(document.querySelectorAll('.display-error')).filter(e => shown(e)
        && getComputedStyle(e).opacity !== '0' && e.getAttribute('aria-hidden') !== 'true').map(T).join(' ')});
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


def parse_canvas(raw: dict, ox: float, oy: float) -> Optional[Grid]:
    """hCaptcha's canvas challenge from :data:`GRID_JS`, offset by the frame's content origin: everything but
    the tiles, which :func:`canvas_grid` cuts from the picture. ``body`` is the canvas under the prompt."""
    area = _box(raw.get("area"), ox, oy)
    if area is None:
        return None
    header = _box(raw.get("header"), ox, oy)
    top = header.y + header.height if header is not None and area.y <= header.y + header.height < area.y + area.height \
        else area.y
    body = Box(area.x, top, area.width, area.y + area.height - top)
    try:
        page, pages = int(raw.get("page") or 0), int(raw.get("pages") or 0)
    except (TypeError, ValueError):
        page = pages = 0
    return Grid(vendor="hcaptcha", instruction=str(raw.get("instruction") or ""), target=str(raw.get("target") or ""),
                rows=0, cols=0, tiles=[], selected=[], srcs=[], area=area, verify=_box(raw.get("verify"), ox, oy),
                reload=_box(raw.get("reload"), ox, oy), error=str(raw.get("error") or ""), canvas=True, body=body,
                page=page, pages=pages)


#: What counts as the grey panel behind the pictures (measured: rgb(235,235,235) and white): a light,
#: colourless pixel. Everything else is picture.
_PANEL_MIN, _PANEL_SPREAD = 200, 10
#: A blob smaller than this share of the panel is a speck or a tick mark, not a picture.
_MIN_PICTURE = 0.012
#: ...and fills at least this share of its own box (a picture ~0.95 measured; an outline a few percent).
_MIN_FILL = 0.35


def _blobs(png: bytes, width: float, height: float, step: int = 2) -> Optional[List[Tuple[float, float, float, float]]]:
    """Bounding boxes (CSS px, relative to the image) of the separate pictures on the panel."""
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(png)).convert("RGB")
    except Exception:
        return None
    w, h = int(round(width)), int(round(height))
    if w < 20 or h < 20:
        return None
    if im.size != (w, h):  # a Retina screenshot is device pixels: back to CSS px
        im = im.resize((w, h))
    px = im.load()
    gw, gh = w // step, h // step
    n = gw * gh
    mask = bytearray(n)
    for gy in range(gh):
        for gx in range(gw):
            r, g, b = px[gx * step, gy * step]
            lo = min(r, g, b)
            if not (lo >= _PANEL_MIN and max(r, g, b) - lo <= _PANEL_SPREAD):
                mask[gy * gw + gx] = 1
    seen = bytearray(n)
    out = []
    for start in range(n):
        if not mask[start] or seen[start]:
            continue
        stack, seen[start] = [start], 1
        x0 = x1 = start % gw
        y0 = y1 = start // gw
        cells = 0
        while stack:
            i = stack.pop()
            cells += 1
            x, y = i % gw, i // gw
            x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
            for j in ((i - 1) if x else -1, (i + 1) if x + 1 < gw else -1, i - gw, i + gw):
                if 0 <= j < n and mask[j] and not seen[j]:
                    seen[j] = 1
                    stack.append(j)
        bw, bh = (x1 - x0 + 1) * step, (y1 - y0 + 1) * step
        # A picture fills its box; a ring does not -- the canvas's teal focus outline (measured after a click
        # on "refresh") runs round the whole panel and would otherwise swallow every picture inside it.
        if bw * bh >= _MIN_PICTURE * w * h and cells >= _MIN_FILL * (x1 - x0 + 1) * (y1 - y0 + 1):
            out.append((float(x0 * step), float(y0 * step), float(bw), float(bh)))
    return out


def _rows_of(blobs: List[Tuple[float, float, float, float]]) -> Optional[List[List[Tuple[float, float, float, float]]]]:
    """The blobs as rows (top to bottom, each left to right) when they form an even grid, else None."""
    ordered = sorted(blobs, key=lambda b: b[1] + b[3] / 2)
    tall = sorted(b[3] for b in blobs)[len(blobs) // 2]
    rows: List[List[Tuple[float, float, float, float]]] = []
    for b in ordered:
        if rows and abs((b[1] + b[3] / 2) - (rows[-1][0][1] + rows[-1][0][3] / 2)) < tall / 2:
            rows[-1].append(b)
        else:
            rows.append([b])
    if len({len(r) for r in rows}) != 1:
        return None
    return [sorted(r, key=lambda b: b[0]) for r in rows]


def slice_canvas(png: bytes, width: float, height: float) -> Optional[Tuple[str, int, int, List[Tuple[float, float, float, float]]]]:
    """What hCaptcha's canvas shows, from its pixels: ``("grid", rows, cols, tiles)`` for separate pictures
    in an even grid (row-major), ``("area", 1, 1, [picture])`` for one picture filling the panel, else None.
    Boxes are CSS px relative to the image. Measured on 2026-09-29's two challenges: nine pictures of
    ~110x104 px on the 500x360 panel (the tiles are drawn slightly warped, so each is its own blob), and
    one 480x330 scene."""
    blobs = _blobs(png, width, height)
    if not blobs:
        return None
    if len(blobs) == 1:
        b = blobs[0]
        return ("area", 1, 1, [b]) if b[2] * b[3] >= 0.4 * width * height else None
    rows = _rows_of(blobs)
    if rows is None or len(rows) < 2 or len(rows[0]) < 2 or len(rows) * len(rows[0]) > 16:
        return None
    return "grid", len(rows), len(rows[0]), [b for r in rows for b in r]


def canvas_grid(g: Grid, png: Optional[bytes]) -> Optional[Grid]:
    """``g`` (from :func:`parse_canvas`) with its tiles cut from ``png``, a screenshot of ``g.body``."""
    if g.body is None or not png:
        return None
    cut = slice_canvas(png, g.body.width, g.body.height)
    if cut is None:
        return None
    mode, rows, cols, boxes = cut
    import hashlib
    tiles = [Box(g.body.x + x, g.body.y + y, w, h) for x, y, w, h in boxes]
    digest = hashlib.sha1(png).hexdigest()[:16]  # the picture itself: a new page / a tick mark changes it
    return Grid(vendor=g.vendor, instruction=g.instruction, target=g.target, rows=rows, cols=cols, tiles=tiles,
                selected=[False] * len(tiles), srcs=[f"{digest}#{i}" for i in range(len(tiles))], area=g.body,
                verify=g.verify, reload=g.reload, error=g.error, canvas=True, mode=mode, body=g.body,
                page=g.page, pages=g.pages, busy=g.busy, digest=digest)


def parse_grid(raw: dict, ox: float, oy: float) -> Optional[Grid]:
    """A :class:`Grid` from :data:`GRID_JS`'s output, offset by the frame's content origin, or None. For
    hCaptcha's canvas this has no tiles yet (:func:`canvas_grid` cuts them)."""
    if not isinstance(raw, dict) or raw.get("none"):
        return None
    if raw.get("canvas"):
        return parse_canvas(raw, ox, oy)
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
    viewport = _in_view(surface.page, int(frame.backend_node_id))  # type: ignore[arg-type]
    origin = _content_origin(surface.page, int(frame.backend_node_id))  # type: ignore[arg-type]
    if origin is None:
        return None
    g = _read_frame(surface, frame, origin)
    return _clip(g, viewport) if g is not None else None


def _in_view(page: Any, backend_node_id: int) -> Optional[Tuple[float, float]]:
    """Scroll so the challenge frame is inside the viewport -- its bottom (the buttons) first when it is taller
    than the viewport -- and return the viewport's CSS size. Measured on the local canvas fake (a 570 px frame in
    a 557 px viewport): the verify button hung 13 px below the fold and every click that picked its lower part
    went nowhere. A box can only be clicked where the viewport shows it."""
    try:
        vv = page.call("Page.getLayoutMetrics").get("cssVisualViewport") or {}
        vw, vh = float(vv.get("clientWidth") or 0), float(vv.get("clientHeight") or 0)
        quad = (page.call("DOM.getBoxModel", {"backendNodeId": backend_node_id}).get("model") or {}).get("border")
    except Exception:
        return None
    if not vw or not vh or not quad or len(quad) < 8:
        return None
    top, bottom = min(quad[1::2]), max(quad[1::2])
    dy = 0.0
    if bottom > vh:
        dy = bottom - vh + 2
    if top - dy < 0 and bottom - top <= vh:
        dy = top - 2
    if dy:
        try:
            page.call("Runtime.evaluate", {"expression": f"window.scrollBy(0, {dy:.0f})"})
        except Exception:
            pass
    return vw, vh


def _clip(g: Grid, viewport: Optional[Tuple[float, float]]) -> Optional[Grid]:
    """Clickable boxes cut to what the viewport shows; a button that is not on screen at all is no button."""
    if viewport is None:
        return g
    vw, vh = viewport

    def cut(b: Optional[Box]) -> Optional[Box]:
        if b is None:
            return None
        x0, y0 = max(b.x, 0.0), max(b.y, 0.0)
        x1, y1 = min(b.x + b.width, vw), min(b.y + b.height, vh)
        return Box(x0, y0, x1 - x0, y1 - y0) if x1 - x0 >= 6 and y1 - y0 >= 6 else None

    g.verify, g.reload = cut(g.verify), cut(g.reload)
    tiles = [cut(t) for t in g.tiles]
    if any(t is None for t in tiles):
        return None  # a tile off screen cannot be looked at or clicked: read again once it is in view
    g.tiles = tiles  # type: ignore[assignment]
    return g


def _read_frame(surface: Any, frame: bc.Frame, origin: Tuple[float, float]) -> Optional[Grid]:
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
                return _finish(surface, g, _frame_opacity(surface.page, int(frame.backend_node_id)))
        except Exception:
            continue
        finally:
            if sid:
                try:
                    surface.conn.call("Target.detachFromTarget", {"sessionId": sid})
                except Exception:
                    pass
    if not cands:
        # The same site as the page (accounts.hcaptcha.com/demo): the frame runs in the page's own process and
        # is no target of its own. Read it in an isolated world of that frame.
        g = _read_in_process(surface, frame.url, origin)
        if g is not None:
            return _finish(surface, g, _frame_opacity(surface.page, int(frame.backend_node_id)))
    return None


def _finish(surface: Any, g: Grid, frame_opacity: float = 1.0) -> Optional[Grid]:
    if frame_opacity < 0.99:
        g.busy = True  # still fading in (or out): what a screenshot shows now is not the challenge
    if not g.canvas:
        if g.vendor == "hcaptcha":
            import hashlib
            shot = surface.screenshot(g.area)
            g.digest = hashlib.sha1(shot).hexdigest()[:16] if shot else ""
        return g
    return canvas_grid(g, surface.screenshot(g.body)) if g.body is not None else None


#: The challenge iframe's effective opacity on the page (its own and every ancestor's): hCaptcha fades its
#: frame's container in and out.
_OPACITY_FN = ("function() { let o = 1, e = this; while (e && e.nodeType === 1) { "
               "o *= parseFloat(getComputedStyle(e).opacity || '1'); e = e.parentElement; } return o; }")


def _frame_opacity(page: Any, backend_node_id: int) -> float:
    try:
        obj = (page.call("DOM.resolveNode", {"backendNodeId": backend_node_id}).get("object") or {}).get("objectId")
        if not obj:
            return 1.0
        r = page.call("Runtime.callFunctionOn", {"objectId": obj, "functionDeclaration": _OPACITY_FN,
                                                 "returnByValue": True})
        return float(((r.get("result") or {}).get("value")))
    except Exception:
        return 1.0


def _read_in_process(surface: Any, url: str, origin: Tuple[float, float]) -> Optional[Grid]:
    try:
        tree = surface.page.call("Page.getFrameTree").get("frameTree") or {}
    except Exception:
        return None
    todo, fid = [tree], None
    while todo and fid is None:
        t = todo.pop()
        f = t.get("frame") or {}
        if str(f.get("url") or "") + str(f.get("urlFragment") or "") == url or f.get("url") == url:
            fid = f.get("id")
        todo.extend(t.get("childFrames") or [])
    if not fid:
        return None
    try:
        ctx = surface.page.call("Page.createIsolatedWorld", {"frameId": fid, "worldName": "hermes-captcha-grid"})
        r = surface.page.call("Runtime.evaluate", {"expression": GRID_JS, "returnByValue": True,
                                                   "contextId": ctx["executionContextId"]})
        return parse_grid(json.loads(((r.get("result") or {}).get("value")) or "{}"), *origin)
    except Exception:
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
