"""Moe's engine-level CAPTCHA ladder: detect -> pass without solving -> solve -> person.

When ``browser_exec``'s own lane sees a challenge (tools/browser_use_cli.py calls :func:`run_for_exec`),
the engine -- not the chat model -- tries the rungs it may, on the owner's own browser, and returns the
outcome as a fact the model reads (``result["captcha"]``)::

    {"kind", "host", "outcome": "passed" | "needs_person" | "hard_stop" | "report_only",
     "reason", "attempts", "tier", "delivery", "steps", "next"}

Why the engine: frontier models are trained to refuse CAPTCHAs and Moe is provider-agnostic, so the chat
model is never asked to solve one -- it is told what happened and what to do next (carry on, or hand the
page to the person with ``browser_handoff``). Owner decision 2026-09-25: Moe acts for the owner, in the
owner's accounts, from the owner's Mac and IP.

Rungs in this slice (tier A -- nothing is solved, nothing leaves the Mac):

* wait out a Cloudflare managed challenge ("Just a moment...") -- it usually clears itself (<= 15 s);
* click a checkbox: Turnstile, reCAPTCHA v2's anchor, hCaptcha's checkbox -- on a person-shaped pointer
  path (tools/browser_captcha_input.py), re-locating the target before every try, at most 3 tries;
* PerimeterX "Press & Hold": press and keep holding while the page is polled -- released when the page
  says it is done or gave up, never after a fixed time.

Tier C (a recognition solver for image grids) is a seam only (tools/browser_captcha_solver.py); with
none configured an image challenge goes to the person.

Hard stops -- always the person, never automated, and never retried by the engine in this task:
a ticketing site (US BOTS Act), DataDome ``t=bv`` (the IP is banned), and the same challenge having
failed 3 times. Never used: audio / accessibility routes. Never sent to a third party: Cloudflare work.

The ladder only ever drives a browser on THIS Mac (a loopback CDP endpoint): a cloud browser is a third
party's machine and IP, so a challenge there is reported, not touched.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from tools import browser_captcha as bc
from tools.browser_captcha_input import Box, Delivery, pick_delivery
from tools.browser_captcha_solver import solver_for

logger = logging.getLogger(__name__)

PASSED = "passed"
NEEDS_PERSON = "needs_person"
HARD_STOP = "hard_stop"
REPORT_ONLY = "report_only"

MAX_FAILURES = 3            # the same challenge failing this many times is a hard stop
TOTAL_DEADLINE_S = 45.0     # the whole ladder, per browser_exec call


@dataclass(frozen=True)
class Budget:
    wait_s: float = 0.0        # pass without solving: how long to let it clear on its own first
    attempts: int = 0          # inputs (clicks / holds), each on a freshly located target
    poll_s: float = 6.0        # after each input: how long to watch for the pass
    hold_max_s: float = 0.0    # press-and-hold ceiling (the page decides when; this only bounds it)


BUDGETS: Dict[str, Budget] = {
    bc.CF_MANAGED: Budget(wait_s=15.0, attempts=1, poll_s=10.0),
    bc.TURNSTILE: Budget(wait_s=4.0, attempts=3, poll_s=8.0),
    bc.RECAPTCHA_V2: Budget(attempts=3, poll_s=6.0),
    bc.HCAPTCHA: Budget(attempts=3, poll_s=6.0),
    bc.PERIMETERX: Budget(attempts=3, poll_s=4.0, hold_max_s=15.0),
}

#: Where the clickable box sits inside each widget's iframe (CSS px from the frame's top-left: x, y, w, h).
#: Widget layouts measured from the vendors' own rendered widgets: Turnstile 300x65 with the box at left
#: 16-44 px, vertically centred; reCAPTCHA's anchor 304x78 with the box at (12..40, 25..53); hCaptcha's
#: checkbox frame 303x78 with the box at (14..42, vertically centred).
CHECKBOX_IN_FRAME = {
    bc.TURNSTILE: lambda fw, fh: (16.0, fh / 2 - 12, 26.0, 24.0),
    bc.CF_MANAGED: lambda fw, fh: (16.0, fh / 2 - 12, 26.0, 24.0),
    bc.RECAPTCHA_V2: lambda fw, fh: (13.0, 26.0, 26.0, 26.0),
    bc.HCAPTCHA: lambda fw, fh: (15.0, fh / 2 - 13, 26.0, 26.0),
}


# ---- failure memory (per task) -----------------------------------------------------------------------

_lock = threading.Lock()
_failures: Dict[Tuple[str, str, str], int] = {}


def _key(task_id: Optional[str], ch: bc.Challenge) -> Tuple[str, str, str]:
    return (str(task_id or "default"), ch.host, ch.kind)


def failures(task_id: Optional[str], ch: bc.Challenge) -> int:
    with _lock:
        return _failures.get(_key(task_id, ch), 0)


def _add_failure(task_id: Optional[str], ch: bc.Challenge) -> int:
    with _lock:
        k = _key(task_id, ch)
        _failures[k] = _failures.get(k, 0) + 1
        return _failures[k]


def _clear_failures(task_id: Optional[str], ch: bc.Challenge) -> None:
    with _lock:
        _failures.pop(_key(task_id, ch), None)


def reset_failures() -> None:
    with _lock:
        _failures.clear()


# ---- the outcome ---------------------------------------------------------------------------------------

@dataclass
class Outcome:
    kind: str
    host: str
    outcome: str
    reason: str = ""
    attempts: int = 0
    tier: str = ""
    delivery: str = ""
    hard_stop: str = ""
    steps: List[str] = field(default_factory=list)
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"kind": self.kind, "host": self.host, "outcome": self.outcome, "reason": self.reason,
             "attempts": self.attempts, "tier": self.tier, "delivery": self.delivery,
             "hard_stop": self.hard_stop, "steps": self.steps, "url": self.url}
        return {k: v for k, v in d.items() if v not in ("", [], None) or k in ("attempts",)}


# ---- the surface the ladder acts on ----------------------------------------------------------------------

class Surface:
    """What the ladder needs from a page. :class:`CdpSurface` is the real one; tests script a fake."""

    def facts(self) -> bc.PageFacts:
        raise NotImplementedError

    def locate(self, ch: bc.Challenge) -> Optional[Box]:
        raise NotImplementedError

    def delivery(self) -> Tuple[Delivery, str]:
        raise NotImplementedError


class Ladder:
    """One run of the state machine against one surface. ``sleep``/``clock``/``rng`` are injectable."""

    def __init__(self, browser_cfg: Optional[dict] = None, *, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, rng: Optional[random.Random] = None,
                 budgets: Optional[Dict[str, Budget]] = None, deadline_s: float = TOTAL_DEADLINE_S,
                 poll_every_s: float = 0.5):
        self.cfg = browser_cfg or {}
        self.sleep, self.clock = sleep, clock
        self.rng = rng or random.Random()
        self.budgets = budgets or BUDGETS
        self.deadline_s = deadline_s
        self.poll_every_s = poll_every_s
        self.hard_stop_hosts = bc.hard_stop_hosts(self.cfg)

    # -- helpers --

    def _classify(self, surface: Surface) -> Tuple[Optional[bc.Challenge], bc.PageFacts]:
        facts = surface.facts()
        return bc.classify(facts, hard_stop_hosts=self.hard_stop_hosts), facts

    @staticmethod
    def _passed(first: bc.Challenge, now: Optional[bc.Challenge], facts: bc.PageFacts) -> bool:
        if first.kind == bc.CF_MANAGED:
            # Solved = cf_clearance issued AND the page is real content, not the interstitial.
            return "cf_clearance" in facts.cookies and (now is None or now.kind != bc.CF_MANAGED)
        if now is None:
            return True  # the challenge is gone from the page
        return now.kind == first.kind and now.solved

    def _watch(self, surface: Surface, first: bc.Challenge, seconds: float, end: float
               ) -> Tuple[str, Optional[bc.Challenge], bc.PageFacts]:
        """Poll until passed / changed / out of time: ``("passed" | "changed" | "waiting", now, facts)``."""
        until = min(self.clock() + seconds, end)
        now, facts = self._classify(surface)
        while True:
            if self._passed(first, now, facts):
                return "passed", now, facts
            if now is not None and (now.kind != first.kind or now.stage != first.stage or now.hard_stop):
                return "changed", now, facts
            if self.clock() >= until:
                return "waiting", now, facts
            self.sleep(self.poll_every_s)
            now, facts = self._classify(surface)

    def _out(self, ch: bc.Challenge, outcome: str, reason: str, **kw: Any) -> Outcome:
        return Outcome(kind=ch.kind, host=ch.host, outcome=outcome, reason=reason, **kw)

    # -- the machine --

    def run(self, surface: Surface, ch: bc.Challenge, task_id: Optional[str] = None) -> Outcome:
        steps: List[str] = []
        end = self.clock() + self.deadline_s

        # 1. Hard stops: never automated.
        if ch.hard_stop == bc.HARD_STOP_TICKETING:
            return self._out(ch, HARD_STOP, f"{ch.host} sells tickets: Moe never works around its checks (US BOTS Act)",
                             hard_stop=ch.hard_stop, steps=["hard stop: ticketing site"])
        if ch.hard_stop == bc.HARD_STOP_IP_BANNED:
            return self._out(ch, HARD_STOP, "DataDome has blocked this network address (t=bv): nothing on the page can pass it",
                             hard_stop=ch.hard_stop, steps=["hard stop: DataDome t=bv"])
        if failures(task_id, ch) >= MAX_FAILURES:
            return self._out(ch, HARD_STOP, f"this {ch.kind} check has already failed {MAX_FAILURES} times",
                             hard_stop=bc.HARD_STOP_REPEATED, steps=["hard stop: failed 3 times already"])

        # 2. Nothing to do.
        if ch.kind == bc.RECAPTCHA_V3:
            return self._out(ch, REPORT_ONLY, "invisible reCAPTCHA (score-based): there is nothing to click",
                             steps=["report only"])
        if ch.solved:
            return self._out(ch, PASSED, "already passed", steps=["already passed"])

        budget = self.budgets.get(ch.kind)

        # 3. Pass without solving: let it clear on its own.
        if budget and budget.wait_s:
            state, now, facts = self._watch(surface, ch, budget.wait_s, end)
            steps.append(f"waited up to {budget.wait_s:g}s: {state}")
            if state == "passed":
                _clear_failures(task_id, ch)
                return self._out(ch, PASSED, "the check cleared on its own", tier="A", steps=steps, url=facts.url)
            if state == "changed" and now is not None and now.hard_stop:
                return self.run(surface, now, task_id)

        # 4. Image / puzzle stages: tier C, or the person.
        if budget is None or budget.attempts == 0 or ch.stage in ("image", "slider"):
            return self._tier_c(ch, steps)

        # 5. Tier A inputs.
        delivery, why = surface.delivery()
        attempts = 0
        current = ch
        for _ in range(budget.attempts):
            if self.clock() >= end:
                steps.append("ladder deadline reached")
                break
            box = surface.locate(current)  # re-located before every try: widgets move and re-render
            if box is None:
                steps.append("target not found")
                if attempts == 0:
                    return self._out(ch, NEEDS_PERSON, f"the {ch.kind} control could not be found on the page",
                                     tier="A", delivery=delivery.name, steps=steps)
                break
            attempts += 1
            try:
                if ch.kind == bc.PERIMETERX:
                    verdict, held = delivery.hold(box, self.rng, lambda: self._hold_verdict(surface, ch),
                                                 max(0.5, min(budget.hold_max_s, end - self.clock())))
                    steps.append(f"held {held:.1f}s: {'done' if verdict else 'failed' if verdict is False else 'no verdict'}")
                else:
                    delivery.click(box, self.rng)
                    steps.append(f"clicked ({delivery.name})")
            except NotImplementedError as exc:
                return self._out(ch, NEEDS_PERSON, str(exc), tier="A", delivery=delivery.name, attempts=attempts, steps=steps)
            state, now, facts = self._watch(surface, ch, budget.poll_s, end)
            if state == "passed":
                _clear_failures(task_id, ch)
                return self._out(ch, PASSED, f"passed after {attempts} {'hold' if ch.kind == bc.PERIMETERX else 'click'}"
                                 f"{'s' if attempts > 1 else ''}", tier="A", delivery=delivery.name, attempts=attempts,
                                 steps=steps, url=facts.url)
            if state == "changed" and now is not None:
                if now.hard_stop:
                    return self.run(surface, now, task_id)
                if now.stage in ("image", "slider"):
                    steps.append(f"escalated to {now.kind} {now.stage}")
                    out = self._tier_c(now, steps)
                    out.attempts, out.tier, out.delivery = attempts, out.tier or "A", delivery.name
                    return out
                if now.kind != ch.kind:
                    steps.append(f"page now shows {now.kind}")
                    return self._out(now, NEEDS_PERSON, f"the page moved on to a different check ({now.kind})",
                                     tier="A", delivery=delivery.name, attempts=attempts, steps=steps)
            if _add_failure(task_id, ch) >= MAX_FAILURES:
                return self._out(ch, HARD_STOP, f"this {ch.kind} check failed {MAX_FAILURES} times",
                                 hard_stop=bc.HARD_STOP_REPEATED, tier="A", delivery=delivery.name,
                                 attempts=attempts, steps=steps)
            current = now or current
        return self._out(ch, NEEDS_PERSON, f"the {ch.kind} check did not pass after {attempts} "
                         f"{'try' if attempts == 1 else 'tries'}", tier="A", delivery=delivery.name,
                         attempts=attempts, steps=steps)

    def _hold_verdict(self, surface: Surface, first: bc.Challenge) -> Optional[bool]:
        """During a press-and-hold: True once passed, False if the page swapped to another challenge, else None."""
        try:
            now, facts = self._classify(surface)
        except Exception:
            return None  # mid-navigation: keep holding, the next poll will see
        if self._passed(first, now, facts):
            return True
        if now is not None and now.kind != first.kind:
            return False
        return None

    def _tier_c(self, ch: bc.Challenge, steps: List[str]) -> Outcome:
        solver, why = solver_for(ch.kind, self.cfg)
        if solver is None:
            steps.append(f"tier C: {why}")
            what = {"image": "an image puzzle", "slider": "a slider puzzle"}.get(ch.stage, "a puzzle")
            return self._out(ch, NEEDS_PERSON, f"the {ch.kind} check is {what}, and {why}", tier="C", steps=steps)
        # A configured solver exists but driving the grid is the next slice: say so, never half-do it.
        steps.append(f"tier C: solver {why} configured; grid driving is not built yet")
        return self._out(ch, NEEDS_PERSON, f"the {ch.kind} check is a puzzle; solving it in the engine is not built yet",
                         tier="C", steps=steps)


# ---- CDP plumbing (a browser on THIS Mac only) --------------------------------------------------------------

_LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def local_browser_ws(env: Dict[str, str], timeout: float = 3.0) -> Optional[str]:
    """The browser-level WebSocket of the CDP endpoint browser_exec used -- only when it is on this Mac."""
    raw = str(env.get("BU_CDP_WS") or env.get("BU_CDP_URL") or "").strip()
    if not raw:
        return None
    u = urlparse(raw)
    if (u.hostname or "") not in _LOOPBACK:
        return None
    if u.scheme in ("ws", "wss") and "/devtools/browser/" in u.path:
        return raw
    root = f"http://{u.hostname}:{u.port}" if u.port else None
    if not root:
        return None
    import urllib.request
    try:
        with urllib.request.urlopen(root + "/json/version", timeout=timeout) as r:  # noqa: S310 -- loopback only
            ws = str(json.loads(r.read().decode("utf-8")).get("webSocketDebuggerUrl") or "")
    except Exception as exc:
        logger.debug("captcha ladder: /json/version failed: %s", exc)
        return None
    return ws if (urlparse(ws).hostname or "") in _LOOPBACK else None


class CdpConn:
    """A small synchronous CDP client: one browser WebSocket, flattened page sessions, events ignored."""

    def __init__(self, ws_url: str, timeout: float = 10.0):
        from websockets.sync.client import connect
        self._cm = connect(ws_url, max_size=None, open_timeout=timeout, close_timeout=1)
        self.ws = self._cm.__enter__()
        self.timeout = timeout
        self._id = 0

    def call(self, method: str, params: Optional[dict] = None, session_id: Optional[str] = None,
             timeout: Optional[float] = None) -> dict:
        self._id += 1
        msg: Dict[str, Any] = {"id": self._id, "method": method, "params": params or {}}
        if session_id:
            msg["sessionId"] = session_id
        self.ws.send(json.dumps(msg))
        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"CDP {method} timed out")
            r = json.loads(self.ws.recv(timeout=left))
            if r.get("id") == self._id:
                if "error" in r:
                    raise RuntimeError(f"CDP {method}: {(r['error'] or {}).get('message')}")
                return r.get("result") or {}

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:
            pass

    def __enter__(self) -> "CdpConn":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class PageSession:
    def __init__(self, conn: CdpConn, target_id: str, session_id: str):
        self.conn, self.target_id, self.session_id = conn, target_id, session_id

    def call(self, method: str, params: Optional[dict] = None) -> dict:
        return self.conn.call(method, params, session_id=self.session_id)


def _frame_ids(tree: dict) -> List[str]:
    out = [str((tree.get("frame") or {}).get("id") or "")]
    for child in tree.get("childFrames") or ():
        out.extend(_frame_ids(child))
    return out


class CdpSurface(Surface):
    def __init__(self, conn: CdpConn, page: PageSession, browser_cfg: Optional[dict] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.conn, self.page, self.cfg, self.sleep = conn, page, browser_cfg or {}, sleep
        self._delivery: Optional[Tuple[Delivery, str]] = None

    def _oopif_urls(self) -> List[str]:
        try:
            ids = set(_frame_ids((self.page.call("Page.getFrameTree").get("frameTree") or {})))
            infos = self.conn.call("Target.getTargets").get("targetInfos") or []
        except Exception:
            return []
        return [str(t.get("url") or "") for t in infos
                if t.get("type") == "iframe" and str(t.get("parentFrameId") or "") in ids]

    def facts(self) -> bc.PageFacts:
        return bc.collect_facts(self.page, oopif_urls=self._oopif_urls())

    def _box(self, backend_node_id: int) -> Optional[Box]:
        try:
            self.page.call("DOM.scrollIntoViewIfNeeded", {"backendNodeId": backend_node_id})
        except Exception:
            pass
        try:
            quad = (self.page.call("DOM.getBoxModel", {"backendNodeId": backend_node_id}).get("model") or {}).get("border")
        except Exception:
            return None
        if not quad or len(quad) < 8:
            return None
        xs, ys = quad[0::2], quad[1::2]
        box = Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        return box if box.width > 4 and box.height > 4 else None

    def locate(self, ch: bc.Challenge) -> Optional[Box]:
        facts = self.facts()  # fresh: the widget may have re-rendered since the last try
        if ch.kind == bc.PERIMETERX:
            el = next((e for e in facts.elements if e.id == "px-captcha" and e.backend_node_id), None)
            return self._box(el.backend_node_id) if el else None
        pattern = {bc.TURNSTILE: bc._RE_TURNSTILE_FRAME, bc.CF_MANAGED: bc._RE_TURNSTILE_FRAME,
                   bc.RECAPTCHA_V2: bc._RE_RECAPTCHA_ANCHOR, bc.HCAPTCHA: bc._RE_HCAPTCHA}.get(ch.kind)
        if pattern is None:
            return None
        frames = [f for f in facts.frames if f.backend_node_id and pattern.search(f.url or "")
                  and not (ch.kind == bc.HCAPTCHA and "frame=challenge" in f.url)
                  and not (ch.kind == bc.RECAPTCHA_V2 and "size=invisible" in f.url)]
        for f in frames:
            fb = self._box(int(f.backend_node_id))  # type: ignore[arg-type]
            if fb is None:
                continue
            dx, dy, w, h = CHECKBOX_IN_FRAME[ch.kind](fb.width, fb.height)
            return Box(fb.x + dx, fb.y + dy, w, h)
        return None

    def delivery(self) -> Tuple[Delivery, str]:
        if self._delivery is None:
            os_clicks = bc.captcha_config(self.cfg).get("os_clicks", False) is True
            self._delivery = pick_delivery(self.page, os_clicks=os_clicks, backend_factory=_cua_backend_for_page,
                                           sleep=self.sleep)
        return self._delivery


def _cua_backend_for_page() -> Any:
    """A cua-driver backend captured on the driven browser's window. Only reached with os_clicks on and a
    visible, non-headless window -- see CuaDriverDelivery (not verified end to end yet). It calls the
    backend directly, outside computer_use's per-action approval, which is exactly why it sits behind the
    explicit ``browser.captcha.os_clicks`` opt-in (default off)."""
    from tools.computer_use.tool import _get_backend
    backend = _get_backend()
    cap = backend.capture(mode="ax", app="Google Chrome for Testing")
    if getattr(cap, "fix", None) or not getattr(cap, "width", 0):
        raise RuntimeError("the driven browser's window could not be captured")
    return backend


def find_challenge(conn: CdpConn, prefer_hosts: Sequence[str] = (), limit: int = 4,
                   hard_stop_hosts: Sequence[str] = bc.DEFAULT_HARD_STOP_HOSTS
                   ) -> Optional[Tuple[PageSession, bc.Challenge]]:
    """The first open http(s) tab showing a challenge -- tabs on ``prefer_hosts`` first."""
    infos = conn.call("Target.getTargets").get("targetInfos") or []
    pages = [t for t in infos if t.get("type") == "page" and str(t.get("url") or "").startswith(("http://", "https://"))]
    prefer = {h.lower() for h in prefer_hosts if h}
    pages.sort(key=lambda t: 0 if bc.host_of(str(t.get("url"))) in prefer else 1)
    for t in pages[:limit]:
        try:
            sid = conn.call("Target.attachToTarget", {"targetId": t["targetId"], "flatten": True})["sessionId"]
        except Exception:
            continue
        page = PageSession(conn, t["targetId"], sid)
        try:
            ch = bc.classify(CdpSurface(conn, page).facts(), hard_stop_hosts=hard_stop_hosts)
        except Exception as exc:
            logger.debug("captcha ladder: probe of %s failed: %s", str(t.get("url"))[:80], exc)
            ch = None
        if ch is not None:
            return page, ch
        try:
            conn.call("Target.detachFromTarget", {"sessionId": sid})
        except Exception:
            pass
    return None


# ---- the browser_exec hook ---------------------------------------------------------------------------------

def handoff_available() -> bool:
    """Whether ``browser_handoff`` (tools/browser_handoff_tool.py, branch browser-handoff) is installed."""
    try:
        return importlib.util.find_spec("tools.browser_handoff_tool") is not None
    except (ImportError, ValueError):
        return False


def next_step(out: Outcome, presence: Optional[dict]) -> str:
    """What the model does with the outcome. It never solves a challenge itself."""
    if out.outcome == PASSED:
        return (f"Moe's browser passed the {out.kind} check itself. Read the page again (page_info()) and carry on. "
                "Never try to solve a CAPTCHA yourself.")
    if out.outcome == REPORT_ONLY:
        return "Nothing to do for this check; carry on."
    live = bool((presence or {}).get("live"))
    stop = (" This is a hard stop: do not retry it, do not try it in another browser, and never try to solve it "
            "yourself." if out.outcome == HARD_STOP else " Do not try to solve it yourself.")
    if not live:
        return (f"The page needs the person ({out.reason}), and this is "
                f"{(presence or {}).get('why') or 'a turn nobody started live'}: report that this step is waiting "
                "for them, and stop there." + stop)
    if handoff_available():
        return (f"The page needs the person ({out.reason}). Call browser_handoff(reason=\"complete the "
                f"{out.kind.replace('_', ' ')} check on {out.host}\") to put it in front of them, then wait." + stop)
    return (f"The page needs the person ({out.reason}). Tell them in one line what the page needs and that you will "
            "carry on when they have done it." + stop)


def run_for_exec(result: Dict[str, Any], stdout: str, env: Dict[str, str], browser_cfg: Optional[dict],
                 task_id: Optional[str], presence: Optional[dict] = None,
                 ladder_factory: Callable[[Optional[dict]], Ladder] = lambda cfg: Ladder(cfg)) -> Optional[str]:
    """browser_exec's hook (own lane). Runs the ladder when the printed output suggests a challenge and the
    browser is on this Mac; puts ``result["captcha"]`` and returns the outcome string, else None."""
    if not bc.ladder_enabled(browser_cfg) or not bc.worth_a_look(stdout or ""):
        return None
    ws = local_browser_ws(env)
    if not ws:
        return None
    hosts = [bc.host_of(u) for u in _urls(stdout)]
    try:
        with CdpConn(ws) as conn:
            found = find_challenge(conn, hosts, hard_stop_hosts=bc.hard_stop_hosts(browser_cfg))
            if not found:
                return None
            page, ch = found
            ladder = ladder_factory(browser_cfg)
            out = ladder.run(CdpSurface(conn, page, browser_cfg, sleep=ladder.sleep), ch, task_id)
    except Exception as exc:
        logger.debug("captcha ladder failed: %s", exc)
        return None
    if out.outcome != PASSED and presence is None:
        try:
            from tools.browser_chrome_extension import turn_presence
            presence = turn_presence()
        except Exception:
            presence = {"live": False, "why": ""}
    apply_outcome(result, out, presence)
    return out.outcome


def apply_outcome(result: Dict[str, Any], out: Outcome, presence: Optional[dict]) -> None:
    d = out.to_dict()
    d["next"] = next_step(out, presence)
    result["captcha"] = d
    if out.outcome == HARD_STOP:
        result["hint"] = d["next"]


_URL_RE = re.compile(r"""https?://[^\s'"\\)<>]+""", re.I)


def _urls(text: str) -> List[str]:
    return _URL_RE.findall(text or "")[:20]
