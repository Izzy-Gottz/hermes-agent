"""Moe's engine-level CAPTCHA ladder: the detector, the state machine and its hard stops, the pointer path
and delivery strategies, the browser_exec hook's result shape, the tier C seam, and the description fixes.

The live tests at the bottom drive a real Chrome for Testing (skipped when it is not installed) against
LOCAL pages only: a fake Turnstile served for challenges.cloudflare.com through --host-resolver-rules,
with every other host mapped to NOTFOUND, so nothing here ever reaches a real challenge vendor.
"""

from __future__ import annotations

import http.server
import json
import math
import os
import random
import signal
import socketserver
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from tools import browser_captcha as bc
from tools import browser_captcha_input as bi
from tools import browser_captcha_ladder as bl
from tools import browser_captcha_solver as bs
from tools.browser_captcha import Element, Frame, PageFacts


# ---- fixtures: what CDP reads off each kind of page -------------------------------------------------
# URLs follow the vendors' real embed shapes (as served in 2026); sitekeys are placeholders.

TS_FRAME = ("https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/b/turnstile/if/ov2/av0/rcv/abc12/"
            "0x4AAAAAAAxyz/auto/fbE/new/normal/auto/")
RC_ANCHOR = "https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LcTEST&co=aHR0cHM6&hl=en&v=abc&size=normal&cb=x"
RC_ANCHOR_INV = "https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LcV3&co=aHR0cHM6&hl=en&v=abc&size=invisible&cb=y"
RC_BFRAME = "https://www.google.com/recaptcha/api2/bframe?hl=en&v=abc&k=6LcTEST"
RC_ENT_ANCHOR = "https://www.google.com/recaptcha/enterprise/anchor?ar=1&k=6LeENT&co=aHR0&size=invisible"
HC_BOX = ("https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html#frame=checkbox&id=0abc&host=shop.example"
          "&sitekey=10000000-ffff-ffff-ffff-000000000001&theme=light")
HC_CHALLENGE = ("https://newassets.hcaptcha.com/captcha/v1/abc/static/hcaptcha.html#frame=challenge&id=0abc"
                "&host=shop.example&sitekey=10000000-ffff-ffff-ffff-000000000001")
DD_FE = ("https://geo.captcha-delivery.com/captcha/?initialCid=AHrlqAAA&hash=ABC&cid=xyz&t=fe&referer=https%3A%2F%2F"
         "www.shop.example%2F&s=12345&e=abc")
DD_BV = DD_FE.replace("t=fe", "t=bv")


def ts(solved=False, host="shop.example", visible=True) -> PageFacts:
    return PageFacts(url=f"https://{host}/login", title="Sign in", frames=[Frame(TS_FRAME, 11, visible)],
                     elements=[Element("div", classes=("cf-turnstile",), attrs={"data-sitekey": "0x4AAAAAAAxyz"},
                                       backend_node_id=10)],
                     tokens={"cf-turnstile-response": "0.tok" if solved else ""})


def cf_managed(cleared=False, cookie=True) -> PageFacts:
    if cleared:
        return PageFacts(url="https://www.producthunt.com/products/moe-2", title="Moe - Product Hunt",
                         cookies={"cf_clearance": "abc"} if cookie else {})
    return PageFacts(url="https://www.producthunt.com/products/moe-2", title="Just a moment...", status=403,
                     headers={"cf-mitigated": "challenge"}, globals={"_cf_chl_opt": True},
                     frames=[Frame(TS_FRAME, 21, True)])


def rc(image=False, solved=False) -> PageFacts:
    return PageFacts(url="https://forms.example/contact", title="Contact",
                     frames=[Frame(RC_ANCHOR, 31, True), Frame(RC_BFRAME, 32, image)],
                     elements=[Element("div", classes=("g-recaptcha",), attrs={"data-sitekey": "6LcTEST"})],
                     tokens={"g-recaptcha-response": "03AF.tok" if solved else ""})


def hc(image=False, solved=False) -> PageFacts:
    return PageFacts(url="https://shop.example/signup", title="Sign up",
                     frames=[Frame(HC_BOX, 41, True), Frame(HC_CHALLENGE, 42, image)],
                     elements=[Element("div", classes=("h-captcha",), attrs={"data-sitekey": "10000000-ffff"})],
                     tokens={"h-captcha-response": "P1_tok" if solved else ""})


def px(solved=False) -> PageFacts:
    if solved:
        return PageFacts(url="https://www.skyscanner.com/transport/flights/", title="Flights", cookies={"_px3": "new"})
    return PageFacts(url="https://www.skyscanner.com/sttc/px/captcha-v2/index.html?url=L3RyYW5z", title="Skyscanner",
                     text="Are you a person or a robot?\nPress & Hold", elements=[Element("div", id="px-captcha", backend_node_id=51)],
                     cookies={"_pxhd": "x"})


def dd(t="fe") -> PageFacts:
    return PageFacts(url="https://www.shop.example/", title="shop.example",
                     frames=[Frame(DD_FE if t == "fe" else DD_BV, 61, True)], cookies={"datadome": "abc"})


# ---- detector -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("facts,kind,stage,solved", [
    (ts(), bc.TURNSTILE, "checkbox", False),
    (ts(solved=True), bc.TURNSTILE, "checkbox", True),
    (cf_managed(), bc.CF_MANAGED, "interstitial", False),
    (rc(), bc.RECAPTCHA_V2, "checkbox", False),
    (rc(solved=True), bc.RECAPTCHA_V2, "checkbox", True),
    (rc(image=True), bc.RECAPTCHA_V2, "image", False),
    (PageFacts(url="https://login.example/", frames=[Frame(RC_ANCHOR_INV, 1, False)]), bc.RECAPTCHA_V3, "report", False),
    (PageFacts(url="https://bank.example/", frames=[Frame(RC_ENT_ANCHOR, 1, False)]), bc.RECAPTCHA_V3, "report", False),
    (hc(), bc.HCAPTCHA, "checkbox", False),
    (hc(image=True), bc.HCAPTCHA, "image", False),
    (hc(solved=True), bc.HCAPTCHA, "checkbox", True),
    (px(), bc.PERIMETERX, "hold", False),
    (dd("fe"), bc.DATADOME, "slider", False),
    (dd("bv"), bc.DATADOME, "banned", False),
    (PageFacts(url="https://www.airline.example/", frames=[Frame("https://client-api.arkoselabs.com/fc/gc/?token=1", 1, True)]),
     bc.ARKOSE, "image", False),
    (PageFacts(url="https://www.game.example/", scripts=["https://static.geetest.com/v4/gcaptcha4.js"]), bc.GEETEST, "image", False),
    (PageFacts(url="https://store.example/", globals={"gokuProps": True},
               scripts=["https://abc123.edge.sdk.awswaf.com/abc123/def/captcha.js"]), bc.AWS_WAF, "image", False),
    (PageFacts(url="https://old.example/form", text="Please enter the characters you see in the image",
               elements=[Element("img", attrs={"src": "/captcha.php?r=1"})]), bc.IMAGE_TEXT, "image", False),
])
def test_detector_classifies_every_kind(facts, kind, stage, solved):
    ch = bc.classify(facts)
    assert ch is not None and (ch.kind, ch.stage, ch.solved) == (kind, stage, solved)
    assert ch.host == bc.host_of(facts.url)


def test_detector_reads_sitekeys_and_frames():
    assert bc.classify(ts()).sitekey == "0x4AAAAAAAxyz" and bc.classify(ts()).frame == TS_FRAME
    assert bc.classify(rc()).sitekey == "6LcTEST"
    assert bc.classify(PageFacts(url="https://a.example/", frames=[Frame(RC_ANCHOR_INV)])).sitekey == "6LcV3"


@pytest.mark.parametrize("facts", [
    PageFacts(url="https://bot.sannysoft.com/", title="Antibot", text="Intoli.com tests + additions"),
    PageFacts(url="https://blog.example/post", title="How reCAPTCHA works", text="an article about CAPTCHAs"),
    PageFacts(url="https://www.producthunt.com/", title="Product Hunt", cookies={"cf_clearance": "x"}),
])
def test_detector_leaves_ordinary_pages_alone(facts):
    assert bc.classify(facts) is None


def test_cf_managed_is_recognised_by_any_one_of_its_signals():
    for facts in (PageFacts(url="https://a.example/", headers={"cf-mitigated": "challenge"}),
                  PageFacts(url="https://a.example/", title="\U0001F434 Just a moment..."),
                  PageFacts(url="https://a.example/?__cf_chl_tk=abc"),
                  PageFacts(url="https://a.example/", globals={"_cf_chl_opt": True})):
        assert bc.classify(facts).kind == bc.CF_MANAGED


def test_hard_stops_are_detected():
    assert bc.classify(dd("bv")).hard_stop == bc.HARD_STOP_IP_BANNED
    assert bc.classify(dd("fe")).hard_stop is None
    for host in ("www.ticketmaster.com", "www.ticketmaster.co.uk", "concerts.livenation.com", "www.stubhub.com",
                 "seatgeek.com", "www.axs.com", "shop.dice.fm"):
        ch = bc.classify(ts(host=host))
        assert ch.hard_stop == bc.HARD_STOP_TICKETING, host
    for host in ("www.dice.com", "axs.example", "tickets.example", "ticketmasters-fan.blog"):
        assert bc.classify(ts(host=host)).hard_stop is None, host


def test_hard_stop_hosts_are_extendable_from_config():
    cfg = {"captcha": {"hard_stop_hosts": ["tickets.example", "boxoffice"]}}
    hosts = bc.hard_stop_hosts(cfg)
    assert bc.classify(ts(host="shop.tickets.example"), hard_stop_hosts=hosts).hard_stop == bc.HARD_STOP_TICKETING
    assert bc.classify(ts(host="boxoffice.co.uk"), hard_stop_hosts=hosts).hard_stop == bc.HARD_STOP_TICKETING
    assert set(bc.DEFAULT_HARD_STOP_HOSTS) <= set(hosts)          # extends, never replaces


def test_one_signature_table_for_the_extension_lane_and_the_detector():
    from tools import browser_chrome_extension as lane
    assert lane._BLOCK_SIGNATURES is bc.BLOCK_SIGNATURES
    assert lane.detect_block("Are you a person or a robot?\nPress & Hold") == "perimeterx"


def test_worth_a_look_triggers_on_walls_and_widgets_only():
    assert bc.worth_a_look("{'url': 'https://a.example/', 'title': 'Just a moment...'}")
    assert bc.worth_a_look('<div class="cf-turnstile" data-sitekey="x">')
    assert not bc.worth_a_look("{'url': 'https://a.example/', 'title': 'Home'}\nWelcome")


def test_collect_facts_reads_closed_shadow_frames_tokens_and_cookies():
    """The pierced DOM (a closed shadow root included) and the JS probe both feed PageFacts."""
    doc = {"root": {"nodeName": "#document", "children": [{"nodeName": "HTML", "children": [
        {"nodeName": "SCRIPT", "attributes": ["src", "https://static.geetest.com/v4/gcaptcha4.js"]},
        {"nodeName": "DIV", "backendNodeId": 7, "attributes": ["class", "cf-turnstile", "data-sitekey", "0x4K"],
         "shadowRoots": [{"nodeName": "#document-fragment", "shadowRootType": "closed", "children": [
             {"nodeName": "IFRAME", "backendNodeId": 8, "attributes": ["src", TS_FRAME]}]}]}]}]}}
    probe = {"url": "https://a.example/", "title": "A", "status": 200, "text": "hi",
             "tokens": {"cf-turnstile-response": ""}, "globals": {"_cf_chl_opt": False, "gokuProps": False},
             "visible_frames": [TS_FRAME]}

    class Page:
        def call(self, method, params=None):
            return {"Runtime.evaluate": {"result": {"value": json.dumps(probe)}}, "DOM.getDocument": doc,
                    "Network.getCookies": {"cookies": [{"name": "cf_clearance", "value": "z"}]}}[method]

    facts = bc.collect_facts(Page(), oopif_urls=["https://other.example/frame"])
    assert [f.url for f in facts.frames] == [TS_FRAME, "https://other.example/frame"]
    assert facts.frames[0].backend_node_id == 8 and facts.frames[0].visible is True
    assert facts.scripts == ["https://static.geetest.com/v4/gcaptcha4.js"]
    assert facts.elements[0].attrs["data-sitekey"] == "0x4K" and facts.cookies == {"cf_clearance": "z"}


# ---- the state machine --------------------------------------------------------------------------------------

class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class FakeDelivery(bi.Delivery):
    name = "fake"

    def __init__(self, surface):
        self.surface = surface
        self.clicks: List[bi.Point] = []
        self.holds = 0

    def available(self):
        return True, "fake"

    def click(self, box, rng):
        p = bi.point_inside(box, rng)
        assert box.contains(p)
        self.clicks.append(p)
        self.surface.on_input()
        return p

    def hold(self, box, rng, until, max_s, poll_s=0.2):
        self.holds += 1
        self.surface.on_input()
        start = self.surface.clock()
        while self.surface.clock() - start < max_s:
            self.surface.clock.sleep(poll_s)
            v = until()
            if v is not None:
                return v, self.surface.clock() - start
        return None, self.surface.clock() - start


class FakeSurface(bl.Surface):
    """A page whose facts come from ``page(state)``; each input bumps ``state['inputs']``."""

    def __init__(self, page, clock, box=bi.Box(100, 200, 26, 24)):
        self.page, self.clock, self.box = page, clock, box
        self.state = {"inputs": 0, "t0": clock()}
        self.locates = 0
        self.fd = FakeDelivery(self)

    def facts(self):
        return self.page(self.state, self.clock() - self.state["t0"])

    def locate(self, ch):
        self.locates += 1
        return self.box

    def delivery(self):
        return self.fd, "fake"

    def on_input(self):
        self.state["inputs"] += 1


@pytest.fixture(autouse=True)
def _fresh_failures():
    bl.reset_failures()
    yield
    bl.reset_failures()


def ladder(clock, cfg=None):
    return bl.Ladder(cfg or {}, sleep=clock.sleep, clock=clock, rng=random.Random(7))


def run(page, cfg=None, task="t1"):
    clock = Clock()
    s = FakeSurface(page, clock)
    first = bc.classify(s.facts(), hard_stop_hosts=bc.hard_stop_hosts(cfg))
    return ladder(clock, cfg).run(s, first, task), s, clock


def test_managed_challenge_is_waited_out_without_input():
    out, s, clock = run(lambda st, dt: cf_managed(cleared=dt >= 6))
    assert out.outcome == bl.PASSED and out.tier == "A" and s.fd.clicks == [] and s.state["inputs"] == 0
    assert clock() - 1000 <= 7


def test_managed_challenge_needs_cf_clearance_not_just_a_new_title():
    out, s, clock = run(lambda st, dt: cf_managed(cleared=dt >= 2, cookie=False))
    assert out.outcome != bl.PASSED


def test_managed_challenge_waits_at_most_fifteen_seconds_then_clicks_its_box_once():
    out, s, clock = run(lambda st, dt: cf_managed(cleared=st["inputs"] >= 1))
    assert out.outcome == bl.PASSED and len(s.fd.clicks) == 1
    assert out.steps[0].startswith("waited up to 15s: waiting")


def test_turnstile_is_clicked_relocated_every_try_and_passes():
    out, s, _ = run(lambda st, dt: ts(solved=st["inputs"] >= 2))
    assert out.outcome == bl.PASSED and out.attempts == 2 and len(s.fd.clicks) == 2
    assert s.locates == 2                                   # re-located before each try
    assert bl.failures("t1", bc.classify(ts())) == 0        # a pass forgets earlier failures


def test_turnstile_that_auto_passes_during_the_wait_gets_no_click():
    out, s, _ = run(lambda st, dt: ts(solved=dt > 2))
    assert out.outcome == bl.PASSED and s.fd.clicks == []


def test_three_failures_is_a_hard_stop_and_stays_one():
    out, s, _ = run(lambda st, dt: ts())
    assert out.outcome == bl.HARD_STOP and out.hard_stop == bc.HARD_STOP_REPEATED and len(s.fd.clicks) == 3
    again, s2, _ = run(lambda st, dt: ts())
    assert again.outcome == bl.HARD_STOP and s2.fd.clicks == [] and s2.locates == 0   # never tried again


def test_failures_are_counted_across_calls_of_one_task():
    bl._add_failure("t1", bc.classify(ts()))
    bl._add_failure("t1", bc.classify(ts()))
    out, s, _ = run(lambda st, dt: ts())
    assert out.outcome == bl.HARD_STOP and len(s.fd.clicks) == 1
    other, s3, _ = run(lambda st, dt: ts(), task="t2")      # another task starts fresh
    assert len(s3.fd.clicks) == 3


def test_ticketing_is_a_hard_stop_before_any_wait_or_input():
    out, s, clock = run(lambda st, dt: ts(host="www.ticketmaster.com", solved=st["inputs"] >= 1))
    assert out.outcome == bl.HARD_STOP and out.hard_stop == bc.HARD_STOP_TICKETING
    assert s.fd.clicks == [] and s.locates == 0 and clock() == 1000.0
    out2, s2, _ = run(lambda st, dt: ts(host="tix.example"), cfg={"captcha": {"hard_stop_hosts": ["tix.example"]}})
    assert out2.outcome == bl.HARD_STOP and s2.fd.clicks == []


def test_datadome_banned_is_a_hard_stop_and_its_slider_goes_to_the_person():
    out, s, _ = run(lambda st, dt: dd("bv"))
    assert out.outcome == bl.HARD_STOP and out.hard_stop == bc.HARD_STOP_IP_BANNED and s.state["inputs"] == 0
    out, s, _ = run(lambda st, dt: dd("fe"))
    assert out.outcome == bl.NEEDS_PERSON and out.tier == "C" and s.state["inputs"] == 0


def test_recaptcha_escalating_to_images_goes_to_the_person():
    out, s, _ = run(lambda st, dt: rc(image=st["inputs"] >= 1))
    assert out.outcome == bl.NEEDS_PERSON and out.tier == "C" and len(s.fd.clicks) == 1
    assert "no recognition solver" in out.reason


def test_hcaptcha_checkbox_click_passes():
    out, s, _ = run(lambda st, dt: hc(solved=st["inputs"] >= 1))
    assert out.outcome == bl.PASSED and len(s.fd.clicks) == 1


def test_recaptcha_v3_is_report_only():
    out, s, _ = run(lambda st, dt: PageFacts(url="https://login.example/", frames=[Frame(RC_ANCHOR_INV)]))
    assert out.outcome == bl.REPORT_ONLY and s.state["inputs"] == 0


def test_press_and_hold_is_released_when_the_page_says_so_not_after_a_fixed_time():
    for done_after in (1.4, 6.3):
        bl.reset_failures()
        out, s, clock = run(lambda st, dt, d=done_after: px(solved=st["inputs"] >= 1 and dt >= d))
        assert out.outcome == bl.PASSED and s.fd.holds == 1
        assert abs((clock() - 1000) - done_after) < 1.0, (clock() - 1000, done_after)


def test_press_and_hold_that_never_completes_is_bounded_and_counted():
    out, s, clock = run(lambda st, dt: px())
    assert out.outcome == bl.HARD_STOP and s.fd.holds == 3 and clock() - 1000 <= bl.TOTAL_DEADLINE_S + 1


def test_missing_target_goes_to_the_person():
    clock = Clock()
    s = FakeSurface(lambda st, dt: ts(), clock)
    s.locate = lambda ch: None
    out = ladder(clock).run(s, bc.classify(ts()), "t1")
    assert out.outcome == bl.NEEDS_PERSON and "could not be found" in out.reason and s.fd.clicks == []


def test_the_whole_ladder_respects_its_deadline():
    clock = Clock()
    s = FakeSurface(lambda st, dt: cf_managed(), clock)
    out = bl.Ladder({}, sleep=clock.sleep, clock=clock, rng=random.Random(1), deadline_s=5).run(s, bc.classify(cf_managed()))
    assert out.outcome != bl.PASSED and clock() - 1000 <= 6


# ---- tier C seam ----------------------------------------------------------------------------------------------

class _Solver:
    name = "fake"

    def supports(self, kind):
        return True

    def solve(self, request):
        return bs.SolveAnswer(tiles=(0, 4, 4, 8))


def test_no_solver_by_default_and_cloudflare_never_gets_one():
    assert bs.solver_for(bc.HCAPTCHA, {})[0] is None
    bs.register_solver("fake", lambda cfg: _Solver())
    try:
        cfg = {"captcha": {"solver": {"provider": "fake"}}}
        assert bs.solver_for(bc.HCAPTCHA, cfg)[0] is not None
        for kind in (bc.TURNSTILE, bc.CF_MANAGED):
            solver, why = bs.solver_for(kind, cfg)
            assert solver is None and "third party" in why
        assert bs.solver_for(bc.PERIMETERX, cfg)[0] is None       # a hold is not a recognition task
    finally:
        bs.unregister_solver("fake")


def test_solver_answers_are_validated_and_the_request_has_no_audio_route():
    assert bs.validate_answer(bs.SolveAnswer(tiles=(0, 4, 4, 8)), (3, 3)) == [0, 4, 8]
    with pytest.raises(ValueError):
        bs.validate_answer(bs.SolveAnswer(tiles=(9,)), (3, 3))
    fields = set(bs.SolveRequest.__dataclass_fields__)
    assert fields == {"kind", "image_png", "instruction", "grid"}


def test_a_configured_solver_still_hands_over_until_grid_driving_exists():
    bs.register_solver("fake", lambda cfg: _Solver())
    try:
        out, s, _ = run(lambda st, dt: hc(image=True), cfg={"captcha": {"solver": {"provider": "fake"}}})
        assert out.outcome == bl.NEEDS_PERSON and out.tier == "C" and "not built yet" in out.reason
    finally:
        bs.unregister_solver("fake")


# ---- pointer path + delivery --------------------------------------------------------------------------------

def test_human_path_shape():
    rng = random.Random(3)
    for trial in range(200):
        start = (rng.uniform(0, 800), rng.uniform(0, 600))
        end = (rng.uniform(0, 800), rng.uniform(0, 600))
        pts, delays = bi.human_path(start, end, rng)
        assert bi.MIN_STEPS <= len(pts) <= bi.MAX_STEPS
        assert 0.3 - 1e-9 <= sum(delays) <= 0.9 + 1e-9
        assert pts[-1] == end
    pts, _ = bi.human_path((0, 0), (400, 0), random.Random(5))
    assert max(abs(y) for _, y in pts) > 10                 # curved, not a straight line


def test_point_inside_is_offset_from_the_centre_and_inside():
    box = bi.Box(10, 20, 30, 30)
    pts = [bi.point_inside(box, random.Random(i)) for i in range(100)]
    assert all(box.contains(p) for p in pts) and len({p for p in pts}) > 90


class RecPage:
    def __init__(self):
        self.events = []

    def call(self, method, params=None):
        self.events.append((method, dict(params or {})))
        return {}


def test_cdp_click_moves_along_a_path_then_presses_inside_the_box():
    page, clock = RecPage(), Clock()
    d = bi.CdpDelivery(page, sleep=clock.sleep, clock=clock)
    box = bi.Box(300, 200, 26, 24)
    target = d.click(box, random.Random(2))
    types = [p["type"] for _, p in page.events]
    moves = types.count("mouseMoved")
    assert bi.MIN_STEPS <= moves <= bi.MAX_STEPS and types[-2:] == ["mousePressed", "mouseReleased"]
    press = page.events[-2][1]
    assert box.contains((press["x"], press["y"])) and (press["x"], press["y"]) == target
    assert press["button"] == "left" and press["clickCount"] == 1
    assert 0.3 <= clock() - 1000 <= 1.3


def test_cdp_hold_releases_on_the_verdict():
    page, clock = RecPage(), Clock()
    d = bi.CdpDelivery(page, sleep=clock.sleep, clock=clock)
    polls = {"n": 0}

    def until():
        polls["n"] += 1
        return True if polls["n"] >= 9 else None

    verdict, held = d.hold(bi.Box(0, 0, 200, 80), random.Random(1), until, max_s=15)
    types = [p["type"] for _, p in page.events]
    assert verdict is True and types.count("mousePressed") == 1 and types[-1] == "mouseReleased"
    assert 1.5 <= held <= 2.5                                # 9 polls of 0.2 s, not a fixed hold time


def test_os_clicks_need_a_visible_window_and_fall_back_to_cdp():
    class GeoPage(RecPage):
        def __init__(self, geo):
            super().__init__()
            self.geo = geo

        def call(self, method, params=None):
            return {"result": {"value": json.dumps(self.geo)}}

    headless = {"sx": 0, "sy": 0, "ix": 0, "iy": 143, "vis": "visible", "ua": "Mozilla/5.0 HeadlessChrome/154"}
    hidden = {**headless, "ua": "Mozilla/5.0 Chrome/154", "vis": "hidden"}
    shown = {**hidden, "vis": "visible", "sx": 300, "sy": 200}
    assert bi.pick_delivery(GeoPage(headless), os_clicks=True, backend_factory=lambda: object())[0].name == "cdp"
    assert bi.pick_delivery(GeoPage(hidden), os_clicks=True, backend_factory=lambda: object())[0].name == "cdp"
    assert bi.pick_delivery(GeoPage(shown), os_clicks=False, backend_factory=lambda: object())[0].name == "cdp"
    d, why = bi.pick_delivery(GeoPage(shown), os_clicks=True, backend_factory=lambda: object())
    assert d.name == "cua_driver" and d.verified is False
    g = bi.window_geometry(GeoPage(shown))
    assert g.to_window((100, 50)) == (100, 193) and g.to_screen((100, 50)) == (400, 393)
    # The strategy's own gate, not just pick_delivery's: a headless or hidden window never takes an OS click.
    assert bi.CuaDriverDelivery(object(), bi.window_geometry(GeoPage(headless))).available()[0] is False
    assert bi.CuaDriverDelivery(object(), bi.window_geometry(GeoPage(hidden))).available()[0] is False
    assert bi.CuaDriverDelivery(None, g).available()[0] is False


def test_cua_driver_click_is_foreground_and_window_local():
    calls = []

    class Backend:
        def click(self, **kw):
            calls.append(kw)
            return type("R", (), {"ok": True})()

    g = bi.WindowGeometry(left=300, top=200, inset_x=0, inset_y=143, visible=True, headless=False)
    d = bi.CuaDriverDelivery(Backend(), g)
    p = d.click(bi.Box(100, 50, 20, 20), random.Random(1))
    assert calls[0]["delivery_mode"] == "foreground"
    assert (calls[0]["x"], calls[0]["y"]) == (round(p[0]), round(p[1] + 143))
    with pytest.raises(NotImplementedError):
        d.hold(bi.Box(0, 0, 1, 1), random.Random(1), lambda: True, 1)


# ---- the browser_exec hook ---------------------------------------------------------------------------------------

@pytest.fixture()
def cli(monkeypatch):
    from tools import browser_use_cli as bu
    calls = {"stdout": "ok\n", "cfg": {}}

    def fake_run(cmd, code, env, timeout):
        return subprocess.CompletedProcess(cmd, 0, calls["stdout"], "")

    def fake_route(env, session, task_id, local):
        env["BU_CDP_URL"] = calls.get("cdp", "http://127.0.0.1:9333")
        return None

    monkeypatch.setattr(bu, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", fake_run)
    monkeypatch.setattr(bu, "_route_backend", fake_route)
    monkeypatch.setattr(bu, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu, "_workspace_dir", lambda task_id: None)
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: calls["cfg"])
    return calls


@pytest.fixture()
def canned(monkeypatch):
    """Everything below the hook's CDP boundary replaced: the ladder 'finds' ``state['challenge']`` and
    ``state['outcome']`` is what running it produced."""
    state = {"ws_for": [], "ran": 0}

    def fake_ws(env, timeout=3.0):
        raw = env.get("BU_CDP_WS") or env.get("BU_CDP_URL") or ""
        state["ws_for"].append(raw)
        return "ws://127.0.0.1:9333/devtools/browser/x" if "127.0.0.1" in raw else None

    class Conn:
        def __init__(self, ws, timeout=10.0):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    class L:
        sleep = staticmethod(lambda s: None)

        def run(self, surface, ch, task_id):
            state["ran"] += 1
            return state["outcome"]

    monkeypatch.setattr(bl, "local_browser_ws", fake_ws)
    monkeypatch.setattr(bl, "CdpConn", Conn)
    monkeypatch.setattr(bl, "find_challenge", lambda conn, hosts, **kw: (object(), state["challenge"]))
    monkeypatch.setattr(bl, "CdpSurface", lambda *a, **kw: None)
    monkeypatch.setattr(bl, "Ladder", lambda cfg: L())
    return state


PH_WALL = "{'url': 'https://www.producthunt.com/products/moe-2', 'title': 'Just a moment...'}"


def _exec(code='new_tab("https://www.producthunt.com/products/moe-2")\nprint(page_info())'):
    from tools import browser_use_cli as bu
    return json.loads(bu.browser_exec(code, task_id="t"))


def test_hook_passed_clears_blocked_by_and_says_carry_on(cli, canned):
    cli["stdout"] = PH_WALL
    canned["challenge"] = bc.classify(cf_managed())
    canned["outcome"] = bl.Outcome(kind=bc.CF_MANAGED, host="www.producthunt.com", outcome=bl.PASSED,
                                   reason="the check cleared on its own", tier="A", steps=["waited"])
    out = _exec()
    assert "blocked_by" not in out and canned["ran"] == 1
    c = out["captcha"]
    assert c["outcome"] == "passed" and c["kind"] == "cf_managed" and c["host"] == "www.producthunt.com"
    assert "carry on" in c["next"] and "Never try to solve" in c["next"]


def test_hook_needs_person_points_at_browser_handoff_or_the_person(cli, canned, monkeypatch):
    cli["stdout"] = PH_WALL
    canned["challenge"] = bc.classify(rc(image=True))
    canned["outcome"] = bl.Outcome(kind=bc.RECAPTCHA_V2, host="forms.example", outcome=bl.NEEDS_PERSON,
                                   reason="the recaptcha_v2 check is an image puzzle", tier="C")
    monkeypatch.setattr("tools.browser_chrome_extension.turn_presence", lambda: {"live": True, "why": ""})
    monkeypatch.setattr(bl, "handoff_available", lambda: True)
    out = _exec()
    assert out["blocked_by"] == "cloudflare" and out["captcha"]["outcome"] == "needs_person"
    assert "browser_handoff(reason=" in out["captcha"]["next"]
    monkeypatch.setattr(bl, "handoff_available", lambda: False)
    out = _exec()
    assert "browser_handoff" not in out["captcha"]["next"] and "Tell them" in out["captcha"]["next"]
    monkeypatch.setattr("tools.browser_chrome_extension.turn_presence", lambda: {"live": False, "why": "a cron job"})
    out = _exec()
    assert "a cron job" in out["captcha"]["next"] and "waiting" in out["captcha"]["next"]


def test_hook_hard_stop_replaces_the_try_it_in_chrome_hint(cli, canned, monkeypatch, tmp_path):
    cli["stdout"] = PH_WALL
    cli["cfg"] = {"chrome_extension": {"enabled": True}}
    from tools import browser_chrome_extension as lane
    monkeypatch.setattr(lane, "read_bridge", lambda hermes_home=None: {"ws_url": "ws://127.0.0.1:1/devtools/browser/x"})
    monkeypatch.setattr(lane, "turn_presence", lambda: {"live": True, "why": ""})
    monkeypatch.setattr(lane, "remember_blocking_host", lambda host, **kw: None)
    canned["challenge"] = bc.classify(ts(host="www.ticketmaster.com"))
    canned["outcome"] = bl.Outcome(kind=bc.TURNSTILE, host="www.ticketmaster.com", outcome=bl.HARD_STOP,
                                   reason="www.ticketmaster.com sells tickets", hard_stop=bc.HARD_STOP_TICKETING)
    out = _exec('new_tab("https://www.ticketmaster.com/")\nprint(page_info())')
    assert out["captcha"]["outcome"] == "hard_stop" and out["captcha"]["hard_stop"] == "ticketing"
    assert 'where="chrome"' not in out["hint"] and "hard stop" in out["hint"]


def test_hook_does_not_run_off_this_mac_when_disabled_or_on_a_normal_page(cli, canned):
    canned["challenge"] = bc.classify(ts())
    canned["outcome"] = bl.Outcome(kind=bc.TURNSTILE, host="shop.example", outcome=bl.PASSED)
    cli["stdout"] = PH_WALL
    cli["cdp"] = "wss://connect.browserbase.com/devtools/browser/abc"     # a cloud browser: a third party
    out = _exec()
    assert canned["ran"] == 0 and "captcha" not in out and out["blocked_by"] == "cloudflare"
    cli["cdp"] = "http://127.0.0.1:9333"
    cli["cfg"] = {"captcha": {"enabled": False}}
    out = _exec()
    assert canned["ran"] == 0 and "captcha" not in out
    cli["cfg"] = {}
    cli["stdout"] = "{'url': 'https://shop.example/', 'title': 'Shop'}\nWelcome"
    out = _exec()
    assert canned["ran"] == 0 and "captcha" not in out and "blocked_by" not in out


def test_a_broken_ladder_never_breaks_browser_exec(cli, monkeypatch):
    cli["stdout"] = PH_WALL
    monkeypatch.setattr(bl, "run_for_exec", lambda *a, **kw: 1 / 0)
    out = _exec()
    assert out["success"] and out["blocked_by"] == "cloudflare" and "captcha" not in out


def test_local_browser_ws_refuses_anything_but_loopback():
    assert bl.local_browser_ws({"BU_CDP_WS": "ws://127.0.0.1:9/devtools/browser/a"}) == "ws://127.0.0.1:9/devtools/browser/a"
    assert bl.local_browser_ws({"BU_CDP_WS": "wss://connect.browserbase.com/devtools/browser/a"}) is None
    assert bl.local_browser_ws({"BU_CDP_URL": "https://cdp.example.com:9222"}) is None
    assert bl.local_browser_ws({}) is None


# ---- description fixes ---------------------------------------------------------------------------------------

def test_browser_vision_no_longer_invites_solving_captchas():
    from tools import browser_tool as bt
    schema = next(s for s in bt.BROWSER_TOOL_SCHEMAS if s["name"] == "browser_vision")
    d = schema["description"]
    assert "especially for CAPTCHAs" not in d and "never try to solve one yourself" in d


def test_bot_detection_warning_points_at_the_person_not_browserbase():
    from tools import browser_tool as bt
    resp = {}
    bt._add_navigate_warnings(resp, "Just a moment...", None)
    w = resp["bot_detection_warning"]
    assert "BROWSERBASE" not in w and "stealth" not in w.lower() and "Scale plan" not in w
    assert "Do not try to solve it yourself" in w and "browser_handoff" in w


# ---- live: Chrome for Testing on local pages only ----------------------------------------------------------------

def _cft() -> Optional[str]:
    root = Path.home() / ".agent-browser" / "browsers"
    for app in sorted(root.glob("chrome-*/Google Chrome for Testing.app"), reverse=True):
        exe = app / "Contents" / "MacOS" / "Google Chrome for Testing"
        if exe.exists():
            return str(exe)
    return None


CFT = _cft()
live = pytest.mark.skipif(CFT is None, reason="Chrome for Testing not installed under ~/.agent-browser/browsers")

_REC = """<script>
window.__ev = [];
function rec(where, e){ const r = {where, type: e.type, screenX: e.screenX, screenY: e.screenY, clientX: e.clientX,
  clientY: e.clientY, isTrusted: e.isTrusted}; window.__ev.push(r);
  if (window.parent !== window) window.parent.postMessage({ev: r}, '*'); }
</script>"""

# A fake Turnstile at Turnstile's own geometry (300x65, the box at left 16, vertically centred). It
# passes only on a trusted click that lands inside the box, like the real thing's first gate.
_FAKE_TS_FRAME = _REC + """<body style="margin:0;width:300px;height:65px;background:#fafafa">
<div id="box" style="position:absolute;left:16px;top:20px;width:26px;height:24px;border:2px solid #666"></div>
<script>
document.addEventListener('mousedown', e => rec('frame', e));
document.addEventListener('click', e => { rec('frame', e);
  const b = document.getElementById('box').getBoundingClientRect();
  if (e.isTrusted && e.clientX >= b.left && e.clientX <= b.right && e.clientY >= b.top && e.clientY <= b.bottom)
    window.parent.postMessage({turnstile: 'XXXX.DUMMY.TOKEN'}, '*');
});
</script></body>"""

_FAKE_PX = """<html><head><title>Access to this page has been denied</title></head><body>
<p>Press &amp; Hold to confirm you are a human (and not a bot).</p>
<div id="px-captcha" style="position:absolute;left:120px;top:160px;width:260px;height:90px;background:#eee"></div>
<script>
document.cookie = "_pxhd=abc; path=/";
let t0 = null, need = 1200 + Math.floor(Math.random() * 800);
const el = document.getElementById('px-captcha');
el.addEventListener('mousedown', e => { if (e.isTrusted) t0 = performance.now(); });
el.addEventListener('mouseup', e => { t0 = null; });
setInterval(() => { if (t0 !== null && performance.now() - t0 >= need) {
  document.cookie = "_px3=passed; path=/"; el.remove(); document.title = "Flights"; t0 = null; } }, 50);
</script></body></html>"""


def _serve(pages, tls: Optional[tuple] = None):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = pages.get(self.path.split("?")[0], pages.get("*", "")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    if tls:
        import ssl
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(*tls)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture(scope="module")
def chrome(tmp_path_factory):
    # cloudflare.com is HSTS-preloaded (subdomains included), so Chrome only ever asks
    # challenges.cloudflare.com over https: serve the fake frame over TLS with a throwaway self-signed
    # certificate, and let this one test browser accept it.
    certdir = tmp_path_factory.mktemp("tls")
    cert, key = str(certdir / "c.pem"), str(certdir / "k.pem")
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj",
                        "/CN=challenges.cloudflare.com", "-keyout", key, "-out", cert],
                       check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("openssl is needed to serve the fake Turnstile frame over https")
    frame_srv = _serve({"*": _FAKE_TS_FRAME}, tls=(cert, key))
    fport = frame_srv.server_address[1]
    top = _REC + f"""<html><head><title>Sign in</title></head><body style="margin:0">
<form><input name="email"><div class="cf-turnstile" data-sitekey="1x00000000000000000000AA"
 style="position:absolute;left:200px;top:150px">
<iframe src="https://challenges.cloudflare.com:{fport}/cdn-cgi/challenge-platform/h/b/turnstile/if/ov2/av0/rcv/x/1x00/auto/fbE/new/normal/auto/"
 style="width:300px;height:65px;border:0"></iframe>
<input type="hidden" name="cf-turnstile-response" value=""></div></form>
<script>window.addEventListener('message', m => {{ if (m.data && m.data.ev) window.__ev.push(m.data.ev);
  if (m.data && m.data.turnstile) document.querySelector('[name=cf-turnstile-response]').value = m.data.turnstile; }});</script>
</body></html>"""
    top_srv = _serve({"/login": top, "/px": _FAKE_PX})
    ud = tempfile.mkdtemp(prefix="captcha-live-")
    rules = (f"MAP challenges.cloudflare.com 127.0.0.1, MAP * ~NOTFOUND, EXCLUDE 127.0.0.1, EXCLUDE localhost")
    proc = subprocess.Popen([CFT, "--headless=new", f"--user-data-dir={ud}", "--remote-debugging-port=0",
                             "--no-first-run", "--no-default-browser-check", f"--host-resolver-rules={rules}",
                             "--ignore-certificate-errors",
                             # This Mac routes browsers through a filtering proxy (127.0.0.1:9176, measured in a
                             # net-log 2026-09-25) that drops the mapped challenge host: go direct.
                             "--no-proxy-server",
                             "--window-size=900,700", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    port_file = Path(ud) / "DevToolsActivePort"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not (port_file.exists() and port_file.read_text().strip()):
        time.sleep(0.3)
    if not (port_file.exists() and port_file.read_text().strip()):
        os.killpg(proc.pid, signal.SIGKILL)
        pytest.skip("Chrome for Testing did not start in 90 s (a loaded Mac is not a result)")
    port, path = port_file.read_text().split("\n")[:2]
    yield {"ws": f"ws://127.0.0.1:{port}{path}", "top": top_srv.server_address[1]}
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    import shutil
    shutil.rmtree(ud, ignore_errors=True)
    frame_srv.shutdown()
    top_srv.shutdown()


@pytest.fixture()
def conn(chrome):
    """A CDP connection whose tabs are closed afterwards, so live tests never see each other's pages."""
    with bl.CdpConn(chrome["ws"], timeout=30) as c:
        yield c
        for t in c.call("Target.getTargets")["targetInfos"]:
            if t.get("type") == "page" and str(t.get("url") or "").startswith("http"):
                c.call("Target.closeTarget", {"targetId": t["targetId"]})


def _open(conn, url, settle=lambda page: True, timeout=45):
    tid = conn.call("Target.createTarget", {"url": url})["targetId"]
    sid = conn.call("Target.attachToTarget", {"targetId": tid, "flatten": True})["sessionId"]
    page = bl.PageSession(conn, tid, sid)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = page.call("Runtime.evaluate", {"expression": "document.readyState", "returnByValue": True})
        if (v.get("result") or {}).get("value") == "complete" and settle(page):
            return page
        time.sleep(0.3)
    pytest.skip(f"{url} did not load in {timeout} s (a loaded Mac is not a result)")


def _frame_ready(conn) -> bool:
    """The fake Turnstile frame has really loaded (an out-of-process frame, not a chrome-error page)."""
    for t in conn.call("Target.getTargets")["targetInfos"]:
        if t.get("type") == "iframe" and "challenges.cloudflare.com" in str(t.get("url")):
            sid = conn.call("Target.attachToTarget", {"targetId": t["targetId"], "flatten": True})["sessionId"]
            try:
                r = conn.call("Runtime.evaluate", {"expression": "location.href + '|' + document.readyState + '|' + "
                              "!!document.getElementById('box')", "returnByValue": True}, session_id=sid)
            finally:
                conn.call("Target.detachFromTarget", {"sessionId": sid})
            v = str((r.get("result") or {}).get("value") or "")
            if v.startswith("https://challenges.cloudflare.com") and v.endswith("|complete|true"):
                return True
    return False


def _events(page):
    raw = page.call("Runtime.evaluate", {"expression": "JSON.stringify(window.__ev || [])", "returnByValue": True})
    return json.loads(raw["result"]["value"])


@live
def test_live_cdp_iframe_screen_coordinates(chrome, conn):
    """Chromium 40280325 says CDP clicks inside an iframe report screenX == clientX. Measured on CfT 154,
    2026-09-25 (window at 300,200): iframe clientX=100 screenX=600, i.e. real screen coordinates."""
    page = _open(conn, f"http://127.0.0.1:{chrome['top']}/login",
                 settle=lambda p: _frame_ready(conn))
    wid = conn.call("Browser.getWindowForTarget", {"targetId": page.target_id})["windowId"]
    conn.call("Browser.setWindowBounds", {"windowId": wid, "bounds": {"left": 300, "top": 200}})
    time.sleep(0.5)
    geo = bi.window_geometry(page)
    x, y = 200 + 100, 150 + 30          # iframe-local (100, 30)
    for t in ("mouseMoved", "mousePressed", "mouseReleased"):
        page.call("Input.dispatchMouseEvent", {"type": t, "x": x, "y": y, "button": "left", "clickCount": 1})
    time.sleep(1.0)
    ev = [e for e in _events(page) if e["type"] == "click"]
    assert ev, "the iframe never reported the click"
    e = ev[-1]
    print("measured:", json.dumps({"window": [geo.left, geo.top], "inset_y": geo.inset_y, **e}))
    assert e["isTrusted"] is True and (e["clientX"], e["clientY"]) == (100, 30)
    assert (e["screenX"], e["screenY"]) != (e["clientX"], e["clientY"]), "Chromium 40280325 is back"
    assert (e["screenX"], e["screenY"]) == (round(geo.left + x), round(geo.top + geo.inset_y + y))


@live
def test_live_ladder_passes_a_local_fake_turnstile(chrome, conn):
    _open(conn, f"http://127.0.0.1:{chrome['top']}/login",
          settle=lambda p: _frame_ready(conn))
    found = bl.find_challenge(conn, ["127.0.0.1"])
    assert found is not None
    page, ch = found
    assert ch.kind == bc.TURNSTILE and not ch.solved and ch.sitekey == "1x00000000000000000000AA"
    fast = {**bl.BUDGETS, bc.TURNSTILE: bl.Budget(wait_s=1.0, attempts=3, poll_s=4.0)}
    out = bl.Ladder({}, rng=random.Random(11), budgets=fast).run(bl.CdpSurface(conn, page), ch, "live")
    assert out.outcome == bl.PASSED, out.to_dict()
    assert out.delivery == "cdp" and out.attempts == 1
    clicks = [e for e in _events(page) if e["type"] == "click"]
    assert clicks and all(e["isTrusted"] for e in clicks)


@live
def test_live_ladder_press_and_hold_until_the_page_lets_go(chrome, conn):
    _open(conn, f"http://127.0.0.1:{chrome['top']}/px")
    found = bl.find_challenge(conn, ["127.0.0.1"])
    assert found is not None and found[1].kind == bc.PERIMETERX
    page, ch = found
    t0 = time.monotonic()
    out = bl.Ladder({}, rng=random.Random(5)).run(bl.CdpSurface(conn, page), ch, "live-px")
    assert out.outcome == bl.PASSED, out.to_dict()
    assert out.attempts == 1 and time.monotonic() - t0 < 15
