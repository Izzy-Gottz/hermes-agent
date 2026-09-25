"""Tier C of the CAPTCHA ladder: the recognition providers (CapSolver, NopeCHA) against their documented
shapes, the key handling, and the grid driver end to end -- on a scripted surface, and live in Chrome for
Testing on a LOCAL fake reCAPTCHA bframe (www.google.com mapped to 127.0.0.1; nothing reaches Google).

The documented shapes, as read 2026-09-25 and quoted here so a drift is a red test, not a silent miss:

CapSolver, docs.capsolver.com/en/guide/recognition/ReCaptchaClassification/::

    POST https://api.capsolver.com/createTask
    { "clientKey": "YOUR_API_KEY", "task": { "type": "ReCaptchaV2Classification",
      "websiteURL": "https://xxxx.com", "image": "/9j/4AAQ...", "question": "/m/0k4j" } }
    -> { "errorId": 0, "status": "ready", "solution": { "type": "multi", "objects": [0, 1, 2, 3],
         "size": 4 /* 3 or 4 */ }, "taskId": "cbb1c730-..." }
    -> { "errorId": 0, "status": "ready", "solution": { "type": "single", "hasObject": true, "size": 1 } }
    errors (docs.capsolver.com/en/guide/api-error/): { "errorId": 1, "errorCode": "...", "errorDescription": "..." }
    ("websiteURL" and "websiteKey" are optional; Moe never sends them.)

NopeCHA, nopecha.com/api-reference::

    POST https://api.nopecha.com/v1/recognition/recaptcha   Authorization: Basic $YOUR_API_KEY
    { "task": "Select all images with traffic lights", "grid": "3x3", "image_data": ["data:image/jpeg;base64,..."] }
    -> { "data": "2xuttwekei7..." }           GET .../recaptcha?id=JOB_ID
    -> { "data": [true, false, true, false, true, false, false, false, true] }
    POST .../hcaptcha  { "data": { "request_type": "image_label_binary", "requester_question": { "en": ... },
                                   "tasklist": [ { "datapoint_uri": "data:image/jpeg;base64,...", "task_key": ... } ] } }
    -> BinaryResultSet: [[true, false, ...]]
    POST .../textcaptcha { "image_data": ["data:image/png;base64,..."] }  -> { "data": ["TEXT"] }
    Incomplete Job: HTTP 409, app code 14, "Recommended wait time before retrying the request is 500ms"
    Invalid API Key: HTTP 401, app code 15, { "message": "Invalid key", "code": 15 }
"""

from __future__ import annotations

import base64
import http.server
import json
import logging
import os
import random
import signal
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from pathlib import Path
from typing import List, Optional

import pytest

from tools import browser_captcha as bc
from tools import browser_captcha_grid as bg
from tools import browser_captcha_input as bi
from tools import browser_captcha_ladder as bl
from tools import browser_captcha_solver as bs
from tools import browser_captcha_solvers as prov
from tools.browser_captcha import Element, Frame, PageFacts

KEY = "CAP-7f3e9a1b2c4d5e6f-SECRET"   # a fake key: if this string shows up anywhere, the test fails
PNG = b"\x89PNG\r\n\x1a\nFAKE"


@pytest.fixture(autouse=True)
def _fresh():
    bl.reset_failures()
    yield
    bl.reset_failures()


class Transport:
    """A scripted provider: records each exchange and answers from ``replies`` in order."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: List[tuple] = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def capsolver(*replies, **kw):
    t = Transport(*replies)
    return prov.CapSolver(KEY, transport=t, **kw), t


def nopecha(*replies):
    t = Transport(*replies)
    clock = {"t": 0.0}
    s = prov.NopeCHA(KEY, transport=t, timeout_s=5, sleep=lambda d: clock.__setitem__("t", clock["t"] + d),
                     clock=lambda: clock["t"])
    return s, t


def req(kind=bc.RECAPTCHA_V2, instruction="Select all images with buses", grid=(3, 3), tiles=()):
    return bs.SolveRequest(kind=kind, image_png=PNG, instruction=instruction, grid=grid, tiles_png=tiles)


# ---- CapSolver ----------------------------------------------------------------------------------------------

def test_capsolver_request_is_the_documented_shape_and_carries_nothing_about_the_page():
    s, t = capsolver((200, {"errorId": 0, "status": "ready", "solution": {"type": "multi", "objects": [0, 1, 2, 3],
                                                                          "size": 3}, "taskId": "cbb1c730"}))
    assert s.solve(req()).tiles == (0, 1, 2, 3)
    method, url, headers, body = t.calls[0]
    assert (method, url) == ("POST", "https://api.capsolver.com/createTask")
    assert body == {"clientKey": KEY, "task": {"type": "ReCaptchaV2Classification",
                                                "image": base64.b64encode(PNG).decode(), "question": "/m/01bjv"}}
    assert "websiteURL" not in body["task"] and "websiteKey" not in body["task"]


def test_capsolver_single_tile_answer():
    s, _ = capsolver((200, {"errorId": 0, "status": "ready", "solution": {"type": "single", "hasObject": True, "size": 1}}),
                     (200, {"errorId": 0, "status": "ready", "solution": {"type": "single", "hasObject": False, "size": 1}}))
    assert s.solve(req(grid=(1, 1))).tiles == (0,)
    assert s.solve(req(grid=(1, 1))).tiles == ()


@pytest.mark.parametrize("instruction,qid", [
    ("Select all images with buses", "/m/01bjv"), ("Select all images with a school bus", "/m/02yvhj"),
    ("Select all squares with traffic lights If there are none, click skip", "/m/015qff"),
    ("Select all images with crosswalks", "/m/014xcs"), ("Select all images with fire hydrants", "/m/01pns0"),
    ("Select all images with cars", "/m/0k4j"), ("Select all images with mountains or hills", "/m/09d_r"),
    ("Select all images with stairs", "/m/01lynh"), ("Select all images with taxis", "/m/0pg52"),
])
def test_capsolver_question_ids_are_its_documented_list(instruction, qid):
    assert prov.capsolver_question(instruction) == qid


def test_capsolver_unknown_object_is_unsupported_without_spending_a_request():
    s, t = capsolver()
    with pytest.raises(bs.SolverUnsupported):
        s.solve(req(instruction="Select all images with ducks"))
    assert t.calls == []
    assert not s.supports(bc.HCAPTCHA) and not s.supports(bc.TURNSTILE)


def test_capsolver_errors_are_reported_without_the_key():
    s, _ = capsolver((400, {"errorId": 1, "errorCode": "ERROR_KEY_DENIED_ACCESS",
                            "errorDescription": f"Wrong account key {KEY}"}),
                     (200, {"errorId": 1, "errorCode": "ERROR_UNKNOWN_QUESTION", "errorDescription": "Wrong Question ID"}),
                     (200, {"errorId": 0, "status": "ready", "solution": {"type": "multi", "objects": [1], "size": 4}}))
    with pytest.raises(bs.SolverError) as e:
        s.solve(req())
    assert "ERROR_KEY_DENIED_ACCESS" in str(e.value) and KEY not in str(e.value)
    with pytest.raises(bs.SolverUnsupported):
        s.solve(req())
    with pytest.raises(bs.SolverError, match="4x4"):
        s.solve(req())                   # a 4x4 answer for a 3x3 grid is never clicked


# ---- NopeCHA --------------------------------------------------------------------------------------------------

def test_nopecha_recaptcha_submits_then_polls_through_incomplete_jobs():
    s, t = nopecha((200, {"data": "job-1"}), (409, {"message": "Incomplete job", "code": 14}),
                   (200, {"data": [True, False, True, False, True, False, False, False, True]}))
    assert s.solve(req(instruction="Select all images with traffic lights")).tiles == (0, 2, 4, 8)
    (m1, u1, h1, b1), (m2, u2, h2, _), (m3, u3, h3, b3) = t.calls
    assert (m1, u1) == ("POST", "https://api.nopecha.com/v1/recognition/recaptcha")
    assert b1 == {"task": "Select all images with traffic lights", "grid": "3x3",
                  "image_data": ["data:image/png;base64," + base64.b64encode(PNG).decode()]}
    assert (m2, u2, m3, u3) == ("GET", "https://api.nopecha.com/v1/recognition/recaptcha?id=job-1",
                                "GET", "https://api.nopecha.com/v1/recognition/recaptcha?id=job-1")
    assert h1 == h2 == h3 == {"Authorization": f"Basic {KEY}"}
    assert all(KEY not in c[1] for c in t.calls) and b3 is None      # the key is never in a URL


def test_nopecha_hcaptcha_sends_one_image_per_tile_as_image_label_binary():
    tiles = tuple(b"tile%d" % i for i in range(9))
    s, t = nopecha((200, {"data": "job-2"}), (200, {"data": [[False, True, False, False, False, True, False, False, False]]}))
    ans = s.solve(req(kind=bc.HCAPTCHA, instruction="Please click each image containing a bicycle", tiles=tiles))
    assert ans.tiles == (1, 5)
    body = t.calls[0][3]
    assert t.calls[0][1] == "https://api.nopecha.com/v1/recognition/hcaptcha"
    assert body["data"]["request_type"] == "image_label_binary"
    assert body["data"]["requester_question"] == {"en": "Please click each image containing a bicycle"}
    assert [x["datapoint_uri"] for x in body["data"]["tasklist"]] == [
        "data:image/png;base64," + base64.b64encode(p).decode() for p in tiles]
    assert "key" not in body


def test_nopecha_text_captcha():
    s, t = nopecha((200, {"data": "job-3"}), (200, {"data": ["x7Kq2"]}))
    assert s.solve(req(kind=bc.IMAGE_TEXT, grid=(1, 1))).text == "x7Kq2"
    assert t.calls[0][3] == {"image_data": ["data:image/png;base64," + base64.b64encode(PNG).decode()]}


def test_nopecha_invalid_key_and_timeouts_never_echo_the_key():
    s, _ = nopecha((401, {"message": f"Invalid key {KEY}", "code": 15}))
    with pytest.raises(bs.SolverError) as e:
        s.solve(req())
    assert "code 15" in str(e.value) and KEY not in str(e.value)
    s, t = nopecha((200, {"data": "job"}), *[(409, {"message": "Incomplete job", "code": 14})] * 20)
    with pytest.raises(bs.SolverError, match="no answer"):
        s.solve(req())
    assert len(t.calls) <= 12            # bounded: 5 s at the documented 500 ms


def test_the_real_transport_sends_the_key_in_the_header_and_reads_error_bodies(monkeypatch):
    seen = {}

    class Resp:
        status = 200

        def read(self):
            return b'{"data": "job"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def urlopen(request, timeout):
        seen.update(url=request.full_url, headers=dict(request.header_items()), body=json.loads(request.data))
        return Resp()

    monkeypatch.setattr(prov.urllib.request, "urlopen", urlopen)
    status, data = prov._urllib_transport("POST", prov.NOPECHA_URL + "recaptcha", {"Authorization": f"Basic {KEY}"},
                                          {"task": "x"}, 5)
    assert (status, data) == (200, {"data": "job"}) and KEY not in seen["url"]
    assert seen["headers"]["Authorization"] == f"Basic {KEY}"

    import io
    import urllib.error

    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"message": "Out of credit", "code": 16}'))

    monkeypatch.setattr(prov.urllib.request, "urlopen", refuse)
    assert prov._urllib_transport("GET", prov.NOPECHA_URL + "recaptcha?id=j", {}, None, 5) == (
        403, {"message": "Out of credit", "code": 16})


# ---- config, keys, and what is never routed -----------------------------------------------------------------

def test_config_string_form_and_none(monkeypatch):
    monkeypatch.setenv("NOPECHA_API_KEY", KEY)
    monkeypatch.setattr(prov, "provider_key", lambda p: os.environ.get(prov.KEY_ENV[p], ""))
    solver, why = bs.solver_for(bc.RECAPTCHA_V2, {"captcha": {"solver": "nopecha"}})
    assert isinstance(solver, prov.NopeCHA) and why == "nopecha"
    solver, why = bs.solver_for(bc.RECAPTCHA_V2, {"captcha": {"solver": {"provider": "nopecha"}}})
    assert isinstance(solver, prov.NopeCHA)
    for off in ("none", "", None):
        assert bs.solver_for(bc.RECAPTCHA_V2, {"captcha": {"solver": off}})[0] is None
    assert bs.solver_for(bc.RECAPTCHA_V2, {})[0] is None
    assert bs.solver_for(bc.HCAPTCHA, {"captcha": {"solver": "capsolver"}})[0] is None   # not a CapSolver kind


def test_no_key_skips_the_tier_silently(monkeypatch, caplog):
    monkeypatch.setattr(prov, "provider_key", lambda p: "")
    caplog.set_level(logging.DEBUG)
    for provider in ("capsolver", "nopecha"):
        solver, why = bs.solver_for(bc.RECAPTCHA_V2, {"captcha": {"solver": provider}})
        assert solver is None and prov.KEY_ENV[provider] in why
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_provider_key_reads_the_hermes_env_like_every_other_key(monkeypatch):
    import hermes_cli.config as hc
    monkeypatch.setattr(hc, "get_env_value", lambda name: {"CAPSOLVER_API_KEY": " k1 "}.get(name))
    assert prov.provider_key("capsolver") == "k1" and prov.provider_key("nopecha") == ""
    from hermes_cli.config_defaults import OPTIONAL_ENV_VARS
    for name in ("CAPSOLVER_API_KEY", "NOPECHA_API_KEY"):
        assert OPTIONAL_ENV_VARS[name]["password"] is True


def test_cloudflare_is_never_routed_to_a_solver_even_with_one_configured(monkeypatch):
    built = []
    monkeypatch.setattr(prov, "provider_key", lambda p: KEY)
    bs.register_solver("spy", lambda cfg: built.append(cfg) or FakeSolver())
    try:
        for kind in (bc.TURNSTILE, bc.CF_MANAGED):
            for cfg in ({"captcha": {"solver": "spy"}}, {"captcha": {"solver": "nopecha"}},
                        {"captcha": {"solver": "capsolver"}}):
                solver, why = bs.solver_for(kind, cfg)
                assert solver is None and "third party" in why
        assert built == []                       # the provider was never even constructed
    finally:
        bs.unregister_solver("spy")


def test_the_key_never_reaches_an_error_a_log_a_repr_or_the_outcome(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    boom = OSError(f"connect failed for https://api.nopecha.com/?key={KEY} Authorization: Basic {KEY}")
    s, _ = nopecha(boom)
    with pytest.raises(bs.SolverError) as e:
        s.solve(req())
    assert KEY not in str(e.value) and KEY not in repr(s)
    s2, _ = capsolver(OSError(f'body {{"clientKey": "{KEY}"}}'))
    with pytest.raises(bs.SolverError) as e2:
        s2.solve(req())
    assert KEY not in str(e2.value)
    # and through the ladder, with the provider failing mid-round
    monkeypatch.setattr(prov, "provider_key", lambda p: KEY)
    t = Transport((200, {"data": "job"}), OSError(f"reset by peer; Authorization: Basic {KEY}"))
    monkeypatch.setattr(prov, "_urllib_transport", t)
    out, page = run_grid(GridPage([Chal({1, 4, 7})]), cfg={"captcha": {"solver": "nopecha"}}, solver=None)
    assert out.outcome == bl.NEEDS_PERSON and out.solver == "nopecha"
    blob = json.dumps(out.to_dict()) + caplog.text + "".join(out.steps)
    assert KEY not in blob and "failed" in blob


# ---- the grid driver on a scripted page ------------------------------------------------------------------------

RC_ANCHOR = "https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LcTEST&co=aHR0cHM6&hl=en&v=abc&size=normal&cb=x"
RC_BFRAME = "https://www.google.com/recaptcha/api2/bframe?hl=en&v=abc&k=6LcTEST"


class Chal:
    """One image challenge: ``red`` tiles hold the object; for a dynamic one, ``repl[i]`` is what the
    replacement images of tile i hold, in order (then nothing)."""

    def __init__(self, red, dynamic=False, repl=None, instruction="Select all images with buses"):
        self.red, self.dynamic, self.repl = set(red), dynamic, {k: list(v) for k, v in (repl or {}).items()}
        self.instruction = instruction + (" Click verify once there are none left." if dynamic else "")


class GridPage:
    """A reCAPTCHA page: anchor + bframe; ``challenges`` are served in order, each passed by a correct
    verify. A wrong verify re-serves the same challenge with new images, like the real one."""
    ORIGIN = (60.0, 120.0)

    def __init__(self, challenges, checkbox_first=False):
        self.challenges, self.i = challenges, 0
        self.passed = False
        self.showing_image = not checkbox_first
        self.gen = 0
        self.verifies = self.reloads = 0
        self.clicks: List[str] = []
        self._load()

    def _load(self):
        c = self.challenges[self.i]
        self.cur = [i in c.red for i in range(9)]
        self.repl = {k: list(v) for k, v in c.repl.items()}
        self.sel = set()
        self.gen += 1
        self.srcs = [f"img-{self.gen}-{i}" for i in range(9)]

    def facts(self) -> PageFacts:
        frames = [Frame(RC_ANCHOR, backend_node_id=11, visible=True),
                  Frame(RC_BFRAME, backend_node_id=12, visible=self.showing_image and not self.passed)]
        return PageFacts(url="https://shop.example/signup", title="Sign up", frames=frames,
                         elements=[Element("div", classes=("g-recaptcha",), attrs={"data-sitekey": "6LcTEST"})],
                         tokens={"g-recaptcha-response": "03AFcWeA" if self.passed else ""})

    def tile(self, i):
        ox, oy = self.ORIGIN
        return bi.Box(ox + (i % 3) * 100, oy + 60 + (i // 3) * 100, 98, 98)

    VERIFY = bi.Box(300, 490, 80, 30)
    RELOAD = bi.Box(70, 490, 30, 30)
    CHECKBOX = bi.Box(20, 30, 28, 28)

    def grid(self, ch):
        if not self.showing_image or self.passed:
            return None
        c = self.challenges[self.i]
        return bg.Grid(vendor="recaptcha", instruction=c.instruction, target="buses", rows=3, cols=3,
                       tiles=[self.tile(i) for i in range(9)], selected=[i in self.sel for i in range(9)],
                       srcs=list(self.srcs), area=bi.Box(self.ORIGIN[0], self.ORIGIN[1] + 60, 300, 300),
                       verify=self.VERIFY, reload=self.RELOAD, dynamic=c.dynamic)

    def screenshot(self, box):
        return PNG + json.dumps({"truth": [i for i in range(9) if self.cur[i]]}).encode()

    def click(self, p):
        if self.CHECKBOX.contains(p):
            self.clicks.append("checkbox")
            self.showing_image = True
            return
        if self.VERIFY.contains(p):
            self.clicks.append("verify")
            self.verifies += 1
            c = self.challenges[self.i]
            ok = not any(self.cur) if c.dynamic else self.sel == c.red
            if ok:
                self.i += 1
                if self.i == len(self.challenges):
                    self.passed = True
                    return
            self._load()
            return
        if self.RELOAD.contains(p):
            self.clicks.append("reload")
            self.reloads += 1
            self._load()
            return
        for i in range(9):
            if self.tile(i).contains(p):
                self.clicks.append(f"tile{i}")
                if self.challenges[self.i].dynamic:
                    nxt = self.repl.get(i) or []
                    self.cur[i] = nxt.pop(0) if nxt else False
                    self.gen += 1
                    self.srcs[i] = f"img-{self.gen}-{i}"
                else:
                    self.sel ^= {i}
                return
        self.clicks.append("miss")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class PageDelivery(bi.Delivery):
    name = "fake"

    def __init__(self, page):
        self.page = page

    def click(self, box, rng):
        p = bi.point_inside(box, rng)
        self.page.click(p)
        return p


class GridSurface(bl.Surface):
    def __init__(self, page):
        self.page = page
        self.d = PageDelivery(page)

    def facts(self):
        return self.page.facts()

    def locate(self, ch):
        return self.page.CHECKBOX

    def delivery(self):
        return self.d, "fake"

    def grid(self, ch):
        return self.page.grid(ch)

    def screenshot(self, box):
        return self.page.screenshot(box)


class FakeSolver:
    """Reads the fake screenshot's truth; ``wrong=True`` always answers tile 0 only."""
    name = "fake"

    def __init__(self, wrong=False, unsupported=0):
        self.wrong, self.unsupported = wrong, unsupported
        self.requests: List[bs.SolveRequest] = []

    def supports(self, kind):
        return kind in (bc.RECAPTCHA_V2, bc.HCAPTCHA)

    def solve(self, request):
        self.requests.append(request)
        if self.unsupported:
            self.unsupported -= 1
            raise bs.SolverUnsupported("no class for this object")
        truth = json.loads(request.image_png[len(PNG):])["truth"]
        return bs.SolveAnswer(tiles=(0,) if self.wrong else tuple(truth))


def run_grid(page, cfg=None, solver: Optional[FakeSolver] = FakeSolver(), task="t1"):
    cfg = cfg or {"captcha": {"solver": "fake"}}
    if solver is not None:
        bs.register_solver("fake", lambda c: solver)
    try:
        clock = Clock()
        s = GridSurface(page)
        first = bc.classify(s.facts())
        out = bl.Ladder(cfg, sleep=clock.sleep, clock=clock, rng=random.Random(5)).run(s, first, task)
        return out, page
    finally:
        bs.unregister_solver("fake")


def test_a_static_grid_is_solved_clicked_and_verified():
    solver = FakeSolver()
    out, page = run_grid(GridPage([Chal({1, 4, 7})]), solver=solver)
    assert out.outcome == bl.PASSED, out.to_dict()
    d = out.to_dict()
    assert (d["tier"], d["solver"], d["rounds"], d["solves"]) == ("C", "fake", 1, 1)
    assert page.clicks == ["tile1", "tile4", "tile7", "verify"]
    r = solver.requests[0]
    assert (r.kind, r.grid, r.instruction) == (bc.RECAPTCHA_V2, (3, 3), "Select all images with buses")


def test_a_dynamic_grid_asks_again_about_the_replaced_tiles_only_then_a_second_round():
    solver = FakeSolver()
    page = GridPage([Chal({1, 4, 7}, dynamic=True, repl={1: [True, False], 4: [False], 7: [False]}),
                     Chal({0, 2, 8})])
    out, _ = run_grid(page, solver=solver)
    assert out.outcome == bl.PASSED, out.to_dict()
    assert out.rounds == 2 and out.solves == 4           # waves: {1,4,7} -> {1} -> none; then the static round
    assert page.clicks == ["tile1", "tile4", "tile7", "tile1", "verify", "tile0", "tile2", "tile8", "verify"]
    assert solver.requests[0].instruction == "Select all images with buses"   # the how-to tail is not sent


def test_tier_a_escalating_to_images_is_solved_and_reported_as_tier_c():
    out, page = run_grid(GridPage([Chal({3, 5})], checkbox_first=True))
    assert out.outcome == bl.PASSED and out.tier == "C" and out.attempts == 1
    assert page.clicks[0] == "checkbox" and page.clicks[-1] == "verify"


def test_three_rounds_at_most_then_the_person_and_it_counts_as_a_failure():
    out, page = run_grid(GridPage([Chal({1, 4, 7})]), solver=FakeSolver(wrong=True))
    assert out.outcome == bl.NEEDS_PERSON and out.rounds == 3 and page.verifies == 3
    assert "did not pass after 3 rounds" in out.reason
    assert bl.failures("t1", bc.classify(page.facts())) == 1


def test_five_solves_per_task_across_calls():
    out1, _ = run_grid(GridPage([Chal({1})]), solver=FakeSolver(wrong=True))
    assert out1.solves == 3
    out2, page2 = run_grid(GridPage([Chal({1})]), solver=FakeSolver(wrong=True))
    assert out2.outcome == bl.NEEDS_PERSON and out2.solves == 2 and "used up" in out2.reason
    assert bg.solves_used("t1") == 5
    out3, page3 = run_grid(GridPage([Chal({1})]), solver=FakeSolver(), task="t1")
    assert out3.outcome == bl.NEEDS_PERSON and page3.clicks == []          # nothing left to spend, nothing clicked
    out4, _ = run_grid(GridPage([Chal({1})]), solver=FakeSolver(), task="t2")
    assert out4.outcome == bl.PASSED                                      # a new task has its own budget


def test_an_object_the_solver_does_not_know_asks_for_another_challenge():
    solver = FakeSolver(unsupported=1)
    out, page = run_grid(GridPage([Chal({2, 6})]), solver=solver)
    assert out.outcome == bl.PASSED and page.reloads == 1 and out.rounds == 2
    assert page.clicks == ["reload", "tile2", "tile6", "verify"]


def test_no_key_means_no_tier_c_and_nothing_is_clicked(monkeypatch):
    monkeypatch.setattr(prov, "provider_key", lambda p: "")
    out, page = run_grid(GridPage([Chal({1})]), cfg={"captcha": {"solver": "capsolver"}}, solver=None)
    assert out.outcome == bl.NEEDS_PERSON and page.clicks == [] and "CAPSOLVER_API_KEY" in out.reason
    assert out.solves is None


def test_cloudflare_never_reaches_the_grid_driver(monkeypatch):
    called = []
    monkeypatch.setattr(bg, "solve_grid", lambda *a, **k: called.append(a))
    bs.register_solver("fake", lambda c: FakeSolver())
    try:
        clock = Clock()
        page = PageFacts(url="https://shop.example/", title="Just a moment...")
        s = GridSurface(GridPage([Chal({1})]))
        s.facts = lambda: page
        ch = bc.classify(page)
        assert ch.kind == bc.CF_MANAGED
        ch.stage = "image"                       # even if it ever looked like a puzzle
        out = bl.Ladder({"captcha": {"solver": "fake"}}, sleep=clock.sleep, clock=clock).run(s, ch, "t9")
        assert called == [] and out.outcome == bl.NEEDS_PERSON and "third party" in out.reason
    finally:
        bs.unregister_solver("fake")


def test_other_puzzles_stay_with_the_person_even_with_a_solver():
    bs.register_solver("fake", lambda c: type("S", (), {"name": "fake", "supports": lambda self, k: True})())
    try:
        clock = Clock()
        facts = PageFacts(url="https://shop.example/", scripts=["https://static.geetest.com/v4/gcaptcha4.js"])
        s = GridSurface(GridPage([Chal({1})]))
        s.facts = lambda: facts
        out = bl.Ladder({"captcha": {"solver": "fake"}}, sleep=clock.sleep, clock=clock).run(s, bc.classify(facts), "t")
        assert out.outcome == bl.NEEDS_PERSON and "does not solve" in out.reason
    finally:
        bs.unregister_solver("fake")


def test_grid_js_output_is_parsed_and_offset_by_the_frame():
    raw = {"vendor": "recaptcha", "instruction": "Select all images with buses Click verify once there are none left.",
           "target": "buses", "rows": 3, "cols": 3, "tiles": [[i % 3 * 100, 60 + i // 3 * 100, 98, 98] for i in range(9)],
           "selected": [False] * 9, "srcs": ["a"] * 9, "busy": False, "dynamic": True, "area": [0, 60, 300, 300],
           "verify": [230, 380, 70, 30], "reload": None, "error": ""}
    g = bg.parse_grid(raw, 40.0, 100.0)
    assert g.tiles[4] == bi.Box(140, 260, 98, 98) and g.verify == bi.Box(270, 480, 70, 30) and g.reload is None
    assert bg.instruction_line(g.instruction) == "Select all images with buses"
    assert bg.parse_grid({**raw, "rows": 4}, 0, 0) is None               # 9 tiles are not a 4x3 grid
    assert bg.parse_grid({"none": True}, 0, 0) is None


# ---- live: Chrome for Testing, a local fake bframe --------------------------------------------------------------

def _cft() -> Optional[str]:
    root = Path.home() / ".agent-browser" / "browsers"
    for app in sorted(root.glob("chrome-*/Google Chrome for Testing.app"), reverse=True):
        exe = app / "Contents" / "MacOS" / "Google Chrome for Testing"
        if exe.exists():
            return str(exe)
    return None


CFT = _cft()
live = pytest.mark.skipif(CFT is None, reason="Chrome for Testing not installed under ~/.agent-browser/browsers")

RED, GREEN = "%23dd0000", "%2300aa00"

# The bframe, in reCAPTCHA's own DOM vocabulary (rc-imageselect-*). Round 1 is dynamic: clicked tiles fade
# and are replaced after 400 ms; round 2 is a static 3x3. A tile only counts on a trusted click.
_FAKE_BFRAME = """<!doctype html><html><body style="margin:0;font:14px sans-serif;width:300px">
<div class="rc-imageselect-desc-wrapper" style="height:60px"><div id="desc" class="rc-imageselect-desc-no-canonical"></div></div>
<table id="t" class="rc-imageselect-table-33" style="border-collapse:collapse;border-spacing:0"><tbody></tbody></table>
<div class="rc-imageselect-incorrect-response" id="err" style="display:none">Please try again.</div>
<div style="height:40px;position:relative"><button id="recaptcha-reload-button" style="position:absolute;left:4px;top:6px;width:30px;height:28px">R</button>
<button id="recaptcha-verify-button" style="position:absolute;right:4px;top:6px;width:80px;height:28px">Verify</button></div>
<script>
const ROUNDS = [{dynamic: true, red: [1, 4, 7], repl: {1: [true, false], 4: [false], 7: [false]}},
                {dynamic: false, red: [0, 2, 8]}];
let r = 0, cur, repl, sel, gen = 0;
window.__log = [];
const src = red => "data:image/svg+xml," + encodeURIComponent('') + "<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100'>"
  + "<rect width='100' height='100' fill='" + (red ? '%RED%' : '%GREEN%') + "'/><!--" + (++gen) + "--></svg>";
function load() {
  const R = ROUNDS[r];
  cur = [...Array(9).keys()].map(i => R.red.includes(i));
  repl = JSON.parse(JSON.stringify(R.repl || {}));
  sel = new Set();
  document.getElementById('desc').innerHTML = 'Select all images with <strong>buses</strong>'
    + (R.dynamic ? ' <span>Click verify once there are none left.</span>' : '');
  const tb = document.querySelector('#t tbody'); tb.innerHTML = '';
  for (let y = 0; y < 3; y++) { const tr = tb.insertRow();
    for (let x = 0; x < 3; x++) { const i = y * 3 + x, td = tr.insertCell();
      td.className = 'rc-imageselect-tile'; td.style.padding = '0'; td.style.width = td.style.height = '100px';
      td.innerHTML = '<img class="rc-image-tile-33" width="100" height="100" style="display:block" src="' + src(cur[i]) + '">';
      td.addEventListener('click', e => tile(e, i, td)); } }
}
function tile(e, i, td) {
  if (!e.isTrusted) return; window.__log.push('tile' + i);
  if (ROUNDS[r].dynamic) {
    td.classList.add('rc-imageselect-dynamic-selected');
    setTimeout(() => { const n = (repl[i] || []).shift(); cur[i] = !!n;
      td.querySelector('img').className = 'rc-image-tile-11'; td.querySelector('img').src = src(cur[i]);
      td.classList.remove('rc-imageselect-dynamic-selected'); }, 400);
  } else { sel.has(i) ? sel.delete(i) : sel.add(i); td.classList.toggle('rc-imageselect-tileselected'); }
}
document.getElementById('recaptcha-verify-button').addEventListener('click', e => {
  if (!e.isTrusted) return; window.__log.push('verify');
  const R = ROUNDS[r];
  const ok = R.dynamic ? !cur.some(x => x) : (sel.size === R.red.length && R.red.every(i => sel.has(i)));
  document.getElementById('err').style.display = ok ? 'none' : 'block';
  if (ok) { r++; if (r === ROUNDS.length) { parent.postMessage({token: '03AFcWeA-FAKE-TOKEN'}, '*'); return; } }
  load();
});
document.getElementById('recaptcha-reload-button').addEventListener('click', e => { if (e.isTrusted) load(); });
load();
</script></body></html>""".replace("%RED%", RED).replace("%GREEN%", GREEN)


def _serve(pages, tls=None):
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
    # google.com is HSTS-preloaded: the fake frames are served over TLS with a throwaway certificate.
    certdir = tmp_path_factory.mktemp("tls")
    cert, key = str(certdir / "c.pem"), str(certdir / "k.pem")
    try:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj",
                        "/CN=www.google.com", "-keyout", key, "-out", cert], check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("openssl is needed to serve the fake bframe over https")
    fsrv = _serve({"/recaptcha/api2/bframe": _FAKE_BFRAME,
                   "*": "<body style='margin:0'><div id='recaptcha-anchor' style='margin:23px 13px;width:28px;height:28px'></div></body>"},
                  tls=(cert, key))
    fp = fsrv.server_address[1]
    top = f"""<html><head><title>Sign up</title></head><body style="margin:0">
<div class="g-recaptcha" data-sitekey="6LcFAKE" style="position:absolute;left:20px;top:10px">
<iframe src="https://www.google.com:{fp}/recaptcha/api2/anchor?ar=1&k=6LcFAKE&hl=en&size=normal" style="width:304px;height:78px;border:0"></iframe>
<textarea name="g-recaptcha-response" style="display:none"></textarea></div>
<iframe id="bf" src="https://www.google.com:{fp}/recaptcha/api2/bframe?hl=en&k=6LcFAKE"
 style="position:absolute;left:40px;top:100px;width:300px;height:420px;border:0"></iframe>
<script>window.addEventListener('message', m => {{ if (m.data && m.data.token) {{
  document.querySelector('[name=g-recaptcha-response]').value = m.data.token;
  document.getElementById('bf').style.top = '-9999px'; }} }});</script></body></html>"""
    tsrv = _serve({"/signup": top})
    ud = tempfile.mkdtemp(prefix="captcha-grid-live-")
    rules = "MAP www.google.com 127.0.0.1, MAP * ~NOTFOUND, EXCLUDE 127.0.0.1, EXCLUDE localhost"
    proc = subprocess.Popen([CFT, "--headless=new", f"--user-data-dir={ud}", "--remote-debugging-port=0",
                             "--no-first-run", "--no-default-browser-check", f"--host-resolver-rules={rules}",
                             "--ignore-certificate-errors", "--no-proxy-server", "--window-size=900,700", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    port_file = Path(ud) / "DevToolsActivePort"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not (port_file.exists() and port_file.read_text().strip()):
        time.sleep(0.3)
    if not (port_file.exists() and port_file.read_text().strip()):
        os.killpg(proc.pid, signal.SIGKILL)
        pytest.skip("Chrome for Testing did not start in 90 s (a loaded Mac is not a result)")
    port, path = port_file.read_text().split("\n")[:2]
    yield {"ws": f"ws://127.0.0.1:{port}{path}", "top": tsrv.server_address[1]}
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    import shutil
    shutil.rmtree(ud, ignore_errors=True)
    fsrv.shutdown()
    tsrv.shutdown()


def _png_pixels(png: bytes):
    """(width, height, rows of RGB tuples) from an 8-bit RGB/RGBA PNG -- enough to read Chrome's screenshots."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h, bpp = 8, b"", 0, 0, 3
    while pos < len(png):
        n, typ = struct.unpack(">I4s", png[pos:pos + 8])
        data = png[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
            assert depth == 8 and ctype in (2, 6)
            bpp = 4 if ctype == 6 else 3
        elif typ == b"IDAT":
            idat += data
        pos += 12 + n
    raw, stride, rows, prev = zlib.decompress(idat), w * bpp, [], bytearray(w * bpp)
    for y in range(h):
        f, line = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b, c = prev[x], prev[x - bpp] if x >= bpp else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append([tuple(line[i:i + 3]) for i in range(0, stride, bpp)])
        prev = line
    return w, h, rows


class PixelSolver:
    """Looks at the real screenshot: a tile is a bus when its centre pixel is red."""
    name = "pixels"

    def __init__(self):
        self.requests = []

    def supports(self, kind):
        return kind == bc.RECAPTCHA_V2

    def solve(self, request):
        self.requests.append(request)
        w, h, px = _png_pixels(request.image_png)
        rows, cols = request.grid
        red = []
        for i in range(rows * cols):
            r, g, b = px[int((i // cols + 0.5) * h / rows)][int((i % cols + 0.5) * w / cols)]
            if r > 180 and g < 60:
                red.append(i)
        return bs.SolveAnswer(tiles=tuple(red))


@live
def test_live_the_ladder_solves_a_local_fake_bframe_end_to_end(chrome):
    with bl.CdpConn(chrome["ws"], timeout=30) as conn:
        tid = conn.call("Target.createTarget", {"url": f"http://127.0.0.1:{chrome['top']}/signup"})["targetId"]
        deadline = time.monotonic() + 45
        found = None
        while time.monotonic() < deadline and found is None:
            time.sleep(0.5)
            f = bl.find_challenge(conn, ["127.0.0.1"])
            if f and f[1].stage == "image":
                page, ch = f
                s = bl.CdpSurface(conn, page)
                if s.grid(ch) is not None:
                    found = f
        if found is None:
            pytest.skip("the fake bframe did not load in 45 s (a loaded Mac is not a result)")
        page, ch = found
        assert ch.kind == bc.RECAPTCHA_V2 and ch.stage == "image"
        surface = bl.CdpSurface(conn, page)
        g = surface.grid(ch)
        assert (g.rows, g.cols, g.dynamic) == (3, 3, True) and g.target == "buses"
        assert g.tiles[0] == bi.Box(40, 160, 100, 100)          # frame at (40,100) + 60 px header
        solver = PixelSolver()
        bs.register_solver("pixels", lambda c: solver)
        try:
            out = bl.Ladder({"captcha": {"solver": "pixels"}}, rng=random.Random(3)).run(surface, ch, "live")
        finally:
            bs.unregister_solver("pixels")
        assert out.outcome == bl.PASSED, out.to_dict()
        assert (out.tier, out.solver, out.rounds, out.delivery) == ("C", "pixels", 2, "cdp")
        assert out.solves == 4
        # the screenshot was the grid itself: 300x300, and the first answer was the three red tiles
        w, h, _ = _png_pixels(solver.requests[0].image_png)
        assert (w, h) == (300, 300)
        assert solver.requests[0].instruction == "Select all images with buses"
        conn.call("Target.closeTarget", {"targetId": tid})


# ---- live provider (only with a key in the environment; costs one request) ----------------------------------

def _grid_png(red=(1, 4, 7)) -> bytes:
    rows = []
    for y in range(300):
        line = b"\x00"
        for x in range(300):
            line += b"\xdd\x00\x00" if (y // 100) * 3 + x // 100 in red else b"\x00\xaa\x00"
        rows.append(line)
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 300, 300, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


@pytest.mark.skipif(not os.environ.get("CAPSOLVER_API_KEY"), reason="no CAPSOLVER_API_KEY in the environment")
def test_live_capsolver_answers_in_its_documented_shape():
    s = prov.CapSolver(os.environ["CAPSOLVER_API_KEY"])
    ans = s.solve(bs.SolveRequest(kind=bc.RECAPTCHA_V2, image_png=_grid_png(), instruction="Select all images with buses"))
    assert all(0 <= t < 9 for t in ans.tiles)


@pytest.mark.skipif(not os.environ.get("NOPECHA_API_KEY"), reason="no NOPECHA_API_KEY in the environment")
def test_live_nopecha_answers_in_its_documented_shape():
    s = prov.NopeCHA(os.environ["NOPECHA_API_KEY"])
    ans = s.solve(bs.SolveRequest(kind=bc.RECAPTCHA_V2, image_png=_grid_png(), instruction="Select all images with buses"))
    assert all(0 <= t < 9 for t in ans.tiles)
