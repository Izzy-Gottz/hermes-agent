"""Tier C providers for the CAPTCHA ladder: recognition only, bring your own key.

Each provider answers "which tiles of this picture match the instruction" and nothing else. The ladder
(tools/browser_captcha_grid.py) screenshots the challenge in Moe's own browser, sends the pixels and the
instruction here, and clicks the answer itself. So, by construction:

* no token is minted, no page URL / sitekey / cookie leaves the Mac, and the owner's IP is never proxied;
* nothing here is ever asked about a Cloudflare challenge (tools/browser_captcha_solver.py refuses first);
* the key goes only into the one request that needs it, and is scrubbed from every message this module
  raises or logs (:func:`_scrub`).

Shapes are the providers' own documentation, read 2026-09-25, and quoted in
tests/tools/test_browser_captcha_solver.py:

* CapSolver ``ReCaptchaV2Classification`` (docs.capsolver.com/en/guide/recognition/ReCaptchaClassification/):
  ``POST https://api.capsolver.com/createTask`` with ``{"clientKey", "task": {"type":
  "ReCaptchaV2Classification", "image": <base64>, "question": "/m/0k4j"}}`` -- answered synchronously:
  ``{"errorId": 0, "status": "ready", "solution": {"type": "multi", "objects": [0, 1, 2, 3], "size": 4}}``
  (size 3 or 4, the grid's side) or ``{"type": "single", "hasObject": true, "size": 1}`` for one tile.
  Errors: ``{"errorId": 1, "errorCode", "errorDescription"}``. ``websiteURL`` / ``websiteKey`` are
  optional and deliberately never sent. reCAPTCHA v2 only.
* NopeCHA Recognition API (nopecha.com/api-reference): ``POST https://api.nopecha.com/v1/recognition/
  {recaptcha|hcaptcha|textcaptcha}`` with ``Authorization: Basic <key>`` returns ``{"data": <job id>}``;
  ``GET ...?id=<job id>`` returns the answer, or HTTP 409 app code 14 "Incomplete job" (retry after
  500 ms). reCAPTCHA: ``{"task", "grid": "3x3"|"4x4"|"1x1", "image_data": [data URI]}`` -> ``{"data":
  [true, false, ...]}``; hCaptcha: ``{"data": {"request_type": "image_label_binary", "requester_question":
  {"en": ...}, "tasklist": [{"datapoint_uri", "task_key"}]}}`` -> ``{"data": [[true, false, ...]]}``;
  text: ``{"image_data": [data URI]}`` -> ``{"data": ["TEXT"]}``. The key goes in the header, never the
  ``key`` query parameter, so it cannot end up in a URL.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

from tools import browser_captcha as bc
from tools.browser_captcha_solver import (SolveAnswer, SolveRequest, SolverError, SolverUnsupported,
                                          register_solver)

logger = logging.getLogger(__name__)

CAPSOLVER_URL = "https://api.capsolver.com/createTask"
NOPECHA_URL = "https://api.nopecha.com/v1/recognition/"
KEY_ENV = {"capsolver": "CAPSOLVER_API_KEY", "nopecha": "NOPECHA_API_KEY"}

#: ``(status, body)``: one HTTP exchange. Injected in tests; :func:`_urllib_transport` is the real one.
Transport = Callable[[str, str, Dict[str, str], Optional[dict], float], Tuple[int, Any]]


class MissingKey(SolverError):
    pass


def provider_key(provider: str) -> str:
    """The provider's key from the environment or ``~/.hermes/.env`` (scope-aware, like every other
    provider key in Hermes), or ''."""
    name = KEY_ENV[provider]
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value(name)
    except Exception:
        val = os.environ.get(name)
    return str(val or "").strip()


def _scrub(text: str, key: str) -> str:
    """``text`` with the key (and anything key-shaped after 'key'/'clientKey'/'Basic') removed."""
    out = str(text or "")
    if key:
        out = out.replace(key, "***")
    out = re.sub(r"(?i)(clientKey\"?\s*[:=]\s*\"?|Basic\s+|[?&]key=)[A-Za-z0-9_\-.]{6,}", r"\1***", out)
    return out[:300]


def _urllib_transport(method: str, url: str, headers: Dict[str, str], body: Optional[dict],
                      timeout: float) -> Tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Accept": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- fixed provider hosts
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read() or b""
    try:
        return status, json.loads(raw.decode("utf-8") or "null")
    except ValueError:
        return status, {"message": raw[:200].decode("utf-8", "replace")}


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode("ascii")


class _Provider:
    name = ""
    kinds: frozenset = frozenset()

    def __init__(self, key: str, transport: Optional[Transport] = None, timeout_s: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        if not key:
            raise MissingKey(f"no {KEY_ENV[self.name]} set")
        self._key = key
        self._transport = transport or _urllib_transport
        self.timeout_s = timeout_s
        self.sleep, self.clock = sleep, clock

    def __repr__(self) -> str:  # never the key
        return f"<{type(self).__name__} key=***>"

    def supports(self, kind: str) -> bool:
        return kind in self.kinds

    def _call(self, method: str, url: str, headers: Dict[str, str], body: Optional[dict]) -> Tuple[int, Any]:
        try:
            status, data = self._transport(method, url, headers, body, self.timeout_s)
        except Exception as exc:  # network: the message may echo the request, so scrub it
            raise SolverError(f"{self.name}: {_scrub(f'{type(exc).__name__}: {exc}', self._key)}") from None
        logger.debug("captcha solver %s: %s %s -> %s", self.name, method, url.split("?")[0], status)
        return status, data

    def _fail(self, what: str) -> SolverError:
        return SolverError(f"{self.name}: {_scrub(what, self._key)}")


# ---- CapSolver ---------------------------------------------------------------------------------------------

#: The objects ReCaptchaV2Classification knows (its documented question list), matched against the
#: challenge's own words; the most specific phrase first ("school bus" before "bus").
CAPSOLVER_QUESTIONS: Tuple[Tuple[str, str], ...] = (
    ("school bus", "/m/02yvhj"), ("taxi", "/m/0pg52"), ("bus", "/m/01bjv"), ("motorcycle", "/m/04_sv"),
    ("tractor", "/m/013xlm"), ("chimney", "/m/01jk_4"), ("crosswalk", "/m/014xcs"),
    ("traffic light", "/m/015qff"), ("bicycle", "/m/0199g"), ("parking meter", "/m/015qbp"),
    ("car", "/m/0k4j"), ("bridge", "/m/015kr"), ("boat", "/m/019jd"), ("palm tree", "/m/0cdl1"),
    ("mountain", "/m/09d_r"), ("hill", "/m/09d_r"), ("fire hydrant", "/m/01pns0"), ("hydrant", "/m/01pns0"),
    ("stair", "/m/01lynh"),
)


def capsolver_question(instruction: str) -> Optional[str]:
    words = " " + re.sub(r"[^a-z ]+", " ", (instruction or "").lower()) + " "
    for phrase, qid in CAPSOLVER_QUESTIONS:
        if re.search(r"\b" + re.escape(phrase) + r"(?:e?s)?\b", words):
            return qid
    return None


class CapSolver(_Provider):
    name = "capsolver"
    kinds = frozenset({bc.RECAPTCHA_V2})

    def solve(self, request: SolveRequest) -> SolveAnswer:
        if request.kind not in self.kinds:
            raise SolverUnsupported(f"capsolver: {request.kind} is not a ReCaptchaV2Classification task")
        qid = capsolver_question(request.instruction)
        if qid is None:
            raise SolverUnsupported(f"capsolver: no question id for {request.instruction[:80]!r}")
        body = {"clientKey": self._key,
                "task": {"type": "ReCaptchaV2Classification", "image": _b64(request.image_png), "question": qid}}
        status, data = self._call("POST", CAPSOLVER_URL, {}, body)
        if not isinstance(data, dict):
            raise self._fail(f"HTTP {status}: not JSON")
        if data.get("errorId"):
            code = str(data.get("errorCode") or "")
            what = f"{code} {data.get('errorDescription') or ''}".strip()
            if code in ("ERROR_UNKNOWN_QUESTION", "ERROR_CAPTCHA_UNSOLVABLE", "ERROR_TASK_NOT_SUPPORTED"):
                raise SolverUnsupported(f"capsolver: {_scrub(what, self._key)}")
            raise self._fail(what or f"HTTP {status}")
        sol = data.get("solution") or {}
        if sol.get("type") == "single":
            return SolveAnswer(tiles=(0,) if sol.get("hasObject") else ())
        size = int(sol.get("size") or 0)
        rows, cols = request.grid
        if size and size != cols:
            raise self._fail(f"answered a {size}x{size} grid for a {rows}x{cols} one")
        objs = sol.get("objects")
        if not isinstance(objs, list):
            raise self._fail("no objects in the solution")
        return SolveAnswer(tiles=tuple(int(o) for o in objs if isinstance(o, int) or str(o).isdigit()))


# ---- NopeCHA -------------------------------------------------------------------------------------------------

class NopeCHA(_Provider):
    name = "nopecha"
    kinds = frozenset({bc.RECAPTCHA_V2, bc.HCAPTCHA, bc.IMAGE_TEXT})
    INCOMPLETE = 14          # "Incomplete job", HTTP 409: retry after 500 ms
    POLL_S = 0.5

    def solve(self, request: SolveRequest) -> SolveAnswer:
        rows, cols = request.grid
        uri = lambda png: "data:image/png;base64," + _b64(png)  # noqa: E731
        if request.kind == bc.RECAPTCHA_V2:
            endpoint = "recaptcha"
            body: dict = {"task": request.instruction, "grid": f"{rows}x{cols}", "image_data": [uri(request.image_png)]}
        elif request.kind == bc.HCAPTCHA:
            if not request.tiles_png:
                raise SolverUnsupported("nopecha: hCaptcha needs one image per tile")
            endpoint = "hcaptcha"
            body = {"data": {"request_type": "image_label_binary", "requester_question": {"en": request.instruction},
                             "tasklist": [{"datapoint_uri": uri(p), "task_key": str(i)}
                                          for i, p in enumerate(request.tiles_png)]}}
        elif request.kind == bc.IMAGE_TEXT:
            endpoint = "textcaptcha"
            body = {"image_data": [uri(request.image_png)]}
        else:
            raise SolverUnsupported(f"nopecha: {request.kind} is not a recognition task here")
        headers = {"Authorization": f"Basic {self._key}"}
        status, data = self._call("POST", NOPECHA_URL + endpoint, headers, body)
        job = data.get("data") if isinstance(data, dict) else None
        if status != 200 or not isinstance(job, str) or not job:
            raise self._error(status, data)
        deadline = self.clock() + self.timeout_s
        while True:
            self.sleep(self.POLL_S)
            status, data = self._call("GET", f"{NOPECHA_URL}{endpoint}?id={job}", headers, None)
            if status == 409 and isinstance(data, dict) and data.get("code") == self.INCOMPLETE:
                if self.clock() >= deadline:
                    raise self._fail(f"no answer in {self.timeout_s:g}s")
                continue
            if status != 200 or not isinstance(data, dict):
                raise self._error(status, data)
            return self._answer(request, data.get("data"))

    def _error(self, status: int, data: Any) -> SolverError:
        msg = data.get("message") if isinstance(data, dict) else ""
        code = data.get("code") if isinstance(data, dict) else None
        return self._fail(f"HTTP {status} code {code}: {msg or 'error'}")

    def _answer(self, request: SolveRequest, got: Any) -> SolveAnswer:
        if request.kind == bc.IMAGE_TEXT:
            if not isinstance(got, list) or not got or not isinstance(got[0], str):
                raise self._fail("no text in the answer")
            return SolveAnswer(text=got[0])
        if request.kind == bc.HCAPTCHA:
            got = [b for page in (got or []) if isinstance(page, list) for b in page]
        if not isinstance(got, list) or not all(isinstance(b, bool) for b in got):
            raise self._fail("the answer is not a list of booleans")
        return SolveAnswer(tiles=tuple(i for i, b in enumerate(got) if b))


def _factory(cls):
    def make(cfg: dict):
        return cls(provider_key(cls.name), timeout_s=float(cfg.get("timeout_s") or 30.0))
    return make


register_solver("capsolver", _factory(CapSolver))
register_solver("nopecha", _factory(NopeCHA))
