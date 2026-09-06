"""`browser_read` — read a Chromium tab over CDP instead of photographing it.

cua-driver ships nine `browser_*` tools that drive Chromium through the
DevTools protocol. Hermes named none of them anywhere in
`tools/computer_use/*.py`, while already passing `--grant existing-profile` to
the driver for exactly this purpose. The door was built and nothing went
through it.

Sizes here are not invented, and the numbers in the schema are not either.
Measured on this Mac against an isolated Chromium on
en.wikipedia.org/wiki/Accessibility, through `handle_computer_use`:

    default (viewport-first)   9,337 B   ~2,334 tokens   1.6 s
    query='search'             3,805 B   ~  951 tokens   1.6 s
    full                      63,309 B  ~15,827 tokens   1.6 s

A set-of-marks screenshot of the same window is ~1,568 image tokens and
~2.2 s. So the widely-repeated claim that "a snapshot is 200-400 tokens
against 3,000-5,000 for a screenshot" (Playwright MCP) is true of a small
page and badly wrong of a real one — an unscoped read is TEN TIMES a
screenshot. That measurement is what shaped the defaults being tested here:
this tool wins on ACTING (an exact ref, no coordinate to guess, no focus
stolen), not on bulk reading, so the cheap shape is the default and the
expensive one has to be asked for by name.
"""

from __future__ import annotations

import json

import pytest

from tools.computer_use import tool as t
from tools.computer_use import stall
from tools.computer_use.tool import handle_computer_use


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    # Own detector map, own session — `_stall_detectors` is module-level and
    # keyed by session, and sharing the default one is how a green file makes
    # a red suite under pytest-randomly.
    monkeypatch.setattr(t, "_stall_detectors", {}, raising=False)


SID = "browser-read-tests"


def _outline(n_lines: int) -> str:
    return "\n".join(f'- button "b{i}" [focusable=true]' for i in range(n_lines))


def _refs(n: int, visible: int = 0):
    out = []
    for i in range(n):
        out.append({
            "ref": f"p1:{i}",
            "role": "button",
            "name": f"b{i}",
            "visibility": "in_viewport" if i < visible else "offscreen",
            "actions": ["click"],
        })
    return out


class FakeBackend:
    """Just enough driver to exercise the wiring, with the real shapes."""

    LIST_WINDOWS_NOTE = None

    def __init__(self, *, bind_status="ok", refusal=None,
                 outline=None, refs=None, content_refs=None,
                 windows=None, snapshot_status="ok"):
        self._bind_status = bind_status
        self._refusal = refusal
        self._outline = outline if outline is not None else _outline(3)
        self._refs = refs if refs is not None else _refs(3, visible=3)
        self._content_refs = content_refs if content_refs is not None else []
        self._snapshot_status = snapshot_status
        self._windows = windows if windows is not None else [
            {"pid": 42, "window_id": 7, "app_name": "Google Chrome",
             "on_current_space": True, "z_index": 0, "off_screen": False},
        ]
        self.target = None
        self.tab = None
        self.binds = []
        self.snapshots = []

    def list_windows(self):
        return list(self._windows)

    def browser_bind(self, *, pid, window_id):
        self.binds.append((pid, window_id))
        if self._bind_status != "ok":
            return {"status": "refused", "refusal": self._refusal or {}}
        self.target, self.tab = "bt-1", "tab-1"
        return {"status": "ok", "target_id": self.target,
                "tabs": [{"tab_id": self.tab, "active": True}]}

    def browser_targets(self):
        return self.target, self.tab

    def browser_snapshot(self, **kw):
        self.snapshots.append(kw)
        if self._snapshot_status != "ok":
            return {"status": "refused",
                    "refusal": {"code": "target_gone", "message": "tab is gone"}}
        return {
            "status": "ok", "target_id": "bt-1", "tab_id": "tab-1",
            "page": {"url": "https://example.test/", "title": "T"},
            "outline": self._outline,
            "refs": list(self._refs),
            "content_refs": list(self._content_refs),
            "snapshot": {"complete": True, "continuation": None},
        }


@pytest.fixture
def backend(monkeypatch):
    b = FakeBackend()
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    return b


def _read(**args):
    return json.loads(handle_computer_use(
        {"action": "browser_read", **args}, session_id=SID))


# ── the wiring ────────────────────────────────────────────────────────────

def test_binds_then_reads(backend):
    out = _read()
    assert out["status"] == "ok"
    assert backend.binds == [(42, 7)]
    assert out["page"]["url"] == "https://example.test/"


def test_binding_is_sticky(backend):
    _read()
    _read(query="anything")          # different args, so the repeat guard is quiet
    assert len(backend.binds) == 1, "a second read must reuse the binding"


def test_rebind_forces_a_fresh_bind(backend):
    _read()
    _read(rebind=True, query="x")
    assert len(backend.binds) == 2


def test_query_is_passed_through(backend):
    _read(query="Send")
    assert backend.snapshots[-1]["query"] == "Send"


def test_scope_ref_and_continuation_pass_through(backend):
    _read(scope_ref="p1:9")
    assert backend.snapshots[-1]["scope_ref"] == "p1:9"
    _read(continuation="tok-1", query="q")
    assert backend.snapshots[-1]["continuation"] == "tok-1"


# ── the budgets, which are the point ──────────────────────────────────────

def test_default_keeps_viewport_refs_first_and_caps_them(monkeypatch):
    b = FakeBackend(refs=_refs(80, visible=5))
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read()
    assert len(out["refs"]) == t._BROWSER_REF_BUDGET
    assert out["refs_omitted"] == 80 - t._BROWSER_REF_BUDGET
    # The five in-viewport refs are the ones a click will use, so they must
    # survive the cut whatever order the driver returned them in.
    kept = {r["ref"] for r in out["refs"]}
    assert {f"p1:{i}" for i in range(5)} <= kept


def test_default_truncates_a_long_outline_and_says_so(monkeypatch):
    b = FakeBackend(outline=_outline(500))
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read()
    assert out["outline_truncated"] is True
    assert len(out["outline"]) <= t._BROWSER_OUTLINE_BUDGET
    assert out["outline"].endswith("]"), "truncate on a line, not mid-node"
    assert "query" in out["hint"]


def test_short_outline_is_not_truncated(backend):
    out = _read()
    assert "outline_truncated" not in out
    assert out["outline"].count("\n") == 2


def test_content_is_dropped_unless_asked_for(monkeypatch):
    content = [{"ref": "p1:c", "role": "statictext", "name": "x" * 500}]
    b = FakeBackend(content_refs=content)
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read()
    assert "content_refs" not in out
    assert out["text_available"] == 1
    out = _read(include_text=True)
    assert out["content_refs"] == content


def test_full_returns_everything_unbudgeted(monkeypatch):
    b = FakeBackend(refs=_refs(80), outline=_outline(500),
                    content_refs=[{"ref": "p1:c", "name": "t"}])
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read(full=True)
    assert len(out["refs"]) == 80
    assert "refs_omitted" not in out
    assert "outline_truncated" not in out
    assert out["content_refs"]


def test_the_default_read_is_smaller_than_the_full_one(monkeypatch):
    """The defaults exist to make the cheap shape the automatic one.

    Guarding the ordering rather than an absolute byte count: the absolute
    numbers belong to a real page (they are in this file's docstring), and a
    test that pinned them would be pinning the fake.
    """
    b = FakeBackend(refs=_refs(80), outline=_outline(500),
                    content_refs=[{"ref": "p1:c", "name": "t" * 900}])
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    small = len(json.dumps(_read()))
    big = len(json.dumps(_read(full=True)))
    assert small * 3 < big


# ── refusals, said in words the model can act on ──────────────────────────

def test_consent_refusal_names_the_setting(monkeypatch):
    b = FakeBackend(bind_status="refused", refusal={
        "code": "browser_consent_required",
        "message": "this standalone browser profile requires explicit "
                   "existing-profile approval"})
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read()
    assert out["code"] == "browser_consent_required"
    assert "grant_existing_profile" in out["hint"]
    assert "capture" in out["hint"], "there must be a route that still works"


def test_other_bind_refusals_fall_back_to_the_screen(monkeypatch):
    b = FakeBackend(bind_status="refused",
                    refusal={"code": "heuristic_binding", "message": "no"})
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read()
    assert out["code"] == "heuristic_binding"
    assert "capture" in out["hint"]


def test_no_browser_window_is_a_named_code_not_a_crash(monkeypatch):
    b = FakeBackend(windows=[{"pid": 1, "window_id": 2, "app_name": "Finder",
                              "on_current_space": True, "z_index": 0}])
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read()
    assert out["code"] == "no_browser_window"
    assert "Safari" in out["hint"], "say which browsers are not CDP targets"


def test_snapshot_refusal_suggests_a_rebind(backend, monkeypatch):
    _read()
    backend._snapshot_status = "refused"
    out = _read(query="x")
    assert out["code"] == "target_gone"
    assert "rebind" in out["hint"]


def test_backend_without_cdp_says_so(monkeypatch):
    class Old:
        pass
    monkeypatch.setattr(t, "_get_backend", lambda **kw: Old())
    out = _read()
    assert "cannot drive a browser" in out["error"]
    assert "capture" in out["hint"]


# ── window choice ─────────────────────────────────────────────────────────

def test_app_argument_selects_the_browser(monkeypatch):
    b = FakeBackend(windows=[
        {"pid": 1, "window_id": 1, "app_name": "Google Chrome",
         "on_current_space": True, "z_index": 1},
        {"pid": 2, "window_id": 2, "app_name": "Brave Browser",
         "on_current_space": True, "z_index": 0},
    ])
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    _read(app="Brave")
    assert b.binds == [(2, 2)]


def test_explicit_pid_and_window_skip_the_search(monkeypatch):
    b = FakeBackend(windows=[])
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    _read(pid=99, window_id=5)
    assert b.binds == [(99, 5)]


def test_a_window_on_another_space_is_still_readable(monkeypatch):
    """Input to another Space is refused by macOS. A CDP read is not — it
    never touches WindowServer — so an off-Space browser must not be skipped
    the way `capture` has to skip it."""
    b = FakeBackend(windows=[
        {"pid": 3, "window_id": 3, "app_name": "Google Chrome",
         "on_current_space": False, "z_index": 0, "off_screen": False},
    ])
    monkeypatch.setattr(t, "_get_backend", lambda **kw: b)
    out = _read()
    assert out["status"] == "ok"
    assert b.binds == [(3, 3)]


# ── the guard that a new verb walks past ──────────────────────────────────

def test_browser_read_counts_as_a_look(monkeypatch):
    """Fourth time in this codebase for this exact shape.

    `stall._READ_ONLY` decides two things, and the second one is the trap:
    `record()` treats anything NOT in that set as an ACT and zeroes the
    looks counter. So a `browser_read` missing from it would not merely be
    invisible to the looking-loop detector — alternating capture and
    browser_read would reset the count every other call and the detector
    would never fire again for anyone.
    """
    assert "browser_read" in stall._READ_ONLY
    d = stall.StallDetector()
    for _ in range(4):
        d.record("browser_read", {}, "{}")
    assert d.looks_without_acting() == 4
    d.record("click", {"element": 1}, "{}")
    assert d.looks_without_acting() == 0


def test_looking_loop_refuses_a_run_of_browser_reads():
    d = stall.StallDetector()
    for _ in range(stall.LOOK_HARD_LIMIT - 1):
        d.record("browser_read", {}, "{}")
    verdict = d.looking_loop("browser_read")
    assert verdict is not None and verdict["refused"] is True


def test_browser_read_is_not_gated_as_destructive():
    """Reading a page has no side effect, like `capture`. If this ever needs
    to change, the acting verbs are what to gate — not the read."""
    assert "browser_read" not in t._DESTRUCTIVE_ACTIONS


def test_browser_read_is_in_the_schema():
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA as S
    props = S["parameters"]["properties"]
    assert "browser_read" in props["action"]["enum"]
    # The schema is what a model reads when the skill is not loaded, so the
    # measured cost has to be in it, not only in the docs.
    for key in ("query", "scope_ref", "continuation", "include_text",
                "full", "rebind"):
        assert key in props, key
    assert "tokens" in props["query"]["description"]
    assert "screenshot" in props["full"]["description"]


def test_two_different_reads_are_not_one_repeated_call():
    """Fifth instance of the repo's oldest shape, in the guard for the fourth.

    `call_fingerprint` enumerates argument names, and every `browser_read`
    discriminator was missing from that list — so reading a page, then
    reading one panel of it, then widening to `full`, all fingerprinted
    identically and the second was refused as "this exact call". The comment
    directly above the list already says why that is wrong, about menu paths.
    Caught by the live sizes script refusing its own last measurement.
    """
    base = {"app": "Google Chrome"}
    seen = {
        stall.call_fingerprint("browser_read", dict(base, **extra))
        for extra in (
            {},
            {"query": "Send"},
            {"query": "Search"},
            {"scope_ref": "p1:4"},
            {"full": True},
            {"include_text": True},
            {"continuation": "tok-2"},
        )
    }
    assert len(seen) == 7, "each read shape must be its own call"


def test_the_same_read_twice_is_still_a_repeat():
    """The point of the fingerprint survives: identical really is identical."""
    a = stall.call_fingerprint("browser_read", {"app": "Chrome", "query": "Send"})
    b = stall.call_fingerprint("browser_read", {"app": "Chrome", "query": "Send"})
    assert a == b
