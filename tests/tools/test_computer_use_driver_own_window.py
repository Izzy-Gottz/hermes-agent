"""Ticket #8, screen half: an unqualified capture must never target cua-driver's own window.

Measured 2026-09-23 on macOS 27.0 with the standalone daemon running
(``/Applications/CuaDriver.app/Contents/MacOS/cua-driver serve``, pid 69038): it owns an on-screen,
untitled, full-screen window, and over the MCP transport with a session that window is index 0 of
``list_windows {"on_screen_only": true}``. ``get_window_state`` on it is refused with the text below.
``_select_capture_target`` took ``pool[0]``, so every unqualified ``computer_use capture`` failed —
7 times in 6 sessions, 09-17 to 09-23 — and the model told the person "you're looking at the
permissions dialog itself", with no dialog on screen.

The records and the refusal are the measured ones, passed through the real MCP result flattener.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tools.computer_use import cua_backend_capture as cap_mod
from tools.computer_use.cua_backend_parse import _extract_tool_result

# Verbatim from the live driver (list_windows over MCP with a session; index 0).
DRIVER_WINDOW = {"app_name": "Cua Driver", "bounds": {"height": 956, "width": 1470, "x": 0, "y": 0},
                 "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 69038,
                 "space_ids": [1, 30, 45, 40], "title": "", "window_id": 405, "z_index": 5}
SAFARI_WINDOW = {"app_name": "Safari", "bounds": {"height": 900, "width": 1400, "x": 20, "y": 30},
                 "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 812,
                 "space_ids": [45], "title": "Flights to Tel Aviv", "window_id": 9911, "z_index": 4}
# Verbatim from `cua-driver call get_window_state '{"pid":69038,"window_id":405}'`.
REFUSAL = "Permission denied: Cua Driver refuses operations that target its own authorization process"

_PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
            "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg==")


def _mcp(*, text=None, structured=None, image=None, is_error=False):
    content = []
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    if image is not None:
        content.append(SimpleNamespace(type="image", data=image, mimeType="image/png"))
    return _extract_tool_result(SimpleNamespace(content=content, structuredContent=structured, isError=is_error))


class _FakeSession:
    """A cua-driver MCP session: list_windows returns ``windows``; get_window_state refuses every
    window whose pid is the driver's, exactly as the live driver does, and captures the rest."""

    capabilities_discovered = True

    def __init__(self, windows, driver_pid=69038):
        self.windows, self.driver_pid, self.calls = windows, driver_pid, []

    def _has_tool(self, name):
        return name != "screenshot"

    def call_tool(self, name, args, *a, **k):
        self.calls.append((name, dict(args)))
        if name == "list_windows":
            return _mcp(structured={"current_space_id": 45, "windows": copy.deepcopy(self.windows)})
        if name == "get_window_state":
            if args.get("pid") == self.driver_pid:
                return _mcp(text=REFUSAL, is_error=True)
            return _mcp(text='AXWindow "Flights to Tel Aviv"\n[0] AXButton "Search"', image=_PNG_B64,
                        structured={"elements": [], "snapshot_id": "s1"})
        return _mcp(structured={})

    def _call_tool_via_cli(self, name, args, timeout):
        return self.call_tool(name, args)

    def gws_targets(self):
        return [(a.get("pid"), a.get("window_id")) for n, a in self.calls if n == "get_window_state"]


def _backend(windows, **kw):
    from tools.computer_use.cua_backend import CuaDriverBackend
    backend = CuaDriverBackend()
    backend._session = _FakeSession(windows, **kw)
    return backend


@pytest.fixture(autouse=True)
def _no_live_process_table(monkeypatch):
    """Unit tests must not depend on whether a cua-driver runs on the machine running them."""
    monkeypatch.setattr(cap_mod, "_driver_pids", lambda: frozenset(), raising=False)


def test_measured_driver_window_at_index_0_is_skipped():
    backend = _backend([DRIVER_WINDOW, SAFARI_WINDOW])
    cap = backend.capture()  # unqualified: what failed 7 times
    assert cap.app == "Safari" and cap.png_b64 == _PNG_B64
    assert (backend._active_pid, backend._active_window_id) == (812, 9911)
    assert (69038, 405) not in backend._session.gws_targets()  # never even asked


def test_driver_window_known_only_by_pid_is_skipped(monkeypatch):
    """A driver whose window carries no recognisable name is still caught by its pid."""
    monkeypatch.setattr(cap_mod, "_driver_pids", lambda: frozenset({69038}))
    backend = _backend([{**DRIVER_WINDOW, "app_name": ""}, SAFARI_WINDOW])
    cap = backend.capture()
    assert cap.app == "Safari"
    assert (69038, 405) not in backend._session.gws_targets()


def test_refusal_falls_through_to_the_next_window():
    """Nothing identifies the window (no name, pid unknown), so it is tried — and the driver's
    refusal moves the default capture on to the next window instead of failing it. The next window
    is untitled too, so macOS ranking (titled first) keeps the driver's window first and the
    fall-through itself is what is tested."""
    backend = _backend([{**DRIVER_WINDOW, "app_name": ""}, {**SAFARI_WINDOW, "title": ""}])
    cap = backend.capture()
    assert cap.app == "Safari" and cap.png_b64 == _PNG_B64
    assert backend._session.gws_targets() == [(69038, 405), (812, 9911)]
    assert backend._last_app == "Safari"


def test_explicit_target_on_the_driver_window_says_whose_it_is():
    backend = _backend([DRIVER_WINDOW, SAFARI_WINDOW])
    with pytest.raises(RuntimeError) as raised:
        backend.capture(pid=69038, window_id=405)
    msg = str(raised.value)
    assert "belongs to the screen-control helper itself" in msg and "capture another app" in msg
    assert "authorization process" not in msg  # the driver's words no longer reach the model
    assert backend._active_pid is None


def test_explicit_target_through_the_tool_is_a_clear_error(monkeypatch):
    import json
    from tools.computer_use import tool as cu_tool
    backend = _backend([DRIVER_WINDOW, SAFARI_WINDOW])
    monkeypatch.setattr(cu_tool, "_get_backend", lambda session_id=None: backend)
    out = json.loads(cu_tool.handle_computer_use({"action": "capture", "pid": 69038, "window_id": 405}))
    assert "belongs to the screen-control helper itself" in out["error"], out


def test_only_the_driver_window_on_screen():
    backend = _backend([DRIVER_WINDOW])
    cap = backend.capture()
    assert cap.png_b64 is None and "screen-control helper itself" in cap.window_title
    assert backend._session.gws_targets() == []


def test_explicit_app_request_is_unchanged():
    """app='Safari' still resolves by name; the driver filter only governs default captures."""
    backend = _backend([DRIVER_WINDOW, SAFARI_WINDOW])
    assert backend.capture(app="Safari").app == "Safari"


def test_select_capture_target_contract():
    from tools.computer_use.cua_backend_parse import _ingest_windows
    windows = _ingest_windows([DRIVER_WINDOW, SAFARI_WINDOW])
    assert cap_mod._select_capture_target(windows, app_requested=False)["app_name"] == "Safari"
    # exact / app-filtered selection is untouched
    assert cap_mod._select_capture_target(windows, app_requested=True)["app_name"] == "Cua Driver"
    bundled = _ingest_windows([{**DRIVER_WINDOW, "app_name": "x", "bundle_id": "com.trycua.driver"}, SAFARI_WINDOW])
    assert cap_mod._select_capture_target(bundled, app_requested=False)["app_name"] == "Safari"


# ── macOS: a strip is not "the screen" ──────────────────────────────────────────────────────────
# Measured 2026-09-23 after the driver-window fix: the unqualified capture picked Chrome window 3408,
# untitled and 1470x41 (a toolbar strip); the real 801 px Chrome window was fourth. Live the same
# day: cmux window 205, untitled, 1470x32, above cmux window 79, "ssh sruly-5", 1470x923.
CHROME_STRIP = {"app_name": "Google Chrome", "bounds": {"height": 41, "width": 1470, "x": 0, "y": 33},
                "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 5120,
                "space_ids": [45], "title": "", "window_id": 3408, "z_index": 4}
CMUX_STRIP = {"app_name": "cmux", "bounds": {"height": 32.0, "width": 1470.0, "x": 0.0, "y": 33.0},
              "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 36280,
              "space_ids": [45], "title": "", "window_id": 205, "z_index": 3}
CHROME_MAIN = {"app_name": "Google Chrome", "bounds": {"height": 801, "width": 1470, "x": 0, "y": 74},
               "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 5120,
               "space_ids": [45], "title": "Google Flights - Find Cheap Flight Options", "window_id": 3391,
               "z_index": 2}


def test_darwin_default_capture_prefers_the_real_window_over_strips(monkeypatch):
    monkeypatch.setattr(cap_mod.sys, "platform", "darwin")
    backend = _backend([DRIVER_WINDOW, CHROME_STRIP, CMUX_STRIP, CHROME_MAIN], driver_pid=69038)
    cap = backend.capture()
    assert (backend._active_pid, backend._active_window_id) == (5120, 3391), cap
    assert backend._session.gws_targets() == [(5120, 3391)]


def test_darwin_ranking_keeps_frontmost_order_within_a_tier(monkeypatch):
    from tools.computer_use.cua_backend_parse import _ingest_windows
    monkeypatch.setattr(cap_mod.sys, "platform", "darwin")
    front_titled = {**CHROME_MAIN, "window_id": 1, "z_index": 9, "title": "A"}
    windows = sorted(_ingest_windows([CHROME_STRIP, CMUX_STRIP, CHROME_MAIN, front_titled]),
                     key=lambda w: w["z_index"], reverse=True)
    order = [w["window_id"] for w in cap_mod._capture_candidates(windows, app_requested=False)]
    assert order == [1, 3391, 3408, 205]  # titled+sized (z order), then the untitled strips (z order)


def test_darwin_ranking_without_bounds_is_plain_frontmost(monkeypatch):
    """A driver that omits bounds/layer keeps the frontmost pick (nothing to rank on); the
    frontmost app's other windows follow it, then every other app in z-order."""
    from tools.computer_use.cua_backend_parse import _ingest_windows
    monkeypatch.setattr(cap_mod.sys, "platform", "darwin")
    bare = [{k: v for k, v in w.items() if k not in ("bounds", "layer")} | {"title": "t"}
            for w in (CHROME_STRIP, CMUX_STRIP, CHROME_MAIN)]
    windows = sorted(_ingest_windows(bare), key=lambda w: w["z_index"], reverse=True)
    assert [w["window_id"] for w in cap_mod._capture_candidates(windows, app_requested=False)] == [3408, 3391, 205]


# ── the ranking never crosses apps: the frontmost app wins ─────────────────────────────────────
MAIL_INBOX = {"app_name": "Mail", "bounds": {"height": 820, "width": 1300, "x": 60, "y": 60},
              "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 700,
              "space_ids": [45], "title": "Inbox", "window_id": 60, "z_index": 1}


def test_darwin_untitled_frontmost_app_beats_a_titled_app_behind(monkeypatch):
    """Spotify is frontmost with an untitled 1200x800 window; Mail "Inbox" is behind it."""
    monkeypatch.setattr(cap_mod.sys, "platform", "darwin")
    spotify = {"app_name": "Spotify", "bounds": {"height": 800, "width": 1200, "x": 100, "y": 50},
               "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 900,
               "space_ids": [45], "title": "", "window_id": 77, "z_index": 3}
    backend = _backend([DRIVER_WINDOW, spotify, MAIL_INBOX], driver_pid=69038)
    backend.capture()
    assert (backend._active_pid, backend._active_window_id) == (900, 77)


def test_darwin_small_frontmost_dialog_beats_a_titled_app_behind(monkeypatch):
    """A Finder "Copy" progress dialog (400x110) is frontmost; Mail is behind it."""
    monkeypatch.setattr(cap_mod.sys, "platform", "darwin")
    copy_dialog = {"app_name": "Finder", "bounds": {"height": 110, "width": 400, "x": 500, "y": 300},
                   "is_on_screen": True, "layer": 0, "on_current_space": True, "pid": 420,
                   "space_ids": [45], "title": "Copy", "window_id": 88, "z_index": 3}
    backend = _backend([DRIVER_WINDOW, copy_dialog, MAIL_INBOX], driver_pid=69038)
    backend.capture()
    assert (backend._active_pid, backend._active_window_id) == (420, 88)
