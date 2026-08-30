"""macOS-native computer_use backend: pure math + a stubbed screencapture.

Nothing here touches the real screen, cursor or keyboard: the capture path
runs against a fake ``screencapture`` (HERMES_SCREENCAPTURE_BIN) that writes
a generated PNG, and every input method is exercised only through the
argument-validation layer or a recorded osascript stub.
"""

from __future__ import annotations

import base64
import io
import os
import stat
import sys

import pytest

from tools.computer_use import macos_native_backend as mnb
from tools.computer_use.macos_native_backend import (
    MAX_LONG_SIDE,
    MacNativeBackend,
    downscale_png,
    fit_scale,
    looks_blank,
    parse_key_combo,
    scale_back,
)


def _png(w: int, h: int, color=(20, 90, 200)) -> bytes:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), color)
    ImageDraw.Draw(im).rectangle((10, 10, w // 3, h // 3), fill=(250, 250, 250))
    out = io.BytesIO()
    im.save(out, "PNG")
    return out.getvalue()


# ── scaling math ───────────────────────────────────────────────────────


def test_fit_scale_only_shrinks():
    assert fit_scale(800, 600) == 1.0
    assert fit_scale(1568, 900) == 1.0
    assert fit_scale(3136, 1000) == pytest.approx(0.5)
    assert fit_scale(1000, 3136) == pytest.approx(0.5)


def test_downscale_png_caps_longest_side_and_reports_scale():
    png, w, h, scale = downscale_png(_png(2940, 1912))
    assert max(w, h) == MAX_LONG_SIDE
    assert (w, h) == (1568, 1020)
    assert scale == pytest.approx(1568 / 2940)
    from PIL import Image

    assert Image.open(io.BytesIO(png)).size == (1568, 1020)


def test_downscale_png_leaves_small_images_alone():
    png, w, h, scale = downscale_png(_png(640, 480))
    assert (w, h, scale) == (640, 480, 1.0)


def test_scale_back_round_trips_retina_and_downscale():
    # 1470x956 pt display, screencapture gives 2940x1912 px, downscaled to
    # 1568x1020 → 1.0667 px per point.
    ppp = 1568 / 1470
    assert scale_back(0, 0, pixels_per_point=ppp) == (0, 0)
    assert scale_back(1568, 1020, pixels_per_point=ppp) == (1470, 956)
    assert scale_back(784, 510, pixels_per_point=ppp) == (735, 478)
    # A zoomed region carries its own origin.
    assert scale_back(100, 50, pixels_per_point=2.0, origin=(300, 200)) == (350, 225)


def test_looks_blank_detects_uniform_capture():
    from PIL import Image

    flat = io.BytesIO()
    Image.new("RGB", (200, 100), (40, 40, 40)).save(flat, "PNG")
    assert looks_blank(flat.getvalue())
    assert not looks_blank(_png(200, 100))


# ── key combos (fixed table, never raw script) ────────────────────────


@pytest.mark.parametrize(
    "combo, mods, key, code",
    [
        ("cmd+s", ["command down"], "s", None),
        ("cmd+shift+s", ["command down", "shift down"], "s", None),
        ("ctrl-alt-t", ["control down", "option down"], "t", None),
        ("return", [], "return", 36),
        ("cmd+return", ["command down"], "return", 36),
        ("escape", [], "escape", 53),
        ("Tab", [], "tab", 48),
    ],
)
def test_parse_key_combo(combo, mods, key, code):
    assert parse_key_combo(combo) == (mods, key, code)


@pytest.mark.parametrize("bad", ["", "cmd", "cmd+", "hyper+x", 'cmd+"', "cmd+ab", "x; rm -rf /"])
def test_parse_key_combo_rejects_everything_else(bad):
    with pytest.raises(ValueError):
        parse_key_combo(bad)


# ── capture through a stubbed screencapture ───────────────────────────


@pytest.fixture
def fake_screencapture(tmp_path, monkeypatch):
    """A ``screencapture`` stand-in that copies a fixture PNG to its last arg
    and records the argv it was called with."""
    fixture = tmp_path / "fixture.png"
    fixture.write_bytes(_png(2940, 1912))
    log = tmp_path / "argv.log"
    stub = tmp_path / "screencapture"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        'for last; do :; done\n'
        f'cp "{fixture}" "$last"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HERMES_SCREENCAPTURE_BIN", str(stub))
    monkeypatch.setenv("HERMES_SCREENSHOT_DIR", str(tmp_path / "shots"))
    return log


def _backend_with_fake_display(monkeypatch) -> MacNativeBackend:
    b = MacNativeBackend()
    monkeypatch.setattr(b, "screen_points", lambda: (1470, 956))
    monkeypatch.setattr(b, "_front_window_info", lambda: {"app": "Safari", "title": "Home", "bounds": [10, 20, 800, 600]})
    return b


def test_capture_downscales_and_records_scale(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    cap = b.capture(mode="vision")
    assert cap.mode == "vision"
    assert (cap.width, cap.height) == (1568, 1020)
    assert cap.image_mime_type == "image/png"
    assert cap.app == "Safari" and cap.window_title == "Home"
    assert "screenshot pixels (1568x1020)" in cap.note
    assert b._ppp == pytest.approx(1568 / 1470)
    from PIL import Image

    assert Image.open(io.BytesIO(base64.b64decode(cap.png_b64))).size == (1568, 1020)
    # Full-screen is still an explicit -R of the main display: a bare
    # screencapture writes one file per display and the extras would be
    # left behind in the temp dir.
    assert "-R 0,0,1470,956" in fake_screencapture.read_text()


def test_som_mode_degrades_to_vision_without_elements(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    cap = b.capture(mode="som")
    assert cap.mode == "vision" and cap.elements == []


def test_zoom_uses_region_in_points_and_maps_clicks_back(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    # Zoom into a 400x300 px box at (784, 510) px of the full shot.
    cap = b.zoom((784, 510, 400, 300))
    argv = fake_screencapture.read_text().splitlines()[-1]
    # (784,510) px → (735,478) pt; 400x300 px → 375x281 pt.
    assert "-R 735,478,375,281" in argv
    assert cap.width == 1568  # fixture is re-used, so it is downscaled again
    assert b._origin == (735.0, 478.0)
    # Clicking the centre of the zoomed image lands mid-region on screen.
    # The zoomed image is 1568 px wide for a 375 pt region: 4.18 px/pt (the
    # stub re-uses the full-screen fixture, so only the width is meaningful).
    ppp = 1568 / 375
    px, py = b._to_points(cap.width // 2, cap.height // 2)
    assert (px, py) == (int(round(735 + 784 / ppp)), int(round(478 + 510 / ppp)))


def test_capture_note_warns_when_image_is_blank(monkeypatch, tmp_path, fake_screencapture):
    from PIL import Image

    flat = tmp_path / "fixture.png"
    Image.new("RGB", (1470, 956), (30, 30, 30)).save(flat, "PNG")
    b = _backend_with_fake_display(monkeypatch)
    cap = b.capture(mode="vision")
    assert "Screen Recording" in cap.note


def test_capture_error_names_screen_recording(monkeypatch, tmp_path):
    stub = tmp_path / "screencapture"
    stub.write_text("#!/bin/sh\necho 'could not create image from display' >&2\nexit 1\n")
    stub.chmod(0o755)
    monkeypatch.setenv("HERMES_SCREENCAPTURE_BIN", str(stub))
    b = _backend_with_fake_display(monkeypatch)
    with pytest.raises(RuntimeError, match="Screen Recording"):
        b.capture(mode="vision")


def test_ax_mode_has_no_image(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    cap = b.capture(mode="ax")
    assert cap.png_b64 is None and "bounds" in cap.note


# ── input goes through validation, then osascript with argv out-of-band ──


# ── focus stub ─────────────────────────────────────────────────────────
# The backend asks System Events "who is frontmost / what is focused" before
# and after every mutating action. ``_focus_stub`` answers those queries from
# a queue of fake states and records everything else (the real keystrokes),
# so nothing touches the screen.


def _focus_line(app="Terminal", bid="com.apple.Terminal", pid=100, title="moe — zsh",
                role="AXTextArea", subrole="", vkind="text", vmatch="unknown") -> str:
    return "\t".join([app, bid, str(pid), title, role, subrole, vkind, vmatch])


FINDER_DESKTOP = _focus_line("Finder", "com.apple.finder", 703, "", "AXGroup", "", "", "unknown")


def _focus_stub(monkeypatch, b, states):
    """Route ``_AS_FOCUS`` queries to successive ``states`` (the last one
    repeats; a BaseException instance is raised); return the list of
    (script, argv) for every *other* AppleScript call, plus the focus argv log."""
    monkeypatch.setattr(mnb, "_SETTLE_S", 0)
    states = list(states)
    sent, focus_argv = [], []

    def fake(script, argv=None):
        if script is b._AS_FOCUS:
            focus_argv.append(list(argv or []))
            st = states.pop(0) if len(states) > 1 else states[0]
            if isinstance(st, BaseException):
                raise st
            return st
        sent.append((script, list(argv or [])))
        return ""

    monkeypatch.setattr(b, "_applescript", fake)
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: sent.append(("jxa", list(argv or []))) or "ok")
    return sent, focus_argv


def test_click_needs_coordinates_not_elements(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    res = b.click(element=3)
    assert not res.ok and res.code == "no_elements"
    with pytest.raises(ValueError):
        b.click(x=None, y=None)


def test_click_posts_scaled_point_via_jxa(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    sent, _ = _focus_stub(monkeypatch, b, [_focus_line()])
    calls = [argv for kind, argv in sent if kind == "jxa"]
    res = b.click(x=784, y=510, click_count=2)
    assert res.ok and res.path == "cgevent_fg" and res.effect == "unverifiable"
    calls = [argv for kind, argv in sent if kind == "jxa"]
    assert calls == [["click", "735", "478", "2", "left"]]
    b.drag(from_xy=(0, 0), to_xy=(1568, 1020))
    calls = [argv for kind, argv in sent if kind == "jxa"]
    assert calls[-1] == ["drag", "0", "0", "1470", "956"]
    b.scroll(direction="down", amount=4)
    calls = [argv for kind, argv in sent if kind == "jxa"]
    assert calls[-1][0] == "scroll" and calls[-1][3:] == ["-4", "0"]


def test_type_and_key_pass_text_as_argv(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    calls, _ = _focus_stub(monkeypatch, b, [_focus_line()])
    b.type_text('say "hi"; rm -rf /')
    script, argv = calls[-1]
    assert argv == ['say "hi"; rm -rf /'] and 'rm -rf' not in script
    b.key("cmd+shift+s")
    script, argv = calls[-1]
    assert argv == ["s"] and "command down, shift down" in script
    b.key("return")
    script, argv = calls[-1]
    assert "key code 36" in script and argv == []
    with pytest.raises(ValueError):
        b.key("cmd+\"")


def test_type_chunks_long_text(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    calls, focus_argv = _focus_stub(monkeypatch, b, [_focus_line()])
    monkeypatch.setattr(mnb.time, "sleep", lambda s: None)
    b.type_text("x" * 120)
    assert [len(a[0]) for _, a in calls] == [50, 50, 20]
    # The post-type verification compares only the last 50 chars, as argv.
    assert focus_argv == [[], ["x" * 50]]


def test_list_windows_falls_back_to_front_window_when_cgwindowlist_fails(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: (_ for _ in ()).throw(RuntimeError("no bridge")))
    (win,) = b.list_windows()
    assert win["app"] == "Safari" and win["bounds_points"] == {"x": 10, "y": 20, "w": 800, "h": 600}
    assert win["bounds_screenshot_px"]["w"] == round(800 * 1568 / 1470)
    assert win["frontmost"] and "front window only" in win["note"]


def _cg_windows(monkeypatch, b, windows, front=None, display=(1470, 956)):
    import json as _json
    payload = {"display": list(display), "front": front or {"name": "Safari", "bundle_id": "com.apple.Safari", "pid": 5},
               "windows": windows}
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: _json.dumps(payload))


def _w(app, pid, wid, title, x, y, w, h, alpha=1):
    return {"app": app, "pid": pid, "window_id": wid, "title": title, "x": x, "y": y, "w": w, "h": h, "alpha": alpha}


def test_list_windows_frontmost_skips_untitled_helper_windows(monkeypatch, fake_screencapture):
    """Verified live: Chrome's tab-strip helper (untitled, 41 px) sits above
    its real window in z-order; `frontmost` must land on the titled window."""
    b = _backend_with_fake_display(monkeypatch)
    _cg_windows(monkeypatch, b, [
        _w("Google Chrome", 5, 1, "", 0, 33, 1470, 41),
        _w("Google Chrome", 5, 2, "", 0, 74, 1470, 47),
        _w("Google Chrome", 5, 3, "Docs - Google Chrome", 0, 121, 1470, 835),
        _w("Finder", 703, 4, "Desktop", 100, 100, 600, 400),
    ], front={"name": "Google Chrome", "bundle_id": "com.google.Chrome", "pid": 5})
    wins = b.list_windows()
    assert [w["frontmost"] for w in wins] == [False, False, True, False]
    assert [w["front_app"] for w in wins] == [True, True, True, False]
    # All untitled → the top one is still marked so something is.
    _cg_windows(monkeypatch, b, [_w("Google Chrome", 5, 1, "", 0, 33, 1470, 41),
                                 _w("Google Chrome", 5, 2, "", 0, 74, 1470, 47)],
                front={"name": "Google Chrome", "bundle_id": "com.google.Chrome", "pid": 5})
    assert [w["frontmost"] for w in b.list_windows()] == [True, False]


def test_list_windows_lists_every_app_z_ordered(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    _cg_windows(monkeypatch, b, [
        _w("Finder", 703, 1, "Desktop", 100, 100, 600, 400),      # a Finder window in front of Safari
        _w("Electron", 9, 2, "", 0, 0, 1, 1),                        # 1-px helper: skipped
        _w("Safari", 5, 3, "Home", 10, 20, 800, 600),
        _w("Safari", 5, 4, "Docs", 50, 60, 800, 600),
        _w("Terminal", 7, 5, "moe — zsh", 0, 0, 1470, 956),         # fullscreen
        _w("Music", 8, 6, "Music", 1470, 0, 500, 500),               # entirely on a second display
        _w("Ghost", 10, 7, "hidden", 0, 0, 300, 300, alpha=0.0),
    ])
    wins = b.list_windows()
    assert [w["app"] for w in wins] == ["Finder", "Safari", "Safari", "Terminal", "Music", "Ghost"]
    assert [w["z"] for w in wins] == list(range(6))
    # Only the front app's TOP window is `frontmost`; every window of the front app is `front_app`.
    assert [w["frontmost"] for w in wins] == [False, True, False, False, False, False]
    assert [w["front_app"] for w in wins] == [False, True, True, False, False, False]
    assert wins[1]["bundle_id"] == "com.apple.Safari" and "bundle_id" not in wins[0]
    assert wins[0]["pid"] == 703 and wins[0]["window_id"] == 1 and wins[0]["title"] == "Desktop"
    assert wins[1]["bounds_points"] == {"x": 10, "y": 20, "w": 800, "h": 600}
    assert wins[1]["bounds_screenshot_px"]["w"] == round(800 * 1568 / 1470)
    assert wins[3]["note"] == "fullscreen"
    assert wins[4]["note"].startswith("off main display")
    assert wins[5]["note"] == "transparent"
    assert "note" not in wins[0]


def test_list_windows_is_capped(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    _cg_windows(monkeypatch, b, [_w("A", 1, i, f"w{i}", 0, 0, 100, 100) for i in range(60)])
    wins = b.list_windows()
    assert len(wins) == mnb.MAX_WINDOWS == 40
    assert "bounds_screenshot_px" not in wins[0]  # no capture yet → no pixel space


def test_list_windows_is_empty_on_a_bare_desktop(monkeypatch):
    # Verified live: Finder frontmost with no windows → zero layer-0 windows.
    b = _backend_with_fake_display(monkeypatch)
    _cg_windows(monkeypatch, b, [], front={"name": "Finder", "bundle_id": "com.apple.finder", "pid": 703})
    assert b.list_windows() == []


# ── landing report: where did the input go? ────────────────────────────


def test_focused_element_parses_query_and_never_returns_value(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    _, focus_argv = _focus_stub(monkeypatch, b, [_focus_line(vmatch="yes")])
    f = b.focused_element("abc")
    assert f == {"app": "Terminal", "bundle_id": "com.apple.Terminal", "pid": 100, "window_title": "moe — zsh",
                 "role": "AXTextArea", "subrole": "", "value_kind": "text", "value_ends_with_typed": True}
    assert focus_argv == [["abc"]]
    # Without a suffix the comparison is not made and the argv is empty.
    assert b.focused_element()["value_ends_with_typed"] is None and focus_argv[-1] == []


def test_focused_element_reports_failure_softly(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    _focus_stub(monkeypatch, b, [RuntimeError("Accessibility permission missing for the app")])
    f = b.focused_element()
    assert f["app"] == "" and f["role"] == "" and "Accessibility" in f["error"]
    assert b._describe_focus(f).startswith("focus unknown (Accessibility")


def test_type_message_reports_landing_and_value_match(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    sent, focus_argv = _focus_stub(monkeypatch, b, [_focus_line(), _focus_line(vmatch="yes")])
    res = b.type_text("testing")
    assert res.ok
    assert res.message == 'typed 7 chars → Terminal "moe — zsh" AXTextArea (value ends with typed text: yes)'
    assert res.meta["front_app"] == {"name": "Terminal", "bundle_id": "com.apple.Terminal", "pid": 100}
    assert res.meta["focused"] == {"role": "AXTextArea", "subrole": "", "window_title": "moe — zsh",
                                   "value_ends_with_typed": True}
    assert res.meta["front_app_changed"] is False and "front_app_before" not in res.meta
    assert focus_argv == [[], ["testing"]]
    assert "testing" not in str(res.meta)  # value/typed text never echoed in meta
    # A mismatch is reported as "no"; a non-text focus as "unknown".
    _focus_stub(monkeypatch, b, [_focus_line(), _focus_line(vmatch="no")])
    assert b.type_text("x").message.endswith("(value ends with typed text: no)")
    _focus_stub(monkeypatch, b, [_focus_line(role="AXWebArea", vkind="")])
    assert b.type_text("x").message.endswith("AXWebArea (value ends with typed text: unknown)")


def test_click_reports_front_app_change(monkeypatch, fake_screencapture):
    # The live failure: a click meant for a chat box fronted Finder.
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    _focus_stub(monkeypatch, b, [_focus_line(), FINDER_DESKTOP])
    res = b.click(x=100, y=100)
    assert res.ok
    assert res.message.endswith("→ Finder AXGroup; front app changed: Terminal → Finder")
    assert res.meta["front_app_changed"] is True
    assert res.meta["front_app"] == {"name": "Finder", "bundle_id": "com.apple.finder", "pid": 703}
    assert res.meta["front_app_before"] == {"name": "Terminal", "bundle_id": "com.apple.Terminal"}
    assert res.meta["focused"] == {"role": "AXGroup", "subrole": "", "window_title": ""}
    # Same app, different window/focus: no "changed" clause.
    _focus_stub(monkeypatch, b, [_focus_line(), _focus_line(title="other", role="AXButton")])
    res = b.click(x=100, y=100)
    assert res.message.endswith('→ Terminal "other" AXButton') and res.meta["front_app_changed"] is False


def test_landing_survives_a_failed_focus_query(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    _focus_stub(monkeypatch, b, [RuntimeError("osascript exit 1")])
    res = b.click(x=1, y=1)
    assert res.ok and "focus unknown (osascript exit 1)" in res.message
    assert res.meta["focused"]["error"] == "osascript exit 1" and res.meta["front_app_changed"] is False


def test_every_mutating_action_carries_landing(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    _focus_stub(monkeypatch, b, [_focus_line()])
    results = [
        b.click(x=1, y=1), b.click(x=1, y=1, click_count=2), b.click(x=1, y=1, button="right"),
        b.click(x=1, y=1, modifiers=["cmd"]), b.drag(from_xy=(0, 0), to_xy=(9, 9)),
        b.scroll(direction="up"), b.type_text("a"), b.key("cmd+s"), b.key("tab"),
    ]
    for r in results:
        assert r.ok and "front_app" in r.meta and "focused" in r.meta and "→ Terminal" in r.message, r
    # Read-only actions do not.
    assert "front_app" not in b.move(1, 1).meta


# ── pre-type guard ─────────────────────────────────────────────────────


def test_type_refuses_on_finder_desktop_unless_forced(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    sent, _ = _focus_stub(monkeypatch, b, [FINDER_DESKTOP])
    res = b.type_text("testing")
    assert not res.ok and res.code == "focus_not_editable" and res.effect == "suspected_noop"
    assert "Finder is frontmost" in res.message and "force=true" in res.message
    assert "focus is Finder AXGroup" in res.message
    assert res.meta["front_app"]["bundle_id"] == "com.apple.finder" and res.meta["focused"]["role"] == "AXGroup"
    assert sent == []  # nothing typed
    res = b.type_text("testing", force=True)
    assert res.ok and [a for _, a in sent] == [["testing"]]


@pytest.mark.parametrize("role, allowed", [
    ("AXTextField", True), ("AXTextArea", True), ("AXComboBox", True), ("AXWebArea", True),
    ("AXGroup", True), ("AXSearchField", True), ("", True),  # unknown: opaque app, fail open
    ("AXButton", False), ("AXImage", False), ("AXOutline", False), ("AXList", False), ("AXStaticText", False),
])
def test_type_guard_by_role_outside_finder(monkeypatch, role, allowed):
    b = _backend_with_fake_display(monkeypatch)
    sent, _ = _focus_stub(monkeypatch, b, [_focus_line(app="Safari", bid="com.apple.Safari", role=role, vkind="")])
    res = b.type_text("hi")
    assert res.ok is allowed, (role, res.message)
    if not allowed:
        assert res.code == "focus_not_editable" and f"focused element is {role}" in res.message and sent == []


def test_finder_text_fields_still_accept_typing(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    for role in ("AXTextField", "AXSearchField", "AXComboBox"):
        _focus_stub(monkeypatch, b, [_focus_line(app="Finder", bid="com.apple.finder", role=role)])
        assert b.type_text("rename").ok, role
    # AXGroup/unknown are lenient elsewhere but not in Finder (desktop reports AXGroup).
    for role in ("AXGroup", "", "AXOutline", "AXImage"):
        _focus_stub(monkeypatch, b, [_focus_line(app="Finder", bid="com.apple.finder", role=role)])
        assert b.type_text("x").code == "focus_not_editable", role


def test_key_return_refused_on_finder_item_unless_forced(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    sent, _ = _focus_stub(monkeypatch, b, [FINDER_DESKTOP])
    for combo in ("return", "enter"):
        res = b.key(combo)
        assert not res.ok and res.code == "focus_not_editable" and "open or rename" in res.message, combo
    assert sent == []
    # Modified return, other keys, and forced return go through.
    assert b.key("cmd+return").ok and b.key("tab").ok and b.key("escape").ok
    assert b.key("return", force=True).ok
    assert any("key code 36" in script for script, _ in sent)
    # Return in a Finder text field (rename/search) is fine.
    _focus_stub(monkeypatch, b, [_focus_line(app="Finder", bid="com.apple.finder", role="AXTextField")])
    assert b.key("return").ok
    # Return outside Finder with a non-text focus (e.g. a default button) is fine.
    _focus_stub(monkeypatch, b, [_focus_line(app="Safari", bid="com.apple.Safari", role="AXButton")])
    assert b.key("return").ok


def test_set_value_is_unsupported():
    res = MacNativeBackend().set_value("x", element=1)
    assert not res.ok and res.code == "unsupported"


# ── tool-level wiring ──────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS backend")
def test_auto_backend_falls_back_to_native_without_cua_driver(monkeypatch):
    from tools.computer_use import tool

    monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
    monkeypatch.setattr("tools.computer_use.cua_backend.cua_driver_binary_available", lambda: False)
    monkeypatch.setattr(tool, "_configured_backend_name", lambda: "auto")
    assert tool._select_backend_name() == "macos"
    assert tool.check_computer_use_requirements() is True
    monkeypatch.setattr("tools.computer_use.cua_backend.cua_driver_binary_available", lambda: True)
    assert tool._select_backend_name() == "cua"


def test_explicit_env_backend_wins(monkeypatch):
    from tools.computer_use import tool

    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "noop")
    assert tool._select_backend_name() == "noop"


def test_config_can_disable_the_tool(monkeypatch):
    from tools.computer_use import tool

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"computer_use": {"enabled": False}})
    assert tool.check_computer_use_requirements() is False


def test_zoom_and_move_dispatch(monkeypatch, fake_screencapture):
    from tools.computer_use import tool

    b = _backend_with_fake_display(monkeypatch)
    calls = []
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: calls.append(list(argv or [])) or "ok")
    tool.reset_backend_for_tests()
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": b)
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "claude-code")
    shot = tool.handle_computer_use({"action": "capture", "mode": "vision"})
    assert isinstance(shot, dict) and shot["_multimodal"]
    assert shot["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    zoomed = tool.handle_computer_use({"action": "zoom", "region": [0, 0, 200, 100]})
    assert isinstance(zoomed, dict) and zoomed["_multimodal"]
    assert "zoom requires" in tool.handle_computer_use({"action": "zoom"})
    moved = tool.handle_computer_use({"action": "move", "coordinate": [10, 10]})
    assert calls[-1][0] == "move" and "pointer" in moved


# ── coordinate truth table (concrete numbers, no display) ─────────────


def _capture_with(monkeypatch, tmp_path, *, px, pts):
    """Backend whose fake screencapture returns a ``px``-sized PNG for a
    ``pts``-point main display; returns (backend, argv log)."""
    fixture = tmp_path / "fixture.png"
    fixture.write_bytes(_png(*px))
    log = tmp_path / "argv.log"
    stub = tmp_path / "screencapture"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{log}"\n'
        'for last; do :; done\n'
        f'cp "{fixture}" "$last"\n'
    )
    stub.chmod(0o755)
    monkeypatch.setenv("HERMES_SCREENCAPTURE_BIN", str(stub))
    monkeypatch.setenv("HERMES_SCREENSHOT_DIR", str(tmp_path / "shots"))
    b = MacNativeBackend()
    monkeypatch.setattr(b, "screen_points", lambda: pts)
    monkeypatch.setattr(b, "_front_window_info", lambda: {"app": "X"})
    b.capture(mode="vision")
    return b, log


@pytest.mark.parametrize(
    "px, pts, image, clicks",
    [
        # 2x Retina, downscaled: 2880x1800 px on a 1440x900 pt display →
        # 1568x980 image, 1.0889 px/pt. (784,450) is the image centre-ish →
        # (720, 413) pt, NOT (720, 450): y scales too.
        ((2880, 1800), (1440, 900), (1568, 980),
         [((784, 450), (720, 413)), ((0, 0), (0, 0)), ((1568, 980), (1440, 900)), ((1567, 979), (1439, 899))]),
        # Non-Retina 1440x900: nothing scales, pixels are points.
        ((1440, 900), (1440, 900), (1440, 900), [((784, 450), (784, 450)), ((1439, 899), (1439, 899))]),
        # MacBook "default" 1470x956 pt renders 2940x1912 px → 1568x1020.
        ((2940, 1912), (1470, 956), (1568, 1020), [((784, 510), (735, 478)), ((1568, 1020), (1470, 956))]),
        # 2x Retina that fits without downscale: 1280x800 pt / 2560x1600 px → 1568x980, 1.225 px/pt.
        ((2560, 1600), (1280, 800), (1568, 980), [((1568, 980), (1280, 800)), ((784, 490), (640, 400))]),
        # A small non-Retina display: 1024x768, untouched.
        ((1024, 768), (1024, 768), (1024, 768), [((512, 384), (512, 384))]),
    ],
)
def test_coordinate_truth_table(monkeypatch, tmp_path, px, pts, image, clicks):
    b, _log = _capture_with(monkeypatch, tmp_path, px=px, pts=pts)
    assert b._last_size == image
    for (ix, iy), (sx, sy) in clicks:
        assert b._to_points(ix, iy) == (sx, sy), f"{px}px/{pts}pt: image ({ix},{iy})"


def test_coordinates_refused_before_any_capture(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    calls = []
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: calls.append(argv) or "ok")
    with pytest.raises(ValueError, match="capture"):
        b.click(x=10, y=10)
    with pytest.raises(ValueError, match="capture"):
        b.move(10, 10)
    assert calls == []


def test_zoom_region_is_clamped_to_the_display(monkeypatch, tmp_path):
    # 1440x900 pt, image 1568 wide (1.0889 px/pt). A region hanging off the
    # bottom-right corner is clamped so the px/pt ratio matches what
    # screencapture actually returns (it crops silently).
    b, log = _capture_with(monkeypatch, tmp_path, px=(2880, 1800), pts=(1440, 900))
    b.zoom((1500, 900, 400, 300))
    assert log.read_text().splitlines()[-1].split()[2] == "1378,827,62,73"
    assert b._origin == (1378.0, 827.0)
    # And a region entirely off-screen degrades to a 1x1-pt sliver, not a crash.
    b._last_size = (1568, 980); b._ppp = 1568 / 1440; b._origin = (0.0, 0.0)
    b.zoom((5000, 5000, 10, 10))
    assert log.read_text().splitlines()[-1].split()[2] == "1439,899,1,1"


def test_screen_points_prefers_coregraphics_main_display(monkeypatch):
    b = MacNativeBackend()
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: "[1470,956]")
    monkeypatch.setattr(b, "_applescript", lambda *a, **k: pytest.fail("Finder should not be asked"))
    assert b.screen_points() == (1470, 956)
    # Finder fallback only when CoreGraphics is unavailable.
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setattr(b, "_applescript", lambda *a, **k: "0, 0, 1440, 900")
    assert b.screen_points() == (1440, 900)


def test_screenshot_dir_and_file_are_private(monkeypatch, tmp_path):
    b, _log = _capture_with(monkeypatch, tmp_path, px=(640, 480), pts=(640, 480))
    shots = tmp_path / "shots"
    assert stat.S_IMODE(shots.stat().st_mode) == 0o700
    assert list(shots.iterdir()) == []  # unlinked after read


def test_force_quit_and_lock_combos_are_hard_blocked(monkeypatch):
    from tools.computer_use import tool

    b = _backend_with_fake_display(monkeypatch)
    monkeypatch.setattr(b, "_applescript", lambda *a, **k: pytest.fail("must not reach osascript"))
    tool.reset_backend_for_tests()
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": b)
    for combo in ("cmd+option+esc", "cmd+alt+escape", "cmd-option-shift-escape", "ctrl+cmd+q", "cmd+shift+q"):
        out = tool.handle_computer_use({"action": "key", "keys": combo})
        assert "blocked key combo" in out, combo


def test_claude_code_profile_does_not_persist_screenshots(monkeypatch):
    from tools.computer_use import tool as cu
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "claude-code")
    assert cu._is_claude_code_profile()
    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE")
    assert not cu._is_claude_code_profile()


# ── tool-level landing / focused_element / force ───────────────────────


def _tool_with(monkeypatch, b):
    from tools.computer_use import tool
    tool.reset_backend_for_tests()
    monkeypatch.setattr(tool, "_get_backend", lambda session_id="": b)
    return tool


def test_tool_payload_hoists_landing_info(monkeypatch, fake_screencapture):
    import json
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    tool = _tool_with(monkeypatch, b)
    _focus_stub(monkeypatch, b, [_focus_line(), FINDER_DESKTOP])
    out = json.loads(tool.handle_computer_use({"action": "click", "coordinate": [100, 100]}))
    assert out["ok"] and out["front_app"]["name"] == "Finder" and out["front_app_changed"] is True
    assert out["front_app_before"]["name"] == "Terminal" and out["focused"]["role"] == "AXGroup"
    assert "front app changed: Terminal → Finder" in out["message"]
    _focus_stub(monkeypatch, b, [_focus_line(), _focus_line(vmatch="yes")])
    out = json.loads(tool.handle_computer_use({"action": "type", "text": "testing"}))
    assert out["ok"] and out["focused"]["value_ends_with_typed"] is True
    assert "value" not in out["focused"] and "testing" not in json.dumps(out["focused"])


def test_tool_type_guard_and_force_passthrough(monkeypatch):
    import json
    b = _backend_with_fake_display(monkeypatch)
    tool = _tool_with(monkeypatch, b)
    sent, _ = _focus_stub(monkeypatch, b, [FINDER_DESKTOP])
    out = json.loads(tool.handle_computer_use({"action": "type", "text": "testing"}))
    assert out["ok"] is False and out["code"] == "focus_not_editable" and out["verdict"]["decision"] == "escalate"
    assert out["front_app"]["bundle_id"] == "com.apple.finder" and out["focused"]["role"] == "AXGroup"
    assert sent == []
    out = json.loads(tool.handle_computer_use({"action": "type", "text": "testing", "force": True}))
    assert out["ok"] and [a for _, a in sent] == [["testing"]]
    out = json.loads(tool.handle_computer_use({"action": "key", "keys": "return"}))
    assert out["ok"] is False and out["code"] == "focus_not_editable"
    out = json.loads(tool.handle_computer_use({"action": "key", "keys": "return", "force": True}))
    assert out["ok"]


def test_force_is_dropped_for_backends_without_it():
    from tools.computer_use import tool

    def type_text(text, *, delivery_mode=None, bring_to_front=False): ...
    def type_text_kw(text, **kw): ...
    def type_text_force(text, *, force=False): ...
    assert tool._force_kw(type_text, {"force": True}) == {}
    assert tool._force_kw(type_text_kw, {"force": True}) == {"force": True}
    assert tool._force_kw(type_text_force, {"force": True}) == {"force": True}
    assert tool._force_kw(type_text_force, {}) == {}


def test_tool_focused_element_action_is_safe_and_valueless(monkeypatch):
    import json
    b = _backend_with_fake_display(monkeypatch)
    tool = _tool_with(monkeypatch, b)
    assert "focused_element" in tool._SAFE_ACTIONS
    sent, focus_argv = _focus_stub(monkeypatch, b, [_focus_line(subrole="AXStandardWindow", vmatch="yes")])
    out = json.loads(tool.handle_computer_use({"action": "focused_element"}))
    assert out == {"front_app": {"name": "Terminal", "bundle_id": "com.apple.Terminal", "pid": 100},
                   "focused": {"role": "AXTextArea", "subrole": "AXStandardWindow", "window_title": "moe — zsh"}}
    assert sent == [] and focus_argv == [[]]
    _focus_stub(monkeypatch, b, [RuntimeError("boom")])
    out = json.loads(tool.handle_computer_use({"action": "focused_element"}))
    assert out["error"] == "boom" and out["focused"]["role"] == ""


def test_tool_list_windows_carries_backend_note(monkeypatch):
    import json
    b = _backend_with_fake_display(monkeypatch)
    tool = _tool_with(monkeypatch, b)
    _cg_windows(monkeypatch, b, [_w("Safari", 5, 3, "Home", 10, 20, 800, 600)])
    out = json.loads(tool.handle_computer_use({"action": "list_windows"}))
    assert out["count"] == 1 and out["windows"][0]["frontmost"] and "z-ordered" in out["note"]


def test_schema_mentions_landing_info_and_new_knobs():
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA as s
    props = s["parameters"]["properties"]
    assert "focused_element" in props["action"]["enum"]
    assert props["force"]["type"] == "boolean" and "focus_not_editable" in props["force"]["description"]
    for needle in ("front_app", "focused", "front_app_changed", "check", "list_windows"):
        assert needle in s["description"], needle
