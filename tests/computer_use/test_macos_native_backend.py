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
    # Full-screen: no -R region flag.
    assert "-R" not in fake_screencapture.read_text()


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


def test_click_needs_coordinates_not_elements(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    res = b.click(element=3)
    assert not res.ok and res.code == "no_elements"
    with pytest.raises(ValueError):
        b.click(x=None, y=None)


def test_click_posts_scaled_point_via_jxa(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    calls = []
    monkeypatch.setattr(b, "_jxa", lambda script, argv=None: calls.append(list(argv or [])) or "ok")
    res = b.click(x=784, y=510, click_count=2)
    assert res.ok and res.path == "cgevent_fg" and res.effect == "unverifiable"
    assert calls == [["click", "735", "478", "2", "left"]]
    b.drag(from_xy=(0, 0), to_xy=(1568, 1020))
    assert calls[-1] == ["drag", "0", "0", "1470", "956"]
    b.scroll(direction="down", amount=4)
    assert calls[-1][0] == "scroll" and calls[-1][3:] == ["-4", "0"]


def test_type_and_key_pass_text_as_argv(monkeypatch):
    b = _backend_with_fake_display(monkeypatch)
    calls = []
    monkeypatch.setattr(b, "_applescript", lambda script, argv=None: calls.append((script, list(argv or []))) or "")
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
    calls = []
    monkeypatch.setattr(b, "_applescript", lambda script, argv=None: calls.append(argv) or "")
    monkeypatch.setattr(mnb.time, "sleep", lambda s: None)
    b.type_text("x" * 120)
    assert [len(a[0]) for a in calls] == [50, 50, 20]


def test_list_windows_reports_front_window_in_both_spaces(monkeypatch, fake_screencapture):
    b = _backend_with_fake_display(monkeypatch)
    b.capture(mode="vision")
    (win,) = b.list_windows()
    assert win["app"] == "Safari" and win["bounds_points"] == {"x": 10, "y": 20, "w": 800, "h": 600}
    assert win["bounds_screenshot_px"]["w"] == round(800 * 1568 / 1470)


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
