"""A plain scroll has to name a window, or it cannot scroll a browser.

Measured 2026-09-06 on the owner's Mac, asking Moe to scroll a Claude
conversation in Chrome:

    scroll  app="Google Chrome"        -> refused, ambiguous_window_target
      "pid 773 owns more than one eligible top-level window; provide window_id."
    scroll  pid=773  window_id=175     -> refused, THE IDENTICAL ERROR

The model did exactly what the error asked and got the same answer, because
`backend.scroll` had no `window_id` parameter and the dispatcher passed none.
The refusal named the one thing the tool could not express. It tried twice,
then gave up on scrolling and read the conversation out of the saved element
JSON instead.

Chrome is where this bites first: it owns a fistful of offscreen helper
windows, so "more than one eligible top-level window" is true on the first
try. The id was known the whole time — `capture()` records
`_active_window_id`.
"""

from __future__ import annotations

import sys

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend


class _Backend(CuaDriverBackend):
    """Records the args that would go to the driver, and goes no further."""

    def __init__(self, active_pid=773, active_window=175):
        self._active_pid = active_pid
        self._active_window_id = active_window
        self._snapshot_tokens = {}
        self._snapshot_id = None
        self.sent = None

        class _S:
            def supports_capability(_s, cap, tool=None):
                return False
        self._session = _S()

    def _action(self, name, args, **kw):
        self.sent = (name, dict(args))
        return type("R", (), {"ok": True, "action": name})()


def test_a_plain_scroll_carries_the_captured_window():
    b = _Backend()
    b.scroll(direction="down", amount=6)
    name, args = b.sent
    assert name == "scroll"
    assert args["window_id"] == 175, "this is the whole bug"
    assert args["pid"] == 773


def test_an_explicit_window_id_wins():
    """The model passed one and was ignored. Whatever else changes, the
    argument a caller actually supplied has to be the one that is used."""
    b = _Backend(active_window=175)
    b.scroll(direction="down", amount=3, window_id=4246)
    assert b.sent[1]["window_id"] == 4246


def test_no_window_anywhere_still_scrolls_by_pid():
    """A single-window app never needed one, and must not start failing."""
    b = _Backend(active_window=None)
    b.scroll(direction="up", amount=2)
    assert "window_id" not in b.sent[1]
    assert b.sent[1]["pid"] == 773


def test_the_amount_is_still_clamped():
    b = _Backend()
    b.scroll(direction="down", amount=9999)
    assert b.sent[1]["amount"] == 50


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
