"""macOS Spaces: a window on another desktop is findable, and says so.

Regression test for the 2026-09-03 meeting-link failure. ``capture(app=...)``
returned ``<no on-screen window matched app='Calendar'>`` for a Calendar window
that was open the whole time, one Space away, because ``_load_windows()`` asks
cua-driver for ``on_screen_only: True`` — documented as "drop windows not on
the current Space". The old message blamed *localized app names*, so the agent
spent ninety seconds re-spelling "Calendar" and calling ``focus_app`` /
``bring_to_front``, which switch Spaces and dragged the user's desktop around.

Every assertion here checks a **side effect** — which records come back, and
what the text actually says — never merely that a call was allowed.

Shape of the real payload, measured on macOS 26.5 with 5 Spaces (137 layer-0
windows): 3 records ``on_current_space=True``, 16 ``False``, 118 ``None`` — and
all 118 of those were also ``is_on_screen=False``.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend, _ingest_windows


def _raw(window_id: int, app: str, *, on_current, on_screen=True,
         spaces=None, pid: int = 7253, z: int = 10) -> Dict[str, Any]:
    return {
        "window_id": window_id, "pid": pid, "app_name": app, "title": "",
        "z_index": z, "is_on_screen": on_screen,
        "on_current_space": on_current, "space_ids": spaces,
    }


# The incident, reduced: Calendar sat on Space 1, Chrome came up on the active
# Space, and 118 hidden backing windows carried no Space information at all.
INCIDENT = [
    _raw(143441, "Terminal", on_current=True, spaces=[2915], z=30),
    _raw(109557, "Google Chrome", on_current=True, spaces=[2915], z=20),
    _raw(122663, "Calendar", on_current=False, spaces=[1], on_screen=False, z=5),
    _raw(122680, "Calendar", on_current=None, spaces=None, on_screen=False, z=1),
    _raw(122679, "Calendar", on_current=None, spaces=None, on_screen=False, z=1),
]


class _Backend(CuaDriverBackend):
    """Only the two seams the code under test uses. No driver, no screen."""

    def __init__(self, raw: List[Dict[str, Any]], *, all_spaces_fails=False):
        self._raw = raw
        self._all_spaces_fails = all_spaces_fails

    def _load_windows_all_spaces_checked(self):
        if self._all_spaces_fails:
            return [], True
        return _ingest_windows(self._raw), False

    def _load_windows_all_spaces(self):
        return self._load_windows_all_spaces_checked()[0]

    def _load_windows(self):
        # What the driver returns for on_screen_only=True: current Space only.
        return _ingest_windows(
            [w for w in self._raw if w.get("on_current_space") is True]
        )


def test_ingest_carries_space_metadata_and_never_invents_it():
    ws = {w["window_id"]: w for w in _ingest_windows(INCIDENT)}
    assert ws[122663]["on_current_space"] is False
    assert ws[122663]["space_ids"] == [1]
    assert ws[109557]["on_current_space"] is True
    # null must stay null: on Linux/Windows the driver says nothing about
    # Spaces, and reading that as "off-Space" would break every non-mac host.
    assert ws[122680]["on_current_space"] is None
    assert ws[122680]["space_ids"] == []


def test_list_windows_surfaces_the_off_space_window_the_agent_could_not_find():
    out = _Backend(INCIDENT).list_windows()
    ids = [w["window_id"] for w in out]
    assert 122663 in ids, "the Calendar window on Space 1 must be listed"
    # ...and the old, Space-scoped view is exactly what hid it.
    assert 122663 not in [w["window_id"] for w in _Backend(INCIDENT)._load_windows()]


def test_list_windows_drops_hidden_backing_windows_but_keeps_off_space_ones():
    out = _Backend(INCIDENT).list_windows()
    ids = [w["window_id"] for w in out]
    # 118 of 137 real records were is_on_screen=False with no Space; listing
    # them buries the signal under two orders of magnitude of noise.
    assert 122680 not in ids and 122679 not in ids
    assert len(out) == 3


def test_reachable_windows_sort_ahead_of_off_space_ones():
    out = _Backend(INCIDENT).list_windows()
    reachable = [w for w in out if w["on_current_space"] is not False]
    off = [w for w in out if w["on_current_space"] is False]
    assert [w["window_id"] for w in out] == \
        [w["window_id"] for w in reachable] + [w["window_id"] for w in off]
    # frontmost first within the reachable tier
    assert out[0]["window_id"] == 143441


def test_list_windows_falls_back_rather_than_returning_nothing():
    out = _Backend(INCIDENT, all_spaces_fails=True).list_windows()
    assert [w["window_id"] for w in out] == [143441, 109557]


def test_off_space_message_names_the_cause_and_the_recovery():
    msg = _Backend(INCIDENT)._no_match_reason("Calendar")
    assert "another macOS Space" in msg
    assert "window_id=122663" in msg and "pid=7253" in msg
    assert "space_ids=[1]" in msg
    # the recovery the incident agent took 90s to stumble onto
    assert "capture(window_id=122663, pid=7253" in msg
    # and it must actively stop the behaviour that wasted those 90s
    assert "NOT a naming problem" in msg
    assert "localized" not in msg.lower()


def test_off_space_message_does_not_claim_reading_is_impossible():
    """cua-driver: "Observation (capture/list_windows) remains available".

    An earlier draft of this message said an off-Space window "cannot be
    captured or clicked from here" and then recommended capturing it — the
    agent that believes the first clause skips the recovery and reaches for
    focus_app, yanking the user's desktop to another Space mid-work. Only
    *input* is refused off-Space.
    """
    msg = _Backend(INCIDENT)._no_match_reason("Calendar")
    assert "cannot be captured" not in msg
    assert "READ it where it is" in msg
    assert "CANNOT do is send it input" in msg
    # som returns an empty element list off-Space (degraded_reason:
    # ax_window_unresolved), so the advice must name the mode that works.
    assert "mode='vision'" in msg


def test_absent_app_still_gets_the_naming_advice_without_overclaiming():
    msg = _Backend(INCIDENT)._no_match_reason("Xcode")
    assert "no window matched" in msg
    assert "list_apps" in msg
    assert "another macOS Space" not in msg
    # discovery is known to come back empty intermittently, so this branch
    # must not assert as fact that nothing is running
    assert "appears to be running" not in msg
    assert "came back empty this call" in msg


def test_running_but_fully_offscreen_is_its_own_message():
    hidden = [_raw(500, "Notes", on_current=None, spaces=None, on_screen=False)]
    msg = _Backend(hidden)._no_match_reason("Notes")
    assert "none of its" in msg and "on-screen" in msg
    assert "minimized_or_hidden_window" in msg
    assert "another macOS Space" not in msg


def test_minimized_on_this_space_ranks_below_a_live_window():
    """`on_current_space: true` is not the reachability test.

    A minimized window is on the current Space and still refuses input
    (`minimized_or_hidden_window`). Ranking it beside a live window teaches
    the model the same wrong reachability test this whole change exists to
    correct.
    """
    ws = [
        _raw(1, "Notes", on_current=True, on_screen=False, spaces=[2915], z=0),
        _raw(2, "Finder", on_current=True, on_screen=True, spaces=[2915], z=5),
        _raw(3, "Chrome", on_current=False, on_screen=False, spaces=[1], z=9),
    ]
    out = _Backend(ws).list_windows()
    assert [w["window_id"] for w in out] == [2, 1, 3]


def test_space_ids_keeps_string_ids_and_rejects_bools():
    ws = _ingest_windows([
        _raw(1, "A", on_current=False, spaces=["main", 4]),
        _raw(2, "B", on_current=False, spaces=[True, 7]),
    ])
    assert ws[0]["space_ids"] == ["main", 4]
    # isinstance(True, int) is True; a JSON `true` must not become Space 1
    assert ws[1]["space_ids"] == [7]


def test_current_space_id_is_ingested_so_space_ids_are_anchored():
    raw = _raw(1, "A", on_current=False, spaces=[4])
    raw["current_space_id"] = 2915
    assert _ingest_windows([raw])[0]["current_space_id"] == 2915


def test_empty_desktop_does_not_trigger_the_twenty_second_cli_refetch():
    """An honest empty and a dropped session both look like []; only the
    second earns a fallback that logs a warning and re-fetches over the CLI."""
    calls = []

    class Counting(_Backend):
        def _load_windows_all_spaces_checked(self):
            return [], False          # empty, but the fetch worked

        def _load_windows(self):
            calls.append("fallback")
            return []

    assert Counting([]).list_windows() == []
    assert calls == [], "an empty desktop must not fall back"


def test_transport_failure_still_falls_back():
    calls = []

    class Failing(_Backend):
        def _load_windows_all_spaces_checked(self):
            return [], True           # the fetch itself failed

        def _load_windows(self):
            calls.append("fallback")
            return _ingest_windows(INCIDENT[:1])

    assert [w["window_id"] for w in Failing([]).list_windows()] == [143441]
    assert calls == ["fallback"]


def test_diagnosis_failure_cannot_mask_the_original_error():
    class Boom(_Backend):
        def _load_windows_all_spaces(self):
            raise RuntimeError("driver session dropped")

    msg = Boom(INCIDENT)._no_match_reason("Calendar")
    assert "no window matched" in msg  # degrades to the safe text, never raises


def test_schema_tells_the_model_about_on_current_space():
    from tools.computer_use.schema import get_computer_use_schema

    desc = get_computer_use_schema()["description"]
    assert "on_current_space" in desc
    assert "other macOS Spaces" in desc
    # must not repeat the false claim that off-Space windows cannot be read
    assert "can only reach" not in desc
    assert "input is refused" in desc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
