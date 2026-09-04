"""Looking without acting — the failure the repeat detector cannot see.

Measured from a real session, 2026-09-04 20:27-20:37, drawing a picture:

    33 calls over 578 s
    tool time      80 s  (14%)
    model thinking 498 s (86%)
    22 captures vs 5 actions
    longest run of looking with nothing done: SEVENTEEN

Nothing failed. There was no error to react to and no identical call to
detect — every screenshot differs, so no two fingerprint alike and the
repeated-call detector is blind to it. The owner called it "very, very slow"
and "stuck in a couple of places", and both were true: it was stuck looking.

This matches the published shape (OSWorld-Human: planning and reflection
75-94% of a task's wall clock, grounding 2-4%), which is why the lever is
step count rather than faster clicking.
"""

from __future__ import annotations

import json
import sys

import pytest

from tools.computer_use.stall import (
    LOOK_HARD_LIMIT,
    LOOK_SOFT_LIMIT,
    StallDetector,
)

OK = json.dumps({"ok": True})


def _look(det, n, action="capture", args=None):
    """n reads in a row; returns the verdicts they produced."""
    out = []
    for i in range(n):
        out.append(det.looking_loop(action))
        det.record(action, args or {"app": f"App{i}"}, json.dumps({"i": i}))
    return out


# --------------------------------------------------------- the real session

def test_the_seventeen_look_run_is_caught():
    det = StallDetector()
    verdicts = _look(det, 17)
    advised = next(i for i, v in enumerate(verdicts) if v) + 1
    refused = next(i for i, v in enumerate(verdicts) if v and v["refused"]) + 1
    assert advised == LOOK_SOFT_LIMIT, f"first word at look {advised}"
    assert refused == LOOK_HARD_LIMIT, f"first refusal at look {refused}"


def test_a_differing_screenshot_defeats_the_repeat_detector_but_not_this_one():
    """Why this exists at all: each capture returns different pixels, so the
    identical-call detector never fires."""
    det = StallDetector()
    for i in range(12):
        args = {"app": "Freeform", "mode": "som"}
        # different result every time — a real screen
        det.record("capture", args, json.dumps({"total_elements": i}))
        assert det.block_reason("capture", args) is None, "repeat detector stays quiet"
    assert det.looking_loop("capture")["refused"] is True


def test_the_advice_names_what_to_do_instead():
    det = StallDetector()
    v = _look(det, LOOK_SOFT_LIMIT)[-1]
    assert "invoke_menu" in v["hint"]
    assert "mac.sh menus" in v["hint"]
    assert "tell the user" in v["hint"]


# ------------------------------------------------------- acting resets it

def test_doing_anything_clears_the_run():
    det = StallDetector()
    _look(det, LOOK_SOFT_LIMIT - 1)
    det.record("click", {"coordinate": [1, 2]}, OK)
    assert det.looks_without_acting() == 0
    assert det.looking_loop("capture") is None


@pytest.mark.parametrize("act", ["click", "invoke_menu", "type", "focus_app",
                                 "drag", "set_value", "double_click"])
def test_every_acting_verb_counts_as_progress(act):
    det = StallDetector()
    _look(det, 8)
    det.record(act, {}, OK)
    assert det.looks_without_acting() == 0


def test_the_real_sessions_mix_never_trips_until_it_stops_acting():
    """The first half of that session interleaved looks with clicks and menus
    and must stay silent; only the tail run is a problem."""
    det = StallDetector()
    opening = [
        "capture", "list_windows", "capture", "capture", "capture",
        "focus_app", "click", "capture", "invoke_menu", "focus_app",
        "focus_app", "invoke_menu", "capture", "click", "capture", "click",
    ]
    for a in opening:
        assert det.looking_loop(a) is None or not det.looking_loop(a)["refused"]
        det.record(a, {}, OK)
    assert det.looks_without_acting() <= 1


# ------------------------------------------------------------- only reads

@pytest.mark.parametrize("act", ["click", "type", "invoke_menu", "drag"])
def test_acting_is_never_refused_by_this(act):
    """Refusing a look is safe. Refusing an action is not, and an agent that
    has looked ten times is not short of pixels."""
    det = StallDetector()
    _look(det, 20)
    assert det.looking_loop(act) is None


def test_verify_state_counts_as_looking():
    # It answers a question about the screen; it changes nothing.
    det = StallDetector()
    for _ in range(LOOK_HARD_LIMIT - 1):
        det.record("verify_state", {}, OK)
    assert det.looking_loop("verify_state")["refused"] is True


def test_a_short_look_is_never_mentioned():
    det = StallDetector()
    assert all(v is None for v in _look(det, LOOK_SOFT_LIMIT - 1))


def test_the_dispatch_refuses_the_read_and_says_why():
    from tools.computer_use.tool import handle_computer_use

    from tools.computer_use import tool as t
    det = t._stall_detector("looking-test")
    for _ in range(LOOK_HARD_LIMIT):
        det.record("capture", {}, OK)
    out = json.loads(handle_computer_use({"action": "capture", "app": "X"},
                                         session_id="looking-test"))
    assert out["code"] == "looking_without_acting"
    assert out["verdict"]["decision"] == "stop_and_report"
    t.release_computer_use_session("looking-test")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
