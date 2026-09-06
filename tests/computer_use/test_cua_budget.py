"""A turn budget: the check that fires on work which looks fine.

The two stall detectors read the SHAPE of what has been tried — the same call
twice, or ten looks without acting. Neither can see the worst failure we have
measured, because every step of it was different and every step was
defensible.

2026-09-06, asked to draw something. Five minutes, seventeen calls: hunt the
environment for image-generation API keys, read a skill, list windows, probe
for `rsvg-convert`, capture, probe again, write an HTML page, write an
AppleScript, launch Chrome on it, capture Chrome, list windows, capture — then
conclude the target tab was on another Space anyway. No repetition. No looking
loop. The owner watched "still running" for five minutes.

Astra's computer-use loop stops at twenty turns and says so: "The task reached
the 20-response limit." The number is the whole point — a limit that only
fired on obviously-bad work would not have fired here either.
"""

from __future__ import annotations

import json
import sys

import pytest

from tools.computer_use.stall import (
    BUDGET_HARD, BUDGET_IDLE_RESET, BUDGET_SOFT, StallDetector,
)


def _spend(d: StallDetector, n: int, action: str = "click", t0: float = 1000.0):
    """n calls at one-second intervals, returning the last refusal (or None)."""
    out = None
    for i in range(n):
        out = d.budget_reason(action, now=t0 + i)
    return out


def test_the_budget_allows_a_real_task_through():
    d = StallDetector()
    assert _spend(d, BUDGET_HARD) is None, "twenty actions is a task, not a loop"


def test_the_call_after_the_limit_is_refused():
    d = StallDetector()
    _spend(d, BUDGET_HARD)
    out = d.budget_reason("click", now=1100.0)
    assert out is not None
    body = json.loads(out)
    assert body["code"] == "budget_spent"
    assert body["verdict"]["decision"] == "stop_and_report"
    assert str(BUDGET_HARD) in body["error"]


def test_the_refusal_asks_for_the_obstacle_by_name():
    """"It didn't work" is what the owner got for five minutes. The whole
    value of stopping is the account that comes with it."""
    d = StallDetector()
    _spend(d, BUDGET_HARD + 1)
    # Inside the idle window on purpose — a later timestamp would refill the
    # allowance and hand back None, which is what this test did first.
    hint = json.loads(d.budget_reason("click", now=1030.0))["verdict"]["hint"]
    assert "naming the specific obstacle" in hint
    assert "ask whether to keep going" in hint


def test_varied_plausible_calls_are_counted_the_same_as_repeats():
    """The regression this exists for. Every action different — exactly the
    trace above — and the budget still runs out."""
    d = StallDetector()
    actions = ["capture", "click", "type", "list_windows", "invoke_menu"]
    out = None
    for i in range(BUDGET_HARD + 1):
        out = d.budget_reason(actions[i % len(actions)], now=1000.0 + i)
    assert out is not None, "a budget that only catches repetition is the detector we already had"


def test_retrying_straight_away_is_refused_again():
    """The refusal only means something if it forces the model to speak. An
    immediate second attempt must not buy a fresh allowance."""
    d = StallDetector()
    _spend(d, BUDGET_HARD + 1)
    assert d.budget_reason("click", now=1030.0) is not None


def test_the_allowance_refills_after_a_real_gap():
    """...and it must refill, or the session is over for good. The gap stands
    in for handing back to the person, which the tool cannot see."""
    d = StallDetector()
    _spend(d, BUDGET_HARD + 1)
    assert d.budget_reason("click", now=1000.0 + BUDGET_IDLE_RESET + 60) is None


def test_waiting_and_scrolling_are_not_spent():
    """Actions whose whole point is repetition are exempt everywhere else in
    this module; the budget must not be the one place they cost."""
    d = StallDetector()
    for i in range(50):
        assert d.budget_reason("wait", now=1000.0 + i) is None
    assert d.spent() == 0


def test_the_advisory_arrives_late_enough_to_be_useful():
    d = StallDetector()
    _spend(d, BUDGET_SOFT - 1)
    assert d.budget_advisory() is None, "'nineteen left' on every call is noise"
    d.budget_reason("click", now=1000.0 + BUDGET_SOFT)
    note = d.budget_advisory()
    assert note is not None
    assert f"{BUDGET_HARD - BUDGET_SOFT} screen action(s) left" in note["budget"]["note"]


def test_the_numbers_are_astras():
    assert BUDGET_HARD == 20
    assert BUDGET_SOFT == 14


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ------------------------------------------------- the wiring, not the rule
#
# Everything above tests StallDetector directly, and all of it passed with the
# call in tool.py disabled — 285 green against a budget nothing consulted.
# That is the "check that cannot fail" this repo keeps writing down: a rule and
# its caller are two things, and only one of them was under test.

def test_the_tool_actually_consults_the_budget(monkeypatch):
    """Drive handle_computer_use past the limit and require a refusal.

    Refusal happens before any backend work — that is the design, so that a
    spent budget never reaches the approval gate — which is what makes this
    testable without a driver or a screen.
    """
    from tools.computer_use import tool as t

    monkeypatch.setattr(t, "_stall_detectors", {}, raising=False)

    out = None
    for _ in range(BUDGET_HARD + 1):
        out = t.handle_computer_use({"action": "list_windows"},
                                    session_id="budget-wiring")
    body = json.loads(out)
    assert body.get("code") == "budget_spent", (
        "the tool ran a 21st action; the budget is not wired in"
    )


def test_a_different_session_has_its_own_allowance(monkeypatch):
    """One person's long task must not spend another session's budget."""
    from tools.computer_use import tool as t

    monkeypatch.setattr(t, "_stall_detectors", {}, raising=False)
    for _ in range(BUDGET_HARD + 1):
        t.handle_computer_use({"action": "list_windows"}, session_id="one")
    out = t.handle_computer_use({"action": "list_windows"}, session_id="two")
    assert json.loads(out).get("code") != "budget_spent"
