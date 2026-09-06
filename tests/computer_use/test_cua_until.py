"""`until` — a batch that repeats until a predicate holds.

The round-trip lever. Astra's version of this is exec_py/exec_js: the model
writes real code with real loops, and OpenAI recommend it over their
structured computer tool precisely because one call can carry a loop.

We are deliberately not doing that, and the reason is specific rather than
timid. Every control in this stack inspects what an action DOES — the send
gate flattens a `steps` batch and judges each one, the approval prompt names
the recipient, both stall detectors read the action and its arguments. A
script is opaque to all of them. Three times this month a new verb walked past
a guard that enumerated the old ones (invoke_menu, then steps, then
`osascript … activate`), and an arbitrary code executor pointed at somebody's
screen is that mistake with the volume turned up.

So: the same actions, repeated, with a predicate deciding when to stop.
"Scroll until the Join button exists" is one round-trip instead of eight, and
every step in it is still a step every existing control can read.
"""

from __future__ import annotations

import json
import sys

import pytest

from tools.computer_use import tool as t
from tools.computer_use.tool import _MAX_ROUNDS, handle_computer_use


# A session of this test's own, and a detector map of its own.
#
# Shared module state is how a green file makes a red suite: `_stall_detectors`
# is a module-level dict keyed by session, and every test here that shares the
# default session also shares its budget and its history. Under pytest-randomly
# that turned into fifteen errors in OTHER files, and the suite passed in fixed
# order — which is the worst way to find out.
@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(t, "_stall_detectors", {}, raising=False)


def _run(**kw):
    return json.loads(handle_computer_use({"action": "steps", **kw},
                                          session_id="until-tests"))


@pytest.fixture
def screen(monkeypatch):
    """A screen where the predicate becomes true after N scrolls."""
    state = {"scrolls": 0, "appears_after": 3, "calls": []}
    real = t.handle_computer_use

    def router(args, **kwargs):
        act = (args.get("action") or "").strip().lower()
        if act == "steps":
            return real(args, **kwargs)
        state["calls"].append(act)
        if act == "scroll":
            state["scrolls"] += 1
            return json.dumps({"ok": True, "action": "scroll"})
        if act == "verify_state":
            met = state["scrolls"] >= state["appears_after"]
            return json.dumps({"ok": True, "action": "verify_state",
                               "verified": True if met else None})
        return json.dumps({"ok": True, "action": act})

    monkeypatch.setattr(t, "handle_computer_use", router)
    return state


def test_it_repeats_until_the_predicate_holds(screen):
    out = _run(steps=[{"action": "scroll", "direction": "down"}],
               until=[{"element": {"selector": {"label_contains": "Join"},
                                   "exists": True}}],
               max_rounds=8)
    assert out["ok"] is True
    assert out["until_met"] is True
    assert out["rounds"] == 3, "stops the moment it is satisfied"
    assert screen["scrolls"] == 3


def test_running_out_of_rounds_is_a_failure_that_says_do_not_just_raise_it(screen):
    screen["appears_after"] = 99
    out = _run(steps=[{"action": "scroll", "direction": "down"}],
               until=[{"element": {"selector": {"label_contains": "Join"},
                                   "exists": True}}],
               max_rounds=3)
    assert out["ok"] is False and out["code"] == "until_unmet"
    assert out["rounds"] == 3
    assert "not simply raise max_rounds" in out["verdict"]["hint"]


def test_unknown_is_not_satisfied(screen):
    """verify_state answers satisfied / unsatisfied / unknown, and the batch
    must not stop on the third — the absence of evidence that something
    appeared is not evidence that it did."""
    screen["appears_after"] = 99          # verified stays None
    out = _run(steps=[{"action": "scroll", "direction": "down"}],
               until=[{"element": {"selector": {"label_contains": "x"},
                                   "exists": True}}],
               max_rounds=2)
    assert out["until_met"] is False


def test_a_failed_step_still_halts_on_a_later_round(screen):
    """Halt-on-first-failure means the same thing on round four as on round
    one: a later step assumed the earlier one landed."""
    calls = {"n": 0}
    real_router = t.handle_computer_use

    def flaky(args, **kwargs):
        act = (args.get("action") or "").strip().lower()
        if act == "scroll":
            calls["n"] += 1
            if calls["n"] == 2:
                return json.dumps({"ok": False, "action": "scroll"})
        return real_router(args, **kwargs)

    t.handle_computer_use = flaky
    try:
        out = _run(steps=[{"action": "scroll", "direction": "down"}],
                   until=[{"element": {"selector": {"label_contains": "z"},
                                       "exists": True}}],
                   max_rounds=5)
    finally:
        t.handle_computer_use = real_router
    assert out["ok"] is False
    assert out["failed_step"]["action"] == "scroll"


# ------------------------------------------------- what may not be repeated

def test_a_send_cannot_be_put_in_a_loop():
    """The gate approves a batch ONCE. A Return inside a looping batch would
    be ten messages sent on one 'yes' — the exact hole the gate exists for."""
    out = _run(steps=[{"action": "key", "keys": "return"}],
               until=[{"window": {"exists": False}}], max_rounds=5)
    assert out["code"] == "bad_steps"
    assert "return/enter is what sends" in out["error"]


@pytest.mark.parametrize("action", ["click", "type", "invoke_menu", "drag"])
def test_only_navigation_may_repeat(action):
    out = _run(steps=[{"action": action}],
               until=[{"window": {"exists": True}}], max_rounds=4)
    assert out["code"] == "bad_steps"
    assert "cannot be repeated" in out["error"]


def test_arrow_keys_may_repeat(screen):
    out = _run(steps=[{"action": "key", "keys": "pagedown"}],
               until=[{"window": {"exists": True}}], max_rounds=2)
    assert out.get("code") != "bad_steps"


def test_a_single_round_is_unrestricted(screen):
    """max_rounds defaults to 1, and a plain batch keeps every action it had.
    The restriction is about repetition, not about batching."""
    out = _run(steps=[{"action": "click", "coordinate": [1, 2]},
                      {"action": "key", "keys": "return"}])
    assert out["ok"] is True and out.get("rounds") == 1


# ------------------------------------------------------------- the guardrails

def test_repeating_with_no_stopping_condition_is_refused():
    out = _run(steps=[{"action": "scroll", "direction": "down"}], max_rounds=5)
    assert out["code"] == "bad_steps"
    assert "loop with no exit" in out["error"]


@pytest.mark.parametrize("bad", [0, -1, _MAX_ROUNDS + 1, "lots", None])
def test_max_rounds_is_bounded(bad):
    out = _run(steps=[{"action": "scroll"}],
               until=[{"window": {"exists": True}}], max_rounds=bad)
    assert out["code"] == "bad_steps"


@pytest.mark.parametrize("bad", [[], "soon", {}, 3])
def test_until_must_be_a_predicate_list(bad):
    out = _run(steps=[{"action": "scroll"}], until=bad, max_rounds=2)
    assert out["code"] == "bad_steps"


def test_the_cap_is_ten():
    assert _MAX_ROUNDS == 10


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
