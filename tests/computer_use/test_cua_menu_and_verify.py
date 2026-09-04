"""invoke_menu and verify_state — two capabilities cua-driver always had.

`invoke_menu` resolves an application-menu path through the accessibility API
and invokes the final item. Clicking a menu is five fragile steps (capture,
ground, click to open, capture again because the menu did not exist a moment
ago, click again); this is one, and its contract is that missing, ambiguous,
disabled or structurally mismatched segments fail closed — it never falls back
to pixels. It also reaches where pixels barely do: a DAW, CAD or 3D app draws
its canvas as one custom surface with no AX tree, while its menu bar stays
standard and fully addressable.

`verify_state` evaluates bounded predicates against live accessibility state
with consecutive stable samples — the "did that actually happen?" the
2026-09-03 incident had no way to ask.

Measured against cua-driver 0.23.2 before writing these, and several of these
tests exist because the first draft got the measurement wrong:

  * a refused invoke_menu returns `isError: true` AND
    `structuredContent.status == "refused"` — an earlier comment claimed there
    was no transport error, which was false;
  * a SUCCESSFUL invoke_menu carries no `status` field at all (ActionResult is
    {effect, route, delivery, evidence, escalation}), so a success branch
    keyed on `status == "invoked"` was dead code against a word the driver
    never emits;
  * a window-exists predicate returns `satisfied` in 18 ms, against roughly
    600 KB and several seconds for the SOM capture it replaces.
"""

from __future__ import annotations

import json
import sys

import pytest

from tools.computer_use.backend import ActionResult
from tools.computer_use.cua_backend import _apply_driver_status
from tools.computer_use.schema import get_computer_use_schema
from tools.computer_use.stall import call_fingerprint
from tools.computer_use.tool import (
    _DESTRUCTIVE_ACTIONS,
    _SAFE_ACTIONS,
    _dispatch,
    _summarize_action,
    _verify_payload,
    handle_computer_use,
)


def _res(name, meta, ok=True, **kw):
    return ActionResult(ok=ok, action=name, message="", meta=meta, **kw)


# ---------------------------------------------------------- refusal honesty

def test_a_refusal_is_a_failure_with_the_drivers_own_code():
    res = _apply_driver_status(_res(
        "invoke_menu",
        {"status": "refused",
         "refusal": {"code": "menu_path_unavailable",
                     "message": "invoke_menu: path segment 1 was not found"}},
        ok=False))
    assert res.ok is False
    assert res.effect == "suspected_noop"
    assert res.code == "menu_path_unavailable"
    assert "segment 1" in res.message


def test_a_refusal_object_alone_is_enough_even_without_a_status_word():
    res = _apply_driver_status(_res("invoke_menu",
                                    {"refusal": {"code": "x"}}, ok=True))
    assert res.ok is False and res.effect == "suspected_noop"


def test_a_plain_success_is_left_exactly_as_the_driver_described_it():
    """A successful invoke_menu carries no `status`. The mapper must not
    invent one — an earlier draft synthesised `verified: True` from a status
    word, and `_classify_action_result` reads `verified is True` BEFORE
    effect, so a result whose own effect said `unverifiable` came back as
    `decision: done`."""
    res = _apply_driver_status(_res("invoke_menu", {"effect": "unverifiable"},
                                    effect="unverifiable"))
    assert res.verified is None, "verified means the driver read it back"
    assert res.effect == "unverifiable"
    assert res.ok is True


def test_the_drivers_own_effect_always_wins():
    res = _apply_driver_status(_res("invoke_menu",
                                    {"status": "refused", "effect": "partial"},
                                    effect="partial"))
    assert res.effect == "partial" and res.ok is False


def test_meta_that_is_not_a_dict_does_not_crash_the_mapper():
    r = ActionResult(ok=True, action="invoke_menu", message="", meta=None)
    assert _apply_driver_status(r).ok is True


# ------------------------------------------- the predicate verdict, its own

def test_satisfied_is_proof_and_says_no_screenshot_is_needed():
    p = _verify_payload(_res("verify_state", {"status": "satisfied"}))
    assert p["status"] == "satisfied" and p["verified"] is True
    assert p["verdict"]["decision"] == "confirmed"


def test_unsatisfied_never_tells_the_model_to_re_issue_input():
    """Routing this through the shared action payload mapped `unsatisfied`
    onto `suspected_noop`, whose hint says "the input likely did not land —
    re-issue by coordinate, or with delivery_mode='foreground'". For a
    read-only check that answered "no", that is an instruction to click
    somewhere and pull the user's desktop to another Space, for an action
    that never happened."""
    p = _verify_payload(_res("verify_state", {"status": "unsatisfied"}))
    assert p["verified"] is False
    assert p["verdict"]["decision"] == "not_yet"
    hint = p["verdict"]["hint"]
    assert "nothing was sent" in hint
    for forbidden in ("suspected_noop", "delivery_mode", "coordinate",
                      "escalate", "re-issue by"):
        assert forbidden not in hint, forbidden
    assert "effect" not in p and "escalation" not in p


def test_unknown_is_neither_and_is_emitted_as_an_explicit_null():
    """An absent `verified` key cannot be told apart from a driver that does
    not report the field."""
    p = _verify_payload(_res("verify_state", {
        "status": "unknown",
        "predicates": [{"index": 0, "status": "unknown",
                        "unknown_reason": "untrusted_source"}]}))
    assert "verified" in p and p["verified"] is None
    assert p["verdict"]["decision"] == "cannot_tell"
    assert "UNKNOWN IS NOT SUCCESS" in p["verdict"]["hint"]
    assert p["predicates"][0]["unknown_reason"] == "untrusted_source"


def test_an_unrecognised_status_is_not_success():
    """The driver's vocabulary is not enumerable from here, so a word this
    build has never heard of must fail toward "cannot tell"."""
    p = _verify_payload(_res("verify_state", {"status": "something_new"}))
    assert p["verified"] is None
    assert p["verdict"]["decision"] == "cannot_tell"


def test_a_missing_status_is_not_success_either():
    p = _verify_payload(_res("verify_state", {}))
    assert p["verified"] is None and p["verdict"]["decision"] == "cannot_tell"


def test_the_evidence_the_driver_gave_is_carried_through():
    p = _verify_payload(_res("verify_state", {
        "status": "satisfied", "elapsed_ms": 18, "samples": 1, "stable": True}))
    assert p["elapsed_ms"] == 18 and p["samples"] == 1 and p["stable"] is True


# ------------------------------------------------------- dispatch + guards

class _Backend:
    def __init__(self, meta=None):
        self.calls = []
        self._meta = meta or {}

    def invoke_menu(self, path, *, pid=None, window_id=None):
        self.calls.append(("invoke_menu", list(path), pid, window_id))
        return _res("invoke_menu", dict(self._meta))

    def verify_state(self, expect, *, pid=None, window_id=None,
                     timeout_ms=None, stable_samples=None):
        self.calls.append(("verify_state", expect, timeout_ms, stable_samples))
        return _res("verify_state", dict(self._meta) or {"status": "satisfied"})


def test_invoke_menu_reaches_the_backend_with_its_path():
    b = _Backend()
    out = json.loads(_dispatch(b, "invoke_menu",
                               {"path": ["File", "Export…"], "pid": 7, "window_id": 9}))
    assert b.calls == [("invoke_menu", ["File", "Export…"], 7, 9)]
    assert out["ok"] is True


def test_the_resolved_target_is_reported_so_a_stale_one_is_visible():
    b = _Backend({"target": {"pid": 7253, "window_id": 122663,
                             "app": "Notes", "path": ["File", "Delete"]}})
    out = json.loads(_dispatch(b, "invoke_menu", {"path": ["File", "Delete"]}))
    assert out["target"]["app"] == "Notes"


def test_verify_state_passes_its_knobs_through():
    b = _Backend({"status": "satisfied"})
    expect = [{"element": {"selector": {"role": "AXButton",
                                        "label_contains": "Join"},
                           "exists": True}}]
    _dispatch(b, "verify_state", {"expect": expect, "timeout_ms": 3000,
                                  "stable_samples": 3})
    assert b.calls == [("verify_state", expect, 3000, 3)]


def test_a_backend_without_these_says_so_rather_than_crashing():
    class Old:
        pass

    for action, args in (("invoke_menu", {"path": ["File"]}),
                         ("verify_state", {"expect": [{"window": {"exists": True}}]})):
        out = json.loads(_dispatch(Old(), action, args))
        assert "not supported by this backend" in out["error"]


# --------------------------------------------- validation BEFORE the prompt

@pytest.mark.parametrize("bad", [None, [], "File", ["", "x"], [1, 2], ["  "]])
def test_a_malformed_path_is_refused_without_costing_an_approval(bad, monkeypatch):
    prompts = []
    monkeypatch.setattr("tools.computer_use.tool._request_approval",
                        lambda *a, **k: prompts.append(a) or None)
    out = json.loads(handle_computer_use({"action": "invoke_menu", "path": bad}))
    assert out["code"] == "bad_menu_path"
    assert prompts == [], "the user must not be asked about a call that cannot run"


@pytest.mark.parametrize("bad", [None, [], "x", [1], [{}] * 9])
def test_a_malformed_predicate_is_refused_early(bad):
    out = json.loads(handle_computer_use({"action": "verify_state", "expect": bad}))
    assert out["code"] == "bad_predicate"


@pytest.mark.parametrize("knob,bad", [("timeout_ms", "abc"), ("timeout_ms", True),
                                      ("stable_samples", 1.5)])
def test_a_non_integer_knob_is_caught_here_not_deep_in_the_backend(knob, bad):
    out = json.loads(handle_computer_use(
        {"action": "verify_state", "expect": [{"window": {"exists": True}}],
         knob: bad}))
    assert out["code"] == "bad_predicate"


# ------------------------------------------------------------ safety wiring

def test_the_approval_dialog_says_which_menu_item():
    """Every other destructive action tells the user what it will do; this one
    used to say only the word "invoke_menu"."""
    assert _summarize_action("invoke_menu", {"path": ["Terminal", "Quit Terminal"]}) \
        == "menu Terminal › Quit Terminal"


def test_approving_one_menu_item_does_not_authorise_every_other():
    """A click needs a capture and coordinates, so each is a separate visible
    decision. One opaque menu approval would otherwise cover every future
    menu item in the session: approving "Terminal > Quit Terminal" once would
    silently authorise "File > Delete Everything" later, because the scope key
    was only (action, delivery_mode)."""
    import inspect

    from tools.computer_use import tool as t

    src = inspect.getsource(t._request_approval)
    assert "scope_key = scope_key + tuple(" in src, \
        "the menu path must be part of the approval scope key"
    assert 'action == "invoke_menu"' in src


def test_invoking_a_menu_needs_approval_like_a_click():
    assert "invoke_menu" in _DESTRUCTIVE_ACTIONS
    assert "invoke_menu" not in _SAFE_ACTIONS


def test_checking_your_own_work_does_not_need_approval():
    assert "verify_state" in _SAFE_ACTIONS
    assert "verify_state" not in _DESTRUCTIVE_ACTIONS


# -------------------------------------------------------- stall interaction

def test_different_menu_items_are_different_calls():
    """Five different menu items in one window otherwise looked like one call
    repeated five times, and the fifth was refused with a message ("these
    exact arguments") that was simply false."""
    base = {"pid": 7, "window_id": 9}
    a = call_fingerprint("invoke_menu", dict(base, path=["File", "New"]))
    b = call_fingerprint("invoke_menu", dict(base, path=["File", "Quit"]))
    assert a != b


def test_different_predicates_are_different_calls():
    a = call_fingerprint("verify_state", {"expect": [{"window": {"exists": True}}]})
    b = call_fingerprint("verify_state", {"expect": [{"element": {"selector": {}}}]})
    assert a != b


# ------------------------------------------------------------------ schema

def test_both_are_offered_to_the_model_and_explained():
    schema = get_computer_use_schema()
    props = schema["parameters"]["properties"]
    assert {"invoke_menu", "verify_state"} <= set(props["action"]["enum"])
    desc = schema["description"]
    assert "invoke_menu" in desc and "verify_state" in desc
    assert "fails closed" in desc
    assert "UNKNOWN IS NOT SUCCESS" in desc
    assert "DAW" in desc
    assert props["path"]["maxItems"] == 16
    assert props["expect"]["maxItems"] == 8


def test_the_schema_is_honest_about_window_exists_false():
    """True for element predicates, false for window ones — and the window
    form is the natural way to check that a window closed."""
    desc = get_computer_use_schema()["parameters"]["properties"]["expect"]["description"]
    assert "WINDOW predicate it is accepted" in desc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
