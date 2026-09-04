"""Two-tier stall detection: advise at 3, refuse at 5 (3 for input actions).

The first draft of this detector failed both ways at once — it did not fire on
the trace it was written for, and it refused a fourth ``cmd+z``. Both failures
are pinned here, so neither can come back.

Every test asserts a side effect: whether a call is refused, which call, and
what the model is actually told. A detector that counts but never blocks is
the thing this replaces.
"""

from __future__ import annotations

import json
import sys
import threading

import pytest

from tools.computer_use.stall import (
    HARD_LIMIT,
    INPUT_HARD_LIMIT,
    SOFT_LIMIT,
    StallDetector,
    call_fingerprint,
    result_fingerprint,
)

EMPTY_CAPTURE = json.dumps({
    "mode": "som", "width": 0, "height": 0, "app": "", "window_title":
    "<no on-screen window matched app='Calendar'>", "total_elements": 0,
})
CAPTURE_ARGS = {"app": "Calendar", "mode": "som"}
UNVERIFIED_CLICK = json.dumps({
    "ok": True, "action": "click", "effect": "unverifiable",
    "verdict": {"decision": "verify_fresh_state"},
})


def multimodal(*, app="Calendar", width=1456, height=931, elements=329,
               png_bytes=180_000, nonce="a"):
    """The envelope a screenshot-bearing capture really returns.

    `screenshot_path` and `elements_file` are uuid4-named per call — the whole
    reason they must not reach the fingerprint.
    """
    return {
        "_multimodal": True,
        "content": [{"type": "text", "text": f"capture {app} {width}x{height}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}],
        "text_summary": f"capture {app} {width}x{height}",
        "meta": {"mode": "som", "width": width, "height": height,
                 "elements": elements, "png_bytes": png_bytes,
                 "screenshot_path": f"/cache/{nonce}-{id(object())}.png",
                 "elements_file": f"/cache/{nonce}-{id(object())}.json"},
    }


# ---------------------------------------------------------------- the incident

# The real 2026-09-03 sequence, in order, from the session transcript. Note the
# interleaving: the agent's own diagnostic calls sit between every repeat.
INCIDENT_TRACE = [
    ("capture", {"app": "Calendar", "mode": "som"}, multimodal()),
    ("click", {"app": "Calendar", "coordinate": [545, 405]}, UNVERIFIED_CLICK),
    ("capture", {"app": "Calendar", "mode": "som"}, multimodal(elements=324)),
    ("click", {"app": "Calendar", "coordinate": [1391, 527]}, UNVERIFIED_CLICK),
    ("list_windows", {}, json.dumps({"count": 5})),
    ("capture", {"app": "Calendar", "mode": "som"}, EMPTY_CAPTURE),
    ("list_apps", {}, json.dumps({"count": 78})),
    ("capture", {"app": "screen", "mode": "vision"}, multimodal(app="screen", elements=0)),
    ("capture", {"app": "Google Chrome", "mode": "som"}, EMPTY_CAPTURE),
    ("focus_app", {"app": "Calendar", "raise_window": True}, json.dumps({"ok": True})),
    ("capture", {"app": "Calendar", "mode": "som"}, EMPTY_CAPTURE),
    ("capture", {"app": "Calendar", "mode": "som"}, EMPTY_CAPTURE),
    ("list_windows", {}, json.dumps({"count": 5})),
    ("capture", {"app": "Calendar", "mode": "som"}, EMPTY_CAPTURE),
    ("capture", {"pid": 7253, "window_id": 122663, "mode": "som"}, EMPTY_CAPTURE),
]


def _replay(trace):
    """Run a trace; return the 1-based index of the first refused call."""
    det = StallDetector()
    for i, (action, args, result) in enumerate(trace, start=1):
        if det.block_reason(action, args) is not None:
            return i, det
        det.record(action, args, result)
    return None, det


def test_the_real_incident_trace_is_caught():
    """The regression that matters.

    A tail-consecutive streak peaks at 1 on this trace — the interleaved
    list_apps / list_windows / focus_app calls reset it every time — so the
    first implementation said nothing at all. Counting occurrences over the
    window is what makes the same thrashing visible.

    The trace contains four identical empty capture(app='Calendar') calls, at
    positions 6, 11, 12 and 14. The soft tier is therefore the tier that fires
    here, at the third of them, telling the agent to change approach before it
    makes the fourth. Refusal is reserved for a fifth, which this agent never
    reached because the user interrupted first.
    """
    det = StallDetector()
    advised_at = None
    for i, (action, args, result) in enumerate(INCIDENT_TRACE, start=1):
        assert det.block_reason(action, args) is None, \
            f"nothing in the real trace warrants a hard refusal (call {i})"
        det.record(action, args, result)
        if advised_at is None and det.advisory(action, args) is not None:
            advised_at = i
    assert advised_at == 12, (
        f"the 3rd identical empty capture is at position 12; advised at "
        f"{advised_at}"
    )
    # ...and the fifth would be refused outright.
    assert det.block_reason("capture", {"app": "Calendar", "mode": "som"}) is not None


def test_the_incident_advisory_says_something_the_agent_can_act_on():
    det = StallDetector()
    for action, args, result in INCIDENT_TRACE:
        det.record(action, args, result)
    note = det.advisory("capture", {"app": "Calendar", "mode": "som"})
    assert note is not None
    assert note["seen"] == 4
    assert "on_current_space" in note["hint"]     # the actual cause that day
    assert "avoids the GUI" in note["hint"]


def test_a_consecutive_run_is_not_required():
    det = StallDetector()
    for _ in range(SOFT_LIMIT):
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
        det.record("list_apps", {}, json.dumps({"count": 78}))
    assert det.advisory("capture", CAPTURE_ARGS) is not None


# ------------------------------------------------------- fingerprint honesty

def test_a_screenshot_does_not_defeat_result_matching():
    a = json.dumps({"ok": True, "action": "click", "effect": "unverifiable",
                    "png_b64": "AAAA", "width": 1456})
    b = json.dumps({"ok": True, "action": "click", "effect": "unverifiable",
                    "png_b64": "BBBB", "width": 1456})
    assert a != b and result_fingerprint(a) == result_fingerprint(b)


def test_two_different_screens_do_not_fingerprint_alike():
    """The multimodal envelope carries none of the flat result keys, so
    reading only those made every image capture identical to every other."""
    assert result_fingerprint(multimodal(app="Calendar", elements=5)) != \
        result_fingerprint(multimodal(app="Chrome", width=800, elements=300))


def test_uuid_named_cache_paths_are_excluded():
    """...and including them would mean nothing ever matches anything."""
    assert result_fingerprint(multimodal(nonce="x")) == \
        result_fingerprint(multimodal(nonce="y"))


def test_six_captures_of_a_changing_screen_are_never_refused():
    """An app loading — Zoom joining, a document rendering — is captured every
    few seconds. Each screen differs, so none of it is a stall."""
    det = StallDetector()
    for i in range(10):
        assert det.block_reason("capture", CAPTURE_ARGS) is None, f"capture #{i+1}"
        det.record("capture", CAPTURE_ARGS, multimodal(elements=i * 7))


def test_incidental_args_do_not_change_call_identity():
    base = {"app": "Calendar", "coordinate": [545, 405]}
    assert call_fingerprint("click", base) == \
        call_fingerprint("click", dict(base, capture_after=True))


def test_typed_text_is_fingerprinted_without_copying_it_wholesale():
    fp = call_fingerprint("type", {"text": "my bank password is hunter2 etc"})
    assert "hunter2" not in fp and "31:" in fp


# ---------------------------------------------- deliberately repeatable verbs

@pytest.mark.parametrize("keys", ["cmd+z", "down", "backspace", "tab", "pagedown"])
def test_repeated_key_presses_are_never_refused(keys):
    """A fourth cmd+z undoes a fourth edit. Their results are byte-identical
    every time and carry no evidence of progress, so any identity-based
    detector would refuse them — which is why they are exempt."""
    det = StallDetector()
    same = json.dumps({"ok": True, "action": "key", "effect": "unverifiable"})
    for i in range(12):
        assert det.block_reason("key", {"keys": keys}) is None, f"{keys} #{i+1}"
        det.record("key", {"keys": keys}, same)
        assert det.advisory("key", {"keys": keys}) is None


def test_repeated_scrolling_through_a_long_document_is_never_refused():
    det = StallDetector()
    same = json.dumps({"ok": True, "action": "scroll", "effect": "unverifiable"})
    for i in range(12):
        args = {"direction": "down", "amount": 5}
        assert det.block_reason("scroll", args) is None, f"scroll #{i+1}"
        det.record("scroll", args, same)


def test_wait_is_never_refused_and_never_wipes_the_history():
    """Clearing on wait let 'capture, wait, capture, wait' run forever —
    inserting a wait changes nothing real, and that is the incident's shape."""
    det = StallDetector()
    refused = False
    for _ in range(12):
        if det.block_reason("capture", CAPTURE_ARGS) is not None:
            refused = True
            break
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
        assert det.block_reason("wait", {"seconds": 1}) is None
        det.record("wait", {"seconds": 1}, json.dumps({"ok": True}))
    assert refused, "a wait between identical captures must not reset the count"


# ------------------------------------------------------------------- tiers

def test_input_is_refused_sooner_than_a_read():
    click = {"app": "Calendar", "coordinate": [1391, 527]}
    det = StallDetector()
    n = None
    for i in range(1, 11):
        if det.block_reason("click", click) is not None:
            n = i
            break
        det.record("click", click, UNVERIFIED_CLICK)
    assert n == INPUT_HARD_LIMIT
    assert INPUT_HARD_LIMIT < HARD_LIMIT


def test_a_read_is_refused_on_the_fifth_identical_attempt():
    det = StallDetector()
    n = None
    for i in range(1, 11):
        if det.block_reason("capture", CAPTURE_ARGS) is not None:
            n = i
            break
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    assert n == HARD_LIMIT
    payload = json.loads(det.block_reason("capture", CAPTURE_ARGS))
    assert payload["code"] == "stalled_repeated_action"
    assert payload["attempt"] == HARD_LIMIT


def test_the_refusal_points_at_the_cause_and_at_asking_the_user():
    det = StallDetector()
    for _ in range(HARD_LIMIT):
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    hint = json.loads(det.block_reason("capture", CAPTURE_ARGS))["verdict"]["hint"]
    for expected in ("on_current_space", "minimized", "non-GUI route",
                     "join URL", "ask", "vary the coordinate"):
        assert expected in hint, expected


def test_the_advisory_countdown_matches_what_actually_happens():
    """The first draft threatened refusal 'next' while allowing two more."""
    det = StallDetector()
    for _ in range(SOFT_LIMIT):
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    note = det.advisory("capture", CAPTURE_ARGS)
    assert note["seen"] == SOFT_LIMIT
    assert "1 more identical attempt(s) will be allowed" in note["hint"]
    det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    assert "next identical attempt will be refused" in \
        det.advisory("capture", CAPTURE_ARGS)["hint"]


def test_a_changing_result_is_progress_and_never_trips():
    det = StallDetector()
    for i in range(20):
        assert det.block_reason("capture", CAPTURE_ARGS) is None
        det.record("capture", CAPTURE_ARGS, json.dumps({"total_elements": i}))


def test_sessions_do_not_share_a_count():
    a, b = StallDetector(), StallDetector()
    for _ in range(HARD_LIMIT):
        a.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    assert a.block_reason("capture", CAPTURE_ARGS) is not None
    assert b.block_reason("capture", CAPTURE_ARGS) is None


def test_concurrent_calls_do_not_mutate_the_deque_mid_iteration():
    det = StallDetector()
    errors = []

    def hammer():
        try:
            for _ in range(400):
                det.block_reason("capture", CAPTURE_ARGS)
                det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
        except Exception as exc:  # RuntimeError: deque mutated during iteration
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, errors


# ------------------------------------------------------------------ wiring

def test_the_advisory_reaches_the_model_on_the_multimodal_path():
    """A top-level `stall` key is dropped by the agent loop, which hands the
    model only `content` / `text_summary` — so the soft tier was inert on
    exactly the screenshot captures it exists for."""
    from tools.computer_use.tool import _attach_stall_advisory

    det = StallDetector()
    for _ in range(SOFT_LIMIT):
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    out = _attach_stall_advisory(det, "capture", CAPTURE_ARGS, multimodal())
    assert out["content"][0]["text"].startswith("[stall]")
    assert out["text_summary"].startswith("[stall]")
    assert out["_multimodal"] is True
    assert out["content"][1]["type"] == "image_url"


def test_the_advisory_rides_alongside_the_verdict_not_instead_of_it():
    from tools.computer_use.tool import _attach_stall_advisory

    det = StallDetector()
    for _ in range(SOFT_LIMIT):
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    out = json.loads(_attach_stall_advisory(
        det, "capture", CAPTURE_ARGS,
        json.dumps({"ok": True, "verdict": {"decision": "verify_fresh_state"}})))
    assert out["verdict"]["decision"] == "verify_fresh_state"
    assert out["stall"]["decision"] == "change_approach"


def test_no_advisory_leaves_the_payload_byte_identical():
    from tools.computer_use.tool import _attach_stall_advisory

    payload = multimodal()
    assert _attach_stall_advisory(StallDetector(), "capture",
                                  CAPTURE_ARGS, payload) is payload


def test_the_session_detector_is_released_with_the_session():
    """A turn ending on three identical captures must not pre-poison the next."""
    from tools.computer_use import tool as t

    sid = "stall-release-test"
    det = t._stall_detector(sid)
    for _ in range(HARD_LIMIT):
        det.record("capture", CAPTURE_ARGS, EMPTY_CAPTURE)
    assert det.block_reason("capture", CAPTURE_ARGS) is not None
    t.release_computer_use_session(sid)
    assert sid not in t._stall_detectors
    assert t._stall_detector(sid).block_reason("capture", CAPTURE_ARGS) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
