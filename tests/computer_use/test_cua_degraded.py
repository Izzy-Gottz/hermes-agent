"""An empty capture has to say why it is empty.

cua-driver returns the screenshot but an EMPTY element list with
``degraded_reason: ax_window_unresolved`` when a window's accessibility surface
cannot be resolved — which is what happens to a window on another macOS Space,
to an app that has only just launched, and to a canvas/Electron surface that
exposes no tree. Hermes read none of it: ``grep degraded_reason`` over
tools/computer_use/ returned nothing. The model was handed "0 interactable
element(s)" with no explanation and no way to tell those three apart, though
each has a different fix.
"""

from __future__ import annotations

import json
import sys

import pytest

from tools.computer_use.backend import CaptureResult, UIElement
from tools.computer_use.cua_backend import _degraded_reason_from
from tools.computer_use.tool import _capture_response, _degraded_note


def _summary(result):
    if isinstance(result, dict):
        return result.get("text_summary") or json.dumps(result)
    return result


def test_the_reason_is_lifted_from_structured_content():
    assert _degraded_reason_from(
        {"structuredContent": {"degraded_reason": "ax_window_unresolved"}}
    ) == "ax_window_unresolved"


def test_it_is_also_found_on_the_flattened_shapes():
    # Older drivers put it on `data`, or at the top level.
    assert _degraded_reason_from({"data": {"degraded_reason": "x"}}) == "x"
    assert _degraded_reason_from({"degraded_reason": "y"}) == "y"


def test_absent_or_malformed_is_simply_no_reason():
    for payload in ({}, {"structuredContent": {}}, {"degraded_reason": ""},
                    {"degraded_reason": 3}, None, "not a dict"):
        assert _degraded_reason_from(payload) is None


def test_an_empty_ax_capture_explains_itself_and_names_the_fix():
    note = _degraded_note("ax_window_unresolved", 0)
    assert note is not None
    assert "another macOS Space" in note
    assert "on_current_space" in note
    assert "mode='vision'" in note
    # the other two causes that produce the same empty list
    assert "just-launched" in note or "launched" in note


def test_a_minimized_window_gets_its_own_explanation():
    note = _degraded_note("minimized_or_hidden_window", 0)
    assert "minimized" in note
    assert "raise_window=true" in note


def test_an_unknown_reason_is_surfaced_verbatim_rather_than_swallowed():
    """The driver may add a value; "no elements, no reason" is the state this
    exists to end, so an unrecognised reason must still reach the model."""
    note = _degraded_note("some_new_driver_reason", 0)
    assert "some_new_driver_reason" in note
    assert "mode='vision'" in note


def test_a_capture_that_found_elements_is_not_apologised_for():
    # A note on every successful call is a note nobody reads.
    assert _degraded_note("ax_window_unresolved", 12) is None


def test_no_reason_means_no_note():
    assert _degraded_note(None, 0) is None
    assert _degraded_note("", 0) is None


def test_the_note_reaches_the_capture_summary_the_model_reads():
    cap = CaptureResult(mode="som", width=1456, height=931, png_b64=None,
                        elements=[], app="Calendar", window_title="Calendar",
                        degraded_reason="ax_window_unresolved")
    text = _summary(_capture_response(cap))
    assert "0 interactable element(s)" in text
    assert "degraded: ax_window_unresolved" in text
    assert "on_current_space" in text


def test_a_healthy_capture_summary_gains_nothing():
    cap = CaptureResult(
        mode="som", width=1456, height=931, png_b64=None,
        elements=[UIElement(index=1, role="AXButton", label="Join",
                            bounds=(10, 10, 40, 20))],
        app="Calendar", window_title="Calendar")
    assert "degraded" not in _summary(_capture_response(cap))


def test_the_field_defaults_to_none_so_every_other_backend_still_builds():
    # macos_native_backend and the noop backend construct CaptureResult
    # without it; a required field here would break both.
    assert CaptureResult(mode="ax", width=0, height=0).degraded_reason is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
