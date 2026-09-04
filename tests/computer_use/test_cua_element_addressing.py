"""Clicking by element index — the path the schema tells the model to prefer.

cua-driver 0.23.2 refuses a bare index: "click: bare element_index is not
accepted; pass element_token, or snapshot_id together with element_index".
Hermes sent neither, because attaching the token is gated on
`supports_capability("accessibility.element_tokens")` — and this driver
advertises 56 tools with ZERO capability tokens on any of them, so the gate is
permanently False.

A guard written to protect OLD drivers was refusing the CURRENT one, and every
element click failed. Measured 2026-09-04 driving Freeform: the click was
refused, and the agent fell back to raw coordinates, which is the rung the
prompt explicitly tells it to avoid.

The driver accepts either addressing route. Keeping only the token meant one
route; capture now also records the `snapshot_id` those indices belong to, so
the index route works whatever a build publishes.
"""

from __future__ import annotations

import sys

import pytest

from tools.computer_use.cua_backend import CuaDriverBackend


class _Backend(CuaDriverBackend):
    def __init__(self, tokens=None, snap=None, capability=False):
        self._snapshot_tokens = tokens or {}
        self._snapshot_id = snap
        self._capability = capability

        class _S:
            def supports_capability(_s, cap, tool=None):
                return capability
        self._session = _S()


def test_the_snapshot_id_is_attached_even_when_the_capability_gate_says_no():
    """The regression. The gate is False on a real driver, so nothing else
    may depend on it for the index route to work."""
    b = _Backend(tokens={7: "tok"}, snap="s0000001c", capability=False)
    args = {"element_index": 7}
    b._maybe_attach_element_token("click", args)
    assert args["snapshot_id"] == "s0000001c"


def test_the_token_still_rides_when_the_driver_advertises_it():
    b = _Backend(tokens={7: "tok"}, snap="s1", capability=True)
    args = {"element_index": 7}
    b._maybe_attach_element_token("click", args)
    assert args["element_token"] == "tok"
    assert args["snapshot_id"] == "s1"


def test_an_index_with_no_token_still_gets_the_snapshot_id():
    """A build that publishes snapshot_id but no per-element tokens must
    still be addressable — that is the whole point of carrying both."""
    b = _Backend(tokens={}, snap="s2")
    args = {"element_index": 3}
    b._maybe_attach_element_token("click", args)
    assert args["snapshot_id"] == "s2"
    assert "element_token" not in args


def test_a_caller_supplied_snapshot_id_is_not_overwritten():
    b = _Backend(tokens={}, snap="ours")
    args = {"element_index": 3, "snapshot_id": "theirs"}
    b._maybe_attach_element_token("click", args)
    assert args["snapshot_id"] == "theirs"


def test_nothing_is_attached_without_an_element_index():
    b = _Backend(tokens={1: "t"}, snap="s3")
    args = {"coordinate": [10, 20]}
    b._maybe_attach_element_token("click", args)
    assert args == {"coordinate": [10, 20]}


@pytest.mark.parametrize("idx", [None, "3", 1.5, True])
def test_a_non_integer_index_is_ignored(idx):
    b = _Backend(tokens={1: "t"}, snap="s4")
    args = {"element_index": idx}
    b._maybe_attach_element_token("click", args)
    assert "snapshot_id" not in args or idx is True


def test_no_snapshot_means_nothing_invented():
    b = _Backend(tokens={}, snap=None)
    args = {"element_index": 3}
    b._maybe_attach_element_token("click", args)
    assert "snapshot_id" not in args


def test_the_snapshot_id_is_cleared_wherever_the_tokens_are():
    """A stale snapshot_id paired with a fresh capture's indices addresses the
    wrong element. Both are per-snapshot and must die together."""
    import inspect

    from tools.computer_use import cua_backend

    src = inspect.getsource(cua_backend)
    clears_tokens = src.count("self._snapshot_tokens = {}")
    clears_snap = src.count("self._snapshot_id = None")
    assert clears_snap >= clears_tokens, (
        f"{clears_tokens} sites clear the token map but only {clears_snap} "
        f"clear the snapshot id"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
