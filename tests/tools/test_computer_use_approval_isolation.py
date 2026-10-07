"""computer_use approval is the shared ``tools.approval`` gate — no private grant store.

Two contracts:

* Moe fork: with NO callback wired (neither computer_use's nor the per-thread CLI one) the action runs —
  Moe's own pre_tool_call gate sits one layer out. Upstream refuses here; see the first test.
* A grant answered through computer_use lives in ``tools.approval``'s store under computer_use's own scope key,
  so ``is_approved`` sees it and ``clear_session`` retires it like any terminal pattern.

A leaked callback still poisons later tests (a raising one becomes deny, a blocking one hangs), so the autouse
reset in ``tests/conftest.py`` stays and the polluter/observer pair below keeps proving it.
"""

import json

import pytest


def _install_backend(cu_tool):
    class _RecordingBackend:
        def __init__(self):
            self.calls = []

        def start(self):
            pass

        def stop(self):
            pass

        def is_available(self):
            return True

        def click(self, **kw):
            self.calls.append(("click", kw))
            from tools.computer_use.backend import ActionResult

            return ActionResult(ok=True, action="click")

        def capture(self, mode="som", app=None):
            from tools.computer_use.backend import CaptureResult

            return CaptureResult(
                mode=mode, width=1, height=1, png_b64=None, elements=[],
                app="X", window_title="",
            )

    backend = _RecordingBackend()
    cu_tool.reset_backend_for_tests()
    cu_tool._backend = backend
    return backend


@pytest.fixture
def _nobody_to_ask(monkeypatch):
    """No interactive CLI, no gateway, no per-thread terminal callback, yolo off."""
    from tools import approval

    for name in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None)
    yield


def test_no_callback_allows_because_the_gate_is_one_layer_out(_nobody_to_ask, monkeypatch):
    """Moe fork: with no callback wired (neither computer_use's nor the per-thread CLI one) the click runs.

    Upstream fails closed here. Moe's gateway and the claude-code hermes-tools server have no callback, so
    upstream's posture would refuse (or card) every click; Moe's pre_tool_call send gate judges what each
    action does instead, and the owner decided the native path carries no per-step gates (2026-09-10).
    A wired callback still gets the shared gate — see the grant-store test below."""
    from tools.computer_use import tool as cu_tool

    backend = _install_backend(cu_tool)
    result = cu_tool.handle_computer_use({"action": "click", "element": 3})
    assert [name for name, _ in backend.calls] == ["click"], result


def test_a_wired_cli_callback_that_denies_is_obeyed(_nobody_to_ask, monkeypatch):
    """The short-circuit is only for 'nobody wired': a per-thread CLI callback answering deny refuses."""
    from tools import approval
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: lambda command, description, **kw: "deny")
    backend = _install_backend(cu_tool)
    result = json.loads(cu_tool.handle_computer_use({"action": "click", "element": 3}))
    assert "error" in result, result
    assert backend.calls == []


def test_a_forgets_a_poisoned_approval_callback():
    """Simulates the polluter: installs a raising callback and deliberately does not reset it."""
    from tools.computer_use import tool as cu_tool

    def poisoned(command, description, **kw):
        raise RuntimeError("dead UI")

    cu_tool.set_approval_callback(poisoned)
    # no reset — the autouse fixture must clean this up


def test_b_still_dispatches_after_the_polluter(monkeypatch):
    """Answers through the per-thread terminal callback only. The explicit computer_use callback takes precedence
    in the shared gate, so if the polluter's raising one had leaked, this click would be denied."""
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: lambda command, description, **kw: "once")
    backend = _install_backend(cu_tool)
    result = cu_tool.handle_computer_use({"action": "click", "element": 3})
    assert [name for name, _ in backend.calls] == ["click"], f"leaked approval callback poisoned this test: {result!r}"
