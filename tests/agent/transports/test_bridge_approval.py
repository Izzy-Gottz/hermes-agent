"""A plugin's approval, asked from the hermes-tools MCP server, reaches the person.

Under the claude-code runtime every Hermes and MCP tool runs in the MCP server
the CLI spawned. That process has no session and runs with
``HERMES_SINGLE_QUERY_SESSION``, so ``request_tool_approval`` refused every
plugin escalation there without asking anybody — a send gate that could never
say yes. The question now goes home over the tool bridge and is asked inside
the turn's own context. These tests use a real bridge (socket, token, thread)
between the two halves, and the real ``make_tool_bridge_dispatch``.

What must hold, in both directions:

* the question reaches the gate in the turn's process, with the tool, the full
  text and the rule key, and the person's answer comes back;
* nothing is approved that the person did not approve: no bridge, a dead
  bridge, a turn that has ended, a scheduled job, a gate that raises.
"""

from __future__ import annotations

import contextvars
import json
import types

import pytest

from agent.transports.hermes_tool_bridge import (
    BRIDGE_SOCKET_ENV,
    BRIDGE_TOKEN_ENV,
    TURN_APPROVAL_QUERY,
    ToolBridge,
)

LONG_BODY = "Dear Dan,\n\n" + ("All of this must be seen before it is sent. " * 120) + "\n\nThe last words."


def _agent(live: bool = True):
    agent = types.SimpleNamespace()
    agent._turn_context = contextvars.copy_context()
    agent._turn_live = live
    return agent


@pytest.fixture
def home(monkeypatch, tmp_path):
    """The turn's process: a bridge whose far end is the real runtime dispatcher."""
    from agent import claude_code_runtime as rt

    asked: list[tuple[str, str, str]] = []
    verdict = {"approved": True, "message": None}

    def fake_gate(tool_name, reason, *, rule_key="", approval_callback=None):
        asked.append((tool_name, reason, rule_key))
        return dict(verdict)

    import tools.approval as approval
    monkeypatch.setattr(approval, "request_tool_approval", fake_gate)
    agent = _agent()
    bridge = ToolBridge(rt.make_tool_bridge_dispatch(agent), directory=str(tmp_path))
    bridge.start()
    ns = types.SimpleNamespace(bridge=bridge, asked=asked, verdict=verdict, agent=agent, real=approval)
    try:
        yield ns
    finally:
        bridge.close()


def _as_mcp_server(monkeypatch, bridge):
    """This process, as the child's MCP server: a profile and a bridge address."""
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "claude-code")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    if bridge is None:
        monkeypatch.delenv(BRIDGE_SOCKET_ENV, raising=False)
        monkeypatch.delenv(BRIDGE_TOKEN_ENV, raising=False)
    else:
        monkeypatch.setenv(BRIDGE_SOCKET_ENV, bridge.socket_path)
        monkeypatch.setenv(BRIDGE_TOKEN_ENV, bridge.token)


def _forward(tool, reason, rule_key):
    """The MCP server's half, called directly: request_tool_approval is monkeypatched at home."""
    from tools.approval import _request_tool_approval_over_bridge
    return _request_tool_approval_over_bridge(tool, reason, rule_key)


def test_the_question_goes_home_whole_and_the_yes_comes_back(home, monkeypatch):
    _as_mcp_server(monkeypatch, home.bridge)
    out = _forward("gmail_send", LONG_BODY, "moe-send:ask-abc")
    assert out["approved"] is True
    assert home.asked == [("gmail_send", LONG_BODY, "moe-send:ask-abc")], \
        "the gate at home must be asked with the tool, every word, and the per-call rule key"


def test_a_no_comes_back_as_a_no_with_the_gates_words(home, monkeypatch):
    _as_mcp_server(monkeypatch, home.bridge)
    home.verdict.update(approved=False, message="BLOCKED: User denied. Do NOT retry.")
    out = _forward("gmail_send", "hi", "moe-send:ask-1")
    assert out["approved"] is False
    assert out["message"] == "BLOCKED: User denied. Do NOT retry."


def test_request_tool_approval_itself_routes_home_in_the_mcp_server(monkeypatch, tmp_path):
    """The public entry point, not the helper: this is what a plugin and the `approve` directive call.
    Before, it read HERMES_SINGLE_QUERY_SESSION and refused without asking anyone."""
    from agent import claude_code_runtime as rt
    import tools.approval as approval

    real = approval.request_tool_approval
    asked = []
    agent = _agent()

    def dispatch(tool, args):
        # The home side runs a fake gate; the MCP side below runs the REAL request_tool_approval.
        if tool == TURN_APPROVAL_QUERY:
            asked.append(dict(args))
            return json.dumps({"approved": True, "message": None})
        return rt.make_tool_bridge_dispatch(agent)(tool, args)

    bridge = ToolBridge(dispatch, directory=str(tmp_path))
    bridge.start()
    try:
        _as_mcp_server(monkeypatch, bridge)
        out = real("slack_send", "Send this?", rule_key="moe-send:ask-2")
    finally:
        bridge.close()
    assert out["approved"] is True
    assert asked == [{"tool_name": "slack_send", "reason": "Send this?", "rule_key": "moe-send:ask-2"}]


def test_no_bridge_is_a_refusal_that_says_nobody_could_be_asked(monkeypatch):
    import tools.approval as approval
    _as_mcp_server(monkeypatch, None)
    out = approval.request_tool_approval("gmail_send", "hi", rule_key="moe-send:x")
    assert out["approved"] is False
    assert "no way to ask you" in out["message"]


def test_a_dead_bridge_is_a_refusal(monkeypatch, tmp_path):
    bridge = ToolBridge(lambda *_a: "x", directory=str(tmp_path))
    bridge.start()
    path, token = bridge.socket_path, bridge.token
    bridge.close()
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "claude-code")
    monkeypatch.setenv(BRIDGE_SOCKET_ENV, path)
    monkeypatch.setenv(BRIDGE_TOKEN_ENV, token)
    out = _forward("gmail_send", "hi", "moe-send:x")
    assert out["approved"] is False
    assert "could not reach you" in out["message"]


def test_a_turn_that_has_ended_is_not_asked(home, monkeypatch):
    _as_mcp_server(monkeypatch, home.bridge)
    home.agent._turn_live = False
    out = _forward("gmail_send", "hi", "moe-send:x")
    assert out["approved"] is False and "already ended" in out["message"]
    assert home.asked == [], "an ended turn must not put a question to the person"


def test_a_scheduled_job_is_not_asked_even_with_cron_mode_approve(home, monkeypatch):
    """cron_mode: approve would make the gate say yes by itself. A send needs a person."""
    from gateway.session_context import _VAR_MAP
    ctx = contextvars.copy_context()
    ctx.run(_VAR_MAP["HERMES_CRON_SESSION"].set, "1")
    home.agent._turn_context = ctx
    _as_mcp_server(monkeypatch, home.bridge)
    out = _forward("gmail_send", "hi", "moe-send:x")
    assert out["approved"] is False and "scheduled job" in out["message"]
    assert home.asked == []


def test_a_gate_that_raises_at_home_is_a_refusal(home, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("gate exploded")
    monkeypatch.setattr(home.real, "request_tool_approval", boom)
    _as_mcp_server(monkeypatch, home.bridge)
    out = _forward("gmail_send", "hi", "moe-send:x")
    assert out["approved"] is False and "failed while asking" in out["message"]


def test_an_answer_that_is_not_the_gates_shape_is_not_a_yes(monkeypatch, tmp_path):
    for reply in ('"approved"', '{"approved": "true"}', "[]", "not json"):
        bridge = ToolBridge(lambda tool, args, r=reply: r, directory=str(tmp_path))
        bridge.start()
        try:
            _as_mcp_server(monkeypatch, bridge)
            out = _forward("gmail_send", "hi", "moe-send:x")
        finally:
            bridge.close()
        assert out["approved"] is False, reply


def test_outside_the_mcp_server_nothing_is_forwarded(monkeypatch):
    """The gateway itself asks in-process as before; only the spawned server goes home."""
    import tools.approval as approval
    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)
    called = []
    monkeypatch.setattr(approval, "_request_tool_approval_over_bridge",
                        lambda *a: called.append(a) or {"approved": True})
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    out = approval.request_tool_approval("gmail_send", "hi", rule_key="moe-send:x")
    assert called == []
    assert out["approved"] is False  # single-query, default deny: the old behaviour, untouched
