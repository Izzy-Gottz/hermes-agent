"""A person being asked is not an idle turn, and a card goes with ITS turn
(tools.approval_human_wait.waiting_on_person; the HUMAN_WAIT_PING dispatcher).

Moe, 2026-09-25: a cron job's WhatsApp send waited on its confirmation card inside the claude-code
runtime's MCP server, where nothing stamps the agent's activity clock while a tool call is out. The
job read as idle, the 600 s watchdog killed it, and the card was left up for nobody. The binding is
by the CLI's own tool_use id, sent on every MCP request as _meta["claudecode/toolUseId"].
"""

import asyncio
import contextvars
import json
import time
import types

import pytest

from agent.transports.hermes_tool_bridge import (
    BRIDGE_SOCKET_ENV,
    BRIDGE_TOKEN_ENV,
    HUMAN_WAIT_PING,
    ToolBridge,
    call_bridged_tool,
)
from tools import approval_human_wait as hw


# --------------------------------------------------------------------------- outside the MCP server

def test_the_entering_threads_activity_callback_is_stamped_while_waiting(monkeypatch):
    from tools.environments.base import set_activity_callback

    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)
    stamps = []
    set_activity_callback(stamps.append)
    try:
        with hw.waiting_on_person("waiting for the person to answer a whatsapp confirmation",
                                  interval=0.05, check_interval=0.05, session_key="s-cb"):
            time.sleep(0.4)
    finally:
        set_activity_callback(None)
    assert len(stamps) >= 3, stamps
    assert all("whatsapp confirmation" in s for s in stamps)


def test_the_heartbeat_stops_when_the_wait_does(monkeypatch):
    from tools.environments.base import set_activity_callback

    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)
    stamps = []
    set_activity_callback(stamps.append)
    try:
        with hw.waiting_on_person("x", interval=0.05, check_interval=0.05, session_key="s-stop"):
            time.sleep(0.2)
        settled = len(stamps)
        time.sleep(0.3)
    finally:
        set_activity_callback(None)
    assert len(stamps) == settled


def test_the_wait_is_counted_as_a_humans(monkeypatch):
    monkeypatch.setattr(hw, "human_wait_ceiling", lambda: 3600.0)
    before = hw.human_wait_seconds("s-count")
    with hw.waiting_on_person("x", interval=10, session_key="s-count"):
        time.sleep(0.3)
    assert hw.human_wait_seconds("s-count") - before >= 0.25


def test_outside_the_server_an_interrupted_turn_is_a_gone_one(monkeypatch):
    from tools.interrupt import set_interrupt

    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)
    gone = []
    try:
        with hw.waiting_on_person("x", on_turn_gone=gone.append, check_interval=0.05, session_key="s-int"):
            set_interrupt(True)
            time.sleep(0.3)
    finally:
        set_interrupt(False)
    assert gone == ["the turn was interrupted"]


# --------------------------------------------------------------------------- inside the MCP server

def _in_server(monkeypatch, replies):
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "moe")
    import agent.transports.hermes_tool_bridge as bridge
    monkeypatch.setattr(bridge, "bridge_available", lambda env=None: True)
    seen = []

    def answer(tool, args=None, **kw):
        seen.append((tool, dict(args or {})))
        return json.dumps({"state": next(replies, "ok")})

    monkeypatch.setattr(bridge, "call_bridged_tool", answer)
    return seen


def test_the_ping_names_its_own_tool_call_never_a_learned_turn(monkeypatch):
    seen = _in_server(monkeypatch, iter([]))
    with hw.bound_tool_use("toolu_A"):
        with hw.waiting_on_person("waiting on the card", check_interval=0.05, session_key="s-ping"):
            time.sleep(0.3)
    assert seen and all(t == HUMAN_WAIT_PING for t, _ in seen)
    assert all(a == {"label": "waiting on the card", "tool_use_id": "toolu_A"} for _, a in seen)


def test_with_no_tool_use_id_nothing_is_bound_or_pinged(monkeypatch):
    seen = _in_server(monkeypatch, iter(["ended"]))
    gone = []
    with hw.waiting_on_person("x", on_turn_gone=gone.append, check_interval=0.05, session_key="s-none"):
        time.sleep(0.3)
    assert seen == [] and gone == []


def test_a_turn_already_gone_is_said_before_the_block_starts(monkeypatch):
    _in_server(monkeypatch, iter(["ended"]))
    gone, order = [], []
    with hw.bound_tool_use("toolu_A"):
        with hw.waiting_on_person("x", on_turn_gone=lambda r: (gone.append(r), order.append("gone")),
                                  check_interval=0.05, session_key="s-first"):
            order.append("body")
    assert gone == ["the turn that asked has ended"]
    assert order == ["gone", "body"]


def test_when_the_turn_ends_the_caller_is_told_once(monkeypatch):
    _in_server(monkeypatch, iter(["ok", "ok", "ended", "ended", "ended"]))
    gone = []
    with hw.bound_tool_use("toolu_A"):
        with hw.waiting_on_person("x", on_turn_gone=gone.append, check_interval=0.05, session_key="s-gone"):
            time.sleep(0.6)
    assert gone == ["the turn that asked has ended"]


def test_no_bridge_is_unknown_never_ended(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "moe")
    import agent.transports.hermes_tool_bridge as bridge
    monkeypatch.setattr(bridge, "bridge_available", lambda env=None: False)
    called = []
    monkeypatch.setattr(bridge, "call_bridged_tool", lambda *a, **k: called.append(a) or '{"state":"ended"}')
    assert hw._ping_the_turn("x", "toolu_A") == "unknown"
    gone = []
    with hw.bound_tool_use("toolu_A"):
        with hw.waiting_on_person("x", on_turn_gone=gone.append, check_interval=0.05, session_key="s-nob"):
            time.sleep(0.3)
    assert gone == [] and called == []


def test_an_unanswered_ping_is_not_a_gone_turn(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "moe")
    import agent.transports.hermes_tool_bridge as bridge
    monkeypatch.setattr(bridge, "bridge_available", lambda env=None: True)

    def broken(*a, **k):
        raise bridge.BridgeError("not answering")

    monkeypatch.setattr(bridge, "call_bridged_tool", broken)
    gone = []
    with hw.bound_tool_use("toolu_A"):
        with hw.waiting_on_person("x", on_turn_gone=gone.append, check_interval=0.05, session_key="s-unk"):
            time.sleep(0.3)
    assert gone == []


# --------------------------------------------------------------------------- the dispatcher

def _agent():
    agent = types.SimpleNamespace(stamps=[])
    agent._touch_activity = agent.stamps.append
    agent._turn_context = contextvars.copy_context()
    return agent


def _ping(agent, call):
    from agent import claude_code_runtime as rt
    return json.loads(rt.make_tool_bridge_dispatch(agent)(HUMAN_WAIT_PING, {"label": "on the card",
                                                                          "tool_use_id": call}))["state"]


def test_a_call_of_the_live_turn_is_stamped_and_ok():
    from agent import claude_code_runtime as rt
    agent = _agent()
    token = rt._turn_begins(agent)
    rt._note_turn_tool_call(agent, "toolu_A")
    assert _ping(agent, "toolu_A") == "ok"
    assert agent.stamps == ["on the card"]
    rt._turn_ends(agent, token)
    assert _ping(agent, "toolu_A") == "ended"
    assert agent.stamps == ["on the card"]


def test_a_card_from_turn_a_cannot_keep_turn_b_alive():
    from agent import claude_code_runtime as rt
    agent = _agent()
    a = rt._turn_begins(agent)
    rt._note_turn_tool_call(agent, "toolu_A")
    rt._turn_ends(agent, a)
    rt._turn_begins(agent)                       # turn B is live now
    assert _ping(agent, "toolu_A") == "ended"    # A's card goes
    assert agent.stamps == []                    # and B was not stamped for it


def test_a_call_made_after_its_turn_ended_is_never_bound_to_the_next():
    """A background subagent sending after its turn ended: its call id was never part of the live
    turn, so the live turn is neither stamped nor made its owner — "unknown", not "ok"."""
    from agent import claude_code_runtime as rt
    agent = _agent()
    rt._turn_begins(agent)
    assert _ping(agent, "toolu_LATE") == "unknown"
    assert _ping(agent, "") == "unknown"
    assert agent.stamps == []


def test_turn_flags_are_set_only_once_the_lock_is_held(monkeypatch):
    """A turn refused with 'another turn is still running' must not touch the live one's flags."""
    from agent import claude_code_runtime as rt
    agent = _agent()
    live = rt._turn_begins(agent)
    before = (agent._turn_id, agent._turn_live)

    def refused(agent, **kw):             # the real body returns here without calling _on_locked
        return {"final_response": "busy", "completed": False}

    monkeypatch.setattr(rt, "_run_claude_code_turn_body", refused)
    rt.run_claude_code_turn(agent, user_message="hi", original_user_message="hi", messages=[],
                            effective_task_id="t")
    assert (agent._turn_id, agent._turn_live) == before
    rt._turn_ends(agent, live)


def test_the_turn_is_unmarked_when_it_ends_even_by_raising(monkeypatch):
    from agent import claude_code_runtime as rt
    agent = _agent()
    seen = {}

    def body(agent, **kw):
        kw["_on_locked"]()
        seen["during"] = agent._turn_live
        raise RuntimeError("child died")

    monkeypatch.setattr(rt, "_run_claude_code_turn_body", body)
    with pytest.raises(RuntimeError):
        rt.run_claude_code_turn(agent, user_message="hi", original_user_message="hi", messages=[],
                                effective_task_id="t")
    assert seen["during"] is True
    assert agent._turn_live is False


def test_only_the_owner_unmarks_a_turn():
    from agent import claude_code_runtime as rt
    agent = _agent()
    first = rt._turn_begins(agent)
    rt._turn_begins(agent)                # a later owner
    rt._turn_ends(agent, first)           # the earlier one finishing late
    assert agent._turn_live is True


def test_each_turn_gets_its_own_id(monkeypatch):
    from agent import claude_code_runtime as rt
    agent = _agent()
    ids = []

    def body(agent, **kw):
        kw["_on_locked"]()
        ids.append(agent._turn_id)
        return {}

    monkeypatch.setattr(rt, "_run_claude_code_turn_body", body)
    for _ in range(2):
        rt.run_claude_code_turn(agent, user_message="hi", original_user_message="hi", messages=[],
                                effective_task_id="t")
    assert len(set(ids)) == 2 and all(ids)


def test_the_event_bridge_records_the_call_for_the_live_turn_only():
    from agent import claude_code_runtime as rt
    agent = _agent()
    agent.tool_progress_callback = agent.tool_start_callback = None
    on_event = rt.make_claude_code_event_bridge(agent)
    on_event({"kind": "tool_started", "call_id": "toolu_BEFORE", "name": "whatsapp_send"})
    token = rt._turn_begins(agent)
    on_event({"kind": "tool_started", "call_id": "toolu_A", "name": "whatsapp_send"})
    assert rt._turn_ledger(agent).current == {"toolu_A"}
    rt._turn_ends(agent, token)


def test_a_card_from_turn_a_is_ended_after_rebind_to_a_fresh_agent():
    """api_server builds a fresh AIAgent per request and session.rebind moves dispatch to it. The
    ledger is the session's, so turn A's card — pinging through the NEW agent's dispatch — is
    answered "ended", not "unknown"."""
    from agent import claude_code_runtime as rt
    session = types.SimpleNamespace()
    first = _agent()
    first._claude_code_session = session
    token = rt._turn_begins(first)
    rt._note_turn_tool_call(first, "toolu_A")
    assert _ping(first, "toolu_A") == "ok"
    rt._turn_ends(first, token)
    fresh = _agent()                       # the next request's agent
    fresh._claude_code_session = session   # rebind: same session, new dispatch
    rt._turn_begins(fresh)
    assert _ping(fresh, "toolu_A") == "ended"
    assert fresh.stamps == []


def test_a_retired_session_still_ends_the_turn_it_began_on():
    from agent import claude_code_runtime as rt
    session = types.SimpleNamespace()
    agent = _agent()
    agent._claude_code_session = session
    token = rt._turn_begins(agent)
    agent._claude_code_session = None       # retired mid-turn
    rt._turn_ends(agent, token)
    assert session._hermes_turn_ledger.live is False


# --------------------------------------------------------------------------- the wire

def test_the_bridge_carries_the_ping_and_its_call_though_it_is_not_a_bridged_tool(tmp_path):
    calls = []
    b = ToolBridge(lambda tool, args: calls.append((tool, dict(args))) or "ok",
                   directory=str(tmp_path))
    b.start()
    try:
        env = {BRIDGE_SOCKET_ENV: b.socket_path, BRIDGE_TOKEN_ENV: b.token}
        assert call_bridged_tool(HUMAN_WAIT_PING, {"label": "x" * 500, "tool_use_id": "toolu_A"},
                                 env=env, timeout=5) == "ok"
    finally:
        b.close()
    assert calls == [(HUMAN_WAIT_PING, {"label": "x" * 120, "tool_use_id": "toolu_A"})]


def test_the_mcp_server_hands_the_clis_tool_use_id_to_the_tool(monkeypatch):
    """Through the real SDK: a tools/call carrying _meta["claudecode/toolUseId"] reaches the handler
    with CURRENT_TOOL_USE_ID set, and the context parameter is not in the tool's schema."""
    from agent.transports import hermes_tools_mcp_server as m
    import model_tools
    from mcp.client.client import Client

    monkeypatch.setattr(m, "discover_external_mcp_servers", lambda: [])
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda quiet_mode=True, **kw: [
        {"type": "function", "function": {"name": "web_search", "parameters": {
            "type": "object", "properties": {"query": {"type": "string"}}}}}])
    seen = {}

    def fake_call(name, args, *a, **k):
        seen["id"] = hw.CURRENT_TOOL_USE_ID.get()
        return "ok"

    monkeypatch.setattr(model_tools, "handle_function_call", fake_call)
    server = m._build_server("claude-code")

    async def go():
        async with Client(server) as c:
            tools = await c.list_tools()
            schema = [t for t in tools.tools if t.name == "web_search"][0].input_schema
            await c.call_tool("web_search", {"query": "x"}, meta={"claudecode/toolUseId": "toolu_ABC"})
            return schema

    schema = asyncio.run(go())
    assert seen["id"] == "toolu_ABC"
    assert list(schema.get("properties", {})) == ["query"]
