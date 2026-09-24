"""The fix fields reach the host app on the ``completed`` tool-progress frame.

Moe's docs/FIXABLE-FAILURES.md slice 3, app half. A tool that fails for a reason the person can fix
returns the ``tools.fix_reasons`` contract; the host (Memoe) reads ``/v1/chat/completions`` SSE, whose
``hermes.tool.progress`` ``completed`` frame used to carry only the tool name, the call id and the
status — so the code never left Hermes. These tests drive the real chain under the Claude Code runtime:

    producer (fix_error) -> MCP bridge (to_mcp_content) -> the child's stream-json tool_result
    -> _TurnProjector -> make_claude_code_event_bridge -> api_server's _on_tool_complete -> SSE

and the HTTP tool loop's shape (the JSON string straight into ``tool_complete_callback``).

``MOE_FIX_FRAME_OUT``: when set, the SSE ``completed`` frame of the Claude Code route is written there
verbatim, so the app's Swift test parses the bytes this server really wrote, not a hand-typed copy.
"""

from __future__ import annotations

import asyncio
import json
import os
import types
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app, _make_adapter
from tools.fix_reasons import CODES, fix_error, host_fields


def _frames(body: str) -> list[dict]:
    """Every ``hermes.tool.progress`` payload, in order."""
    out, lines = [], body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "event: hermes.tool.progress":
            continue
        for follow in lines[i + 1: i + 4]:
            if follow.startswith("data: "):
                out.append(json.loads(follow[len("data: "):]))
                break
    return out


def _raw_completed_frame(body: str) -> str:
    """The ``completed`` SSE event exactly as written: the event line, the data line, the blank."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "event: hermes.tool.progress" and '"completed"' in lines[i + 1]:
            return "\n".join(lines[i:i + 2]) + "\n\n"
    raise AssertionError("no completed frame")


def _denied_chrome() -> str:
    return fix_error(
        "Memoe isn't allowed to read Google Chrome's data.", "tcc_app_data",
        subject="Google Chrome", retry=True, path="/Users/x/Library/Application Support/Google/Chrome")


async def _post(run_agent) -> str:
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", side_effect=run_agent):
            resp = await cli.post("/v1/chat/completions", json={
                "model": "test", "messages": [{"role": "user", "content": "find a flight"}],
                "stream": True})
            assert resp.status == 200
            return await resp.text()


def _claude_code_route(tool_result_text: str, *, is_error: bool):
    """A ``_run_agent`` that plays one tool call through the REAL Claude Code projector and bridge."""
    from agent.claude_code_runtime import make_claude_code_event_bridge
    from agent.transports.claude_code_session import TurnResult, _TurnProjector

    async def run_agent(**kwargs):
        agent = types.SimpleNamespace(
            tool_start_callback=kwargs.get("tool_start_callback"),
            tool_complete_callback=kwargs.get("tool_complete_callback"),
            tool_progress_callback=None, show_commentary=False, _stream_needs_break=False)
        on_event = make_claude_code_event_bridge(agent)
        projector = _TurnProjector(types.SimpleNamespace(_emit=on_event), silent=False)
        result = TurnResult()
        projector.handle({"type": "assistant", "message": {"content": [{
            "type": "tool_use", "id": "toolu_chrome_1", "name": "mcp__hermes__browser_navigate",
            "input": {"url": "https://www.google.com/travel/flights"}}]}}, result)
        projector.handle({"type": "user", "message": {"content": [{
            "type": "tool_result", "tool_use_id": "toolu_chrome_1", "is_error": is_error,
            "content": [{"type": "text", "text": tool_result_text}]}]}}, result)
        cb = kwargs.get("stream_delta_callback")
        if cb:
            await asyncio.sleep(0.05)
            cb("I can't read Chrome's data yet.")
        return ({"final_response": "I can't read Chrome's data yet.", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    return run_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [False, True], ids=["plain", "flagged-is_error"])
async def test_claude_code_route_carries_the_fix(is_error):
    from agent.transports.hermes_tools_mcp_server import to_mcp_content
    text = to_mcp_content(_denied_chrome())
    body = await _post(_claude_code_route(text, is_error=is_error))
    frames = _frames(body)
    assert [f["status"] for f in frames] == ["running", "completed"], frames
    done = frames[1]
    assert done["toolCallId"] == "toolu_chrome_1"
    assert done["fix"] == {
        "code": "tcc_app_data", "owner": "app", "pane": "Privacy_AllFiles", "subject": "Google Chrome",
        "retry": True, "path": "/Users/x/Library/Application Support/Google/Chrome"}, done
    # The model's words stay the model's: the host is handed the fields, not the sentence.
    assert "error" not in done["fix"]
    assert "fix" not in frames[0]
    out = os.environ.get("MOE_FIX_FRAME_OUT")
    if out and not is_error:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(_raw_completed_frame(body))


@pytest.mark.asyncio
async def test_http_loop_shape_carries_the_fix():
    async def run_agent(**kwargs):
        kwargs["tool_start_callback"]("call_1", "computer_use", {"action": "capture"})
        kwargs["tool_complete_callback"]("call_1", "computer_use", {"action": "capture"}, fix_error(
            "CuaDriver isn't allowed to see your screen.", "tcc_driver_screen",
            subject="CuaDriver", retry=True, restart="driver"))
        return ({"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    done = _frames(await _post(run_agent))[1]
    assert done["fix"]["code"] == "tcc_driver_screen"
    assert done["fix"]["owner"] == "driver"
    assert done["fix"]["pane"] == "Privacy_ScreenCapture"
    assert done["fix"]["restart"] == "driver"


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    "ok",
    json.dumps({"error": "boom"}),                                  # a plain tool_error: no code
    json.dumps({"error": "x", "code": "ENOENT"}),                   # a code of its own, not ours
    json.dumps({"success": True, "code": "tcc_app_data_lookalike"}),
    None,
], ids=["ok", "plain-error", "foreign-code", "lookalike", "none"])
async def test_no_fix_without_a_known_code(result):
    async def run_agent(**kwargs):
        kwargs["tool_start_callback"]("call_1", "terminal", {"command": "ls"})
        kwargs["tool_complete_callback"]("call_1", "terminal", {"command": "ls"}, result)
        return ({"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    frames = _frames(await _post(run_agent))
    assert frames[1]["status"] == "completed"
    assert "fix" not in frames[1], frames[1]


def test_host_fields_every_code_round_trips():
    for code in sorted(CODES):
        got = host_fields(fix_error("words", code))
        assert got["code"] == code
        assert set(got) >= {"code", "owner", "pane", "subject", "retry"}


def test_host_fields_drops_nested_and_caps_text():
    raw = json.loads(fix_error("w", "driver_not_running", retry=False, detail="d" * 5000))
    raw["nested"] = {"a": 1}
    got = host_fields(json.dumps(raw))
    assert "nested" not in got
    assert len(got["detail"]) == 512
    assert host_fields("[error] " + json.dumps(raw))["code"] == "driver_not_running"
    assert host_fields(raw)["retry"] is False
