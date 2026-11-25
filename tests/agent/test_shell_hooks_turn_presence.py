"""A tool hook's payload says who started the turn, on Hermes's own loop.

A hook is a subprocess: the turn's origin lives in this process's contextvars,
which its environment does not carry. Under the claude-code runtime a hook asks
over the tool bridge; on the native loop there is none, so a scheduled job's
send used to wait out the hook ceiling on a confirm card nobody saw (Memoe's
confirm-send.sh). ``extra.turn_presence`` is that answer, judged in-process.
"""

import contextvars

from agent.shell_hooks import _payload_fields
from gateway.session_context import clear_session_vars, set_session_vars


def _in_fresh_context(fn):
    return contextvars.copy_context().run(fn)


def test_a_cron_turn_is_not_live_and_says_why(monkeypatch):
    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)

    def run():
        tokens = set_session_vars(platform="cron", session_id="cron_job1_20261125_090000", cron_session="1")
        try:
            return _payload_fields({"tool_name": "send_message", "args": {}})
        finally:
            clear_session_vars(tokens)

    assert _in_fresh_context(run)["extra"]["turn_presence"] == {"live": False, "why": "a scheduled job"}


def test_a_persons_chat_turn_is_live(monkeypatch):
    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)

    def run():
        tokens = set_session_vars(platform="telegram", chat_id="1", session_id="s1")
        try:
            return _payload_fields({"tool_name": "send_message", "args": {}})
        finally:
            clear_session_vars(tokens)

    verdict = _in_fresh_context(run)["extra"]["turn_presence"]
    assert verdict["live"] is True


def test_inside_the_tools_mcp_server_it_is_unknown_and_the_bridge_answers(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "claude-code")

    def run():
        tokens = set_session_vars(platform="cron", cron_session="1")
        try:
            return _payload_fields({"tool_name": "send_message", "args": {}})
        finally:
            clear_session_vars(tokens)

    assert _in_fresh_context(run)["extra"]["turn_presence"] is None


def test_only_tool_events_carry_it():
    assert "turn_presence" not in _payload_fields({"session_id": "s"})["extra"]
