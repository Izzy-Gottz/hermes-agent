"""A voice turn's opening runs at a lower effort, and nothing else does.

The fake CLI answers the two control requests the way the real one was
measured to (see ``fake_claude_cli.py``), and logs the effort in force at each
user message — so these read what a turn RAN at, not what was asked for. An
acknowledgement is not an effect: the real CLI acknowledges a level it does not
know and a thinking budget it then ignores.
"""

from __future__ import annotations

import json
import logging
import stat
import sys
from pathlib import Path

import pytest

from agent.transports.claude_code_session import CLI_EFFORTS, ClaudeCodeSession, effort_from_args

_FAKE = Path(__file__).with_name("fake_claude_cli.py")


@pytest.fixture
def cli(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-setup-token")
    log = tmp_path / "effort.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_EFFORT_LOG", str(log))
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {json.dumps(sys.executable)} {json.dumps(str(_FAKE))} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    def read() -> list[tuple[str, str]]:
        if not log.exists():
            return []
        return [(row["what"], row["effort"]) for row in map(json.loads, log.read_text().splitlines())]

    def make(**kwargs) -> ClaudeCodeSession:
        kwargs.setdefault("expose_hermes_tools", False)
        kwargs.setdefault("startup_timeout", 20.0)
        return ClaudeCodeSession(claude_bin=str(wrapper), **kwargs)

    return make, read


def _ran_at(rows, text: str) -> str:
    """The effort in force when the user message starting ``text`` arrived."""
    return next(effort for what, effort in rows if what == f"user {text}")


def test_a_spoken_turn_opens_low_and_the_typed_turn_after_it_is_untouched(cli):
    make, read = cli
    with make() as session:
        session.ensure_started()
        assert session.run_turn("typed one").final_text == "echo: typed one"
        assert session.run_turn("spoken", opening_effort="low").final_text == "echo: spoken"
        assert session.run_turn("typed two").final_text == "echo: typed two"
    rows = read()
    assert _ran_at(rows, "typed one") == "medium"
    assert _ran_at(rows, "spoken") == "low"
    # The side effect that matters most: the turn AFTER is back where it was.
    assert _ran_at(rows, "typed two") == "medium"
    whats = [what for what, _ in rows]
    assert whats.index("apply low") < whats.index("user spoken") < whats.index("apply None")


def test_the_session_effort_is_put_back_by_name_not_by_null(cli):
    """``null`` returns the real CLI to ITS default, not to the --effort it
    was spawned with (measured). A session spawned at high must end at high."""
    make, read = cli
    with make(extra_args=["--effort", "high"]) as session:
        session.ensure_started()
        session.run_turn("spoken", opening_effort="low")
        session.run_turn("typed")
    rows = read()
    assert _ran_at(rows, "spoken") == "low"
    assert _ran_at(rows, "typed") == "high"
    assert ("apply high", "high") in rows and "apply None" not in [w for w, _ in rows]


def test_the_work_after_a_tool_call_runs_at_the_session_effort(cli):
    """Only the opening is lowered. The fake holds its tool result until the
    effort is changed again, so a session that put the effort back only when
    the turn ended would never get a result at all — it would time out."""
    make, read = cli
    with make() as session:
        session.ensure_started()
        result = session.run_turn("please TOOLWAIT", opening_effort="low", idle_timeout=5.0)
        assert result.error is None and result.tool_iterations == 1
        assert result.final_text == "The version is 6.2"
    rows = read()
    assert _ran_at(rows, "please TOOLWAIT") == "low"
    # The tool's result — and so everything the model did with it — at medium.
    assert ("tool result", "medium") in rows
    assert [what for what, _ in rows].count("apply None") == 1


def test_a_session_already_at_that_effort_sends_nothing(cli):
    make, read = cli
    with make(extra_args=["--effort=low"]) as session:
        session.ensure_started()
        session.run_turn("spoken", opening_effort="low")
    assert not [what for what, _ in read() if what.startswith("apply")]


def test_a_level_the_cli_does_not_have_is_never_sent(cli):
    make, read = cli
    with make() as session:
        session.ensure_started()
        session.run_turn("spoken", opening_effort="none")
    assert not [what for what, _ in read() if what.startswith("apply")]


def test_a_cli_that_never_answers_costs_one_short_wait_and_is_not_asked_again(cli, monkeypatch, caplog):
    make, read = cli
    monkeypatch.setenv("FAKE_CLAUDE_EFFORT", "silent")
    with make() as session:
        session.ensure_started()
        with caplog.at_level(logging.WARNING, logger="agent.transports.claude_code_session"):
            assert session.run_turn("spoken one", opening_effort="low").final_text == "echo: spoken one"
            assert session.run_turn("spoken two", opening_effort="low").final_text == "echo: spoken two"
        assert session._effort_adjustable is False
    assert caplog.text.count("did not take a per-turn effort") == 1


def test_an_acknowledgement_without_the_effect_is_not_believed(cli, monkeypatch, caplog):
    """The real CLI says ``success`` to things it then ignores. The session
    reads the effort back, and when it is not the one asked for it stops."""
    make, read = cli
    monkeypatch.setenv("FAKE_CLAUDE_EFFORT", "ack-only")
    with make() as session:
        session.ensure_started()
        with caplog.at_level(logging.WARNING, logger="agent.transports.claude_code_session"):
            session.run_turn("spoken", opening_effort="low")
        assert session._effort_adjustable is False and session._opening_effort_active is False
    assert "did not take a per-turn effort" in caplog.text
    assert _ran_at(read(), "spoken") == "medium"


def test_effort_from_args():
    assert effort_from_args(["--effort", "high"]) == "high"
    assert effort_from_args(["--effort=max", "--effort", "low"]) == "low"
    assert effort_from_args(["--effort", "none"]) is None
    assert effort_from_args(None) is None
    assert CLI_EFFORTS[0] == "low"


class TestSpawnReason:
    """Every spawn says why. Twenty-six of thirty-two respawns under a spoken
    turn on one Mac had no cause in any log."""

    def test_first_then_a_child_that_died_then_a_changed_prompt(self, cli, caplog):
        make, _ = cli
        with make(system_prompt="ONE") as session:
            with caplog.at_level(logging.INFO, logger="agent.transports.claude_code_session"):
                session.ensure_started()
                assert session.last_spawn_reason == "first"
                assert session.run_turn("CRASH").should_retire
                session.ensure_started()
                assert session.last_spawn_reason.startswith("child-exited (pid ")
                assert "code 3" in session.last_spawn_reason
                session.restart(system_prompt="TWO")
                assert session.last_spawn_reason == "prompt-changed"
            assert caplog.text.count("claude-code session spawned") == 3
            assert "reason=first" in caplog.text and "reason=prompt-changed" in caplog.text

    def test_the_callers_account_wins_and_is_used_once(self, cli):
        make, _ = cli
        with make() as session:
            session.note_spawn_reason("evicted-lru 40s ago")
            session.ensure_started()
            assert session.last_spawn_reason == "evicted-lru 40s ago"
            session.restart()
            assert session.last_spawn_reason == "restart"

    def test_closing_a_live_session_says_who_did_it(self, cli, caplog):
        make, _ = cli
        session = make()
        session.ensure_started()
        with caplog.at_level(logging.INFO, logger="agent.transports.claude_code_session"):
            session.close()
        assert "closing a live session" in caplog.text
        assert "test_closing_a_live_session_says_who_did_it" in caplog.text
