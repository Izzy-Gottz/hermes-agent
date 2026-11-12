"""The warm path: a conversation keeps its child, every spawn says why, a
voice turn opens at a lower effort, and one line a turn says where the time went.

Same harness as ``test_claude_code_runtime.py`` (minimal agent stand-ins, the
fake ``claude``), kept in its own file so it travels as one piece.
"""

from __future__ import annotations

import asyncio
import json
import logging
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import model_tools
from agent import claude_code_runtime as rt

_FAKE = Path(__file__).parent / "transports" / "fake_claude_cli.py"


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-setup-token")
    effort_log = tmp_path / "effort.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_EFFORT_LOG", str(effort_log))
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {json.dumps(sys.executable)} {json.dumps(str(_FAKE))} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    cfg = {"binary": str(wrapper), "expose_hermes_tools": False, "prewarm": False}
    monkeypatch.setattr(rt, "_claude_code_config", lambda: dict(cfg))
    with rt._REGISTRY_LOCK:
        rt._REGISTRY.clear()
    with rt._GONE_LOCK:
        rt._GONE.clear()
    yield SimpleNamespace(cfg=cfg, effort_log=effort_log)
    for key in list(rt._REGISTRY):
        rt.evict_session(key)


def _agent(session_id: str, *, key: str = "", ephemeral=None, cached="BASE") -> SimpleNamespace:
    a = SimpleNamespace(
        session_id=session_id, _gateway_session_key=key, _cached_system_prompt=cached,
        ephemeral_system_prompt=ephemeral, model="sonnet", api_mode="claude_code",
        provider="claude-code-cli", base_url="claude-code://local", api_key="x",
        _interrupt_requested=False, _interrupt_message=None, _skill_nudge_interval=0,
        _iters_since_skill=0, valid_tool_names=set(), _session_db=None, session_api_calls=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_cost_status=None, session_cost_source=None,
        context_compressor=None, show_commentary=True,
    )
    a.clear_interrupt = lambda: None
    a._sync_external_memory_for_turn = lambda **kw: None
    a._spawn_background_review = lambda **kw: None
    return a


def _turn(agent, text="hello"):
    return rt.run_claude_code_turn(
        agent, user_message=text, original_user_message=text,
        messages=[{"role": "user", "content": text}], effective_task_id="t")


def _ran_at(env, text: str) -> str:
    rows = [json.loads(line) for line in env.effort_log.read_text().splitlines()]
    return next(row["effort"] for row in rows if row["what"] == f"user {text}")


class TestTheConversationKeepsItsChild:
    def test_one_off_sessions_do_not_push_the_named_conversation_out(self, _env):
        """max_sessions 2, the conversation used FIRST (so least recently), then
        three one-off sessions. Plain LRU evicts the conversation at the second
        of them; kept warm, it is still the same process at the end."""
        _env.cfg.update(max_sessions=2, prewarm_session_keys=["moe"])
        person = _agent("conversation", key="moe")
        _turn(person)
        pid = person._claude_code_session.pid
        for i in range(3):
            _turn(_agent(f"cron-{i}", key=f"cron-{i}"))
        again = _agent("conversation", key="moe")
        _turn(again)
        assert again._claude_code_session.pid == pid
        assert again._claude_code_session.is_alive()
        assert rt.registered_session_count() == 2          # the bound still holds

    def test_without_the_setting_it_is_the_plain_lru_it_was(self, _env):
        _env.cfg.update(max_sessions=2)
        person = _agent("conversation", key="moe")
        _turn(person)
        first = person._claude_code_session
        for i in range(2):
            _turn(_agent(f"cron-{i}", key=f"cron-{i}"))
        assert not first.is_alive()

    def test_with_only_kept_warm_sessions_the_oldest_still_goes(self, _env, caplog):
        _env.cfg.update(max_sessions=2, prewarm_session_keys=["moe"])
        a, b, c = (_agent(f"c{i}", key="moe") for i in range(3))
        _turn(a); _turn(b)
        with caplog.at_level(logging.INFO, logger="agent.claude_code_runtime"):
            _turn(c)
        assert not a._claude_code_session.is_alive()
        assert rt.registered_session_count() == 2
        assert "nothing else was left to evict" in caplog.text

    def test_a_lane_or_a_subagent_is_never_kept_warm(self, _env):
        _env.cfg.update(prewarm_session_keys=["moe"])
        assert rt._kept_warm(_agent("s", key="moe")) is True
        assert rt._kept_warm(_agent("s", key="other")) is False
        lane = _agent("s", key="moe"); lane._claude_code_lane = "background_review"
        assert rt._kept_warm(lane) is False
        child = _agent("s", key="moe"); child._parent_session_id = "p"
        assert rt._kept_warm(child) is False


class TestEverySpawnSaysWhy:
    def test_new_then_evicted_then_prompt_changed(self, _env, caplog):
        _env.cfg.update(max_sessions=1)
        with caplog.at_level(logging.INFO):
            first = _agent("A", ephemeral="one")
            _turn(first)
            assert first._claude_code_session.last_spawn_reason == "first (new conversation)"
            _turn(_agent("B"))                       # evicts A
            back = _agent("A", ephemeral="one")
            _turn(back)
            assert back._claude_code_session.last_spawn_reason.startswith("evicted-lru ")
            changed = _agent("A", ephemeral="two!")
            _turn(changed)
            reason = changed._claude_code_session.last_spawn_reason
            assert reason.startswith("prompt-changed (the host's half, ")
        assert 'reason="evicted-lru ' in caplog.text       # in the turn's timing line
        assert "reason=prompt-changed (the host's half" in caplog.text   # and the spawn's own

    def test_a_change_in_hermes_own_half_is_named_as_that(self, _env):
        a = _agent("A", ephemeral="host", cached="memory v1")
        _turn(a)
        b = _agent("A", ephemeral="host", cached="memory v2 longer")
        _turn(b)
        assert "hermes's half" in b._claude_code_session.last_spawn_reason

    def test_a_failed_turn_is_the_reason_for_the_next_spawn(self, _env):
        a = _agent("A")
        _turn(a)
        _turn(_agent("A"), "CRASH")
        b = _agent("A")
        _turn(b)
        assert b._claude_code_session.last_spawn_reason.startswith("retired-after-turn-error ")

    def test_a_session_closed_behind_the_registrys_back(self, _env):
        a = _agent("A")
        _turn(a)
        a._claude_code_session.close()               # what AIAgent.close() does
        b = _agent("A")
        _turn(b)
        assert b._claude_code_session.last_spawn_reason.startswith("closed-by-its-agent ")


class TestOneTimingLineATurn:
    def test_warm_and_cold_turns_each_log_one_and_no_words(self, _env, caplog):
        with caplog.at_level(logging.INFO, logger="agent.claude_code_runtime"):
            a = _agent("A"); a._turn_received_at = time.perf_counter(); a._turn_agent_built_at = time.perf_counter()
            _turn(a, "the secret word is plover")
            _turn(_agent("A"), "and again plover")
        lines = [r.getMessage() for r in caplog.records if "turn timing:" in r.getMessage()]
        assert len(lines) == 2
        assert "spawned=yes" in lines[0] and 'reason="first (new conversation)"' in lines[0]
        assert "build_agent_ms=" in lines[0] and "to_runtime_ms=" in lines[0] and "total_ms=" in lines[0]
        assert "spawned=no" in lines[1] and "reason=" not in lines[1]
        assert "plover" not in "".join(lines)


class TestAVoiceTurnsOpening:
    def _voice(self, agent, config):
        agent._voice_route_state = {"reasoning_config": None}   # what begin_voice_turn_route leaves
        agent.reasoning_config = config
        return agent

    def test_only_a_marked_turn_is_lowered(self, _env):
        _turn(_agent("A"), "typed first")
        _turn(self._voice(_agent("A"), {"enabled": False}), "spoken")
        _turn(_agent("A"), "typed after")
        assert _ran_at(_env, "typed first") == "medium"
        assert _ran_at(_env, "spoken") == "low"
        assert _ran_at(_env, "typed after") == "medium"

    def test_a_reasoning_setting_alone_is_not_a_voice_turn(self, _env):
        """``reasoning_config`` is also how a typed turn carries the person's
        own effort choice. Without the voice mark it is left to the session."""
        typed = _agent("A"); typed.reasoning_config = {"enabled": False}
        assert rt._opening_effort_for(typed) is None

    @pytest.mark.parametrize("config, expected", [
        ({"enabled": False}, "low"),                      # off -> the CLI's lowest; it has no off
        ({"enabled": True, "effort": "none"}, "low"),
        ({"enabled": True, "effort": "minimal"}, "low"),
        ({"enabled": True, "effort": "medium"}, "medium"),
        ({"enabled": True, "effort": "ultra"}, None),     # not on the CLI's scale: leave it
        ({"enabled": True}, None),
        (None, None),
    ])
    def test_the_level_is_on_the_clis_own_scale(self, config, expected):
        assert rt._opening_effort_for(self._voice(_agent("A"), config)) == expected


class TestTheVoiceHeader:
    """``X-Hermes-Voice-Turn`` marks the agent the way the CLI, TUI and
    messaging surfaces already mark theirs — and only for a person's turn."""

    def _marks(self, headers_list, monkeypatch):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter

        seen = []

        class _Agent:
            session_id = "s1"
            session_prompt_tokens = session_completion_tokens = session_total_tokens = 0

            def run_conversation(self, **_kw):
                seen.append((bool(getattr(self, "_voice_turn_pending", False)),
                             isinstance(getattr(self, "_turn_received_at", None), float)))
                return {"final_response": "ok", "messages": [], "api_calls": 0, "tools": []}

        async def go():
            adapter = APIServerAdapter(PlatformConfig(enabled=True))
            monkeypatch.setattr(adapter, "_create_agent", lambda **kw: _Agent())
            app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])

            async def handler(request):
                await adapter._run_agent("hello", [], session_id="s1")
                return web.json_response({})

            app.router.add_post("/turn", handler)
            async with TestClient(TestServer(app)) as client:
                for headers in headers_list:
                    assert (await client.post("/turn", headers=headers)).status == 200

        asyncio.run(go())
        return seen

    def test_marked_only_when_a_person_spoke(self, monkeypatch):
        person = {"X-Hermes-Turn-Origin": "person"}
        seen = self._marks([
            {**person, "X-Hermes-Voice-Turn": "1"},
            person,                                              # typed
            {"X-Hermes-Turn-Origin": "background", "X-Hermes-Voice-Turn": "1"},
            {"X-Hermes-Voice-Turn": "1"},                        # undeclared origin
            {**person, "X-Hermes-Voice-Turn": "0"},
            person,                                              # the mark does not outlive its request
        ], monkeypatch)
        assert [voice for voice, _ in seen] == [True, False, False, False, False, False]
        assert all(stamped for _, stamped in seen)


class TestToolDefinitionsCache:
    @pytest.fixture(autouse=True)
    def _clean(self):
        model_tools._tool_defs_cache.clear(); model_tools._tool_defs_last_key.clear()
        yield
        model_tools._tool_defs_cache.clear(); model_tools._tool_defs_last_key.clear()

    def test_the_selection_in_use_is_not_the_one_evicted(self, monkeypatch):
        """The conversation's selection is filled first and used every turn;
        other selections come and go. First-in eviction rebuilt it every time
        the others filled the cache; least-recently-used never does."""
        built = []
        monkeypatch.setattr(model_tools, "_compute_tool_definitions",
                            lambda enabled, disabled, quiet, skip_tool_search_assembly=False:
                            built.append(tuple(enabled or ())) or [{"type": "function", "function": {"name": "t"}}])
        monkeypatch.setattr(model_tools, "_TOOL_DEFS_CACHE_MAX", 3)
        conversation = ["conversation"]
        model_tools.get_tool_definitions(conversation, quiet_mode=True)
        for i in range(8):
            model_tools.get_tool_definitions([f"one-off-{i}"], quiet_mode=True)
            model_tools.get_tool_definitions(conversation, quiet_mode=True)
        assert built.count(("conversation",)) == 1
        assert len(model_tools._tool_defs_cache) == 3

    def test_a_miss_says_why(self, monkeypatch, caplog):
        monkeypatch.setattr(model_tools, "_compute_tool_definitions",
                            lambda *a, **k: [{"type": "function", "function": {"name": "t"}}])
        monkeypatch.setattr(model_tools, "_TOOL_DEFS_CACHE_MAX", 1)
        with caplog.at_level(logging.INFO, logger="model_tools"):
            model_tools.get_tool_definitions(["a"], quiet_mode=True)
            model_tools.get_tool_definitions(["b"], quiet_mode=True)     # pushes "a" out
            model_tools.get_tool_definitions(["a"], quiet_mode=True)
            monkeypatch.setattr(model_tools.registry, "_generation", model_tools.registry._generation + 1)
            model_tools.get_tool_definitions(["a"], quiet_mode=True)
        text = caplog.text
        assert text.count("first for this selection — cache miss)") == 2
        assert text.count("evicted — cache miss)") == 1
        assert "registry generation" in text and text.count("cache miss)") == 4
