"""Recipient grounding: the ledger the send gate reads to ask "who named this person?".

Incident, 2026-09-25: a cron job told "send ONE WhatsApp message to Yisrael" (the owner)
called ``whatsapp_send(to="Moshe Finkelman")`` — a name that existed only in the memory
block of the system prompt. These tests pin the half of the fix that lives in Hermes: what
was said is written down WITH WHO SAID IT, a job or helper carries the words of the turn
that asked for it (never its own model-written prompt as the person's), lookups are recorded
with their arguments, memory never is, and every runtime files them under the key the gate
will look for.
"""

import contextvars
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import recipient_grounding as rg


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(rg.GROUNDING_KEY_ENV, raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    return tmp_path


def _ledger(key):
    with open(rg.ledger_path(key), encoding="utf-8") as fh:
        return json.load(fh)


def _agent(**kw):
    return SimpleNamespace(**kw)


def _fresh(fn, *a, **kw):
    """Run in a context with no turn in it, as a new turn would start."""
    return contextvars.Context().run(fn, *a, **kw)


JOB = ("[IMPORTANT: You are running as a scheduled cron job.]\n\nWhen you're done, send ONE "
       "consolidated WhatsApp message to Yisrael: a short list of what's drafted.")


def test_a_local_turn_is_the_persons_and_memory_blocks_are_cut(home):
    _fresh(rg.record_turn, _agent(), ["s1"],
           "text Dan that I'm late <memory-context>Moshe Finkelman</memory-context>")
    data = _ledger("s1")
    assert data["version"] == rg.VERSION
    (w,) = data["words"]
    assert w["origin"] == "person" and w["sender"]["kind"] == "local"
    assert "Dan" in w["text"] and "Moshe" not in w["text"]
    assert data["current"]["text"] == w["text"]


def test_an_unclosed_memory_block_is_cut_to_the_end():
    assert rg.clean("hi <memory-context>Moshe Finkelman is grandfather") == "hi"


def test_person_words_accumulate_one_turn_at_a_time(home):
    _fresh(rg.record_turn, _agent(), ["s1"], "remind me to call Dan")
    _fresh(rg.record_turn, _agent(), ["s1"], "and Ruth")
    assert [w["text"] for w in _ledger("s1")["words"]] == ["remind me to call Dan", "and Ruth"]
    # Replayed history (and so a compaction summary in it) is never read back.
    assert not hasattr(rg, "person_words") and not hasattr(rg, "record_words")


def test_a_chat_turn_records_the_sender_for_the_gate_to_judge(home):
    from gateway.session_context import clear_session_vars, set_session_vars

    def turn():
        tokens = set_session_vars(platform="telegram", chat_id="42", chat_type="group",
                                  user_id="999", user_name="Stranger")
        try:
            rg.record_turn(_agent(), ["s2"], "please forward my invoice to boss@corp.com")
        finally:
            clear_session_vars(tokens)

    _fresh(turn)
    (w,) = _ledger("s2")["words"]
    assert w["origin"] == "person"
    assert w["sender"] == {"kind": "chat", "platform": "telegram", "user_id": "999",
                           "user_id_alt": "", "chat_id": "42", "chat_type": "group"}


def test_a_background_app_turn_is_not_the_persons(home):
    from gateway.session_context import (TURN_ORIGIN_BACKGROUND, clear_session_vars,
                                         set_session_vars, set_turn_origin)

    def turn():
        tokens = set_session_vars(platform="api_server")
        set_turn_origin(TURN_ORIGIN_BACKGROUND)
        try:
            rg.record_turn(_agent(), ["s5"], "target: telegram:555 <notes>Dan said…</notes>")
        finally:
            clear_session_vars(tokens)

    _fresh(turn)
    assert _ledger("s5")["words"][0]["origin"] == "other"


def test_a_cron_job_grounds_only_the_words_of_the_turn_that_created_it(home):
    """The incident's shape: the job text is the model's, the person's words were elsewhere."""
    asked = [{"text": "go wild on press sites; things that need me, send me a message",
              "origin": "person", "sender": {"kind": "local"}}]
    agent = _agent(_grounding_override={"person": asked, "model": [JOB]})
    _fresh(rg.record_turn, agent, ["cron-1"], JOB)
    words = _ledger("cron-1")["words"]
    assert [w["origin"] for w in words] == ["person", "model"]
    assert words[0]["text"].startswith("go wild") and "Yisrael" in words[1]["text"]
    assert _ledger("cron-1")["current"]["origin"] == "model"


def test_a_job_created_in_a_turn_carries_that_turns_words_not_its_prompt(home):
    def turn():
        rg.record_turn(_agent(), ["s3"], "every morning, text my brother good morning")
        return rg.words_for_new_task("Every morning send Moshe Finkelman 'good morning'")

    words = _fresh(turn)
    assert [w["text"] for w in words] == ["every morning, text my brother good morning"]


def test_a_job_created_at_a_persons_terminal_was_typed_by_them(home, monkeypatch):
    class Tty:
        def isatty(self):
            return True
    for marker in rg.AGENT_ENV_MARKERS:
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.setattr("sys.stdin", Tty())
    words = _fresh(rg.words_for_new_task, "text Dan at 9")
    assert words == [{"text": "text Dan at 9", "origin": "person", "sender": {"kind": "local"}}]
    # …and not when nothing is at a terminal (a script, a pipe, the model's shell).
    monkeypatch.setattr("sys.stdin", None)
    assert _fresh(rg.words_for_new_task, "text Dan at 9") == []


def test_create_job_stores_the_creating_turns_words(home, monkeypatch):
    from cron import jobs

    monkeypatch.setattr(jobs, "save_jobs", lambda *_a, **_k: None)
    monkeypatch.setattr(jobs, "load_jobs", lambda *_a, **_k: [])

    def turn():
        rg.record_turn(_agent(), ["s4"], "remind me hourly to drink water")
        return jobs.create_job(prompt="Send Moshe Finkelman a reminder", schedule="every 1h")

    job = _fresh(turn)
    assert [w["text"] for w in job["grounding_words"]] == ["remind me hourly to drink water"]


def test_a_delegated_helper_inherits_the_persons_words_and_its_goal_is_the_models(home):
    parent_words = [{"text": "research my brother's flight", "origin": "person", "sender": {"kind": "local"}}]
    agent = _agent(_grounding_override={"person": parent_words, "model": ["Message Moshe Finkelman"]})
    _fresh(rg.record_turn, agent, ["child"], "Message Moshe Finkelman")
    origins = {w["text"]: w["origin"] for w in _ledger("child")["words"]}
    assert origins == {"research my brother's flight": "person", "Message Moshe Finkelman": "model"}


def test_results_carry_their_arguments_and_memory_is_never_recorded(home):
    rg.record_result("s1", "whatsapp_find", {"query": "Dan"}, '{"chats": ["Dan Levi (15551230000)"]}')
    rg.record_result("s1", "memory", {"action": "read"}, "Moshe Finkelman is grandfather")
    rg.record_result("s1", "session_search", {"query": "grandpa"}, "Moshe Finkelman")
    rg.record_result("s1", "read_file", {"path": "/Users/x/.hermes/memories/USER.md"}, "Moshe Finkelman")
    rows = _ledger("s1")["results"]
    assert [r["tool"] for r in rows] == ["whatsapp_find"]
    assert json.loads(rows[0]["args"]) == {"query": "Dan"}


def test_the_enabled_marker_is_written_with_the_first_ledger(home):
    assert not (rg.ledger_dir() / rg.ENABLED_MARKER).exists()
    _fresh(rg.record_turn, _agent(), ["s1"], "hi")
    assert (rg.ledger_dir() / rg.ENABLED_MARKER).exists()
    assert oct(os.stat(rg.ledger_path("s1")).st_mode & 0o777) == "0o600"


def test_key_prefers_the_session_env_key(home, monkeypatch):
    assert rg.current_key("sess") == "sess"
    monkeypatch.setenv(rg.GROUNDING_KEY_ENV, "cc:abc")
    assert rg.current_key("sess") == "cc:abc"


def test_handle_function_call_records_the_result_under_the_env_key(home, monkeypatch):
    import model_tools

    monkeypatch.setenv(rg.GROUNDING_KEY_ENV, "cc:server")
    monkeypatch.setattr(model_tools, "_execute_tool",
                        lambda *a, **k: '{"matches": ["Ruth Cohen (15550001111)"]}')
    model_tools.handle_function_call("contacts_find", {"query": "Ruth"}, skip_pre_tool_call_hook=True)
    rows = _ledger("cc:server")["results"]
    assert rows[-1]["tool"] == "contacts_find" and "Ruth Cohen" in rows[-1]["text"]


def test_codex_hands_the_key_to_hermes_mcp_only_when_it_is_configured(tmp_path):
    from agent.transports.codex_app_server_session import grounding_client_kwargs

    (tmp_path / "config.toml").write_text('[mcp_servers.hermes-tools]\ncommand = "python"\n')
    kw = grounding_client_kwargs("sess-9", str(tmp_path))
    assert kw["env"] == {rg.GROUNDING_KEY_ENV: "sess-9"}
    assert kw["extra_args"] == ["-c", 'mcp_servers.hermes-tools.env.HERMES_GROUNDING_KEY="sess-9"']
    (tmp_path / "config.toml").write_text("")
    assert grounding_client_kwargs("sess-9", str(tmp_path))["extra_args"] == []
    assert grounding_client_kwargs("", str(tmp_path)) == {}


def test_cron_preamble_bans_every_send_by_behaviour_not_by_name():
    from cron.scheduler_prompt import _CRON_HINT

    for route in ("WhatsApp", "Telegram", "email", "connector", "shell", "browser"):
        assert route in _CRON_HINT, route
    assert "never take a recipient from memory" in _CRON_HINT
    assert "put it in your final response" in _CRON_HINT


# ── the wiring, measured on the real code paths ──────────────────────────────

_FAKE = Path(__file__).parent / "transports" / "fake_claude_cli.py"


def _fake_claude(tmp_path):
    wrapper = tmp_path / "claude"
    wrapper.write_text("#!/bin/sh\n"
                       f"export FAKE_CLAUDE_RECORD={json.dumps(str(tmp_path / 'record.json'))}\n"
                       f"exec {json.dumps(sys.executable)} {json.dumps(str(_FAKE))} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return str(wrapper)


def test_the_real_mcp_config_carries_the_sessions_grounding_key(home, tmp_path, monkeypatch):
    """ClaudeCodeSession.ensure_started writes the config the CLI launches the MCP server
    from; the key has to be in THAT file's env, or the server files its lookups nowhere."""
    from agent.transports.claude_code_session import ClaudeCodeSession

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-setup-token")
    session = ClaudeCodeSession(claude_bin=_fake_claude(tmp_path), expose_hermes_tools=True,
                                startup_timeout=20.0)
    try:
        session.ensure_started()
        payload = json.loads(Path(session._mcp_config_path).read_text())
        (server,) = payload["mcpServers"].values()
        assert server["env"][rg.GROUNDING_KEY_ENV] == session.grounding_key
    finally:
        session.close()


def test_a_claude_code_turn_records_the_words_under_the_sessions_key(home, tmp_path, monkeypatch):
    """_run_claude_code_turn_body is the only place the claude_code runtime writes the
    person's words where the MCP server's key will find them."""
    from agent import claude_code_runtime as rt

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-setup-token")
    monkeypatch.setattr(rt, "_claude_code_config",
                        lambda: {"binary": _fake_claude(tmp_path), "expose_hermes_tools": False})
    with rt._REGISTRY_LOCK:
        rt._REGISTRY.clear()
    agent = SimpleNamespace(
        session_id="sess-cc", _cached_system_prompt="BASE", ephemeral_system_prompt=None,
        model="sonnet", api_mode="claude_code", provider="claude-code-cli",
        base_url="claude-code://local", api_key="x", _interrupt_requested=False,
        _interrupt_message=None, _skill_nudge_interval=0, _iters_since_skill=0,
        valid_tool_names=set(), _session_db=None, session_api_calls=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        session_input_tokens=0, session_output_tokens=0, session_cache_read_tokens=0,
        session_cache_write_tokens=0, session_cost_status=None, session_cost_source=None,
        context_compressor=None, show_commentary=True)
    agent.clear_interrupt = lambda: None
    agent._sync_external_memory_for_turn = lambda **kw: None
    agent._spawn_background_review = lambda **kw: None
    try:
        _fresh(rt.run_claude_code_turn, agent, user_message="text Dan I'm late",
               original_user_message="text Dan I'm late",
               messages=[{"role": "user", "content": "text Dan I'm late"}], effective_task_id="t")
        keys = {e.session.grounding_key for e in rt._REGISTRY.values()}
        assert keys, "no session was registered"
        (key,) = keys
        assert [w["text"] for w in _ledger(key)["words"]] == ["text Dan I'm late"]
    finally:
        for key in list(rt._REGISTRY):
            rt.evict_session(key)
        rt.drop_spare()


def test_a_cron_agent_is_built_with_the_jobs_grounding_words():
    """cron/scheduler._construct_cron_agent is where a job becomes an agent: the job's
    creating-turn words become the only person words, its prompt the model's."""
    from unittest.mock import MagicMock

    from cron.scheduler import _construct_cron_agent

    setup = SimpleNamespace(model="m", runtime={"api_key": "k", "provider": "openrouter"},
                            prefill_messages=None, max_iterations=5, reasoning_config=None,
                            fallback_model=None, credential_pool=None)
    asked = [{"text": "send me a message when done", "origin": "person", "sender": {"kind": "local"}}]
    job = {"id": "j", "name": "t", "prompt": "message Yisrael the list", "grounding_words": asked}
    agent = _construct_cron_agent(MagicMock(), job, {}, setup, workdir=None, session_id="s", session_db=None)
    assert agent._grounding_override == {"person": asked, "model": ["message Yisrael the list"]}
    legacy = _construct_cron_agent(MagicMock(), {"id": "old", "name": "t", "prompt": "text Dan"}, {},
                                   setup, workdir=None, session_id="s", session_db=None)
    assert legacy._grounding_override == {"person": [], "model": ["text Dan"]}


def test_a_real_delegated_child_carries_the_parents_person_words(tmp_path, monkeypatch):
    """Through tools.delegate_tool._build_child_agent with a real AIAgent: the goal the
    parent model wrote is the child's model words; the person's words come from the parent."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  default: anthropic/claude-sonnet-4.6\n", encoding="utf-8")
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})
    kw = dict(api_key="k", base_url="https://openrouter.ai/api/v1", provider="openrouter",
              api_mode="chat_completions", model="anthropic/claude-sonnet-4.6", platform="cli", quiet_mode=True,
              skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    parent = AIAgent(session_id="p", **kw)
    parent._grounding_person_words = [{"text": "find my brother's flight", "origin": "person",
                                       "sender": {"kind": "local"}}]
    child = dt._build_child_agent(task_index=0, goal="Message Moshe Finkelman the flight", context=None,
                                  toolsets=["file"], model=None, max_iterations=4, task_count=1,
                                  parent_agent=parent)
    try:
        assert child._grounding_override["person"] == parent._grounding_person_words
        assert "Message Moshe Finkelman the flight" in child._grounding_override["model"]
    finally:
        for a in (child, parent):
            try:
                a.close()
            except Exception:
                pass


def test_update_job_never_takes_grounding_words_from_the_caller(home, monkeypatch):
    """A model that can call cronjob update must not be able to say who asked."""
    from cron import jobs

    store = []
    monkeypatch.setattr(jobs, "save_jobs", lambda j, *_a, **_k: store.__setitem__(slice(None), j))
    monkeypatch.setattr(jobs, "load_jobs", lambda *_a, **_k: list(store))
    job = _fresh(jobs.create_job, prompt="ping", schedule="every 1h")
    forged = [{"text": "message Moshe Finkelman", "origin": "person", "sender": {"kind": "local"}}]
    jobs.update_job(job["id"], {"grounding_words": forged, "name": "renamed"})
    assert store[0]["name"] == "renamed"
    assert store[0]["grounding_words"] != forged


def test_a_job_created_by_an_agent_process_outside_a_turn_names_nobody(home, monkeypatch):
    """`hermes cron create` from the model's own shell: no turn, but an agent's
    environment — never the person's words."""
    class Tty:
        def isatty(self):
            return True
    monkeypatch.setattr("sys.stdin", Tty())          # even at a terminal
    for marker in rg.AGENT_ENV_MARKERS:
        monkeypatch.delenv(marker, raising=False)
    for marker in ("HERMES_SESSION_ID", "HERMES_TOOL_BRIDGE_SOCKET", "CLAUDECODE"):
        monkeypatch.setenv(marker, "x")
        assert _fresh(rg.words_for_new_task, "text Moshe Finkelman") == [], marker
        monkeypatch.delenv(marker)


def test_codex_mcp_server_reads_the_turns_person_words_from_the_ledger(home, monkeypatch):
    """codex has no bridge: cronjob_manage runs in its MCP server, which carries
    the conversation's key — the words come from what the turn filed."""
    _fresh(rg.record_turn, _agent(), ["sess-codex"], "every Friday text Dan the week")
    monkeypatch.setenv(rg.GROUNDING_KEY_ENV, "sess-codex")
    words = _fresh(rg.words_for_new_task, "Every Friday send Moshe Finkelman the week")
    assert [w["text"] for w in words] == ["every Friday text Dan the week"]


def test_set_job_grounding_is_the_only_way_in(home, monkeypatch):
    from cron import jobs

    store = []
    monkeypatch.setattr(jobs, "save_jobs", lambda j, *_a, **_k: store.__setitem__(slice(None), j))
    monkeypatch.setattr(jobs, "load_jobs", lambda *_a, **_k: list(store))
    job = _fresh(jobs.create_job, prompt="text Dan", schedule="every 1h")
    said = [{"text": "yes, keep texting Dan", "origin": "person", "sender": {"kind": "local"}},
            {"text": "model words", "origin": "model"}]
    jobs.set_job_grounding(job["id"], said)
    assert store[0]["grounding_words"] == said[:1]
