"""Recipient grounding: the ledger the send gate reads to ask "who named this person?".

Incident, 2026-09-25: a cron job told "send ONE WhatsApp message to Yisrael" (the owner)
called ``whatsapp_send(to="Moshe Finkelman")`` — a name that existed only in the memory
block of the system prompt. These tests pin the half of the fix that lives in Hermes: the
person's words and the conversation's lookups are written down where the gate can read
them, keyed so the MCP server's tool calls and the runtime's turn meet, and memory is
never counted as a lookup.
"""

import json
import os

import pytest

from agent import recipient_grounding as rg


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv(rg.GROUNDING_KEY_ENV, raising=False)
    return tmp_path


def _ledger(key):
    with open(rg.ledger_path(key), encoding="utf-8") as fh:
        return json.load(fh)


JOB = ("[IMPORTANT: You are running as a scheduled cron job.]\n\nWhen you're done, send ONE "
       "consolidated WhatsApp message to Yisrael: a short list of what's drafted.")


def test_words_are_the_user_turns_with_memory_fenced_out(home):
    messages = [
        {"role": "system", "content": "Moshe Finkelman is grandfather (WhatsApp contact)."},
        {"role": "user", "content": "earlier: remind me to call Dan"},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": [{"type": "text", "text": JOB},
                                     {"type": "text", "text": "<memory-context>Moshe Finkelman</memory-context>"}]},
    ]
    rg.record_words("s1", messages)
    data = _ledger("s1")
    assert data["current"].startswith("[IMPORTANT")
    joined = "\n".join(data["words"])
    assert "Yisrael" in joined and "call Dan" in joined
    # The system prompt and the fenced memory block are context, not the person's words.
    assert "Moshe" not in joined


def test_results_are_recorded_but_memory_is_not(home):
    rg.record_result("s1", "whatsapp_chats", {}, '{"chats": ["Dan Levi (15551230000)"]}')
    rg.record_result("s1", "memory", {"action": "read"}, "Moshe Finkelman is grandfather")
    rg.record_result("s1", "read_file", {"path": "/Users/x/.hermes/memories/USER.md"}, "Moshe Finkelman")
    rows = _ledger("s1")["results"]
    assert [r["tool"] for r in rows] == ["whatsapp_chats"]
    assert "Dan Levi" in rows[0]["text"]


def test_words_replace_and_results_accumulate(home):
    rg.record_result("s1", "contacts_find", {}, "Dan Levi +1 555 123 0000")
    rg.record_words("s1", [{"role": "user", "content": "text Dan"}])
    rg.record_words("s1", [{"role": "user", "content": "text Dan"}, {"role": "user", "content": "and Ruth"}])
    data = _ledger("s1")
    assert data["words"] == ["text Dan", "and Ruth"]
    assert data["current"] == "and Ruth"
    assert len(data["results"]) == 1


def test_ledger_file_is_private(home):
    rg.record_words("s1", [{"role": "user", "content": "hi"}])
    assert oct(os.stat(rg.ledger_path("s1")).st_mode & 0o777) == "0o600"


def test_key_prefers_the_session_env_key(home, monkeypatch):
    assert rg.current_key("sess") == "sess"
    monkeypatch.setenv(rg.GROUNDING_KEY_ENV, "cc:abc")
    assert rg.current_key("sess") == "cc:abc"
    assert rg.current_key("") == "cc:abc"


def test_handle_function_call_records_the_result_under_the_env_key(home, monkeypatch):
    """The MCP server's tool calls carry no session id: the env key is what files them."""
    import model_tools

    monkeypatch.setenv(rg.GROUNDING_KEY_ENV, "cc:server")
    monkeypatch.setattr(model_tools, "_execute_tool",
                        lambda *a, **k: '{"matches": ["Ruth Cohen (15550001111)"]}')
    model_tools.handle_function_call("contacts_lookup_for_test", {"query": "Ruth"},
                                     skip_pre_tool_call_hook=True)
    rows = _ledger("cc:server")["results"]
    assert rows and rows[-1]["tool"] == "contacts_lookup_for_test"
    assert "Ruth Cohen" in rows[-1]["text"]


def test_mcp_config_carries_the_grounding_key(home, tmp_path):
    from agent.transports.claude_code_session import write_mcp_config

    path = write_mcp_config(directory=str(tmp_path), credential_env={}, grounding_key="cc:k1")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    (server,) = payload["mcpServers"].values()
    assert server["env"][rg.GROUNDING_KEY_ENV] == "cc:k1"


def test_each_session_has_its_own_grounding_key():
    from agent.transports.claude_code_session import ClaudeCodeSession

    a, b = ClaudeCodeSession(), ClaudeCodeSession()
    assert a.grounding_key.startswith("cc:") and a.grounding_key != b.grounding_key


def test_cron_preamble_bans_every_send_by_behaviour_not_by_name():
    from cron.scheduler_prompt import _CRON_HINT

    hint = _CRON_HINT
    # The old text banned `send_message` by name; the model sent with whatsapp_send.
    for route in ("WhatsApp", "Telegram", "email", "connector", "shell", "browser"):
        assert route in hint, route
    assert "never take a recipient from memory" in hint
    assert "put it in your final response" in hint
