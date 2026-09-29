"""Background helpers the model was never told about — Moe, 2026-09-29.

Seven helpers (``deleg_fe9575cd``) finished; a ``deliver_detached_completion`` plugin took the completion for an
inbox nobody read; the ledger said ``delivered``; and in a new conversation ``delegate_task(action="list")``
answered "count 0 … children that already finished have delivered their results", so the model told the owner it
had never started any. These tests pin the three repairs against the real ledger (a sandboxed state.db):

1. a plugin hand-off settles as ``plugin`` (unconfirmed), and the next client turn of the originating session
   hands the result over once — idempotent against the plugin's own delivery arriving later;
2. ``list`` reports finished helpers from this conversation and the owner's others, truthfully;
3. a fresh/respawned conversation is told, in its first user turn, about helpers from another conversation.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from tools import async_delegation as ad
from tools import async_delegation_recall as recall

SESSION = "eb88d002-3c5b-f5a3-b635-524a5a255810"
OTHER = "0f1e2d3c-new-conversation"
OWNER = "moe"


@pytest.fixture(autouse=True)
def _clean():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _finished(delegation_id, *, session=SESSION, owner=OWNER, completed_ago=600.0, goals=None, summaries=None):
    """A finished background batch as the real dispatch + completion write it."""
    now = time.time()
    goals = goals or ["Research AI marketing platforms. Write findings to /Users/x/research/01-platforms.md",
                      "Research agent fleets. Write findings to /Users/x/research/02-fleets.md"]
    summaries = summaries or ["Platforms: 12 compared, Jasper and Copy.ai lead.", "Fleets: n8n + Claude Code."]
    ad._persist_dispatch({
        "delegation_id": delegation_id, "session_key": owner, "origin_ui_session_id": "",
        "parent_session_id": session, "origin_session_id": session, "dispatched_at": now - completed_ago - 270,
        "goal": f"{len(goals)} parallel subagents: research", "goals": goals, "is_batch": True,
    })
    results = [{"task_index": i, "status": "completed", "summary": s, "api_calls": 3}
               for i, s in enumerate(summaries)]
    event = {"type": "async_delegation", "delegation_id": delegation_id, "session_key": owner,
             "origin_session_id": session, "parent_session_id": session, "goals": goals,
             "goal": f"{len(goals)} parallel subagents: research", "is_batch": True, "results": results,
             "status": "completed", "completed_at": now - completed_ago, "dispatched_at": now - completed_ago - 270}
    ad._persist_completion(event, {"results": results})
    return event


def _hand_to_plugin(delegation_id):
    assert ad.claim_completion_delivery(delegation_id, "c1")
    assert ad.hand_completion_to_plugin(delegation_id, "c1")


def _state(delegation_id):
    row = ad.get_durable_delegation(delegation_id)
    with ad._DB_LOCK, ad._transaction() as conn:
        via = conn.execute("SELECT delivered_via FROM async_delegations WHERE delegation_id=?",
                           (delegation_id,)).fetchone()[0]
    return row["delivery_state"], via


# ── 1. durable fallback ─────────────────────────────────────────────────────

class TestPluginHandOff:
    def test_a_plugin_hand_off_is_not_delivered(self):
        _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        assert _state("deleg_a") == ("plugin", "plugin")
        # Terminal for the gateway's replay: a restart must not push it at the plugin again.
        import queue
        q = queue.Queue()
        ad.restore_undelivered_completions(q)
        assert q.empty()

    def test_within_the_grace_window_the_plugin_keeps_it(self):
        _finished("deleg_a", completed_ago=30)
        _hand_to_plugin("deleg_a")
        assert recall.turn_preamble([SESSION], OWNER, "what's up", fresh=False) == ""
        assert _state("deleg_a")[0] == "plugin"

    def test_next_turn_of_the_session_hands_it_over_exactly_once(self):
        _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        first = recall.turn_preamble([SESSION], OWNER, "hi again", fresh=False)
        assert "[ASYNC DELEGATION BATCH COMPLETE — deleg_a]" in first
        assert "Platforms: 12 compared" in first and "never reached you" in first
        assert _state("deleg_a") == ("delivered", "fallback")
        assert recall.turn_preamble([SESSION], OWNER, "and again", fresh=False) == ""

    def test_the_plugins_own_delivery_arriving_marks_it_seen_without_a_second_copy(self):
        evt = _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        from tools.process_registry_notifications import format_process_notification
        plugin_turn = "(Background note…)\n<helper-result>\n" + format_process_notification(evt) + "\n</helper-result>"
        assert recall.turn_preamble([SESSION], OWNER, plugin_turn, fresh=False) == ""
        assert _state("deleg_a") == ("delivered", "plugin-turn")

    def test_a_late_plugin_delivery_after_the_fallback_is_flagged_as_already_given(self):
        evt = _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        assert recall.turn_preamble([SESSION], OWNER, "hi", fresh=False)
        from tools.process_registry_notifications import format_process_notification
        late = recall.turn_preamble([SESSION], OWNER, format_process_notification(evt), fresh=False)
        assert "already given to you" in late
        assert "Platforms: 12 compared" not in late  # the preamble does not inject it a second time

    def test_another_sessions_plugin_row_is_not_injected_into_this_one(self):
        _finished("deleg_a", session=OTHER)
        _hand_to_plugin("deleg_a")
        assert recall.turn_preamble([SESSION], OWNER, "hi", fresh=False) == ""
        assert _state("deleg_a")[0] == "plugin"


@pytest.mark.asyncio
async def test_gateway_settles_a_plugin_taken_completion_as_plugin(monkeypatch):
    """Through the real delivery path: claim → inject → api_server self-post → plugin hook → settle."""
    import hermes_cli.plugins as plugins
    import gateway.wake as wake
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from unittest.mock import AsyncMock

    evt = _finished("deleg_g")
    evt = {k: v for k, v in evt.items() if k != "parent_session_id"}  # skip the session-liveness preflight
    persist = AsyncMock()
    monkeypatch.setattr(wake, "persist_delegation_delivery", persist)
    monkeypatch.setattr(plugins, "has_hook", lambda name: name == "deliver_detached_completion")
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kw: [True])
    runner = GatewayRunner(GatewayConfig())
    runner.adapters = {Platform.API_SERVER: SimpleNamespace(supports_async_delivery=False, _ensure_session_db=lambda: object())}
    assert await runner._deliver_completion_notification("[ASYNC DELEGATION COMPLETE — deleg_g]", evt) is True
    persist.assert_not_awaited()
    assert _state("deleg_g") == ("plugin", "plugin")


# ── 2. truthful list ────────────────────────────────────────────────────────

def _parent(session_id=SESSION):
    return SimpleNamespace(session_id=session_id, _session_db=None)


def _list(monkeypatch, session_id, owner=OWNER):
    from tools import delegate_tool_registry as reg
    import tools.approval_context as ac
    monkeypatch.setattr(ac, "get_current_session_key", lambda default="": owner)
    return json.loads(reg._handle_control_action("list", None, None, _parent(session_id)))


class TestTruthfulList:
    def test_finished_helpers_of_this_conversation_are_listed_with_whether_they_arrived(self, monkeypatch):
        _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        out = _list(monkeypatch, SESSION)
        assert out["count"] == 0
        [entry] = out["finished"]
        assert entry["delegation_id"] == "deleg_a" and entry["reached_model"] is False
        assert entry["delivery_state"] == "plugin"
        assert "/Users/x/research/01-platforms.md" in entry["result_paths"]
        assert entry["tasks"][0]["summary_line"].startswith("Platforms: 12 compared")
        assert "never reached you" in out["note"]
        assert "have delivered" not in out["note"]

    def test_other_conversations_of_the_same_owner_are_flagged(self, monkeypatch):
        _finished("deleg_a")
        out = _list(monkeypatch, OTHER)
        assert "finished" not in out
        [entry] = out["other_conversations"]
        assert entry["from_another_conversation"] is True and entry["delegation_id"] == "deleg_a"

    def test_another_owners_helpers_are_not_listed(self, monkeypatch):
        _finished("deleg_a", owner="someone-else")
        out = _list(monkeypatch, OTHER)
        assert "other_conversations" not in out and "finished" not in out
        assert "No background helpers" in out["note"]

    def test_older_than_a_day_is_not_listed(self, monkeypatch):
        _finished("deleg_a", completed_ago=25 * 3600)
        out = _list(monkeypatch, OTHER)
        assert "other_conversations" not in out

    def test_result_reads_it_in_full_and_counts_as_seen(self, monkeypatch):
        from tools import delegate_tool_registry as reg
        import tools.approval_context as ac
        _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        monkeypatch.setattr(ac, "get_current_session_key", lambda default="": OWNER)
        out = json.loads(reg._handle_control_action("result", "deleg_a", None, _parent(OTHER)))
        assert "Fleets: n8n + Claude Code." in out["result"] and out["from_another_conversation"] is True
        assert _state("deleg_a") == ("delivered", "result-read")
        refused = json.loads(reg._handle_control_action("result", "deleg_zzz", None, _parent(OTHER)))
        assert "error" in refused


# ── 3. the first turn of a fresh conversation ───────────────────────────────

class TestFreshConversationLine:
    def test_a_fresh_conversation_is_told_about_the_owners_unseen_helpers(self):
        _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        line = recall.turn_preamble([OTHER], OWNER, "did the research finish?", fresh=True)
        assert "You have 2 background helpers from a recent conversation" in line
        assert "finished at" in line and "/Users/x/research/01-platforms.md" in line
        assert "delegate_task(action='result', subagent_id='deleg_a')" in line
        # Telling is not delivering: the row stays unseen until it is read.
        assert _state("deleg_a")[0] == "plugin"

    def test_not_on_a_warm_turn(self):
        _finished("deleg_a")
        _hand_to_plugin("deleg_a")
        assert recall.turn_preamble([OTHER], OWNER, "hi", fresh=False) == ""

    def test_nothing_when_they_already_reached_the_model(self):
        _finished("deleg_a")
        assert ad.claim_completion_delivery("deleg_a", "c1")
        assert ad.complete_completion_delivery("deleg_a", "c1")
        assert recall.turn_preamble([OTHER], OWNER, "hi", fresh=True) == ""

    def test_nothing_without_an_owner_or_rows(self):
        assert recall.turn_preamble([OTHER], "", "hi", fresh=True) == ""
        assert recall.turn_preamble([OTHER], OWNER, "hi", fresh=True) == ""
