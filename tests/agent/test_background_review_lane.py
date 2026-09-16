"""A background review must not take the conversation's process — or its place.

The incident, on the owner's Mac, 2026-09-15: Moe finished a turn at 21:21:01
and its post-turn skill review forked 70 ms later under the SAME Hermes
session id. On the claude_code runtime that id names a warm ``claude``
process and its turn lock, so the review took both — and the two messages the
owner typed over the next 74 seconds came back to the screen as "another turn
is still running for session 1e495ecf-d44; try again".

``build_cache_parity_fork`` therefore stamps a LANE on every fork, and a
laned fork is shown the conversation inside its own message, because it no
longer shares the transcript it used to read it from. These tests cover both
halves: without the second one the review would run, cost a process, and have
nothing to review.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent import background_review as br


class TestTheForkAnnouncesItsLane:
    def test_every_parity_fork_is_laned_by_its_write_origin(self, monkeypatch):
        built = {}

        def _fake_init_kwargs(agent, rt, routed, max_iterations):
            return {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                built["made"] = True
                self.api_mode = "claude_code"

        monkeypatch.setattr(br, "_fork_init_kwargs", _fake_init_kwargs)
        monkeypatch.setattr(br, "_resolve_review_runtime", lambda a, c=None: {"routed": False})
        monkeypatch.setattr(br, "_detach_fork_compression", lambda a: None)
        monkeypatch.setattr(br, "_inherit_parent_tool_surface", lambda f, a: None)
        monkeypatch.setattr(br, "_review_input_token_budget", lambda c: 0)
        import run_agent

        monkeypatch.setattr(run_agent, "AIAgent", _FakeAgent)
        parent = SimpleNamespace(
            session_id="sess-1", _memory_store=None, _memory_enabled=False,
            _user_profile_enabled=False, _cached_system_prompt="P", session_start=0,
        )
        fork, _rt, _routed = br.build_cache_parity_fork(
            parent, None, max_iterations=3, write_origin="background_review")
        assert built["made"]
        assert fork.session_id == "sess-1"          # cache parity, unchanged
        assert fork._claude_code_lane == "background_review"

        fork2, _, _ = br.build_cache_parity_fork(
            parent, None, max_iterations=3, write_origin="side_question")
        assert fork2._claude_code_lane == "side_question"


class TestTheForkCanStillSeeTheConversation:
    snapshot = [
        {"role": "system", "content": "SYSTEM PROMPT — must not travel"},
        {"role": "user", "content": "book the dentist"},
        {"role": "assistant", "content": "Booked for Tuesday."},
    ]

    def test_claude_code_carries_the_transcript_in_the_message(self):
        fork = SimpleNamespace(api_mode="claude_code")
        prefix, history = br.fork_conversation_context(fork, self.snapshot, routed=False)
        assert "book the dentist" in prefix
        assert "Booked for Tuesday." in prefix
        assert "SYSTEM PROMPT" not in prefix
        assert prefix.endswith("# What to do with it\n\n")
        # The child owns the context; sending the messages as well would be a
        # second copy of the same conversation that the runtime drops anyway.
        assert history == []

    def test_other_runtimes_are_untouched(self):
        fork = SimpleNamespace(api_mode="chat_completions")
        prefix, history = br.fork_conversation_context(fork, self.snapshot, routed=False)
        assert prefix == ""
        assert history == self.snapshot

    def test_a_routed_fork_still_gets_its_digest(self):
        fork = SimpleNamespace(api_mode="chat_completions")
        long_snapshot = [{"role": "user", "content": f"turn {i}"} for i in range(60)]
        _prefix, history = br.fork_conversation_context(fork, long_snapshot, routed=True)
        assert len(history) < len(long_snapshot)
        assert "digest" in str(history[0]["content"]).lower()

    def test_an_empty_conversation_adds_nothing(self):
        fork = SimpleNamespace(api_mode="claude_code")
        assert br.fork_conversation_context(fork, [], routed=False) == ("", [])
