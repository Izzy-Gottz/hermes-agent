"""A person waiting on a rate limit is told, in their own provider's name.

Written 2026-09-15 after the owner of a Moe messaged it from a phone, with the
Mac shut, and got nothing at all for twenty minutes. The machine's log had
`Retrying API call in 600s (attempt 1/3)` twice over; the chat had silence.

Two faults, and the second is worse than it looks:

1. The rate-limit branch BUFFERED every wait however long, while the 5xx branch
   fifteen lines below had surfaced long waits since #26293 for exactly the
   stated reason — "buffering that wait would leave the user silent for
   minutes". The argument was never carried across.
2. `_clear_status_buffer` drops the buffer on successful recovery. So the
   buffered message was not delivered late — it was never delivered at all.
   Silence, then an answer, with nothing to say where the time had gone.
"""
import types

import pytest

from agent import turn_recovery


class _Agent:
    """Only what `compute_error_backoff` touches."""

    def __init__(self, provider="anthropic"):
        self.provider, self.base_url, self.model = provider, "https://x", "m"
        self.emitted, self.buffered = [], []

    def _emit_status(self, m): self.emitted.append(m)
    def _buffer_status(self, m): self.buffered.append(m)
    def _client_log_context(self): return ""


def _429(retry_after=None):
    err = Exception("429")
    if retry_after is not None:
        err.body = {"error": {"retry_after": retry_after}}
    return err


def _backoff(agent, err, **kw):
    kw.setdefault("retry_count", 0)
    kw.setdefault("max_retries", 3)
    kw.setdefault("is_rate_limited", True)
    kw.setdefault("is_zai_coding_overload", False)
    kw.setdefault("base_url", agent.base_url)
    kw.setdefault("model", agent.model)
    return turn_recovery.compute_error_backoff(agent, err, **kw)


def test_a_long_rate_limit_wait_is_said_not_buffered():
    """The regression itself: 600s of silence is not acceptable."""
    agent = _Agent()
    wait = _backoff(agent, _429(600))
    assert wait >= turn_recovery.SPEAK_UP_AFTER_S
    assert agent.emitted, (
        "a %.0fs rate-limit wait was buffered, and the buffer is dropped on "
        "recovery — the person is never told anything happened" % wait)
    assert not agent.buffered
    said = agent.emitted[0]
    assert "600" not in said and "attempt" not in said.lower(), (
        "the engine's counter reached the person: %r" % said)


def test_a_brief_stumble_is_still_swallowed():
    """The other half. Moe should not narrate a hiccup it recovers from."""
    agent = _Agent()
    wait = _backoff(agent, _429(5))
    assert wait < turn_recovery.SPEAK_UP_AFTER_S
    assert not agent.emitted, "a %.0fs wait interrupted the person" % wait
    assert agent.buffered


def test_the_provider_is_never_hardcoded():
    """Somebody on a ChatGPT plan must never be told about Claude.

    Read out of `CANONICAL_PROVIDERS`, so a provider added there is named
    without touching this code — and an unknown slug still says something
    truer than a guess.
    """
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS

    claude = turn_recovery.rate_limit_sentence("claude-code-cli", 600)
    chatgpt = turn_recovery.rate_limit_sentence("openai-codex", 600)
    assert "Claude" in claude and "Claude" not in chatgpt
    assert "ChatGPT" in chatgpt
    # A subscription is the person's; a service they call is not.
    assert "your Claude plan" in claude
    assert "your LM Studio plan" not in turn_recovery.rate_limit_sentence("lmstudio", 600)

    # Every canonical provider produces a sentence that names it and carries
    # no other vendor — the guard against a hardcoded word creeping back in.
    for entry in CANONICAL_PROVIDERS:
        said = turn_recovery.rate_limit_sentence(entry.slug, 600)
        assert said.startswith("I've hit the limit on "), said
        assert "600" not in said, said

    # Unknown and missing both degrade honestly rather than inventing a vendor.
    assert "nobody-has-added-this" in turn_recovery.rate_limit_sentence(
        "nobody-has-added-this", 600)
    assert "your model provider" in turn_recovery.rate_limit_sentence(None, 600)


@pytest.mark.parametrize("seconds,expected", [
    (25, "in under a minute"), (59, "in under a minute"),
    (60, "in about a minute"), (89, "in about a minute"),
    (171, "in about 3 minutes"), (600, "in about 10 minutes"),
])
def test_the_wait_is_in_words_a_person_uses(seconds, expected):
    assert turn_recovery._wait_in_words(seconds) == expected
