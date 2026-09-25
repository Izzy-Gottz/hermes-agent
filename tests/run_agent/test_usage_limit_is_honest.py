"""A usage limit that outlasts any retry is said truthfully, and not slept on.

Measured 2026-09-25 on the owner's away machine (provider=anthropic, a Claude
subscription setup-token whose plan usage was spent)::

    API call failed (attempt 1/3) error_type=RateLimitError ... HTTP 429: This
      request would exceed your account's rate limit. Please try again later.
    Retrying API call in 600s (attempt 1/3) ... policy=default

and on Telegram: "I've hit the limit on your Claude plan — I'll answer as soon
as it clears, in about 10 minutes." Neither half was true. `compute_error_backoff`
clamps Retry-After to 600 s, the clamp was then read aloud as the provider's
word, and the loop slept 3 x 600 s against a limit that resets in hours.

The header names below are Claude Code's own (its client turns
``anthropic-ratelimit-unified-status: rejected`` + ``-reset`` (epoch seconds)
into ``retry-after = reset - now``); none is invented here.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

import hermes_time
from agent import turn_recovery
from run_agent import AIAgent

# Friday 2026-09-25 10:40 America/New_York.
_TZ = "America/New_York"
_NOW = datetime(2026, 9, 25, 10, 40, tzinfo=ZoneInfo(_TZ)).timestamp()


@pytest.fixture(autouse=True)
def _timezone(monkeypatch):
    monkeypatch.setenv("HERMES_TIMEZONE", _TZ)
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


def _tools(*names):
    return [{"type": "function", "function": {"name": n, "description": n,
             "parameters": {"type": "object", "properties": {}}}} for n in names]


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_tools("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    a.client = MagicMock()
    # The measured shape: native Anthropic, signed in with a subscription setup-token.
    a.provider = "anthropic"
    a._is_anthropic_oauth = True
    a._persist_session = lambda *args, **kwargs: None
    a._save_trajectory = lambda *args, **kwargs: None
    return a


def _limit_429(headers):
    """The measured error: an Anthropic RateLimitError with the measured body."""

    class RateLimitError(Exception):
        def __init__(self):
            super().__init__(
                "Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', "
                "'message': \"This request would exceed your account's rate limit. "
                "Please try again later.\"}}")
            self.status_code = 429
            self.response = SimpleNamespace(headers=headers)
            self.body = {"type": "error", "error": {
                "type": "rate_limit_error",
                "message": "This request would exceed your account's rate limit. Please try again later."}}

    return RateLimitError()


def _drive(agent, error, *, fallback_results=()):
    """Run a turn whose every API call raises ``error``. The backoff sleep is replaced by a
    recorder that ends the turn, so a test never waits and can see whether it was asked to."""
    calls = {"api": 0}

    def _api(api_kwargs):
        calls["api"] += 1
        raise error

    agent._interruptible_api_call = _api
    emitted = []
    agent._emit_status = emitted.append
    sleeps = []

    def _sleep(_agent, wait_time, *a, **kw):
        sleeps.append(wait_time)
        return {"final_response": "(slept)", "messages": [], "api_calls": calls["api"],
                "completed": False, "interrupted": True}

    fallbacks = list(fallback_results)
    fallback_calls = []

    def _fallback(reason=None):
        fallback_calls.append(reason)
        return fallbacks.pop(0) if fallbacks else False

    agent._try_activate_fallback = _fallback
    agent._has_pending_fallback = lambda: bool(fallbacks)
    with (
        patch("agent.turn_api_error.interruptible_backoff_sleep", side_effect=_sleep),
        patch("agent.turn_recovery._clock", return_value=_NOW),
    ):
        result = agent.run_conversation("hello")
    return result, SimpleNamespace(sleeps=sleeps, emitted=emitted, api=calls["api"],
                                   fallbacks=fallback_calls)


# ── The turn ──────────────────────────────────────────────────────────────


def test_a_retry_after_of_hours_is_not_slept_on_and_is_said_truthfully(agent):
    """The regression: Retry-After 5 h 20 m. No 600 s sleep, no "about 10 minutes"."""
    result, seen = _drive(agent, _limit_429({"retry-after": str(5 * 3600 + 20 * 60)}))

    assert seen.sleeps == [], "slept %r on a limit that resets in hours" % seen.sleeps
    assert seen.api == 1, "retried a spent plan %d times" % seen.api
    said = result["final_response"]
    assert said == ("I've hit the usage limit on your Claude plan. It resets around 4:00 PM, "
                    "in about 5 hours — or switch Moe to another plan or key in "
                    "Settings › Brain to keep going now."), said
    assert "10 minutes" not in said and "600" not in said
    assert result["failed"] is True and result["failure_retryable"] is False
    assert result["failure_reason"] == "rate_limit"
    # Said once, as the outcome — not also pushed as a status bubble.
    assert not any("usage limit" in m for m in seen.emitted), seen.emitted


def test_the_subscription_reset_header_is_the_clock(agent):
    """Anthropic's unified window names its own reset instant; that is what is said."""
    reset = datetime(2026, 9, 26, 9, 5, tzinfo=ZoneInfo(_TZ)).timestamp()
    result, seen = _drive(agent, _limit_429({
        "anthropic-ratelimit-unified-status": "rejected",
        "anthropic-ratelimit-unified-reset": str(int(reset)),
        # The capped value a proxy or SDK might still carry must not win over the instant.
        "retry-after": "600",
    }))
    assert seen.sleeps == [] and seen.api == 1
    assert "It resets tomorrow around 9:05 AM, in about 22 hours" in result["final_response"]
    assert result["rate_limit_reset_at"] == reset


def test_a_rejected_window_with_no_reset_invents_no_number(agent):
    result, seen = _drive(agent, _limit_429({"anthropic-ratelimit-unified-status": "rejected"}))
    said = result["final_response"]
    assert seen.sleeps == [] and seen.api == 1
    assert "can't tell exactly when" in said, said
    assert not any(ch.isdigit() for ch in said), "a number nobody stated: %r" % said
    assert result["rate_limit_reset_at"] is None


def test_a_long_limit_goes_to_the_fallback_chain_first(agent):
    """With a fallback configured, the turn moves on to it instead of ending."""
    result, seen = _drive(agent, _limit_429({"retry-after": "18000"}), fallback_results=[True])
    assert seen.fallbacks, "the fallback chain was never tried"
    assert seen.sleeps == [], "slept %r instead of falling back" % seen.sleeps
    # The fallback hit the same wall (this fake raises every call) and the chain ran out:
    # only then is the person told.
    assert seen.api == 2
    assert "usage limit" in result["final_response"]


def test_a_short_rate_limit_keeps_todays_wait_and_wording(agent):
    """Retry-After within the cap: still a burst, still waited out, still "in about N minutes"."""
    result, seen = _drive(agent, _limit_429({"retry-after": "300"}))
    assert seen.sleeps == [300.0]
    assert seen.emitted == [
        "I've hit the limit on Anthropic — I'll answer as soon as it clears, in about 5 minutes."]


def test_exactly_the_cap_is_still_a_burst(agent):
    _, seen = _drive(agent, _limit_429({"retry-after": "600"}))
    assert seen.sleeps == [600.0]


# ── What the person on Telegram actually receives ─────────────────────────


@pytest.mark.parametrize("platform", ["telegram", "slack"])
def test_the_gateway_delivers_the_sentence_itself(agent, platform):
    """The gateway rewrites provider-error envelopes into "wait a moment and try again".
    The honest sentence must pass through untouched — it is the whole point."""
    from gateway.run import _normalize_empty_agent_response, _sanitize_gateway_final_response

    result, _ = _drive(agent, _limit_429({"retry-after": "18000"}))
    delivered = _sanitize_gateway_final_response(
        platform, _normalize_empty_agent_response(result, result["final_response"]))
    assert delivered == result["final_response"]
    assert delivered.startswith("I've hit the usage limit on your Claude plan. It resets around "), delivered
    # What the same turn used to end with, and what the gateway made of it.
    old = "API call failed after 3 retries: HTTP 429: This request would exceed your account's rate limit."
    assert "wait a moment" in _sanitize_gateway_final_response(platform, old)


# ── The pieces ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("headers,expected", [
    ({"retry-after": "30"}, None),
    ({"Retry-After": "600"}, None),
    ({"retry-after": "601"}, 601.0),
    ({"retry-after": "18000"}, 18000.0),
    # Overage still allowed: the window is not what refused this request.
    ({"anthropic-ratelimit-unified-status": "rejected",
      "anthropic-ratelimit-unified-overage-status": "allowed", "retry-after": "20"}, None),
    ({"anthropic-ratelimit-unified-status": "allowed", "retry-after": "20"}, None),
    ({}, None),
])
def test_what_counts_as_beyond_retry(headers, expected):
    wait = turn_recovery.rate_limit_beyond_retry(_limit_429(headers), now=_NOW)
    assert (None if wait is None else wait.seconds) == expected


def test_a_body_retry_after_counts_too():
    err = _limit_429({})
    err.body = {"error": {"retry_after": 7200}}
    assert turn_recovery.rate_limit_beyond_retry(err, now=_NOW).seconds == 7200.0


@pytest.mark.parametrize("reset,expected", [
    (datetime(2026, 9, 25, 11, 25, tzinfo=ZoneInfo(_TZ)), "around 11:25 AM, in about 45 minutes"),
    (datetime(2026, 9, 25, 15, 40, tzinfo=ZoneInfo(_TZ)), "around 3:40 PM, in about 5 hours"),
    (datetime(2026, 9, 26, 0, 10, tzinfo=ZoneInfo(_TZ)), "tomorrow around 12:10 AM, in about 14 hours"),
    (datetime(2026, 9, 28, 12, 0, tzinfo=ZoneInfo(_TZ)), "on Monday around 12:00 PM, in about 3 days"),
    (datetime(2026, 10, 9, 8, 0, tzinfo=ZoneInfo(_TZ)), "on October 9 around 8:00 AM, in about 14 days"),
])
def test_the_reset_is_in_the_persons_clock(reset, expected):
    said = turn_recovery.usage_limit_sentence("claude-code-cli", reset.timestamp(), now=_NOW)
    assert said.startswith("I've hit the usage limit on your Claude plan. It resets " + expected), said


def test_the_clock_follows_the_configured_timezone(monkeypatch):
    reset = datetime(2026, 9, 25, 15, 40, tzinfo=ZoneInfo(_TZ)).timestamp()
    monkeypatch.setenv("HERMES_TIMEZONE", "Europe/London")
    hermes_time.reset_cache()
    assert "around 8:40 PM" in turn_recovery.usage_limit_sentence("anthropic", reset, now=_NOW)


def test_no_counter_no_emoji_and_the_provider_is_not_hardcoded():
    for slug, name in (("openai-codex", "your ChatGPT plan"), ("openrouter", "OpenRouter")):
        said = turn_recovery.usage_limit_sentence(slug, _NOW + 18000, now=_NOW)
        assert ("usage limit on %s." % name) in said and "Claude" not in said, said
        assert all(ord(c) < 0x2000 or c in "—›" for c in said), said
