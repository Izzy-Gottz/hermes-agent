"""The CLI vision lane, wired into call_llm.

A rule and its caller are two things. An earlier change in this repo passed
285 tests with the caller disabled, because every test drove the rule
directly — so these drive `call_llm` / `async_call_llm` and assert on what
came back, not on the helper in isolation.

Both seams are covered, because the live failure needed the second one:

  1. Nothing pinned -> the CLI serves the call.
  2. A lane IS pinned and returns 402 -> the CLI rescues it. This is the one
     that was actually broken on the owner's Mac: `auxiliary.vision.provider`
     was pinned to a real provider whose balance was empty, the fallback chain
     contains no CLI (it is not an HTTP provider), so the subscription sitting
     right there was never tried and every screenshot died as
     `402 Insufficient Balance`.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from agent import auxiliary_client as A
from agent import claude_code_vision as V


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode()
MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "what is on screen?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}},
]}]


class _Paid(Exception):
    """Stands in for openai.APIStatusError(402)."""


@pytest.fixture
def cli(monkeypatch):
    """The CLI answers; record what it was asked."""
    calls = []

    def fake(image, question, *, model=None, timeout=120.0, **kw):
        calls.append({"image": image, "question": question, "model": model})
        return "a wikipedia article"

    monkeypatch.setattr(V, "describe_image", fake)
    monkeypatch.setattr(V, "cli_available", lambda command="claude": True)
    return calls


@pytest.fixture
def brain_is_claude_code(monkeypatch):
    monkeypatch.setattr(A, "_read_main_provider", lambda: "claude-code-cli")


@pytest.fixture
def nothing_pinned(monkeypatch):
    monkeypatch.setattr(
        A, "_resolve_task_provider_model",
        lambda *a, **k: ("auto", None, None, None, None))


# ── seam 1: nothing pinned, the CLI serves ────────────────────────────────

def test_the_cli_serves_a_vision_call(cli, brain_is_claude_code, nothing_pinned,
                                      monkeypatch):
    def must_not_run(**kw):
        raise AssertionError("the HTTP chain ran; the CLI should have served")

    monkeypatch.setattr(A, "_call_llm_impl", must_not_run)
    out = A.call_llm(task="vision", messages=MESSAGES)
    assert out.choices[0].message.content == "a wikipedia article"
    assert cli[0]["question"] == "what is on screen?"


def test_a_pinned_text_lane_is_untouched(cli, brain_is_claude_code, monkeypatch):
    """A text lane the user pinned keeps its own chain.

    This asserted that compression ALWAYS kept the HTTP chain, and went red
    when the text lane landed — the second of my own tests to catch a contract
    change rather than a bug. Inverted rather than deleted: what still matters
    is that an explicit pin wins over the CLI, exactly as it does for vision.
    """
    seen = {}

    def impl(**kw):
        seen["ran"] = True
        return "http response"

    monkeypatch.setattr(
        A, "_resolve_task_provider_model",
        lambda *a, **k: ("openai", "gpt-4o-mini", None, None, None))
    monkeypatch.setattr(A, "_call_llm_impl", impl)
    assert A.call_llm(task="compression", messages=MESSAGES) == "http response"
    assert seen["ran"] and not cli


def test_another_brain_is_untouched(cli, monkeypatch, nothing_pinned):
    """An HTTP provider has a working chain and must keep using it."""
    monkeypatch.setattr(A, "_read_main_provider", lambda: "openai")
    monkeypatch.setattr(A, "_call_llm_impl", lambda **kw: "http response")
    assert A.call_llm(task="vision", messages=MESSAGES) == "http response"
    assert not cli


def test_a_pinned_working_lane_wins_over_the_cli(cli, brain_is_claude_code,
                                                 monkeypatch):
    """An explicit `auxiliary.vision.provider` is a real answer."""
    monkeypatch.setattr(
        A, "_resolve_task_provider_model",
        lambda *a, **k: ("openai", "gpt-4o", None, None, None))
    monkeypatch.setattr(A, "_call_llm_impl", lambda **kw: "http response")
    assert A.call_llm(task="vision", messages=MESSAGES) == "http response"
    assert not cli


def test_messages_without_an_image_fall_through(cli, brain_is_claude_code,
                                                nothing_pinned, monkeypatch):
    monkeypatch.setattr(A, "_call_llm_impl", lambda **kw: "http response")
    text_only = [{"role": "user", "content": "no image here"}]
    assert A.call_llm(task="vision", messages=text_only) == "http response"
    assert not cli


def test_an_unavailable_cli_falls_through(brain_is_claude_code, nothing_pinned,
                                          monkeypatch):
    monkeypatch.setattr(V, "cli_available", lambda command="claude": False)
    monkeypatch.setattr(A, "_call_llm_impl", lambda **kw: "http response")
    assert A.call_llm(task="vision", messages=MESSAGES) == "http response"


# ── seam 2: the pinned lane cannot pay ────────────────────────────────────

def test_a_402_on_a_pinned_lane_is_rescued(cli, brain_is_claude_code,
                                           monkeypatch):
    """The measured failure, end to end."""
    monkeypatch.setattr(
        A, "_resolve_task_provider_model",
        lambda *a, **k: ("deepseek", "deepseek-vision", None, None, None))
    monkeypatch.setattr(A, "_is_payment_error", lambda exc: isinstance(exc, _Paid))

    def broke(**kw):
        raise _Paid("402 Insufficient Balance")

    monkeypatch.setattr(A, "_call_llm_impl", broke)
    out = A.call_llm(task="vision", messages=MESSAGES)
    assert out.choices[0].message.content == "a wikipedia article"


def test_a_non_payment_error_still_raises(cli, brain_is_claude_code, monkeypatch):
    """A bad request must surface as itself, not be quietly answered by
    another lane — that would hide real bugs behind a plausible sentence."""
    monkeypatch.setattr(
        A, "_resolve_task_provider_model",
        lambda *a, **k: ("deepseek", "deepseek-vision", None, None, None))
    monkeypatch.setattr(A, "_is_payment_error", lambda exc: False)

    def broke(**kw):
        raise ValueError("malformed request")

    monkeypatch.setattr(A, "_call_llm_impl", broke)
    with pytest.raises(ValueError):
        A.call_llm(task="vision", messages=MESSAGES)
    assert not cli


def test_a_402_on_a_text_side_task_is_also_rescued(brain_is_claude_code,
                                                   monkeypatch):
    """The same empty balance took compression, session_search and
    title_generation down with vision — the gateway logged it once per turn.
    This test asserted the opposite until the text lane existed; it is kept,
    inverted, rather than deleted, because the contract it guards is which
    tasks the CLI may serve."""
    monkeypatch.setattr(V, "answer_text", lambda prompt, **kw: "a title")
    monkeypatch.setattr(A, "_is_payment_error", lambda exc: isinstance(exc, _Paid))

    def broke(**kw):
        raise _Paid("402")

    monkeypatch.setattr(A, "_call_llm_impl", broke)
    out = A.call_llm(task="compression", messages=MESSAGES)
    assert out.choices[0].message.content == "a title"


def test_a_402_on_an_unlisted_task_still_raises(cli, brain_is_claude_code,
                                                monkeypatch):
    """TEXT_TASKS is a small named set, not "anything not vision". A task
    added upstream later must not silently inherit a lane nobody chose."""
    monkeypatch.setattr(A, "_is_payment_error", lambda exc: isinstance(exc, _Paid))

    def broke(**kw):
        raise _Paid("402")

    monkeypatch.setattr(A, "_call_llm_impl", broke)
    with pytest.raises(_Paid):
        A.call_llm(task="embedding", messages=MESSAGES)
    assert not cli


def test_a_402_with_another_brain_still_raises(cli, monkeypatch):
    """Rescue is only for the brain that has no HTTP chain of its own."""
    monkeypatch.setattr(A, "_read_main_provider", lambda: "openai")
    monkeypatch.setattr(A, "_is_payment_error", lambda exc: isinstance(exc, _Paid))
    monkeypatch.setattr(
        A, "_resolve_task_provider_model",
        lambda *a, **k: ("openai", "gpt-4o", None, None, None))

    def broke(**kw):
        raise _Paid("402")

    monkeypatch.setattr(A, "_call_llm_impl", broke)
    with pytest.raises(_Paid):
        A.call_llm(task="vision", messages=MESSAGES)
    assert not cli


# ── the async twin, which is the one vision_analyze actually calls ────────

def test_the_async_path_serves_from_the_cli(cli, brain_is_claude_code,
                                            nothing_pinned, monkeypatch):
    """`vision_analyze` and the computer_use capture path call this one.
    Wiring only the sync twin would have fixed nothing a user can see."""
    async def must_not_run(**kw):
        raise AssertionError("the HTTP chain ran")

    monkeypatch.setattr(A, "_async_call_llm_impl", must_not_run)
    out = asyncio.run(A.async_call_llm(task="vision", messages=MESSAGES))
    assert out.choices[0].message.content == "a wikipedia article"


def test_the_async_path_rescues_a_402(cli, brain_is_claude_code, monkeypatch):
    monkeypatch.setattr(
        A, "_resolve_task_provider_model",
        lambda *a, **k: ("deepseek", "deepseek-vision", None, None, None))
    monkeypatch.setattr(A, "_is_payment_error", lambda exc: isinstance(exc, _Paid))

    async def broke(**kw):
        raise _Paid("402 Insufficient Balance")

    monkeypatch.setattr(A, "_async_call_llm_impl", broke)
    out = asyncio.run(A.async_call_llm(task="vision", messages=MESSAGES))
    assert out.choices[0].message.content == "a wikipedia article"


def test_the_async_path_does_not_block_the_loop(cli, brain_is_claude_code,
                                                nothing_pinned, monkeypatch):
    """The CLI is a subprocess taking seconds. Running it inline would stall
    every other task sharing the loop."""
    import time

    def slow(image, question, **kw):
        time.sleep(0.25)
        return "answer"

    monkeypatch.setattr(V, "describe_image", slow)

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        await A.async_call_llm(task="vision", messages=MESSAGES)
        await task
        return ticks

    assert asyncio.run(main()) >= 5, "the event loop was blocked"


# ── the response shape every caller relies on ─────────────────────────────

def test_the_response_reads_like_any_other(cli, brain_is_claude_code,
                                           nothing_pinned, monkeypatch):
    """`extract_content_or_reasoning` is what vision_tools uses, and it
    reaches for reasoning fields when content is empty. The shim must survive
    that probe rather than raising AttributeError."""
    monkeypatch.setattr(A, "_call_llm_impl", lambda **kw: None)
    out = A.call_llm(task="vision", messages=MESSAGES)
    assert A.extract_content_or_reasoning(out) == "a wikipedia article"


def test_usage_is_zero_not_invented(cli, brain_is_claude_code, nothing_pinned,
                                    monkeypatch):
    """A subscription is not metered per token here. Guessing numbers would
    put fiction into cost accounting."""
    monkeypatch.setattr(A, "_call_llm_impl", lambda **kw: None)
    out = A.call_llm(task="vision", messages=MESSAGES)
    assert out.usage is None


# ── extracting the question and image ─────────────────────────────────────

def test_the_last_image_wins():
    """A caller may retry a downscaled copy in the same list."""
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
    ]}]
    image, _ = V.image_and_question_from_messages(msgs)
    assert image.endswith("BBB")


def test_the_system_prompt_is_not_the_question():
    msgs = [
        {"role": "system", "content": "You are a vision model."},
        {"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]},
    ]
    _, question = V.image_and_question_from_messages(msgs)
    assert question == "what is this?"


def test_a_text_task_is_served_when_no_lane_is_configured(brain_is_claude_code,
                                                          nothing_pinned,
                                                          monkeypatch):
    """The gap the live Mac exposed after the pin was removed.

    With `auxiliary` on `auto` and nothing in the chain configured, the
    gateway logged

        No LLM provider configured for task=title_generation provider=auto

    which is NOT a payment error, so the rescue never fired — and the first
    seam only served vision. The rescue is for a lane that breaks; this is for
    a lane that was never there.
    """
    monkeypatch.setattr(V, "answer_text", lambda prompt, **kw: "a title")

    def must_not_run(**kw):
        raise AssertionError("the HTTP chain ran; the CLI should have served")

    monkeypatch.setattr(A, "_call_llm_impl", must_not_run)
    out = A.call_llm(task="title_generation", messages=MESSAGES)
    assert out.choices[0].message.content == "a title"


def test_moes_sweep_and_monitor_tasks_are_served_too(brain_is_claude_code,
                                                     nothing_pinned, monkeypatch):
    """`classify` is what moe-screen/sweep.py asks for at the end of a
    conversation and `monitor` is the cron triage; both are text in, JSON
    out, and both were falling through to a provider nobody configured."""
    monkeypatch.setattr(V, "answer_text", lambda prompt, **kw: '[{"slug": "x"}]')

    def must_not_run(**kw):
        raise AssertionError("the HTTP chain ran; the CLI should have served")

    monkeypatch.setattr(A, "_call_llm_impl", must_not_run)
    for task in ("classify", "monitor"):
        out = A.call_llm(task=task, messages=MESSAGES)
        assert out.choices[0].message.content == '[{"slug": "x"}]'
    assert {"classify", "monitor"} <= V.TEXT_TASKS


def test_the_async_path_serves_a_text_task_too(brain_is_claude_code,
                                               nothing_pinned, monkeypatch):
    monkeypatch.setattr(V, "answer_text", lambda prompt, **kw: "a summary")

    async def must_not_run(**kw):
        raise AssertionError("the HTTP chain ran")

    monkeypatch.setattr(A, "_async_call_llm_impl", must_not_run)
    out = asyncio.run(A.async_call_llm(task="compression", messages=MESSAGES))
    assert out.choices[0].message.content == "a summary"


def test_an_unlisted_task_still_uses_the_chain(cli, brain_is_claude_code,
                                               nothing_pinned, monkeypatch):
    monkeypatch.setattr(A, "_call_llm_impl", lambda **kw: "http response")
    assert A.call_llm(task="embedding", messages=MESSAGES) == "http response"
