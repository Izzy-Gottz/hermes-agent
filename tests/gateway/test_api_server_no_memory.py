"""X-Hermes-No-Memory: a turn the person asked not to be remembered (Memoe's "Don't remember this
chat") reaches the turn's context, the built-in memory tool refuses to write in it, and no
background review is spawned after it. Driven through the adapter's own middleware and
``_run_agent``, the way test_api_server_turn_origin does."""

import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


class _Agent:
    session_id = "s1"
    session_prompt_tokens = session_completion_tokens = session_total_tokens = 0

    def __init__(self, seen):
        self.seen = seen

    def run_conversation(self, **_kw):
        from gateway.session_context import get_turn_no_memory
        from tools.memory_tool import memory_tool

        class _Store:  # reached only when the write is allowed
            def __getattr__(self, name):
                raise LookupError("store reached")
        try:
            wrote = json.loads(memory_tool(action="add", target="memory", content="likes tea", store=_Store()))
            refused = wrote.get("success") is False and "not to be remembered" in json.dumps(wrote)
        except LookupError:
            refused = False
        self.seen.append((get_turn_no_memory(), refused))
        return {"final_response": "ok", "messages": [], "api_calls": 0, "tools": []}


async def _turns(headers_list, monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    seen = []
    monkeypatch.setattr(adapter, "_create_agent", lambda **kw: _Agent(seen))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])

    async def handler(request):
        await adapter._run_agent("hello", [], session_id="s1")
        return web.json_response({})

    app.router.add_post("/turn", handler)
    async with TestClient(TestServer(app)) as client:
        for headers in headers_list:
            assert (await client.post("/turn", headers=headers)).status == 200
    from gateway.session_context import get_turn_no_memory
    loop = asyncio.get_running_loop()
    seen.append(("after", sorted({await loop.run_in_executor(None, get_turn_no_memory) for _ in range(16)})))
    return seen


def test_header_reaches_the_turn_and_the_memory_tool_refuses(monkeypatch):
    seen = asyncio.run(_turns([{"X-Hermes-No-Memory": "1", "X-Hermes-Turn-Origin": "person"}, {},
                               {"X-Hermes-No-Memory": "0"}], monkeypatch))
    # Executor threads are reused: a "don't remember" must never leak into a later turn.
    assert seen.pop() == ("after", [False])
    assert [no for no, _ in seen] == [True, False, False]
    assert [r for _, r in seen] == [True, False, False], "the memory tool wrote in a turn that may not be remembered"


def test_no_review_after_a_turn_not_to_be_remembered(monkeypatch):
    from gateway.session_context import reset_turn_no_memory, set_turn_no_memory
    import run_agent

    spawned = []
    monkeypatch.setattr("agent.background_review.load_background_review_settings",
                        lambda: spawned.append("loaded") or (False, None))
    agent = object.__new__(run_agent.AIAgent)
    token = set_turn_no_memory(True)
    try:
        agent._spawn_background_review([], review_memory=True)
    finally:
        reset_turn_no_memory(token)
    assert spawned == [], "a review was considered for a turn that may not be remembered"
    agent._spawn_background_review([], review_memory=True)
    assert spawned == ["loaded"]


def test_turn_signals_carry_it_for_an_admitted_turn():
    from gateway.session_api_turn import _valid_turn_signals, bind_api_turn_signals, reset_api_turn_signals
    from gateway.session_context import get_turn_no_memory
    assert _valid_turn_signals({"origin": "person", "no_memory": True})
    assert not _valid_turn_signals({"origin": "person", "no_memory": "yes"})
    tokens = bind_api_turn_signals({"turn_signals": {"origin": "person", "no_memory": True}})
    try:
        assert get_turn_no_memory() is True
    finally:
        reset_api_turn_signals(tokens)
    assert get_turn_no_memory() is False
