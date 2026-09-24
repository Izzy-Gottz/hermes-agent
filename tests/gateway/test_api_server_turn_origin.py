"""X-Hermes-Turn-Origin: the client's declaration of who started a turn reaches the turn's context.

The Chrome-extension browser lane opens tabs on the person's screen only in a person's own live
turn (tools.browser_chrome_extension.local_turn_presence). On api_server — the platform the Memoe
app's turns arrive on — that is decided by this header, so the test drives a real request through
the adapter's own middleware and ``_run_agent`` and reads the presence verdict from inside the
agent's ``run_conversation``, where the turn really runs.
"""

import asyncio

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
        from gateway.session_context import get_turn_origin
        from tools.browser_chrome_extension import local_turn_presence
        self.seen.append((get_turn_origin(), local_turn_presence()))
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
    # Other work the same executor threads run later — /v1/runs has its own agent lifecycle and
    # never passes through _run_agent — must not inherit a turn's "person".
    from gateway.session_context import get_turn_origin
    loop = asyncio.get_running_loop()
    seen.append(("after", sorted({await loop.run_in_executor(None, get_turn_origin) for _ in range(16)})))
    return seen


def test_header_reaches_the_turn_and_decides_presence(monkeypatch):
    seen = asyncio.run(_turns([{"X-Hermes-Turn-Origin": "person"}, {"X-Hermes-Turn-Origin": "background"},
                         {}, {"X-Hermes-Turn-Origin": "admin"}, {"X-Hermes-Turn-Origin": "PERSON"}], monkeypatch))
    assert seen.pop() == ("after", [""])
    assert [origin for origin, _ in seen] == ["person", "background", "", "", "person"]
    assert [p["live"] for _, p in seen] == [True, False, False, False, True]
    assert seen[1][1]["why"] == "a background note from the app"


def test_a_person_turn_never_leaks_into_the_next_turn_on_a_reused_thread(monkeypatch):
    # run_in_executor reuses threads; a "person" left bound would make the next, undeclared turn live.
    seen = asyncio.run(_turns([{"X-Hermes-Turn-Origin": "person"}] * 3, monkeypatch))
    assert seen.pop() == ("after", [""])
    assert [p["live"] for _, p in seen] == [True] * 3
