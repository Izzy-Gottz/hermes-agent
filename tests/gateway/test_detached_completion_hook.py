"""An async-delegation completion for an api_server session is offered to a plugin
before it is parked on the transcript.

Moe ticket #4, 2026-09-16. The API adapter cannot push, so upstream parks a
finished helper's result as a delivery row for the next client turn — right for
a stateless client, wrong for a desktop app that polls its own inbox and could
tell the person now. The ``deliver_detached_completion`` hook lets the plugin
that owns that surface take the completion; when nothing takes it, the row is
persisted exactly as before.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def _runner():
    return GatewayRunner(GatewayConfig())


def _evt():
    return {"type": "async_delegation", "delegation_id": "d1", "origin_session_id": "moe-sess",
            "session_key": "moe-sess", "results": [{"status": "completed", "task_index": 0}]}


@pytest.mark.asyncio
async def test_a_plugin_that_takes_the_completion_means_no_row(monkeypatch):
    import gateway.wake as wake
    import hermes_cli.plugins as plugins

    got: list = []
    persist = AsyncMock()
    monkeypatch.setattr(wake, "persist_delegation_delivery", persist)
    monkeypatch.setattr(plugins, "has_hook", lambda name: name == "deliver_detached_completion")
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kw: (got.append((name, kw)), [True])[1])

    runner = _runner()
    ok = await runner._self_post_api_server(SimpleNamespace(), "[IMPORTANT: done]", "moe-sess", _evt())
    assert ok is True
    persist.assert_not_awaited()
    name, kw = got[0]
    assert name == "deliver_detached_completion"
    assert kw["session_id"] == "moe-sess" and kw["text"] == "[IMPORTANT: done]"
    assert kw["event"]["delegation_id"] == "d1"


@pytest.mark.asyncio
async def test_nobody_taking_it_persists_the_row_as_before(monkeypatch):
    import gateway.wake as wake
    import hermes_cli.plugins as plugins

    persist = AsyncMock()
    monkeypatch.setattr(wake, "persist_delegation_delivery", persist)
    monkeypatch.setattr(plugins, "has_hook", lambda name: True)
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kw: [None, False])

    runner = _runner()
    ok = await runner._self_post_api_server(SimpleNamespace(), "[IMPORTANT: done]", "moe-sess", _evt())
    assert ok is True
    persist.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_hook_that_raises_never_loses_the_completion(monkeypatch):
    import gateway.wake as wake
    import hermes_cli.plugins as plugins

    persist = AsyncMock()
    monkeypatch.setattr(wake, "persist_delegation_delivery", persist)
    monkeypatch.setattr(plugins, "has_hook", lambda name: True)

    def _boom(name, **kw):
        raise RuntimeError("plugin broke")

    monkeypatch.setattr(plugins, "invoke_hook", _boom)
    runner = _runner()
    assert await runner._self_post_api_server(SimpleNamespace(), "x", "moe-sess", _evt()) is True
    persist.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_hook_registered_costs_nothing(monkeypatch):
    import gateway.wake as wake
    import hermes_cli.plugins as plugins

    persist = AsyncMock()
    monkeypatch.setattr(wake, "persist_delegation_delivery", persist)
    monkeypatch.setattr(plugins, "has_hook", lambda name: False)
    monkeypatch.setattr(plugins, "invoke_hook", lambda name, **kw: pytest.fail("invoked with no hook"))
    runner = _runner()
    assert await runner._self_post_api_server(SimpleNamespace(), "x", "moe-sess", _evt()) is True
    persist.assert_awaited_once()
