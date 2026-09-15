"""`link_request` / `link_result` (Moe slice E3) — gateway side.

The connector asks this gateway to become a second device of the person's
own Telegram or WhatsApp; the registered handler answers, and the answer
goes back on the same socket as a `link_result`. Against a real in-process
websockets server, in the shape of test_relay_going_idle.py.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio

from gateway import relay as relay_module
from gateway.relay.ws_transport import WebSocketRelayTransport, WEBSOCKETS_AVAILABLE

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets


DESCRIPTOR = {
    "contract_version": 1,
    "platform": "telegram",
    "label": "Telegram",
    "max_message_length": 4096,
    "supports_draft_streaming": False,
    "supports_edit": True,
    "supports_threads": False,
    "markdown_dialect": "plain",
    "len_unit": "chars",
}


class _AskingServer:
    """Connector stub: descriptor on hello, then pushes the link_requests it
    was given and records every link_result that comes back."""

    def __init__(self, to_push=None):
        self.results: list[dict] = []
        self.received: list[dict] = []
        self._to_push = list(to_push or [])
        self._server = None
        self.url = ""
        self.got_result = asyncio.Event()

    async def start(self):
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        sock = next(iter(self._server.sockets))
        self.url = f"ws://127.0.0.1:{sock.getsockname()[1]}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws):
        try:
            async for raw in ws:
                for line in str(raw).split("\n"):
                    if not line.strip():
                        continue
                    frame = json.loads(line)
                    self.received.append(frame)
                    if frame.get("type") == "hello":
                        await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
                        for f in self._to_push:
                            await ws.send(json.dumps(f) + "\n")
                    elif frame.get("type") == "link_result":
                        self.results.append(frame)
                        self.got_result.set()
        except Exception:
            pass


async def _wait_for_result(server: _AskingServer, timeout: float = 3.0) -> dict:
    await asyncio.wait_for(server.got_result.wait(), timeout)
    return server.results[-1]


@pytest_asyncio.fixture
async def server():
    srv = _AskingServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture(autouse=True)
def _no_module_handler():
    relay_module.set_link_handler(None)
    yield
    relay_module.set_link_handler(None)


@pytest.mark.asyncio
async def test_a_plain_handler_answers_and_the_result_carries_its_body(server):
    server._to_push = [{"type": "link_request", "requestId": "r-1", "link": "telegram", "body": {}}]
    asked = []

    def handler(link, body):
        asked.append((link, body))
        return {"key": "QUJD"}

    t = WebSocketRelayTransport(server.url, "telegram", "appShared")
    t.set_link_handler(handler)
    await t.connect()
    try:
        await t.handshake()
        result = await _wait_for_result(server)
        assert asked == [("telegram", {})]
        assert result == {"type": "link_result", "requestId": "r-1", "ok": True, "body": {"key": "QUJD"}}
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_a_coroutine_handler_is_awaited_and_the_body_is_passed_through(server):
    server._to_push = [{"type": "link_request", "requestId": "r-2", "link": "whatsapp",
                        "body": {"phone": "447700900000"}}]
    seen = {}

    async def handler(link, body):
        seen["link"], seen["body"] = link, body
        return {"code": "ABCD1234"}

    t = WebSocketRelayTransport(server.url, "telegram", "appShared")
    t.set_link_handler(handler)
    await t.connect()
    try:
        await t.handshake()
        result = await _wait_for_result(server)
        assert seen == {"link": "whatsapp", "body": {"phone": "447700900000"}}
        assert result["ok"] is True and result["body"] == {"code": "ABCD1234"}
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_no_handler_is_an_honest_refusal_not_silence(server):
    server._to_push = [{"type": "link_request", "requestId": "r-3", "link": "telegram", "body": {}}]
    t = WebSocketRelayTransport(server.url, "telegram", "appShared")
    await t.connect()
    try:
        await t.handshake()
        result = await _wait_for_result(server)
        assert result["ok"] is False
        assert "no link handler" in result["body"]["error"]
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_the_module_level_handler_is_consulted_when_the_transport_has_none(server):
    """The transport is built before any plugin loads, so a plugin registers
    with `gateway.relay.set_link_handler` and the transport finds it at
    frame time rather than at construction."""
    server._to_push = [{"type": "link_request", "requestId": "r-4", "link": "telegram", "body": {}}]
    relay_module.set_link_handler(lambda link, body: {"already": True})
    t = WebSocketRelayTransport(server.url, "telegram", "appShared")
    await t.connect()
    try:
        await t.handshake()
        result = await _wait_for_result(server)
        assert result == {"type": "link_result", "requestId": "r-4", "ok": True, "body": {"already": True}}
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_a_handler_that_raises_or_answers_with_error_is_ok_false(server):
    server._to_push = [
        {"type": "link_request", "requestId": "r-5", "link": "telegram", "body": {}},
        {"type": "link_request", "requestId": "r-6", "link": "whatsapp", "body": {}},
    ]

    def handler(link, body):
        if link == "telegram":
            raise RuntimeError("boom")
        return {"error": "no number was given"}

    t = WebSocketRelayTransport(server.url, "telegram", "appShared")
    t.set_link_handler(handler)
    await t.connect()
    try:
        await t.handshake()
        deadline = asyncio.get_running_loop().time() + 3
        while len(server.results) < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.02)
        by_id = {r["requestId"]: r for r in server.results}
        assert by_id["r-5"]["ok"] is False and "failed" in by_id["r-5"]["body"]["error"]
        assert by_id["r-6"] == {"type": "link_result", "requestId": "r-6", "ok": False,
                                "body": {"error": "no number was given"}}
        # The socket survived both: a bad handler is a bad answer, not a dead link.
        assert t._ws is not None
    finally:
        await t.disconnect()


@pytest.mark.asyncio
async def test_a_link_request_without_an_id_is_dropped_and_nothing_is_sent(server):
    server._to_push = [{"type": "link_request", "link": "telegram", "body": {}}]
    t = WebSocketRelayTransport(server.url, "telegram", "appShared")
    t.set_link_handler(lambda link, body: {"key": "x"})
    await t.connect()
    try:
        await t.handshake()
        await asyncio.sleep(0.2)
        assert server.results == []
    finally:
        await t.disconnect()
