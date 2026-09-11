"""A credential refused at the HTTP upgrade — the revocation the 4401 latch never saw.

The transport reads a **4401 close after a successful handshake** as a revoked
credential and stops. That covers a connector revoking a *live* socket. A
connector that authenticates the upgrade itself refuses a dead credential
*before* ``accept``, as an **HTTP status on the handshake response**, and no
WebSocket ever exists for a Close frame to travel on. So a device revoked *while
its socket was down* saw ``InvalidStatus`` on every re-dial and re-dialled for
ever, each attempt a "reconnect failed" line and none of them saying why.

The connector's word for it is **HTTP 410 Gone**, sent only to a bearer that
matches a credential it has revoked (contract §3.1). These tests run the
production transport against a real ``websockets`` server whose
``process_request`` plays that gate: accept, or refuse with a status before any
socket exists. What they pin:

* **one** 410 latches ``auth_revoked`` — by ``connect()`` on a fresh transport
  (which is what a restarted gateway does) and by the reconnect supervisor —
  and exactly one log line says why;
* **nothing else does**: three 403s in a row followed by a healthy connector
  must leave the transport reconnected and *not* revoked (a review found the
  earlier three-in-a-row counter disabling a gateway whose ``.env`` had merely
  lost its auth mode); 5xx and a closed port stay retryable and recover;
* the adapter turns a transport that latched on its first dial into a
  non-retryable ``relay_disabled`` fatal, which is how the gateway's retry
  watcher stops building fresh transports.
"""

from __future__ import annotations

import asyncio
import http
import json
import logging

import pytest

from gateway.relay.ws_transport import (
    _UPGRADE_REVOKED_STATUS,
    WEBSOCKETS_AVAILABLE,
    WebSocketRelayTransport,
)

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets
    from websockets.exceptions import InvalidStatus


DESCRIPTOR = {
    "contract_version": 1,
    "platform": "discord",
    "label": "Discord",
    "max_message_length": 2000,
    "supports_draft_streaming": False,
    "supports_edit": True,
    "supports_threads": True,
    "markdown_dialect": "discord",
    "len_unit": "chars",
}

LOGGER = "gateway.relay.ws_transport"
#: The one line the latch writes. Chosen so it cannot be confused with the
#: post-handshake 4401 line, which also says "not reconnecting".
LATCH_LINE = "upgrade refused"


class _UpgradeAuthConnector:
    """A connector that authenticates the upgrade itself.

    ``answers`` is consumed one per dial: ``None`` accepts, an int refuses with
    that HTTP status before any WebSocket exists — the shape of a Starlette
    endpoint that answers before ``accept``. When the list runs out,
    ``default`` applies. Every accepted socket is answered with one descriptor
    and then closed normally (the lid, not a revocation), so a reconnecting
    transport comes straight back to the gate.
    """

    def __init__(self, answers=None, default=None):
        self.answers = list(answers or [])
        self.default = default
        self.dials = 0
        self.handshakes = 0
        self._server = None
        self.url = ""

    async def start(self):
        self._server = await websockets.serve(
            self._handle, "127.0.0.1", 0, process_request=self._gate)
        port = next(iter(self._server.sockets)).getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"

    async def stop(self):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _gate(self, connection, request):
        self.dials += 1
        answer = self.answers.pop(0) if self.answers else self.default
        if answer is None:
            return None
        return connection.respond(http.HTTPStatus(answer), "refused\n")

    async def _handle(self, ws):
        self.handshakes += 1
        async for raw in ws:
            for line in str(raw).split("\n"):
                if line.strip() and json.loads(line).get("type") == "hello":
                    await ws.send(json.dumps({"type": "descriptor", "descriptor": DESCRIPTOR}) + "\n")
                    await asyncio.sleep(0.05)
                    return


def _transport(url: str, **kwargs) -> WebSocketRelayTransport:
    return WebSocketRelayTransport(
        url, "discord", "appShared", gateway_id="gw-x", upgrade_secret="secret-x",
        connect_timeout_s=5, **kwargs)


async def _until(predicate, tries: int = 400, every: float = 0.02) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(every)
    return predicate()


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


async def _quietly_disconnect(t: WebSocketRelayTransport) -> None:
    try:
        await t.disconnect()
    except Exception:  # noqa: BLE001 - teardown of a transport that may never have connected
        pass


# ── the latch: one sighting ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_one_410_on_a_fresh_transport_latches_auth_revoked(caplog):
    """The restarted-gateway route. `register_relay_adapter()` builds a NEW
    transport on every attempt, so nothing survives from one dial to the next
    — the signal has to be read off a single response. Three fresh transports,
    each dialling once: each latches, each writes one line naming the status."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    srv = _UpgradeAuthConnector(default=_UPGRADE_REVOKED_STATUS)
    await srv.start()
    try:
        for attempt in range(3):
            t = _transport(srv.url)
            with pytest.raises(InvalidStatus):
                await t.connect()
            assert t.auth_revoked is True, f"fresh transport {attempt} did not latch on one dial"
            await _quietly_disconnect(t)
        assert srv.dials == 3
        latch = [m for m in _messages(caplog) if LATCH_LINE in m]
        assert len(latch) == 3, _messages(caplog)
        assert all(f"HTTP {_UPGRADE_REVOKED_STATUS}" in m and "not reconnecting" in m for m in latch)
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_a_device_revoked_while_its_socket_was_down_stops_redialling(caplog):
    """The supervisor route, which is where the bug lived: handshake once, lose
    the socket normally, and find the connector saying *revoked* on the
    re-dial. The supervisor must end, the dialling must stop, and the log must
    carry exactly one sentence about why — not a "reconnect failed" under it."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    srv = _UpgradeAuthConnector()
    await srv.start()
    t = _transport(srv.url, reconnect=True, reconnect_backoff_s=0.02,
                   reconnect_max_backoff_s=0.05)
    try:
        await t.connect()
        await t.handshake()
        # Revoked between the socket closing and the first re-dial.
        srv.default = _UPGRADE_REVOKED_STATUS
        assert await _until(lambda: t.auth_revoked), "never latched"
        assert t._supervisor is not None
        assert await _until(lambda: t._supervisor.done()), "the supervisor kept running"
        dials = srv.dials
        await asyncio.sleep(0.2)
        assert srv.dials == dials, "kept dialling after latching"
        assert srv.dials == 2, "more than one re-dial was needed to read a single response"
        assert srv.handshakes == 1
        messages = _messages(caplog)
        assert sum(LATCH_LINE in m for m in messages) == 1, messages
        assert not any("reconnect failed" in m for m in messages), messages
    finally:
        await _quietly_disconnect(t)
        await srv.stop()


# ── nothing else latches — the review's scenario A ───────────────────────────


@pytest.mark.asyncio
async def test_three_transient_403s_then_a_healthy_connector_is_not_a_revocation(caplog):
    """The finding that killed the counter: a proxy incident answers 403 three
    times and clears. The credential was never revoked. The transport must
    reconnect and `auth_revoked` must be False — an adapter polls that flag
    and disables the relay off it, permanently."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    srv = _UpgradeAuthConnector(answers=[None, 403, 403, 403], default=None)
    await srv.start()
    t = _transport(srv.url, reconnect=True, reconnect_backoff_s=0.02,
                   reconnect_max_backoff_s=0.05)
    try:
        await t.connect()
        await t.handshake()
        assert await _until(lambda: srv.handshakes == 2), "did not reconnect once the 403s cleared"
        assert t.auth_revoked is False, "three transient 403s disabled a device that was never revoked"
        assert not [m for m in _messages(caplog) if LATCH_LINE in m]
        assert sum("reconnect failed" in m for m in _messages(caplog)) == 3
    finally:
        await _quietly_disconnect(t)
        await srv.stop()


@pytest.mark.asyncio
async def test_a_403_for_ever_is_a_configuration_to_fix_not_a_revocation():
    """A Mac whose `.env` lost `GATEWAY_RELAY_AUTH_MODE=raw`, or whose edge
    address points at the wrong host, is refused 403 on every dial. It must
    keep trying — the fix is to the file — and never be told it was revoked."""
    srv = _UpgradeAuthConnector(answers=[None], default=403)
    await srv.start()
    t = _transport(srv.url, reconnect=True, reconnect_backoff_s=0.02,
                   reconnect_max_backoff_s=0.02)
    try:
        await t.connect()
        await t.handshake()
        assert await _until(lambda: srv.dials >= 8), f"only {srv.dials} dial(s)"
        assert t.auth_revoked is False
        assert not t._supervisor.done(), "the supervisor gave up on a 403"
    finally:
        await _quietly_disconnect(t)
        await srv.stop()


@pytest.mark.asyncio
async def test_a_connector_that_is_down_is_a_cold_start_and_stays_retryable(caplog):
    """A 503 is what a proxy answers while the connector behind it is not
    there. The transport keeps re-dialling, never latches, and reconnects the
    moment the connector is back."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    srv = _UpgradeAuthConnector()
    await srv.start()
    t = _transport(srv.url, reconnect=True, reconnect_backoff_s=0.02,
                   reconnect_max_backoff_s=0.02)
    try:
        await t.connect()
        await t.handshake()
        srv.default = 503
        assert await _until(lambda: srv.dials >= 7), f"only {srv.dials} dial(s)"
        assert t.auth_revoked is False
        assert not t._supervisor.done(), "the supervisor gave up on a 503"
        assert not [m for m in _messages(caplog) if LATCH_LINE in m]
        srv.default = None
        assert await _until(lambda: srv.handshakes == 2), "did not reconnect when the connector returned"
        assert t.auth_revoked is False
    finally:
        await _quietly_disconnect(t)
        await srv.stop()


@pytest.mark.asyncio
async def test_a_closed_port_is_a_cold_start_too(caplog):
    """Connection refused carries no status at all and is retried on the
    ordinary backoff — measured by the failed re-dial lines, since there is no
    server left to count on."""
    caplog.set_level(logging.WARNING, logger=LOGGER)
    srv = _UpgradeAuthConnector()
    await srv.start()
    t = _transport(srv.url, reconnect=True, reconnect_backoff_s=0.02,
                   reconnect_max_backoff_s=0.02)
    try:
        await t.connect()
        await t.handshake()
        await srv.stop()
        failed = lambda: sum("reconnect failed" in m for m in _messages(caplog))  # noqa: E731
        assert await _until(lambda: failed() >= 4)
        assert t.auth_revoked is False
        assert not t._supervisor.done()
        assert not [m for m in _messages(caplog) if LATCH_LINE in m]
    finally:
        await _quietly_disconnect(t)
        await srv.stop()


# ── the classifier on its own ────────────────────────────────────────────────


def test_the_status_is_read_off_both_shapes_the_library_has_used():
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    current = InvalidStatus(Response(410, "Gone", Headers()))
    assert WebSocketRelayTransport._upgrade_status_of(current) == 410

    class Legacy(Exception):  # websockets < 13: InvalidStatusCode.status_code
        status_code = 410

    assert WebSocketRelayTransport._upgrade_status_of(Legacy()) == 410
    assert WebSocketRelayTransport._upgrade_status_of(OSError()) is None
    assert WebSocketRelayTransport._upgrade_status_of(asyncio.TimeoutError()) is None


def test_only_the_revoked_status_latches():
    """Every other status a dial can be refused with — the ones a connector
    that does not know this credential sends, and the ones a proxy sends with
    the connector perfectly healthy — leaves the transport retryable."""
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    t = _transport("ws://unused")
    for status in (400, 401, 403, 404, 429, 500, 502, 503, 504):
        t._note_dial_failure(InvalidStatus(Response(status, "x", Headers())))
        assert t.auth_revoked is False, status
    t._note_dial_failure(OSError("connection refused"))
    assert t.auth_revoked is False
    t._note_dial_failure(InvalidStatus(Response(_UPGRADE_REVOKED_STATUS, "Gone", Headers())))
    assert t.auth_revoked is True


# ── the adapter ends the gateway's retry loop ────────────────────────────────


class _RefusedDial:
    """A transport whose dial the connector refused — with or without the
    revoked signal — the way `WebSocketRelayTransport.connect()` raises."""

    def __init__(self, revoked: bool):
        self.auth_revoked = revoked

    def set_inbound_handler(self, h):
        self._h = h

    async def connect(self):
        raise RuntimeError("server rejected WebSocket connection: HTTP 4xx")


@pytest.mark.asyncio
async def test_the_adapter_turns_a_first_dial_revocation_into_a_non_retryable_fatal():
    """gateway/run.py builds a fresh adapter and transport on every reconnect
    attempt, so a latch left on a raised connect() is thrown away with the
    transport. run.py reads `connect() -> False` with a non-retryable fatal as
    "remove from the retry queue"; this is the adapter producing that."""
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from tests.gateway.relay.test_relay_adapter import make_desc

    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=_RefusedDial(revoked=True))
    assert await a.connect() is False
    assert a.has_fatal_error
    assert a.fatal_error_code == "relay_disabled"
    assert a.fatal_error_retryable is False


@pytest.mark.asyncio
async def test_the_adapter_leaves_every_other_dial_failure_transient():
    """The other half: a dial that failed without the revoked signal must
    still RAISE out of connect(), which run.py queues as transient — a
    connector that is down comes back."""
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from tests.gateway.relay.test_relay_adapter import make_desc

    a = RelayAdapter(PlatformConfig(), make_desc(platform="discord"), transport=_RefusedDial(revoked=False))
    with pytest.raises(RuntimeError):
        await a.connect()
    assert not a.has_fatal_error
