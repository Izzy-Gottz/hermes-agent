"""The raw-bearer upgrade mode, and the property that forced it.

**Why this mode exists.** ``make_upgrade_token`` is an HMAC, and verifying an
HMAC needs the key. A connector that deliberately stores only
``sha256(secret)`` — so that a database dump yields nothing that can act —
therefore cannot verify one, ever, no matter what it does at request time.
Moe's edge is exactly that connector, and until this mode existed no real
gateway could dial it: the gateway spoke HMAC and the connector could only
hash. ``GATEWAY_RELAY_AUTH_MODE=raw`` sends the secret itself, the connector
hashes what arrives, and neither side gains a second credential.

Every assertion below names its artefact — the header bytes that leave the
process, or the value ``relay_upgrade_auth_mode()`` returns — never "the code
path exists". The two modes are also checked *against each other*: a test that
only asserts raw mode sends the secret would pass unchanged if token mode had
silently become raw too, so each mode carries a negative control naming the
other.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway import relay
from gateway.relay.auth import verify_token
from gateway.relay.ws_transport import WebSocketRelayTransport, WEBSOCKETS_AVAILABLE

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not installed")

if WEBSOCKETS_AVAILABLE:
    import websockets


SECRET = "c2VjcmV0LXRoaXJ0eS10d28tYnl0ZXMtb2Yta2V5IQ=="


def _bearer(**kw) -> str | None:
    """The Authorization value a transport built with ``kw`` would present."""
    t = WebSocketRelayTransport("ws://unused", "discord", "bot1", **kw)
    return t._upgrade_headers().get("Authorization")


# ───────────────────────── the mode reader ─────────────────────────

def test_mode_defaults_to_token_with_nothing_configured(monkeypatch):
    monkeypatch.delenv("GATEWAY_RELAY_AUTH_MODE", raising=False)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config", lambda: {}, raising=False)
    assert relay.relay_upgrade_auth_mode() == "token"


def test_mode_reads_env(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_AUTH_MODE", "raw")
    assert relay.relay_upgrade_auth_mode() == "raw"


def test_mode_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_AUTH_MODE", "  RAW  ")
    assert relay.relay_upgrade_auth_mode() == "raw"


def test_mode_falls_back_to_config_yaml(monkeypatch):
    """Env first, then ``gateway.relay_auth_mode`` — the same two-source order
    every other relay setting uses (relay_connection_auth, relay_endpoint)."""
    monkeypatch.delenv("GATEWAY_RELAY_AUTH_MODE", raising=False)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"gateway": {"relay_auth_mode": "raw"}},
        raising=False,
    )
    assert relay.relay_upgrade_auth_mode() == "raw"


def test_env_wins_over_config_yaml(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_AUTH_MODE", "token")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"gateway": {"relay_auth_mode": "raw"}},
        raising=False,
    )
    assert relay.relay_upgrade_auth_mode() == "token"


def test_unknown_mode_is_token_not_an_exception(monkeypatch):
    """An unrecognised mode must not stop a gateway booting. The connector
    rejects an unusable upgrade with 4401; a crash at import time would take
    the whole gateway down for a typo in one env var."""
    monkeypatch.setenv("GATEWAY_RELAY_AUTH_MODE", "raww")
    assert relay.relay_upgrade_auth_mode() == "token"


def test_broken_config_yaml_is_token_not_an_exception(monkeypatch):
    def _boom():
        raise RuntimeError("config.yaml is unparseable")

    monkeypatch.delenv("GATEWAY_RELAY_AUTH_MODE", raising=False)
    monkeypatch.setattr("gateway.run._load_gateway_config", _boom, raising=False)
    assert relay.relay_upgrade_auth_mode() == "token"


# ───────────────────────── the header bytes ─────────────────────────

def test_token_mode_is_unchanged_and_verifies():
    """The default is still a token ``verify_token`` accepts under the secret.
    This is the upstream-behaviour-preserved assertion."""
    bearer = _bearer(gateway_id="gw-1", upgrade_secret=SECRET)
    assert bearer is not None and bearer.startswith("Bearer ")
    assert verify_token(bearer.removeprefix("Bearer "), [SECRET]) == "gw-1"


def test_raw_mode_presents_the_secret_verbatim():
    bearer = _bearer(gateway_id="gw-1", upgrade_secret=SECRET, upgrade_auth_mode="raw")
    assert bearer == f"Bearer {SECRET}"


def test_raw_mode_is_not_a_valid_upgrade_token():
    """The negative control for the test above. If raw mode ever started
    emitting a token, ``test_raw_mode_presents_the_secret_verbatim`` would
    fail — but if token mode started emitting a raw secret, nothing else here
    would notice. This is what notices."""
    raw = _bearer(gateway_id="gw-1", upgrade_secret=SECRET, upgrade_auth_mode="raw")
    assert verify_token(raw.removeprefix("Bearer "), [SECRET]) is None


def test_token_mode_does_not_leak_the_secret_into_the_header():
    """The other half of the pair: an HMAC bearer must not simply contain the
    secret. Cheap, and it is the whole difference between the two modes."""
    bearer = _bearer(gateway_id="gw-1", upgrade_secret=SECRET)
    assert SECRET not in bearer


def test_raw_mode_needs_no_gateway_id():
    """In raw mode the credential identifies the instance by itself, so
    requiring a gateway_id would be a configured value the connector never
    reads — this repo's 'a well-formed identifier that resolves to nothing'."""
    assert _bearer(upgrade_secret=SECRET, upgrade_auth_mode="raw") == f"Bearer {SECRET}"


def test_token_mode_still_needs_a_gateway_id():
    """Unchanged: the connector peeks the payload to index its verify list, so
    an HMAC token with no gateway_id is unusable and no header is sent."""
    assert _bearer(upgrade_secret=SECRET) is None


def test_no_secret_sends_no_header_in_either_mode():
    assert _bearer(gateway_id="gw-1") is None
    assert _bearer(gateway_id="gw-1", upgrade_auth_mode="raw") is None


def test_unknown_mode_on_the_transport_is_token():
    """The transport normalises too, not only the config reader — a caller that
    passes a mode straight through must not get an un-normalised third state."""
    bearer = _bearer(gateway_id="gw-1", upgrade_secret=SECRET, upgrade_auth_mode="RAW")
    assert verify_token(bearer.removeprefix("Bearer "), [SECRET]) == "gw-1"


# ──────────────── it arrives, on a real socket ────────────────

@pytest.mark.asyncio
async def test_raw_bearer_arrives_at_a_real_server():
    """The artefact is the header a real ``websockets`` server received on the
    upgrade — not the dict ``_upgrade_headers`` returned. §18.29's rule: an
    in-process test client can bypass the transport layer entirely, so the one
    assertion that matters runs against a real socket.
    """
    seen: list[str | None] = []

    async def handler(ws):
        seen.append(ws.request.headers.get("Authorization"))
        await asyncio.sleep(0.05)

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        t = WebSocketRelayTransport(
            f"ws://127.0.0.1:{port}", "discord", "bot1",
            upgrade_secret=SECRET, upgrade_auth_mode="raw",
        )
        await t.connect()
        for _ in range(100):
            if seen:
                break
            await asyncio.sleep(0.02)
        await t.disconnect(budget_s=0)
    finally:
        server.close()
        await server.wait_closed()

    assert seen == [f"Bearer {SECRET}"]
