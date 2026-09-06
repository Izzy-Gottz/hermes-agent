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


#: `wss://`, not `ws://`. Raw mode refuses a cleartext dial to anywhere but
#: loopback (see `test_raw_mode_refuses_a_cleartext_dial`), so a helper that
#: used `ws://unused` would be asserting the header of a dial the transport
#: would not make.
DIAL = "wss://edge.example/relay"


def _bearer(url: str = DIAL, **kw) -> str | None:
    """The Authorization value a transport built with ``kw`` would present."""
    t = WebSocketRelayTransport(url, "discord", "bot1", **kw)
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


# ─────────── the reader that feeds the header ───────────

def test_the_reader_hands_over_a_secret_with_no_gateway_id(monkeypatch):
    """``relay_connection_auth`` returns ``(None, secret)`` when only the
    secret is configured, so ``raw`` — which sends no gatewayId — gets a
    credential to present.

    Written after mis-reading this function as dropping the secret unless both
    were set: the ``if not (gateway_id and secret)`` there guards only the
    *config.yaml fallback*, not the return. The guard that actually dropped a
    raw credential was in ``_upgrade_headers``, and it is fixed above. Kept as
    a test because the wiring between the two is the half no header test can
    see, and the next reader will make the same mistake.
    """
    monkeypatch.setenv("GATEWAY_RELAY_AUTH_MODE", "raw")
    monkeypatch.setenv("GATEWAY_RELAY_SECRET", SECRET)
    monkeypatch.delenv("GATEWAY_RELAY_ID", raising=False)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {}, raising=False)
    gateway_id, secret = relay.relay_connection_auth()
    assert (gateway_id, secret) == (None, SECRET)
    # ...and that pair, handed to the transport, produces the raw header.
    assert _bearer(gateway_id=gateway_id, upgrade_secret=secret,
                   upgrade_auth_mode="raw") == f"Bearer {SECRET}"


def test_the_same_pair_in_token_mode_sends_nothing(monkeypatch):
    """The negative control, and the reason the mode exists: an unindexable
    token is not a credential, so token mode declines to send one."""
    assert _bearer(gateway_id=None, upgrade_secret=SECRET) is None


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
    """An unrecognised mode reaches the token path.

    The transport does NOT normalise — a normaliser here was deleted for
    killing no test — so what this pins is the `== "raw"` comparison: every
    value but the exact string takes the default. (An earlier version of this
    docstring claimed the transport normalises, contradicting the commit that
    removed it.)"""
    bearer = _bearer(gateway_id="gw-1", upgrade_secret=SECRET, upgrade_auth_mode="RAW")
    assert verify_token(bearer.removeprefix("Bearer "), [SECRET]) == "gw-1"


# ───────── cleartext, which raw mode makes permanent ─────────
#
# A `token` bearer expires in 300 s. A `raw` one IS the secret and never
# expires, so a cleartext dial leaks a permanent credential for every route it
# opens rather than a five-minute one for this socket. `_ws_dial_url` maps
# `http://` to `ws://` silently, so one character in GATEWAY_RELAY_URL is the
# whole distance between "fine" and that. Loopback stays allowed: nothing
# leaves the machine, and the real-socket test below dials it.

@pytest.mark.parametrize("url", [
    "ws://edge.example/relay",
    "ws://10.0.0.5:8080/relay",
    "ws://[2001:db8::1]/relay",
])
def test_raw_mode_refuses_a_cleartext_dial(url):
    with pytest.raises(RuntimeError) as caught:
        _bearer(url, upgrade_secret=SECRET, upgrade_auth_mode="raw")
    assert "wss://" in str(caught.value)


@pytest.mark.parametrize("url", [
    "ws://localhost:9/relay",
    "ws://127.0.0.1:9/relay",
    "ws://[::1]:9/relay",
])
def test_raw_mode_allows_a_cleartext_dial_to_loopback(url):
    assert _bearer(url, upgrade_secret=SECRET,
                   upgrade_auth_mode="raw") == f"Bearer {SECRET}"


def test_token_mode_is_not_refused_over_cleartext():
    """The negative control, and it is a deliberate asymmetry rather than an
    oversight: a signed token is worth far less to whoever picks it up, and
    refusing it too would change behaviour for every existing deployment,
    which this change promised not to do."""
    bearer = _bearer("ws://edge.example/relay", gateway_id="gw-1", upgrade_secret=SECRET)
    assert verify_token(bearer.removeprefix("Bearer "), [SECRET]) == "gw-1"


def test_an_unparseable_url_fails_closed_in_raw_mode():
    """Not-loopback is the default for anything the parser cannot read. The
    cost of guessing wrong is a permanent credential in the clear."""
    with pytest.raises(RuntimeError):
        _bearer("ws://[not-an-address/relay", upgrade_secret=SECRET,
                upgrade_auth_mode="raw")


# ───────── the wiring: registration, not the constructor ─────────
#
# **This is the guard a review found nothing covering, and it was the one that
# mattered.** Deleting `upgrade_auth_mode=upgrade_auth_mode` from
# `register_relay_adapter` left every test above green — all 310, plus the
# edge's 563, plus the two end-to-end falsifiers that drive the fork's real
# transport — because every one of them either builds the transport by hand or
# calls `_upgrade_headers` directly. `register_relay_adapter` is the ONLY
# production constructor of `WebSocketRelayTransport` (all 19 other call sites
# are tests), so with that line gone every real gateway silently falls back to
# `token`, every upgrade gets a 403, and the feature is dead while the suite
# says otherwise.
#
# The commit that added the mode said a header test cannot see the wiring
# between the reader and the header, and wrote a test for `relay_connection_auth`
# on exactly that reasoning. It did not write this one. So the shape was
# understood and applied one function short of where it was needed.

def _adapter_transport(tmp_path, monkeypatch, **env):
    """Build the relay adapter the way production does, and hand back its
    transport. Nothing here constructs the transport itself — that is the whole
    point of the test."""
    from gateway.config import Platform, load_gateway_config
    from gateway.platform_registry import platform_registry
    from gateway.relay import register_relay_adapter

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(exist_ok=True)
    (hermes_home / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://edge.example/relay")
    for name in ("GATEWAY_RELAY_AUTH_MODE", "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    platform_registry.unregister("relay")
    assert register_relay_adapter() is True
    config = load_gateway_config()
    adapter = platform_registry.create_adapter("relay", config.platforms[Platform.RELAY])
    transport = adapter._transport
    assert transport is not None, "a URL-configured registration builds a live transport"
    return transport


def test_registration_carries_raw_all_the_way_to_the_header(tmp_path, monkeypatch):
    """The artefact is the header bytes the *registered* transport would send.

    Asserted on the header rather than on ``_upgrade_auth_mode`` deliberately:
    an attribute check would still pass if the transport stopped acting on it,
    and this is the one assertion in the fork that spans reader → registration
    → header.
    """
    transport = _adapter_transport(tmp_path, monkeypatch,
                                   GATEWAY_RELAY_AUTH_MODE="raw",
                                   GATEWAY_RELAY_SECRET=SECRET,
                                   GATEWAY_RELAY_ID="gw-1")
    assert transport._upgrade_headers() == {"Authorization": f"Bearer {SECRET}"}


def test_registration_leaves_the_default_signing(tmp_path, monkeypatch):
    """The negative control. Without it, the test above would pass just as well
    on a transport that had been hard-wired to raw."""
    transport = _adapter_transport(tmp_path, monkeypatch,
                                   GATEWAY_RELAY_SECRET=SECRET,
                                   GATEWAY_RELAY_ID="gw-1")
    bearer = transport._upgrade_headers()["Authorization"]
    assert verify_token(bearer.removeprefix("Bearer "), [SECRET]) == "gw-1"
    assert SECRET not in bearer


# There is deliberately no registration-level test of the config.yaml source.
# `gateway/run.py`'s config home is resolved into a module global at import
# (`_hermes_home`, read through `_gateway_config_home`), so a test that sets
# HERMES_HOME after the fact fights a cache that is not this change's to
# reset — and the resulting test would be measuring import order. The two
# sources are covered where each one is real: config.yaml at the reader
# (`test_mode_falls_back_to_config_yaml`), and the environment here, which is
# also what `~/.hermes/.env` becomes — the gateway loads it into os.environ at
# boot, so the documented configuration location arrives as env either way.


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
