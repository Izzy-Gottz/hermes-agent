"""What this gateway does with an inbound frame that arrived sealed.

**This file tests the policy, not the cryptography, and that split is
deliberate.** The derivation lives in the connector's own payload — one file,
loaded by path, shared with the two other implementations that must agree with
it byte for byte. Writing a stand-in derivation here to test "opening works"
would be a fourth copy of the rule, and four copies agree until one is edited.
So every opener below is a stub that returns or raises on demand, and what is
asserted is the thing this repository actually decides: which text reaches the
model, what is logged, and what happens to a frame that cannot be opened.

The cross-implementation proof — a blob the real connector sealed, opened
through the real rule by this module — is a test in the connector's repository,
where both sides are reachable at once. Neither half is sufficient alone and
neither duplicates the other.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import textwrap

import pytest

from gateway.relay import inbound_seal


PLAINTEXT = "the sealed words"
#: Deliberately different from PLAINTEXT everywhere below. A test where the
#: seal and the fallback say the same thing cannot tell which one was used.
FALLBACK = "the plaintext beside it"


@pytest.fixture(autouse=True)
def _forget_what_was_loaded(monkeypatch):
    """Every cache in the module under test, cleared around each case.

    Without this, case order decides the result: the rule and the key material
    are cached process-wide on purpose (a gateway opens thousands of these) and
    the once-only warning is a module global.
    """
    monkeypatch.delenv(inbound_seal.OPENER_ENV, raising=False)
    monkeypatch.delenv(inbound_seal.KEYS_ENV, raising=False)
    monkeypatch.setattr(inbound_seal, "_rule", None, raising=False)
    monkeypatch.setattr(inbound_seal, "_keys", None, raising=False)
    monkeypatch.setattr(inbound_seal, "_warned_unconfigured", False, raising=False)
    yield


def _identity(tmp_path, device_id="cloud-abc12345"):
    """A device identity file of the shape the connector's vault writes."""
    path = tmp_path / "device-identity.json"
    path.write_text(json.dumps({
        "device": device_id,
        "agreement_private": base64.b64encode(bytes(range(32))).decode(),
        "agreement_public": base64.b64encode(bytes(range(32, 64))).decode(),
        "signing_private": base64.b64encode(bytes(32)).decode(),
        "signing_public": base64.b64encode(bytes(32)).decode(),
    }))
    return path


def _opener(tmp_path, body: str, name: str = "rule.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


RETURNS_PLAINTEXT = '''
    def open_sealed(sealed, **kw):
        return %r
''' % PLAINTEXT

REFUSES = '''
    class Refused(ValueError):
        pass

    def open_sealed(sealed, **kw):
        raise Refused("that message does not open with this device's key")
'''


def _configure(monkeypatch, tmp_path, body=RETURNS_PLAINTEXT):
    monkeypatch.setenv(inbound_seal.OPENER_ENV, str(_opener(tmp_path, body)))
    monkeypatch.setenv(inbound_seal.KEYS_ENV, str(_identity(tmp_path)))


def _sealed_frame(*, text=FALLBACK, **overrides):
    frame = {
        "sealed": {
            "ciphertext": base64.b64encode(b"x" * 60).decode(),
            "ephemeralPublic": base64.b64encode(bytes(32)).decode(),
            "scheme": "x25519-hkdf-sha256.aes-gcm-256.inbound.v1",
            "messageId": "telegram:chat-1001:77",
            "toDevice": "cloud-abc12345",
            "channel": "telegram",
        },
        "message_type": "text",
        "source": {"platform": "telegram", "chat_id": "chat-1001"},
    }
    if text is not None:
        frame["text"] = text
    frame.update(overrides)
    return frame


# ─────────────────── the default: nothing changes for anyone ───────────────────

def test_an_unsealed_frame_is_the_text_it_always_was():
    assert inbound_seal.resolve_text({"text": "hello"}) == "hello"


def test_a_frame_with_no_text_at_all_is_empty_not_an_error():
    """The pre-existing behaviour of `raw.get("text", "")`, kept exactly.

    A connector that sends an attachment with no caption sends no `text`, and
    that was never an error. Only a frame that is *sealed* and textless is one.
    """
    assert inbound_seal.resolve_text({"source": {}}) == ""


def test_nothing_is_configured_by_default():
    assert inbound_seal.configured() is False
    assert inbound_seal.rule() is None
    assert inbound_seal.keys() is None


# ──────────────────────── a seal this gateway can open ────────────────────────

def test_the_seal_wins_over_the_plaintext_beside_it(monkeypatch, tmp_path):
    """The whole point: once we can open one, the seal is what we believe."""
    _configure(monkeypatch, tmp_path)
    assert inbound_seal.resolve_text(_sealed_frame()) == PLAINTEXT


def test_the_seal_is_used_even_with_no_plaintext_to_fall_back_on(monkeypatch, tmp_path):
    """The shape of the world after the connector drops `text`."""
    _configure(monkeypatch, tmp_path)
    assert inbound_seal.resolve_text(_sealed_frame(text=None)) == PLAINTEXT


def test_a_disagreement_between_the_two_is_reported(monkeypatch, tmp_path, caplog):
    _configure(monkeypatch, tmp_path)
    with caplog.at_level(logging.WARNING, logger=inbound_seal.__name__):
        inbound_seal.resolve_text(_sealed_frame())
    assert any("disagree" in r.getMessage() for r in caplog.records)


def test_no_log_line_carries_anybody_s_words(monkeypatch, tmp_path, caplog):
    """The canary reports LENGTHS. A log that quotes the message defeats the
    reason the message was sealed in the first place."""
    _configure(monkeypatch, tmp_path)
    with caplog.at_level(logging.WARNING, logger=inbound_seal.__name__):
        inbound_seal.resolve_text(_sealed_frame())
    written = "\n".join(r.getMessage() for r in caplog.records)
    assert PLAINTEXT not in written and FALLBACK not in written


def test_the_derivation_fields_reach_the_rule_unchanged(monkeypatch, tmp_path):
    """Every field the seal binds into its key is handed over, and `messageId`
    is the seal's — NOT `source.message_id`, which is the channel's own id and
    would derive a different key."""
    _configure(monkeypatch, tmp_path, '''
        seen = {}

        def open_sealed(sealed, **kw):
            seen.clear()
            seen.update(kw)
            seen["sealed"] = sealed
            return "opened"
    ''')
    frame = _sealed_frame()
    frame["source"]["message_id"] = "77"
    assert inbound_seal.resolve_text(frame) == "opened"
    kw = inbound_seal.rule().seen
    assert kw["channel"] == "telegram"
    assert kw["message_id"] == "telegram:chat-1001:77"
    assert kw["to_device"] == "cloud-abc12345"
    assert kw["device_id"] == "cloud-abc12345"
    assert kw["agreement_public"] == bytes(range(32, 64))
    assert kw["ephemeral_public"] == bytes(32)
    assert kw["sealed"] == b"x" * 60
    assert kw["scheme"] == "x25519-hkdf-sha256.aes-gcm-256.inbound.v1"


def test_a_missing_scheme_is_not_defaulted_to_ours(monkeypatch, tmp_path):
    """It is passed through absent so the RULE decides, because the scheme
    names the derivation and substituting ours for an unknown one is how a
    future scheme gets opened as this one."""
    _configure(monkeypatch, tmp_path, '''
        seen = {}

        def open_sealed(sealed, **kw):
            seen.clear()
            seen.update(kw)
            return "opened"
    ''')
    frame = _sealed_frame()
    del frame["sealed"]["scheme"]
    inbound_seal.resolve_text(frame)
    assert "scheme" not in inbound_seal.rule().seen


def test_a_foreign_scheme_is_handed_over_verbatim_not_dropped(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, '''
        seen = {}

        def open_sealed(sealed, **kw):
            seen.clear()
            seen.update(kw)
            return "opened"
    ''')
    frame = _sealed_frame()
    frame["sealed"]["scheme"] = "something.v2"
    inbound_seal.resolve_text(frame)
    assert inbound_seal.rule().seen["scheme"] == "something.v2"


# ─────────────────── a seal this gateway CANNOT open ───────────────────

def test_a_refused_seal_falls_back_to_the_plaintext_and_says_so(monkeypatch, tmp_path, caplog):
    """The rollout window, and it is loud on purpose: every one of these is a
    dropped message the day the connector stops sending `text`."""
    _configure(monkeypatch, tmp_path, REFUSES)
    with caplog.at_level(logging.WARNING, logger=inbound_seal.__name__):
        assert inbound_seal.resolve_text(_sealed_frame()) == FALLBACK
    assert any("would not open" in r.getMessage() for r in caplog.records)


def test_a_refused_seal_with_no_plaintext_is_an_error_not_an_empty_message(
        monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, REFUSES)
    with pytest.raises(inbound_seal.SealError):
        inbound_seal.resolve_text(_sealed_frame(text=None))


def test_a_malformed_seal_block_is_refused_by_shape(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    frame = _sealed_frame(text=None)
    del frame["sealed"]["channel"]
    with pytest.raises(inbound_seal.SealError):
        inbound_seal.resolve_text(frame)


def test_a_seal_that_is_not_an_object_is_refused_when_there_is_no_text():
    with pytest.raises(inbound_seal.SealError):
        inbound_seal.resolve_text({"sealed": "not an object"})


def test_an_opener_returning_something_other_than_text_is_refused(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "def open_sealed(sealed, **kw):\n    return b'bytes'\n")
    with pytest.raises(inbound_seal.SealError):
        inbound_seal.resolve_text(_sealed_frame(text=None))


# ─────────────────── configured badly, or not at all ───────────────────

def test_a_sealed_frame_at_an_unconfigured_gateway_uses_text_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger=inbound_seal.__name__):
        assert inbound_seal.resolve_text(_sealed_frame()) == FALLBACK
    assert any("no opener is configured" in r.getMessage() for r in caplog.records)


def test_that_warning_is_said_once_not_once_per_message(caplog):
    with caplog.at_level(logging.WARNING, logger=inbound_seal.__name__):
        for _ in range(5):
            inbound_seal.resolve_text(_sealed_frame())
    said = [r for r in caplog.records if "no opener is configured" in r.getMessage()]
    assert len(said) == 1


def test_a_sealed_textless_frame_at_an_unconfigured_gateway_is_an_error():
    """The failure mode that matters after the connector drops `text`: a
    gateway nobody finished configuring must fail visibly, not deliver
    nothing."""
    with pytest.raises(inbound_seal.SealError):
        inbound_seal.resolve_text(_sealed_frame(text=None))


def test_half_configured_is_not_configured(monkeypatch, tmp_path):
    monkeypatch.setenv(inbound_seal.OPENER_ENV, str(_opener(tmp_path, RETURNS_PLAINTEXT)))
    assert inbound_seal.configured() is False
    assert inbound_seal.resolve_text(_sealed_frame()) == FALLBACK


def test_a_file_with_no_open_sealed_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv(inbound_seal.OPENER_ENV, str(_opener(tmp_path, "VALUE = 1\n")))
    monkeypatch.setenv(inbound_seal.KEYS_ENV, str(_identity(tmp_path)))
    with pytest.raises(inbound_seal.SealError):
        inbound_seal.rule()


def test_an_identity_with_a_short_key_is_refused(monkeypatch, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"device": "cloud-1",
                                "agreement_private": base64.b64encode(b"short").decode(),
                                "agreement_public": base64.b64encode(bytes(32)).decode()}))
    monkeypatch.setenv(inbound_seal.KEYS_ENV, str(path))
    with pytest.raises(inbound_seal.SealError):
        inbound_seal.keys()


def test_an_identity_that_is_not_json_is_refused(monkeypatch, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    monkeypatch.setenv(inbound_seal.KEYS_ENV, str(path))
    with pytest.raises(ValueError):
        inbound_seal.keys()


# ─────────────────── the payload is upgraded underneath us ───────────────────

def test_the_rule_is_reread_when_its_file_changes(monkeypatch, tmp_path):
    """A long-running gateway outlives the connector payload it loaded from.
    Caching the rule for ever would fail every message after an upgrade with a
    tag error nobody could explain."""
    path = _opener(tmp_path, "def open_sealed(sealed, **kw):\n    return 'first'\n")
    monkeypatch.setenv(inbound_seal.OPENER_ENV, str(path))
    monkeypatch.setenv(inbound_seal.KEYS_ENV, str(_identity(tmp_path)))
    assert inbound_seal.resolve_text(_sealed_frame()) == "first"
    os.utime(path, (0, 0))
    path.write_text("def open_sealed(sealed, **kw):\n    return 'second'\n")
    os.utime(path, (1, 1))
    assert inbound_seal.resolve_text(_sealed_frame()) == "second"


def test_a_rule_that_vanishes_keeps_serving_the_one_already_loaded(monkeypatch, tmp_path):
    """An upgrade that replaces the payload is momentarily a missing file.
    Failing every message during that instant would be worse than serving the
    rule we already hold."""
    path = _opener(tmp_path, RETURNS_PLAINTEXT)
    monkeypatch.setenv(inbound_seal.OPENER_ENV, str(path))
    monkeypatch.setenv(inbound_seal.KEYS_ENV, str(_identity(tmp_path)))
    assert inbound_seal.resolve_text(_sealed_frame()) == PLAINTEXT
    path.unlink()
    assert inbound_seal.resolve_text(_sealed_frame()) == PLAINTEXT


def test_loading_the_rule_leaves_no_droppings_beside_it(monkeypatch, tmp_path):
    """The opener lives in somebody else's installed tree. A `__pycache__`
    appearing in it is our mess in their directory."""
    _configure(monkeypatch, tmp_path)
    inbound_seal.resolve_text(_sealed_frame())
    assert not (tmp_path / "__pycache__").exists()


# ─────────────────── and what the transport does with all that ───────────────────

def test_the_transport_drops_a_frame_it_cannot_open_rather_than_delivering_a_blank(
        monkeypatch, tmp_path, caplog):
    """The wiring, asserted at the transport rather than at the module.

    A `SealError` escaping `_event_from_wire` would take down the reader task;
    swallowing it would hand the model a MessageEvent with no text, which reads
    as a person having said nothing at all. Neither: the frame is dropped and
    the reason is logged at ERROR.
    """
    import asyncio

    from gateway.relay.ws_transport import WebSocketRelayTransport

    _configure(monkeypatch, tmp_path, REFUSES)
    delivered = []
    transport = WebSocketRelayTransport("wss://edge.example/relay", "telegram", "bot1")
    transport.set_inbound_handler(lambda event: delivered.append(event))
    frame = json.dumps({"type": "inbound", "event": _sealed_frame(text=None)})
    with caplog.at_level(logging.ERROR):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            transport._handle_frame(frame))
    assert delivered == []
    assert any("cannot be opened" in r.getMessage() for r in caplog.records)


def test_the_transport_delivers_the_opened_words_not_the_plaintext(
        monkeypatch, tmp_path):
    import asyncio

    from gateway.relay.ws_transport import WebSocketRelayTransport

    _configure(monkeypatch, tmp_path)
    delivered = []

    async def handler(event):
        delivered.append(event)

    transport = WebSocketRelayTransport("wss://edge.example/relay", "telegram", "bot1")
    transport.set_inbound_handler(handler)
    frame = json.dumps({"type": "inbound", "event": _sealed_frame()})
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        transport._handle_frame(frame))
    assert [e.text for e in delivered] == [PLAINTEXT]
