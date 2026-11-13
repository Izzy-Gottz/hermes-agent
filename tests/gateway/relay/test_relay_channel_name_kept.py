"""A relayed chat keeps the name of the app it came from (Memoe F7).

Email to Moe's own address arrives from the connector as ``moemail``, a
channel this build has no adapter for. It used to become ``RELAY``, so the
session, its memory and its turns were filed under "relay" and no longer read
as the person talking. It now keeps its own name, and the reply still goes
back through the relay transport, tagged for that channel.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.relay.ws_transport import _event_from_wire


def _wire(platform):
    return {"text": "hello", "message_type": "text",
            "source": {"platform": platform, "chat_id": "c-1", "chat_type": "dm", "user_id": "u-1"}}


def test_a_channel_with_no_adapter_here_keeps_its_name():
    event = _event_from_wire(_wire("moemail"))
    assert event.source.platform is not Platform.RELAY
    assert event.source.platform.value == "moemail"
    assert Platform("moemail") is event.source.platform


@pytest.mark.parametrize("name,expected", [
    ("telegram", "telegram"),
    ("line", "line"),
    ("relay", "relay"),
    ("Not A Name!", "relay"),
    ("", "relay"),
    (None, "relay"),
])
def test_known_names_resolve_and_odd_ones_are_still_relay(name, expected):
    assert _event_from_wire(_wire(name)).source.platform.value == expected


@pytest.mark.asyncio
async def test_the_reply_to_a_kept_name_goes_back_through_the_relay_tagged_for_it():
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
    from tests.gateway.relay.stub_connector import StubConnector

    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="relay", label="Relay",
        max_message_length=4096, supports_draft_streaming=False, supports_edit=True,
        supports_threads=False, markdown_dialect="plain", len_unit="chars")
    stub = StubConnector(descriptor)
    stub._identities = [("telegram", "bot-9"), ("moemail", "addr-1")]
    adapter = RelayAdapter(PlatformConfig(), descriptor, transport=stub)
    await adapter.connect()

    await stub.push_inbound(_event_from_wire(_wire("moemail")))
    result = await adapter.send("c-1", "a reply by email")
    assert getattr(result, "success", True) is not False
    assert stub.sent_platforms[-1] == "moemail"
