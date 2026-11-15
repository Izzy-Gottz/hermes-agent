"""A catch-all relay connector's deliveries are admitted natively, whatever platform they carry.

Memoe's Mac dials its connector with no ``GATEWAY_RELAY_PLATFORMS`` and no ``GATEWAY_RELAY_ID``:
one generic ``relay`` hello, authenticated by the raw secret alone. The connector then delivers the
person's Telegram (or any channel) DMs over that one socket, with the UNDERLYING platform on the
source and no ``user_id``. ``_relay_connector`` required a descriptor and a bot id hello'd for that
exact platform, plus a gateway id, so every one of those deliveries was refused
``permission_denied`` and acknowledged: the person's message was dropped without a reply.

Real ``GatewayRunner`` + session authority + ``RelayAdapter`` + ``WebSocketRelayTransport`` against
a fixture connector; only the model call is stubbed.
"""
import asyncio
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from tests.waiting import async_wait_until

websockets = pytest.importorskip('websockets')


async def _deliver(tmp_path, monkeypatch, *, hello_platform, gateway_id, secret, mode, sources):
    """Send each source as one inbound frame; return (texts run, acked buffer ids, log)."""
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport

    outgoing, ready = [], asyncio.Queue()

    async def connector(ws):
        hello = json.loads(await ws.recv())
        # The connector answers a hello with the descriptor for the platform it named.
        descriptor = CapabilityDescriptor(1, hello.get('platform') or 'relay', 'Fixture', 4096,
                                          False, True, False, 'plain', 'chars')
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')
        await ready.put(ws)
        async for line in ws:
            frame = json.loads(line)
            outgoing.append(frame)
            if frame['type'] == 'outbound':
                await ws.send(json.dumps({'type': 'outbound_result', 'requestId': frame['requestId'],
                                          'result': {'success': True, 'message_id': 'sent'}}) + '\n')

    turns = []

    async def turn(event, *_args, **_kwargs):
        turns.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    placeholder = CapabilityDescriptor(1, hello_platform, 'Relay', 4096, False, True, False, 'plain', 'chars')
    async with websockets.serve(connector, '127.0.0.1', 0) as relay:
        transport = WebSocketRelayTransport(
            f'ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay', hello_platform, '',
            gateway_id=gateway_id, upgrade_secret=secret, upgrade_auth_mode=mode, reconnect=False)
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
        monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='catch-all')
        adapter = RelayAdapter(PlatformConfig(enabled=True), placeholder, transport)
        adapter.gateway_runner = runner
        runner.adapters[Platform.RELAY] = adapter
        runner._wire_adapter_handlers(adapter)
        try:
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)
            for index, (text, source) in enumerate(sources):
                event = {'text': text, 'message_id': f'm{index}', 'source': source}
                await ws.send(json.dumps({'type': 'inbound', 'event': event, 'bufferId': f'buf-{index}'}) + '\n')
            acked = lambda: {f.get('bufferId') for f in outgoing if f.get('type') == 'inbound_ack'}
            await async_wait_until(lambda: len(acked()) == len(sources), timeout=20,
                                   what='every frame acknowledged')
            await asyncio.sleep(0.5)  # an admitted turn drains after its ACK
            return turns, acked()
        finally:
            await adapter.disconnect()
            authority.db.close()


TELEGRAM_DM = {'platform': 'telegram', 'chat_id': '8975409107', 'chat_type': 'dm', 'user_id': None}


@pytest.mark.asyncio
async def test_catch_all_connector_telegram_dm_without_user_id_is_answered(tmp_path, monkeypatch, caplog):
    # The Mac's exact shape: generic hello, no gateway id, raw secret, no user_id on the source.
    turns, _ = await _deliver(tmp_path, monkeypatch, hello_platform='relay', gateway_id=None,
                              secret='mac-secret', mode='raw', sources=[('hello moe', TELEGRAM_DM)])
    assert 'relay inbound refused' not in caplog.text
    await async_wait_until(lambda: turns == ['hello moe'], timeout=10, what='the telegram DM run')


@pytest.mark.asyncio
async def test_catch_all_connector_needs_its_secret(tmp_path, monkeypatch, caplog):
    # An unauthenticated socket is no upstream authority: still refused.
    turns, _ = await _deliver(tmp_path, monkeypatch, hello_platform='relay', gateway_id=None,
                              secret=None, mode='raw', sources=[('hello moe', TELEGRAM_DM)])
    assert 'relay inbound refused (permission_denied)' in caplog.text
    assert turns == []


@pytest.mark.asyncio
async def test_token_mode_still_needs_a_gateway_id(tmp_path, monkeypatch, caplog):
    # A signed-token upgrade without a gateway id never authenticated anything.
    turns, _ = await _deliver(tmp_path, monkeypatch, hello_platform='relay', gateway_id=None,
                              secret='mac-secret', mode='token', sources=[('hello moe', TELEGRAM_DM)])
    assert 'relay inbound refused (permission_denied)' in caplog.text
    assert turns == []


@pytest.mark.asyncio
async def test_a_platform_specific_connector_still_refuses_other_platforms(tmp_path, monkeypatch, caplog):
    # Only the catch-all hello widens: a socket that hello'd telegram is not a discord connector.
    discord = {**TELEGRAM_DM, 'platform': 'discord', 'user_id': 'u1'}
    turns, _ = await _deliver(tmp_path, monkeypatch, hello_platform='telegram', gateway_id='gw',
                              secret='s', mode='token',
                              sources=[('for discord', discord), ('for telegram', {**TELEGRAM_DM, 'user_id': 'u1'})])
    assert 'relay inbound refused (permission_denied) for discord' in caplog.text
    await async_wait_until(lambda: turns == ['for telegram'], timeout=10, what='the telegram frame run')


def test_direct_telegram_from_an_unlisted_user_is_still_refused(monkeypatch):
    # Not over the relay: the Telegram allowlist decides, and an unlisted sender is denied.
    from gateway.session import SessionSource
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '8975409107')
    monkeypatch.delenv('GATEWAY_ALLOW_ALL_USERS', raising=False)
    monkeypatch.delenv('TELEGRAM_ALLOW_ALL_USERS', raising=False)
    runner = GatewayRunner(GatewayConfig())
    stranger = SessionSource(platform=Platform.TELEGRAM, chat_id='111', chat_type='dm', user_id='111')
    owner = SessionSource(platform=Platform.TELEGRAM, chat_id='8975409107', chat_type='dm', user_id='8975409107')
    assert runner._is_user_authorized_for_source(stranger, allow_adapter_delegation=True) is False
    assert runner._is_user_authorized_for_source(owner, allow_adapter_delegation=True) is True
