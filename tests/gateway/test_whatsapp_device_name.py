"""The linked-device name reaches the bridge the gateway spawns.

bridge.js lists the link on the person's phone under WHATSAPP_DEVICE_NAME
(default 'Hermes Agent'). The gateway builds the bridge's environment from an
explicit passthrough list rather than the whole process env, so a name that
is set but not on that list is a name the phone never sees.
"""

from gateway.config import PlatformConfig


class TestDeviceNamePassthrough:
    def _env(self, monkeypatch, value):
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
        if value is None:
            monkeypatch.delenv("WHATSAPP_DEVICE_NAME", raising=False)
        else:
            monkeypatch.setenv("WHATSAPP_DEVICE_NAME", value)
        adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
        return adapter._bridge_env()

    def test_device_name_is_passed_to_the_bridge(self, monkeypatch):
        env = self._env(monkeypatch, "Moe")
        assert env.get("WHATSAPP_DEVICE_NAME") == "Moe"

    def test_unset_device_name_is_not_invented(self, monkeypatch):
        env = self._env(monkeypatch, None)
        assert "WHATSAPP_DEVICE_NAME" not in env

    def test_device_name_is_on_the_passthrough_list(self):
        from plugins.platforms.whatsapp.adapter import _BRIDGE_PASSTHROUGH_ENV
        assert "WHATSAPP_DEVICE_NAME" in _BRIDGE_PASSTHROUGH_ENV
