"""The history store's two settings reach the bridge the gateway spawns.

bridge.js keeps a SQLite history when WHATSAPP_HISTORY_DB names a file, and
asks the phone for everything when WHATSAPP_SYNC_FULL_HISTORY is on. The
bridge's environment starts as a copy of os.environ, and only the names on
_BRIDGE_PASSTHROUGH_ENV are then read through the profile's secret scope
(_wenv). So a value a profile's .env holds — which is where Moe's connect
step writes these two — reaches the bridge only if its name is on that list.
Off the list, the store stays off and the reader reads nothing.

The values below are therefore set ONLY in a profile scope, with os.environ
emptied of them: a value in os.environ would reach the bridge through the
copy whether or not the list names it, and the test could not fail.

These names share one tuple with WHATSAPP_DEVICE_NAME, and that tuple is
where slice B and slice C conflicted — a merge that keeps only one side's
line passes every other test in the tree.
"""

import pytest

from agent import secret_scope as ss
from gateway.config import PlatformConfig

HISTORY_NAMES = ("WHATSAPP_HISTORY_DB", "WHATSAPP_SYNC_FULL_HISTORY")
ALL_NAMES = HISTORY_NAMES + ("WHATSAPP_DEVICE_NAME",)


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _bridge_env_from_profile(tmp_path, monkeypatch, dotenv: str) -> dict:
    """The bridge env for a profile whose .env is ``dotenv``, with nothing
    in os.environ for any of the names under test."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    for name in ALL_NAMES:
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(dotenv)
    ss.set_multiplex_active(True)
    tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
    try:
        adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
        return adapter._bridge_env()
    finally:
        ss.reset_secret_scope(tok)


@pytest.mark.parametrize("name", HISTORY_NAMES)
def test_history_setting_is_on_the_passthrough_list(name):
    from plugins.platforms.whatsapp.adapter import _BRIDGE_PASSTHROUGH_ENV
    assert name in _BRIDGE_PASSTHROUGH_ENV


def test_a_profiles_history_settings_and_device_name_reach_the_bridge(tmp_path, monkeypatch):
    env = _bridge_env_from_profile(
        tmp_path, monkeypatch,
        "WHATSAPP_HISTORY_DB=/tmp/store/messages.db\n"
        "WHATSAPP_SYNC_FULL_HISTORY=true\n"
        "WHATSAPP_DEVICE_NAME=Moe\n",
    )
    assert env.get("WHATSAPP_HISTORY_DB") == "/tmp/store/messages.db"
    assert env.get("WHATSAPP_SYNC_FULL_HISTORY") == "true"
    assert env.get("WHATSAPP_DEVICE_NAME") == "Moe"


def test_unset_history_settings_are_not_invented(tmp_path, monkeypatch):
    env = _bridge_env_from_profile(tmp_path, monkeypatch, "WHATSAPP_MODE=bot\n")
    for name in HISTORY_NAMES:
        assert name not in env
