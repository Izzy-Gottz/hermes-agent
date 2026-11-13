"""Memoe: what the engine says on its own reaches a person, so it reads like one.

Measured on 2026-11-13 in the owner's Telegram: "⚠️ Hermes is shutting down —
your current task will be interrupted…" five times in one morning (one per idle
stop of a cloud machine) and once in the middle of his "hi", plus "Low disk
space: 1.7 GB free of 1.9 GB … /autoprune on". These pin three things:

* every lifecycle ping, the startup one included, passes the one diagnostics
  gate, so one global switch quiets a platform name nobody listed (``relay``);
* none of the notices a person can receive says "Hermes" or hands them a
  command, a path or a config key;
* "low disk" means low for the volume it is on.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3

# Text a person must never be handed: the engine's name, a slash command, a CLI
# invocation, inline code, a file or config path.
TECHNICAL = [
    (re.compile(r"\bhermes\b", re.I), "names Hermes"),
    (re.compile(r"\bgateway\b", re.I), "names the gateway"),
    (re.compile(r"`"), "carries inline code"),
    (re.compile(r"(^|\s)/[a-z]"), "hands over a slash command"),
    (re.compile(r"\.(ya?ml|db|py|json)\b"), "names a file"),
    (re.compile(r"\b(cron run|config\.yaml|state\.db)\b"), "names machinery"),
]


def _problems(text: str) -> list[str]:
    return [why for rx, why in TECHNICAL if rx.search(text)]


def _en() -> dict:
    return yaml.safe_load((ROOT / "locales" / "en.yaml").read_text(encoding="utf-8"))


def _key(tree: dict, dotted: str) -> str:
    node = tree
    for part in dotted.split("."):
        node = node[part]
    return node


PERSON_FACING_KEYS = [
    "gateway.shutdown.notice_shutdown",
    "gateway.shutdown.notice_restart",
    "gateway.shutdown.cron_interrupted",
    "gateway.startup.online",
    "gateway.startup.restarted",
]


@pytest.mark.parametrize("key", PERSON_FACING_KEYS)
def test_lifecycle_notice_is_plain(key):
    text = _key(_en(), key)
    assert not _problems(text), f"{key} {_problems(text)}: {text!r}"


def test_restart_replies_are_plain():
    from gateway import session_settlement_recovery as rec
    import gateway.session_api_turn as api
    src = (ROOT / "gateway" / "session_ingress.py").read_text(encoding="utf-8")
    m = re.search(r"if reason == 'runtime_draining':.*?return '([^']*)'", src, re.S)
    assert m, "the runtime_draining reply moved; re-point this test"
    texts = {"session_ingress runtime_draining": m.group(1),
             "RESTART_INTERRUPT_NOTICE": rec.RESTART_INTERRUPT_NOTICE,
             "SHUTDOWN_INTERRUPT_NOTICE": rec.SHUTDOWN_INTERRUPT_NOTICE}
    for name, obj in vars(api).items():
        msg = getattr(obj, "MESSAGE", None)
        if isinstance(obj, type) and isinstance(msg, str):
            texts[f"{name}.MESSAGE"] = msg
    bad = {k: _problems(v) for k, v in texts.items() if _problems(v)}
    assert not bad, bad


@pytest.mark.parametrize("free_gb,total_gb,low", [
    (1.7, 1.9, False),   # the owner's cloud machine: nearly empty, was reported low
    (0.15, 1.9, True),   # the same volume, actually full
    (1.0, 500, True),    # a big disk nearly full: the absolute floor still applies
    (60, 500, False),
])
def test_low_disk_is_low_for_the_volume_it_is_on(free_gb, total_gb, low):
    from hermes_cli.prune_advisor import DiskSnapshot
    snap = DiskSnapshot(db_bytes=0, free_bytes=int(free_gb * GIB), total_bytes=int(total_gb * GIB))
    assert snap.low is low


@pytest.mark.asyncio
async def test_startup_ping_honours_the_diagnostics_gate(tmp_path, monkeypatch):
    """The startup "I'm back" ping did not pass the gate every other lifecycle notice
    passes, so the one global switch Memoe sets could not silence it."""
    from unittest.mock import AsyncMock

    import gateway.run as gateway_run
    from gateway.config import HomeChannel, Platform
    from gateway.platforms.base import SendResult
    from tests.gateway.restart_test_helpers import make_restart_runner

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="42", name="Home")
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="m"))

    import gateway.warning_notifications as wn
    monkeypatch.setattr(wn, "effective_user_config",
                        lambda: {"display": {"suppress_warning_notifications": True}})
    assert await runner._send_home_channel_startup_notifications() == set()
    adapter.send.assert_not_called()

    monkeypatch.setattr(wn, "effective_user_config", lambda: {})
    assert await runner._send_home_channel_startup_notifications() == {("telegram", "42", None)}
