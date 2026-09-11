"""The rate-limit headroom is written to $HERMES_HOME/rate-limit.json.

The CLI reports ``rate_limit_info.unifiedWindows`` on every turn. Until now
Hermes read it once, to say "heads up" at 90 %, and dropped it. A host that
plans around the subscription — pausing discretionary background work above
half of the five-hour window — had nothing to read. Now it has a file.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from agent.transports import claude_code_session as S


@pytest.fixture(autouse=True)
def hermes_home(tmp_path: Path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


class _Session:
    def __init__(self):
        self.events = []

    def _emit(self, event):
        self.events.append(event)


def _event(five=0.42, seven=0.07, **extra):
    windows = {
        "five_hour": {"utilization": five, "resetsAt": 1757520000, "status": "allowed"},
        "seven_day": {"utilization": seven, "resetsAt": 1757980800, "status": "allowed"},
    }
    windows.update(extra)
    return {"type": "rate_limit_event", "rate_limit_info": {"unifiedWindows": windows}}


def _snapshot(home: Path) -> dict:
    return json.loads((home / "rate-limit.json").read_text())


# ── written from the projector, on the real event shape ───────────────────

def test_every_rate_limit_event_is_persisted(hermes_home):
    session = _Session()
    projector = S._TurnProjector(session, silent=False)
    before = int(time.time())
    projector._on_rate_limit(_event())

    snap = _snapshot(hermes_home)
    assert snap["provider"] == "claude-code-cli"
    assert before <= snap["at"] <= int(time.time())
    assert set(snap["windows"]) == {"five_hour", "seven_day"}
    assert snap["windows"]["five_hour"] == {"utilization": 0.42, "resets_at": 1757520000}
    assert snap["windows"]["seven_day"]["utilization"] == 0.07
    # Below 90 %: no status line for the person, but the file is there.
    assert not [e for e in session.events if e.get("kind") == "status"]


def test_window_names_are_kept_as_claude_sends_them(hermes_home):
    projector = S._TurnProjector(_Session(), silent=False)
    projector._on_rate_limit(_event(some_new_window={"utilization": 0.5}))
    assert "some_new_window" in _snapshot(hermes_home)["windows"]
    assert "five hour" not in _snapshot(hermes_home)["windows"]


def test_the_status_line_at_ninety_percent_still_fires(hermes_home):
    session = _Session()
    projector = S._TurnProjector(session, silent=False)
    projector._on_rate_limit(_event(five=0.93))
    statuses = [e for e in session.events if e.get("kind") == "status"]
    assert statuses and "93%" in statuses[0]["text"]
    assert _snapshot(hermes_home)["windows"]["five_hour"]["utilization"] == 0.93


def test_a_silent_projector_still_writes_the_file(hermes_home):
    session = _Session()
    projector = S._TurnProjector(session, silent=True)
    projector._on_rate_limit(_event(five=0.95))
    assert not session.events
    assert _snapshot(hermes_home)["windows"]["five_hour"]["utilization"] == 0.95


def test_a_later_event_replaces_the_earlier_one(hermes_home):
    projector = S._TurnProjector(_Session(), silent=False)
    projector._on_rate_limit(_event(five=0.10))
    projector._on_rate_limit(_event(five=0.60))
    assert _snapshot(hermes_home)["windows"]["five_hour"]["utilization"] == 0.60


def test_resets_at_is_passed_through_whatever_its_spelling(hermes_home):
    projector = S._TurnProjector(_Session(), silent=False)
    projector._on_rate_limit({"rate_limit_info": {"unifiedWindows": {
        "five_hour": {"utilization": 0.2, "resets_at": "2026-09-10T20:00:00Z"},
        "seven_day": {"utilization": 0.1},
    }}})
    snap = _snapshot(hermes_home)
    assert snap["windows"]["five_hour"]["resets_at"] == "2026-09-10T20:00:00Z"
    assert snap["windows"]["seven_day"]["resets_at"] is None


def test_a_non_numeric_utilization_is_null_not_a_crash(hermes_home):
    projector = S._TurnProjector(_Session(), silent=False)
    projector._on_rate_limit({"rate_limit_info": {"unifiedWindows": {
        "five_hour": {"utilization": "n/a"},
        "junk": "not a window",
    }}})
    snap = _snapshot(hermes_home)
    assert snap["windows"]["five_hour"]["utilization"] is None
    assert "junk" not in snap["windows"]


def test_an_event_without_windows_writes_nothing(hermes_home):
    projector = S._TurnProjector(_Session(), silent=False)
    projector._on_rate_limit({"rate_limit_info": {}})
    projector._on_rate_limit({"rate_limit_info": {"unifiedWindows": []}})
    assert not (hermes_home / "rate-limit.json").exists()


# ── the file itself ───────────────────────────────────────────────────────

def test_the_file_is_owner_only(hermes_home):
    S.write_rate_limit_snapshot({"five_hour": {"utilization": 0.1}})
    mode = stat.S_IMODE((hermes_home / "rate-limit.json").stat().st_mode)
    assert mode == 0o600


def test_the_write_is_atomic_and_leaves_no_temp_file(hermes_home, monkeypatch):
    """A reader mid-read must see the old snapshot or the new one, never a
    half-written file."""
    path = hermes_home / "rate-limit.json"
    S.write_rate_limit_snapshot({"five_hour": {"utilization": 0.1}})
    old = path.read_text()

    real_replace = os.replace
    seen = {}

    def spy(src, dst):
        seen["src"], seen["dst"] = src, dst
        seen["target_before_rename"] = path.read_text()
        return real_replace(src, dst)

    monkeypatch.setattr(S.os, "replace", spy)
    S.write_rate_limit_snapshot({"five_hour": {"utilization": 0.9}})
    assert seen["dst"] == str(path)
    assert seen["src"] != str(path)
    assert seen["target_before_rename"] == old, "the target was written in place"
    assert json.loads(path.read_text())["windows"]["five_hour"]["utilization"] == 0.9
    assert [p.name for p in hermes_home.iterdir()] == ["rate-limit.json"]


def test_a_failed_write_keeps_the_old_snapshot_and_no_temp_file(hermes_home, monkeypatch):
    path = hermes_home / "rate-limit.json"
    S.write_rate_limit_snapshot({"five_hour": {"utilization": 0.1}})
    old = path.read_text()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(S.json, "dump", boom)
    assert S.write_rate_limit_snapshot({"five_hour": {"utilization": 0.9}}) is None
    assert path.read_text() == old
    assert [p.name for p in hermes_home.iterdir()] == ["rate-limit.json"]


def test_a_write_failure_never_reaches_the_turn(tmp_path, monkeypatch, caplog):
    """HERMES_HOME pointing at a regular file: the write cannot succeed, the
    turn that carried the event must not notice."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")
    monkeypatch.setenv("HERMES_HOME", str(blocker))
    projector = S._TurnProjector(_Session(), silent=False)
    with caplog.at_level("WARNING"):
        projector._on_rate_limit(_event())  # must not raise
    assert any("rate-limit snapshot" in r.getMessage() for r in caplog.records)


def test_the_path_is_under_hermes_home(hermes_home):
    assert S.rate_limit_snapshot_path() == str(hermes_home / "rate-limit.json")


def test_at_is_epoch_seconds(hermes_home):
    S.write_rate_limit_snapshot({"five_hour": {"utilization": 0.1}}, now=1757500000.7)
    assert _snapshot(hermes_home)["at"] == 1757500000
