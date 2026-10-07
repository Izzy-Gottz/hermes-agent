"""The cron inactivity watchdog counts idle time only while the machine is awake (Memoe, 2026-09-30:
a Mac's 15-minute maintenance sleep was read as a 905 s idle and the job killed 6 s after the wake,
five mornings running). The fork's own ``_awake_idle_seconds`` was replaced by upstream's
``AwakeIdleMeter`` (agent/session_activity.py), which the watchdog loop applies; these keep the
fork's scenarios as its contract."""
from agent import session_activity
from agent.session_activity import AwakeIdleMeter


def _clock(monkeypatch):
    clock = {"wall": 1000.0, "mono": 50.0}
    monkeypatch.setattr(session_activity.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(session_activity.time, "monotonic", lambda: clock["mono"])
    return clock


def test_a_sleep_is_not_idle_time(monkeypatch):
    clock = _clock(monkeypatch)
    meter = AwakeIdleMeter()
    # Working: 5 s of real idle, awake.
    clock["wall"] += 5; clock["mono"] += 5
    assert meter.measure(5) == 5
    # The Mac sleeps 900 s: the wall clock moves, the monotonic one does not.
    clock["wall"] += 900; clock["mono"] += 5
    assert meter.measure(910) < 60
    # Awake and truly idle: it accumulates to the limit like before.
    idle = 910
    for _ in range(60):
        clock["wall"] += 10; clock["mono"] += 10; idle += 10
        measured = meter.measure(idle)
    assert measured >= 600


def test_new_activity_restarts_the_count(monkeypatch):
    clock = _clock(monkeypatch)
    meter = AwakeIdleMeter()
    idle = 0
    for _ in range(30):
        clock["wall"] += 10; clock["mono"] += 10; idle += 10
        meter.measure(idle)
    clock["wall"] += 10; clock["mono"] += 10        # a tool just returned
    assert meter.measure(2) <= 10
