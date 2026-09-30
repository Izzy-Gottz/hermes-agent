"""The cron inactivity watchdog counts idle time only while the machine is awake (Memoe, 2026-09-30:
a Mac's 15-minute maintenance sleep was read as a 905 s idle and the job killed 6 s after the wake,
five mornings running)."""
from cron import scheduler


class _Agent:
    def __init__(self):
        self.idle = 0.0

    def get_activity_summary(self):
        return {"seconds_since_activity": self.idle}


def test_a_sleep_is_not_idle_time(monkeypatch):
    clock = {"wall": 1000.0, "mono": 50.0}
    monkeypatch.setattr(scheduler.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: clock["mono"])
    agent, state = _Agent(), {"at": None, "idle": 0.0, "mono": clock["mono"]}
    # Working: 5 s of real idle, awake.
    clock["wall"] += 5; clock["mono"] += 5; agent.idle = 5
    assert scheduler._awake_idle_seconds(agent, state) == 5
    # The Mac sleeps 900 s: the wall clock moves, the monotonic one does not.
    clock["wall"] += 900; clock["mono"] += 5; agent.idle = 910
    assert scheduler._awake_idle_seconds(agent, state) == 10
    # Awake and truly idle: it accumulates to the limit like before.
    for _ in range(60):
        clock["wall"] += 10; clock["mono"] += 10; agent.idle += 10
        idle = scheduler._awake_idle_seconds(agent, state)
    assert idle == 610


def test_new_activity_restarts_the_count(monkeypatch):
    clock = {"wall": 1000.0, "mono": 50.0}
    monkeypatch.setattr(scheduler.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(scheduler.time, "monotonic", lambda: clock["mono"])
    agent, state = _Agent(), {"at": None, "idle": 0.0, "mono": clock["mono"]}
    for _ in range(30):
        clock["wall"] += 10; clock["mono"] += 10; agent.idle += 10
        scheduler._awake_idle_seconds(agent, state)
    clock["wall"] += 10; clock["mono"] += 10; agent.idle = 2        # a tool just returned
    assert scheduler._awake_idle_seconds(agent, state) <= 10
