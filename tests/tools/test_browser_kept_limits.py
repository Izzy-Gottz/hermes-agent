"""(fork) The kept browser on a cloud computer: launched small, killed first, capped, and floored."""

from __future__ import annotations

import sys

import pytest

from tools import browser_kept_limits as kl
from tools import browser_tool_cloud as cloud
from tools import browser_tool_real_profile as rp


@pytest.fixture
def cloud_lane(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(cloud, "_use_kept_profile", lambda: True)
    cfg = {}
    monkeypatch.setattr(kl, "_cfg", lambda key, default, parse: parse(cfg[key]) if key in cfg else default)
    return cfg


def test_a_mac_never_takes_the_cloud_lane(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(cloud, "_use_kept_profile", lambda: True)
    assert kl.active() is False and kl.kept_flags() == ()


def test_linux_without_the_kept_profile_is_not_the_cloud_lane(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(cloud, "_use_kept_profile", lambda: False)
    assert kl.active() is False


def test_the_cloud_lane_launches_small(cloud_lane, monkeypatch):
    monkeypatch.setattr(rp._fidelity, "fidelity_enabled", lambda: False)
    flags = rp._driven_browser_flags(None, True)
    for flag in kl.KEPT_FLAGS:
        assert flag in flags
    assert not any(f in flags for f in ("--no-sandbox", "--disable-site-isolation-trials", "--single-process"))


def test_the_browser_is_the_oom_killers_first_choice(cloud_lane, tmp_path):
    (tmp_path / "4242").mkdir()
    (tmp_path / "4242" / "oom_score_adj").write_text("0")
    assert kl.mark_expendable(4242, proc_root=str(tmp_path)) is True
    assert (tmp_path / "4242" / "oom_score_adj").read_text() == "1000"


def test_off_the_cloud_lane_nothing_is_marked(monkeypatch, tmp_path):
    monkeypatch.setattr(kl, "active", lambda: False)
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "oom_score_adj").write_text("0")
    assert kl.mark_expendable(7, proc_root=str(tmp_path)) is False
    assert (tmp_path / "7" / "oom_score_adj").read_text() == "0"


def test_trim_closes_the_oldest_pages_beyond_the_cap():
    closed = []
    pages = lambda port: [{"id": "new"}, {"id": "mid"}, {"id": "old1"}, {"id": "old2"}]  # noqa: E731
    n = kl.trim_tabs(1, 2, pages=pages, close=lambda port, tid: closed.append(tid))
    assert n == 2 and closed == ["old1", "old2"]
    assert kl.trim_tabs(1, 0, pages=pages, close=lambda *a: closed.append("x")) == 0


def test_enforce_caps_tabs(cloud_lane):
    cloud_lane["max_tabs"] = 3
    seen = []
    did = kl.enforce(9222, lambda: seen.append("released"), available=lambda: 5000,
                     trim=lambda port, keep: seen.append(("trim", keep)) or 1)
    assert did == "trimmed" and seen == [("trim", 3)]


def test_under_the_floor_tabs_go_first_then_the_browser(cloud_lane):
    cloud_lane["memory_floor_mb"] = 300
    seen = []
    did = kl.enforce(9222, lambda: seen.append("released"), available=lambda: 120,
                     trim=lambda port, keep: seen.append(("trim", keep)) or 1)
    assert did == "released" and seen == [("trim", 1), "released"]


def test_a_trim_that_frees_enough_keeps_the_browser(cloud_lane):
    cloud_lane["memory_floor_mb"] = 300
    free = iter([120, 900])
    seen = []
    did = kl.enforce(9222, lambda: seen.append("released"), available=lambda: next(free),
                     trim=lambda port, keep: 1)
    assert did == "trimmed" and seen == []


def test_off_the_cloud_lane_enforce_does_nothing(monkeypatch):
    monkeypatch.setattr(kl, "active", lambda: False)
    assert kl.enforce(9222, lambda: pytest.fail("released on a Mac"), available=lambda: 1,
                      trim=lambda *a: pytest.fail("trimmed on a Mac")) == ""


def test_meminfo_is_read_in_megabytes(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal:        2000000 kB\nMemFree:          10 kB\nMemAvailable:     512000 kB\n")
    assert kl.mem_available_mb(str(f)) == 500
    assert kl.mem_available_mb(str(tmp_path / "missing")) is None


def test_every_launch_of_the_kept_browser_is_marked(cloud_lane, monkeypatch, tmp_path):
    marked = []
    monkeypatch.setattr(kl, "mark_expendable", lambda pid: marked.append(pid) or True)

    class Proc:
        pid = 31337

        def poll(self):
            return None

    monkeypatch.setattr(rp.subprocess, "Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(rp, "_read_devtools_port", lambda d: "9222")
    monkeypatch.setattr(rp._origin(), "_build_browser_env", lambda: {})
    proc, port, err = rp._spawn_browser_on_copy("/bin/chrome", str(tmp_path), (), "browser", headless=True)
    assert (port, err) == (9222, None) and marked == [31337]
    rp._origin()._real_profile_chrome_procs.remove(proc)
