"""The kept profile: the real-profile lane on a cloud computer, where there is no person's browser.

What it is for (Moe's Away machine, 2026-09-29): the cloud computer browsed signed out, every call
on a throwaway profile, so a site the person signed in to once was signed out again next time. With
``browser.kept_profile`` the driven browser runs on its OWN profile, ``browser-profile/kept``, and
keeps it across launches. So what must hold: nothing of the person's is looked for or copied, the
profile on disk is launched as it was left (a snapshot or a hand-over would overwrite the only copy
of those sign-ins), and it is created owner-only.
"""
import os
import stat
from unittest.mock import patch

import pytest

import hermes_cli.browser_connect as bc
from tests.tools.test_browser_real_profile import TestRealProfileCdpLaunch as _Launch, _FakeChrome
from tools import browser_tool_cloud as bt_cloud
from tools import browser_tool_install as bt_install
from tools import browser_tool_real_profile as bt_real_profile

DRIVEN = "/opt/data/.agent-browser/browsers/chrome-153.0.8010.47/chrome-linux64/chrome"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hh"
    monkeypatch.setattr(bc, "get_hermes_home", lambda: h)
    return h


def _cold_start(home, *, launches, cdp_calls, detect=None, snapshot=None):
    """``_real_profile_cdp`` cold, with ``kept_profile`` on and only process launch and CDP faked."""
    import tools.browser_tool as bt
    kept = home / "browser-profile" / "kept"

    def fake_popen(argv, **kw):
        launches.append(argv)
        kept.mkdir(parents=True, exist_ok=True)
        (kept / "DevToolsActivePort").write_text("41000\n/devtools/browser/x\n")
        return _FakeChrome()

    def fake_cdp(port, method, params=None, timeout=20.0):
        cdp_calls.append(method)
        return {}

    ok = type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    patches = [
        patch.object(bt_cloud, "_use_real_profile", return_value=True),
        patch.object(bt_cloud, "_use_kept_profile", return_value=True),
        patch.object(bt_cloud, "_is_headed_mode", return_value=False),
        patch("hermes_cli.browser_connect.detect_default_chromium", side_effect=detect or (lambda *a: None)),
        patch("hermes_cli.browser_connect.snapshot_real_profile", side_effect=snapshot or (lambda *a, **k: (None, "snapshot ran"))),
        patch.object(bt_real_profile, "driven_browser_executable", return_value=DRIVEN),
        patch.object(bt_real_profile, "_cdp_call", side_effect=fake_cdp),
        patch.object(bt_real_profile, "_browsers_on_data_dir", return_value=[]),
        patch.object(bt.subprocess, "Popen", side_effect=fake_popen),
        patch.object(bt_real_profile, "_agent_browser_get_cdp", side_effect=[None, "http://127.0.0.1:41000"]),
        patch.object(bt_install, "_find_agent_browser", return_value="/usr/bin/agent-browser"),
        patch.object(bt.subprocess, "run", side_effect=lambda *a, **k: ok),
    ]
    for p in patches:
        p.start()
    try:
        return bt_real_profile._real_profile_cdp()
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.fixture(autouse=True)
def _clean():
    _Launch()._reset()
    yield
    _Launch()._reset()


def test_a_cloud_computer_with_no_default_browser_launches_the_kept_profile(home):
    """No Chromium default (a Linux machine has none) is the fail-closed case for the person's lane;
    for the kept one it is simply the situation. Only the driven browser runs, on ``kept``."""
    launches, cdp_calls = [], []
    cdp, err = _cold_start(home, launches=launches, cdp_calls=cdp_calls)
    assert err is None and cdp == "http://127.0.0.1:41000"
    assert [argv[0] for argv in launches] == [DRIVEN]
    assert f"--user-data-dir={home / 'browser-profile' / 'kept'}" in launches[0]
    assert "--headless=new" in launches[0]


def test_nothing_is_copied_into_the_kept_profile(home):
    """The kept profile is the only copy of its sign-ins: no snapshot overlay, no cookie hand-over
    (no Storage.* call), and a cookie store already on disk is still there after the launch."""
    kept = home / "browser-profile" / "kept" / "Default" / "Network"
    kept.mkdir(parents=True)
    (kept / "Cookies").write_text("signed in to x.com yesterday")
    snapshots, launches, cdp_calls = [], [], []
    cdp, err = _cold_start(home, launches=launches, cdp_calls=cdp_calls,
                           snapshot=lambda *a, **k: snapshots.append(a) or (None, "must not run"))
    assert err is None and cdp
    assert snapshots == []
    assert not [m for m in cdp_calls if m.startswith("Storage.")]
    assert (kept / "Cookies").read_text() == "signed in to x.com yesterday"


def test_the_kept_profile_is_created_owner_only(home):
    launches, cdp_calls = [], []
    cdp, err = _cold_start(home, launches=launches, cdp_calls=cdp_calls)
    assert err is None
    mode = stat.S_IMODE(os.stat(home / "browser-profile" / "kept").st_mode)
    assert mode == 0o700


def test_the_persons_default_browser_is_never_consulted(home):
    """Even where a default browser exists, the kept lane never reads it: a kept profile that
    followed it would be a snapshot of someone's browser again."""
    asked = []
    launches, cdp_calls = [], []
    cdp, err = _cold_start(home, launches=launches, cdp_calls=cdp_calls,
                           detect=lambda *a: asked.append(1) or "chrome")
    assert err is None and cdp
    assert asked == []


def test_kept_profile_off_keeps_the_persons_lane(home):
    """Without ``kept_profile`` a host with no Chromium default fails closed, as before."""
    with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
         patch.object(bt_cloud, "_use_kept_profile", return_value=False), \
         patch("hermes_cli.browser_connect.detect_default_chromium", return_value=None):
        cdp, err = bt_real_profile._real_profile_cdp()
    assert cdp is None and "not a supported Chromium" in err


def test_kept_profile_reads_its_config_key():
    import tools.browser_tool as bt
    with patch.object(bt, "_browser_cfg", side_effect=lambda key, default, *a: key == "kept_profile"):
        assert bt_cloud._use_kept_profile() is True
    with patch.object(bt, "_browser_cfg", side_effect=lambda key, default, *a: default):
        assert bt_cloud._use_kept_profile() is False
