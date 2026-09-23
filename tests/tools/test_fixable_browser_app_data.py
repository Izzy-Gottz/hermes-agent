"""Ticket #8, browser half: macOS app-data protection is not a profile lock.

Measured 2026-09-23 on macOS 27.0: ``~/Library/Application Support/Google/Chrome`` returns EPERM
("Operation not permitted", Errno 1) to every process without Full Disk Access. Hermes read that
PermissionError as "chrome is running and has its profile locked… Fully quit the browser", and the
model told the person to quit Chrome, which cannot help. It is now ``tcc_app_data``.

Every test here produces the REAL condition — a directory or file chmod'd to 000 (the kernel
raises PermissionError), or ``open`` raising ``PermissionError(1, 'Operation not permitted')``
exactly as macOS TCC does — and never a string that merely looks like the error.
"""

import builtins
import json
import os
import sys
from unittest.mock import patch

import pytest

import hermes_cli.browser_connect as bc
from tools import fix_reasons

pytestmark = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                reason="chmod 000 does not deny root, and the condition is macOS-only")

EPERM = (1, "Operation not permitted")


def _auth_db(path, value):
    import sqlite3
    from contextlib import closing
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as con:
        con.execute("CREATE TABLE t (v TEXT)")
        con.execute("INSERT INTO t VALUES (?)", (value,))
        con.commit()


def _profile(root):
    """A Chrome user-data dir shaped like the real one: Local State + a signed-in profile."""
    (root / "Profile 6" / "Network").mkdir(parents=True)
    (root / "Default").mkdir(parents=True)
    (root / "Local State").write_text('{"profile": {"last_used": "Profile 6"}}')
    _auth_db(root / "Profile 6" / "Network" / "Cookies", "SESSION")
    _auth_db(root / "Profile 6" / "Login Data", "logins")
    (root / "Profile 6" / "Preferences").write_text("{}")
    return root


@pytest.fixture
def darwin(monkeypatch, tmp_path):
    monkeypatch.setattr(bc.sys, "platform", "darwin")
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    monkeypatch.delenv("HERMES_HOST_APP_NAME", raising=False)
    return tmp_path


@pytest.fixture
def chmod000():
    """chmod paths to 000 and always restore them, so pytest can clean tmp_path."""
    locked = []

    def lock(path):
        locked.append((path, path.stat().st_mode))
        path.chmod(0)

    yield lock
    for path, mode in reversed(locked):
        path.chmod(mode)


def _assert_tcc_app_data(err, src):
    assert err, "expected an error"
    fields = fix_reasons.fields_of(err)
    assert fields == {"code": "tcc_app_data", "owner": "app", "pane": "Privacy_AllFiles",
                      "subject": "Google Chrome", "retry": True, "browser": "chrome", "path": str(src)}
    text = str(err)
    assert not text.startswith(bc._PROFILE_LOCKED_PREFIX)
    assert "profile locked" not in text and "close-profile" not in text
    assert "Full Disk Access" in text and "Google Chrome" in text
    assert "Don't quit Google Chrome" in text


class TestSnapshotOnDarwin:
    def test_user_data_dir_chmod_000(self, darwin, chmod000):
        """The whole browser dir refuses to list: the shape of macOS app-data protection."""
        src = _profile(darwin / "Chrome")
        chmod000(src)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_tcc_app_data(err, src)

    def test_cookie_db_chmod_000(self, darwin, chmod000):
        """The dir lists and Local State reads, but the profile's cookie DB is denied — the very
        probe (_profile_is_locked) that used to answer "locked"."""
        src = _profile(darwin / "Chrome")
        chmod000(src / "Profile 6" / "Network" / "Cookies")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_tcc_app_data(err, src)

    def test_login_data_chmod_000_is_not_database_unavailable(self, darwin, chmod000):
        """Denied only on a DB the probes never open: surfaces in _copy_auth_file, which used to
        count it as "1 database(s) unavailable. Close chrome and retry"."""
        src = _profile(darwin / "Chrome")
        chmod000(src / "Profile 6" / "Login Data")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert "unavailable" not in str(err) and "Close chrome" not in str(err)
        _assert_tcc_app_data(err, src)

    def test_open_raises_eperm_errno_1(self, darwin, monkeypatch):
        """``open`` raising exactly what TCC raises, for every path under the browser dir. Local
        State is the first thing read, and an unreadable Local State used to resolve silently to
        "Default"."""
        src = _profile(darwin / "Chrome")
        real_open = builtins.open

        def tcc_open(path, *a, **k):
            if str(path).startswith(str(src)):
                raise PermissionError(*EPERM)
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", tcc_open)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_tcc_app_data(err, src)

    def test_stat_eperm_is_not_directory_not_found(self, darwin, monkeypatch):
        """os.path.isdir swallows EPERM; a protected dir must not read as "not found"."""
        src = _profile(darwin / "Chrome")
        real_stat = os.stat

        def tcc_stat(path, *a, **k):
            if str(path) == str(src):
                raise PermissionError(*EPERM)
            return real_stat(path, *a, **k)

        monkeypatch.setattr(os, "stat", tcc_stat)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert "was not found" not in str(err)
        _assert_tcc_app_data(err, src)

    def test_host_app_name_hook(self, darwin, chmod000, monkeypatch):
        monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
        src = _profile(darwin / "Chrome")
        chmod000(src)
        _, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert str(err).startswith("Memoe isn't allowed to read Google Chrome's data")
        assert "Turn Memoe on in System Settings › Privacy & Security › Full Disk Access" in str(err)

    def test_readable_profile_still_snapshots(self, darwin):
        """The probes must not refuse a profile that is actually readable."""
        src = _profile(darwin / "Chrome")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None and dst


class TestWindowsKeepsTheLock:
    def test_permissionerror_on_windows_is_still_profile_locked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bc.sys, "platform", "win32")
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
        monkeypatch.setattr(bc, "_real_profile_autoclose", lambda: False)
        src = _profile(tmp_path / "Chrome")
        real_open = builtins.open

        def share_violation(path, *a, **k):
            if str(path).endswith("Cookies"):
                raise PermissionError(13, "The process cannot access the file")
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", share_violation)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert err.startswith(bc._PROFILE_LOCKED_PREFIX)
        assert fix_reasons.fields_of(err)["code"] == "profile_locked"


# ── the error reaches the model with its code, by every route ───────────────────────────────────

def _cold_real_profile_patches(src):
    """Consent on, default browser Chrome at ``src``, nothing already running: the real
    snapshot_real_profile runs against ``src``."""
    from tools import browser_tool_cloud as bt_cloud
    from tools import browser_tool_real_profile as bt_rp
    return [
        patch.object(bt_cloud, "_use_real_profile", return_value=True),
        patch("hermes_cli.browser_connect.detect_default_chromium", return_value="chrome"),
        patch("hermes_cli.browser_connect.real_profile_data_dir", return_value=str(src)),
        patch.object(bt_rp, "_agent_browser_get_cdp", return_value=None),
        patch.object(bt_rp, "_terminate_orphaned_browsers_on_dir", return_value=None),
        patch.object(bt_rp, "_surviving_chrome_cdp", return_value=None),
    ]


@pytest.fixture
def cold(darwin):
    import tools.browser_tool as bt
    bt._real_profile_cdp_cache.clear()
    yield
    bt._real_profile_cdp_cache.clear()


def _contract(payload):
    return {k: payload.get(k) for k in ("code", "owner", "pane", "subject", "retry")}


EXPECTED = {"code": "tcc_app_data", "owner": "app", "pane": "Privacy_AllFiles",
            "subject": "Google Chrome", "retry": True}


class TestReachesTheModel:
    def test_real_profile_cdp_passes_the_fix_message_through(self, darwin, cold, chmod000):
        from tools import browser_tool_real_profile as bt_rp
        src = _profile(darwin / "Chrome")
        chmod000(src)
        patches = _cold_real_profile_patches(src)
        for p in patches:
            p.start()
        try:
            cdp, err = bt_rp._real_profile_cdp()
        finally:
            for p in reversed(patches):
                p.stop()
        assert cdp is None
        _assert_tcc_app_data(err, src)
        assert not str(err).startswith(bt_rp._RP)  # the person's words, not "use_real_profile is on, but"

    def test_browser_exec(self, darwin, cold, chmod000, monkeypatch, tmp_path):
        """browser_exec (Browser Use mode, what Moe drives) returns the contract."""
        from tools import browser_use_cli as bu_cli
        src = _profile(darwin / "Chrome")
        chmod000(src)
        cli = tmp_path / "browser-use"
        cli.write_text("#!/bin/sh\necho SHOULD-NOT-RUN\n")
        cli.chmod(0o755)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override_raw", lambda: "")
        monkeypatch.setattr("tools.browser_tool_cloud._get_cloud_provider", lambda: None)
        patches = _cold_real_profile_patches(src)
        for p in patches:
            p.start()
        try:
            out = json.loads(bu_cli.browser_exec("print(1)"))
        finally:
            for p in reversed(patches):
                p.stop()
        assert _contract(out) == EXPECTED, out
        assert out["path"] == str(src)
        assert "Full Disk Access" in out["error"] and "profile locked" not in out["error"]
        assert "SHOULD-NOT-RUN" not in json.dumps(out)

    def test_builtin_browser_tools(self, darwin, cold, chmod000):
        """The built-in browser tools: session creation raises FixableError; the command runner and
        the registry both keep the fields."""
        from tools import browser_tool_session as bt_session
        from tools.registry import ToolRegistry
        import tools.browser_tool as bt
        src = _profile(darwin / "Chrome")
        chmod000(src)
        patches = _cold_real_profile_patches(src) + [
            patch.object(bt_session, "_browser_command_preflight", return_value={"browser_cmd": "agent-browser"}),
            patch("tools.browser_tool_cdp._get_cdp_override", return_value=""),
            patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None),
        ]
        for p in patches:
            p.start()
        try:
            with pytest.raises(fix_reasons.FixableError) as raised:
                bt_session._create_local_session("t-8")
            result = bt_session._run_browser_command("t-8b", "snapshot")
            failed = json.loads(bt._failed_response(result, "Failed"))
            reg = ToolRegistry()
            reg.register("probe_nav", "browser", {"name": "probe_nav", "parameters": {"type": "object"}},
                         lambda args, **kw: bt_session._create_local_session("t-8c"))
            dispatched = json.loads(reg.dispatch("probe_nav", {}))
        finally:
            for p in reversed(patches):
                p.stop()
        assert fix_reasons.fields_of(raised.value)["code"] == "tcc_app_data"
        assert _contract(result) == EXPECTED and result["success"] is False
        assert _contract(failed) == EXPECTED
        assert _contract(dispatched) == EXPECTED, dispatched
        assert "Tool execution failed" not in dispatched["error"]
