"""Ticket #8, browser half: macOS app-data protection is not a profile lock, and not EACCES either.

Measured 2026-09-23 on macOS 27.0: ``~/Library/Application Support/Google/Chrome`` returns **EPERM**
(errno 1, "Operation not permitted") to every process without Full Disk Access. Hermes read that
PermissionError as "chrome is running and has its profile locked… Fully quit the browser". It is
now ``tcc_app_data``.

Python raises EPERM and EACCES (errno 13, "Permission denied" — ordinary mode bits) as the same
PermissionError, and only EPERM is the privacy grant. So:

* EPERM is produced the way TCC produces it: ``PermissionError(errno.EPERM, 'Operation not
  permitted', <path>)`` from the call that touched the protected path. chmod cannot produce EPERM.
* EACCES is produced for real: chmod 000 on a source file or dir, and a snapshot destination
  whose parent is mode 0500. Neither may carry a code; Full Disk Access fixes neither.
* An EPERM on Hermes' OWN snapshot dir is not the browser's data being protected: no code.
"""

import builtins
import errno
import json
import os

import pytest

import hermes_cli.browser_connect as bc
from tools import fix_reasons

pytestmark = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                reason="chmod 000 does not deny root, and the condition is macOS-only")


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


def _eperm(path):
    return PermissionError(errno.EPERM, os.strerror(errno.EPERM), str(path))


@pytest.fixture
def darwin(monkeypatch, tmp_path):
    monkeypatch.setattr(bc.sys, "platform", "darwin")
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    monkeypatch.delenv("HERMES_HOST_APP_NAME", raising=False)
    return tmp_path


@pytest.fixture
def tcc(monkeypatch):
    """Make every open()/listdir()/scandir() under a path raise EPERM with that path, as TCC does."""
    def install(root):
        real_open, real_listdir, real_scandir = builtins.open, os.listdir, os.scandir
        target = str(root)

        def hit(path):
            return isinstance(path, (str, os.PathLike)) and os.fspath(path).startswith(target)

        def tcc_open(path, *a, **k):
            if hit(path):
                raise _eperm(path)
            return real_open(path, *a, **k)

        def tcc_listdir(path="."):
            if hit(path):
                raise _eperm(path)
            return real_listdir(path)

        def tcc_scandir(path="."):
            if hit(path):
                raise _eperm(path)
            return real_scandir(path)

        monkeypatch.setattr(builtins, "open", tcc_open)
        monkeypatch.setattr(os, "listdir", tcc_listdir)
        monkeypatch.setattr(os, "scandir", tcc_scandir)
    return install


@pytest.fixture
def chmod():
    """chmod paths and always restore them, so pytest can clean tmp_path."""
    changed = []

    def apply(path, mode):
        changed.append((path, path.stat().st_mode))
        path.chmod(mode)

    yield apply
    for path, mode in reversed(changed):
        path.chmod(mode)


def _assert_tcc_app_data(err, src):
    assert err, "expected an error"
    fields = fix_reasons.fields_of(err)
    assert fields.pop(fix_reasons.MAC_KEY) and fields.pop(fix_reasons.NONCE_KEY)
    assert fields == {"code": "tcc_app_data", "owner": "app", "pane": "Privacy_AllFiles",
                      "subject": "Google Chrome", "retry": True, "browser": "chrome", "path": str(src)}
    text = str(err)
    assert not text.startswith(bc._PROFILE_LOCKED_PREFIX)
    assert "profile locked" not in text and "close-profile" not in text
    assert "Full Disk Access" in text and "Don't quit Google Chrome" in text


def _assert_plain_permission_denied(err, path):
    assert err, "expected an error"
    assert fix_reasons.fields_of(err) == {}, "EACCES / a non-source path must carry no fix code"
    text = str(err)
    assert "Full Disk Access" not in text and "profile locked" not in text
    assert "permission denied on" in text.lower() and str(path) in text, text


# ── EPERM on the source: tcc_app_data ────────────────────────────────────────────────────────────

class TestEpermOnSource:
    def test_user_data_dir_refuses_to_list(self, darwin, tcc):
        """The shape measured on Chrome's dir: every read under it is EPERM."""
        src = _profile(darwin / "Chrome")
        tcc(src)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_tcc_app_data(err, src)

    def test_cookie_db_only(self, darwin, tcc):
        """The dir lists and Local State reads, but the cookie DB is EPERM — the probe that used to
        answer "locked"."""
        src = _profile(darwin / "Chrome")
        tcc(src / "Profile 6" / "Network" / "Cookies")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_tcc_app_data(err, src)

    def test_login_data_only_is_not_database_unavailable(self, darwin, tcc):
        """EPERM on a DB no probe opens: surfaces in _copy_auth_file, which used to count it as
        "1 database(s) unavailable. Close chrome and retry"."""
        src = _profile(darwin / "Chrome")
        tcc(src / "Profile 6" / "Login Data")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert "unavailable" not in str(err) and "Close chrome" not in str(err)
        _assert_tcc_app_data(err, src)

    def test_stat_eperm_is_not_directory_not_found(self, darwin, monkeypatch):
        """os.path.isdir swallows EPERM; a protected dir must not read as "not found"."""
        src = _profile(darwin / "Chrome")
        real_stat = os.stat

        def tcc_stat(path, *a, **k):
            if str(path) == str(src):
                raise _eperm(path)
            return real_stat(path, *a, **k)

        monkeypatch.setattr(os, "stat", tcc_stat)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None and "was not found" not in str(err)
        _assert_tcc_app_data(err, src)

    def test_host_app_name_hook(self, darwin, tcc, monkeypatch):
        monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
        src = _profile(darwin / "Chrome")
        tcc(src)
        _, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert str(err).startswith("Memoe isn't allowed to read Google Chrome's data")
        assert "Turn Memoe on in System Settings › Privacy & Security › Full Disk Access" in str(err)

    def test_readable_profile_still_snapshots(self, darwin):
        src = _profile(darwin / "Chrome")
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None and dst


# ── EACCES, and EPERM off the source: no code ────────────────────────────────────────────────────

class TestNotAppDataProtection:
    def test_eacces_source_dir_chmod_000(self, darwin, chmod):
        src = _profile(darwin / "Chrome")
        chmod(src, 0)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_plain_permission_denied(err, src)

    def test_eacces_cookie_db_chmod_000(self, darwin, chmod):
        src = _profile(darwin / "Chrome")
        cookies = src / "Profile 6" / "Network" / "Cookies"
        chmod(cookies, 0)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_plain_permission_denied(err, cookies)

    def test_eacces_login_data_chmod_000(self, darwin, chmod):
        src = _profile(darwin / "Chrome")
        login = src / "Profile 6" / "Login Data"
        chmod(login, 0)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_plain_permission_denied(err, login)

    def test_eacces_destination_parent_0500(self, darwin, chmod):
        """The reviewer's case: Hermes' own browser-profile dir is read-only."""
        src = _profile(darwin / "Chrome")
        parent = darwin / "hh" / "browser-profile"
        parent.mkdir(parents=True)
        chmod(parent, 0o500)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        _assert_plain_permission_denied(err, parent / "chrome")

    def test_eperm_on_the_destination_is_not_the_browsers_data(self, darwin, monkeypatch):
        """Even a real EPERM, if it is on Hermes' own snapshot dir, is not app-data protection."""
        src = _profile(darwin / "Chrome")
        dst_dir = str(darwin / "hh" / "browser-profile" / "chrome")
        real_makedirs = os.makedirs

        def denied_makedirs(path, *a, **k):
            if os.fspath(path).startswith(dst_dir):
                raise _eperm(path)
            return real_makedirs(path, *a, **k)

        monkeypatch.setattr(os, "makedirs", denied_makedirs)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert fix_reasons.fields_of(err) == {}
        assert "Full Disk Access" not in str(err)


class TestWindowsKeepsTheLock:
    def test_permissionerror_on_windows_is_still_profile_locked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bc.sys, "platform", "win32")
        monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
        monkeypatch.setattr(bc, "_real_profile_autoclose", lambda: False)
        src = _profile(tmp_path / "Chrome")
        real_open = builtins.open

        def share_violation(path, *a, **k):
            if str(path).endswith("Cookies"):
                raise PermissionError(errno.EACCES, "The process cannot access the file", str(path))
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", share_violation)
        dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert dst is None
        assert err.startswith(bc._PROFILE_LOCKED_PREFIX)
        assert fix_reasons.fields_of(err)["code"] == "profile_locked"


# ── every route to the model keeps the contract ──────────────────────────────────────────────────
# One real tcc_app_data failure (EPERM on the Chrome dir, the real snapshot_real_profile) sent
# through every way a browser failure reaches a model. FixMessage is a str, so any f-string,
# concatenation or .strip() on the way would silently drop the fields; this is what catches it.

EXPECTED = {"code": "tcc_app_data", "owner": "app", "pane": "Privacy_AllFiles",
            "subject": "Google Chrome", "retry": True}


@pytest.fixture
def denied_chrome(darwin, tcc, tmp_path, monkeypatch):
    """Consented real-profile browsing whose Chrome dir is EPERM; nothing running, no cloud, no
    CDP override, and a browser-use CLI that must never run."""
    import tools.browser_tool as bt
    from tools import browser_tool_cloud as bt_cloud
    from tools import browser_tool_real_profile as bt_rp
    from tools import browser_tool_session as bt_session
    from tools import browser_use_cli as bu_cli
    src = _profile(darwin / "Chrome")
    cli = tmp_path / "browser-use"
    cli.write_text("#!/bin/sh\necho SHOULD-NOT-RUN\n")
    cli.chmod(0o755)
    monkeypatch.setattr(bt_cloud, "_use_real_profile", lambda: True)
    monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr("hermes_cli.browser_connect.detect_default_chromium", lambda *a, **k: "chrome")
    monkeypatch.setattr("hermes_cli.browser_connect.real_profile_data_dir", lambda *a, **k: str(src))
    monkeypatch.setattr(bt_rp, "_agent_browser_get_cdp", lambda *a, **k: None)
    monkeypatch.setattr(bt_rp, "_terminate_orphaned_browsers_on_dir", lambda *a, **k: None)
    monkeypatch.setattr(bt_rp, "_surviving_chrome_cdp", lambda *a, **k: None)
    monkeypatch.setattr(bt_session, "_browser_command_preflight", lambda: {"browser_cmd": "agent-browser"})
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override", lambda *a, **k: "")
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override_raw", lambda *a, **k: "")
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])
    bt._real_profile_cdp_cache.clear()
    tcc(src)
    yield src
    bt._real_profile_cdp_cache.clear()


def _via_registry(name, args):
    import tools.browser_tool  # noqa: F401 — registers the browser tools
    import tools.browser_use_cli  # noqa: F401 — registers browser_exec
    from tools.registry import registry
    return registry.dispatch(name, args)


def _via_claude_code_bridge(name, args, monkeypatch):
    """The hop Moe uses: the claude-code child calls Hermes' tools through the hermes-tools MCP
    server, whose per-tool handler runs handle_function_call then to_mcp_content. Built for real
    by _build_server; only the SDK server object records the handlers instead of serving them."""
    import mcp.server as mcp_server
    import model_tools
    from agent.transports import hermes_tools_mcp_server as m

    class _RecordingServer:
        def __init__(self, name, instructions=None):
            self.tools = {}

        def add_tool(self, fn, name=None, description=None, meta=None, structured_output=None):
            self.tools[name or fn.__name__] = fn

    monkeypatch.setattr(mcp_server, "MCPServer", _RecordingServer)
    monkeypatch.setattr(m, "discover_external_mcp_servers", lambda: [])
    monkeypatch.setattr(m, "tools_to_offer", lambda *a, **k: [name])
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda *a, **k: [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {
            k: {"type": "string"} for k in args}}}}])
    server = m._build_server("claude-code")
    out = server.tools[name](**args)
    assert isinstance(out, str), f"the bridge must hand the child text, got {type(out)}"
    return out


def _session_runner(mp):
    from tools import browser_tool_session as bt_session
    return json.dumps(bt_session._run_browser_command("t-route", "snapshot"))


def _browser_exec(mp):
    from tools import browser_use_cli as bu_cli
    return bu_cli.browser_exec("print(1)")


ROUTES = {
    "browser_use_cli/browser_exec": _browser_exec,
    "registry/browser_exec": lambda mp: _via_registry("browser_exec", {"code": "print(1)"}),
    "registry/browser_navigate": lambda mp: _via_registry("browser_navigate", {"url": "https://example.com/"}),
    "registry/browser_console(expression)": lambda mp: _via_registry("browser_console", {"expression": "1+1"}),
    "registry/browser_console": lambda mp: _via_registry("browser_console", {}),
    "registry/browser_vision": lambda mp: _via_registry("browser_vision", {"question": "what is on it?"}),
    "registry/browser_snapshot": lambda mp: _via_registry("browser_snapshot", {}),
    "session/_run_browser_command": _session_runner,
    "claude-code-bridge/browser_exec": lambda mp: _via_claude_code_bridge("browser_exec", {"code": "print(1)"}, mp),
    "claude-code-bridge/browser_navigate": lambda mp: _via_claude_code_bridge(
        "browser_navigate", {"url": "https://example.com/"}, mp),
    "claude-code-bridge/browser_console": lambda mp: _via_claude_code_bridge(
        "browser_console", {"expression": "1+1"}, mp),
    "claude-code-bridge/browser_vision": lambda mp: _via_claude_code_bridge(
        "browser_vision", {"question": "what is on it?"}, mp),
}


@pytest.mark.parametrize("route", list(ROUTES))
def test_every_route_delivers_the_contract(route, denied_chrome, monkeypatch):
    raw = ROUTES[route](monkeypatch)
    out = json.loads(raw) if isinstance(raw, str) else raw
    assert {k: out.get(k) for k in EXPECTED} == EXPECTED, f"{route} dropped the contract: {out}"
    assert out["path"] == str(denied_chrome) and out["browser"] == "chrome"
    assert out["error"].startswith("The app isn't allowed to read Google Chrome's data"), out["error"]
    assert "profile locked" not in out["error"] and "SHOULD-NOT-RUN" not in raw
