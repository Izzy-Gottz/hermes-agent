"""Tests for browser fidelity (browser.stealth_fidelity): the real-profile browser tells sites the
truth, that it is the person's Chrome on a Mac, instead of "HeadlessChrome" with
navigator.webdriver set.

Values quoted here were measured on 2026-09-23, not invented: macOS 27, Chrome for Testing
153.0.8010.52, the owner's Google Chrome 153.0.8010.53, and a 1470x956 @2x display with a 33 pt
menu bar and a 72 pt Dock. The unbranded brand list is what the driven browser reported for
itself. The --screen-info read-backs are what the browser returned.
"""
import http.server
import json
import os
import plistlib
import socketserver
import threading
import time
from unittest.mock import patch

import pytest

from tools import browser_tool_fidelity as fid
from tools import browser_tool_real_profile as rp


CHROME_153 = {"brand": "Google Chrome", "ua_suffix": None, "full_version": "153.0.8010.53", "major": 153}
MACBOOK = {"width": 1470.0, "height": 956.0, "scale": 2.0, "top": 33.0, "bottom": 72.0}


def _fake_app(tmp_path, name="Google Chrome", version="153.0.8010.53"):
    app = tmp_path / f"{name}-{version}" / f"{name}.app" / "Contents"
    (app / "MacOS").mkdir(parents=True, exist_ok=True)
    (app / "MacOS" / name).write_text("")
    with open(app / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleShortVersionString": version}, fh)
    return str(app / "MacOS" / name)


class TestIdentity:
    def test_version_is_read_from_the_installed_apps_plist(self, tmp_path):
        binary = _fake_app(tmp_path)
        assert fid.installed_browser_version(binary) == "153.0.8010.53"
        assert fid.persons_browser_identity("chrome", binary) == CHROME_153

    def test_unreadable_or_odd_version_is_no_identity(self, tmp_path):
        assert fid.installed_browser_version("/usr/bin/google-chrome") is None  # not an .app
        assert fid.persons_browser_identity("chrome", _fake_app(tmp_path, version="beta")) is None
        assert fid.persons_browser_identity("firefox", _fake_app(tmp_path)) is None

    def test_edge_keeps_its_own_brand_and_ua_token(self, tmp_path):
        ident = fid.persons_browser_identity("edge", _fake_app(tmp_path, "Microsoft Edge", "153.0.3405.10"))
        assert ident["brand"] == "Microsoft Edge"
        assert fid.user_agent(ident).endswith("Safari/537.36 Edg/153.0.0.0")

    def test_chrome_updated_first_claims_the_engines_version(self, tmp_path):
        """Chrome auto-updates before Chrome for Testing: never claim 154 on a 153 engine."""
        ident = fid.persons_browser_identity("chrome", _fake_app(tmp_path, version="154.0.8100.2"),
                                             engine_version="153.0.8010.52")
        assert ident == {"brand": "Google Chrome", "ua_suffix": None, "full_version": "153.0.8010.52", "major": 153}
        assert "Chrome/153.0.0.0" in fid.user_agent(ident)

    def test_same_major_keeps_the_persons_version(self, tmp_path):
        assert fid.persons_browser_identity("chrome", _fake_app(tmp_path), engine_version="153.0.8010.52") == CHROME_153

    def test_engine_version_from_json_version(self):
        assert fid._engine_version("HeadlessChrome/153.0.8010.52") == "153.0.8010.52"
        assert fid._engine_version("Chrome/154.0.1.2") == "154.0.1.2"
        assert fid._engine_version("") is None


class TestWhatItSends:
    def test_unbranded_list_matches_what_chrome_for_testing_reports_for_itself(self):
        # Measured: navigator.userAgentData.brands on CfT 153 = "Chromium;153, Not_A Brand;8"
        assert fid.brand_list(None, 153, "153") == [
            {"brand": "Chromium", "version": "153"}, {"brand": "Not_A Brand", "version": "8"}]
        # Measured by the reviewer: CfT 154 natively = "Not A(Brand;99, Chromium;154"
        assert fid.brand_list(None, 154, "154") == [
            {"brand": "Not A(Brand", "version": "99"}, {"brand": "Chromium", "version": "154"}]
        # CfT 153 fullVersionList = "Chromium;153.0.8010.52, Not_A Brand;8.0.0.0"
        assert fid.brand_list(None, 153, "153.0.8010.52") == [
            {"brand": "Chromium", "version": "153.0.8010.52"}, {"brand": "Not_A Brand", "version": "8.0.0.0"}]

    def test_chrome_153_brands(self):
        assert [b["brand"] for b in fid.brand_list("Google Chrome", 153, "153")] == [
            "Google Chrome", "Not_A Brand", "Chromium"]

    def test_user_agent_is_the_reduced_mac_chrome_string(self):
        ua = fid.user_agent(CHROME_153)
        assert ua == ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36")

    def test_metadata_is_this_mac(self):
        with patch.object(fid.platform, "machine", return_value="arm64"), \
             patch.object(fid.platform, "mac_ver", return_value=("27.0", ("", "", ""), "arm64")):
            md = fid.user_agent_metadata(CHROME_153)
        assert md["platform"] == "macOS" and md["platformVersion"] == "27.0.0"
        assert (md["architecture"], md["bitness"], md["mobile"]) == ("arm", "64", False)
        assert md["fullVersion"] == "153.0.8010.53" and md["formFactors"] == ["Desktop"]
        assert {"brand": "Google Chrome", "version": "153.0.8010.53"} in md["fullVersionList"]

    def test_launch_flags(self):
        flags = fid.launch_flags(CHROME_153, headless=True, display=MACBOOK)
        assert "--disable-blink-features=AutomationControlled" in flags
        assert f"--user-agent={fid.user_agent(CHROME_153)}" in flags
        # Chrome's own default window on that display (measured headed: 1440x853)...
        assert "--window-size=1440,853" in flags
        # ...and --screen-info in device pixels: this read back as 1470x956, availTop 33,
        # availHeight 851, colorDepth 30.
        assert "--screen-info={2940x1912 devicePixelRatio=2 workAreaTop=66 workAreaBottom=144 colorDepth=30}" in flags

    def test_no_identity_claims_no_user_agent(self):
        assert not [f for f in fid.launch_flags(None, True, MACBOOK) if f.startswith("--user-agent")]

    def test_headed_or_unknown_display_gets_no_synthetic_screen(self):
        for flags in (fid.launch_flags(None, False, MACBOOK), fid.launch_flags(None, True, None)):
            assert flags == ["--disable-blink-features=AutomationControlled"]

    @pytest.mark.skipif(not os.path.exists("/System/Library/Frameworks/AppKit.framework"), reason="macOS only")
    def test_display_probe_on_this_mac(self):
        d = fid.main_display()
        assert d and d["width"] > 0 and d["scale"] >= 1 and d["top"] >= 0 and d["bottom"] >= 0


class TestConfig:
    def test_default_on(self):
        with patch("tools.browser_tool._browser_cfg", side_effect=lambda key, default, parse, label: default):
            assert fid.fidelity_enabled() is True

    @pytest.mark.parametrize("raw", [False, "false", "off", 0, "no"])
    def test_off(self, raw):
        with patch("tools.browser_tool._browser_cfg", side_effect=lambda key, default, parse, label: parse(raw)):
            assert fid.fidelity_enabled() is False

    def test_reads_the_real_config_file(self):
        from hermes_cli.config import get_hermes_home
        home = get_hermes_home()
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("browser:\n  stealth_fidelity: false\n")
        assert fid.fidelity_enabled() is False
        (home / "config.yaml").write_text("browser:\n  use_real_profile: true\n")
        assert fid.fidelity_enabled() is True


class _FakeWs:
    def __init__(self, events, block=False):
        self.sent, self._events, self.closed = [], list(events), False
        self._block = block
        self._closed_evt = threading.Event()

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self, timeout=None):
        if self._events:
            return json.dumps(self._events.pop(0))
        if self._block:  # a wedged read: never times out, only a close() ends it
            self._closed_evt.wait(30)
            raise ConnectionError("closed")
        time.sleep(0.02)
        raise TimeoutError

    def close(self):
        self.closed = True
        self._closed_evt.set()


def _attached(session, kind, url="https://example.com/", opener=None):
    info = {"targetId": session.upper(), "type": kind, "url": url}
    if opener:
        info["openerId"] = opener
    return {"method": "Target.attachedToTarget",
            "params": {"sessionId": session, "targetInfo": info, "waitingForDebugger": True}}


@pytest.fixture(autouse=True)
def _no_keepers_left():
    """Every lane's keepers, before and after: a test elsewhere that really launches a signed-out
    browser leaves its keeper to the session's close, which that test never sends."""
    fid.stop_keepers(lane=None)
    yield
    fid.stop_keepers(lane=None)


@pytest.fixture(autouse=True, scope="module")
def _no_files_left():
    """Nothing this module creates may stay in $TMPDIR: lock files of gone browsers and status
    files of gone keepers are swept once the module is done (``now`` is pushed past the 60 s
    grace a live gateway's files get)."""
    yield
    fid.sweep_lock_files(now=time.time() + 120)


class TestKeeper:
    UA = fid.user_agent(CHROME_153)

    def _run(self, events):
        ws = _FakeWs(events)
        keeper = fid.FidelityKeeper(9, CHROME_153, connect=lambda port: ws)
        keeper.start()
        deadline = time.time() + 3
        while ws._events and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(0.05)
        keeper.stop()
        keeper.join(2)
        return ws, keeper

    def test_browser_level_auto_attach_pauses_every_new_target(self):
        ws, _ = self._run([])
        first = ws.sent[0]
        assert first["method"] == "Target.setAutoAttach" and "sessionId" not in first
        assert first["params"] == {"autoAttach": True, "waitForDebuggerOnStart": True, "flatten": True}

    def test_new_tab_popup_frame_and_workers_each_get_the_ua_and_brands_before_they_run(self):
        ws, keeper = self._run([
            _attached("tab1", "page"),                                        # a new tab
            _attached("pop1", "page", "https://example.org/", opener="TAB1"),  # a popup it opened
            _attached("frm1", "iframe"),                                      # an out-of-process frame
            _attached("wrk1", "worker", "blob:https://example.com/x"),        # a dedicated worker
            _attached("shw1", "shared_worker", "https://example.com/s.js"),
            _attached("sw1", "service_worker", "https://example.com/sw.js"),
        ])
        by_session = {}
        for msg in ws.sent[1:]:
            by_session.setdefault(msg.get("sessionId"), []).append(msg)
        for s in ("tab1", "pop1", "frm1"):
            assert [m["method"] for m in by_session[s]] == [
                "Emulation.setUserAgentOverride", "Page.addScriptToEvaluateOnNewDocument", "Target.setAutoAttach",
                "Runtime.runIfWaitingForDebugger"], s
        assert [m["method"] for m in by_session["wrk1"]] == ["Network.setUserAgentOverride", "Runtime.runIfWaitingForDebugger"]
        for s in ("shw1", "sw1"):
            assert [m["method"] for m in by_session[s]] == [
                "Network.setUserAgentOverride", "Target.setAutoAttach", "Runtime.runIfWaitingForDebugger"], s
        for s in ("tab1", "pop1", "frm1", "wrk1", "shw1", "sw1"):
            override = by_session[s][0]["params"]
            assert override["userAgent"] == self.UA, s  # the ONLY source of the UA string now
            assert [b["brand"] for b in override["userAgentMetadata"]["brands"]] == [
                "Google Chrome", "Not_A Brand", "Chromium"], s
            assert override["userAgentMetadata"]["architecture"] and override["userAgentMetadata"]["platformVersion"]
        assert keeper.applied == {"page": 2, "iframe": 1, "worker": 1, "shared_worker": 1, "service_worker": 1}

    def test_chrome_ui_targets_are_only_resumed(self):
        ws, _ = self._run([_attached("ui1", "browser_ui", "chrome://omnibox-popup.top-chrome/")])
        assert [m["method"] for m in ws.sent if m.get("sessionId") == "ui1"] == ["Runtime.runIfWaitingForDebugger"]

    def test_a_dead_browser_ends_the_keeper_as_failed(self):
        def refuse(port):
            raise ConnectionRefusedError("gone")
        keeper = fid.FidelityKeeper(9, CHROME_153, connect=refuse)
        keeper.start()
        keeper.join(2)
        assert not keeper.is_alive() and keeper.state == "failed" and "gone" in keeper.error



class _FakeProc:
    """Stands in for a KeeperProcess: the surface ensure_keeper and the monitor use."""
    made = []

    def __init__(self, port, identity, state="serving", age=0.0, lane=fid.REAL_PROFILE_LANE):
        self.port, self.identity, self._state, self.age, self.lane = port, identity, state, age, lane
        self.alive, self.killed, self.stopped = True, False, False
        self.ua = fid.user_agent(identity)
        _FakeProc.made.append(self)

    state = property(lambda self: self._state)
    error = None

    def heartbeat_age(self):
        return self.age

    def is_alive(self):
        return self.alive

    def healthy(self):
        return self.alive and self._state == "serving" and self.age <= fid.STALL_SECONDS

    def wait_ready(self, wait):
        pass

    def stop(self):
        self.stopped, self.alive = True, False

    def kill(self):
        self.killed, self.alive = True, False

    def join(self, timeout=None):
        pass

    def describe(self):
        return {"port": self.port, "state": self._state, "applied": {}, "error": None}


class TestMonitor:
    def test_a_silent_keeper_is_replaced_at_once(self):
        """Stuck for longer than STALL_SECONDS: killed (Chrome releases its pauses) and replaced
        straight away, not at the next acquire, so running tabs get their brands back."""
        _FakeProc.made = []
        with patch.object(fid, "KeeperProcess", _FakeProc), patch.object(fid, "_ensure_monitor"):
            stuck = fid._start_keeper(4700, CHROME_153)
            stuck.age = fid.STALL_SECONDS + 1
            fid._monitor_once()
        assert stuck.killed and len(_FakeProc.made) == 2 and fid._keepers[4700] is _FakeProc.made[1]

    def test_a_short_hitch_is_left_alone(self):
        """A 4 s hitch only delays new tabs, and they come out right. Killing would strip the
        brands from every running tab (measured by the reviewer), so the bound is 10 s."""
        assert fid.STALL_SECONDS >= 10
        _FakeProc.made = []
        with patch.object(fid, "KeeperProcess", _FakeProc), patch.object(fid, "_ensure_monitor"):
            k = fid._start_keeper(4800, CHROME_153)
            k.age = 4.0
            fid._monitor_once()
        assert not k.killed and len(_FakeProc.made) == 1


class TestOneKeeperPerBrowser:
    def test_the_lock_admits_one_holder(self, tmp_path):
        lock = str(tmp_path / "k.lock")
        a, b = fid.FidelityKeeper(9, CHROME_153), fid.FidelityKeeper(9, CHROME_153)
        assert a._claim(lock) is True
        assert b._claim(lock) is False  # a second pausing client could only add a way to wedge
        a._release()
        assert b._claim(lock) is True  # the holder went away: the next acquire takes over
        b._release()

    def test_lock_is_per_browser_instance(self):
        assert fid._lock_path(9222, "ws://127.0.0.1:9222/devtools/browser/abc-1") != \
               fid._lock_path(9222, "ws://127.0.0.1:9222/devtools/browser/def-2")

    def test_holder_heartbeat(self, tmp_path):
        lock = tmp_path / "k.lock"
        lock.write_text("1")
        assert fid.holder_is_stale(str(lock)) is False
        old = time.time() - 60
        os.utime(lock, (old, old))
        assert fid.holder_is_stale(str(lock)) is True

    def test_a_stale_keeper_holder_is_taken_over(self, tmp_path):
        """A holder that stopped heartbeating (SIGSTOPped) is killed and the lock taken, but only
        when its command line shows it is a keeper process."""
        import subprocess
        import sys
        lock = str(tmp_path / "k.lock")
        holder = subprocess.Popen([sys.executable, "-c", f"""
import fcntl, os, time
fd = os.open({lock!r}, os.O_RDWR | os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX)
os.write(fd, str(os.getpid()).encode()); print("held", flush=True); time.sleep(60)
""", fid._KEEPER_MARK, "keeper", "9"], stdout=subprocess.PIPE, text=True)  # argv of a keeper for port 9
        try:
            assert holder.stdout.readline().strip() == "held"
            b = fid.FidelityKeeper(9, CHROME_153)
            assert b._claim(lock) is False  # heartbeat fresh: a live keeper, leave it
            old = time.time() - 60
            os.utime(lock, (old, old))
            assert b._claim(lock) is True  # stale: taken over
            assert holder.wait(5) is not None
            b._release()
        finally:
            holder.kill()
            holder.wait(5)

    def test_a_keeper_for_another_port_is_never_killed(self, tmp_path):
        """Same pid in the lock file, a real keeper command line, but for port 9999: not ours."""
        import subprocess
        import sys
        lock = str(tmp_path / "k.lock")
        holder = subprocess.Popen([sys.executable, "-c", f"""
import fcntl, os, time
fd = os.open({lock!r}, os.O_RDWR | os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX)
os.write(fd, str(os.getpid()).encode()); print("held", flush=True); time.sleep(60)
""", fid._KEEPER_MARK, "keeper", "9999"], stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "held"
            old = time.time() - 60
            os.utime(lock, (old, old))
            assert fid.FidelityKeeper(9, CHROME_153)._claim(lock) is False
            assert holder.poll() is None
        finally:
            holder.kill()
            holder.wait(5)

    def test_a_keeper_for_this_port_that_does_not_hold_the_lock_is_never_killed(self, tmp_path):
        import subprocess
        import sys
        lock = tmp_path / "k.lock"
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", fid._KEEPER_MARK, "keeper", "9"])
        try:
            lock.write_text(str(other.pid))  # a pid file that points at it, but no lock and no open file
            old = time.time() - 60
            os.utime(lock, (old, old))
            import fcntl
            fd = os.open(str(lock), os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX)  # someone else (this test) holds it
            try:
                assert fid.FidelityKeeper(9, CHROME_153)._claim(str(lock)) is False
            finally:
                os.close(fd)
            assert other.poll() is None
        finally:
            other.kill()
            other.wait(5)

    def test_sweep_removes_only_unheld_locks_of_gone_browsers(self, tmp_path):
        import fcntl
        gone = tmp_path / "hermes-browser-fidelity-1-deadbeef.lock"
        held = tmp_path / "hermes-browser-fidelity-2-cafe.lock"
        gone.write_text("1")
        held.write_text("2")
        fd = os.open(str(held), os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        old_status = tmp_path / "hermes-fidelity-1-abc.json"
        old_status.write_text(json.dumps({"pid": 999999}))  # a keeper that no longer exists
        live_status = tmp_path / "hermes-fidelity-2-def.json"
        live_status.write_text(json.dumps({"pid": os.getpid()}))
        for f in (old_status, live_status):
            os.utime(f, (time.time() - 120, time.time() - 120))
        try:
            with patch("tempfile.gettempdir", return_value=str(tmp_path)):
                assert fid.sweep_lock_files() == 2
            assert not gone.exists() and held.exists()
            assert not old_status.exists() and live_status.exists()
        finally:
            os.close(fd)

    def test_a_stale_holder_that_is_not_a_keeper_is_never_killed(self, tmp_path):
        import subprocess
        import sys
        lock = str(tmp_path / "k.lock")
        holder = subprocess.Popen([sys.executable, "-c", f"""
import fcntl, os, time
fd = os.open({lock!r}, os.O_RDWR | os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX)
os.write(fd, str(os.getpid()).encode()); print("held", flush=True); time.sleep(60)
"""], stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "held"
            old = time.time() - 60
            os.utime(lock, (old, old))
            assert fid.FidelityKeeper(9, CHROME_153)._claim(lock) is False
            assert holder.poll() is None
        finally:
            holder.kill()
            holder.wait(5)

    def test_a_follower_adds_no_pause_and_is_retried(self):
        _FakeProc.made = []
        with patch.object(fid, "KeeperProcess", lambda port, ident, **kw: _FakeProc(port, ident, state="follower", **kw)), \
             patch.object(fid, "_ensure_monitor"), patch.object(fid, "fidelity_enabled", return_value=True):
            first = fid.ensure_keeper(4400, CHROME_153)
            assert first.state == "follower"
            second = fid.ensure_keeper(4400, CHROME_153)
        assert second is not first  # retried on the next acquire


class TestEnsureKeeper:
    def test_restarts_a_dead_keeper_and_resolves_identity_lazily(self):
        calls = []

        def identity():
            calls.append(1)
            return CHROME_153

        _FakeProc.made = []
        with patch.object(fid, "KeeperProcess", _FakeProc), patch.object(fid, "_ensure_monitor"), \
             patch.object(fid, "fidelity_enabled", return_value=True):
            a = fid.ensure_keeper(4500, identity)
            assert a.healthy() and calls == [1]
            assert fid.ensure_keeper(4500, identity) is a and calls == [1]  # healthy: identity not even read
            a.alive = False  # it died
            b = fid.ensure_keeper(4500, identity)
            assert b is not a and b.healthy() and calls == [1, 1]

    def test_off_or_unknown_identity_starts_nothing(self):
        with patch.object(fid, "KeeperProcess") as K, patch.object(fid, "fidelity_enabled", return_value=False):
            assert fid.ensure_keeper(4600, CHROME_153) is None
        with patch.object(fid, "KeeperProcess") as K2, patch.object(fid, "fidelity_enabled", return_value=True):
            assert fid.ensure_keeper(4600, None) is None
            assert fid.ensure_keeper(4600, lambda: None) is None
        K.assert_not_called()
        K2.assert_not_called()


class TestStatus:
    def _status(self, keepers):
        path = fid._status_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"pid": os.getpid(), "updated": time.time(), "keepers": keepers}, fh)

    def test_serving(self):
        self._status([{"state": "serving", "claims": "Google Chrome 153.0.8010.53", "applied": {"page": 3}}])
        with patch.object(fid, "fidelity_enabled", return_value=True):
            ok, line = fid.status_summary()
        assert ok and "serving" in line and "Google Chrome 153" in line

    def test_failed_is_a_warning_that_says_what_the_browser_reports(self):
        self._status([{"state": "failed", "error": "ConnectionRefusedError: gone"}])
        with patch.object(fid, "fidelity_enabled", return_value=True):
            ok, line = fid.status_summary()
        assert not ok and "NOT applied" in line and "gone" in line and "plain Chrome for Testing" in line

    def test_stale_holder_is_a_warning(self):
        self._status([{"state": "follower", "holder_stale": True}])
        with patch.object(fid, "fidelity_enabled", return_value=True):
            ok, line = fid.status_summary()
        assert not ok and "stopped responding" in line

    def test_doctor_reports_it(self):
        from hermes_cli import doctor_tools
        self._status([{"state": "failed", "error": "boom"}])
        with patch.object(fid, "fidelity_enabled", return_value=True), \
             patch.object(doctor_tools, "check_warn") as warn:
            doctor_tools._check_browser_fidelity()
        assert "NOT applied" in warn.call_args[0][0]


class TestDrivenBrowserLaunch:
    """The real-profile launcher passes the fidelity switches and keeps the keeper alive."""

    def _launch(self, tmp_path, enabled, headless=True):
        captured = {}

        def fake_spawn(binary, copy_dir, extra_flags, what, headless):
            captured["flags"], captured["headless"] = list(extra_flags), headless
            return object(), 41000, None

        with patch.object(rp, "_spawn_browser_on_copy", side_effect=fake_spawn), \
             patch.object(rp, "_driven_browser_headless", return_value=headless), \
             patch.object(fid, "fidelity_enabled", return_value=enabled), \
             patch.object(fid, "main_display", return_value=MACBOOK), \
             patch.object(fid, "ensure_keeper") as keeper:
            port, err = rp._launch_driven_browser("/cft", str(tmp_path), CHROME_153)
        return captured, keeper, port, err

    def test_fidelity_on(self, tmp_path):
        captured, keeper, port, err = self._launch(tmp_path, enabled=True)
        assert (port, err) == (41000, None)
        flags = captured["flags"]
        assert "--disable-blink-features=AutomationControlled" in flags
        assert f"--user-agent={fid.user_agent(CHROME_153)}" in flags
        assert any(f.startswith("--screen-info={2940x1912 ") for f in flags)
        assert captured["headless"] is True  # never a window: see the module docstring for why
        keeper.assert_called_once_with(41000, CHROME_153)

    def test_fidelity_off_is_the_old_flag_set(self, tmp_path):
        captured, _, _, _ = self._launch(tmp_path, enabled=False)
        assert captured["flags"] == list(rp._mock_keychain_flags())

    def test_terminate_stops_the_keeper(self):
        with patch.object(fid, "stop_keepers") as stop:
            rp._terminate_real_profile_chrome()
        stop.assert_called_once()

    def test_a_cache_hit_still_checks_the_keeper(self):
        import tools.browser_tool as bt
        from tools import browser_tool_cloud as bt_cloud
        bt._real_profile_cdp_cache["cdp"] = "http://127.0.0.1:41000"
        try:
            with patch.object(bt_cloud, "_use_real_profile", return_value=True), \
                 patch.object(rp._lp, "_using_lightpanda_engine", return_value=False), \
                 patch.object(rp, "_cdp_http_ready", return_value=True), \
                 patch.object(rp._session, "_prepare_session_socket_dir"), \
                 patch.object(fid, "ensure_keeper") as keeper:
                assert rp._real_profile_cdp() == ("http://127.0.0.1:41000", None)
            assert keeper.call_args[0][0] == 41000 and callable(keeper.call_args[0][1])
        finally:
            bt._real_profile_cdp_cache.clear()

    def test_every_tab_hermes_opens_starts_blank(self):
        """The pause holds the renderer, not a navigation already under way, so a tab created at a
        URL sends its first request before the brands are set. The attach opens about:blank."""
        import inspect
        src = inspect.getsource(rp._attach_agent_browser_to_real_profile)
        assert '"open", "about:blank"' in src


# ---------------------------------------------------------------------------
# Live: a real throwaway Chrome for Testing and a local server that records what it is sent
# ---------------------------------------------------------------------------

_HINTS = ("Sec-CH-UA, Sec-CH-UA-Full-Version-List, Sec-CH-UA-Arch, Sec-CH-UA-Bitness, "
          "Sec-CH-UA-Platform-Version, Sec-CH-UA-Model, Sec-CH-UA-Mobile, Sec-CH-UA-Platform")
_REPORT = ("async function report(where){const o={where,ua:navigator.userAgent,"
           "brands:navigator.userAgentData.brands.map(b=>b.brand).join(',')};"
           "try{const h=await navigator.userAgentData.getHighEntropyValues(['architecture','platformVersion']);"
           "o.arch=h.architecture;o.pv=h.platformVersion}catch(e){o.err=String(e)}"
           "await fetch('/report',{method:'POST',body:JSON.stringify(o)})}")
_PAGE = ("<html><head><script>" + _REPORT + "</script></head><body><script>report('page');"
         "new Worker('/dw.js');new SharedWorker('/shw.js');navigator.serviceWorker.register('/sw.js');"
         "</script></body></html>")


class _Recorder(http.server.BaseHTTPRequestHandler):
    log = []

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-CH", _HINTS)
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        self.log.append(("GET", self.path, {k.lower(): v for k, v in self.headers.items()}))
        path = self.path.split("?")[0]
        if path in ("/dw.js", "/shw.js", "/sw.js"):
            self._send(_REPORT + f"\nreport('{path[1:-3]}');", "text/javascript")
        elif path == "/plain":
            self._send("<html><body>plain</body></html>")
        else:
            self._send(_PAGE)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.log.append(("POST", self.path, json.loads(self.rfile.read(n) or b"{}")))
        self._send("ok", "text/plain")


# The live tests claim what a real acquire claims: the person's brand at the ENGINE's version
# (with_engine_version), so they hold on whichever Chrome for Testing is installed. Set by the
# live_browser fixture; the expected brand strings are derived from it, not written down.
_LIVE_ID = dict(CHROME_153)


def _live_brands(brand="Google Chrome"):
    return ",".join(b["brand"] for b in fid.brand_list(brand, _LIVE_ID["major"], str(_LIVE_ID["major"])))


@pytest.fixture
def live_browser(tmp_path):
    """(cdp_call, keeper, server_port, recorder_log). The whole browser process tree is killed after."""
    import subprocess
    import psutil
    import requests
    from websockets.sync.client import connect
    binary = rp.driven_browser_executable()
    if not binary:
        pytest.skip("no packaged Chrome for Testing")
    global _LIVE_ID
    _LIVE_ID = fid.with_engine_version(CHROME_153, fid.installed_browser_version(binary))
    _Recorder.log = []

    class _TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = _TS(("127.0.0.1", 0), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    udd = tmp_path / "udd"
    udd.mkdir()
    flags = [*rp._REAL_PROFILE_CHROME_FLAGS, *rp._mock_keychain_flags(), *fid.launch_flags(_LIVE_ID, True, MACBOOK), "--headless=new"]
    proc = subprocess.Popen([binary, f"--user-data-dir={udd}", *flags], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    ws = None
    port = None
    try:
        for _ in range(120):
            try:
                port = int((udd / "DevToolsActivePort").read_text().splitlines()[0])
                break
            except (OSError, ValueError, IndexError):
                time.sleep(0.25)
        assert port, "browser never exposed a debug port"
        with patch.object(fid, "fidelity_enabled", return_value=True):
            keeper = fid.ensure_keeper(port, _LIVE_ID)
        assert keeper is not None and keeper.state == "serving", keeper and keeper.error
        ws = connect(requests.get(f"http://127.0.0.1:{port}/json/version", timeout=5).json()["webSocketDebuggerUrl"],
                     max_size=None)
        ids = iter(range(1, 100_000))

        def call(method, params=None, session=None, timeout=30):
            i = next(ids)
            msg = {"id": i, "method": method, "params": params or {}}
            if session:
                msg["sessionId"] = session
            ws.send(json.dumps(msg))
            deadline = time.time() + timeout
            while True:
                m = json.loads(ws.recv(timeout=max(0.1, deadline - time.time())))
                if m.get("id") == i:
                    assert "error" not in m, m
                    return m.get("result", {})

        call("Target.createTarget", {"url": "about:blank"})  # a tab to keep the browser up
        yield call, keeper, server.server_address[1], _Recorder.log
    finally:
        fid.stop_keepers()
        if ws is not None:
            ws.close()
        server.shutdown()
        # The whole tree: renderer, GPU and crashpad helpers must not be orphaned under launchd.
        try:
            tree = [psutil.Process(proc.pid), *psutil.Process(proc.pid).children(recursive=True)]
        except psutil.NoSuchProcess:
            tree = []
        for p in tree:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(tree, timeout=10)
        for p in alive:
            p.kill()
        try:
            proc.wait(5)  # reap it, so it is not left a zombie
        except subprocess.TimeoutExpired:
            pass
        for p in psutil.process_iter(["cmdline"]):  # helpers re-parented before we looked
            try:
                if any(str(udd) in a for a in (p.info["cmdline"] or [])):
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        # The browser is gone, so its lock file must go too: nothing may pile up in $TMPDIR.
        import glob
        import tempfile
        fid.sweep_lock_files()
        if port:
            assert not glob.glob(os.path.join(tempfile.gettempdir(), f"hermes-browser-fidelity-{port}-*.lock"))


def _wait_for(pred, secs=15):
    deadline = time.time() + secs
    while time.time() < deadline:
        got = pred()
        if got:
            return got
        time.sleep(0.2)
    return None


def _report(log, where):
    return next((r[2] for r in log if r[0] == "POST" and r[2].get("where") == where), None)


LIVE = pytest.mark.skipif(os.environ.get("HERMES_BROWSER_LIVE_TESTS") != "1",
                          reason="launches a real throwaway Chrome for Testing; set HERMES_BROWSER_LIVE_TESTS=1")


def _blank_then_navigate(call, url):
    tab = call("Target.createTarget", {"url": "about:blank"})["targetId"]
    s = call("Target.attachToTarget", {"targetId": tab, "flatten": True})["sessionId"]
    call("Page.navigate", {"url": url}, s)
    return s


@LIVE
@pytest.mark.live_system_guard_bypass
class TestLiveBrowser:
    def test_first_request_of_a_blank_then_navigate_tab_and_every_context(self, live_browser):
        """The way Hermes opens tabs. The FIRST request's headers, the page, a dedicated worker, a
        shared worker and a service worker must all say Google Chrome. The shared and service
        workers take their brands from Network.setUserAgentOverride and their navigator.userAgent
        ONLY from the --user-agent switch (measured: no override reaches it)."""
        call, keeper, sp, log = live_browser
        _blank_then_navigate(call, f"http://localhost:{sp}/page?first")
        assert _wait_for(lambda: all(_report(log, w) for w in ("page", "dw", "shw", "sw"))), f"missing reports: {log}"
        first = next(h for verb, path, h in log if verb == "GET" and path == "/page?first")
        assert first["user-agent"] == keeper.ua
        major = _LIVE_ID["major"]
        assert f'"Google Chrome";v="{major}"' in first["sec-ch-ua"] and first["sec-ch-ua-platform"] == '"macOS"'
        for where in ("page", "dw", "shw", "sw"):
            r = _report(log, where)
            assert r["ua"] == keeper.ua, where
            assert r["brands"] == _live_brands(), where
            assert r.get("arch") and r.get("pv"), where  # high-entropy hints are present, not empty

    def test_a_popup_is_covered(self, live_browser):
        call, keeper, sp, log = live_browser
        s = _blank_then_navigate(call, f"http://localhost:{sp}/plain")
        _wait_for(lambda: any(p == "/plain" for _, p, _ in log))
        call("Runtime.evaluate", {"expression": f"window.open('http://localhost:{sp}/page?popup'); 1",
                                  "userGesture": True}, s)
        got = _wait_for(lambda: _report(log, "page"))
        assert got and got["brands"] == _live_brands() and got["ua"] == keeper.ua
        assert _wait_for(lambda: keeper.applied.get("page", 0) >= 3, secs=5)  # counters reach status each second

    @pytest.mark.xfail(strict=True, reason="KNOWN LIMIT: waitForDebuggerOnStart holds the renderer, not the "
                       "browser-side navigation, so a tab created straight at a URL sends its first request "
                       "with Chromium brands. Hermes never opens a tab that way (blank, then navigate).")
    def test_first_request_of_a_tab_created_straight_at_a_url(self, live_browser):
        call, keeper, sp, log = live_browser
        call("Target.createTarget", {"url": f"http://localhost:{sp}/page?direct"})
        first = _wait_for(lambda: next((h for verb, path, h in log if verb == "GET" and path == "/page?direct"), None))
        assert first and '"Google Chrome"' in first.get("sec-ch-ua", "")

    def test_a_dead_keeper_is_reported_and_the_next_acquire_restores_it(self, live_browser):
        """Kill the keeper. While none serves, a new tab is half-spoofed (the switch's UA over
        Chromium brands, empty hints): that is the cost of the switch the workers need. So the gap
        must be visible in status, and the next acquire must close it."""
        call, keeper, sp, log = live_browser
        port = keeper.port
        os.kill(keeper.proc.pid, 9)  # a crash, not a stop
        keeper.proc.wait(5)
        assert keeper.state == "failed"
        fid.write_status()
        with patch.object(fid, "fidelity_enabled", return_value=True):
            ok, line = fid.status_summary()
        assert not ok and "NOT applied" in line
        _blank_then_navigate(call, f"http://localhost:{sp}/page?dead")
        got = _wait_for(lambda: _report(log, "page"))
        assert got["brands"] == _live_brands(None)  # measured degraded state, documented
        with patch.object(fid, "fidelity_enabled", return_value=True):
            again = fid.ensure_keeper(port, _LIVE_ID)
        assert again is not keeper and again.state == "serving"
        log.clear()
        _blank_then_navigate(call, f"http://localhost:{sp}/page?back")
        got = _wait_for(lambda: _report(log, "page"))
        assert got["brands"] == _live_brands()

    def test_a_stopped_second_process_cannot_wedge_new_tabs(self, live_browser):
        """The reviewer's case: another Hermes process attaches to the same browser, and is then
        SIGSTOPped. With two pausing keepers, new tabs hung until timeout. Now the second process
        finds the browser's lock held and adds no pause of its own, so stopping it changes nothing."""
        import signal
        import subprocess
        import sys
        call, keeper, sp, log = live_browser
        fork_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        child = subprocess.Popen([sys.executable, "-c", f"""
import sys, time; sys.path.insert(0, {fork_root!r})
from unittest.mock import patch
from tools import browser_tool_fidelity as fid
with patch.object(fid, "fidelity_enabled", return_value=True):
    k = fid.ensure_keeper({keeper.port}, {_LIVE_ID!r})
print(k.state, flush=True)
time.sleep(120)
"""], stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == "follower"
            os.kill(child.pid, signal.SIGSTOP)
            t0 = time.time()
            _blank_then_navigate(call, f"http://localhost:{sp}/page?stopped")
            got = _wait_for(lambda: _report(log, "page"), secs=10)
            assert got and got["brands"] == _live_brands()
            assert time.time() - t0 < 8
        finally:
            os.kill(child.pid, signal.SIGCONT)
            child.kill()
            child.wait(5)

    def test_a_gateway_stall_changes_nothing(self, live_browser):
        """The reviewer's case: the gateway holds the GIL for 4 s. With the keeper in a thread, its
        watchdog closed the connection and every open tab fell to Chromium brands. With the
        keeper in its own process, open tabs keep the brands, and a tab another process opens
        during the stall gets them without waiting for the stall to end."""
        import subprocess
        import sys
        call, keeper, sp, log = live_browser
        _blank_then_navigate(call, f"http://localhost:{sp}/page?t1")
        assert _wait_for(lambda: _report(log, "page"))["brands"] == _live_brands()
        # A peer process opens a tab during the stall, against its own server (ours is stalled too).
        child = subprocess.Popen([sys.executable, "-c", _PEER_TAB % {"port": keeper.port}],
                                 stdout=subprocess.PIPE, text=True)
        t0 = time.monotonic()
        per = time.monotonic(); sum(range(10 ** 8)); per = time.monotonic() - per
        sum(range(int(10 ** 8 * 4.5 / max(per, 1e-3))))  # one C call: holds the GIL for about 4.5 s
        stall = time.monotonic() - t0
        peer = json.loads(child.communicate(timeout=60)[0].strip().splitlines()[-1])
        assert stall >= 3.5, stall
        assert peer["brands"] == _live_brands(), peer
        assert peer["secs"] < stall, peer  # it did not wait for the gateway
        log.clear()
        s = _blank_then_navigate(call, f"http://localhost:{sp}/plain")
        brands = call("Runtime.evaluate", {"expression": "1", "returnByValue": True}, s)
        tabs = [t for t in call("Target.getTargets")["targetInfos"] if t["type"] == "page" and "t1" in t["url"]]
        s1 = call("Target.attachToTarget", {"targetId": tabs[0]["targetId"], "flatten": True})["sessionId"]
        t = time.monotonic()
        brands = call("Runtime.evaluate", {"expression": "navigator.userAgentData.brands.map(b=>b.brand).join()",
                                           "returnByValue": True}, s1)["result"]["value"]
        assert brands == _live_brands()  # the open tab kept them
        assert time.monotonic() - t < 2  # and answers at once (no 20 s hang)
        assert keeper.healthy()

    def test_a_stuck_keeper_is_replaced_and_open_tabs_get_their_brands_back(self, live_browser):
        """SIGSTOP the keeper process past STALL_SECONDS: the monitor kills and replaces it without
        waiting for an acquire, and the replacement re-applies the brands to the open tab."""
        import signal
        call, keeper, sp, log = live_browser
        s1 = _blank_then_navigate(call, f"http://localhost:{sp}/page?t1")
        assert _wait_for(lambda: _report(log, "page"))
        os.kill(keeper.proc.pid, signal.SIGSTOP)
        replaced = _wait_for(lambda: fid._keepers.get(keeper.port) is not keeper and fid._keepers.get(keeper.port),
                             secs=fid.STALL_SECONDS + 10)
        assert replaced, "the monitor never replaced the stopped keeper"
        assert keeper.proc.wait(5) is not None  # SIGKILL reaches a stopped process
        replaced.wait_ready(15)
        assert replaced.state == "serving"
        time.sleep(0.5)
        brands = call("Runtime.evaluate", {"expression": "navigator.userAgentData.brands.map(b=>b.brand).join()",
                                           "returnByValue": True}, s1)["result"]["value"]
        assert brands == _live_brands()


_PEER_TAB = r'''
import http.server, json, socketserver, threading, time
import requests
from websockets.sync.client import connect
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
        self.wfile.write(b"<html><body>peer</body></html>")
class TS(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads = True
srv = TS(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
time.sleep(0.5)  # let the gateway's stall begin
t0 = time.monotonic()
ws = connect(requests.get("http://127.0.0.1:%(port)d/json/version", timeout=5).json()["webSocketDebuggerUrl"], max_size=None)
n = [0]
def call(method, params=None, session=None):
    n[0] += 1; msg = {"id": n[0], "method": method, "params": params or {}}
    if session: msg["sessionId"] = session
    ws.send(json.dumps(msg))
    while True:
        m = json.loads(ws.recv(timeout=30))
        if m.get("id") == n[0]: return m.get("result", {})
tab = call("Target.createTarget", {"url": "about:blank"})["targetId"]
s = call("Target.attachToTarget", {"targetId": tab, "flatten": True})["sessionId"]
call("Page.navigate", {"url": "http://localhost:%%d/" %% srv.server_address[1]}, s)
for _ in range(100):
    if call("Runtime.evaluate", {"expression": "document.readyState", "returnByValue": True}, s)["result"]["value"] == "complete": break
    time.sleep(0.05)
b = call("Runtime.evaluate", {"expression": "navigator.userAgentData.brands.map(b=>b.brand).join()", "returnByValue": True}, s)["result"]["value"]
print(json.dumps({"brands": b, "secs": time.monotonic() - t0}), flush=True)
'''


@LIVE
@pytest.mark.live_system_guard_bypass
def test_the_keeper_dies_with_the_gateway(live_browser):
    """The incident class of 2026-09-20: something of ours outliving its owner. A gateway (a child
    process here) starts a keeper and is SIGKILLed. The keeper must be gone within 2 s: it watches
    its stdin, which the kernel closes when the gateway dies."""
    import signal
    import subprocess
    import sys
    import psutil
    call, keeper, sp, log = live_browser
    port = keeper.port
    fid.stop_keepers()  # hand the browser to the child's keeper (only one may serve it)
    fork_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gateway = subprocess.Popen([sys.executable, "-c", f"""
import sys, time; sys.path.insert(0, {fork_root!r})
from unittest.mock import patch
from tools import browser_tool_fidelity as fid
with patch.object(fid, "fidelity_enabled", return_value=True):
    k = fid.ensure_keeper({port}, {_LIVE_ID!r})
print(k.state, k.proc.pid, flush=True)
time.sleep(120)
"""], stdout=subprocess.PIPE, text=True)
    try:
        state, pid = gateway.stdout.readline().split()
        assert state == "serving"
        keeper_proc = psutil.Process(int(pid))
        os.kill(gateway.pid, signal.SIGKILL)
        gateway.wait(5)
        t0 = time.monotonic()
        try:
            keeper_proc.wait(2)
        except psutil.TimeoutExpired:
            keeper_proc.kill()
            pytest.fail("the keeper outlived its gateway by more than 2 s")
        assert time.monotonic() - t0 < 2
    finally:
        if gateway.poll() is None:
            gateway.kill()
            gateway.wait(5)


class _BeatingFake(_FakeProc):
    def describe(self):
        return {**super().describe(), "beat": 1234.5, "heartbeat_age": self.age, "pending": 0,
                "resume_failures": 0, "resume_retries": 0, "unresumed": 0}


class TestLiveStatus:
    def test_the_status_is_rewritten_on_every_monitor_tick(self):
        """A healthy keeper's status must not freeze at start: each monitor tick rewrites it,
        carrying the keeper's heartbeat, so a reader can tell a live keeper from a dead one."""
        _BeatingFake.made = []
        with patch.object(fid, "KeeperProcess", _BeatingFake), patch.object(fid, "_ensure_monitor"):
            fid._start_keeper(4900, CHROME_153)
            path = fid._status_path()
            if os.path.exists(path):
                os.unlink(path)
            before = time.time()
            fid._monitor_once()  # a healthy keeper: nothing replaced, and still written
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        assert data["updated"] >= before
        (k,) = data["keepers"]
        assert k["beat"] == 1234.5 and k["port"] == 4900 and "resume_failures" in k

    def test_describe_carries_the_keepers_heartbeat_and_counters(self, tmp_path):
        """What the parent writes comes from the keeper process's per-second file."""
        kp = object.__new__(fid.KeeperProcess)
        kp.port, kp.identity, kp.ua, kp._stopped = 4901, CHROME_153, "ua", False
        kp.status_path = str(tmp_path / "hermes-fidelity-4901-x.json")

        class _Alive:
            pid, returncode = 4242, None

            def poll(self):
                return None
        kp.proc = _Alive()
        beat = time.time() - 1
        (tmp_path / "hermes-fidelity-4901-x.json").write_text(json.dumps(
            {"state": "serving", "applied": {"page": 3}, "beat": beat, "pending": 1,
             "resume_failures": 2, "resume_retries": 1, "unresumed": 1}))
        d = kp.describe()
        assert d["beat"] == beat and 0 < d["heartbeat_age"] < 5
        assert (d["applied"], d["pending"], d["resume_failures"], d["resume_retries"], d["unresumed"]) == \
            ({"page": 3}, 1, 2, 1, 1)


class TestResumeFailures:
    def _keeper(self):
        ws = _FakeWs([])
        keeper = fid.FidelityKeeper(9, CHROME_153, connect=lambda port: ws)
        keeper._ws = ws
        return ws, keeper

    @staticmethod
    def _resumes(ws, session):
        return [m for m in ws.sent if m["method"] == "Runtime.runIfWaitingForDebugger" and m.get("sessionId") == session]

    def test_a_failed_resume_warns_is_retried_once_and_is_counted(self, caplog):
        ws, keeper = self._keeper()
        keeper.handle(_attached("s1", "page"))
        (first,) = self._resumes(ws, "s1")
        with caplog.at_level("WARNING", logger=fid.logger.name):
            keeper.handle({"id": first["id"], "error": {"code": -32000, "message": "Target closed"}})
            assert any("resume" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)
            assert keeper.resume_failures == 1
            with patch.object(fid, "RESUME_SECONDS", 0.0):
                keeper._tick()  # past the bound: one more resume
                assert len(self._resumes(ws, "s1")) == 2 and keeper.resume_retries == 1
                keeper._tick()  # still no answer: given up on, counted, logged
                keeper._tick()
        assert len(self._resumes(ws, "s1")) == 2  # one retry, never more
        snap = keeper.snapshot()
        assert (snap["resume_failures"], snap["resume_retries"], snap["unresumed"], snap["pending"]) == (1, 1, 1, 0)
        assert any("still not resumed" in r.getMessage() for r in caplog.records)

    def test_an_acknowledged_resume_is_not_retried(self):
        ws, keeper = self._keeper()
        keeper.handle(_attached("s2", "page"))
        (first,) = self._resumes(ws, "s2")
        keeper.handle({"id": first["id"], "result": {}})
        with patch.object(fid, "RESUME_SECONDS", 0.0):
            keeper._tick()
        assert len(self._resumes(ws, "s2")) == 1 and keeper.snapshot()["pending"] == 0
        assert keeper.resume_retries == keeper.unresumed == 0


class TestStatusAcrossProcesses:
    """Two MCP servers (measured: two live on the owner's Mac) share one HERMES_HOME. Each writes
    its own file; the reader merges the live ones, so neither can erase the other's report."""

    @staticmethod
    def _write(pid, keepers):
        path = fid._status_path(pid)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"pid": pid, "updated": time.time(), "keepers": keepers}, fh)
        return path

    @pytest.fixture
    def other_process(self):
        import subprocess
        import sys
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        yield p.pid
        p.kill()
        p.wait(5)

    def test_a_process_without_keepers_does_not_erase_anothers_failure(self, other_process):
        theirs = self._write(other_process, [{"state": "failed", "error": "ConnectionRefusedError: gone"}])
        assert not fid._keepers
        fid.write_status()  # this process has no keepers: it writes nothing
        assert not os.path.exists(fid._status_path())
        with patch.object(fid, "fidelity_enabled", return_value=True):
            ok, line = fid.status_summary()
        assert os.path.exists(theirs)
        assert not ok and "NOT applied" in line and "gone" in line

    def test_two_processes_are_merged_and_a_failure_wins(self, other_process):
        self._write(other_process, [{"state": "failed", "error": "ConnectionRefusedError: gone"}])
        _BeatingFake.made = []
        with patch.object(fid, "KeeperProcess", _BeatingFake), patch.object(fid, "_ensure_monitor"):
            fid._start_keeper(5000, CHROME_153)
            for _ in range(3):  # tick after tick: the report does not flap
                fid._monitor_once()
                with patch.object(fid, "fidelity_enabled", return_value=True):
                    ok, line = fid.status_summary()
                assert not ok and "gone" in line
        mine = fid._status_path()
        with open(mine) as fh:
            assert json.load(fh)["keepers"][0]["port"] == 5000
        assert {d["pid"] for d in fid.read_statuses()} == {os.getpid(), other_process}

    def test_a_healthy_pair_reports_serving(self, other_process):
        self._write(other_process, [{"state": "follower", "holder_stale": False}])
        self._write(os.getpid(), [{"state": "serving", "claims": "Google Chrome 154.0.8037.57", "applied": {"page": 1}}])
        with patch.object(fid, "fidelity_enabled", return_value=True):
            ok, line = fid.status_summary()
        assert ok and "serving" in line and "154" in line

    def test_a_dead_processs_file_is_ignored_and_removed(self):
        import subprocess
        import sys
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait(10)
        dead = self._write(p.pid, [{"state": "failed", "error": "old news"}])
        legacy = os.path.join(os.path.dirname(dead), "browser-fidelity-status.json")
        with open(legacy, "w") as fh:
            json.dump({"pid": p.pid, "updated": 1, "keepers": [{"state": "failed"}]}, fh)
        with patch.object(fid, "fidelity_enabled", return_value=True):
            assert fid.status_summary() is None
        assert not os.path.exists(dead) and not os.path.exists(legacy)

    def test_stopping_the_keepers_removes_this_processs_file(self):
        with patch.object(fid, "KeeperProcess", _BeatingFake), patch.object(fid, "_ensure_monitor"):
            fid._start_keeper(5100, CHROME_153)
            fid.write_status()
            assert os.path.exists(fid._status_path())
            fid.stop_keepers()
        assert not os.path.exists(fid._status_path())
