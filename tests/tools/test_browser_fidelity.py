"""Tests for browser fidelity (browser.stealth_fidelity): the real-profile browser tells sites the
truth — the person's Chrome on a Mac — instead of "HeadlessChrome" with navigator.webdriver set.

Values quoted here were measured on 2026-09-23 (macOS 27, Chrome for Testing 153.0.8010.52, the
owner's Google Chrome 153.0.8010.53), not invented: the unbranded brand list is what the driven
browser reported for itself, and the headless defaults (800x600 screen) are what it reported
without these switches.
"""
import json
import os
import plistlib
import threading
import time
from unittest.mock import patch

import pytest

from tools import browser_tool_fidelity as fid
from tools import browser_tool_real_profile as rp


CHROME_153 = {"brand": "Google Chrome", "ua_suffix": None, "full_version": "153.0.8010.53", "major": 153}


def _fake_app(tmp_path, name="Google Chrome", version="153.0.8010.53"):
    app = tmp_path / f"{name}.app" / "Contents"
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


class TestWhatItSends:
    def test_unbranded_list_matches_what_chrome_for_testing_reports_for_itself(self):
        # Measured: navigator.userAgentData.brands on CfT 153 = "Chromium;153, Not_A Brand;8"
        assert fid.brand_list(None, 153, "153") == [
            {"brand": "Chromium", "version": "153"}, {"brand": "Not_A Brand", "version": "8"}]
        # ...and its fullVersionList = "Chromium;153.0.8010.52, Not_A Brand;8.0.0.0"
        assert fid.brand_list(None, 153, "153.0.8010.52") == [
            {"brand": "Chromium", "version": "153.0.8010.52"}, {"brand": "Not_A Brand", "version": "8.0.0.0"}]

    def test_chrome_153_brands(self):
        assert [b["brand"] for b in fid.brand_list("Google Chrome", 153, "153")] == [
            "Google Chrome", "Not_A Brand", "Chromium"]

    def test_user_agent_is_the_reduced_mac_chrome_string(self):
        ua = fid.user_agent(CHROME_153)
        assert ua == ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36")
        assert "Headless" not in ua

    def test_metadata_is_this_mac(self):
        with patch.object(fid.platform, "machine", return_value="arm64"), \
             patch.object(fid.platform, "mac_ver", return_value=("27.0", ("", "", ""), "arm64")):
            md = fid.user_agent_metadata(CHROME_153)
        assert md["platform"] == "macOS" and md["platformVersion"] == "27.0.0"
        assert (md["architecture"], md["bitness"], md["mobile"]) == ("arm", "64", False)
        assert md["fullVersion"] == "153.0.8010.53"
        assert {"brand": "Google Chrome", "version": "153.0.8010.53"} in md["fullVersionList"]

    def test_launch_flags(self):
        flags = fid.launch_flags(CHROME_153, headless=True, display=(1470, 956, 2.0))
        assert "--disable-blink-features=AutomationControlled" in flags
        assert f"--user-agent={fid.user_agent(CHROME_153)}" in flags
        # Chrome's own default window on that display (measured headed: 1440x853)...
        assert "--window-size=1440,853" in flags
        # ...and --screen-info takes device pixels: {1470x956 devicePixelRatio=2} read back as 735x478.
        assert "--screen-info={2940x1912 devicePixelRatio=2}" in flags

    def test_headed_gets_no_synthetic_screen(self):
        flags = fid.launch_flags(CHROME_153, headless=False)
        assert not [f for f in flags if f.startswith(("--window-size", "--screen-info"))]

    def test_no_identity_still_drops_webdriver_but_claims_no_ua(self):
        flags = fid.launch_flags(None, headless=True, display=(1470, 956, 2.0))
        assert "--disable-blink-features=AutomationControlled" in flags
        assert not [f for f in flags if f.startswith("--user-agent")]


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
    def __init__(self, events):
        self.sent, self._events, self.closed = [], list(events), False

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self, timeout=None):
        if self._events:
            return json.dumps(self._events.pop(0))
        time.sleep(0.02)
        raise TimeoutError

    def close(self):
        self.closed = True


def _attached(session, kind, url="https://example.com/", opener=None):
    info = {"targetId": session.upper(), "type": kind, "url": url}
    if opener:
        info["openerId"] = opener
    return {"method": "Target.attachedToTarget",
            "params": {"sessionId": session, "targetInfo": info, "waitingForDebugger": True}}


class TestKeeper:
    UA = fid.user_agent(CHROME_153)
    MD = fid.user_agent_metadata(CHROME_153)

    def _run(self, events):
        ws = _FakeWs(events)
        keeper = fid.FidelityKeeper(9, self.UA, self.MD, connect=lambda port: ws)
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

    def test_new_tab_popup_and_worker_each_get_the_brands_before_they_run(self):
        ws, keeper = self._run([
            _attached("tab1", "page"),                                     # a new tab
            _attached("pop1", "page", "https://example.org/", opener="TAB1"),  # a popup it opened
            _attached("frm1", "iframe"),                                   # an out-of-process frame
            _attached("wrk1", "worker", "blob:https://example.com/x"),     # a dedicated worker
            _attached("sw1", "service_worker", "https://example.com/sw.js"),
        ])
        by_session = {}
        for msg in ws.sent[1:]:
            by_session.setdefault(msg.get("sessionId"), []).append(msg["method"])
        for s in ("tab1", "pop1", "frm1"):
            assert by_session[s] == ["Emulation.setUserAgentOverride", "Target.setAutoAttach",
                                     "Runtime.runIfWaitingForDebugger"], s
        assert by_session["wrk1"] == ["Network.setUserAgentOverride", "Runtime.runIfWaitingForDebugger"]
        assert by_session["sw1"] == ["Network.setUserAgentOverride", "Target.setAutoAttach",
                                     "Runtime.runIfWaitingForDebugger"]
        override = next(m for m in ws.sent if m.get("sessionId") == "pop1" and "Override" in m["method"])
        assert override["params"]["userAgent"] == self.UA
        assert [b["brand"] for b in override["params"]["userAgentMetadata"]["brands"]] == [
            "Google Chrome", "Not_A Brand", "Chromium"]
        assert keeper.applied == {"page": 2, "iframe": 1, "worker": 1, "service_worker": 1}

    def test_chrome_ui_targets_are_only_resumed(self):
        ws, _ = self._run([_attached("ui1", "browser_ui", "chrome://omnibox-popup.top-chrome/")])
        assert [m["method"] for m in ws.sent if m.get("sessionId") == "ui1"] == ["Runtime.runIfWaitingForDebugger"]

    def test_a_dead_browser_ends_the_keeper_quietly(self):
        keeper = fid.FidelityKeeper(9, self.UA, self.MD, connect=lambda port: (_ for _ in ()).throw(ConnectionRefusedError("gone")))
        keeper.start()
        keeper.join(2)
        assert not keeper.is_alive() and "gone" in keeper.error

    def test_ensure_keeper_is_once_per_port_and_off_when_disabled(self):
        fid.stop_keepers()
        started = []

        class _K(threading.Thread):
            def __init__(self, port, ua, md):
                super().__init__(daemon=True)
                self.ready, self.error = threading.Event(), None
                self._halt = threading.Event()
                started.append(port)

            def run(self):
                self.ready.set()
                self._halt.wait(5)

            def stop(self):
                self._halt.set()

        with patch.object(fid, "FidelityKeeper", _K), patch.object(fid, "fidelity_enabled", return_value=True):
            a = fid.ensure_keeper(4100, CHROME_153)
            b = fid.ensure_keeper(4100, CHROME_153)
            assert a is b and started == [4100]
            assert fid.ensure_keeper(4200, None) is None
        with patch.object(fid, "FidelityKeeper", _K), patch.object(fid, "fidelity_enabled", return_value=False):
            assert fid.ensure_keeper(4300, CHROME_153) is None
        assert started == [4100]
        fid.stop_keepers()
        a.join(2)
        assert not a.is_alive()


class TestDrivenBrowserLaunch:
    """The real-profile launcher passes the fidelity switches and starts the keeper on the port."""

    def _launch(self, tmp_path, enabled, headless=True):
        captured = {}

        def fake_spawn(binary, copy_dir, extra_flags, what, headless):
            captured["flags"], captured["headless"] = list(extra_flags), headless
            return object(), 41000, None

        with patch.object(rp, "_spawn_browser_on_copy", side_effect=fake_spawn), \
             patch.object(rp, "_driven_browser_headless", return_value=headless), \
             patch.object(fid, "fidelity_enabled", return_value=enabled), \
             patch.object(fid, "main_display", return_value=(1470, 956, 2.0)), \
             patch.object(fid, "ensure_keeper") as keeper:
            port, err = rp._launch_driven_browser("/cft", str(tmp_path), CHROME_153)
        return captured, keeper, port, err

    def test_fidelity_on(self, tmp_path):
        captured, keeper, port, err = self._launch(tmp_path, enabled=True)
        assert (port, err) == (41000, None)
        flags = captured["flags"]
        assert "--disable-blink-features=AutomationControlled" in flags
        assert f"--user-agent={fid.user_agent(CHROME_153)}" in flags
        assert "--screen-info={2940x1912 devicePixelRatio=2}" in flags
        assert captured["headless"] is True  # never a window: see the module docstring for why
        keeper.assert_called_once_with(41000, CHROME_153)

    def test_fidelity_off_is_the_old_flag_set(self, tmp_path):
        captured, _, _, _ = self._launch(tmp_path, enabled=False)
        assert captured["flags"] == list(rp._mock_keychain_flags())

    def test_terminate_stops_the_keeper(self):
        with patch.object(fid, "stop_keepers") as stop:
            rp._terminate_real_profile_chrome()
        stop.assert_called_once()


@pytest.mark.skipif(os.environ.get("HERMES_BROWSER_LIVE_TESTS") != "1",
                    reason="launches a real throwaway Chrome for Testing; set HERMES_BROWSER_LIVE_TESTS=1")
class TestLiveBrowser:
    """End to end on a real Chrome for Testing with a temp profile: every new target reports the
    person's brands, including a tab opened after the keeper, a popup and a worker."""

    PROBE = ("(async () => ({ua: navigator.userAgent, webdriver: navigator.webdriver,"
             " brands: navigator.userAgentData.brands.map(b => b.brand).join(','),"
             " worker: await new Promise(r => { const w = new Worker(URL.createObjectURL(new Blob("
             "['postMessage(navigator.userAgentData.brands.map(b => b.brand).join(\",\"))']))); "
             "w.onmessage = e => r(e.data); setTimeout(() => r('timeout'), 4000) })}))()")

    def test_new_tab_popup_and_worker(self, tmp_path):
        import subprocess
        import requests
        from websockets.sync.client import connect
        binary = rp.driven_browser_executable()
        if not binary:
            pytest.skip("no packaged Chrome for Testing")
        with patch.object(fid, "fidelity_enabled", return_value=True):
            flags = [*rp._REAL_PROFILE_CHROME_FLAGS, *rp._mock_keychain_flags(),
                     *fid.launch_flags(CHROME_153, True, (1470, 956, 2.0)), "--headless=new"]
            proc = subprocess.Popen([binary, f"--user-data-dir={tmp_path}", *flags], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                port = None
                for _ in range(120):
                    try:
                        port = int((tmp_path / "DevToolsActivePort").read_text().splitlines()[0])
                        break
                    except (OSError, ValueError, IndexError):
                        time.sleep(0.25)
                assert port
                keeper = fid.ensure_keeper(port, CHROME_153)
                assert keeper is not None and keeper.error is None
                ws = connect(requests.get(f"http://127.0.0.1:{port}/json/version", timeout=3).json()["webSocketDebuggerUrl"],
                             max_size=None)
                ids = iter(range(1, 10_000))

                def call(method, params=None, session=None):
                    i = next(ids)
                    msg = {"id": i, "method": method, "params": params or {}}
                    if session:
                        msg["sessionId"] = session
                    ws.send(json.dumps(msg))
                    while True:
                        m = json.loads(ws.recv(timeout=30))
                        if m.get("id") == i:
                            assert "error" not in m, m
                            return m.get("result", {})

                def probe(target_id):
                    s = call("Target.attachToTarget", {"targetId": target_id, "flatten": True})["sessionId"]
                    for _ in range(40):
                        if call("Runtime.evaluate", {"expression": "location.protocol + document.readyState",
                                                     "returnByValue": True}, s)["result"]["value"] == "https:complete":
                            break
                        time.sleep(0.25)
                    return call("Runtime.evaluate", {"expression": self.PROBE, "awaitPromise": True,
                                                     "returnByValue": True}, s)["result"]["value"], s

                call("Target.createTarget", {"url": "about:blank"})
                tab = call("Target.createTarget", {"url": "https://example.com/"})["targetId"]
                got, session = probe(tab)
                assert got["ua"] == fid.user_agent(CHROME_153) and got["webdriver"] is False
                assert got["brands"] == "Google Chrome,Not_A Brand,Chromium"
                assert got["worker"] == "Google Chrome,Not_A Brand,Chromium"
                call("Runtime.evaluate", {"expression": "window.open('https://example.org/', '_blank'); 1",
                                          "userGesture": True}, session)
                popup = None
                for _ in range(40):
                    popup = next((t["targetId"] for t in call("Target.getTargets")["targetInfos"]
                                  if t["type"] == "page" and "example.org" in t["url"]), None)
                    if popup:
                        break
                    time.sleep(0.25)
                assert popup, "popup never opened"
                got, _ = probe(popup)
                assert got["brands"] == "Google Chrome,Not_A Brand,Chromium"
                assert keeper.applied.get("page", 0) >= 3
                ws.close()
            finally:
                fid.stop_keepers()
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
