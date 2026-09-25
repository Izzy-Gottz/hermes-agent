"""Fidelity on every lane, not only the signed-in one (browser.stealth_fidelity).

The signed-out lane is the browser agent-browser launches for itself: real-profile browsing off, or a
default browser that is not Chromium (Safari, Firefox, Arc), the hybrid local sidecar, and the cloud
machine. Values quoted here were measured on 2026-09-25 on this Mac (macOS 27, 1470x956 @2x main
display, agent-browser 0.37.1, Chrome for Testing 154.0.8037.57), through ``_run_browser_command``
against a local page, not invented:

BEFORE (the fork at 9e8bca71f3, signed-out lane)::

    navigator.webdriver       true
    navigator.userAgent       ... HeadlessChrome/154.0.0.0 Safari/537.36   (and the User-Agent header)
    userAgentData.brands      Not A(Brand;99, Chromium;154
    screen                    800x600, avail 800x600, devicePixelRatio 1, outerWidth x outerHeight 0x0
    SharedWorker UA           ... HeadlessChrome/154.0.0.0 ...

AFTER (this change)::

    navigator.webdriver       false
    navigator.userAgent       ... Chrome/154.0.0.0 Safari/537.36            (and the User-Agent header)
    userAgentData.brands      Not A(Brand;99, Chromium;154                  (the engine's own, unchanged)
    fullVersionList           Not A(Brand;99.0.0.0, Chromium;154.0.8037.57
    platformVersion           27.0.0
    screen                    1470x956, avail 1470x861, devicePixelRatio 2, outer 1280x720
    SharedWorker UA           ... Chrome/154.0.0.0 ...

What did NOT work, and why the code is shaped as it is (also measured that day):

* A UA in ``AGENT_BROWSER_ARGS`` (or the config file's ``args``, a string, not a list): agent-browser
  splits it on the comma in "KHTML, like Gecko" and Chrome exits "Multiple targets are not supported
  in headless mode". Hence the launch wrapper.
* ``AGENT_BROWSER_USER_AGENT``: a per-tab override with no ``userAgentMetadata``; ``brands`` read ``""``.
* The UA switch alone (wrapper, no keeper): ``fullVersionList`` and ``platformVersion`` read ``""``.
  Chrome blanks the high-entropy hints under a custom UA. Hence the keeper on this lane too.
* ``--window-size`` through agent-browser (``1440,853`` split, ``1440x853`` not parsed): agent-browser
  drops its own 1280x720 and the window read back 756x556. Hence no window size on this lane.
"""
import os
import stat
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools import browser_tool_fidelity as fid
from tools import browser_tool_session as sess

MACBOOK = {"width": 1470.0, "height": 956.0, "scale": 2.0, "top": 33.0, "bottom": 72.0}
CFT_154 = {"brand": None, "ua_suffix": None, "full_version": "154.0.8037.57", "major": 154}
UA_154_MAC = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/154.0.0.0 Safari/537.36")


@pytest.fixture(autouse=True)
def _no_keepers_left():
    fid.stop_keepers(lane=None)
    yield
    fid.stop_keepers(lane=None)


def _fake_cft(tmp_path, version="154.0.8037.57"):
    import plistlib
    app = tmp_path / "chrome-x" / "Google Chrome for Testing.app" / "Contents"
    (app / "MacOS").mkdir(parents=True, exist_ok=True)
    (app / "MacOS" / "Google Chrome for Testing").write_text("")
    with open(app / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleShortVersionString": version}, fh)
    return str(app / "MacOS" / "Google Chrome for Testing")


class TestEngineIdentity:
    def test_the_engine_claims_itself_and_no_brand_it_is_not(self, tmp_path):
        ident = fid.engine_identity(_fake_cft(tmp_path))
        assert ident == CFT_154
        # What Chrome for Testing 154 reported for itself, before any of this: GREASE + Chromium only.
        assert ", ".join(f"{b['brand']};{b['version']}" for b in fid.brand_list(None, 154, "154")) \
            == "Not A(Brand;99, Chromium;154"
        assert fid.user_agent_metadata(ident)["fullVersionList"][1] == {"brand": "Chromium", "version": "154.0.8037.57"}

    def test_the_ua_drops_only_the_headless_token(self):
        with patch.object(fid.sys, "platform", "darwin"):
            assert fid.user_agent(CFT_154) == UA_154_MAC
        assert "Headless" not in fid.user_agent(CFT_154)

    def test_unreadable_engine_is_no_identity(self, tmp_path):
        assert fid.engine_identity(None) is None
        assert fid.engine_identity(_fake_cft(tmp_path, version="beta")) is None

    def test_linux_version_comes_from_the_binary_itself(self, tmp_path):
        binary = tmp_path / "chrome"
        binary.write_text("#!/bin/sh\necho 'Google Chrome for Testing 151.0.7922.34 '\n")
        binary.chmod(0o755)
        fid._engine_versions.clear()
        with patch.object(fid.sys, "platform", "linux"):
            assert fid.engine_identity(str(binary)) == {"brand": None, "ua_suffix": None,
                                                        "full_version": "151.0.7922.34", "major": 151}

    def test_linux_ua_and_metadata(self):
        class Uname:
            release = "6.8.0-1024-aws"
        with patch.object(fid.sys, "platform", "linux"), patch.object(fid.os, "uname", return_value=Uname()):
            assert fid.user_agent(CFT_154).startswith("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36")
            meta = fid.user_agent_metadata(CFT_154)
        assert (meta["platform"], meta["platformVersion"]) == ("Linux", "6.8.0")


class TestSignedOutLaunch:
    def _launch(self, tmp_path, env=None, headless=True, display=MACBOOK, enabled=True):
        env = {"AGENT_BROWSER_EXECUTABLE_PATH": _fake_cft(tmp_path), **(env or {})}
        with patch.object(fid, "fidelity_enabled", return_value=enabled), \
             patch.object(fid, "launch_display", return_value=display):
            return fid.signed_out_launch(env, headless)

    def test_flags_are_the_real_display_and_no_automation_flag(self, tmp_path):
        launch = self._launch(tmp_path)
        assert launch["args"] == ["--disable-blink-features=AutomationControlled",
                                  "--screen-info={2940x1912 devicePixelRatio=2 workAreaTop=66 "
                                  "workAreaBottom=144 colorDepth=30}"]
        assert launch["identity"] == CFT_154

    def test_nothing_agent_browser_would_split_or_drop(self, tmp_path):
        """No comma in any switch (agent-browser splits on it) and no --window-size (it drops its own
        1280x720 for it; measured 756x556)."""
        for arg in self._launch(tmp_path)["args"]:
            assert "," not in arg and not arg.startswith("--window-size")

    def test_headed_gets_no_synthetic_screen(self, tmp_path):
        assert self._launch(tmp_path, headless=False)["args"] == ["--disable-blink-features=AutomationControlled"]

    def test_off_is_nothing(self, tmp_path):
        assert self._launch(tmp_path, enabled=False) == {}

    def test_an_operators_own_ua_is_left_alone(self, tmp_path):
        for env in ({"AGENT_BROWSER_USER_AGENT": "mine/1.0"}, {"AGENT_BROWSER_ARGS": "--user-agent=mine/1.0"}):
            launch = self._launch(tmp_path, env)
            assert launch["identity"] is None and launch["executable"] is None

    @pytest.mark.skipif(os.name != "posix", reason="the wrapper is a sh script")
    def test_the_wrapper_passes_everything_and_the_ua_as_one_argument(self, tmp_path):
        """Run the wrapper for real, around a stand-in that prints its argv, in a folder with a space
        (Moe's home has one)."""
        folder = tmp_path / "Application Support"
        folder.mkdir()
        echo = folder / "Google Chrome for Testing"
        echo.write_text("#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done\n")
        echo.chmod(0o755)
        wrapper = fid.launch_wrapper(str(echo), UA_154_MAC)
        assert wrapper and os.stat(wrapper).st_mode & stat.S_IXUSR
        out = subprocess.run([wrapper, "--headless=new", "--screen-info={2940x1912 devicePixelRatio=2}"],
                             capture_output=True, text=True, timeout=10).stdout.splitlines()
        assert out == ["--headless=new", "--screen-info={2940x1912 devicePixelRatio=2}", f"--user-agent={UA_154_MAC}"]
        assert fid.launch_wrapper(str(echo), UA_154_MAC) == wrapper  # same browser, same UA: same file

    def test_a_noexec_home_is_passed_over(self, tmp_path):
        """A wrapper the kernel will not run would stop the browser starting at all."""
        real = os.statvfs

        class NoExec:
            def __init__(self, st):
                self.f_flag = st.f_flag | getattr(os, "ST_NOEXEC", 1)

        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
        with patch.object(fid.os, "statvfs", side_effect=lambda p: NoExec(real(p)) if str(p).startswith(home) else real(p)), \
             patch.object(fid.os, "ST_NOEXEC", getattr(os, "ST_NOEXEC", 1), create=True):
            wrapper = fid.launch_wrapper("/cft", UA_154_MAC)
        assert wrapper and not wrapper.startswith(home)
        with patch.object(fid.os, "statvfs", side_effect=lambda p: NoExec(real(p))), \
             patch.object(fid.os, "ST_NOEXEC", getattr(os, "ST_NOEXEC", 1), create=True):
            assert fid.launch_wrapper("/cft", UA_154_MAC) is None

    def test_launch_uses_the_wrapper_around_the_engine(self, tmp_path):
        launch = self._launch(tmp_path)
        body = open(launch["executable"], encoding="utf-8").read()
        assert "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing" in body
        assert fid.user_agent(CFT_154) in body

    def test_linux_virtual_display_is_declared_not_invented(self, tmp_path):
        from tools.browser_tool_origin import origin_module
        with patch.object(fid.sys, "platform", "linux"), \
             patch.object(origin_module(), "_browser_cfg", side_effect=lambda k, d, parse, _: parse("1920x1080")):
            display = fid.launch_display()
        assert display == {"width": 1920.0, "height": 1080.0, "scale": 1.0, "top": 0.0, "bottom": 0.0, "depth": 24}
        args = self._launch(tmp_path, display=display)["args"]
        assert args[-1] == "--screen-info={1920x1080 devicePixelRatio=1 workAreaTop=0 workAreaBottom=0 colorDepth=24}"
        with patch.object(fid.sys, "platform", "linux"), \
             patch.object(origin_module(), "_browser_cfg", side_effect=lambda k, d, parse, _: d):
            assert fid.launch_display() is None  # unset: Chrome's own screen stays, nothing is made up


class TestLaunchEnv:
    LAUNCH = {"args": ["--disable-blink-features=AutomationControlled", "--screen-info={1 2}"],
              "executable": "/h/browser-fidelity/launch-x.sh", "identity": CFT_154}

    def test_after_the_sandbox_bypass_not_instead_of_it(self):
        env = {"AGENT_BROWSER_ARGS": "--no-sandbox,--disable-dev-shm-usage"}
        fid.apply_launch_env(env, self.LAUNCH)
        assert env["AGENT_BROWSER_ARGS"] == ("--no-sandbox,--disable-dev-shm-usage,"
                                             "--disable-blink-features=AutomationControlled,--screen-info={1 2}")
        assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == "/h/browser-fidelity/launch-x.sh"

    def test_a_switch_the_operator_set_is_not_repeated(self):
        env = {"AGENT_BROWSER_ARGS": "--screen-info={9 9}"}
        fid.apply_launch_env(env, self.LAUNCH)
        assert env["AGENT_BROWSER_ARGS"] == "--screen-info={9 9},--disable-blink-features=AutomationControlled"

    def test_nothing_when_off(self):
        env = {"A": "1"}
        fid.apply_launch_env(env, {})
        assert env == {"A": "1"}


class TestSignedOutSession:
    """``_run_browser_command`` on a session agent-browser launches: the browser is started on its
    blank tab and the keeper is serving BEFORE the model's first command runs."""

    def _run(self, session, commands, keeper_state="serving"):
        calls, keepers = [], []

        def spawn(task_id, info, cmd, command, engine, timeout):
            calls.append(cmd[cmd.index("--json") + 1:])
            if cmd[-2:] == ["get", "cdp-url"]:
                return {"success": True, "data": {"cdpUrl": "ws://127.0.0.1:53111/devtools/browser/abc"}}
            return {"success": True, "data": {}}

        class K:
            state = keeper_state

        def ensure(port, identity, wait=15.0, lane=fid.REAL_PROFILE_LANE):
            keepers.append((port, identity, lane))
            return K()

        with patch.object(sess, "_browser_command_preflight", return_value={"browser_cmd": "/ab"}), \
             patch.object(sess, "_get_session_info", return_value=session), \
             patch.object(sess._cdp, "_ensure_cdp_supervisor"), \
             patch.object(sess._cloud, "_get_browser_engine", return_value="auto"), \
             patch.object(sess._cloud, "_is_headed_mode", return_value=False), \
             patch("tools.browser_tool._is_camofox_mode", return_value=False), \
             patch.object(sess, "_spawn_and_collect", side_effect=spawn), \
             patch.object(fid, "signed_out_launch", return_value={"args": [], "executable": None, "identity": CFT_154}), \
             patch.object(fid, "ensure_keeper", side_effect=ensure), \
             patch.object(fid, "stop_keeper") as stop:
            for command, args in commands:
                sess._run_browser_command("t1", command, args, timeout=5)
        return calls, keepers, stop

    def test_first_command_launches_blank_and_starts_the_keeper_first(self):
        session = {"session_name": "h_1", "cdp_url": None, "features": {"local": True}}
        calls, keepers, _ = self._run(session, [("open", ["https://example.com/"]), ("snapshot", [])])
        assert calls == [["get", "cdp-url"], ["open", "https://example.com/"], ["snapshot"]]
        assert keepers == [(53111, CFT_154, fid.SIGNED_OUT_LANE), (53111, CFT_154, fid.SIGNED_OUT_LANE)]
        assert session["fidelity_port"] == 53111

    def test_a_gone_browser_is_found_again(self):
        session = {"session_name": "h_1", "cdp_url": None, "features": {"local": True}, "fidelity_port": 50000,
                   "fidelity_launch": {"args": [], "executable": None, "identity": CFT_154}}
        calls, _, stop = self._run(session, [("snapshot", [])], keeper_state="failed")
        stop.assert_any_call(50000)
        assert calls[0] == ["get", "cdp-url"] and session["fidelity_port"] == 53111

    def test_a_fidelity_failure_never_fails_the_command(self):
        session = {"session_name": "h_1", "cdp_url": None, "features": {"local": True}}
        with patch.object(sess, "_ensure_signed_out_fidelity", side_effect=OSError("no fork")):
            calls, _, _ = self._run(session, [("snapshot", [])])
        assert calls == [["snapshot"]]

    def test_close_stops_its_keeper(self):
        session = {"session_name": "h_1", "cdp_url": None, "features": {"local": True}, "fidelity_port": 53111,
                   "fidelity_launch": {"args": [], "executable": None, "identity": CFT_154}}
        calls, keepers, stop = self._run(session, [("close", [])])
        assert calls == [["close"]] and keepers == []
        stop.assert_called_once_with(53111)

    @pytest.mark.parametrize("session", [
        {"session_name": "rp_1", "cdp_url": "ws://127.0.0.1:41000/devtools/browser/x", "features": {"local": True, "real_profile": True}},
        {"session_name": "lp_1", "cdp_url": None, "features": {"local": True, "lightpanda": True}},
        {"session_name": "cdp_1", "cdp_url": "wss://cloud.example/x", "features": {"cdp_override": True}},
    ])
    def test_other_lanes_are_not_touched_here(self, session):
        calls, keepers, _ = self._run(session, [("snapshot", [])])
        assert calls == [["snapshot"]] and keepers == []

    def test_every_command_carries_the_launch_env(self, tmp_path):
        """Whichever command (re)launches the browser, the switches and the wrapper are in its env."""
        captured = {}
        session = {"session_name": "h_1", "fidelity_launch": {
            "args": ["--disable-blink-features=AutomationControlled"], "executable": "/w.sh", "identity": CFT_154}}

        def popen(argv, env, socket_dir, tag):
            captured.update(env)
            raise RuntimeError("stop here")

        with patch.object(sess, "_prepare_session_socket_dir", return_value=str(tmp_path)), \
             patch.object(sess, "_popen_agent_browser", side_effect=popen), \
             patch.object(sess, "_needs_chromium_sandbox_bypass", return_value=True):
            with pytest.raises(RuntimeError):
                sess._spawn_and_collect("t1", session, ["/ab", "open", "x"], "open", "auto", 5)
        assert captured["AGENT_BROWSER_ARGS"] == \
            "--no-sandbox,--disable-dev-shm-usage,--disable-blink-features=AutomationControlled"
        assert captured["AGENT_BROWSER_EXECUTABLE_PATH"] == "/w.sh"


class _Proc:
    def __init__(self, port, identity, lane=fid.REAL_PROFILE_LANE):
        self.port, self.identity, self.lane, self.alive, self.stopped = port, identity, lane, True, False
        self.state, self.error = "serving", None

    def is_alive(self):
        return self.alive

    def healthy(self):
        return self.alive

    def heartbeat_age(self):
        return 0.0

    def wait_ready(self, wait):
        pass

    def stop(self):
        self.stopped, self.alive = True, False

    kill = stop

    def join(self, timeout=None):
        pass

    def describe(self):
        return {"port": self.port, "lane": self.lane, "state": "failed" if not self.alive else self.state}


class TestLanes:
    def _two(self):
        rp_keeper = fid._start_keeper(41000, CFT_154)
        so_keeper = fid._start_keeper(53111, CFT_154, fid.SIGNED_OUT_LANE)
        return rp_keeper, so_keeper

    def test_closing_the_real_profile_browser_leaves_a_signed_out_one_served(self):
        with patch.object(fid, "KeeperProcess", _Proc), patch.object(fid, "_ensure_monitor"):
            rp_keeper, so_keeper = self._two()
            fid.stop_keepers()  # what _terminate_real_profile_chrome calls
        assert rp_keeper.stopped and not so_keeper.stopped and list(fid._keepers) == [53111]

    def test_a_session_close_stops_only_its_own(self):
        with patch.object(fid, "KeeperProcess", _Proc), patch.object(fid, "_ensure_monitor"):
            rp_keeper, so_keeper = self._two()
            fid.stop_keeper(53111)
        assert so_keeper.stopped and not rp_keeper.stopped and list(fid._keepers) == [41000]

    def test_a_closed_signed_out_browser_is_not_reported_as_a_failure(self):
        """agent-browser closes its browser (idle timeout, close) and the keeper ends with it: the
        monitor forgets it. A real-profile keeper that ended is still reported, as before."""
        with patch.object(fid, "KeeperProcess", _Proc), patch.object(fid, "_ensure_monitor"):
            rp_keeper, so_keeper = self._two()
            rp_keeper.alive = so_keeper.alive = False
            fid._monitor_once()
        assert list(fid._keepers) == [41000]


# ---------------------------------------------------------------------------
# Live: the signed-out lane through _run_browser_command, a real agent-browser and Chrome for Testing
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skipif(os.environ.get("HERMES_BROWSER_LIVE_TESTS") != "1" or sys.platform != "darwin",
                          reason="launches a real throwaway Chrome for Testing; set HERMES_BROWSER_LIVE_TESTS=1")

_PAGE = b"""<!doctype html><html><body><script>
(async()=>{const n=navigator;const h=await n.userAgentData.getHighEntropyValues(["fullVersionList","platformVersion"]);
window.__fid={webdriver:n.webdriver,ua:n.userAgent,brands:n.userAgentData.brands.map(b=>b.brand+";"+b.version).join(", "),
full:h.fullVersionList.map(b=>b.brand+";"+b.version).join(", "),pv:h.platformVersion,screen:screen.width+"x"+screen.height,
dpr:devicePixelRatio};})();</script></body></html>"""

_SHARED_WORKER_UA = ("new Promise(r=>{const w=new SharedWorker(URL.createObjectURL(new Blob(["
                     "'onconnect=e=>e.ports[0].postMessage(navigator.userAgent)'],{type:'text/javascript'})));"
                     "w.port.onmessage=m=>r(m.data);w.port.start();setTimeout(()=>r('timeout'),8000)})")


@pytest.fixture
def page_server():
    import http.server
    import socketserver
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = _PAGE
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    server = TS(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


@LIVE
def test_live_signed_out_lane_tells_the_truth(page_server):
    import json
    import time
    from tools import browser_tool_real_profile as rp
    binary = rp.driven_browser_executable()
    if not binary:
        pytest.skip("no packaged Chrome for Testing")
    engine = fid.engine_identity(binary)
    display = fid.main_display()
    task = "fidelity-live-signed-out"
    with patch.object(fid, "fidelity_enabled", return_value=True), \
         patch.object(sess._real_profile._cloud, "_use_real_profile", return_value=False):
        try:
            assert sess._run_browser_command(task, "open", [page_server], timeout=120).get("success")
            got = None
            for _ in range(40):
                res = sess._run_browser_command(task, "eval", ["JSON.stringify(window.__fid || null)"], timeout=30)
                got = json.loads(res.get("data", {}).get("result") or "null")
                if got:
                    break
                time.sleep(0.25)
            worker_ua = sess._run_browser_command(task, "eval", [_SHARED_WORKER_UA], timeout=30)["data"]["result"]
        finally:
            sess._run_browser_command(task, "close", [], timeout=30)
    ua = fid.user_agent(engine)
    assert got["webdriver"] is False                          # before: true
    assert got["ua"] == ua and "Headless" not in got["ua"]    # before: HeadlessChrome/154.0.0.0
    assert worker_ua == ua                                    # before: HeadlessChrome in the SharedWorker too
    assert got["brands"] == ", ".join(f"{b['brand']};{b['version']}"
                                      for b in fid.brand_list(None, engine["major"], str(engine["major"])))
    assert got["full"].endswith(f"Chromium;{engine['full_version']}")  # blank with the UA switch alone
    assert got["pv"] == fid._mac_platform_version()
    if display:
        assert got["screen"] == f"{int(display['width'])}x{int(display['height'])}"  # before: 800x600
        assert got["dpr"] == display["scale"]                                         # before: 1
