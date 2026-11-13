"""Fidelity on every lane, not only the signed-in one (browser.stealth_fidelity).

The signed-out lane is the browser upstream's launcher (``tools.browser_launcher``) starts for a fresh
session: real-profile browsing off, or a default browser that is not Chromium (Safari, Firefox, Arc), the
hybrid local sidecar, and the cloud machine. Before the rewire agent-browser launched it. Values quoted here were measured on 2026-09-25 on this Mac (macOS 27, 1470x956 @2x main
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
  drops its own 1280x720 and the window read back 756x556.

Since the rewire the launcher passes argv as a list (``build_chromium_argv``), so none of the
agent-browser workarounds remain: the UA goes through whole as one argument, the window size is the
display's, and there is no wrapper script.
"""
import os
import stat
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools import browser_tool_fidelity as fid
from tools import browser_launcher as launcher
from tools import browser_session as bs

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
    def _launch(self, tmp_path, operator_args=(), headless=True, display=MACBOOK, enabled=True):
        with patch.object(fid, "fidelity_enabled", return_value=enabled), \
             patch.object(fid, "launch_display", return_value=display), \
             patch.object(fid.sys, "platform", "darwin"):
            return fid.signed_out_launch_args(_fake_cft(tmp_path), headless, operator_args)

    def test_flags_are_the_engines_own_ua_the_real_display_and_no_automation_flag(self, tmp_path):
        switches, identity = self._launch(tmp_path)
        assert switches == ["--disable-blink-features=AutomationControlled",
                            f"--user-agent={UA_154_MAC}",
                            "--window-size=1440,853",
                            "--screen-info={2940x1912 devicePixelRatio=2 workAreaTop=66 "
                            "workAreaBottom=144 colorDepth=30}"]
        assert identity == CFT_154

    def test_headed_gets_no_synthetic_screen(self, tmp_path):
        switches, _ = self._launch(tmp_path, headless=False)
        assert switches == ["--disable-blink-features=AutomationControlled", f"--user-agent={UA_154_MAC}"]

    def test_off_is_nothing(self, tmp_path):
        assert self._launch(tmp_path, enabled=False) == ([], None)

    def test_an_operators_own_ua_is_left_alone(self, tmp_path):
        switches, identity = self._launch(tmp_path, operator_args=["--user-agent=mine/1.0"])
        assert identity is None and not any(a.startswith("--user-agent") for a in switches)

    def test_the_ua_reaches_chrome_as_one_argument(self, tmp_path):
        """The comma in "KHTML, like Gecko" is what agent-browser split on; the launcher's argv is a list."""
        switches, _ = self._launch(tmp_path)
        with patch.object(launcher, "_configured_chromium_args", return_value=[]), \
             patch.object(launcher, "needs_chromium_sandbox_bypass", return_value=False):
            argv = launcher.build_chromium_argv("/cft", str(tmp_path), packaged=True, headed=False,
                                                sandbox_bypass=True, extra_args=switches)
        assert f"--user-agent={UA_154_MAC}" in argv and argv.count("--headless=new") == 1

    def test_linux_virtual_display_is_declared_not_invented(self, tmp_path):
        with patch.object(fid.sys, "platform", "linux"), \
             patch("tools.browser_session_cloud.browser_section", return_value={"virtual_display": "1920x1080"}):
            display = fid.launch_display()
        assert display == {"width": 1920.0, "height": 1080.0, "scale": 1.0, "top": 0.0, "bottom": 0.0, "depth": 24}
        switches, _ = self._launch(tmp_path, display=display)
        assert switches[-1] == "--screen-info={1920x1080 devicePixelRatio=1 workAreaTop=0 workAreaBottom=0 colorDepth=24}"
        with patch.object(fid.sys, "platform", "linux"), \
             patch("tools.browser_session_cloud.browser_section", return_value={}):
            assert fid.launch_display() is None  # unset: Chrome's own screen stays, nothing is made up


class TestSignedOutSession:
    """The session layer's fresh browser (``browser_session._create_chromium_session``): the launch carries
    the switches and the keeper is serving before the session is handed out -- and again on every reuse."""

    CDP = "ws://127.0.0.1:53111/devtools/browser/abc"

    @pytest.fixture(autouse=True)
    def _host(self, monkeypatch):
        from types import SimpleNamespace
        from tools.bot_desktop import placement
        from tools import browser_lightpanda
        self.launches, self.keepers = [], []
        monkeypatch.setattr("tools.bot_desktop.runtime.tool_placement", lambda: placement.GATEWAY)
        monkeypatch.setattr("tools.bot_desktop.runtime.published_env", lambda: {})
        monkeypatch.setattr(bs, "ensure_screen_for_headed_chromium", lambda: None)
        monkeypatch.setattr(browser_lightpanda, "using_lightpanda_engine", lambda: False)
        monkeypatch.setattr(launcher, "_resolve_executable", lambda: "/cft")
        monkeypatch.setattr(launcher, "launch_chromium", lambda key, **kw: self.launches.append((key, kw)) or (
            SimpleNamespace(cdp_url=self.CDP, pid=4242), None))
        monkeypatch.setattr(fid, "ensure_keeper", lambda port, identity, wait=15.0, lane=fid.REAL_PROFILE_LANE:
                            self.keepers.append((port, identity, lane)))
        monkeypatch.setattr(fid, "signed_out_launch_args",
                            lambda binary, headless, operator_args=(): (["--disable-blink-features=AutomationControlled"], CFT_154))

    def test_the_launch_carries_the_switches_and_the_keeper_starts(self):
        info = bs._create_chromium_session("t1")
        ((key, kwargs),) = self.launches
        assert kwargs["extra_args"] == ["--disable-blink-features=AutomationControlled"]
        assert self.keepers == [(53111, CFT_154, fid.SIGNED_OUT_LANE)]
        assert info["fidelity_identity"] == CFT_154

    def test_a_reused_session_checks_its_keeper_again(self, monkeypatch):
        """A keeper that died or stalled is restarted on the next acquire (ensure_keeper's own rule)."""
        info = bs._create_chromium_session("t1")
        self.keepers.clear()
        monkeypatch.setattr(bs, "_reuse_or_evict", lambda key: info)
        assert bs.get_session_info("t1") is info
        assert self.keepers == [(53111, CFT_154, fid.SIGNED_OUT_LANE)]

    def test_a_fidelity_failure_never_fails_the_launch(self, monkeypatch):
        def broken(*a, **k):
            raise OSError("no display probe")
        monkeypatch.setattr(fid, "signed_out_launch_args", broken)
        info = bs._create_chromium_session("t1")
        assert info["cdp_url"] == self.CDP and self.keepers == []
        assert "extra_args" not in self.launches[0][1]

    def test_an_operators_own_ua_gets_no_keeper(self, monkeypatch):
        monkeypatch.setattr(fid, "signed_out_launch_args", lambda binary, headless, operator_args=(): (["--x"], None))
        info = bs._create_chromium_session("t1")
        assert self.keepers == [] and "fidelity_identity" not in info

    def test_the_bot_desktop_browser_is_left_as_upstream_starts_it(self, monkeypatch):
        """A person's visible window on the shared screen: no switches, no keeper."""
        monkeypatch.setattr(bs, "_bot_desktop_target", lambda: ("profile:/bd", {"executable": "/bd/chrome",
                                                                               "user_data_dir": "/bd", "ephemeral": False}))
        bs._create_chromium_session("t1")
        assert "extra_args" not in self.launches[0][1] and self.keepers == []


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
        """The session layer closes its browser (idle timeout, close) and the keeper ends with it: the
        monitor forgets it. A real-profile keeper that ended is still reported, as before."""
        with patch.object(fid, "KeeperProcess", _Proc), patch.object(fid, "_ensure_monitor"):
            rp_keeper, so_keeper = self._two()
            rp_keeper.alive = so_keeper.alive = False
            fid._monitor_once()
        assert list(fid._keepers) == [41000]
