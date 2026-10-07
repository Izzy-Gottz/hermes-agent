"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os
import shutil
import sys

import pytest

from tools import browser_tool as bt
from tools import browser_tool_install as bt_install
from tools import browser_tool_cloud as bt_cloud


@pytest.fixture(autouse=True)
def _no_pm_chromium(monkeypatch):
    """PM's store is the machine's; these tests decide what is installed."""
    monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package", lambda name: None)


class TestChromiumSearchRoots:
    def test_respects_playwright_browsers_path_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        roots = bt_install._chromium_search_roots()
        assert str(tmp_path) == roots[0]


    def test_always_includes_default_ms_playwright_cache(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        roots = bt_install._chromium_search_roots()
        home = os.path.expanduser("~")
        assert any(r == os.path.join(home, ".cache", "ms-playwright") for r in roots)


class TestChromiumInstalled:
    def test_person_chromium_on_path_is_not_the_driven_browser(self, monkeypatch, tmp_path):
        """Upstream (5e4a2a3d24) and the fork agree: the person's own browser is never a candidate."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, path=None: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert bt_install._chromium_installed() is False

    def test_pm_chromium_counts(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path / "none")])
        binary = tmp_path / "chromium"
        binary.write_text("")

        class _Installed:
            pass

        installed = _Installed()
        installed.binary = binary
        monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package",
                            lambda name: installed if name == "chromium" else None)
        assert bt_install._chromium_installed() is True


class TestDrivenBrowserExecutable:
    """The fork's driven browser keeps preferring agent-browser's Chrome for Testing; PM's Chromium is last."""

    def _cft(self, root):
        exe = root / "chrome-152.0.7977.82" / "Google Chrome for Testing.app" / "Contents" / "MacOS"
        exe.mkdir(parents=True)
        (exe / "Google Chrome for Testing").write_text("")
        return str(exe / "Google Chrome for Testing")

    def test_chrome_for_testing_before_pm_chromium(self, monkeypatch, tmp_path):
        from tools import browser_tool_real_profile as rp

        if rp.sys.platform != "darwin":
            pytest.skip("macOS layout")
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        cft = self._cft(tmp_path)
        pm_bin = tmp_path / "pm-chromium"
        pm_bin.write_text("")
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package",
                            lambda name: type("I", (), {"binary": pm_bin})() if name == "chromium" else None)
        assert rp.driven_browser_executable() == cft

    def test_pm_chromium_when_nothing_packaged(self, monkeypatch, tmp_path):
        from tools import browser_tool_real_profile as rp

        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        pm_bin = tmp_path / "pm-chromium"
        pm_bin.write_text("")
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path / "none")])
        monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package",
                            lambda name: type("I", (), {"binary": pm_bin})() if name == "chromium" else None)
        assert rp.driven_browser_executable() == str(pm_bin)


class TestHermesNodePrefix:
    def test_hermes_home_node_bin_is_a_candidate(self, monkeypatch, tmp_path):
        """A seeding host (Moe) installs agent-browser with npm into $HERMES_HOME/node/bin."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        dirs = bt_install._browser_candidate_path_dirs()
        assert str(tmp_path / "node" / "bin") in dirs




class TestCheckBrowserRequirementsChromium:

    def test_local_mode_with_chromium_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "/usr/local/bin/agent-browser")
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        from tools.browser_tool_real_profile import _driven_browser_candidates
        exe = _driven_browser_candidates(str(tmp_path), "chromium-1208")[0]
        os.makedirs(os.path.dirname(exe))
        open(exe, "w").close()

        assert bt_install.check_browser_requirements() is True


    def test_camofox_mode_does_not_require_chromium(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)
        # Even with no chromium on disk, camofox drives its own backend.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert bt_install.check_browser_requirements() is True






class TestAgentBrowserOwnDownload:
    """agent-browser stopped using Playwright's cache, and this check did not.

    Measured on macOS, agent-browser 0.26 and 0.36, `agent-browser install`:

        Downloading Chrome 152.0.7977.82 for mac-arm64
        ✓ Chrome 152.0.7977.82 installed successfully
          Location: ~/.agent-browser/browsers/chrome-152.0.7977.82

    Chrome for Testing, named ``chrome-*`` not ``chromium-*``, in
    agent-browser's own directory not Playwright's. The detector scanned
    Playwright's cache for ``chromium-*``, so a correct install reported
    "Chromium browser is missing" and the whole browser toolset was
    advertised-and-dead on every Mac that had one — while the same CLI
    opened a Wikipedia page in 12 s from the same shell.

    The docstring on ``_chromium_installed`` asserted the old layout as
    fact, which is why nobody looked. A guard built for an old layout
    refusing the current one.
    """

    def test_agent_browser_own_dir_is_searched(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        roots = bt_install._chromium_search_roots()
        home = os.path.expanduser("~")
        expected = os.path.join(home, ".agent-browser", "browsers")
        assert expected in roots, roots

    def _cft(self, root, name="chrome-152.0.7977.82"):
        exe = root / name / "Google Chrome for Testing.app" / "Contents" / "MacOS"
        exe.mkdir(parents=True)
        (exe / "Google Chrome for Testing").write_text("")

    @pytest.mark.skipif(not sys.platform == "darwin", reason="macOS layout")
    def test_chrome_for_testing_counts_as_installed(self, monkeypatch, tmp_path):
        """The exact directory name the installer prints, with the app inside it."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.shutil, "which", lambda name, path=None: None)
        self._cft(tmp_path)
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bt_install._chromium_installed() is True

    def test_an_empty_build_directory_is_not_a_browser(self, monkeypatch, tmp_path):
        """Upstream's rule, kept: a name is not a binary. An empty ``chrome-*``/``chromium-*`` directory
        (an interrupted download, a stray cache) must not advertise a browser that cannot start."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.shutil, "which", lambda name, path=None: None)
        (tmp_path / "chrome-152.0.7977.82").mkdir()
        (tmp_path / "chromium-1187").mkdir()
        (tmp_path / "chromium_headless_shell-1187").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bt_install._chromium_installed() is False

    def test_an_unrelated_directory_is_not_a_browser(self, monkeypatch, tmp_path):
        """`chrome-*` must not become a wildcard that says yes to anything."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.shutil, "which", lambda name, path=None: None)
        (tmp_path / "firefox-1234").mkdir()
        (tmp_path / "webkit-2020").mkdir()
        (tmp_path / "chromedriver").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bt_install._chromium_installed() is False
