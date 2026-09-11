"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os
import shutil

import pytest

from tools import browser_tool as bt
from tools import browser_tool_install as bt_install
from tools import browser_tool_cloud as bt_cloud


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    bt._cached_chromium_installed = None
    yield
    bt._cached_chromium_installed = None


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
    def test_true_when_plain_chromium_on_path(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            shutil,
            "which",
            lambda name, path=None: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert bt_install._chromium_installed() is True


    def test_result_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert bt_install._chromium_installed() is True
        # Delete after first call — cached True should still return True.
        (tmp_path / "chromium-1208").rmdir()
        assert bt_install._chromium_installed() is True


class TestCheckBrowserRequirementsChromium:

    def test_local_mode_with_chromium_returns_true(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: False)
        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "/usr/local/bin/agent-browser")
        monkeypatch.setattr("tools.browser_tool_install._requires_real_termux_browser_install", lambda _: False)
        monkeypatch.setattr(bt_cloud, "_get_cloud_provider", lambda: None)
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()

        assert bt_install.check_browser_requirements() is True


    def test_camofox_mode_does_not_require_chromium(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_is_camofox_mode", lambda: True)
        # Even with no chromium on disk, camofox drives its own backend.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert bt_install.check_browser_requirements() is True


class TestRunBrowserCommandChromiumGuard:
    """Verify _run_browser_command fails fast (no timeout hang) when
    Chromium is missing in local mode.
    """




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

    def test_chrome_for_testing_counts_as_installed(self, monkeypatch, tmp_path):
        """The exact directory name the installer prints."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.shutil, "which", lambda name, path=None: None)
        (tmp_path / "chrome-152.0.7977.82").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bt_install._chromium_installed() is True

    def test_playwright_chromium_still_counts(self, monkeypatch, tmp_path):
        """The old layout has not gone away — Docker images still use it."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.shutil, "which", lambda name, path=None: None)
        (tmp_path / "chromium-1187").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bt_install._chromium_installed() is True

    def test_headless_shell_still_counts(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.shutil, "which", lambda name, path=None: None)
        (tmp_path / "chromium_headless_shell-1187").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bt_install._chromium_installed() is True

    def test_an_unrelated_directory_is_not_a_browser(self, monkeypatch, tmp_path):
        """`chrome-*` must not become a wildcard that says yes to anything."""
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(bt_install.shutil, "which", lambda name, path=None: None)
        (tmp_path / "firefox-1234").mkdir()
        (tmp_path / "webkit-2020").mkdir()
        (tmp_path / "chromedriver").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert bt_install._chromium_installed() is False
