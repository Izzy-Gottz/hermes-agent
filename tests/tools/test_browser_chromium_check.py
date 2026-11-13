"""Chromium-presence detection for the launcher and the driven browser (fork).

Upstream's launcher (``tools.browser_launcher``) knows one browser: PM's pinned Chromium (or the
operator's ``browser.executable_path``). The fork keeps the rule its ``_chromium_installed`` had: a
packaged Chrome for Testing that a host app (Memoe) already has counts too -- the same build the
driven real-profile browser runs (``driven_browser_executable``) -- and the person's own installed
browser never does. An empty or unrelated directory is not a browser.
"""

import os
import shutil
import sys

import pytest

from tools import browser_launcher as launcher
from tools import browser_tool_install as bt_install


@pytest.fixture(autouse=True)
def _no_pm_chromium(monkeypatch):
    """PM's store is the machine's; these tests decide what is installed."""
    monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package", lambda name: None)
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr(launcher, "_autoinstall_attempted", True)  # never download here


def _installed() -> bool:
    return launcher._resolve_executable() is not None


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

    def test_agent_browser_own_dir_is_searched(self, monkeypatch):
        """A Mac that has agent-browser's download keeps its Chrome for Testing after the rewire."""
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        roots = bt_install._chromium_search_roots()
        assert os.path.join(os.path.expanduser("~"), ".agent-browser", "browsers") in roots, roots


class TestChromiumInstalled:
    def test_person_chromium_on_path_is_not_the_driven_browser(self, monkeypatch, tmp_path):
        """Upstream (5e4a2a3d24) and the fork agree: the person's own browser is never a candidate."""
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        monkeypatch.setattr(shutil, "which",
                            lambda name, path=None: "/usr/bin/chromium" if name == "chromium" else None)
        assert _installed() is False
        assert launcher.chromium_ready()[0] is False

    def test_pm_chromium_counts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path / "none")])
        binary = tmp_path / "chromium"
        binary.write_text("")
        installed = type("I", (), {"binary": binary})()
        monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package",
                            lambda name: installed if name == "chromium" else None)
        assert launcher._resolve_executable() == str(binary)

    def _cft(self, root, name="chrome-152.0.7977.82"):
        exe = root / name / "Google Chrome for Testing.app" / "Contents" / "MacOS"
        exe.mkdir(parents=True)
        (exe / "Google Chrome for Testing").write_text("")
        return str(exe / "Google Chrome for Testing")

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS layout")
    def test_chrome_for_testing_counts_as_installed(self, monkeypatch, tmp_path):
        """The exact directory name agent-browser's installer printed, with the app inside it."""
        monkeypatch.setattr(shutil, "which", lambda name, path=None: None)
        cft = self._cft(tmp_path)
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert launcher._resolve_executable() == cft
        assert launcher.chromium_ready() == (True, "")

    def test_an_empty_build_directory_is_not_a_browser(self, monkeypatch, tmp_path):
        """A name is not a binary: an empty ``chrome-*``/``chromium-*`` directory (an interrupted
        download, a stray cache) must not advertise a browser that cannot start."""
        monkeypatch.setattr(shutil, "which", lambda name, path=None: None)
        (tmp_path / "chrome-152.0.7977.82").mkdir()
        (tmp_path / "chromium-1187").mkdir()
        (tmp_path / "chromium_headless_shell-1187").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert _installed() is False

    def test_an_unrelated_directory_is_not_a_browser(self, monkeypatch, tmp_path):
        """`chrome-*` must not become a wildcard that says yes to anything."""
        monkeypatch.setattr(shutil, "which", lambda name, path=None: None)
        (tmp_path / "firefox-1234").mkdir()
        (tmp_path / "webkit-2020").mkdir()
        (tmp_path / "chromedriver").mkdir()
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        assert _installed() is False


class TestDrivenBrowserExecutable:
    """The fork's driven browser keeps preferring a packaged Chrome for Testing; PM's Chromium is last."""

    def _cft(self, root):
        exe = root / "chrome-152.0.7977.82" / "Google Chrome for Testing.app" / "Contents" / "MacOS"
        exe.mkdir(parents=True)
        (exe / "Google Chrome for Testing").write_text("")
        return str(exe / "Google Chrome for Testing")

    def test_chrome_for_testing_before_pm_chromium(self, monkeypatch, tmp_path):
        from tools import browser_tool_real_profile as rp

        if rp.sys.platform != "darwin":
            pytest.skip("macOS layout")
        cft = self._cft(tmp_path)
        pm_bin = tmp_path / "pm-chromium"
        pm_bin.write_text("")
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path)])
        monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package",
                            lambda name: type("I", (), {"binary": pm_bin})() if name == "chromium" else None)
        assert rp.driven_browser_executable() == cft

    def test_pm_chromium_when_nothing_packaged(self, monkeypatch, tmp_path):
        from tools import browser_tool_real_profile as rp

        pm_bin = tmp_path / "pm-chromium"
        pm_bin.write_text("")
        monkeypatch.setattr(bt_install, "_chromium_search_roots", lambda: [str(tmp_path / "none")])
        monkeypatch.setattr("hermes_cli.browser_runtime.pm.installed_package",
                            lambda name: type("I", (), {"binary": pm_bin})() if name == "chromium" else None)
        assert rp.driven_browser_executable() == str(pm_bin)
