"""The extension's own guards, run for real: extensions/chrome/background.js under node, fake chrome.*.

tests/tools/chrome_extension_harness.js loads the shipped background.js into a VM with a fake
``chrome`` (one person's tab, ungrouped) and plays the native host. The mutation tests at the end
load a copy with a guard gutted and require the same harness to see the damage, so a harness that
cannot fail would be caught here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BACKGROUND = HERE.parents[1] / "extensions" / "chrome" / "background.js"
HARNESS = HERE / "chrome_extension_harness.js"


def _node():
    for cand in (shutil.which("node"), os.path.expanduser("~/.hermes/node/bin/node")):
        if cand and os.access(cand, os.X_OK):
            return cand
    return None


NODE = _node()
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run(src: Path = BACKGROUND) -> dict:
    out = subprocess.run([NODE, str(HARNESS), str(src)], capture_output=True, text=True, timeout=60)
    report = json.loads(out.stdout or "{}")
    assert "crashed" not in report, report.get("crashed")
    return report


@pytest.fixture(scope="module")
def report():
    return run()


def test_the_persons_tab_is_refused_before_chrome_debugger_is_called(report):
    assert {k: v["error"] for k, v in report["foreign"].items()} == {"cdp": "not_owned", "attach": "not_owned", "close": "not_owned"}
    assert report["foreignDebuggerCalls"] == {"attach": 0, "send": 0}
    assert report["foreignStillOpen"] is True


def test_a_tab_dragged_out_of_the_group_is_the_persons_again(report):
    assert report["afterDragOut"]["error"] == "not_owned"
    assert report["create"]["result"]["tabId"] not in report["targets"]


def test_create_opens_only_http_https_about_blank(report):
    assert report["moeNavigate"] == [["Page.navigate", "https://example.com/"]]
    for url, r in report["createRefused"].items():
        assert r == {"error": "url_refused", "tabsMade": 0}, url


def test_every_command_rechecks_where_the_tab_is(report):
    assert report["moeRead"]["result"]["result"]["value"].endswith("https://example.com/")
    assert report["moeNavFile"]["error"] == "url_refused"
    # already on file: -> refused, nothing read, and the tab is NOT navigated away (a dragged-in
    # chrome:// tab is the person's to keep); Moe can still send it somewhere allowed
    assert report["moeOnFile"]["error"] == "url_refused"
    assert report["moeOnFileSends"] == []
    assert report["moeUrlAfter"] == "file:///etc/hosts"
    assert "result" in report["moeNavAway"] and "result" in report["moeReadAfterNavAway"]
    # the frame, not getTargets(), decides: an error page is refused even when the listing says http
    assert report["moeErrorPage"]["error"] == "url_refused"
    assert report["moeChromeErrorShown"]["error"] == "url_refused"
    # moved to chrome: while the command ran -> the result is withheld
    assert report["moeRace"]["error"] == "url_refused"


def test_nothing_focuses_and_moes_window_is_behind_the_persons(report):
    assert report["activate"]["error"] == "unknown_op" and report["focusCallsAfterActivate"] == 0
    assert report["createWindows"] == [{"url": "about:blank", "focused": False}]
    assert report["refocus"] == [{"id": 10, "props": {"focused": True}}]  # the person's window, handed back
    assert report["secondSameWindow"] is True
    assert report["hello"]["result"]["fileAccess"] is False


@pytest.mark.parametrize("anchor,gutted,expect", [
    ("async function requireOwned(tabId) {\n", "async function requireOwned(tabId) {\n  return;\n",
     lambda r: r["foreign"]["cdp"].get("error") is None and r["foreignDebuggerCalls"]["send"] > 0),
    ("function urlAllowed(url) {\n", "function urlAllowed(url) {\n  return true;\n",
     lambda r: r["createRefused"]["file:///etc/hosts"]["error"] is None),
])
def test_the_harness_goes_red_when_a_guard_is_gutted(tmp_path, anchor, gutted, expect):
    src = BACKGROUND.read_text()
    assert anchor in src
    mutant = tmp_path / "background.js"
    mutant.write_text(src.replace(anchor, gutted, 1))
    assert expect(run(mutant))
