"""What was typed on the page survives the relaunch that shows Moe's browser to the person -- live, in Chrome for
Testing, on a local page shaped like forums.macrumors.com's XenForo register form, whose field NAMES are random
per load (measured 2026-09-29: "99bc30e37fc7fd1f998b" one load, "855d4ba6311427a7ab64" the next), so a field is
found again by id, name, or its place among the fields with the same kind and label."""

from __future__ import annotations

import http.server
import json
import os
import secrets
import shutil
import signal
import socketserver
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from tools import browser_tool_real_profile as rp


def _cft() -> Optional[str]:
    root = Path.home() / ".agent-browser" / "browsers"
    for app in sorted(root.glob("chrome-*/Google Chrome for Testing.app"), reverse=True):
        exe = app / "Contents" / "MacOS" / "Google Chrome for Testing"
        if exe.exists():
            return str(exe)
    return None


CFT = _cft()
pytestmark = pytest.mark.skipif(CFT is None, reason="Chrome for Testing not installed under ~/.agent-browser/browsers")


def _register_page(layout: str = "") -> str:
    n = lambda: secrets.token_hex(10)  # noqa: E731 -- a new random name on every load, like XenForo
    extra = '<label for="x">Referral code</label><input id="x" type="text">' if layout == "extra" else ""
    return f"""<!doctype html><html><body><form>
<input type="hidden" name="_xfToken" value="t0k">
{extra}
<label>Username <input type="text" name="{n()}"></label>
<label>Email <input type="email" name="{n()}"></label>
<label>Password <input type="password" name="{n()}" autocomplete="new-password"></label>
<label><input type="checkbox" name="mr_is_developer"> Yes, I make apps</label>
<label>Newsletter <select name="freq"><option value="none" selected>none</option><option value="weekly">weekly</option></select></label>
<textarea name="about" placeholder="About you"></textarea>
<div class="h-captcha"><textarea name="h-captcha-response"></textarea></div>
</form></body></html>"""


@pytest.fixture(scope="module")
def browser():
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = _register_page("extra" if "extra" in self.path else "").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ud = tempfile.mkdtemp(prefix="handoff-form-live-")
    proc = subprocess.Popen([CFT, "--headless=new", f"--user-data-dir={ud}", "--remote-debugging-port=0",
                             "--no-first-run", "--no-default-browser-check", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    port_file = Path(ud) / "DevToolsActivePort"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not (port_file.exists() and port_file.read_text().strip()):
        time.sleep(0.3)
    try:
        if not (port_file.exists() and port_file.read_text().strip()):
            pytest.skip("Chrome for Testing did not start in 90 s (a loaded Mac is not a result)")
        yield {"port": int(port_file.read_text().split()[0]), "site": f"http://127.0.0.1:{srv.server_address[1]}"}
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        shutil.rmtree(ud, ignore_errors=True)
        srv.shutdown()


def _tab(port: int, url: str) -> dict:
    from tools.browser_person_step import _evaluate, page_targets
    tid = rp._cdp_call(port, "Target.createTarget", {"url": url})["targetId"]
    for _ in range(50):
        tab = next((t for t in page_targets(f"http://127.0.0.1:{port}") if t["id"] == tid), None)
        if tab and _evaluate(tab["ws"], "document.readyState === 'complete'", 2.0):
            return tab
        time.sleep(0.2)
    pytest.skip("the local page did not load (a loaded Mac is not a result)")


TYPE_IN = r"""(() => { const q = s => document.querySelector(s);
  q('input[type=text]').value = 'moe-dev-2026'; q('input[type=email]').value = 'owner@example.com';
  q('input[type=password]').value = 'correct horse'; q('input[type=checkbox]').checked = true;
  q('select').value = 'weekly'; q('textarea[name=about]').value = 'I build Memoe.';
  q('[name="h-captcha-response"]').value = 'P1_SHOULD_NOT_TRAVEL'; return 1; })()"""
READ_BACK = r"""(() => { const q = s => document.querySelector(s);
  return JSON.stringify([q('input[type=text]').value, q('input[type=email]').value, q('input[type=password]').value,
    q('input[type=checkbox]').checked, q('select').value, q('textarea[name=about]').value,
    q('[name="h-captcha-response"]').value]); })()"""


def test_typed_fields_come_back_after_a_reload_that_renamed_them(browser):
    from tools.browser_person_step import _evaluate
    first = _tab(browser["port"], browser["site"] + "/register")
    _evaluate(first["ws"], TYPE_IN, 3.0)
    saved = rp._capture_form(first["ws"])
    assert saved is not None and len(saved["fields"]) == 6            # not the hidden token, not the CAPTCHA
    assert all("captcha" not in (f["name"] or "") for f in saved["fields"])
    rp._cdp_call(browser["port"], "Target.closeTarget", {"targetId": first["id"]})

    second = _tab(browser["port"], browser["site"] + "/register")   # the relaunch: same address, new random names
    names = json.loads(_evaluate(second["ws"], "JSON.stringify(Array.from(document.querySelectorAll('input[type=text]'))"
                                               ".map(e => e.name))", 3.0))
    assert names[0] not in [f["name"] for f in saved["fields"]]     # the names really did change
    got = rp._restore_form(browser["port"], second["id"], saved)
    assert got == {"restored": 6, "missing": []}
    assert json.loads(_evaluate(second["ws"], READ_BACK, 3.0)) == [
        "moe-dev-2026", "owner@example.com", "correct horse", True, "weekly", "I build Memoe.", ""]
    assert "correct horse" not in json.dumps(got)                    # what is said never carries a value
    rp._cdp_call(browser["port"], "Target.closeTarget", {"targetId": second["id"]})


def test_a_page_that_came_back_different_is_not_filled_by_guesswork(browser):
    from tools.browser_person_step import _evaluate
    first = _tab(browser["port"], browser["site"] + "/register")
    _evaluate(first["ws"], TYPE_IN, 3.0)
    saved = rp._capture_form(first["ws"])
    rp._cdp_call(browser["port"], "Target.closeTarget", {"targetId": first["id"]})
    other = _tab(browser["port"], browser["site"] + "/register?extra")   # one more field: positions moved
    got = rp._restore_form(browser["port"], other["id"], saved)
    # named / id'd fields still find their place; the renamed-per-load ones are not guessed at
    assert got["restored"] == 3 and sorted(got["missing"]) == ["Email", "Password", "Username"]
    back = json.loads(_evaluate(other["ws"], READ_BACK, 3.0))
    assert back[:3] == ["", "", ""] and back[3:6] == [True, "weekly", "I build Memoe."]
    assert _evaluate(other["ws"], "document.getElementById('x').value", 3.0) == ""
    rp._cdp_call(browser["port"], "Target.closeTarget", {"targetId": other["id"]})
