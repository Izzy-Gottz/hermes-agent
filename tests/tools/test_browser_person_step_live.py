"""The person-step probe in a REAL browser: a throwaway headless Chrome for Testing, a scratch profile,
local pages only. Gated like the fidelity live tests (HERMES_BROWSER_LIVE_TESTS=1).

What it proves, that node cannot: PROBE_JS runs in Chrome's own V8 against real layout (visibility,
bounding boxes, innerText), across navigations -- ``Target.createTarget`` then a ``location.href``
change -- and a login page that only OFFERS a passkey stays quiet. No script is injected into any page
(the WebAuthn hook was dropped; tools/browser_person_step.py says why), so there is nothing to find
in a second document or a frame: the probe reads what each page says, when it is asked.
"""

from __future__ import annotations

import functools
import http.server
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from tools import browser_person_step as ps

LIVE = pytest.mark.skipif(os.environ.get("HERMES_BROWSER_LIVE_TESTS") != "1",
                          reason="launches a real throwaway Chrome for Testing; set HERMES_BROWSER_LIVE_TESTS=1")


def _cft():
    import pwd
    root = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".agent-browser" / "browsers"
    for app in sorted(root.glob("chrome-*/Google Chrome for Testing.app"), reverse=True):
        exe = app / "Contents" / "MacOS" / "Google Chrome for Testing"
        if exe.exists():
            return str(exe)
    return None


PAGES = {
    "pk.html": "<h1>Sign in</h1><p>Use your passkey to confirm it’s really you</p><button>Try another way</button>",
    "login.html": ("<h1>Sign in to GitHub</h1><label>Username or email address<input name=login></label>"
                   "<label>Password<input type=password name=password></label><button>Sign in</button>"
                   "<p>or</p><button>Sign in with a passkey</button>"),
    "article.html": ("<title>Sign in with a passkey - Help</title><h1>Sign in with a passkey</h1>"
                     + "<p>Use your passkey to sign in to your account on any device, instead of typing a password "
                       "every time you visit.</p>" * 6),
    "hidden.html": ("<title>Sign in - Accounts</title><h1>Sign in</h1>"
                    "<div style='position:absolute;left:-9999px'><input type=email name=identifier></div>"
                    "<input type=text style='opacity:0' name=a><div style='display:none'><input name=b></div>"
                    "<select name=hl><option>English</option></select>"
                    "<p>Use your passkey to confirm it’s really you</p><button>Try another way</button>"),
}


@pytest.fixture()
def browser():
    exe = _cft()
    if exe is None:
        pytest.skip("Chrome for Testing is not installed under ~/.agent-browser/browsers")
    site = Path(tempfile.mkdtemp(prefix="person-step-site-"))
    for name, body in PAGES.items():
        (site / name).write_text(f"<!doctype html><meta charset=utf-8><body>{body}</body>", encoding="utf-8")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix="person-step-profile-")
    proc = subprocess.Popen([exe, f"--user-data-dir={profile}", "--remote-debugging-port=0", "--no-first-run",
                             "--no-default-browser-check", "--use-mock-keychain", "--headless=new",
                             "--no-startup-window"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        port = ""
        for _ in range(120):
            try:
                port = open(os.path.join(profile, "DevToolsActivePort")).readline().strip()
            except OSError:
                port = ""
            if port:
                break
            time.sleep(0.25)
        assert port, "the throwaway browser exposed no debug port"
        yield f"http://127.0.0.1:{port}", f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        srv.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
        shutil.rmtree(site, ignore_errors=True)


def _eval_in_active(cdp, expression):
    return ps._evaluate(ps.page_targets(cdp)[0]["ws"], expression, 5)


@LIVE
def test_the_probe_reads_each_page_it_is_navigated_to(browser):
    from tools.browser_tool_real_profile import _cdp_call
    cdp, site = browser
    port = int(cdp.rsplit(":", 1)[1])
    _cdp_call(port, "Target.createTarget", {"url": f"{site}/pk.html"})
    time.sleep(1.5)
    got = ps.probe_active_page(cdp, budget=3)
    assert got and got["kind"] == "passkey" and got["page_says"] == "Use your passkey to confirm it’s really you"

    _eval_in_active(cdp, f"location.href = {site!r} + '/login.html'")   # a second document, same tab
    time.sleep(1.5)
    assert ps.page_targets(cdp)[0]["url"].endswith("/login.html")
    assert ps.probe_active_page(cdp, budget=3) is None                 # a passkey OFFERED beside a form

    _eval_in_active(cdp, f"location.href = {site!r} + '/article.html'")
    time.sleep(1.5)
    assert ps.probe_active_page(cdp, budget=3) is None                 # an article about passkeys

    _eval_in_active(cdp, f"location.href = {site!r} + '/hidden.html'")
    time.sleep(1.5)
    got = ps.probe_active_page(cdp, budget=3)                          # fields not really on screen
    assert got and got["kind"] == "passkey"


@LIVE
def test_no_script_is_left_in_the_page(browser):
    from tools.browser_tool_real_profile import _cdp_call
    cdp, site = browser
    port = int(cdp.rsplit(":", 1)[1])
    _cdp_call(port, "Target.createTarget", {"url": f"{site}/login.html"})
    time.sleep(1.5)
    ps.probe_active_page(cdp, budget=3)
    assert _eval_in_active(cdp, "Object.getOwnPropertySymbols(window).length") == 0
    assert _eval_in_active(cdp, "Function.prototype.toString.call(navigator.credentials.get)") == \
        "function get() { [native code] }"
