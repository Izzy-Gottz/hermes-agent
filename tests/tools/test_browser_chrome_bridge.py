"""The Memoe Chrome bridge (tools/browser_chrome_bridge.py), run as Chrome runs it.

The host is spawned as a real subprocess and this test plays the extension on its
stdio with native-messaging framing — so what is measured is the process Chrome
would start, not a mock of it. Then a CDP client connects to the socket it
publishes, the way browser_harness does.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

websockets = pytest.importorskip("websockets")
from websockets.asyncio.client import connect  # noqa: E402
from websockets.exceptions import InvalidStatus  # noqa: E402

from tools import browser_chrome_bridge as bridge  # noqa: E402

BRIDGE = Path(bridge.__file__)
ORIGIN = "chrome-extension://ljblhmlmgflmbffbfelamefleikodmjh/"


class FakeExtension:
    """Plays chrome.runtime.connectNative's other end: owns tab 7 (Moe's) and knows of tab 1 (the person's)."""

    def __init__(self, proc):
        self.proc = proc
        self.ops = []
        self.moe_tabs = {7: {"targetId": "T7", "tabId": 7, "url": "about:blank", "title": "", "attached": False}}
        self.next_tab = 8
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, obj):
        self.proc.stdin.write(bridge.encode_native_message(obj))
        self.proc.stdin.flush()

    def _loop(self):
        while True:
            msg = bridge.read_native_message(self.proc.stdout)
            if msg is None:
                return
            self.ops.append(msg)
            op, args, rid = msg["op"], msg.get("args") or {}, msg["id"]
            if op == "hello":
                self.send({"id": rid, "result": {"protocol": 1, "userAgent": "Mozilla/5.0 Chrome/153.0.0.0 Safari/537.36"}})
            elif op == "targets":
                self.send({"id": rid, "result": list(self.moe_tabs.values())})
            elif op == "create":
                tab = self.next_tab
                self.next_tab += 1
                self.moe_tabs[tab] = {"targetId": f"T{tab}", "tabId": tab, "url": args["url"], "title": "", "attached": False}
                self.send({"id": rid, "result": {"tabId": tab, "targetId": f"T{tab}"}})
            elif op in ("attach", "detach", "close"):
                if args.get("tabId") not in self.moe_tabs:
                    self.send({"id": rid, "error": {"code": "not_owned", "message": "not in Moe's group"}})
                else:
                    self.send({"id": rid, "result": {}})
            elif op == "cdp":
                if args["method"] == "Runtime.evaluate":
                    self.send({"event": "cdp", "tabId": args["tabId"], "method": "Runtime.consoleAPICalled", "params": {"type": "log"}})
                    self.send({"event": "log", "text": "evaluated"})
                    self.send({"id": rid, "result": {"result": {"type": "number", "value": 2}}})
                elif args["method"] == "Accessibility.getFullAXTree":
                    pass  # never answers: must not stall other requests on the socket
                else:
                    self.send({"id": rid, "result": {}})


def _set_lane(home: Path, enabled: bool) -> None:
    (home / "config.yaml").write_text(f"browser:\n  chrome_extension:\n    enabled: {'true' if enabled else 'false'}\n")


def _spawn(home: Path):
    env = dict(os.environ, HERMES_HOME=str(home))
    proc = subprocess.Popen([sys.executable, str(BRIDGE), ORIGIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    return proc, FakeExtension(proc)


def _wait(pred, timeout=20):
    deadline = time.time() + timeout
    while not pred() and time.time() < deadline:
        time.sleep(0.05)
    return pred()


@pytest.fixture()
def host(tmp_path):
    _set_lane(tmp_path, True)
    proc, ext = _spawn(tmp_path)
    state = tmp_path / "chrome-bridge" / "bridge.json"
    assert _wait(state.exists), proc.stderr.read1(4000) if proc.poll() is not None else "bridge.json never appeared"
    yield proc, ext, state
    proc.stdin.close()
    proc.wait(timeout=10)


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


class Client:
    def __init__(self, ws):
        self.ws, self.n, self.events, self.pending = ws, 0, [], {}

    async def pump(self):
        async for raw in self.ws:
            m = json.loads(raw)
            fut = self.pending.pop(m.get("id"), None) if "id" in m else None
            (fut.set_result(m) if fut else self.events.append(m))

    async def call(self, method, params=None, sid=None, timeout=10):
        self.n += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.n] = fut
        msg = {"id": self.n, "method": method, "params": params or {}}
        if sid:
            msg["sessionId"] = sid
        await self.ws.send(json.dumps(msg))
        return await asyncio.wait_for(fut, timeout)


def test_state_file_is_private_and_names_the_calling_extension(host):
    proc, _, state = host
    st = state.stat()
    assert stat.S_IMODE(st.st_mode) == 0o600
    assert stat.S_IMODE(state.parent.stat().st_mode) == 0o700
    data = json.loads(state.read_text())
    assert data["pid"] == proc.pid and data["extension_origin"] == ORIGIN
    assert data["ws_url"].startswith("ws://127.0.0.1:") and len(data["ws_url"].rsplit("/", 1)[1]) >= 40


def test_socket_refuses_browser_pages_and_guessed_paths(host):
    _, _, state = host
    url = json.loads(state.read_text())["ws_url"]

    async def go():
        codes = {}
        for label, kw, u in (("origin", {"origin": "https://evil.example"}, url),
                             ("path", {}, url.rsplit("/", 1)[0] + "/not-the-token")):
            try:
                async with connect(u, **kw):
                    codes[label] = 101
            except InvalidStatus as e:
                codes[label] = e.response.status_code
        async with connect(url) as ws:  # the harness sends no Origin
            codes["harness"] = 101
        return codes

    assert _run(go()) == {"origin": 403, "path": 404, "harness": 101}


def test_targets_are_only_moes_tabs_and_new_tabs_open_in_the_background(host):
    _, ext, state = host
    url = json.loads(state.read_text())["ws_url"]

    async def go():
        async with connect(url) as ws:
            c = Client(ws)
            asyncio.ensure_future(c.pump())
            targets = (await c.call("Target.getTargets"))["result"]["targetInfos"]
            foreign = await c.call("Target.attachToTarget", {"targetId": "T1", "flatten": True})
            created = (await c.call("Target.createTarget", {"url": "about:blank", "background": False}))["result"]
            return targets, foreign, created

    targets, foreign, created = _run(go())
    assert [t["targetId"] for t in targets] == ["T7"]
    assert "error" in foreign and "Moe's group" in foreign["error"]["message"]
    create_op = next(o for o in ext.ops if o["op"] == "create")
    assert create_op["args"] == {"url": "about:blank"}  # the client's background=False never reaches the extension
    assert created == {"targetId": "T8"}


def test_session_commands_forward_with_events_and_policy(host, tmp_path):
    _, ext, state = host
    url = json.loads(state.read_text())["ws_url"]

    async def go():
        async with connect(url) as ws:
            c = Client(ws)
            asyncio.ensure_future(c.pump())
            sid = (await c.call("Target.attachToTarget", {"targetId": "T7", "flatten": True}))["result"]["sessionId"]
            slow = asyncio.ensure_future(c.call("Accessibility.getFullAXTree", {}, sid, timeout=5))
            evaluated = await c.call("Runtime.evaluate", {"expression": "1+1"}, sid)  # not stuck behind the slow one
            refused = [(await c.call(m, p, sid)).get("error", {}).get("message", "FORWARDED") for m, p in (
                ("Network.getAllCookies", {}), ("Storage.getCookies", {}), ("Browser.close", {}),
                ("Page.navigate", {"url": "file:///etc/passwd"}), ("Page.navigate", {"url": "chrome://settings"}))]
            ok_nav = await c.call("Page.navigate", {"url": "https://example.com/"}, sid)
            await asyncio.sleep(0.2)
            slow.cancel()
            return sid, evaluated, refused, ok_nav, c.events

    sid, evaluated, refused, ok_nav, events = _run(go())
    assert evaluated["result"]["result"]["value"] == 2 and evaluated["sessionId"] == sid
    assert {"method": "Runtime.consoleAPICalled", "params": {"type": "log"}, "sessionId": sid} in events
    assert "FORWARDED" not in refused, refused
    assert "result" in ok_nav
    forwarded = [o["args"]["method"] for o in ext.ops if o["op"] == "cdp"]
    assert "Network.getAllCookies" not in forwarded and "Storage.getCookies" not in forwarded
    assert (tmp_path / "chrome-bridge" / "bridge.log").read_text().strip().endswith("evaluated")


def test_host_exits_and_removes_state_when_chrome_closes_the_port(host):
    proc, _, state = host
    proc.stdin.close()
    proc.wait(timeout=10)
    assert not state.exists()


# Measured holes from the security review (2026-09-23), each of which must now be refused:
REVIEW_HOLES = [
    ("Network.setCookie", {"name": "sid", "value": "x", "domain": "bank.example"}),
    ("Network.deleteCookies", {"name": "sid", "domain": "mail.example"}),
    ("Network.getAllCookies", {}), ("Network.getCookies", {}),
    ("DOM.setFileInputFiles", {"files": ["/etc/hosts"], "nodeId": 5}),
    ("Fetch.enable", {}), ("Fetch.continueRequest", {"requestId": "1"}),
    ("Storage.getCookies", {}), ("Storage.clearDataForOrigin", {"origin": "https://mail.example", "storageTypes": "all"}),
    ("Browser.close", {}), ("Browser.setDownloadBehavior", {"behavior": "allow"}),
    ("Page.setDownloadBehavior", {"behavior": "allow"}), ("Target.createTarget", {"url": "about:blank"}),
    ("Page.bringToFront", {}), ("Emulation.setUserAgentOverride", {"userAgent": "x"}),
    ("Page.navigate", {"url": "file:///etc/hosts"}), ("Page.navigate", {"url": "chrome://settings"}),
    ("Page.navigate", {"url": "chrome-extension://ljblhmlmgflmbffbfelamefleikodmjh/manifest.json"}),
    ("Page.navigate", {"url": "javascript:alert(1)"}), ("Page.navigate", {"url": "view-source:https://a.example/"}),
    ("Page.navigate", {"url": "data:text/html,<script>1</script>"}),
    ("Page.navigate", {"url": "https://a.example/", "frameId": "sub"}),          # unknown parameter
    ("Runtime.evaluate", {"expression": "1", "contextId": 3, "includeCommandLineAPI": True}),
]


@pytest.mark.parametrize("method,params", REVIEW_HOLES)
def test_review_holes_are_refused(method, params):
    with pytest.raises(bridge.RelayError):
        bridge.check_session_method(method, params)


@pytest.mark.parametrize("method,params", [
    ("Page.enable", {}), ("DOM.enable", {}), ("Runtime.enable", {}), ("Network.enable", {}), ("Network.disable", {}),
    ("Page.navigate", {"url": "https://a.example/x?y=1"}), ("Page.navigate", {"url": "about:blank"}),
    ("Runtime.evaluate", {"expression": "document.title", "returnByValue": True, "awaitPromise": True}),
    ("Input.dispatchMouseEvent", {"type": "mousePressed", "x": 1, "y": 2, "button": "left", "clickCount": 1}),
    ("Input.dispatchKeyEvent", {"type": "keyDown", "key": "a", "text": "a"}), ("Input.insertText", {"text": "hi"}),
    ("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}),
    ("DOM.getDocument", {"depth": -1}), ("DOM.querySelector", {"nodeId": 1, "selector": "#q"}),
    ("DOM.getBoxModel", {"backendNodeId": 4}), ("Accessibility.getFullAXTree", {}),
    ("Page.handleJavaScriptDialog", {"accept": True}),
])
def test_what_browser_harness_needs_is_allowed(method, params):
    bridge.check_session_method(method, params)


def _harness_methods():
    """Every CDP method name the installed browser_harness sends (the allowlist's source)."""
    import re
    roots = [Path.home() / ".local/share/uv/tools/browser-use/lib/python3.11/site-packages/browser_harness"]
    root = next((r for r in roots if (r / "helpers.py").exists()), None)
    if root is None:
        return None
    text = (root / "helpers.py").read_text() + (root / "daemon.py").read_text()
    names = set(re.findall(r'"((?:Page|DOM|Runtime|Network|Input|Accessibility|Emulation|Fetch|Storage|Browser)\.[a-z][A-Za-z]+)"', text))
    names |= {f"{d}.enable" for d in re.findall(r'for d in \(([^)]*)\)', text) for d in re.findall(r'"(\w+)"', d)}
    return {n for n in names if not n.split(".")[1][0].isupper()} - {
        "Page.loadEventFired", "Page.domContentEventFired", "Page.javascriptDialogOpening",
        "Page.javascriptDialogClosed", "Network.requestWillBeSent", "Network.loadingFinished",
        "Network.loadingFailed"}  # events it listens to, not commands


def test_allowlist_covers_every_command_browser_harness_sends_except_upload():
    used = _harness_methods()
    if used is None:
        pytest.skip("browser_harness is not installed here")
    browser_level = {"Browser.getVersion"}  # answered by the relay itself, never forwarded (like Target.*)
    missing = sorted(m for m in used if m not in bridge._ALLOWED_SESSION_METHODS and m not in browser_level)
    assert missing == ["DOM.setFileInputFiles"], missing  # upload is refused on purpose


def test_create_target_applies_the_url_policy_before_the_extension_sees_it(host):
    _, ext, state = host
    url = json.loads(state.read_text())["ws_url"]
    bad = ["file:///etc/hosts", "chrome://settings", "chrome-extension://ljblhmlmgflmbffbfelamefleikodmjh/manifest.json",
           "javascript:alert(1)", "view-source:https://example.com/", "data:text/html,x"]

    async def go():
        async with connect(url) as ws:
            c = Client(ws)
            asyncio.ensure_future(c.pump())
            out = [await c.call("Target.createTarget", {"url": u}) for u in bad]
            good = await c.call("Target.createTarget", {"url": "https://example.com/"})
            return out, good

    out, good = _run(go())
    assert all("error" in r for r in out), out
    assert "result" in good
    assert [o["args"]["url"] for o in ext.ops if o["op"] == "create"] == ["https://example.com/"]


def test_activate_target_is_a_no_op_and_sessions_belong_to_their_socket(host):
    _, ext, state = host
    url = json.loads(state.read_text())["ws_url"]

    async def go():
        async with connect(url) as a, connect(url) as b:
            ca, cb = Client(a), Client(b)
            asyncio.ensure_future(ca.pump())
            asyncio.ensure_future(cb.pump())
            act = [await ca.call("Target.activateTarget", {"targetId": "T7"}) for _ in range(3)]
            sid = (await ca.call("Target.attachToTarget", {"targetId": "T7", "flatten": True}))["result"]["sessionId"]
            stolen = await cb.call("Runtime.evaluate", {"expression": "1"}, sid)
            await cb.call("Target.detachFromTarget", {"sessionId": sid})
            still = await ca.call("Runtime.evaluate", {"expression": "1"}, sid)
            return act, stolen, still

    act, stolen, still = _run(go())
    assert all(r.get("result") == {} for r in act)
    assert not [o for o in ext.ops if o["op"] == "activate"]
    assert "error" in stolen and "not found" in stolen["error"]["message"]
    assert "result" in still  # b's detach could not take a's session either


def test_with_the_lane_off_no_socket_exists_and_it_follows_the_config(tmp_path):
    _set_lane(tmp_path, False)
    proc, _ = _spawn(tmp_path)
    state = tmp_path / "chrome-bridge" / "bridge.json"
    try:
        assert not _wait(state.exists, timeout=3)
        _set_lane(tmp_path, True)
        assert _wait(state.exists, timeout=15)
        _set_lane(tmp_path, False)
        assert _wait(lambda: not state.exists(), timeout=15)
        assert proc.poll() is None  # still connected to the extension, just serving nothing
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


@pytest.mark.parametrize("url,ok", [
    ("https://a.example/", True), ("http://a.example", True), ("about:blank", True), ("about:blank#x", True),
    ("about:srcdoc", False), ("https://", False), ("file:///x", False), ("", False), (None, False),
    ("HTTPS://A.EXAMPLE/", True), ("chrome-untrusted://x", False), ("blob:https://a.example/1", False),
])
def test_url_allowed(url, ok):
    assert bridge.url_allowed(url) is ok
