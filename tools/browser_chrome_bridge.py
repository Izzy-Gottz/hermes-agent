"""Native-messaging host for the Memoe Chrome extension: a CDP relay into Moe's own tab group.

Chrome starts this process when the extension (``extensions/chrome``) calls
``chrome.runtime.connectNative("app.memoe.chrome_bridge")``, and talks to it over
stdin/stdout with native-messaging framing (4-byte native-endian length + JSON).
Only an extension id listed in the host manifest's ``allowed_origins`` can start
it; a web page cannot reach it at all.

The host then serves a *browser-level* CDP WebSocket on ``127.0.0.1`` so the
existing ``browser_exec`` machinery (browser_harness → cdp_use) drives the
owner's real Chrome with no second way of thinking: ``page_info()``, ``goto_url``,
``click_at_xy``, ``type_text``/``fill_input``, ``js`` and ``capture_screenshot`` are
unchanged. The relay emulates the handful of browser-level ``Target.*`` calls the
harness makes and forwards page-session commands to ``chrome.debugger``:

* ``Target.getTargets`` lists ONLY tabs in Moe's group (the extension decides);
* ``Target.createTarget`` opens the tab in that group, in Moe's own window behind the person's
  (Chrome stalls never-shown background tabs), and passes :func:`check_url`;
* ``Target.activateTarget`` is a no-op: Moe's window works unfocused, and a turn must never be
  able to pull it in front of the person, once or in a loop;
* page-session commands pass :func:`check_session_method`, an ALLOWLIST of exactly the methods
  browser_harness uses (plus the two the tool description teaches the model), each with its
  parameters checked. Everything else — cookies, ``Storage.*``, ``Browser.*``, ``Fetch.*``,
  ``DOM.setFileInputFiles`` — is refused.

THIS RELAY IS THE SECURITY BOUNDARY, not ``browser.chrome_extension.enabled``. ``browser_exec``
runs arbitrary Python as the user; code that reads ``bridge.json`` can open the socket itself,
lane or no lane. So what the socket will do — the allowlist, the URL policy, the tab scope the
extension enforces — is what protects the person's browser. The config flag only decides whether
the socket exists at all: with the lane off, this host starts, finds it off, and serves nothing
(it rechecks every few seconds, so turning the lane on or off needs no Chrome restart).

Authentication of the local WebSocket: it binds loopback only, its path carries a
256-bit secret (``/devtools/browser/<token>``), it refuses any request carrying an
``Origin`` header (every browser page sends one on a WebSocket upgrade; the
harness never does), and the URL is published only in a 0600 file inside a 0700
directory under the Hermes home. Its lifetime is the extension's port: when Chrome
closes stdin the file is removed and the process exits.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import urlparse

NATIVE_HOST_NAME = "app.memoe.chrome_bridge"
BRIDGE_PROTOCOL = 1
_EXT_TIMEOUT_S = 60.0
_MAX_WS_FRAME = 64 * 1024 * 1024

_CONFIG_POLL_S = 5.0

#: Page-session methods the relay forwards, each with the parameter names it may carry. Derived by
#: grepping browser_harness 0.1.13 (helpers.py, daemon.py) for every CDP method it sends —
#: Page/DOM/Runtime/Network.enable, Network.disable, Page.navigate, Page.captureScreenshot,
#: Runtime.evaluate, Input.dispatchMouseEvent/dispatchKeyEvent/insertText, DOM.getDocument,
#: DOM.querySelector — plus Accessibility.getFullAXTree and DOM.getBoxModel, which browser_exec's
#: tool description teaches the model to call through cdp(), and Page.handleJavaScriptDialog, without
#: which a page's alert() freezes the tab. Not listed = refused.
#:
#: DOM.setFileInputFiles (the harness's upload_file) is refused, not gated: it hands a local file to
#: whatever page the tab is on, and the page decides what an upload means — so a prompt-injected
#: page in the person's signed-in session could exfiltrate any file. Upload stays in the own lane.
_ALLOWED_SESSION_METHODS: Dict[str, frozenset] = {
    "Page.enable": frozenset(),
    "DOM.enable": frozenset(),
    "Runtime.enable": frozenset(),
    "Network.enable": frozenset({"maxTotalBufferSize", "maxResourceBufferSize", "maxPostDataSize"}),
    "Network.disable": frozenset(),
    "Page.navigate": frozenset({"url", "referrer", "transitionType"}),
    "Page.captureScreenshot": frozenset({"format", "quality", "clip", "captureBeyondViewport", "fromSurface",
                                         "optimizeForSpeed"}),
    "Page.handleJavaScriptDialog": frozenset({"accept", "promptText"}),
    "Runtime.evaluate": frozenset({"expression", "returnByValue", "awaitPromise", "userGesture", "timeout",
                                   "silent", "generatePreview", "objectGroup", "throwOnSideEffect"}),
    "Input.dispatchMouseEvent": frozenset({"type", "x", "y", "button", "buttons", "clickCount", "deltaX", "deltaY",
                                           "modifiers", "pointerType", "force", "tangentialPressure", "tiltX",
                                           "tiltY", "twist"}),
    "Input.dispatchKeyEvent": frozenset({"type", "key", "code", "text", "unmodifiedText", "keyIdentifier",
                                         "windowsVirtualKeyCode", "nativeVirtualKeyCode", "modifiers",
                                         "autoRepeat", "isKeypad", "isSystemKey", "location", "commands"}),
    "Input.insertText": frozenset({"text"}),
    "DOM.getDocument": frozenset({"depth", "pierce"}),
    "DOM.querySelector": frozenset({"nodeId", "selector"}),
    "DOM.getBoxModel": frozenset({"nodeId", "backendNodeId", "objectId"}),
    "Accessibility.getFullAXTree": frozenset({"depth", "frameId"}),
}


class RelayError(Exception):
    """A CDP-shaped error returned to the relay client."""

    def __init__(self, message: str, code: int = -32000):
        super().__init__(message)
        self.code = code


def url_allowed(url: Any) -> bool:
    """The lane's one URL policy (mirrored in the extension's urlAllowed): http(s) and about:blank only."""
    if not isinstance(url, str):
        return False
    if url == "about:blank" or url.startswith("about:blank#") or url.startswith("about:blank?"):
        return True
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def check_url(url: Any) -> None:
    if not url_allowed(url):
        raise RelayError(f"{str(url)[:80]!r} is refused in the owner's Chrome: Moe's tabs open only http, https "
                         "and about:blank (no file:, chrome:, chrome-extension:, view-source:, javascript: or data:)",
                         -32602)


def check_session_method(method: str, params: Optional[dict]) -> None:
    """Raise RelayError unless a page-session CDP call is on the allowlist with allowed parameters."""
    if not isinstance(method, str) or "." not in method:
        raise RelayError(f"malformed CDP method {method!r}", -32600)
    allowed = _ALLOWED_SESSION_METHODS.get(method)
    if allowed is None:
        raise RelayError(f"{method} is not available in the Chrome-extension lane (the owner's real browser): only "
                         "page reading, navigation, input and screenshots are", -32601)
    params = params or {}
    if not isinstance(params, dict):
        raise RelayError(f"{method} params must be an object", -32602)
    extra = set(params) - allowed
    if extra:
        raise RelayError(f"{method} parameter(s) {sorted(extra)} are not available in the Chrome-extension lane", -32602)
    if method == "Page.navigate":
        check_url(params.get("url"))


# ---- native messaging framing -------------------------------------------------

def encode_native_message(obj: Any) -> bytes:
    data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return struct.pack("=I", len(data)) + data


def read_native_message(stream) -> Optional[Any]:
    header = stream.read(4)
    if not header or len(header) < 4:
        return None
    (length,) = struct.unpack("=I", header)
    data = stream.read(length)
    if len(data) < length:
        return None
    return json.loads(data.decode("utf-8"))


class ExtensionLink:
    """Request/response + event channel to the extension over stdio."""

    def __init__(self, loop: asyncio.AbstractEventLoop, reader, writer):
        self._loop = loop
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self.on_event = None  # callable(dict)
        self.closed = asyncio.Event()

    def start(self) -> None:
        threading.Thread(target=self._read_loop, name="chrome-bridge-stdin", daemon=True).start()

    def _read_loop(self) -> None:
        try:
            while True:
                msg = read_native_message(self._reader)
                if msg is None:
                    break
                self._loop.call_soon_threadsafe(self._dispatch, msg)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._close)

    def _close(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RelayError("the Chrome extension disconnected"))
        self._pending.clear()
        self.closed.set()

    def _dispatch(self, msg: dict) -> None:
        if "event" in msg:
            if self.on_event:
                self.on_event(msg)
            return
        fut = self._pending.pop(msg.get("id"), None)
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg.get("error") or {}
            fut.set_exception(RelayError(str(err.get("message") or err), -32000))
        else:
            fut.set_result(msg.get("result"))

    async def request(self, op: str, **args) -> Any:
        if self.closed.is_set():
            raise RelayError("the Chrome extension disconnected")
        self._next_id += 1
        rid = self._next_id
        fut = self._loop.create_future()
        self._pending[rid] = fut
        frame = encode_native_message({"id": rid, "op": op, "args": args})
        with self._write_lock:
            self._writer.write(frame)
            self._writer.flush()
        try:
            return await asyncio.wait_for(fut, _EXT_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise RelayError(f"the Chrome extension did not answer {op} within {_EXT_TIMEOUT_S:.0f}s") from None
        finally:
            self._pending.pop(rid, None)


class CDPRelay:
    """Browser-level CDP endpoint whose only pages are the tabs in Moe's group."""

    def __init__(self, link: ExtensionLink, browser_info: Optional[dict] = None):
        self.link = link
        self.browser_info = browser_info or {}
        # sessionId -> (client, tabId, targetId)
        self.sessions: Dict[str, Tuple[Any, int, str]] = {}
        self.clients: Set[Any] = set()
        self.log_path: Optional[Path] = None
        link.on_event = self.on_extension_event

    # -- target bookkeeping
    async def _targets(self) -> list:
        return list(await self.link.request("targets") or [])

    async def _tab_for_target(self, target_id: str) -> dict:
        for t in await self._targets():
            if t.get("targetId") == target_id:
                return t
        raise RelayError(f"No target with given id found (only tabs in Moe's group are visible): {target_id}", -32602)

    @staticmethod
    def _target_info(t: dict) -> dict:
        return {"targetId": t["targetId"], "type": "page", "title": t.get("title", ""), "url": t.get("url", ""),
                "attached": bool(t.get("attached")), "canAccessOpener": False, "browserContextId": "moe"}

    async def handle_browser_method(self, client: Any, method: str, params: dict) -> dict:
        if method == "Browser.getVersion":
            ua = self.browser_info.get("userAgent", "")
            product = next((p for p in ua.split() if p.startswith("Chrome/")), "Chrome")
            return {"protocolVersion": "1.3", "product": product, "revision": "", "userAgent": ua, "jsVersion": ""}
        if method == "Target.getTargets":
            return {"targetInfos": [self._target_info(t) for t in await self._targets()]}
        if method == "Target.createTarget":
            # The client's background flag is ignored: a new Moe tab never takes the person's focus.
            url = params.get("url") or "about:blank"
            check_url(url)
            created = await self.link.request("create", url=url)
            return {"targetId": created["targetId"]}
        if method == "Target.attachToTarget":
            if params.get("flatten") is not True:
                raise RelayError("only flattened sessions are supported", -32602)
            t = await self._tab_for_target(str(params.get("targetId") or ""))
            await self.link.request("attach", tabId=t["tabId"])
            sid = secrets.token_hex(16).upper()
            self.sessions[sid] = (client, t["tabId"], t["targetId"])
            return {"sessionId": sid}
        if method == "Target.detachFromTarget":
            sid = str(params.get("sessionId") or "")
            if sid in self.sessions and self.sessions[sid][0] is client:
                await self._drop_session(sid)
            return {}
        if method == "Target.closeTarget":
            t = await self._tab_for_target(str(params.get("targetId") or ""))
            await self.link.request("close", tabId=t["tabId"])
            return {"success": True}
        if method == "Target.activateTarget":
            # No-op by design: Moe's window works unfocused and must never jump in front of the person.
            await self._tab_for_target(str(params.get("targetId") or ""))
            return {}
        if method == "Target.getTargetInfo":
            tid = params.get("targetId")
            if not tid:
                return {"targetInfo": {"targetId": "browser", "type": "browser", "title": "", "url": "",
                                       "attached": True, "canAccessOpener": False}}
            return {"targetInfo": self._target_info(await self._tab_for_target(str(tid)))}
        if method in ("Target.setDiscoverTargets", "Target.setAutoAttach"):
            return {}
        raise RelayError(f"{method} is not available in the Chrome-extension lane", -32601)

    async def handle_session_method(self, client: Any, session_id: str, method: str, params: dict) -> dict:
        entry = self.sessions.get(session_id)
        if entry is None or entry[0] is not client:  # a session belongs to the socket that attached it
            raise RelayError(f"Session with given id not found: {session_id}", -32001)
        check_session_method(method, params)
        return await self.link.request("cdp", tabId=entry[1], method=method, params=params) or {}

    async def _drop_session(self, session_id: str) -> None:
        entry = self.sessions.pop(session_id, None)
        if entry is None:
            return
        tab_id = entry[1]
        if not any(tab == tab_id for _, tab, _ in self.sessions.values()):
            try:
                await self.link.request("detach", tabId=tab_id)
            except RelayError:
                pass

    async def handle_message(self, client: Any, raw: str) -> Optional[dict]:
        try:
            msg = json.loads(raw)
        except ValueError:
            return None
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        sid = msg.get("sessionId")
        reply: Dict[str, Any] = {"id": mid}
        if sid:
            reply["sessionId"] = sid
        try:
            if sid:
                reply["result"] = await self.handle_session_method(client, sid, method, params)
            else:
                reply["result"] = await self.handle_browser_method(client, method, params)
        except RelayError as e:
            reply.pop("result", None)
            reply["error"] = {"code": e.code, "message": str(e)}
        return reply

    # -- events from the extension
    def on_extension_event(self, msg: dict) -> None:
        kind, tab_id = msg.get("event"), msg.get("tabId")
        if kind == "log":
            if self.log_path is not None:
                try:
                    if not self.log_path.exists() or self.log_path.stat().st_size < 1_000_000:
                        with open(self.log_path, "a", encoding="utf-8") as fh:
                            fh.write(f"{time.strftime('%H:%M:%S')} {str(msg.get('text'))[:500]}\n")
                except OSError:
                    pass
            return
        if kind == "cdp":
            for sid, (client, tab, _) in list(self.sessions.items()):
                if tab == tab_id:
                    self._send(client, {"method": msg.get("method"), "params": msg.get("params") or {}, "sessionId": sid})
        elif kind in ("detached", "released"):
            for sid, (client, tab, target_id) in list(self.sessions.items()):
                if tab == tab_id:
                    self.sessions.pop(sid, None)
                    self._send(client, {"method": "Target.detachedFromTarget",
                                        "params": {"sessionId": sid, "targetId": target_id}})

    def _send(self, client: Any, obj: dict) -> None:
        asyncio.ensure_future(_safe_send(client, json.dumps(obj)))

    def forget_client(self, client: Any) -> None:
        self.clients.discard(client)
        for sid, (owner, _, _) in list(self.sessions.items()):
            if owner is client:
                asyncio.ensure_future(self._drop_session(sid))


async def _safe_send(ws, text: str) -> None:
    try:
        await ws.send(text)
    except Exception:
        pass


def bridge_state_dir(hermes_home: Optional[str] = None) -> Path:
    home = hermes_home or os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "chrome-bridge"


def write_bridge_state(state_dir: Path, state: dict) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)
    path = state_dir / "bridge.json"
    tmp = state_dir / f".bridge.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)
    return path


def make_request_check(token: str):
    """websockets ``process_request`` hook: exact secret path, and no browser Origin."""
    expected_path = f"/devtools/browser/{token}"

    def check(connection, request):
        if request.headers.get("Origin") is not None:
            return connection.respond(403, "browser pages may not connect to the Moe Chrome bridge\n")
        if not secrets.compare_digest(request.path.encode(), expected_path.encode()):
            return connection.respond(404, "not found\n")
        return None

    return check


def lane_enabled_in_config(hermes_home: Optional[str] = None) -> bool:
    """``browser.chrome_extension.enabled`` from the Hermes config, literal True only; unreadable = off."""
    home = Path(hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    try:
        import yaml
        cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        return (cfg.get("browser") or {}).get("chrome_extension", {}).get("enabled") is True
    except Exception:
        return False


async def serve_bridge(link: ExtensionLink, extension_origin: str, state_dir: Path) -> None:
    """Serve the socket only while the lane is enabled; with it off, no socket and no bridge.json exist."""
    info = await link.request("hello")
    while not link.closed.is_set():
        if lane_enabled_in_config(str(state_dir.parent)):
            await _serve_socket(link, info, extension_origin, state_dir)
        else:
            try:
                await asyncio.wait_for(link.closed.wait(), _CONFIG_POLL_S)
            except asyncio.TimeoutError:
                pass


async def _serve_socket(link: ExtensionLink, info: dict, extension_origin: str, state_dir: Path) -> None:
    from websockets.asyncio.server import serve

    relay = CDPRelay(link, info)
    relay.log_path = state_dir / "bridge.log"
    token = secrets.token_urlsafe(32)

    async def handler(ws):
        relay.clients.add(ws)

        async def answer(raw):
            reply = await relay.handle_message(ws, raw)
            if reply is not None:
                await _safe_send(ws, json.dumps(reply))

        try:
            # CDP clients pipeline requests (the harness enables four domains at once), so each
            # message is its own task: one slow command never stalls the rest of the socket.
            async for raw in ws:
                asyncio.ensure_future(answer(raw))
        finally:
            relay.forget_client(ws)

    async with serve(handler, "127.0.0.1", 0, process_request=make_request_check(token),
                     max_size=_MAX_WS_FRAME, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]
        state = {"protocol": BRIDGE_PROTOCOL, "ws_url": f"ws://127.0.0.1:{port}/devtools/browser/{token}",
                 "pid": os.getpid(), "extension_origin": extension_origin, "extension": info,
                 "started_at": time.time()}
        path = write_bridge_state(state_dir, state)
        try:
            while not link.closed.is_set() and lane_enabled_in_config(str(state_dir.parent)):
                try:
                    await asyncio.wait_for(link.closed.wait(), _CONFIG_POLL_S)
                except asyncio.TimeoutError:
                    pass
        finally:
            for ws in list(relay.clients):
                try:
                    await ws.close()
                except Exception:
                    pass
            for sid in list(relay.sessions):
                await relay._drop_session(sid)
            try:
                current = json.loads(path.read_text())
                if current.get("pid") == os.getpid():
                    path.unlink()
            except Exception:
                pass


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Chrome passes the calling extension's origin as the first argument.
    origin = next((a for a in argv if a.startswith("chrome-extension://")), "")
    state_dir = bridge_state_dir()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    link = ExtensionLink(loop, sys.stdin.buffer, sys.stdout.buffer)
    link.start()
    try:
        loop.run_until_complete(serve_bridge(link, origin, state_dir))
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
