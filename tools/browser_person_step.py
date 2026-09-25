"""A page step only the person can take: a passkey, a CAPTCHA, an identity check -- and what the page says.

Moe's own browser runs out of sight. On 2026-09-24 a Google sign-in in it reached "Use your passkey to
confirm it's you": a sheet that the page's own browser raises on the Mac, which nobody could see. The
model told the owner, three times, to "approve it on your phone". Nothing had been sent to a phone; the
words came from a tool hint and a skill, not from the page.

So the engine looks at the page itself, for every site, and says two things:

* ``kind`` -- what the step is: ``passkey`` (WebAuthn / security key / Touch ID), ``captcha`` (a visible
  challenge, not the invisible badge half the web loads), ``identity_check`` ("verify it's you"), or
  ``device_prompt`` -- the one kind where the SITE ITSELF says a phone or another device is involved.
* ``page_says`` -- the page's own line, verbatim. A reply may repeat it; it may not add to it. Only
  ``device_prompt`` makes "check your phone" a true thing to say.

Detection is by what the page is DOING, never a site list and never the bare word (see the block
above IMPERATIVE): a WebAuthn request in flight (the hook the fidelity keeper installs in every page),
a visible challenge frame, or imperative challenge phrasing on a short page with no form to fill.
The phrasing is English; a non-English page is caught only by the hook or a challenge frame (an
honest gap, not a guess).

:func:`suggests_person_step` decides from a tool's output whether the page is worth a look;
:func:`probe_active_page` runs :data:`PROBE_JS` in the active tab, within a tight budget;
:func:`classify_page` is the same decision in Python.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PASSKEY = "passkey"
CAPTCHA = "captcha"
IDENTITY_CHECK = "identity_check"
DEVICE_PROMPT = "device_prompt"
KINDS = (DEVICE_PROMPT, PASSKEY, CAPTCHA, IDENTITY_CHECK)

# ── The verdict: what the page is DOING ────────────────────────────────────────
#
# Measured by review, 2026-09-25: the bare word is not a challenge. github.com/login says "Sign in with
# a passkey" as ONE option beside a password form; a Google help article and Wikipedia's "Passkey" page
# are full of the word. A step only the person can take is a page that (1) has asked the browser for a
# passkey right now -- a WebAuthn call in flight, seen by the hook below -- or (2) tells them, in the
# imperative, to do one thing, and offers no other way forward: no form to fill, a short page.

#: ``(kind, pattern)``, most specific first: imperative challenge phrasing only. Written in the regex
#: subset Python and JavaScript share (no lookbehind, no inline flags): PROBE_JS runs the same strings.
IMPERATIVE = (
    # The site says a device is involved -- the only kind where "check your phone" is grounded.
    (DEVICE_PROMPT, r"check your (?:phone|device|iphone|android)\b|tap (?:yes|approve|allow) on your (?:phone|device)"
                    r"|approve (?:the |this )?(?:sign[- ]?in|request|login)[^.\n]{0,40} on your (?:phone|device|other device)"
                    r"|we sent a (?:notification|prompt|push)[^.\n]{0,40} to your"),
    (PASSKEY, r"use your (?:passkey|security key|fingerprint|face|screen lock|device) to (?:confirm|verify|sign in|continue|log in)"
              r"|insert your security key|touch your security key|confirm it['’]?s (?:really )?you with your passkey"),
    (CAPTCHA, r"verify (?:that )?you(?: are|['’]re) (?:a )?human|are you a (?:person or a )?robot|press (?:&|and) hold"
              r"|complete the security check|solve (?:the|this) (?:puzzle|challenge)"),
    (IDENTITY_CHECK, r"^(?:verify|confirm) it['’]?s (?:really )?you|^(?:verify|confirm) your identity"),
)
#: The old name: tests and callers that want the verdict patterns.
PATTERNS = IMPERATIVE
_COMPILED = tuple((kind, re.compile(p, re.I | re.M)) for kind, p in IMPERATIVE)

#: A challenge page is short; an article or a dashboard is not. Visible text above this is not one.
SHORT_PAGE_CHARS = 1500

#: Visible challenge frames and widgets. The invisible reCAPTCHA badge (``size=invisible``) is on
#: countless ordinary login pages and is NOT a challenge; the checkbox (``size=normal|compact``), the
#: image challenge (``bframe``), hCaptcha, Turnstile, PerimeterX, DataDome and Arkose are.
CHALLENGE_FRAME = (r"recaptcha/(?:api2|enterprise)/(?:anchor[^\"']*size=(?:normal|compact)|bframe)|hcaptcha\.com"
                   r"|challenges\.cloudflare\.com|captcha-delivery\.com|arkoselabs\.com|funcaptcha\.com")
CHALLENGE_SELECTOR = "#px-captcha, .cf-turnstile, .h-captcha, [data-sitekey][class*=captcha i]"

# ── The trigger: output that suggests a stall, so the page is worth a look ────────
#
# Cheap and loose on purpose: it only decides whether to spend one probe of the active tab. It is
# never a verdict, and never reaches the person.
_STALL_HINT = re.compile(
    r"passkey|security key|webauthn|captcha|robot|human|verify it|confirm it|verify your|identity|"
    r"check your (?:phone|device)|two[- ]step|2-step|/challenge|/signin/|press (?:&|and) hold|just a moment", re.I)

_SAYS_LIMIT = 200


def suggests_person_step(text: str) -> bool:
    """Whether a tool's output hints that the page may be waiting on the person (worth a probe)."""
    return bool(text) and bool(_STALL_HINT.search(text))


def _line_around(text: str, start: int) -> str:
    """The line of ``text`` holding ``start``, trimmed: what the page says, verbatim."""
    begin = text.rfind("\n", 0, start) + 1
    end = text.find("\n", start)
    line = " ".join(text[begin:end if end >= 0 else len(text)].split())
    return line[:_SAYS_LIMIT]


def classify_page(text: str, *, fillable_inputs: int = 0, webauthn_pending: bool = False) -> Optional[Dict[str, str]]:
    """The verdict for one page from what it shows: ``{"kind", "page_says"}`` or None. Python twin of
    PROBE_JS's decision (tests hold them to the same fixtures)."""
    text = text or ""
    if webauthn_pending:
        found = next(((k, m) for k, p in _COMPILED if (m := p.search(text))), None)
        return {"kind": PASSKEY, "page_says": _line_around(text, found[1].start()) if found else ""}
    if len(text) > SHORT_PAGE_CHARS or fillable_inputs > 0:
        return None
    for kind, pattern in _COMPILED:
        m = pattern.search(text)
        if m:
            return {"kind": kind, "page_says": _line_around(text, m.start())}
    return None


#: Installed in every page of the driven browser before its scripts run (the fidelity keeper,
#: tools/browser_tool_fidelity.py): it notes a WebAuthn request in flight. ``mediation:
#: "conditional"`` is passkey AUTOFILL (github.com/login offers it quietly beside the password form)
#: and is not a request. A Proxy keeps ``toString`` native, so the page sees nothing new.
WEBAUTHN_HOOK_JS = """(() => {
  const K = Symbol.for("hermes.webauthn");
  if (window[K]) return;
  const st = {pending: 0, last: null};
  try { Object.defineProperty(window, K, {value: st, enumerable: false}); } catch (e) { return; }
  const C = window.CredentialsContainer && window.CredentialsContainer.prototype;
  if (!C) return;
  for (const op of ["get", "create"]) {
    const orig = C[op];
    if (typeof orig !== "function") continue;
    const wrapped = new Proxy(orig, {apply(target, self, args) {
      const o = (args && args[0]) || {};
      const p = Reflect.apply(target, self, args);
      if (!o.publicKey || o.mediation === "conditional") return p;
      st.pending++; st.last = {op, at: Date.now()};
      const done = () => { st.pending = Math.max(0, st.pending - 1); };
      try { Promise.resolve(p).then(done, done); } catch (e) { done(); }
      return p;
    }});
    try { Object.defineProperty(C, op, {value: wrapped, writable: true, configurable: true, enumerable: true}); }
    catch (e) { /* leave the page alone */ }
  }
})();"""


PROBE_JS = """(() => {
  const P = __PATTERNS__;
  const text = ((document.body && document.body.innerText) || "").slice(0, 20000);
  const visible = (el) => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 30 && r.height > 30 && s.visibility !== "hidden" && s.display !== "none";
  };
  const line = (i) => {
    const b = text.lastIndexOf("\\n", i) + 1; let e = text.indexOf("\\n", i); if (e < 0) e = text.length;
    return text.slice(b, e).replace(/\\s+/g, " ").trim().slice(0, __LIMIT__);
  };
  const out = (kind, says, why) => JSON.stringify({kind, page_says: says, why, url: location.href, title: document.title});
  const hook = window[Symbol.for("hermes.webauthn")];
  const find = () => { for (const [kind, src] of P) { const m = new RegExp(src, "im").exec(text); if (m) return [kind, m]; } return null; };
  if (hook && hook.pending > 0) { const f = find(); return out("passkey", f ? line(f[1].index) : "", "webauthn"); }
  const frame = Array.from(document.querySelectorAll("iframe")).find(
    (f) => visible(f) && new RegExp(__FRAME__, "i").test(f.src || ""));
  const widget = Array.from(document.querySelectorAll(__SELECTOR__)).find(visible);
  if (frame || widget) return out("captcha", "", "challenge_frame");
  // Another way forward: any visible field to fill (a password form beside "sign in with a passkey").
  const fillable = Array.from(document.querySelectorAll("input, textarea, select")).filter((el) => {
    const t = (el.type || "").toLowerCase();
    if (["hidden", "submit", "button", "reset", "image", "checkbox", "radio", "search"].includes(t)) return false;
    return visible(el) || (el.getBoundingClientRect().width > 0 && el.getBoundingClientRect().height > 0);
  }).length;
  if (text.length > __SHORT__ || fillable > 0) return null;
  const f = find();
  return f ? out(f[0], line(f[1].index), "imperative") : null;
})()"""


def probe_js() -> str:
    return (PROBE_JS.replace("__PATTERNS__", json.dumps([list(p) for p in IMPERATIVE]))
            .replace("__FRAME__", json.dumps(CHALLENGE_FRAME)).replace("__SELECTOR__", json.dumps(CHALLENGE_SELECTOR))
            .replace("__LIMIT__", str(_SAYS_LIMIT)).replace("__SHORT__", str(SHORT_PAGE_CHARS)))


def parse_probe(raw) -> Optional[Dict[str, str]]:
    """The probe's JSON (possibly double-encoded by a CDP layer) as a dict, or None."""
    for _ in range(2):
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None
    if isinstance(raw, dict) and raw.get("kind") in KINDS:
        return {k: str(raw.get(k) or "") for k in ("kind", "page_says", "why", "url", "title")}
    return None


def _local_http_cdp(cdp: str) -> Optional[str]:
    """``http://127.0.0.1:<port>`` for a LOCAL http CDP root, else None (a cloud endpoint is never probed)."""
    try:
        u = urlparse(cdp or "")
    except ValueError:
        return None
    if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost") or not u.port:
        return None
    return f"http://127.0.0.1:{u.port}"


def page_targets(cdp: str, timeout: float = 2.0) -> List[Dict[str, str]]:
    """Open http(s) tabs of a local browser, as ``/json/list`` orders them (``id``, ``url``, ``title``, ``ws``)."""
    root = _local_http_cdp(cdp)
    if not root:
        return []
    import requests
    try:
        listed = requests.get(f"{root}/json/list", timeout=timeout).json()
    except Exception as exc:
        logger.debug("person-step: /json/list failed: %s", exc)
        return []
    return [{"id": str(t.get("id") or ""), "url": str(t.get("url") or ""), "title": str(t.get("title") or ""),
             "ws": str(t.get("webSocketDebuggerUrl") or "")}
            for t in listed if isinstance(t, dict) and t.get("type") == "page"
            and str(t.get("url") or "").startswith(("http://", "https://"))]


def _evaluate(ws_url: str, expression: str, timeout: float) -> object:
    from websockets.sync.client import connect
    with connect(ws_url, max_size=None, open_timeout=timeout, close_timeout=1) as ws:
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                            "params": {"expression": expression, "returnByValue": True}}))
        while True:
            msg = json.loads(ws.recv(timeout=timeout))
            if msg.get("id") == 1:
                return ((msg.get("result") or {}).get("result") or {}).get("value")


def probe_active_page(cdp: str, *, budget: float = 0.8) -> Optional[Dict[str, str]]:
    """Run :data:`PROBE_JS` in the ACTIVE tab of a local browser (``/json/list``'s first page: Chrome
    lists the most recently activated tab first, measured 2026-09-25), within ``budget`` seconds all
    told. Best effort by design: a page that does not answer in time is not waited on."""
    import time as _time
    deadline = _time.monotonic() + budget
    pages = page_targets(cdp, timeout=max(0.1, budget / 2))
    if not pages or not pages[0]["ws"]:
        return None
    left = deadline - _time.monotonic()
    if left <= 0.05:
        return None
    try:
        return parse_probe(_evaluate(pages[0]["ws"], probe_js(), left))
    except Exception as exc:
        logger.debug("person-step: probe of %s failed: %s", pages[0]["url"][:80], exc)
        return None


#: Kept for callers written against the first version; now the active tab only.
def probe_pages(cdp: str, *, limit: int = 1, timeout: float = 0.8) -> Optional[Dict[str, str]]:
    return probe_active_page(cdp, budget=timeout)


def host_of(url: str) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower()
    except ValueError:
        return ""


def describe(step: Dict[str, str]) -> str:
    """Plain words for what the page needs, never adding to what it says."""
    kind = step.get("kind")
    what = {PASSKEY: "a passkey or security key", CAPTCHA: "a CAPTCHA / 'are you human' check",
            IDENTITY_CHECK: "an identity check ('verify it's you')",
            DEVICE_PROMPT: "an approval the page says is on another device"}.get(kind or "", "a step only the person can do")
    says = step.get("page_says") or ""
    return f"{what}" + (f' -- the page says: "{says}"' if says else "")


GROUNDING_RULE = ("Only tell the person that something was sent, or is waiting on their phone or another device, "
                  "when a tool result shows the site said so (page_says). Otherwise say what the page shows.")
