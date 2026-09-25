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

Detection is by what the page shows (its text and its visible challenge frames), never a site list.
The text patterns are English; a non-English page still trips the challenge-frame check for CAPTCHAs,
and a passkey page in another language is not recognised (an honest gap, not a guess).

:func:`classify_text` reads whatever a tool printed; :data:`PROBE_JS` runs in the page itself (one
source of patterns for both); :func:`probe_pages` runs it over a local CDP endpoint's open tabs.
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

#: ``(kind, pattern)``, most specific first. Written in the subset of regex syntax Python and
#: JavaScript share (no lookbehind, no inline flags): the same strings run in the page (PROBE_JS).
PATTERNS = (
    # The site says a device is involved -- the only kind where "check your phone" is grounded.
    (DEVICE_PROMPT, r"check your (?:phone|device|iphone|android)|tap (?:yes|approve|allow) on your (?:phone|device)"
                    r"|approve (?:the |this )?(?:sign[- ]?in|request|login)[^.\n]{0,40} on your (?:phone|device|other device)"
                    r"|we sent a (?:notification|prompt|push)[^.\n]{0,40} to your|open the [a-z0-9 ]{1,30} app on your (?:phone|device)"),
    (PASSKEY, r"\bpasskey|security key|touch id|face id|windows hello|use your (?:fingerprint|face|screen lock)"),
    (CAPTCHA, r"verify (?:that )?you(?: are|['’]re) (?:a )?human|are you a (?:person or a )?robot|i['’]?m not a robot"
              r"|press (?:&|and) hold|complete the security check|solve (?:the|this) (?:puzzle|challenge)"),
    (IDENTITY_CHECK, r"(?:verify|confirm) it['’]?s (?:really )?you|(?:verify|confirm) your identity|identity verification"),
)
_COMPILED = tuple((kind, re.compile(p, re.I)) for kind, p in PATTERNS)

#: Visible challenge frames and widgets. The invisible reCAPTCHA badge (``size=invisible``) is on
#: countless ordinary login pages and is NOT a challenge; the checkbox (``size=normal|compact``), the
#: image challenge (``bframe``), hCaptcha, Turnstile, PerimeterX, DataDome and Arkose are.
CHALLENGE_FRAME = (r"recaptcha/(?:api2|enterprise)/(?:anchor[^\"']*size=(?:normal|compact)|bframe)|hcaptcha\.com"
                   r"|challenges\.cloudflare\.com|captcha-delivery\.com|arkoselabs\.com|funcaptcha\.com")
CHALLENGE_SELECTOR = "#px-captcha, .cf-turnstile, .h-captcha, [data-sitekey][class*=captcha i]"

_SAYS_LIMIT = 200


def _line_around(text: str, start: int) -> str:
    """The line of ``text`` holding ``start``, trimmed: what the page says, verbatim."""
    begin = text.rfind("\n", 0, start) + 1
    end = text.find("\n", start)
    line = " ".join(text[begin:end if end >= 0 else len(text)].split())
    return line[:_SAYS_LIMIT]


def classify_text(text: str) -> Optional[Dict[str, str]]:
    """``{"kind", "page_says"}`` for the first person-only step the text shows, else None."""
    if not text:
        return None
    for kind, pattern in _COMPILED:
        m = pattern.search(text)
        if m:
            return {"kind": kind, "page_says": _line_around(text, m.start())}
    return None


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
  const out = (kind, says) => JSON.stringify({kind, page_says: says, url: location.href, title: document.title});
  for (const [kind, src] of P) {
    const m = new RegExp(src, "i").exec(text);
    if (m) return out(kind, line(m.index));
  }
  const frame = Array.from(document.querySelectorAll("iframe")).find(
    (f) => visible(f) && new RegExp(__FRAME__, "i").test(f.src || ""));
  const widget = Array.from(document.querySelectorAll(__SELECTOR__)).find(visible);
  if (frame || widget) return out("captcha", "");
  return null;
})()"""


def probe_js() -> str:
    return (PROBE_JS.replace("__PATTERNS__", json.dumps([list(p) for p in PATTERNS]))
            .replace("__FRAME__", json.dumps(CHALLENGE_FRAME)).replace("__SELECTOR__", json.dumps(CHALLENGE_SELECTOR))
            .replace("__LIMIT__", str(_SAYS_LIMIT)))


def parse_probe(raw) -> Optional[Dict[str, str]]:
    """The probe's JSON (possibly double-encoded by a CDP layer) as a dict, or None."""
    for _ in range(2):
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None
    if isinstance(raw, dict) and raw.get("kind") in KINDS:
        return {k: str(raw.get(k) or "") for k in ("kind", "page_says", "url", "title")}
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


def probe_pages(cdp: str, *, limit: int = 3, timeout: float = 1.5) -> Optional[Dict[str, str]]:
    """Run :data:`PROBE_JS` in up to ``limit`` open tabs of a local browser; the first step found, or None.
    Best effort by design: a page that does not answer in ``timeout`` is skipped, never waited on."""
    expression = probe_js()
    for target in page_targets(cdp)[:limit]:
        if not target["ws"]:
            continue
        try:
            found = parse_probe(_evaluate(target["ws"], expression, timeout))
        except Exception as exc:
            logger.debug("person-step: probe of %s failed: %s", target["url"][:80], exc)
            continue
        if found:
            return found
    return None


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
