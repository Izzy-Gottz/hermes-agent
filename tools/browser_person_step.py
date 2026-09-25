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

Detection is a HINT by what the page SAYS AND OFFERS, never a site list and never the bare word (see
the block above IMPERATIVE): imperative challenge wording, in a challenge context, with no field on
screen to fill and no article around it -- or a visible challenge frame. English first, a few French,
German, Spanish and Hebrew lines; other languages are not recognised. No script is injected into pages
(the note above PROBE_JS says why). The rule the model follows is the real path: a check only the
person can complete is browser_handoff, whether or not this note fired.

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
# Detection is a HINT, not the rule. The rule the model is held to -- in the soul and in every tool
# description -- is "a check only the person can complete means browser_handoff"; this only puts a
# `needs_person` note on a result so the model does not miss one. It is tuned for precision: a note
# on an ordinary page costs the person an interruption, a missed one only leaves the rule to act.
#
# Measured by review, 2026-09-25: the bare word is not a challenge. github.com/login says "Sign in with
# a passkey" as ONE option beside a password form; a Google help article and Wikipedia's "Passkey" page
# are full of the word; a cart page can say "verify your identity". A page counts only when ALL hold:
#   1. imperative challenge wording (IMPERATIVE; English first, a few French, German, Spanish and
#      Hebrew lines -- other languages are not recognised, an honest gap);
#   2. challenge context: the title or a heading reads like a sign-in / verification step, or a
#      button offers the way out challenge pages offer ("Try another way", "Use another method",
#      "Continue"); an identity check needs the title or heading;
#   3. no other way forward: no field a person could fill that is really on screen (see the
#      visibility rules in PROBE_JS: hidden, off-screen, transparent, clipped, aria-hidden, disabled,
#      read-only fields, search boxes and <select> language pickers do not count);
#   4. not an article: short visible text, no <article>, not paragraph after paragraph.

#: ``(kind, pattern)``, most specific first: imperative challenge phrasing only. Written in the regex
#: subset Python and JavaScript share (no lookbehind, no inline flags): PROBE_JS runs the same strings.
IMPERATIVE = (
    # The site says a device is involved -- the only kind where "check your phone" is grounded.
    (DEVICE_PROMPT, r"check your (?:phone|device|iphone|android)\b|tap (?:yes|approve|allow) on your (?:phone|device)"
                    r"|approve (?:the |this )?(?:sign[- ]?in|request|login)[^.\n]{0,40} on your (?:phone|device|other device)"
                    r"|we sent a (?:notification|prompt|push)[^.\n]{0,40} to your"
                    r"|(?:verification )?code (?:has been |was )?sent to your (?:trusted )?(?:apple )?devices"
                    r"|allow (?:this )?sign[- ]?in on your (?:iphone|ipad|mac)"
                    r"|consultez votre (?:téléphone|appareil)|prüfen sie ihr (?:telefon|smartphone|gerät)"
                    r"|revisa tu (?:teléfono|dispositivo)|בדק(?:ו|י)? את הטלפון"),
    (PASSKEY, r"use your (?:passkey|security key|fingerprint|face|screen lock|device) to (?:confirm|verify|sign in|continue|log in)"
              r"|use your face, fingerprint, pin,? or security key|use touch id or your passkey|use your passkey\b"
              r"|authenticate with (?:a|your) passkey|sign in with your passkey|insert your security key"
              r"|touch your security key|confirm it['’]?s (?:really )?you with your passkey"
              r"|utilisez votre (?:clé d['’]accès|clé de sécurité)|verwenden sie ihren (?:passkey|sicherheitsschlüssel)"
              r"|usa (?:tu|su) (?:llave de acceso|clave de acceso|llave de seguridad)|השתמש(?:ו|י)? במפתח (?:הגישה|אבטחה)"),
    (CAPTCHA, r"verify (?:that )?you(?: are|['’]re) (?:a )?human|are you a (?:person or a )?robot|press (?:&|and) hold"
              r"|complete the security check|solve (?:the|this) (?:puzzle|challenge)"),
    (IDENTITY_CHECK, r"^(?:verify|confirm) it['’]?s (?:really )?you|^(?:verify|confirm) your identity"
                     r"|^(?:confirmez|vérifiez) (?:qu['’]il s['’]agit bien de vous|votre identité)"
                     r"|^(?:bestätigen sie, dass sie es sind|identität bestätigen|bestätigen sie ihre identität)"
                     r"|^(?:confirma que eres tú|verifica tu identidad)|^(?:אמת(?:ו|י)? שזה את(?:ה|ם|ן)|אימות זהות)"),
)
#: The old name: tests and callers that want the verdict patterns.
PATTERNS = IMPERATIVE
_COMPILED = tuple((kind, re.compile(p, re.I | re.M)) for kind, p in IMPERATIVE)

#: Title / heading wording of a sign-in or verification step.
CONTEXT_TITLE = (r"sign[- ]?in|log[- ]?in|verif|confirm|security|2-step|two[- ]step|two[- ]factor|authenticat|passkey"
                 r"|it['’]?s you|identity|apple account|apple id|microsoft account|connexion|vérif|anmeld|bestätig"
                 r"|sicherheit|iniciar sesión|verific|התחבר|אימות|כניסה")
#: The way out a challenge page offers, as a whole button label.
CONTEXT_BUTTON = (r"^(?:continue|next|try another way|use another (?:method|way|option)|use a different (?:method|option)"
                  r"|more ways to (?:verify|sign in)|sign in another way|other ways to sign in|cancel|continuer"
                  r"|essayer une autre méthode|weiter|andere methode|continuar|probar otra manera|המשך|נסה דרך אחרת)$")
_CONTEXT_TITLE = re.compile(CONTEXT_TITLE, re.I)
_CONTEXT_BUTTON = re.compile(CONTEXT_BUTTON, re.I)

#: A challenge page is short; an article or a dashboard is not. Visible text above this is not one.
SHORT_PAGE_CHARS = 1500
#: Paragraphs longer than this, this many times over, are an article.
ARTICLE_PARAGRAPH_CHARS = 80
ARTICLE_PARAGRAPHS = 3

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


def classify_page(text: str, *, fillable_inputs: int = 0, title: str = "", headings=(), buttons=(),
                  article: bool = False) -> Optional[Dict[str, str]]:
    """The verdict for one page: ``{"kind", "page_says"}`` or None. Python twin of PROBE_JS's decision
    (tests hold them to the same fixtures). ``fillable_inputs`` counts only fields really on screen."""
    text = text or ""
    if article or len(text) > SHORT_PAGE_CHARS or fillable_inputs > 0:
        return None
    titled = any(_CONTEXT_TITLE.search(t or "") for t in [title, *headings])
    buttoned = any(_CONTEXT_BUTTON.search(" ".join(str(b or "").split())) for b in buttons)
    for kind, pattern in _COMPILED:
        m = pattern.search(text)
        if not m:
            continue
        if titled or (buttoned and kind != IDENTITY_CHECK):
            return {"kind": kind, "page_says": _line_around(text, m.start())}
        return None
    return None


# No page script. A WebAuthn hook (wrapping navigator.credentials.get/create) was built and dropped,
# 2026-09-25: V8 prints a Proxy as "function () { [native code] }", not "function get() { [native
# code] }", so any page can tell; a throwing ``publicKey`` getter would show the wrapper's frame in its
# stack; and an init script on every document is one more thing a site can fingerprint. The CDP
# WebAuthn domain is no substitute: enabling it routes every request to virtual authenticators, which
# would break the person's real passkey. So the verdict comes from what the page SAYS AND OFFERS: the
# imperative wording on a short page with nothing to fill, or a visible challenge frame.

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
  const frame = Array.from(document.querySelectorAll("iframe")).find(
    (f) => visible(f) && new RegExp(__FRAME__, "i").test(f.src || ""));
  const widget = Array.from(document.querySelectorAll(__SELECTOR__)).find(visible);
  if (frame || widget) return out("captcha", "", "challenge_frame");
  // A field a person could fill, really on screen. Hidden-by-design fields (Google keeps its email
  // field on the passkey step, off screen) are not another way forward.
  const skipTypes = ["hidden", "submit", "button", "reset", "image", "checkbox", "radio", "search", "file", "range", "color"];
  const onScreen = (el) => {
    if (el.disabled || el.readOnly) return false;
    if (el.tagName === "SELECT") return false;                          // language pickers
    const t = (el.type || "").toLowerCase();
    if (skipTypes.includes(t)) return false;
    const attr = (n) => (el.getAttribute && el.getAttribute(n)) || "";
    if (attr("role") === "searchbox" || /search|query/i.test([el.name, el.id, attr("aria-label"), el.placeholder].join(" ")))
      return false;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;                        // sr-only, 1x1 clipped
    if (r.right + (window.scrollX || 0) <= 0 || r.bottom + (window.scrollY || 0) <= 0) return false;  // off screen
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      const s = getComputedStyle(n);
      if (s.display === "none" || s.visibility === "hidden" || s.visibility === "collapse") return false;
      if (parseFloat(s.opacity) === 0) return false;
      if (n.getAttribute && n.getAttribute("aria-hidden") === "true") return false;
      if (s.clip && /rect\\(\\s*0(?:px)?[ ,]+0(?:px)?[ ,]+0(?:px)?[ ,]+0/.test(s.clip)) return false;
      if (s.clipPath && /inset\\(\\s*(?:50|100)%|circle\\(\\s*0/.test(s.clipPath)) return false;
    }
    return true;
  };
  const fillable = Array.from(document.querySelectorAll("input, textarea, select")).filter(onScreen).length;
  const long = Array.from(document.querySelectorAll("p")).filter((p) => (p.textContent || "").trim().length > __PARA__).length;
  const article = document.querySelectorAll("article").length > 0 || long >= __PARAS__;
  if (article || text.length > __SHORT__ || fillable > 0) return null;
  const heads = [document.title || ""].concat(Array.from(document.querySelectorAll("h1, h2, [role=heading]"),
    (h) => h.textContent || ""));
  const titled = heads.some((h) => new RegExp(__CTITLE__, "i").test(h));
  const buttoned = Array.from(document.querySelectorAll("button, [role=button], input[type=submit], input[type=button]"),
    (b) => ((b.innerText || b.textContent || b.value || "") + "").replace(/\\s+/g, " ").trim())
    .some((b) => new RegExp(__CBUTTON__, "i").test(b));
  for (const [kind, src] of P) {
    const m = new RegExp(src, "im").exec(text);
    if (!m) continue;
    if (titled || (buttoned && kind !== "identity_check")) return out(kind, line(m.index), "imperative");
    return null;
  }
  return null;
})()"""


def probe_js() -> str:
    return (PROBE_JS.replace("__PATTERNS__", json.dumps([list(p) for p in IMPERATIVE]))
            .replace("__FRAME__", json.dumps(CHALLENGE_FRAME)).replace("__SELECTOR__", json.dumps(CHALLENGE_SELECTOR))
            .replace("__LIMIT__", str(_SAYS_LIMIT)).replace("__SHORT__", str(SHORT_PAGE_CHARS))
            .replace("__PARA__", str(ARTICLE_PARAGRAPH_CHARS)).replace("__PARAS__", str(ARTICLE_PARAGRAPHS))
            .replace("__CTITLE__", json.dumps(CONTEXT_TITLE)).replace("__CBUTTON__", json.dumps(CONTEXT_BUTTON)))


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
