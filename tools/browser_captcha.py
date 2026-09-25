"""What CAPTCHA (if any) a page in Moe's own browser is showing, read from what CDP sees.

This is the detector half of Moe's engine-level CAPTCHA ladder (``tools/browser_captcha_ladder.py``).
Owner decision, 2026-09-25: Moe acts in the owner's own accounts from the owner's own Mac and IP, and
handles CAPTCHAs itself *at the engine level* where it can -- never through the chat model, which is
trained to refuse and is not ours to choose (Moe is provider-agnostic) -- and hands the page to the person
as the last resort. The ladder is detect -> pass without solving -> solve -> person.

Detection reads the page's facts, not its prose: the frame tree and the iframes in the DOM (walked with
``pierce: true`` so a closed shadow root -- Turnstile's -- hides nothing), a few marker elements, the
response-token inputs a widget fills when it is passed, the cookies a vendor sets on a pass, the title,
and (when a caller has them) the main response's headers. :func:`classify` turns those into one
:class:`Challenge`::

    {kind, vendor, host, frame, sitekey, solved, hard_stop, stage}

``hard_stop`` is a reason string when the ladder must never automate this challenge and it goes to the
person at once: a ticketing site (the US BOTS Act makes circumventing a ticket seller's access control
illegal -- whatever the purpose), a DataDome ``t=bv`` page (the IP is banned: nothing on the page can
pass it), or the same challenge having failed three times (``tools/browser_captcha_ladder.py`` counts).

It also owns the text signatures ``browser_exec`` has always used to label a bot wall in what the code
printed (:data:`BLOCK_SIGNATURES`, re-exported by ``tools.browser_chrome_extension``) so there is one
table of what a wall looks like, not two.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

# ---- kinds -----------------------------------------------------------------------------------------

TURNSTILE = "turnstile"
CF_MANAGED = "cf_managed"
RECAPTCHA_V2 = "recaptcha_v2"
RECAPTCHA_V3 = "recaptcha_v3"          # invisible score / enterprise: nothing to click -- report only
HCAPTCHA = "hcaptcha"
PERIMETERX = "perimeterx"
DATADOME = "datadome"
ARKOSE = "arkose"
GEETEST = "geetest"
AWS_WAF = "aws_waf"
IMAGE_TEXT = "image_text"              # a plain "type the characters" / "what is 3 + 4" image

KINDS = (TURNSTILE, CF_MANAGED, RECAPTCHA_V2, RECAPTCHA_V3, HCAPTCHA, PERIMETERX, DATADOME, ARKOSE,
         GEETEST, AWS_WAF, IMAGE_TEXT)

#: kind -> the vendor label ``browser_exec`` reports in ``blocked_by`` (the labels predate this module).
VENDOR = {TURNSTILE: "cloudflare", CF_MANAGED: "cloudflare", RECAPTCHA_V2: "captcha", RECAPTCHA_V3: "captcha",
          HCAPTCHA: "captcha", PERIMETERX: "perimeterx", DATADOME: "datadome", ARKOSE: "captcha",
          GEETEST: "captcha", AWS_WAF: "captcha", IMAGE_TEXT: "captcha"}

#: Kinds whose challenge work must never leave this Mac (no third-party solver, ever): Cloudflare's
#: clearance is bound to the browser and IP that earned it, and handing its challenge to a solving
#: service is exactly the traffic the owner ruled out.
NEVER_THIRD_PARTY = frozenset({TURNSTILE, CF_MANAGED})

# ---- hard stops ------------------------------------------------------------------------------------

HARD_STOP_TICKETING = "ticketing"
HARD_STOP_IP_BANNED = "ip_banned"
HARD_STOP_REPEATED = "failed_3_times"

#: Ticket sellers and resellers. The BOTS Act (15 U.S.C. 45c) forbids circumventing a ticket issuer's
#: access control or purchase limits -- so on these hosts Moe never touches a challenge, it hands over.
#: A bare name ("ticketmaster") matches that label under ANY suffix (ticketmaster.co.uk, .com.au ...);
#: a dotted name matches that domain and its subdomains. ``browser.captcha.hard_stop_hosts`` adds more.
DEFAULT_HARD_STOP_HOSTS: Tuple[str, ...] = (
    "ticketmaster", "livenation", "stubhub", "eventbrite", "eventim", "ticketek", "seetickets",
    "axs.com", "seatgeek.com", "vividseats.com", "tickpick.com", "gametime.co", "ticketweb.com",
    "etix.com", "tickets.com", "showclix.com", "frontgatetickets.com", "tixr.com", "dice.fm",
    "universe.com", "ticketswap.com", "viagogo.com", "ticketnetwork.com", "gotickets.com",
)


def host_of(url: str) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def host_matches(host: str, entries: Iterable[str]) -> Optional[str]:
    """The entry ``host`` falls under, or None. See :data:`DEFAULT_HARD_STOP_HOSTS` for the two forms."""
    host = (host or "").lower().rstrip(".")
    if not host:
        return None
    labels = host.split(".")
    for raw in entries:
        e = str(raw or "").strip().lower().lstrip("*.").rstrip(".")
        if not e:
            continue
        if "." in e:
            if host == e or host.endswith("." + e):
                return e
        elif e in labels[:-1]:  # a bare brand label, never the TLD itself
            return e
    return None


def hard_stop_hosts(browser_cfg: Optional[dict]) -> Tuple[str, ...]:
    extra = captcha_config(browser_cfg).get("hard_stop_hosts") or ()
    if isinstance(extra, str):
        extra = [extra]
    return DEFAULT_HARD_STOP_HOSTS + tuple(str(h) for h in extra if isinstance(h, str) and h.strip())


def captcha_config(browser_cfg: Optional[dict]) -> dict:
    cfg = (browser_cfg or {}).get("captcha") if isinstance(browser_cfg, dict) else None
    return cfg if isinstance(cfg, dict) else {}


def ladder_enabled(browser_cfg: Optional[dict]) -> bool:
    """``browser.captcha.enabled`` -- on unless explicitly false."""
    return captcha_config(browser_cfg).get("enabled", True) is not False


# ---- text signatures (what browser_exec printed) -----------------------------------------------------

#: Page-text hints of a bot wall, most specific first: (vendor label, pattern). They only label a
#: browser_exec result; ``tools.browser_chrome_extension.wall_host`` is the signature-grade test the
#: lane router learns from. Measured 2026-09-23, Skyscanner search in Chrome for Testing: url
#: .../sttc/px/captcha-v2/..., text "Are you a person or a robot?".
BLOCK_SIGNATURES = (
    ("perimeterx", re.compile(r"px-captcha|/px/captcha|captcha-v2|press\s*(?:&|and)\s*hold|perimeterx|_pxhd", re.I)),
    ("datadome", re.compile(r"datadome|captcha-delivery\.com", re.I)),
    ("cloudflare", re.compile(r"just a moment\.\.\.|cf-chl|challenges\.cloudflare\.com|attention required! \| cloudflare"
                              r"|verify you are human|error 1020", re.I)),
    ("akamai", re.compile(r"access denied.{0,80}reference #|akamai", re.I | re.S)),
    ("captcha", re.compile(r"are you a (?:person or a )?robot|not a robot|unusual traffic|google\.com/sorry|recaptcha|hcaptcha", re.I)),
)

#: Further evidence that a page MAY hold a challenge -- enough to make the ladder look at the page, never
#: enough to label a result ``blocked_by`` (that stays :data:`BLOCK_SIGNATURES`, unchanged).
CHALLENGE_HINTS = re.compile(r"cf-turnstile|turnstile|arkoselabs|funcaptcha|geetest|awswaf|gokuProps|captcha", re.I)


def text_signature(text: str) -> Optional[str]:
    """Vendor label of the first :data:`BLOCK_SIGNATURES` hit in ``text``, or None."""
    if not text:
        return None
    for label, pattern in BLOCK_SIGNATURES:
        if pattern.search(text):
            return label
    return None


def worth_a_look(text: str) -> bool:
    """Whether what browser_exec printed suggests a challenge, so the ladder should read the page."""
    return bool(text) and (text_signature(text) is not None or bool(CHALLENGE_HINTS.search(text)))


# ---- page facts -------------------------------------------------------------------------------------

@dataclass
class Frame:
    """An iframe the page holds: its ``src`` (or its target's URL), and how to find its box."""
    url: str
    backend_node_id: Optional[int] = None
    visible: Optional[bool] = None


@dataclass
class Element:
    """A marker element: tag, id, classes and the attributes the detector reads."""
    tag: str
    id: str = ""
    classes: Tuple[str, ...] = ()
    attrs: Dict[str, str] = field(default_factory=dict)
    backend_node_id: Optional[int] = None


@dataclass
class PageFacts:
    url: str = ""
    title: str = ""
    status: Optional[int] = None                                 # main navigation's HTTP status, when known
    text: str = ""                                               # body innerText, capped
    frames: List[Frame] = field(default_factory=list)            # iframes in the DOM + OOPIF targets
    scripts: List[str] = field(default_factory=list)             # <script src> URLs
    elements: List[Element] = field(default_factory=list)
    tokens: Dict[str, str] = field(default_factory=dict)         # response-token input name -> value
    cookies: Dict[str, str] = field(default_factory=dict)        # cookie name -> value (for the page URL)
    headers: Dict[str, str] = field(default_factory=dict)        # main response headers, lower-cased
    globals: Dict[str, bool] = field(default_factory=dict)       # e.g. {"_cf_chl_opt": True, "gokuProps": True}


@dataclass
class Challenge:
    kind: str
    host: str
    vendor: str = ""
    frame: str = ""                     # the challenge frame's URL, when there is one
    sitekey: str = ""
    solved: bool = False
    hard_stop: Optional[str] = None     # HARD_STOP_* reason, or None
    stage: str = ""                     # "checkbox" | "image" | "interstitial" | "hold" | "slider" | "report"
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in ("", None, {}, [])} | {"solved": self.solved}


# ---- matchers ---------------------------------------------------------------------------------------

_RE_TURNSTILE_FRAME = re.compile(r"challenges\.cloudflare\.com(?::\d+)?/.*(?:turnstile|/cdn-cgi/challenge-platform/)", re.I)
_RE_CF_CHALLENGE_PATH = re.compile(r"__cf_chl_|/cdn-cgi/challenge-platform/", re.I)
_RE_CF_TITLE = re.compile(r"^(?:\W+\s*)?(?:just a moment\.*|attention required! \| cloudflare|un instant\.*|einen moment\.*)$", re.I)
_RE_RECAPTCHA_ANCHOR = re.compile(r"(?:google\.com|recaptcha\.net)(?::\d+)?/recaptcha/(?:api2|enterprise)/anchor", re.I)
_RE_RECAPTCHA_BFRAME = re.compile(r"(?:google\.com|recaptcha\.net)(?::\d+)?/recaptcha/(?:api2|enterprise)/bframe", re.I)
_RE_HCAPTCHA = re.compile(r"^https?://(?:[a-z0-9-]+\.)*hcaptcha\.com(?::\d+)?/", re.I)
_RE_DATADOME = re.compile(r"(?:geo\.)?captcha-delivery\.com/(?:captcha|interstitial)", re.I)
_RE_ARKOSE = re.compile(r"arkoselabs\.com|funcaptcha\.com|arkoselabs\.cn", re.I)
_RE_GEETEST = re.compile(r"static\.geetest\.com|gcaptcha4\.js|api\.geetest\.com|gcaptcha4\.geetest\.com", re.I)
_RE_AWSWAF = re.compile(r"captcha\.awswaf\.com|\.awswaf\.com/.*(?:captcha|challenge)\.js|token\.awswaf\.com", re.I)
_RE_PX_PATH = re.compile(r"/px/captcha|/sttc/px/|captcha\.px-cdn\.net|px-cloud\.net/.*captcha", re.I)
_RE_PX_TEXT = re.compile(r"press\s*(?:&|and)\s*hold", re.I)
_RE_IMAGE_TEXT = re.compile(r"(?:type|enter) the (?:characters|text|letters|code|word)s? (?:you see|shown|in the (?:image|picture))"
                            r"|what is \d+\s*[+\-x*]\s*\d+|solve (?:this|the) (?:math|sum)", re.I)


def _frame_urls(facts: PageFacts) -> List[str]:
    return [f.url for f in facts.frames if f.url]


def _first(urls: Sequence[str], pattern: "re.Pattern[str]") -> str:
    return next((u for u in urls if pattern.search(u)), "")


def _el(facts: PageFacts, *, id_: str = "", cls: str = "", attr: str = "", tag: str = "") -> Optional[Element]:
    for e in facts.elements:
        if id_ and e.id != id_:
            continue
        if cls and cls not in e.classes:
            continue
        if attr and attr not in e.attrs:
            continue
        if tag and e.tag.lower() != tag:
            continue
        return e
    return None


def _token(facts: PageFacts, *names: str) -> bool:
    return any((facts.tokens.get(n) or "").strip() for n in names)


def _datadome_t(url: str) -> str:
    try:
        return (parse_qs(urlparse(url).query).get("t") or [""])[0].lower()
    except ValueError:
        return ""


def _is_cf_interstitial(facts: PageFacts) -> bool:
    return (facts.headers.get("cf-mitigated", "").lower() == "challenge"
            or bool(_RE_CF_TITLE.match((facts.title or "").strip()))
            or bool(_RE_CF_CHALLENGE_PATH.search(facts.url or ""))
            or bool(facts.globals.get("_cf_chl_opt")))


def classify(facts: PageFacts, *, hard_stop_hosts: Sequence[str] = DEFAULT_HARD_STOP_HOSTS) -> Optional[Challenge]:
    """The challenge the page is showing, or None. Order matters: a hard stop's evidence first, then the
    interstitials that own the whole page, then embedded widgets, then report-only signals."""
    ch = _classify(facts)
    if ch is None:
        return None
    hit = host_matches(ch.host, hard_stop_hosts)
    if hit and not ch.hard_stop:
        ch.hard_stop = HARD_STOP_TICKETING
        ch.detail["hard_stop_host"] = hit
    return ch


def _classify(facts: PageFacts) -> Optional[Challenge]:
    host = host_of(facts.url)
    urls = _frame_urls(facts)
    scripts = facts.scripts

    # DataDome: its challenge is an iframe on geo.captcha-delivery.com; ``t=bv`` means the IP is banned.
    dd = _first(urls, _RE_DATADOME) or (facts.url if _RE_DATADOME.search(facts.url or "") else "")
    if dd:
        t = _datadome_t(dd)
        ch = Challenge(DATADOME, host, frame=dd, stage="slider" if t != "bv" else "banned",
                       solved=False, detail={"t": t} if t else {})
        if t == "bv":
            ch.hard_stop = HARD_STOP_IP_BANNED
        return ch

    # PerimeterX / HUMAN "Press & Hold".
    px_el = _el(facts, id_="px-captcha")
    if px_el or _RE_PX_PATH.search(facts.url or "") or (_RE_PX_TEXT.search(facts.text or "") and "_pxhd" in facts.cookies):
        # A pass sets a fresh _px3 and takes the page back to content (no #px-captcha).
        return Challenge(PERIMETERX, host, stage="hold", solved=bool(px_el is None and "_px3" in facts.cookies
                                                                    and not _RE_PX_PATH.search(facts.url or "")))

    # Cloudflare managed challenge ("Just a moment..."): the whole page is the interstitial.
    if _is_cf_interstitial(facts):
        frame = _first(urls, _RE_TURNSTILE_FRAME)
        return Challenge(CF_MANAGED, host, frame=frame, stage="interstitial", solved=False,
                         detail={"cf_mitigated": facts.headers.get("cf-mitigated", "")} if facts.headers.get("cf-mitigated") else {})

    # Arkose (FunCaptcha), GeeTest, AWS WAF: all puzzle-grade, tier C or the person.
    ark = _first(urls, _RE_ARKOSE) or _first(scripts, _RE_ARKOSE)
    if ark:
        return Challenge(ARKOSE, host, frame=_first(urls, _RE_ARKOSE), stage="image",
                         solved=_token(facts, "fc-token", "arkose-token", "verification-token"))
    gt = _first(scripts, _RE_GEETEST) or _first(urls, _RE_GEETEST)
    if gt or any(c.startswith("geetest_") for e in facts.elements for c in e.classes):
        return Challenge(GEETEST, host, frame=_first(urls, _RE_GEETEST), stage="image",
                         solved=_token(facts, "geetest_validate", "lot_number"))
    if facts.globals.get("gokuProps") or _first(scripts, _RE_AWSWAF):
        return Challenge(AWS_WAF, host, stage="image", solved="aws-waf-token" in facts.cookies and not facts.globals.get("gokuProps"))

    # Turnstile widget embedded in an ordinary page.
    ts_frame = _first(urls, _RE_TURNSTILE_FRAME)
    ts_el = _el(facts, cls="cf-turnstile")
    if ts_frame or ts_el or "cf-turnstile-response" in facts.tokens:
        return Challenge(TURNSTILE, host, frame=ts_frame, sitekey=(ts_el.attrs.get("data-sitekey", "") if ts_el else ""),
                         stage="checkbox", solved=_token(facts, "cf-turnstile-response"))

    # hCaptcha: #frame=checkbox is the box, #frame=challenge the image grid.
    hc_urls = [u for u in urls if _RE_HCAPTCHA.search(u)]
    hc_el = _el(facts, cls="h-captcha")
    if hc_urls or hc_el:
        challenge_frames = [f for f in facts.frames if _RE_HCAPTCHA.search(f.url) and "frame=challenge" in f.url]
        image = any(f.visible for f in challenge_frames)
        return Challenge(HCAPTCHA, host, frame=next((u for u in hc_urls if "frame=checkbox" in u), hc_urls[0] if hc_urls else ""),
                         sitekey=(hc_el.attrs.get("data-sitekey", "") if hc_el else _qs(hc_urls, "sitekey")),
                         stage="image" if image else "checkbox",
                         solved=_token(facts, "h-captcha-response", "g-recaptcha-response") and not image)

    # reCAPTCHA: a visible anchor (size=normal|compact) is v2's checkbox; bframe is its image challenge;
    # an invisible anchor alone is v3 / enterprise scoring -- report only, nothing to click.
    anchors = [u for u in urls if _RE_RECAPTCHA_ANCHOR.search(u)]
    bframes = [f for f in facts.frames if _RE_RECAPTCHA_BFRAME.search(f.url)]
    rc_el = _el(facts, cls="g-recaptcha")
    visible_anchor = [u for u in anchors if "size=invisible" not in u]
    if visible_anchor or any(f.visible for f in bframes) or (rc_el and rc_el.attrs.get("data-size") != "invisible"):
        image = any(f.visible for f in bframes)
        return Challenge(RECAPTCHA_V2, host, frame=(visible_anchor or anchors or [""])[0],
                         sitekey=(rc_el.attrs.get("data-sitekey", "") if rc_el else _qs(anchors, "k")),
                         stage="image" if image else "checkbox",
                         solved=_token(facts, "g-recaptcha-response") and not image)
    if anchors:
        return Challenge(RECAPTCHA_V3, host, frame=anchors[0], sitekey=_qs(anchors, "k"), stage="report", solved=False)

    # A plain image / arithmetic CAPTCHA.
    img = next((e for e in facts.elements if e.tag.lower() == "img"
                and re.search(r"captcha", " ".join([e.id, *e.classes, e.attrs.get("src", ""), e.attrs.get("alt", "")]), re.I)), None)
    if img or _RE_IMAGE_TEXT.search(facts.text or ""):
        return Challenge(IMAGE_TEXT, host, stage="image", solved=False)
    return None


def _qs(urls: Sequence[str], key: str) -> str:
    for u in urls:
        for part in (urlparse(u).query, urlparse(u).fragment):
            v = (parse_qs(part).get(key) or [""])[0]
            if v:
                return v
    return ""


# ---- collecting facts over CDP ---------------------------------------------------------------------

#: Marker selectors worth a node in PageFacts.elements (checked against the pierced DOM, closed shadow
#: roots included).
_MARK_IDS = {"px-captcha", "captcha-container", "challenge-form", "challenge-stage", "cf-chl-widget"}
_MARK_CLASSES = {"cf-turnstile", "g-recaptcha", "h-captcha", "geetest_holder", "geetest_panel", "geetest_box"}
_TOKEN_NAMES = ("cf-turnstile-response", "g-recaptcha-response", "h-captcha-response", "fc-token",
                "arkose-token", "verification-token", "geetest_validate", "lot_number")
_KEEP_ATTRS = ("data-sitekey", "data-size", "src", "alt", "name", "data-callback")

#: Runs in the top page: what only JavaScript can see (a response token's live value, globals, status).
PAGE_PROBE_JS = """(() => {
  const names = %s;
  const tokens = {};
  const roots = [document];
  for (let i = 0; i < roots.length && i < 200; i++) {
    const r = roots[i];
    for (const n of names) {
      for (const el of r.querySelectorAll('[name="' + n + '"], #' + n.replace(/[^a-z0-9_-]/gi, ''))) {
        if (el.value) tokens[n] = String(el.value).slice(0, 64); else if (!(n in tokens)) tokens[n] = '';
      }
    }
    for (const el of r.querySelectorAll('*')) if (el.shadowRoot) roots.push(el.shadowRoot);
  }
  let status = null;
  try { const nav = performance.getEntriesByType('navigation')[0]; status = nav && nav.responseStatus || null; } catch (e) {}
  return JSON.stringify({url: location.href, title: document.title, status,
    text: ((document.body && document.body.innerText) || '').slice(0, 6000), tokens,
    globals: {_cf_chl_opt: typeof window._cf_chl_opt !== 'undefined', gokuProps: typeof window.gokuProps !== 'undefined'},
    visible_frames: Array.from(document.querySelectorAll('iframe')).filter(f => {
      const b = f.getBoundingClientRect(), s = getComputedStyle(f);
      // on the page, not parked off it: reCAPTCHA's bframe and hCaptcha's challenge frame wait at top -9999
      // (measured with both vendors' test sitekeys, 2026-09-25) until an image challenge is shown.
      return b.width > 30 && b.height > 30 && s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0'
        && b.bottom + scrollY > 0 && b.right + scrollX > 0;
    }).map(f => f.src || '')});
})()""" % json.dumps(list(_TOKEN_NAMES))


def _walk(node: dict, out_frames: List[Frame], out_scripts: List[str], out_elements: List[Element], depth: int = 0) -> None:
    if depth > 400:
        return
    name = str(node.get("nodeName") or "").upper()
    raw = node.get("attributes") or []
    attrs = {raw[i]: raw[i + 1] for i in range(0, len(raw) - 1, 2)}
    if name == "IFRAME":
        out_frames.append(Frame(url=attrs.get("src", ""), backend_node_id=node.get("backendNodeId")))
    elif name == "SCRIPT" and attrs.get("src"):
        out_scripts.append(attrs["src"])
    if name and not name.startswith("#"):
        el_id = attrs.get("id", "")
        classes = tuple((attrs.get("class") or "").split())
        is_img = name == "IMG" and re.search(r"captcha", " ".join([el_id, attrs.get("class", ""), attrs.get("src", ""),
                                                                   attrs.get("alt", "")]), re.I)
        if el_id in _MARK_IDS or _MARK_CLASSES.intersection(classes) or any(c.startswith("geetest_") for c in classes) or is_img:
            out_elements.append(Element(tag=name.lower(), id=el_id, classes=classes,
                                        attrs={k: attrs[k] for k in _KEEP_ATTRS if k in attrs},
                                        backend_node_id=node.get("backendNodeId")))
    for key in ("children", "shadowRoots"):
        for child in node.get(key) or ():
            _walk(child, out_frames, out_scripts, out_elements, depth + 1)
    if node.get("contentDocument"):
        _walk(node["contentDocument"], out_frames, out_scripts, out_elements, depth + 1)
    if node.get("templateContent"):
        _walk(node["templateContent"], out_frames, out_scripts, out_elements, depth + 1)


def collect_facts(page: Any, *, oopif_urls: Sequence[str] = (), headers: Optional[Dict[str, str]] = None) -> PageFacts:
    """Read :class:`PageFacts` from a page over CDP. ``page`` has ``call(method, params) -> dict`` bound to
    the page's flattened session (tools.browser_captcha_ladder.PageSession). ``oopif_urls`` are the URLs
    of the browser's iframe targets (cross-site frames run out of process and appear there too)."""
    probe_raw = page.call("Runtime.evaluate", {"expression": PAGE_PROBE_JS, "returnByValue": True})
    probe = json.loads(((probe_raw.get("result") or {}).get("value")) or "{}")
    frames: List[Frame] = []
    scripts: List[str] = []
    elements: List[Element] = []
    try:
        doc = page.call("DOM.getDocument", {"depth": -1, "pierce": True})
        _walk(doc.get("root") or {}, frames, scripts, elements)
    except Exception:  # a page mid-navigation: the JS probe still stands
        pass
    visible = set(probe.get("visible_frames") or [])
    for f in frames:
        f.visible = f.url in visible if f.url else None
    known = {f.url for f in frames}
    frames.extend(Frame(url=u, visible=None) for u in oopif_urls if u and u not in known)
    cookies: Dict[str, str] = {}
    url = str(probe.get("url") or "")
    try:
        for c in (page.call("Network.getCookies", {"urls": [url]}).get("cookies") or []) if url.startswith("http") else []:
            cookies[str(c.get("name"))] = str(c.get("value") or "")
    except Exception:
        pass
    return PageFacts(url=url, title=str(probe.get("title") or ""), status=probe.get("status"),
                     text=str(probe.get("text") or ""), frames=frames, scripts=scripts, elements=elements,
                     tokens={k: str(v) for k, v in (probe.get("tokens") or {}).items()}, cookies=cookies,
                     headers={str(k).lower(): str(v) for k, v in (headers or {}).items()},
                     globals={k: bool(v) for k, v in (probe.get("globals") or {}).items()})
