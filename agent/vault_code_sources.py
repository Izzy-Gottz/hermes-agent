"""Where a one-time code comes from when no authenticator key is saved: the person's own mail.

A site that signs someone in by emailing them a code is asking them to go and read their mail.
An assistant that can read that mail should do exactly that. On 2026-09-24 (Moe ticket #13) a
site emailed the owner a 6-digit code, Moe could read his Gmail, and it still stopped -- the
code-entry tool had two sources, an authenticator seed and asking the person, and neither was
"look in the inbox you already read for them".

So the engine owns the whole of it, once, for every site: which message is the one (sent in the
last few minutes, and from or about the site being signed in to), which token in it is the code,
and putting it in the page. A *source* only answers "what mail arrived recently", for whatever
mailboxes the host has connected -- Moe registers its Gmail accounts; another host registers its
own. The code is registered for redaction and never returned to the model.

What keeps this from being a way to lift codes: the message must name the site the page is ON
(its sender's domain, or the site's name in the subject or text), it must be fresh, and the code
is written only into that page, by origin-checked JS, over the same path a saved password takes.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class MailMessage:
    """One recent message, as a source hands it over. ``at`` is epoch seconds (0 = unknown)."""

    sender: str
    subject: str
    text: str
    at: float
    mailbox: str = ""  # which of the person's mailboxes, for the "where from" line only


#: ``fn(site_label, since_epoch) -> messages``. May raise; a failing source is skipped.
CodeSource = Callable[[str, float], Iterable[MailMessage]]

_lock = threading.Lock()
_sources: List[Tuple[str, CodeSource]] = []

#: How far back a message may have arrived. The model usually presses "send me a code" a few
#: calls before it asks for the code; ten minutes is longer than any site keeps one valid.
FRESH_SECONDS = 600
#: How long enter_code waits for the email to land, polling.
WAIT_SECONDS = 60.0
POLL_SECONDS = 5.0


def register_code_source(name: str, fn: CodeSource) -> None:
    """Add (or replace, by name) a mail source."""
    with _lock:
        _sources[:] = [(n, f) for n, f in _sources if n != name] + [(name, fn)]


def registered_sources() -> List[str]:
    with _lock:
        return [n for n, _ in _sources]


_SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu"}
_HOST_NOISE = {"www", "app", "auth", "login", "accounts", "account", "signin", "id", "my", "secure", "mail"}


def site_label(origin: str) -> str:
    """``https://app.tinylaunch.com`` -> ``tinylaunch``: the name a site's mail carries."""
    host = (urlparse(origin).hostname or origin or "").lower().strip(".")
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _SECOND_LEVEL:
        parts = parts[:-2] + [parts[-2] + "." + parts[-1]]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""


def _sent_by_site(msg: MailMessage, label: str) -> bool:
    sender_domain = msg.sender.lower().rsplit("@", 1)[-1]
    return bool(label) and bool(re.search(r"(^|[.@])" + re.escape(label) + r"\.", sender_domain))


def _names_site(msg: MailMessage, label: str) -> bool:
    if not label or len(label) < 3:
        return False
    if _sent_by_site(msg, label):
        return True
    squashed = re.sub(r"[^a-z0-9]", "", label)
    hay = re.sub(r"[^a-z0-9]", "", (msg.subject + " " + msg.sender + " " + msg.text[:4000]).lower())
    return squashed in hay


_KEYWORDS = re.compile(
    r"(code|verif|one[\s-]?time|otp|passcode|pin\b|log\s?in|sign\s?in|confirm|security|2fa|authenticat|token)",
    re.I)
_CANDIDATE = re.compile(r"(?<![\w$€£#+./:-])(\d{4,8}|(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6,8})(?![\w%/:-])")
_DIGIT_GROUPS = re.compile(r"(?<![\w$€£#+./:-])(\d{3})[ -](\d{3})(?![\w%/:-])")


def extract_code(text: str) -> Optional[str]:
    """The one-time code in a message: a 4-8 digit (or 6-8 mixed upper-case) token that sits
    near words like "code", "verification", "sign in". Years, prices, times and phone-number
    fragments are not codes. ``None`` when nothing reads as a code -- a guess would be typed
    into someone's login."""
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    best: Tuple[int, int, str] = (0, 0, "")
    for rx, join in ((_CANDIDATE, False), (_DIGIT_GROUPS, True)):
        for m in rx.finditer(text):
            token = (m.group(1) + m.group(2)) if join else m.group(1)
            if re.fullmatch(r"(19|20)\d{2}", token):
                continue  # a year
            window = text[max(0, m.start() - 80): m.end() + 40]
            hits = len(_KEYWORDS.findall(window))
            if not hits:
                continue
            score = hits * 10 + (5 if token.isdigit() and len(token) == 6 else 0)
            # Earlier in the message wins ties: the code is what the mail is for.
            if score > best[0] or (score == best[0] and best[2] == ""):
                best = (score, -m.start(), token)
    return best[2] or None


def find_in(messages: Iterable[MailMessage], label: str, since: float) -> Optional[Tuple[str, MailMessage]]:
    """The fresh message that names the site and carries a code: mail SENT BY the site before mail
    that only mentions it (measured 2026-09-24: a Google security notice naming the site, with a
    number in it, sat beside the site's own code mail), newest first within each."""
    usable = [m for m in messages if isinstance(m, MailMessage)
              and (not m.at or m.at >= since) and _names_site(m, label)]
    for msg in sorted(usable, key=lambda m: (_sent_by_site(m, label), m.at or 0), reverse=True):
        code = extract_code(msg.subject + " \n " + msg.text)
        if code:
            return code, msg
    return None


def find_code(origin: str, *, wait: Optional[float] = None, poll: float = POLL_SECONDS,
              now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
              ) -> Tuple[Optional[str], str]:
    """``(code, where)`` from the person's mail for the site at ``origin``, waiting up to
    ``wait`` seconds for it to arrive; ``(None, why)`` when there is none."""
    with _lock:
        sources = list(_sources)
    if not sources:
        return None, "no mailbox is connected"
    label = site_label(origin)
    if not label:
        return None, "the page has no site name to look for"
    since = now() - FRESH_SECONDS
    deadline = now() + max(0.0, WAIT_SECONDS if wait is None else wait)
    while True:
        for name, fn in sources:
            try:
                got = find_in(fn(label, since), label, since)
            except Exception as exc:  # a broken source is skipped, never fatal
                logger.info("vault code source %s failed: %s", name, exc)
                continue
            if got:
                code, msg = got
                return code, (msg.mailbox or name)
        if now() + poll > deadline:
            return None, "no email from %s with a code arrived in the last %d minutes" % (label, FRESH_SECONDS // 60)
        sleep(poll)
