"""Passwords Hermes makes for a new account, inside the tool process, never shown to anyone.

A signup page says what it accepts in three places a browser can read: ``minlength`` /
``maxlength``, a ``pattern`` regex, and Apple's ``passwordrules`` attribute
(``required: upper; required: digit; allowed: [-_!]; max-consecutive: 2; minlength: 12``).
:func:`policy_from_controls` merges what the page's new-password fields say;
:func:`policy_from_request` folds in what the model read off a rejection ("must be 8-16
characters, no symbols"). :func:`generate` draws from ``secrets`` only.

The default is 20 characters of upper, lower, digits and a symbol set almost every site
takes. The value is returned to the caller in the tool process and goes straight into the
vault and the page: never into a result, a log, argv or the environment.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional

UPPER = "ABCDEFGHJKLMNPQRSTUVWXYZ"
LOWER = "abcdefghijkmnopqrstuvwxyz"
DIGIT = "23456789"
#: Accepted by nearly every site that takes symbols at all (no quotes, no backslash, no spaces, no <>).
DEFAULT_SYMBOLS = "!@#$%*-_.?"
DEFAULT_LENGTH = 20
_ABSOLUTE_MIN, _ABSOLUTE_MAX = 8, 128

_CLASSES = {"upper": UPPER, "lower": LOWER, "digit": DIGIT}
# Apple passwordrules class names (https://developer.apple.com/password-rules/).
_APPLE_CLASS = {"upper": "upper", "lower": "lower", "digit": "digit", "special": "symbol",
                "ascii-printable": "any", "unicode": "any"}
_APPLE_SPECIAL = "-~!@#$%^&*_+=`|(){}[:;\"'<>,.?]/ "


@dataclass
class PasswordPolicy:
    min_length: int = 0
    max_length: int = 0            # 0 = no cap
    length: int = 0                # preferred length (0 = DEFAULT_LENGTH)
    required: List[str] = field(default_factory=lambda: ["upper", "lower", "digit", "symbol"])
    symbols: str = DEFAULT_SYMBOLS  # "" = no symbols allowed
    max_consecutive: int = 0
    patterns: List[str] = field(default_factory=list)


def _clamp(policy: PasswordPolicy) -> int:
    lo = max(policy.min_length or 0, _ABSOLUTE_MIN if not policy.max_length or policy.max_length >= _ABSOLUTE_MIN else 1)
    hi = min(policy.max_length or _ABSOLUTE_MAX, _ABSOLUTE_MAX)
    want = policy.length or DEFAULT_LENGTH
    if hi < lo:
        hi = lo
    return max(lo, min(want, hi))


def _parse_apple_rules(text: str, policy: PasswordPolicy) -> PasswordPolicy:
    """Apple ``passwordrules``: ``required``/``allowed`` classes or ``[...]`` sets, ``minlength``,
    ``maxlength``, ``max-consecutive``. Unknown keys are ignored."""
    required: List[str] = []
    allowed_symbols: Optional[str] = None
    required_symbols: Optional[str] = None
    classes_allowed: set = set()
    for part in str(text or "").split(";"):
        if ":" not in part:
            continue
        key, value = (p.strip().lower() for p in part.split(":", 1))
        raw_value = part.split(":", 1)[1].strip()
        if key in ("minlength", "maxlength", "max-consecutive"):
            try:
                n = int(value)
            except ValueError:
                continue
            if key == "minlength":
                policy.min_length = max(policy.min_length, n)
            elif key == "maxlength":
                policy.max_length = n if not policy.max_length else min(policy.max_length, n)
            else:
                policy.max_consecutive = n
            continue
        if key not in ("required", "allowed"):
            continue
        for token in re.findall(r"\[[^\]]*\]|[a-z-]+", raw_value if "[" in raw_value else value):
            if token.startswith("["):
                chars = "".join(ch for ch in token[1:-1] if ch in _APPLE_SPECIAL and ch not in " \"'\\")
                if key == "required":
                    required_symbols = (required_symbols or "") + chars
                    required.append("symbol")
                else:
                    allowed_symbols = (allowed_symbols or "") + chars
                continue
            cls = _APPLE_CLASS.get(token)
            if cls is None:
                continue
            if key == "required":
                required.append(cls)
            classes_allowed.add(cls)
    if required:
        policy.required = sorted({r for r in required if r in ("upper", "lower", "digit", "symbol")})
    symbol_pool = required_symbols or allowed_symbols
    if symbol_pool is not None:
        policy.symbols = "".join(dict.fromkeys(symbol_pool))
    elif required or classes_allowed:
        if not ({"symbol", "any"} & (set(required) | classes_allowed)):
            policy.symbols = ""  # the rules name the classes it takes, and symbols are not among them
    if not policy.symbols and "symbol" in policy.required:
        policy.required.remove("symbol")
    return policy


def policy_from_controls(controls: Iterable[Any]) -> PasswordPolicy:
    """Merge the rules the page's new-password fields carry (``LoginControl`` objects)."""
    policy = PasswordPolicy()
    for c in controls:
        if getattr(c, "min_length", None):
            policy.min_length = max(policy.min_length, int(c.min_length))
        if getattr(c, "max_length", None):
            policy.max_length = int(c.max_length) if not policy.max_length else min(policy.max_length, int(c.max_length))
        pattern = str(getattr(c, "pattern", "") or "")
        if pattern and pattern not in policy.patterns:
            policy.patterns.append(pattern)
        rules = str(getattr(c, "password_rules", "") or "")
        if rules:
            policy = _parse_apple_rules(rules, policy)
    return policy


def policy_from_request(base: PasswordPolicy, request: Any) -> PasswordPolicy:
    """Fold in the rules the model read off the site (never a password): a dict with any of
    ``length``, ``min_length``, ``max_length``, ``symbols`` (bool), ``allowed_symbols`` (str),
    ``require`` (list of upper/lower/digit/symbol), ``max_consecutive``, ``passwordrules`` (Apple
    syntax), or that Apple string on its own."""
    policy = replace(base, required=list(base.required), patterns=list(base.patterns))
    if isinstance(request, str):
        return _parse_apple_rules(request, policy) if request.strip() else policy
    if not isinstance(request, dict):
        return policy
    for key in ("length", "min_length", "max_length", "max_consecutive"):
        try:
            if request.get(key) not in (None, ""):
                n = int(request[key])
                if key == "max_length":
                    policy.max_length = n if not policy.max_length else min(policy.max_length, n)
                elif key == "min_length":
                    policy.min_length = max(policy.min_length, n)
                else:
                    setattr(policy, key, n)
        except (TypeError, ValueError):
            continue
    if isinstance(request.get("allowed_symbols"), str):
        policy.symbols = "".join(ch for ch in dict.fromkeys(request["allowed_symbols"]) if ch.isprintable() and not ch.isalnum() and not ch.isspace())
    if request.get("symbols") is False:
        policy.symbols = ""
    if isinstance(request.get("require"), list):
        policy.required = sorted({str(r).lower() for r in request["require"]} & {"upper", "lower", "digit", "symbol"})
    if isinstance(request.get("passwordrules"), str):
        policy = _parse_apple_rules(request["passwordrules"], policy)
    if not policy.symbols and "symbol" in policy.required:
        policy.required.remove("symbol")
    return policy


def _pattern_ok(pattern: str, candidate: str) -> bool:
    try:
        return re.fullmatch(pattern, candidate) is not None
    except re.error:
        return True  # a JS-only construct Python cannot read: the page is the judge, not us


def _consecutive_ok(candidate: str, limit: int) -> bool:
    if limit <= 0:
        return True
    run = 1
    for a, b in zip(candidate, candidate[1:]):
        run = run + 1 if a == b else 1
        if run > limit:
            return False
    return True


def _attempt(policy: PasswordPolicy, tries: int) -> Optional[str]:
    length = _clamp(policy)
    pools: Dict[str, str] = dict(_CLASSES)
    if policy.symbols:
        pools["symbol"] = policy.symbols
    required = [r for r in policy.required if r in pools] or ["lower", "digit"]
    alphabet = "".join(pools.values())
    rng = secrets.SystemRandom()
    for _ in range(tries):
        chars = [secrets.choice(pools[r]) for r in required][:length]
        chars += [secrets.choice(alphabet) for _ in range(length - len(chars))]
        rng.shuffle(chars)
        candidate = "".join(chars)
        if _consecutive_ok(candidate, policy.max_consecutive) and all(_pattern_ok(p, candidate) for p in policy.patterns):
            return candidate
    return None


def generate(policy: Optional[PasswordPolicy] = None) -> str:
    """A fresh password obeying ``policy``: every required class at least once, the rest drawn from
    the union of allowed classes, shuffled with ``secrets.SystemRandom``. A page ``pattern`` that
    shuts symbols out (``[A-Za-z0-9]{8,}``) gets a symbol-free password rather than a failure."""
    policy = policy or PasswordPolicy()
    found = _attempt(policy, 200)
    if found is None and policy.patterns and policy.symbols:
        found = _attempt(replace(policy, symbols="", required=[r for r in policy.required if r != "symbol"],
                                 patterns=list(policy.patterns)), 200)
    if found is None:
        raise ValueError("could not make a password that satisfies this page's rules")
    return found


def describe(policy: PasswordPolicy) -> Dict[str, Any]:
    """The rules used, for the result (never the password)."""
    return {"length": _clamp(policy), "symbols": bool(policy.symbols),
            "required": list(policy.required), **({"pattern": True} if policy.patterns else {})}
