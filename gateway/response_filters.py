"""Gateway response filtering helpers.

These decide whether a completed agent turn should be delivered to the chat,
not what should be persisted in conversation history.
"""

from __future__ import annotations

import unicodedata
from typing import Any

# Exact whole-response markers meaning "the agent intentionally chose not to
# reply". Keep small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({"[SILENT]", "SILENT", "NO_REPLY", "NO REPLY"})

# Longer than any marker could plausibly be, even with stray punctuation.
_MARKER_LENGTH_CAP = 64


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _is_edge_punctuation(ch: str) -> bool:
    # Square brackets stay structural so malformed ``[SILENT`` cannot become ``SILENT``.
    return ch not in "[]" and unicodedata.category(ch).startswith("P")


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation (``.NO_REPLY``, ``*NO_REPLY*``) without erasing marker structure."""
    start, end = 0, len(text)
    while start < end and _is_edge_punctuation(text[start]):
        start += 1
    while end > start and _is_edge_punctuation(text[end - 1]):
        end -= 1
    return text[start:end].strip()


def _canonical_silence_candidates(text: Any) -> tuple[str, ...]:
    """Canonical forms of a short marker-sized response; ``()`` when not a candidate at all."""
    stripped = text.strip() if isinstance(text, str) else ""
    if not 0 < len(stripped) <= _MARKER_LENGTH_CAP:
        return ()
    depunctuated = _strip_edge_silence_punctuation(stripped)
    forms = (stripped,) if depunctuated == stripped else (stripped, depunctuated)
    return tuple(_canonical_silence_candidate(f) for f in forms)


def is_intentional_silence_response(response: Any) -> bool:
    """True only when ``response`` is exactly a silence marker.

    Prose that merely mentions ``NO_REPLY`` must be delivered normally. A blank
    response is not silence either — that is the empty-response failure path.
    """
    return any(c in LIVE_GATEWAY_SILENT_MARKERS for c in _canonical_silence_candidates(response))


# A silence marker may carry a note this long and still mean silence ("[SILENT] No changes
# detected"). Anything longer beside the marker is a report the person was meant to read.
_SILENCE_NOTE_MAX_LINES = 1
_SILENCE_NOTE_MAX_CHARS = 200


def _split_silence_marker(response: Any) -> tuple[bool, str]:
    """``(marker_present, remainder)`` for the autonomous lanes. The marker counts when it is the
    whole response, its own first/last line, or the bracketed sentinel opening the first line;
    ``remainder`` is everything else, joined by newlines. A token buried mid-sentence is not a
    marker (``(False, response)``)."""
    stripped = response.strip() if isinstance(response, str) else ""
    if not stripped:
        return False, ""
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if _canonical_silence_candidate(stripped) in LIVE_GATEWAY_SILENT_MARKERS:
        return True, ""
    # Bracketed form only for the prefix rule, so a bare "Silent retry succeeded" is NOT swallowed.
    if lines[0].upper().startswith("[SILENT]"):
        return True, "\n".join([lines[0][len("[SILENT]"):].strip(), *lines[1:]]).strip()
    if _canonical_silence_candidate(lines[0]) in LIVE_GATEWAY_SILENT_MARKERS:
        return True, "\n".join(lines[1:])
    if _canonical_silence_candidate(lines[-1]) in LIVE_GATEWAY_SILENT_MARKERS:
        return True, "\n".join(lines[:-1])
    return False, stripped


def is_autonomous_silence_response(response: Any) -> bool:
    """Loose silence matcher for autonomous lanes (cron, webhook).

    Models reliably bracket ``[SILENT]`` with a short note, so unlike the
    interactive EXACT rule this also suppresses when a marker sits on its own
    first/last line or the bracketed sentinel opens the response (``[SILENT] No
    changes detected``).  A token buried mid-sentence is still delivered.
    Shares :data:`LIVE_GATEWAY_SILENT_MARKERS` so the two sets cannot drift.

    A marker beside a *report* is not silence. A cron job that could not sign in
    wrote four paragraphs saying so and ended with ``[SILENT]``; the last-line
    rule swallowed the whole thing and the run was booked ok. Only a note of at
    most :data:`_SILENCE_NOTE_MAX_LINES` line(s) and :data:`_SILENCE_NOTE_MAX_CHARS`
    characters rides along with the marker; anything more is delivered — see
    :func:`autonomous_silence_report`.
    """
    present, remainder = _split_silence_marker(response)
    if not present:
        return False
    note_lines = [ln for ln in remainder.splitlines() if ln.strip()]
    return len(note_lines) <= _SILENCE_NOTE_MAX_LINES and len(remainder) <= _SILENCE_NOTE_MAX_CHARS


def autonomous_silence_report(response: Any) -> str | None:
    """The report to deliver when a silence marker was stapled onto real content, with the
    marker removed; ``None`` when the response is silence or carries no marker."""
    present, remainder = _split_silence_marker(response)
    if not present or is_autonomous_silence_response(response):
        return None
    return remainder


# A cron/webhook agent that could not do its job says so with this on its first line. The
# scheduler books the run *blocked* — never ok — and always delivers the words that follow.
BLOCKED_MARKER = "[BLOCKED]"


def blocked_reason(response: Any) -> str | None:
    """The reason after a leading :data:`BLOCKED_MARKER`, or ``None`` when the response does not
    open with it. The reason is the rest of the first line, else the next non-empty line, else the
    marker's own word — never empty, so a booked reason is always something a person can read."""
    stripped = response.strip() if isinstance(response, str) else ""
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines or not lines[0].upper().startswith(BLOCKED_MARKER):
        return None
    rest = lines[0][len(BLOCKED_MARKER):].strip(" :-—\t")
    if not rest and len(lines) > 1:
        rest = lines[1]
    return rest or "blocked"


def strip_blocked_marker(response: Any) -> str:
    """``response`` with a leading :data:`BLOCKED_MARKER` removed — the agent's own words, which
    are what the person gets."""
    stripped = response.strip() if isinstance(response, str) else ""
    if stripped.upper().startswith(BLOCKED_MARKER):
        return stripped[len(BLOCKED_MARKER):].lstrip(" :-—\t\n")
    return stripped


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    return isinstance(agent_result, dict) and not agent_result.get("failed") and is_intentional_silence_response(response)


def is_partial_silence_marker(text: Any) -> bool:
    """True while streamed ``text`` could still resolve to a silence marker.

    A buffer whose canonical form is a non-empty *prefix* of a marker (``"NO"`` on
    the way to ``"NO_REPLY"``, or an exact marker not yet terminated by stream-end)
    is held back so a raw marker is never shown and then retracted.  Divergence
    from every marker, or exceeding the cap, resumes normal streaming.
    """
    return any(
        c and any(marker.startswith(c) for marker in LIVE_GATEWAY_SILENT_MARKERS)
        for c in _canonical_silence_candidates(text)
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

SILENT_REPLY_TOKEN = "NO_REPLY"
# ---- END PLUGIN-COMPAT ----
