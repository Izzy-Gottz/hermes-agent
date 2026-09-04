"""Two-tier stall detection for ``computer_use``.

Why this exists
---------------
On 2026-09-03 an agent asked to open a meeting link made fourteen
``computer_use`` calls in a hundred seconds. Two were clicks; the other twelve
were failed attempts to *see* — ``capture(app='Calendar')`` returning an
identical empty result while the window sat one Space away. Nothing detected
it. Hermes's generic guardrail (``agent/tool_guardrails.py``) hashes the whole
result, and a capture result carries a screenshot whose bytes differ every
frame, so identical-call matching never fired; its hard stop is default-off
regardless.

Three things this had to get right, each of which a first draft got wrong
-------------------------------------------------------------------------
**Count occurrences in a window, not a consecutive run.** The real trace never
repeats a call twice in a row — a thrashing agent varies its approach, so its
own diagnostic calls (``list_apps``, ``list_windows``, ``focus_app``)
interleave between the repeats. A tail-consecutive streak peaks at 1 on that
trace and refuses nothing. The signal is "N of the last M calls were this same
inert call", which is what OpenHands' detector measures and why it strips ids
before comparing.

**Fingerprint what the result said, through whatever envelope it arrived in.**
A ``capture`` that carries a screenshot returns a ``_multimodal`` dict whose
top level is ``_multimodal``/``content``/``text_summary``/``meta`` — none of
the fields a flat JSON action result has. Reading only the flat keys makes
every image capture fingerprint identically to every other, which both hides
real loops and refuses a sixth capture of a screen that is visibly changing.
``meta`` carries the dimensions and element count; ``screenshot_path`` and
``elements_file`` are uuid4-named and must be excluded, or nothing ever
matches anything.

**Exempt the verbs whose whole purpose is repetition.** ``scroll`` and ``key``
are "do it again" actions: a fourth ``cmd+z`` undoes a fourth edit, a fourth
arrow-key moves a fourth row, a fourth ``pagedown`` reaches page four. Their
results are byte-identical every time and carry no evidence of progress, so
any identity-based detector will refuse them. browser-use exempts exactly this
class (``_LOOP_EXEMPT_ACTIONS``); so do we. The cost is that a scroll stuck at
the bottom of a page goes unflagged — cheap and harmless, unlike interrupting
someone mid-document to ask about a scroll that was working.

Thresholds follow browser-see's pair: change strategy before total failure,
not after it. Input actions refuse sooner because repeating unverified input
sends the message twice, whereas repeating a read only wastes time.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

# "The Nth identical call is refused." Soft tier advises from the 3rd.
SOFT_LIMIT = 3
HARD_LIMIT = 5
INPUT_HARD_LIMIT = 3
WINDOW = 20

# Deliberately repeatable: repetition is the point, and the result carries no
# evidence of progress. Never counted, never refused.
_EXEMPT = frozenset({"wait", "scroll", "key"})

# Reading the screen. None of these change anything, so a run of them is not a
# loop in the identical-call sense — every screenshot differs, so no two
# fingerprint alike and the repeat detector cannot see it at all.
#
# Measured on a real session, 2026-09-04 20:27-20:37: 33 calls over 578 s. Tool
# time was 80 s (14%); the model thinking was 498 s (86%). The shape was 22
# captures against 5 actions, ending in SEVENTEEN consecutive looks with
# nothing done in between — no failure, no error, just looking. The owner's
# word for it was "stuck", and he was right. It matches the published
# measurement (OSWorld-Human: planning and reflection 75-94% of wall clock,
# grounding 2-4%), and it is the thing worth detecting: not a repeated call,
# an unproductive one.
_READ_ONLY = frozenset({
    "capture", "zoom", "list_windows", "list_apps", "focused_element",
    "verify_state",
})

# Six looks with nothing done is worth a word; ten is worth stopping.
# Generous on purpose — real exploration takes several looks, and the run that
# prompted this was seventeen.
LOOK_SOFT_LIMIT = 6
LOOK_HARD_LIMIT = 10

# Actions that send input somewhere. Repeating one of these unverified is how
# a message goes twice; repeating a read only costs time.
_INPUT = frozenset({
    "click", "double_click", "right_click", "middle_click", "drag",
    "type", "set_value",
})

# uuid4-named per call: including them means no two results ever match.
_UNSTABLE_META = frozenset({"screenshot_path", "elements_file"})

_STABLE_KEYS = (
    "ok", "action", "effect", "code", "path", "degraded", "delivery_mode",
    "mode", "width", "height", "app", "window_title", "total_elements",
    "count", "error",
)


def call_fingerprint(action: str, args: Dict[str, Any]) -> str:
    """What the call *meant* — target identity, not incidental arguments."""
    target = {
        k: args.get(k)
        for k in (
            "app", "pid", "window_id", "element", "coordinate", "mode",
            "direction", "amount", "keys", "region", "from_element",
            "to_element", "from_coordinate", "to_coordinate",
            # Without these, five DIFFERENT menu items in one window look like
            # one call repeated five times, and the fifth is refused with a
            # message ("these exact arguments") that is simply false.
            "path", "expect",
        )
        if args.get(k) is not None
    }
    text = args.get("text")
    if isinstance(text, str):
        # Length plus a short prefix: enough to tell two strings apart without
        # copying the user's content into a key that gets logged.
        target["text"] = f"{len(text)}:{text[:16]}"
    return json.dumps([action, target], sort_keys=True, default=str)


def result_fingerprint(result: Any) -> str:
    """What the result *said*, with everything unstable removed."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return result[:200]
    if not isinstance(result, dict):
        return str(result)[:200]

    if result.get("_multimodal"):
        meta = result.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        return json.dumps(
            {k: v for k, v in sorted(meta.items()) if k not in _UNSTABLE_META},
            sort_keys=True, default=str,
        )

    stable = {k: result.get(k) for k in _STABLE_KEYS if k in result}
    verdict = result.get("verdict")
    if isinstance(verdict, dict):
        stable["decision"] = verdict.get("decision")
    for k in ("front_app", "focused"):
        v = result.get(k)
        if isinstance(v, dict):
            stable[k] = {kk: v.get(kk)
                         for kk in ("name", "bundle_id", "role", "window_title")}
    return json.dumps(stable, sort_keys=True, default=str)


class StallDetector:
    """Per-session history of what was tried and what came back.

    Thread-safe: two ``computer_use`` calls can be in flight in one session,
    and the check runs outside the backend's per-session call lock.
    """

    def __init__(self, window: int = WINDOW) -> None:
        self._history: Deque[Tuple[str, str]] = deque(maxlen=window)
        self._lock = threading.Lock()
        # Consecutive read-only calls since anything last changed the screen.
        self._looks = 0

    def _occurrences(self, call_fp: str) -> int:
        """How many times this call produced its own most recent result.

        Occurrences across the window, not a consecutive run: the interleaved
        diagnostic calls a thrashing agent makes must not reset the count.
        """
        with self._lock:
            entries = [r for c, r in self._history if c == call_fp]
        if not entries:
            return 0
        return entries.count(entries[-1])

    def _limit(self, action: str) -> int:
        return INPUT_HARD_LIMIT if action in _INPUT else HARD_LIMIT

    def block_reason(self, action: str, args: Dict[str, Any]) -> Optional[str]:
        """A JSON error string when this call must not run, else None.

        Called *before* the call, so the pending attempt is counted as one
        more than what history holds: with ``HARD_LIMIT = 5`` the fifth
        identical call is the one refused.
        """
        if action in _EXEMPT:
            return None
        limit = self._limit(action)
        attempt = self._occurrences(call_fingerprint(action, args)) + 1
        if attempt < limit:
            return None
        noun = "input" if action in _INPUT else "call"
        return json.dumps({
            "error": (
                f"refusing to repeat this {noun}: {action} with these exact "
                f"arguments has already returned the same result "
                f"{attempt - 1} times. Attempt {attempt} cannot produce a "
                f"different outcome."
            ),
            "code": "stalled_repeated_action",
            "attempt": attempt,
            "verdict": {
                "decision": "stop_and_report",
                "hint": (
                    "Change approach or stop. Do NOT vary the coordinate "
                    "slightly and retry. Consider: is the window on another "
                    "Space (check list_windows for on_current_space)? Is it "
                    "minimized? Is there a non-GUI route — a join URL on the "
                    "calendar event, an AppleScript, a CLI — that skips the "
                    "screen entirely? If none of those, tell the user exactly "
                    "what you are seeing and ask; do not keep trying."
                ),
            },
        })

    def advisory(self, action: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The soft tier. Called *after* record, so the count includes this call."""
        if action in _EXEMPT:
            return None
        seen = self._occurrences(call_fingerprint(action, args))
        if seen < SOFT_LIMIT:
            return None
        limit = self._limit(action)
        remaining = max(0, limit - seen - 1)
        tail = ("The next identical attempt will be refused."
                if remaining <= 0
                else f"{remaining} more identical attempt(s) will be allowed, "
                     f"then refused.")
        return {
            "decision": "change_approach",
            "seen": seen,
            "hint": (
                f"{action} has now returned the same result {seen} times. "
                f"Repeating it is not working. Before trying again, change "
                f"something real: a different window (list_windows shows "
                f"on_current_space), a different capture mode, or a route "
                f"that avoids the GUI. {tail}"
            ),
        }

    def looks_without_acting(self) -> int:
        with self._lock:
            return self._looks

    def looking_loop(self, action: str) -> Optional[Dict[str, Any]]:
        """Advise, or refuse, a read that is going nowhere.

        Only reads are ever refused here. Stopping someone from *looking* is
        safe; stopping them from acting is not, and an agent that has looked
        ten times without acting is not short of pixels.
        """
        if action not in _READ_ONLY:
            return None
        seen = self.looks_without_acting()
        if seen + 1 < LOOK_SOFT_LIMIT:
            return None
        hard = seen + 1 >= LOOK_HARD_LIMIT
        return {
            "decision": "stop_and_report" if hard else "change_approach",
            "looks_without_acting": seen + 1,
            "refused": hard,
            "hint": (
                f"This is look number {seen + 1} with nothing done in between. "
                f"Another screenshot will not tell you what the last {seen} "
                f"did not. Either do the next thing — invoke_menu is usually "
                f"the surest, and `mac.sh menus <app>` lists the paths — or "
                f"tell the user plainly what you can see and what is stopping "
                f"you."
                + (" Further reads are refused until you act or answer."
                   if hard else "")
            ),
        }

    def record(self, action: str, args: Dict[str, Any], result: Any) -> None:
        with self._lock:
            if action in _READ_ONLY:
                self._looks += 1
            else:
                self._looks = 0
        if action in _EXEMPT:
            # Not counted — but the history is NOT cleared either. Clearing it
            # let "capture, wait, capture, wait" run forever: inserting a wait
            # changes nothing real, and that is the shape of the 2026-09-03
            # loop.
            return
        entry = (call_fingerprint(action, args), result_fingerprint(result))
        with self._lock:
            self._history.append(entry)

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
