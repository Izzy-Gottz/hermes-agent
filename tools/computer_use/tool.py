"""`computer_use` tool entry point: any-model desktop control (macOS/Windows/Linux) via cua-driver.
Return contract: text-only results are a JSON string; captures / `capture_after=True` return
``{"_multimodal": True, "content": [text, image_url], "text_summary": <fallback>}`` (run_agent.py /
the Anthropic adapter turn it into provider-specific image tool content).

Landing info (macOS-native backend)
-----------------------------------
Every mutating action result carries `front_app` (name, bundle_id, pid),
`focused` (AX role, window title, and for `type` whether the field's value
now ends with the typed text — never the value itself) and
`front_app_changed`. The model must check `front_app`/`focused` in every
result before continuing: a click that lands on the desktop or a window
behind the intended one is only visible there. `action='focused_element'`
returns the same information without acting.
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import json
import logging
import os
import re
import sys
import threading
import uuid
from collections import namedtuple
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement, image_dimensions_from_bytes
from tools.computer_use.stall import StallDetector

logger = logging.getLogger(__name__)

# ── Approval & safety ───────────────────────────────────────────────────────
_approval_callback = None

def set_approval_callback(cb) -> None:
    """Register the CLI approval prompt (terminal_tool pattern); ``cb(action, args, summary)`` ->
    "approve_once" | "approve_session" | "always_approve" | "deny"."""
    global _approval_callback
    _approval_callback = cb

# Actions that read, not mutate. Always allowed.
_SAFE_ACTIONS = frozenset({
    "capture", "wait", "list_apps", "list_windows", "focused_element", "zoom", "move",
    # verify_state only reads accessibility state and window geometry. It is
    # the answer to "did that work?", and putting an approval prompt in front
    # of checking your own work would train the user to click through the
    # prompts that matter.
    "verify_state",
})

# Actions that mutate user-visible state. Go through approval.
_DESTRUCTIVE_ACTIONS = frozenset({
    "click", "double_click", "right_click", "middle_click",
    "drag", "scroll", "type", "key", "set_value", "focus_app",
    # invoke_menu is a real invocation — "Delete" and "Quit" are menu items
    # like any other. It is safer than a click in every way except this one,
    # so it earns the same gate rather than a lighter one.
    "invoke_menu",
})

# Hard-blocked regardless of approval level (e.g. logout kills the session Hermes runs in). Alt is
# canonicalized to option, so the Windows variants are blocked before any backend sees them.
# See #4562. The Force Quit combos are Moe's addition: a force-quit dialog or a silent force-quit of
# the front app is not something a background assistant may ever send.
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}), frozenset({"cmd", "option", "backspace"}),  # empty trash / force delete
    frozenset({"cmd", "ctrl", "q"}), frozenset({"cmd", "shift", "q"}),                    # lock screen / log out
    frozenset({"cmd", "option", "shift", "q"}), frozenset({"win", "l"}),                  # force log out / lock
    frozenset({"cmd", "option", "escape"}), frozenset({"cmd", "option", "esc"}),          # Force Quit dialog
    frozenset({"cmd", "option", "shift", "escape"}), frozenset({"cmd", "option", "shift", "esc"}),  # force-quit front app
    frozenset({"ctrl", "option", "delete"}), frozenset({"ctrl", "option", "del"}), frozenset({"option", "f4"}),
}
_KEY_ALIASES = {"command": "cmd", "control": "ctrl", "alt": "option", "⌘": "cmd", "⌥": "option",
                "windows": "win", "super": "win", "meta": "win"}
_BLOCKED_TYPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (  # dangerous shell patterns for `type` (last one: fork bomb)
    r"curl\s+[^|]*\|\s*bash", r"curl\s+[^|]*\|\s*sh", r"wget\s+[^|]*\|\s*bash",
    r"\bsudo\s+rm\s+-[rf]", r"\brm\s+-rf\s+/\s*$", r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}")]

def _canon_key_combo(keys: str) -> frozenset:
    # Split on "+" AND "-": cua-driver accepts hyphenated combos, so "ctrl-alt-delete" would bypass otherwise.
    return frozenset(_KEY_ALIASES.get(p, p) for p in (q.strip().lower() for q in re.split(r"\s*[+\-]\s*", keys)) if p)

def _reject_unsafe(action: str, args: Dict[str, Any]) -> Optional[str]:
    """JSON error for hard-blocked input, else None. Runs BEFORE the approval prompt."""
    if action == "type" and (pat := next((p.pattern for p in _BLOCKED_TYPE_PATTERNS if p.search(args.get("text", ""))), None)):
        return json.dumps({"error": f"blocked pattern in type text: {pat!r}",
                           "hint": "Dangerous shell patterns cannot be typed via computer_use."})
    if action == "key" and (blocked := next((b for b in _BLOCKED_KEY_COMBOS
                                             if b.issubset(_canon_key_combo(args.get("keys", "")))), None)) is not None:
        return json.dumps({"error": f"blocked key combo: {sorted(blocked)}", "hint": "Destructive system shortcuts are hard-blocked."})
    if args.get("bring_to_front") and args.get("delivery_mode") != "foreground":
        return json.dumps({"error": "bring_to_front requires delivery_mode='foreground'",
                           "code": "bring_to_front_requires_foreground"})
    # Argument validation before the dialog, for the same reason the blocked key/type patterns are checked
    # above it: a malformed path should not cost the user an approval prompt and then error anyway.
    if action == "invoke_menu" and (err := _bad_menu_path(args)) is not None:
        return err
    if action == "verify_state" and (err := _bad_predicate(args)) is not None:
        return err
    return None

def _bad_menu_path(args: Dict[str, Any]) -> Optional[str]:
    path = args.get("path")
    if isinstance(path, list) and path and all(isinstance(p, str) and p.strip() for p in path):
        return None
    return json.dumps({"error": "invoke_menu requires path=['Menu', 'Item'] — a non-empty list of exact menu labels",
                       "code": "bad_menu_path"})

def _bad_predicate(args: Dict[str, Any]) -> Optional[str]:
    expect = args.get("expect")
    if not (isinstance(expect, list) and 1 <= len(expect) <= 8 and all(isinstance(e, dict) for e in expect)):
        return json.dumps({"error": "verify_state requires expect=[{...}] — 1 to 8 predicate objects, ANDed together",
                           "code": "bad_predicate"})
    for knob in ("timeout_ms", "stable_samples"):
        v = args.get(knob)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            return json.dumps({"error": f"verify_state {knob} must be an integer", "code": "bad_predicate"})
    return None

def _input_target_mismatch(backend, requested_app: str) -> Optional[str]:
    """Current sticky-target app when it provably differs from *requested_app*: both known and neither a substring
    of the other ('Google-chrome' vs 'chrome'). Unknown target -> None (fail open; the verify ladder catches it)."""
    last_app = getattr(backend, "_last_app", None)
    current, wanted = (last_app or "").strip().lower(), requested_app.strip().lower()
    return None if not current or not wanted or wanted in current or current in wanted else last_app

# ── Backend selection — env-swappable for tests ─────────────────────────────
# Per-Hermes-session cached backends (own cua-driver session, native target, refs, grant namespace).
_backend_lock = threading.Lock()
_backend: Optional[ComputerUseBackend] = None  # backward-compatible empty-session injection hook (older tests)
_backends: Dict[str, ComputerUseBackend] = {}
_backend_call_locks: Dict[str, threading.RLock] = {}
_backend_permission_modes: Dict[str, str] = {}
_AUX_VISION_ROUTE_CACHE: Dict[Tuple[str, str], bool] = {}  # process-scoped: (provider, model) → bool
# Per-session stall history. Keyed the same way as _backend_call_locks so a
# second tenant can never inherit another session's streak — this module is
# imported once per process and serves every session on it.
_stall_detectors: Dict[str, StallDetector] = {}
# Approval state keyed by session_id so a gateway serving concurrent sessions can't leak one run's
# "always approve" into another; callers without a session_id share "".
# Falls back to a shared "" bucket for callers that don't pass a session_id (e.g. the classic single-run
# CLI). Values: _session_auto_approve[sid] -> bool   ("always_approve everything") _always_allow[sid]
# -> set of (action, delivery_mode) scope keys See NousResearch/hermes-agent#67052 gap 4.
_approval_lock = threading.Lock()
_session_auto_approve: Dict[str, bool] = {}   # sid -> "always_approve everything"
_always_allow: Dict[str, set] = {}            # sid -> set of (action, delivery_mode) scope keys
_escalation_warned: set = set()               # sids already warned that a bypass widened the driver mode

def _stall_detector(session_id: str) -> StallDetector:
    """The session's detector, created on first use.

    Deliberately not under ``_backend_lock``: that lock is held across
    ``backend.start()``, a multi-second cua-driver spawn, and this needs none
    of it. ``setdefault`` on a plain dict is atomic under the GIL, and the
    detector guards its own history.
    """
    det = _stall_detectors.get(session_id)
    if det is None:
        det = _stall_detectors.setdefault(session_id, StallDetector())
    return det

def _cua_permission_mode(session_id: str) -> str:
    """Map Hermes's approval bypass onto Cua's immutable mode; fails closed. Both identity namespaces are consulted
    (DB ``session_id`` and gateway ``session_key`` contextvar) or a gateway ``/yolo`` would be invisible here.
    Warns once per session that ``-z``/``--yolo`` swapped the driver onto a private ``unrestricted`` daemon, dropping
    the configured ceiling: deliberate (``unrestricted`` is not a config value) but easy to trigger by accident."""
    # Configured cua mode (standard | bounded; "standard" if unresolvable). bounded needs a
    # computer_use.capability_manifest — the backend fails loudly without it.
    configured = "standard"
    with contextlib.suppress(Exception):
        from tools.computer_use.cua_backend import _cua_configured_permission_mode
        configured = _cua_configured_permission_mode()
    with contextlib.suppress(Exception):
        from tools.approval import is_approval_bypass_active_for_session
        from tools.approval_context import get_current_session_key
        if is_approval_bypass_active_for_session(session_id) or (
                bool(key := get_current_session_key(default="")) and is_approval_bypass_active_for_session(key)):
            with _approval_lock:
                warn = (key := str(session_id or "")) not in _escalation_warned
                _escalation_warned.add(key)
            if warn:
                logger.warning(
                    "computer_use: approval bypass (--yolo / -z) escalated the cua-driver permission mode from the "
                    "configured '%s' to 'unrestricted' for this session. Runtime approval prompts are disabled and the "
                    "driver's residual ceilings no longer apply. Drop the bypass flag to keep '%s', or declare a "
                    "version-3 computer_use.capability_manifest to keep a ceiling on bypassed runs.", configured, configured)
            return "unrestricted"
    return configured

def _configured_backend_name() -> str:
    """``computer_use.backend`` from config.yaml (``auto`` when unset)."""
    try:
        from hermes_cli.config import load_config
        raw = ((load_config() or {}).get("computer_use") or {}).get("backend", "auto")
    except Exception:
        raw = "auto"
    return str(raw or "auto").strip().lower()

def _select_backend_name() -> str:
    """Which backend to construct.

    ``$HERMES_COMPUTER_USE_BACKEND`` wins (tests, one-off runs), then
    ``computer_use.backend`` in config.yaml. ``auto`` (the default) means
    cua-driver when its binary is installed, else — on macOS only — the
    dependency-free native backend (``screencapture`` + CGEvent + System
    Events), so a Mac without cua-driver still gets eyes and hands.
    """
    env = (os.environ.get("HERMES_COMPUTER_USE_BACKEND") or "").strip().lower()
    name = env or _configured_backend_name()
    if name != "auto":
        return name
    try:
        from tools.computer_use.cua_backend import cua_driver_binary_available
        if cua_driver_binary_available():
            return "cua"
    except Exception:
        pass
    from tools.computer_use.macos_native_backend import native_backend_available
    return "macos" if native_backend_available() else "cua"

def _new_backend(permission_mode: str) -> ComputerUseBackend:
    backend_name = _select_backend_name()
    if backend_name in {"cua", "cua-driver", ""}:
        from tools.computer_use.cua_backend import CuaDriverBackend
        return CuaDriverBackend(permission_mode=permission_mode)
    if backend_name in {"macos", "macos-native", "native"}:
        from tools.computer_use.macos_native_backend import MacNativeBackend
        return MacNativeBackend()
    if backend_name != "noop":
        raise RuntimeError(f"Unknown HERMES_COMPUTER_USE_BACKEND={backend_name!r}")
    return _NoopBackend()  # pragma: no cover

def _install_backend(sid: str, backend: ComputerUseBackend, permission_mode: str) -> ComputerUseBackend:
    """Record a backend in the session caches (the empty session also mirrors it onto the ``_backend`` hook).
    Caller holds ``_backend_lock``."""
    global _backend
    _backends[sid], _backend_permission_modes[sid] = backend, permission_mode
    _backend_call_locks[sid] = threading.RLock()
    _backend = backend if sid == "" else _backend
    return backend

def _detach_locked(sid: str) -> Tuple[Optional[ComputerUseBackend], Optional[threading.RLock]]:
    """Remove one session's cache entries, plus the ``_backend`` injection hook when it aliases the empty session
    (older callers/tests may populate only the hook). Caller holds ``_backend_lock``."""
    global _backend
    _backend_permission_modes.pop(sid, None)
    backend, call_lock = _backends.pop(sid, None), _backend_call_locks.pop(sid, None)
    if sid == "":
        backend = _backend if backend is None else backend
        _backend = None if _backend is backend else _backend
    return backend, call_lock

def _stop_backend(backend: ComputerUseBackend, call_lock: Optional[threading.RLock], on_error: Callable[[Exception], None]) -> None:
    """Stop under the session call lock (if any) so an in-flight action finishes first. Never called under
    ``_backend_lock`` (unrelated sessions stay free). ``on_error`` absorbs the failure (never raises)."""
    try:
        with call_lock if call_lock is not None else contextlib.nullcontext():
            backend.stop()
    except Exception as e:
        on_error(e)

def _get_backend(session_id: str = "") -> ComputerUseBackend:
    sid = str(session_id or "")
    while True:
        with _backend_lock:
            # Mode resolved under the cache lock; YOLO mutation never holds the approval lock while releasing it.
            permission_mode = _cua_permission_mode(sid)
            if sid == "" and _backend is not None and sid not in _backends:
                _install_backend(sid, _backend, permission_mode)  # fold the injection hook into the cache
            if (cached := _backends.get(sid)) is None:
                backend = _new_backend(permission_mode)
                backend.start()  # under the cache lock: one backend per session; a concurrent toggle releases it
                return _install_backend(sid, backend, permission_mode)
            if _backend_permission_modes.get(sid, "standard") == permission_mode:
                return cached
            # Cua's mode is immutable after daemon startup: a /yolo toggle replaces only this session's backend.
            _, stale_lock = _detach_locked(sid)  # stopped outside the cache lock; the loop re-reads the mode first
        _stop_backend(cached, stale_lock, lambda e: None)

def release_computer_use_session(session_id: str) -> bool:
    """Release one session-owned backend (lifecycle seam for hosts/plugins); idempotent, True iff one was released.
    Cache entries are removed BEFORE stopping so new lookups cannot retain the stale target/ref namespace; approval
    state is cleared even without a backend."""
    sid = str(session_id or "")
    with _backend_lock:
        backend, call_lock = _detach_locked(sid)
        # Otherwise a turn that ended on three identical captures leaves a live count behind, and the NEXT turn's
        # first capture — different request, changed screen — arrives already at the advisory tier.
        _stall_detectors.pop(sid, None)
    with _approval_lock:
        _session_auto_approve.pop(sid, None), _always_allow.pop(sid, None)
    if backend is None:
        return False
    _stop_backend(backend, call_lock,
                  lambda e: logger.debug("computer_use backend release failed for session %s", sid, exc_info=True))
    return True

@atexit.register
def _shutdown_backend_atexit() -> None:
    """Stop all cached backends so cua-driver subprocesses don't outlive us. atexit only, no signal handlers: a
    ``SystemExit`` from a prompt_toolkit key binding corrupts its coroutine state and makes the process unkillable.
    Never raises. Drops the global lock before stop(): teardown budgets 5s and must not block spawns.

    Each session backend holds a long-lived ``cua-driver`` subprocess, so without this a driver can survive
    the Hermes process that spawned it (#28152 item 3). #69903 kept the orphan from burning a core by
    disabling the cursor overlay; the process itself still lingered.
    """
    global _backend
    with _backend_lock:
        unique = {id(b): (b, _backend_call_locks.get(sid)) for sid, b in _backends.items()}
        if _backend is not None:
            unique.setdefault(id(_backend), (_backend, _backend_call_locks.get("")))
        _backend = None
        _backends.clear(), _backend_call_locks.clear(), _backend_permission_modes.clear(), _stall_detectors.clear()
    with _approval_lock:
        _session_auto_approve.clear(), _always_allow.clear(), _escalation_warned.clear()
    for backend, call_lock in unique.values():
        _stop_backend(backend, call_lock, lambda e: logger.debug("cua-driver atexit teardown failed: %s", e))

def reset_backend_for_tests() -> None:  # pragma: no cover — tear down the cached backend and per-session state
    _shutdown_backend_atexit()
    _AUX_VISION_ROUTE_CACHE.clear()

def _noop_stub(name: str, *params: str, result: Any = None):
    # Recording stub: positional args are folded in under *params* (declared params default to None). ``result`` may
    # be a factory of the recorded kwargs; None -> a trivial ok ActionResult.
    def method(self, *pos, **kw):
        self.calls.append((name, call := {**dict.fromkeys(params), **dict(zip(params, pos)), **kw}))
        return result(call) if callable(result) else ActionResult(ok=True, action=name) if result is None else result
    return method

class _NoopBackend(ComputerUseBackend):  # pragma: no cover
    """Test/CI stub (HERMES_COMPUTER_USE_BACKEND=noop). Records ``(name, kwargs)`` calls; returns trivial results."""

    def __init__(self) -> None: self.calls: List[Tuple[str, Dict[str, Any]]] = []
    start = stop = lambda self: None
    def is_available(self) -> bool: return True

    capture = _noop_stub("capture", "mode", "app", "pid", "window_id", result=lambda kw: CaptureResult(
        mode=kw["mode"] or "som", width=1024, height=768, png_b64=None, elements=[], app=kw["app"] or "", window_title=""))
    click, drag, scroll = _noop_stub("click"), _noop_stub("drag"), _noop_stub("scroll")
    type_text, key, set_value = _noop_stub("type", "text"), _noop_stub("key", "keys"), _noop_stub("set_value", "value", "element")
    list_apps, list_windows = _noop_stub("list_apps", result=[]), _noop_stub("list_windows", result=[])
    focus_app = _noop_stub("focus_app", "app", "raise_window")

# ── steps: several actions, one round-trip ──────────────────────────────────
# A batch is a round-trip saver, not a macro language: small enough that the
# approval summary stays readable and a failure is easy to locate.
_MAX_STEPS = 8
# Rounds of a `steps` batch under an `until`. Eight steps × ten rounds is
# eighty actions in one round-trip, which is already far past anything a
# person would sit through; the point is to remove the round-trip, not to
# hand over the afternoon.
_MAX_ROUNDS = 10

# What may be repeated.
#
# The approval gate sees a `steps` call ONCE. It flattens the batch and judges
# each action, the person answers, and the call runs — so under `max_rounds` a
# single "yes" would authorise up to ten of everything in it. A Return inside
# a looping batch is ten messages sent on one approval, which is precisely the
# hole the gate exists to close.
#
# So looping is confined to navigation: the actions whose whole purpose is
# repetition and which send nothing. `key` is allowed only for keys that move
# a view — return/enter is what SENDS in mail, messages and chat windows, and
# it is not on this list on purpose. Anything else, run as a normal batch and
# decide for yourself between rounds.
_REPEATABLE = frozenset({
    "scroll", "wait", "capture", "verify_state", "move", "zoom",
    "list_windows", "key",
})
_REPEATABLE_KEYS = frozenset({
    "up", "down", "left", "right", "pageup", "page_up", "pagedown",
    "page_down", "home", "end", "tab", "space",
})

def _as_jsonable(out: Any) -> Any:
    """One step's result, flattened for the batch payload.

    A capture returns a `_multimodal` dict whose image cannot ride inside a
    JSON string; keep its summary and drop the picture rather than corrupt the
    payload. A model that needs the image should capture on its own.
    """
    if isinstance(out, dict):
        if out.get("_multimodal"):
            return {"summary": out.get("text_summary", ""),
                    "meta": out.get("meta", {}),
                    "note": "image omitted inside a batch; capture alone to see it"}
        return out
    if isinstance(out, str):
        try:
            return json.loads(out)
        except Exception:
            return {"text": out[:2000]}
    return {"text": str(out)[:2000]}

def _step_failed(out: Any) -> bool:
    """Whether a step's result should halt the batch.

    Deliberately conservative: anything that is not clearly a success stops
    the run. A batch that continues past an ambiguous result is exactly the
    "assume it worked" failure this whole area keeps paying for.
    """
    payload = _as_jsonable(out)
    if not isinstance(payload, dict):
        return False
    if payload.get("error") is not None:
        return True
    if payload.get("ok") is False:
        return True
    verdict = payload.get("verdict")
    if isinstance(verdict, dict):
        if verdict.get("decision") in ("stop_and_report", "escalate"):
            return True
    # verify_state answering "no" or "cannot tell" is a reason to stop and
    # look, not to run the next three actions anyway.
    if payload.get("action") == "verify_state" and payload.get("verified") is not True:
        return True
    return False

def _run_steps_once(steps, names, kwargs):
    """One pass over the batch. Returns the results list, or a JSON string
    when a step failed and the whole call is already shaped as a refusal.

    Split out of the round loop so that halt-on-first-failure keeps meaning
    exactly what it did before `until` existed: a later step assumes the
    earlier one landed, and that is as true on round four as on round one.
    """
    results = []
    for index, step in enumerate(steps):
        out = handle_computer_use(dict(step), _nested=True, **kwargs)
        results.append(out)
        if _step_failed(out):
            return json.dumps({
                "ok": False,
                "action": "steps",
                "completed": index,
                "of": len(steps),
                "failed_step": {"index": index, "action": names[index]},
                "results": [_as_jsonable(r) for r in results],
                "verdict": {
                    "decision": "stop_and_report",
                    "hint": (f"Step {index + 1} of {len(steps)} "
                             f"({names[index]}) failed, so the remaining "
                             f"{len(steps) - index - 1} were not run. The "
                             f"screen is part-way through what you "
                             f"planned: look before you continue."),
                },
            })
    return results

def _until_met(until, kwargs) -> bool:
    """Ask verify_state whether the batch can stop.

    Reuses the predicate machinery rather than growing a second one — and
    `unknown` is not success here for the same reason it is not there: the
    absence of evidence that something appeared is not evidence that it did.
    """
    raw = handle_computer_use({"action": "verify_state", "expect": until}, _nested=True, **kwargs)
    try:
        body = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return False
    return isinstance(body, dict) and body.get("verified") is True

def _handle_steps(args: Dict[str, Any], kwargs: Dict[str, Any]) -> str:
    """`steps`: several actions, one round-trip.

    Measured elsewhere (OSWorld-Human): planning and reflection are 75-94%
    of a computer-use task's wall clock and grounding is 2-4%. The win is
    fewer model round-trips, not faster clicking — which is what Astra's
    code-execution mode and Anthropic's batched computer tool both buy.

    Each step RE-ENTERS handle_computer_use rather than reaching the backend
    directly. That is the whole safety argument: blocked type/key patterns,
    argument validation, the approval gate and the stall detector all apply
    to every step exactly as they would to a single call, so batching adds
    no new bypass surface inside Hermes. (Moe's own pre-tool hook sits
    outside this and must iterate `steps` itself — a batch would otherwise
    be a way to smuggle a gated Return past a guard that keys on the
    outer action name.)
    """
    steps = args.get("steps")
    if not (isinstance(steps, list) and 1 <= len(steps) <= _MAX_STEPS
            and all(isinstance(st, dict) for st in steps)):
        return json.dumps({
            "error": f"steps requires a list of 1-{_MAX_STEPS} action "
                     f"objects, e.g. [{{'action':'invoke_menu','path':[…]}}]",
            "code": "bad_steps",
        })
    names = [str(st.get("action") or "").strip().lower() for st in steps]
    if "steps" in names:
        # Nesting would multiply the cap and make the approval summary lie about what is being authorised.
        return json.dumps({"error": "steps cannot contain steps", "code": "bad_steps"})
    if any(not n for n in names):
        return json.dumps({"error": "every step needs an `action`", "code": "bad_steps"})

    # `until` + `max_rounds`: the batch repeats until a predicate holds.
    #
    # This is the round-trip lever, and the reason it is DECLARATIVE
    # rather than a script. Astra's equivalent is exec_py/exec_js — the
    # model writes code with real loops — and that would be the wrong
    # shape here for one specific reason: every control in this stack
    # inspects what an action DOES, and a script is opaque to all of them.
    # The send gate flattens `steps` and judges each one; it cannot judge
    # a for-loop. Three times this month a new verb walked past a guard
    # that enumerated the old ones, and an arbitrary code executor over
    # somebody's screen is that mistake with the volume turned up.
    #
    # So: the same actions, repeated, with a predicate deciding when to
    # stop. "Scroll until the Join button exists" is one round-trip
    # instead of eight, and every individual step is still a step the
    # gate, the approval prompt and both stall detectors can read.
    until = args.get("until")
    if until is not None and not (isinstance(until, list) and until):
        return json.dumps({
            "error": "until must be a non-empty list of verify_state predicates, e.g. [{'element': {...}}]",
            "code": "bad_steps"})
    try:
        max_rounds = int(args.get("max_rounds", 1))
    except (TypeError, ValueError):
        max_rounds = -1
    if max_rounds < 1 or max_rounds > _MAX_ROUNDS:
        return json.dumps({"error": f"max_rounds must be 1..{_MAX_ROUNDS}", "code": "bad_steps"})
    if max_rounds > 1:
        for name, step in zip(names, steps):
            if name not in _REPEATABLE:
                return json.dumps({
                    "error": (f"`{name}` cannot be repeated: a looping batch is approved once and would run it "
                              f"up to {max_rounds} times. Repeatable: " + ", ".join(sorted(_REPEATABLE))),
                    "code": "bad_steps"})
            if name == "key":
                keys = str(step.get("keys") or "").lower()
                parts = [k for k in re.split(r"[+\-\s]+", keys) if k]
                if not parts or any(k not in _REPEATABLE_KEYS for k in parts):
                    return json.dumps({
                        "error": ("`key` may only be repeated for keys that move a view ("
                                  + ", ".join(sorted(_REPEATABLE_KEYS)) + "); return/enter is what sends."),
                        "code": "bad_steps"})
    if until is None and max_rounds > 1:
        return json.dumps({
            "error": "max_rounds needs an `until` — repeating a fixed batch with no stopping condition is a loop "
                     "with no exit",
            "code": "bad_steps"})

    results: List[Any] = []
    rounds = 0
    for _round in range(max_rounds):
        rounds += 1
        round_results = _run_steps_once(steps, names, kwargs)
        if isinstance(round_results, str):
            return round_results          # a step failed; already shaped
        results = round_results
        if until is None:
            break
        if _until_met(until, kwargs):
            return json.dumps({
                "ok": True, "action": "steps",
                "completed": len(steps), "of": len(steps),
                "rounds": rounds, "until_met": True,
                "results": [_as_jsonable(r) for r in results],
            })
    if until is not None:
        return json.dumps({
            "ok": False, "action": "steps",
            "completed": len(steps), "of": len(steps),
            "rounds": rounds, "until_met": False,
            "code": "until_unmet",
            "results": [_as_jsonable(r) for r in results],
            "verdict": {
                "decision": "stop_and_report",
                "hint": (f"Ran the batch {rounds} time(s) and the condition never held. Do not simply raise "
                         f"max_rounds — look at the screen and say what is actually there, because the predicate "
                         f"may be describing something that is not going to appear."),
            },
        })
    return json.dumps({
        "ok": True, "action": "steps",
        "completed": len(steps), "of": len(steps),
        "rounds": rounds,
        "results": [_as_jsonable(r) for r in results],
    })

# ── Dispatch ────────────────────────────────────────────────────────────────
def handle_computer_use(args: Dict[str, Any], _nested: bool = False, **kwargs) -> Any:
    """Main entry point (tools.registry): a JSON string (text-only) or a dict marked `_multimodal`. Order: steps
    (re-entered per step) -> hard blocks + argument validation (_reject_unsafe) -> stall detector (budget, repeats,
    looking without acting) -> approval scopes (destructive action, then 'bring_to_front' — persistent focus is a
    separate visible side effect with its own scope) -> backend -> dispatch under the session call lock."""
    action = (args.get("action") or "").strip().lower()
    if not action:
        return json.dumps({"error": "missing `action`"})
    session_id = str(kwargs.get("session_id") or "")  # approval-state / daemon-mode isolation key
    if action == "steps":
        return _handle_steps(args, kwargs)
    if (err := _reject_unsafe(action, args)) is not None:
        return err

    # Stall check first: every input action is destructive, so leaving this
    # below the approval gate showed the user a dialog on their Mac, took
    # their approval, and only then refused the call.
    detector = _stall_detector(session_id)

    # The budget, first, because it is the only check that fires on work which
    # looks fine. The two below it read the shape of what has been tried; this
    # one only counts, which is why it catches the case they cannot — varied,
    # plausible, and going nowhere for five minutes. See stall.BUDGET_HARD.
    #
    # Ahead of the approval gate for the same reason everything else here is:
    # refusing after showing the user a dialog is the worst order.
    over_budget = None if _nested else detector.budget_reason(action)
    if over_budget is not None:
        logger.warning("computer_use %s refused: budget spent (%d)", action, detector.spent())
        return over_budget

    blocked = detector.block_reason(action, args)
    if blocked is not None:
        logger.warning("computer_use %s refused: repeated identical call", action)
        return blocked

    # Looking without acting. Every screenshot differs, so the repeat detector
    # above cannot see this at all — and it is the shape a real session took
    # on 2026-09-04: seventeen consecutive captures, nothing done between
    # them, 86% of ten minutes spent thinking rather than in a tool.
    looping = detector.looking_loop(action)
    if looping is not None and looping.get("refused"):
        logger.warning("computer_use %s refused: %d looks without acting", action, looping["looks_without_acting"])
        return json.dumps({
            "error": (f"refusing another read: {looping['looks_without_acting']} reads in a row without doing "
                      f"anything."),
            "code": "looking_without_acting",
            "verdict": looping,
        })

    scopes = ([action] if action in _DESTRUCTIVE_ACTIONS else []) + (
        ["bring_to_front"] if args.get("bring_to_front") or (action == "focus_app" and args.get("raise_window")) else [])
    for scope in scopes:
        if (err := _request_approval(scope, args, session_id)) is not None:
            return err
    try:
        backend = _get_backend(session_id=session_id)
    except Exception as e:
        return json.dumps({"error": f"computer_use backend unavailable: {e}",
                           "hint": "If the cua-driver binary is missing, run `hermes computer-use install`. "
                                   "If a Python dependency is missing, the error above shows the exact install command."})
    try:
        with _backend_lock:
            call_lock = _backend_call_locks.setdefault(session_id, threading.RLock())
        with call_lock:
            result = _dispatch(backend, action, args)
    except Exception as e:
        logger.exception("computer_use %s failed", action)
        failed = json.dumps({"error": f"{action} failed: {e}"})
        detector.record(action, args, failed)
        return failed
    detector.record(action, args, result)
    return _attach_stall_advisory(detector, action, args, result)

def _attach_stall_advisory(detector: StallDetector, action: str, args: Dict[str, Any], result: Any) -> Any:
    """Fold the soft-tier warning into the channel the model actually reads.

    A `stall` key on a multimodal envelope is dropped on the floor: the agent
    loop hands the model only ``content`` (or ``text_summary``), so the soft
    tier — the half that is supposed to change strategy *before* total failure
    — was silently inert on exactly the screenshot-bearing captures it exists
    for. `_maybe_follow_capture` already solved this; do what it does.

    Kept alongside the existing ``verdict`` rather than replacing it: the
    verdict says whether *this* action landed, the stall note says the
    *strategy* is not working, and both are true at once.
    """
    note = detector.advisory(action, args) or detector.looking_loop(action)
    if note is None:
        # Nothing is wrong with the strategy, but the road may still be
        # running out. Said late rather than early: "six left" is useful, and
        # "nineteen left" is noise on every call.
        budget = detector.budget_advisory()
        if budget is None:
            return result
        note = {"hint": budget["budget"]["note"]}
    banner = f"[stall] {note['hint']}"

    if isinstance(result, dict):
        result = dict(result)
        result["stall"] = note
        if result.get("_multimodal"):
            content = result.get("content")
            if isinstance(content, list) and content:
                head = content[0]
                if isinstance(head, dict) and isinstance(head.get("text"), str):
                    content = list(content)
                    content[0] = dict(head, text=f"{banner}\n\n{head['text']}")
                    result["content"] = content
            summary = result.get("text_summary")
            if isinstance(summary, str):
                result["text_summary"] = f"{banner}\n\n{summary}"
        return result

    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except Exception:
            return f"{banner}\n\n{result}"
        if isinstance(payload, dict):
            payload["stall"] = note
            return json.dumps(payload)
        return f"{banner}\n\n{result}"
    return result

def _request_approval(action: str, args: Dict[str, Any], session_id: str = "") -> Optional[str]:
    """None if approved, else a JSON error string. Scoped by (action, delivery_mode) AND session_id: foreground
    delivery is a visible focus change, so a background ``approve_session`` must NOT cover it; the blanket
    ``always_approve`` does. No CLI approval wired -> default allow (gateway approval runs one layer out).

    ``always_approve`` (the blanket "auto-approve everything" unlock) still covers foreground, since the
    user explicitly opted into unattended operation. State is keyed on session_id so concurrent runs don't
    leak unlocks into one another. See #67052.
    """
    scope_key = (action, "foreground" if args.get("delivery_mode") == "foreground" else "background")
    # A click needs a capture and coordinates, and each one is a separate
    # visible decision. A menu path is one opaque approval that would
    # otherwise cover every future menu item in the session — approving
    # "Terminal › Quit Terminal" once would silently authorise
    # "File › Delete Everything" later. Scope it by the path itself.
    if action == "invoke_menu":
        path = args.get("path")
        if isinstance(path, list):
            scope_key = scope_key + tuple(str(p) for p in path)
    with _approval_lock:
        if _session_auto_approve.get(session_id) or scope_key in _always_allow.get(session_id, set()):
            return None
    if (cb := _approval_callback) is None:
        return None
    try:
        verdict = cb(action, args, _summarize_action(action, args))
    except Exception as e:
        logger.warning("approval callback failed: %s", e)
        verdict = "deny"
    if verdict in ("approve_session", "always_approve"):
        with _approval_lock:
            _always_allow.setdefault(session_id, set()).add(scope_key)
            if verdict == "always_approve":
                _session_auto_approve[session_id] = True
    if verdict in ("approve_once", "approve_session", "always_approve"):
        return None
    return json.dumps({"error": ("approval prompt timed out — the user did not respond. Silence is not consent; "
                                 "do not retry without the user.") if verdict == "timeout" else "denied by user",
                       "action": action})

def _summarize_action(action: str, args: Dict[str, Any]) -> str:
    fg = " [FOREGROUND — briefly raises the window / changes focus]" if args.get("delivery_mode") == "foreground" else ""
    return _ACTIONS.get(action, _ActionSpec(None)).summarize(action, args, fg)

def _force_kw(fn, args: Dict[str, Any]) -> Dict[str, Any]:
    """``{"force": True}`` when the model asked for it AND the backend's
    method takes it (only macOS-native's focus guards do); else ``{}``."""
    if not args.get("force"):
        return {}
    try:
        import inspect
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if "force" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {"force": True}
    return {}

# --- handlers: (backend, action, args, **delivery) -> ActionResult (_dispatch applies the follow-up capture) or a
#     final str/dict result. `delivery` = delivery_mode + bring_to_front; only input actions use it.

def _xy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Click semantics: a coordinate only counts when its x is set (a bare y is not a point)."""
    return dict(x=coord[0], y=coord[1]) if (coord := args.get("coordinate")) and coord[0] is not None else dict(x=None, y=None)

def _scroll_xy(args: Dict[str, Any]) -> Dict[str, Any]:
    """Scroll semantics: axes are independent — ``coordinate=[null, 100]`` scrolls at y=100 with x unset."""
    coord = args.get("coordinate") or (None, None)
    return dict(x=coord[0] if coord and coord[0] is not None else None,
                y=coord[1] if coord and coord[1] is not None else None)

def _do_click(backend, action, args, button=None, count=1, **delivery):
    return backend.click(element=args.get("element"), **_xy(args), button=button or args.get("button") or "left",
                         click_count=count, modifiers=args.get("modifiers"), **delivery)

def _do_drag(backend, action, args, **delivery):
    src, dst = args.get("from_coordinate"), args.get("to_coordinate")
    if (args.get("from_element") is None or args.get("to_element") is None) and not (src and dst):
        return json.dumps({"error": "drag requires from_coordinate/to_coordinate or from_element/to_element"})
    return backend.drag(from_element=args.get("from_element"), to_element=args.get("to_element"),
                        from_xy=tuple(src) if src else None, to_xy=tuple(dst) if dst else None,
                        button=args.get("button", "left"), modifiers=args.get("modifiers"), **delivery)

def _do_scroll(backend, action, args, **delivery):
    # window_id rides along: a plain scroll in an app owning several windows (Chrome) is otherwise refused with
    # "provide window_id" — the one thing the tool could not express until it took the argument.
    kwargs = dict(direction=args.get("direction", "down"), amount=int(args.get("amount", 3)),
                  element=args.get("element"), **_scroll_xy(args), modifiers=args.get("modifiers"), **delivery)
    if args.get("window_id") is not None:
        kwargs["window_id"] = args.get("window_id")
    return backend.scroll(**kwargs)

def _do_type(backend, action, args, **delivery):
    return backend.type_text(args.get("text", ""), **delivery, **_force_kw(backend.type_text, args))

def _do_key(backend, action, args, **delivery):
    return backend.key(args.get("keys", ""), **delivery, **_force_kw(backend.key, args))

def _do_capture(backend, action, args, **_):
    if (mode := str(args.get("mode", "som"))) not in {"som", "vision", "ax"}:
        return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
    # pid/window_id forwarded only when given so older backends keep their defaults.
    return _capture_response(backend.capture(mode=mode, app=args.get("app"),
                                             **{k: args[k] for k in ("pid", "window_id") if args.get(k) is not None}))

def _do_zoom(backend, action, args, **_):
    region = args.get("region")
    if not (isinstance(region, (list, tuple)) and len(region) == 4):
        return json.dumps({"error": "zoom requires region=[x, y, width, height] in screenshot pixels"})
    if (zoom_fn := getattr(backend, "zoom", None)) is None:
        return json.dumps({"error": "zoom is not supported by this backend; use capture"})
    return _capture_response(zoom_fn(tuple(int(v) for v in region)))

def _do_move(backend, action, args, **_):
    coord = args.get("coordinate") or (None, None)
    move_fn = getattr(backend, "move", None)
    if move_fn is None or coord[0] is None:
        return json.dumps({"error": "move requires coordinate=[x, y] and a backend that supports it"})
    return move_fn(int(coord[0]), int(coord[1]))

def _do_invoke_menu(backend, action, args, **_):
    if (err := _bad_menu_path(args)) is not None:
        return err
    if (fn := getattr(backend, "invoke_menu", None)) is None:
        return json.dumps({"error": "invoke_menu is not supported by this backend",
                           "hint": "The macOS-native fallback backend has no menu API; cua-driver does. "
                                   "Run `hermes computer-use doctor`."})
    return _text_response(fn(args["path"], pid=args.get("pid"), window_id=args.get("window_id")))

def _do_verify_state(backend, action, args, **_):
    if (err := _bad_predicate(args)) is not None:
        return err
    if (fn := getattr(backend, "verify_state", None)) is None:
        return json.dumps({"error": "verify_state is not supported by this backend",
                           "hint": "The macOS-native fallback backend cannot evaluate predicates; cua-driver can. "
                                   "Re-capture and read the result instead."})
    return json.dumps(_verify_payload(fn(args["expect"], pid=args.get("pid"), window_id=args.get("window_id"),
                                         timeout_ms=args.get("timeout_ms"), stable_samples=args.get("stable_samples"))))

def _do_focused_element(backend, action, args, **_):
    if (fe := getattr(backend, "focused_element", None)) is None:
        return json.dumps({"error": "focused_element is not supported by this backend; use capture(mode='ax')"})
    f = fe()
    # Never the field's value — only where the focus is.
    return json.dumps({
        "front_app": {"name": f.get("app", ""), "bundle_id": f.get("bundle_id", ""), "pid": f.get("pid")},
        "focused": {"role": f.get("role", ""), "subrole": f.get("subrole", ""), "window_title": f.get("window_title", "")},
        **({"error": f["error"]} if f.get("error") else {}),
    })

def _do_listing(backend, action, args, key, **_):
    out: Dict[str, Any] = {key: (items := getattr(backend, action)()), "count": len(items)}
    # A backend that differs from cua-driver's contract (the macOS-native fallback) says so with every listing.
    if action == "list_windows" and (note := getattr(backend, "LIST_WINDOWS_NOTE", None)):
        out["note"] = note
    return json.dumps(out)

def _summarize_click(action: str, args: Dict[str, Any], fg: str) -> str:
    where = (f" element #{args['element']}" if args.get("element") is not None
             else f" at {tuple(args['coordinate'])}" if args.get("coordinate") else "")
    return f"{action}{where}{fg}"

def _summarize_menu(action: str, args: Dict[str, Any], fg: str) -> str:
    # Every other destructive action tells the user what it will do; this one used to say only "invoke_menu".
    path = args.get("path")
    return "menu " + " › ".join(str(p) for p in path) if isinstance(path, list) and path else "menu (no path)"

# One `action`. ``input``: native input to the backend's sticky target (gets delivery kwargs + the `app=` mismatch
# guard). ``destructive``: mutates user-visible state -> approval prompt (the rest only read) — kept in step with
# ``_DESTRUCTIVE_ACTIONS``/``_SAFE_ACTIONS`` above. ``summarize(action, args, fg_suffix)`` renders the one-line
# approval prompt.
_ActionSpec = namedtuple("_ActionSpec", "handler input destructive summarize",
                         defaults=(False, False, lambda a, args, fg: a + fg))
_input = partial(_ActionSpec, input=True, destructive=True)

_ACTIONS: Dict[str, _ActionSpec] = {
    "click": _input(_do_click, summarize=_summarize_click),
    "double_click": _input(partial(_do_click, count=2), summarize=_summarize_click),
    "right_click": _input(partial(_do_click, button="right"), summarize=_summarize_click),
    "middle_click": _input(partial(_do_click, button="middle"), summarize=_summarize_click),
    "drag": _input(_do_drag, summarize=lambda a, args, fg: (f"drag {args.get('from_element') or args.get('from_coordinate')} → "
                                                             f"{args.get('to_element') or args.get('to_coordinate')}{fg}")),
    "scroll": _input(_do_scroll, summarize=lambda a, args, fg: f"scroll {args.get('direction', '?')} x{args.get('amount', 3)}{fg}"),
    "type": _input(_do_type,
                   summarize=lambda a, args, fg: f"type {args.get('text', '')[:60]!r}" + ("..." if len(args.get("text", "")) > 60 else "") + fg),
    "key": _input(_do_key, summarize=lambda a, args, fg: f"key {args.get('keys', '')!r}{fg}"),
    "set_value": _input(lambda backend, action, args, **_: (
        json.dumps({"error": "set_value requires `value`"}) if args.get("value") is None
        else backend.set_value(value=str(args["value"]), element=args.get("element")))),
    "focus_app": _ActionSpec(lambda backend, action, args, **_: (
        json.dumps({"error": "focus_app requires `app`"}) if not args.get("app")
        else backend.focus_app(args["app"], raise_window=bool(args.get("raise_window")))), destructive=True,
        summarize=lambda a, args, fg: f"focus {args.get('app', '')!r}" + (" (raise)" if args.get("raise_window") else "")),
    "invoke_menu": _ActionSpec(_do_invoke_menu, destructive=True, summarize=_summarize_menu),
    "verify_state": _ActionSpec(_do_verify_state),
    "capture": _ActionSpec(_do_capture),
    "zoom": _ActionSpec(_do_zoom),
    "move": _ActionSpec(_do_move),
    "focused_element": _ActionSpec(_do_focused_element),
    "wait": _ActionSpec(lambda backend, action, args, **_: _text_response(backend.wait(float(args.get("seconds", 1.0))))),
    "list_apps": _ActionSpec(partial(_do_listing, key="apps")),
    "list_windows": _ActionSpec(partial(_do_listing, key="windows")),
}
# Native input actions deliver to the backend's sticky target; `app=` is NOT a targeting parameter (guard in _dispatch).
_INPUT_ACTIONS = frozenset(a for a, s in _ACTIONS.items() if s.input)

# Unknown actions are never aliased (no repairing bad model output), but the nearest real action is named as guidance.
_ACTION_SUGGESTIONS = {
    "hotkey": "key", "press_key": "key", "keypress": "key", "key_combo": "key", "shortcut": "key", "type_text": "type",
    "input_text": "type", "screenshot": "capture", "get_window_state": "capture", "left_click": "click", "mouse_click": "click",
}

def _dispatch(backend: ComputerUseBackend, action: str, args: Dict[str, Any]) -> Any:
    spec = _ACTIONS.get(action)
    if spec is None:
        return json.dumps({"error": f"unknown action {action!r}" + (f" — did you mean {hint!r}? See the action enum in the tool schema."
                                                                 if (hint := _ACTION_SUGGESTIONS.get(str(action))) else "")})
    # app= guard: input goes to the sticky target from the last capture/focus_app and the backend drops app=
    # silently — refuse a clear mismatch rather than type into the wrong window while reporting ok:true.
    if (spec.input and isinstance(requested_app := args.get("app"), str) and requested_app.strip()
            and (mismatch := _input_target_mismatch(backend, requested_app)) is not None):
        return json.dumps({"ok": False, "action": action, "code": "input_target_mismatch", "error": (
            f"{action} would go to the current target {mismatch!r}, not {requested_app.strip()!r} "
            "— input actions always hit the sticky target from the last capture/focus_app. "
            f"Call capture(app={requested_app.strip()!r}) or focus_app first, then retry.")})
    # delivery_mode / bring_to_front thread through every input action (background → foreground ladder).
    res = spec.handler(backend, action, args, delivery_mode=args.get("delivery_mode"),
                       bring_to_front=bool(args.get("bring_to_front")))
    return res if isinstance(res, (str, dict)) else _maybe_follow_capture(backend, res, bool(args.get("capture_after")))

# ── Response shaping ────────────────────────────────────────────────────────
def _classify_action_result(res: ActionResult) -> Dict[str, Any]:
    """Next ladder step from semantic evidence, in precedence order. Escalation is advisory: it never overrides
    a confirmed effect nor licenses repeating input."""
    if res.effect == "confirmed" or res.verified is True:
        return {"decision": "done"}
    if res.effect == "unverifiable":
        return {"decision": "verify_fresh_state", "hint": ("Input was delivered but not confirmed. Re-capture and check the "
                "result BEFORE any retry — do not repeat the input on an escalation recommendation alone.")}
    if res.effect == "suspected_noop" or not res.ok or res.code is not None:
        return {"decision": "escalate", **({"recommended": res.escalation.get("recommended")}
                                           if isinstance(res.escalation, dict) else {}), "hint": (
            "The input likely did not land. Climb one rung following `recommended`: 'px' → re-issue by coordinate; "
            "'foreground' (or a failed pixel click) → re-issue with delivery_mode='foreground' (separate approval). "
            "Do not predict the rung from the app being Electron/Chromium — react to this signal.")}
    return {"decision": "verify_fresh_state",  # transport success without semantic proof is not proof of effect
            "hint": "Transport succeeded but the effect is unproven. Re-capture and confirm before continuing."}

def _verify_payload(res: ActionResult) -> Dict[str, Any]:
    """A predicate verdict, in its own vocabulary.

    `verify_state` sent no input, so the input-delivery ladder does not apply
    to it. Routing it through the shared payload mapped `unsatisfied` onto
    `suspected_noop`, whose hint tells the model "the input likely did not
    land — re-issue by coordinate, or with delivery_mode='foreground'". For a
    read-only check that answered "no", that is an instruction to click
    somewhere and to pull the user's desktop to another Space, for an action
    that never happened.

    The three values stay three, and `verified` is emitted explicitly —
    including as `null` — because "I could not tell" and "this driver does not
    report it" are different answers and an absent key cannot distinguish them.
    """
    meta = res.meta if isinstance(res.meta, dict) else {}
    status = meta.get("status")
    if not isinstance(status, str):
        status = "unknown"
    decision, hint = {
        "satisfied": ("confirmed",
                      "The predicate holds. This is proof; you do not need a screenshot to confirm it again."),
        "unsatisfied": ("not_yet",
                        "The predicate does not hold. The world is not in the state you expected — decide whether to "
                        "wait longer (raise timeout_ms), act differently, or tell the user. Do NOT re-issue input on "
                        "the strength of this alone: nothing was sent."),
    }.get(status, ("cannot_tell",
                   "The driver could not establish the predicate. UNKNOWN IS NOT SUCCESS and it is not failure "
                   "either — absence of an element is not proof of absence unless the search was exhaustive. Try a "
                   "different selector, a longer timeout_ms, or capture and look."))
    payload: Dict[str, Any] = {
        "ok": res.ok, "action": "verify_state", "status": status,
        # Explicitly present, explicitly nullable.
        "verified": {"satisfied": True, "unsatisfied": False}.get(status),
        "verdict": {"decision": decision, "hint": hint},
    }
    if res.message:
        payload["message"] = res.message
    for key in ("predicates", "elapsed_ms", "samples", "stable"):
        if key in meta:
            payload[key] = meta[key]
    if res.code:
        payload["code"] = res.code
    return payload

def _present(**fields: Any) -> Dict[str, Any]:
    return {k: v for k, v in fields.items() if v}  # only the truthy optional fields, in the given order

# Landing info is hoisted to the top level so it is impossible to miss: where the input went matters more than that
# it was sent.
_LANDING_KEYS = ("front_app", "focused", "front_app_changed", "front_app_before", "target")

def _action_payload(res: ActionResult) -> Dict[str, Any]:
    # cua-driver's structured verdict fields only when returned (None = old driver). ok is transport success;
    # effect/escalation are the semantic verdict.
    meta = res.meta if isinstance(res.meta, dict) else {}
    return {"ok": res.ok, "action": res.action, **_present(message=res.message),
            **{k: v for k in ("verified", "effect", "escalation", "path", "degraded", "delivery_mode", "code")
               if (v := getattr(res, k)) is not None}, **_present(meta=res.meta),
            **{k: meta[k] for k in _LANDING_KEYS if k in meta},
            "verdict": _classify_action_result(res)}

def _text_response(res: ActionResult) -> str:
    return json.dumps(_action_payload(res))

# AX `elements` cap: dense UIs publish 500+ nodes (one capture would exhaust context); the full tree spills to a file.
_DEFAULT_MAX_ELEMENTS = 100
# Some providers reject images below 8x8 before the model sees the result; such captures fall back to text.
_MIN_PROVIDER_IMAGE_DIMENSION = 8
# Some AX trees (Discord/Slack via UIA, Electron chat clients) expose ENTIRE message bodies as labels; uncapped
# they blew the tool-result budget and leaked private chat text. Labels identify a control, not text extraction.
_MAX_ELEMENT_LABEL_CHARS = 120
# Bounded cache trails: every dense capture can spill, and CLI-only sessions never run the gateway's media cleanup.
_MAX_SPILL_FILES = _MAX_CAPTURE_FILES = 20

_DEGRADED_HINTS = {
    "ax_window_unresolved": (
        "the window is live but its accessibility surface could not be "
        "resolved — usual causes are a window on another macOS Space, a "
        "just-launched app whose tree is not built yet, or a canvas/Electron "
        "surface that exposes none. Check list_windows for on_current_space; "
        "if it is elsewhere, read it with mode='vision' rather than 'som'. "
        "If the app only just launched, wait and re-capture once"
    ),
    "minimized_or_hidden_window": (
        "the window is minimized or hidden, so it has no on-screen surface "
        "to read or click. focus_app(raise_window=true) un-minimizes it"
    ),
}

# The same reasons, read from inside mode='vision'.
#
# In vision mode an empty element list is the CONTRACT, not a symptom:
# CaptureResult documents "vision → png_b64 only". So the som/ax advice — "read
# it with mode='vision'" — arrives as a circle, telling the model to do the
# thing it has just done. That is the failure mode from §3.2 of Moe's
# docs/COMPUTER-USE.md wearing a friendlier face: not a wrong identifier, but
# advice whose precondition is already met, which reads as actionable and is
# not. Say instead what would actually change the answer.
_DEGRADED_HINTS_VISION = {
    "ax_window_unresolved": (
        "expected in this mode — a vision capture returns the picture and no "
        "element list. But the accessibility surface could not be resolved "
        "either, so 'som' would not have found elements here: the window is "
        "on another macOS Space, only just launched, or is a canvas/Electron "
        "surface with no tree at all. Work from this image with coordinates, "
        "or get the window onto the current Space if you need addressable "
        "elements. Do not re-capture in vision hoping for a list"
    ),
    "minimized_or_hidden_window": (
        "the window is minimized or hidden, so this picture is of nothing "
        "that is on screen. focus_app(raise_window=true) un-minimizes it"
    ),
}

def _degraded_note(reason: Optional[str], total_elements: int, mode: Optional[str] = None) -> Optional[str]:
    """One line explaining a thin capture, or None when there is nothing to say.

    Only speaks up when the elements are actually missing: a full capture that
    happens to carry a degraded flag needs no apology, and a note on every
    successful call is a note nobody reads.

    `mode` matters because the advice does. See _DEGRADED_HINTS_VISION.
    """
    if not reason or total_elements:
        return None
    vision = str(mode or "").strip().lower() == "vision"
    hint = (_DEGRADED_HINTS_VISION if vision else _DEGRADED_HINTS).get(reason)
    if hint:
        return f"  (degraded: {reason} — {hint}.)"
    # An unknown value is still worth surfacing verbatim; the driver may have
    # added one, and "no elements, no reason" is the state this exists to end.
    if vision:
        return (f"  (degraded: {reason} — the driver returned no elements for this window. This is already a "
                f"vision capture, so read the image; for elements try another window from list_windows.)")
    return (f"  (degraded: {reason} — the driver returned no elements for this window. Try mode='vision', or a "
            f"different window from list_windows.)")

def _is_claude_code_profile() -> bool:
    return (os.environ.get("HERMES_MCP_TOOL_PROFILE") or "").strip().lower() == "claude-code"

def _capture_image_format(cap: CaptureResult) -> Tuple[str, str]:
    # (MIME, file extension): cua-driver's explicit MIME type, else sniff the base64 prefix (JPEG starts with /9j/,
    # PNG with iVBOR). The extension matches the on-disk bytes for MIME sniffing.
    mime = cap.image_mime_type or ("image/jpeg" if (cap.png_b64 or "").startswith("/9j/") else "image/png")
    return mime, (".jpg" if mime.lower() == "image/jpeg" else ".png")

def _bounds_unknown(bounds) -> bool:
    # No real geometry: KDE/Qt apps report [0, 0, 0, 0] for elements clickable by index; serializing that as a
    # rect invites coordinate=[0, 0] clicks.
    with contextlib.suppress(TypeError, ValueError):
        return all(int(v) == 0 for v in bounds)
    return False

def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    # A zero rect is "geometry unknown", not a position — null it so no coordinate= is ever derived from it (the index still works).
    return {"index": e.index, "role": e.role, "label": e.label[:_MAX_ELEMENT_LABEL_CHARS],
            "bounds": None if _bounds_unknown(e.bounds) else list(e.bounds), "app": e.app,
            **({"label_truncated": True} if len(e.label) > _MAX_ELEMENT_LABEL_CHARS else {})}

def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out = [f"  #{e.index} {e.role} {e.label.replace(chr(10), ' ')[:60]!r} "
           + ("@ bounds-unknown (click by element index)" if _bounds_unknown(e.bounds) else f"@ {e.bounds}")
           + (f" [{e.app}]" if e.app else "") for e in elements[:max_lines]]
    return out + ([f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)"] if len(elements) > max_lines else [])

def _bounds_hints(elements: List[UIElement], image_width: int, image_height: int) -> Tuple[Optional[float], Optional[str]]:
    """(scale, note) when element bounds live in a different coordinate space than the screenshot, else (None, None).
    On HiDPI displays AX bounds are native while the screenshot is downscaled, so coordinate= clicks read off the
    screenshot miss by the scale factor. 5% slack: window chrome can hang a few px past the captured frame without
    implying a different space. Scale heuristic: larger axis ratio wins."""
    if not elements or image_width <= 0 or image_height <= 0:
        return None, None
    max_x = max_y = 0
    for e in elements:
        try:
            x, y, w, h = e.bounds
        except (TypeError, ValueError):
            continue
        max_x, max_y = max(max_x, int(x) + int(w)), max(max_y, int(y) + int(h))
    if max_x <= image_width * 1.05 and max_y <= image_height * 1.05:
        return None, None
    note = (f"element bounds are in native desktop coordinates (extend to ~{max_x}x{max_y}), "
            f"NOT screenshot pixels ({image_width}x{image_height}). coordinate= clicks expect the native "
            "space — derive click points from element bounds, or scale screenshot positions up accordingly")
    return round(max(max_x / image_width, max_y / image_height), 2), note

_bounds_scale = lambda elements, image_width, image_height: _bounds_hints(elements, image_width, image_height)[0]  # noqa: E731
_bounds_space_note = lambda elements, image_width, image_height: _bounds_hints(elements, image_width, image_height)[1]  # noqa: E731

def _capture_lost_detail(cap: CaptureResult, visible_elements: List[UIElement], truncated_elements: int) -> bool:
    """True when the in-context response drops information the full tree has."""
    return bool(truncated_elements) or any(len(e.label) > _MAX_ELEMENT_LABEL_CHARS for e in visible_elements)

def _capture_view(cap: CaptureResult, max_elements: int) -> SimpleNamespace:
    """One capture's derived facts, computed once for every response branch: ``visible`` is the capped element list,
    ``dims_omitted`` an image below the provider minimum."""
    visible, dims = cap.elements[:max_elements], None
    with contextlib.suppress(Exception):  # (width, height) of the inline PNG/JPEG screenshot, else the backend's
        dims = image_dimensions_from_bytes(base64.b64decode(cap.png_b64, validate=False)) if cap.png_b64 else None
    width, height = dims or (cap.width, cap.height)
    scale, note = _bounds_hints(visible, width, height)
    truncated = len(cap.elements) - len(visible)
    too_small = bool(dims) and min(dims) < _MIN_PROVIDER_IMAGE_DIMENSION
    has_image = bool(cap.png_b64) and cap.mode != "ax" and not too_small
    # Under the claude-code MCP profile the screenshot goes to the model as an image block and nowhere else: a
    # desktop assistant must not leave a rolling cache of the user's screen in ~/.hermes/cache/images.
    return SimpleNamespace(cap=cap, visible=visible, total=len(cap.elements), width=width, height=height,
                           truncated=truncated, bounds_scale=scale, bounds_note=note,
                           # Capped labels / capped element array: spill the complete tree for on-demand reads.
                           elements_file=_spill_elements_to_file(cap) if _capture_lost_detail(cap, visible, truncated) else None,
                           screenshot_path=_persist_capture_image(cap) if has_image and not _is_claude_code_profile() else None,
                           dims_omitted=dims if too_small else None, has_image=has_image)

def _capture_summary_lines(v: SimpleNamespace) -> List[str]:
    """Human-readable capture summary; line ORDER is contract. Lists only what `elements` surfaces, otherwise the
    summary names indices the model can't find."""
    notes = (
        # An empty element list has to say why. "0 interactable element(s)" alone cannot distinguish an app with no
        # accessibility tree from a window on another Space, and the two have different fixes — one needs pixels,
        # the other needs a different window. The driver already knows; it just was never asked.
        (degraded := _degraded_note(v.cap.degraded_reason, v.total, v.cap.mode)) and degraded.strip()[1:-1],
        v.bounds_note and v.bounds_note + (f"; estimated scale ~{v.bounds_scale}x (screenshot position x "
                                           f"{v.bounds_scale} ≈ native coordinate)" if v.bounds_scale else ""),
        v.screenshot_path and f"shareable screenshot saved to {v.screenshot_path}",
        v.cap.note,
        v.elements_file and (f"full element tree with untruncated labels saved to {v.elements_file} — "
                             "read_file/search_files it if you need dropped label text or elements beyond the cap"),
    )
    return [
        f"capture mode={v.cap.mode} {v.width}x{v.height}"
        + (f" app={v.cap.app}" if v.cap.app else "") + (f" window={v.cap.window_title!r}" if v.cap.window_title else ""),
        f"{v.total} interactable element(s):",
        *(f"  ({note})" for note in notes if note),
        *_format_elements(v.visible),
        *([f"  (screenshot omitted: {v.dims_omitted[0]}x{v.dims_omitted[1]} is below the "
           f"{_MIN_PROVIDER_IMAGE_DIMENSION}x{_MIN_PROVIDER_IMAGE_DIMENSION} provider minimum)"] if v.dims_omitted else []),
    ]

def _text_capture_payload(v: SimpleNamespace, summary: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """JSON text payload shared by the AX, vision-unavailable and aux-vision branches. Key order is contract:
    fixed fields, ``extra`` branch markers, then set optionals."""
    return json.dumps({
        "mode": v.cap.mode, "width": v.width, "height": v.height, "app": v.cap.app, "window_title": v.cap.window_title,
        "elements": [_element_to_dict(e) for e in v.visible], "total_elements": v.total, "summary": summary,
        **(extra or {}),
        **_present(truncated_elements=v.truncated, elements_file=v.elements_file, screenshot_path=v.screenshot_path,
                   bounds_scale=v.bounds_scale),
    })

def _capture_response(cap: CaptureResult, max_elements: int = _DEFAULT_MAX_ELEMENTS) -> Any:
    v = _capture_view(cap, max_elements)
    lines = _capture_summary_lines(v)
    summary, extra = "\n".join(lines), None  # multimodal/aux paths use this; text paths append notes and rebuild
    if v.has_image:
        # Hand the screenshot to auxiliary.vision (text-only result) when the main model may not consume images
        # natively; returning the multimodal envelope unconditionally tripped HTTP 404/400 at the provider.
        if not _should_route_through_aux_vision():  # envelope carrying the screenshot (not the elements array, so no truncation note)
            return {
                "_multimodal": True,
                "content": [{"type": "text", "text": summary},
                            {"type": "image_url", "image_url": {"url": f"data:{_capture_image_format(cap)[0]};base64,{cap.png_b64}"}}],
                "text_summary": summary,
                "meta": {"mode": cap.mode, "width": v.width, "height": v.height, "elements": v.total, "png_bytes": cap.png_bytes_len,
                         **_present(screenshot_path=v.screenshot_path, elements_file=v.elements_file, bounds_scale=v.bounds_scale)},
            }
        # Decide whether to hand the screenshot to the auxiliary.vision pipeline (text-only result) or keep
        # the multimodal envelope (main model handles vision natively). Issue #24015: previously the
        # multimodal envelope was returned unconditionally, so non-vision main models tripped HTTP 404 / 400
        # at the provider boundary even when auxiliary.vision was explicitly configured to handle this.
        routed = _route_capture_through_aux_vision(
            cap, summary, visible_elements=v.visible, truncated_elements=v.truncated,
            elements_file=v.elements_file, screenshot_path=v.screenshot_path)
        if routed is not None:
            return routed
        # Aux routing requested but failed (vision node down, empty analysis...): the multimodal envelope could
        # now break with a provider error, so degrade to text.
        lines.append("  (vision unavailable: the auxiliary vision model could not be reached; screenshot "
                     "omitted. Element-index actions still work — drive via the element list above.)")
        extra = {"vision_unavailable": True}
    if v.truncated:  # text paths carry the `elements` array, so the truncation note applies
        lines.append(f"  (response truncated to {len(v.visible)} of {v.total} elements; the full tree is in "
                     "elements_file — read_file/search_files it, or pass app= to narrow scope)")
    return _text_capture_payload(v, "\n".join(lines), extra)

def _maybe_follow_capture(backend: ComputerUseBackend, res: ActionResult, do_capture: bool) -> Any:
    # No follow-up capture after a failed action: a normal-looking screenshot would suggest success.
    if not do_capture or not res.ok:
        return _text_response(res)
    try:
        # Recapture the exact window when known: on Linux several unrelated windows may share an app name, so
        # app-only recapture can switch targets.
        exact = {k: (getattr(backend, "_last_target", None) or {}).get(k) for k in ("pid", "window_id")}
        cap = backend.capture(mode=_capture_after_mode(), **(exact if None not in exact.values()
                                                            else {"app": getattr(backend, "_last_app", None)}))
    except Exception as e:
        logger.warning("follow-up capture failed: %s", e)
        return _text_response(res)
    resp, payload = _capture_response(cap), _action_payload(res)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        # Keep the evidence/verdict contract visible alongside the image — it governs whether input may repeat.
        resp["content"][0]["text"] = resp["text_summary"] = json.dumps(payload) + "\n\n" + resp["text_summary"]
        resp["action_result"] = payload
        return resp
    return json.dumps({**json.loads(resp), **payload})  # text capture: merge the action payload in

# ── Cache files (screenshots, element spills, vision temps) ─────────────────
def _cache_file(subdir: str, legacy: str, name: str, pattern: str = "", cap: int = 0):
    """Path for a new file under ``$HERMES_HOME/<subdir>`` (dir created). With ``pattern``/``cap``, first unlinks the
    oldest matching files so at most ``cap - 1`` remain (best-effort)."""
    from hermes_constants import get_hermes_dir  # lazy so tests can patch get_hermes_dir
    cache_dir = get_hermes_dir(subdir, legacy)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        files = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime) if pattern else []
        for stale in files[: max(0, len(files) - (cap - 1))]:
            stale.unlink(missing_ok=True)
    return cache_dir / name

def _write_cache_file(what: str, subdir: str, legacy: str, name: str, pattern: str, cap: int,
                      write: Callable[[Any], None]) -> Optional[str]:
    """Bounded cache write via ``write(path)``; the path, or None on any failure — an unwritable cache must never
    break control (a capture keeps working without its spill/screenshot copy)."""
    try:
        write(path := _cache_file(subdir, legacy, name, pattern, cap))
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: %s failed: %s", what, exc)
        return None

def _persist_capture_image(cap: CaptureResult) -> Optional[str]:
    """Copy of the capture in Hermes' media cache so attachment surfaces can deliver it (None without an image)."""
    return _write_cache_file(
        "screenshot persistence", "cache/images", "image_cache", f"computer_use_{uuid.uuid4().hex}{_capture_image_format(cap)[1]}",
        "computer_use_*.*", _MAX_CAPTURE_FILES, lambda p: p.write_bytes(base64.b64decode(cap.png_b64, validate=False)),
    ) if cap.png_b64 else None

def _spill_elements_to_file(cap: CaptureResult) -> Optional[str]:
    """FULL element tree (untruncated labels) in a cache file — the read_file/search_files escape hatch for capped text."""
    payload = {"app": cap.app, "window_title": cap.window_title, "total_elements": len(cap.elements),
               "elements": [{"index": e.index, "role": e.role, "label": e.label, "bounds": list(e.bounds), "app": e.app}
                            for e in cap.elements]}
    return _write_cache_file("element spill", "cache/computer_use", "computer_use_cache", f"elements_{uuid.uuid4().hex}.json",
                             "elements_*.json", _MAX_SPILL_FILES,
                             lambda p: p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"))

# ── auxiliary.vision routing for captured screenshots ───────────────────────
_MAX_VISION_DIM = 1456  # longest image side handed to the aux vision model: full-resolution desktop captures tokenize heavily
# and can overflow small local-model context windows; ~1456px keeps SOM badges legible while cutting vision latency.

def _shrink_capture_for_vision(raw: bytes, ext: str, max_dim: int = _MAX_VISION_DIM) -> tuple[bytes, Optional[str]]:
    """Downscale encoded image bytes so the longest side is <= max_dim -> ``(bytes, scale_note)``. note is None when
    unchanged (fits, or Pillow unavailable/failed), else it tells the vision model the factor so reported
    coordinates map back to the real screen instead of being silently wrong."""
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        if max(img.size) <= max_dim:
            return raw, None
        (orig_w, orig_h), out = img.size, BytesIO()
        img.thumbnail((max_dim, max_dim))
        new_w, new_h = img.size
        img.save(out, format="JPEG" if ext == ".jpg" else "PNG")
        fx, fy = (orig_w / new_w if new_w else 1.0), (orig_h / new_h if new_h else 1.0)
        factor_clause = (f"multiply any coordinates you report by {fx:.2f} to map back to the real screen." if f"{fx:.2f}" == f"{fy:.2f}"
                         else f"multiply any x coordinates you report by {fx:.2f} and any y coordinates by {fy:.2f} to map back to the real screen.")
        return out.getvalue(), f"Screenshot downscaled from {orig_w}x{orig_h} to {new_w}x{new_h} for vision; {factor_clause}"
    except Exception as exc:
        logger.debug("computer_use: vision downscale skipped: %s", exc)
        return raw, None

def _should_route_through_aux_vision() -> bool:
    """True when ``_capture_response`` should hand the PNG to aux vision. Any failure returns False (fail open) so a
    broken config never silently drops the screenshot for vision-capable main models.

    Inside the hermes-tools MCP server's ``claude-code`` profile the model on the other end of the bridge is a
    Claude model reading MCP image blocks, so the screenshot must stay in the multimodal envelope regardless of
    what config.yaml says about the provider."""
    if _is_claude_code_profile():
        return False
    stage = "import"
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config
        from tools.computer_use.vision_routing import should_route_capture_to_aux_vision
        stage = "config read"
        provider, model = _read_main_provider() or "", _read_main_model() or ""
        if (cached := _AUX_VISION_ROUTE_CACHE.get(key := (str(provider), str(model)))) is not None:
            return cached
        stage = "decision"
        _AUX_VISION_ROUTE_CACHE[key] = decision = bool(should_route_capture_to_aux_vision(provider, model, load_config()))
        return decision
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing %s failed: %s", stage, exc)
        return False

def _capture_after_mode() -> str:
    """Mode for ``capture_after`` follow-ups. Default ``som`` (screenshot)."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        mode = str(((load_config() or {}).get("computer_use") or {}).get("capture_after_mode", "som") or "som")
        return mode if (mode := mode.strip().lower()) in {"som", "vision", "ax"} else "som"
    return "som"

_VISION_PROMPT = ("Describe what is visible in this desktop application screenshot in concise but specific terms. Mention "
                  "the app name and window title if visible, the overall layout, any labelled buttons, menus or text fields, "
                  "and any prominent text content the user would need to know about. Do not invent details that are not "
                  "actually visible.\n\nAX/SOM index for cross-reference:\n")

def _route_capture_through_aux_vision(cap: CaptureResult, summary: str, *, visible_elements: Optional[List[UIElement]] = None,
                                      truncated_elements: int = 0, elements_file: Optional[str] = None,
                                      screenshot_path: Optional[str] = None) -> Optional[str]:
    """Pre-analyse the capture via ``vision_analyze_tool`` (temp file under ``$HERMES_HOME/cache/vision/``) and merge
    the description with the AX/SOM summary into one text payload. JSON, or None on any failure."""
    if not cap.png_b64:
        return None
    problem, temp_image_path = "aux-vision import failed", None
    try:
        from model_tools import _run_async
        from tools.vision_tools import vision_analyze_tool
        problem = "failed to decode capture base64"
        raw = base64.b64decode(cap.png_b64, validate=False)
        problem = None  # from here on failures are loud (warning)
        ext = _capture_image_format(cap)[1]
        temp_image_path = _cache_file("cache/vision", "temp_vision_images", f"computer_use_{uuid.uuid4().hex}{ext}")
        raw, scale_note = _shrink_capture_for_vision(raw, ext)
        temp_image_path.write_bytes(raw)
        prompt = _VISION_PROMPT + summary + (f"\n\nNote: {scale_note}" if scale_note else "")
        result_json = _run_async(vision_analyze_tool(str(temp_image_path), prompt))
    except Exception as exc:
        if problem:
            logger.debug("computer_use: %s: %s", problem, exc)
        else:
            logger.warning("computer_use: auxiliary.vision pre-analysis failed (%s); "
                           "returning to caller without aux analysis", exc)
        return None
    finally:
        if temp_image_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(str(temp_image_path))
    # The ``analysis`` field of vision_analyze_tool's JSON result; raw text when it isn't JSON; empty -> no merge.
    analysis_text = result_json.strip() if isinstance(result_json, str) else ""
    with contextlib.suppress(TypeError, json.JSONDecodeError):
        parsed = json.loads(analysis_text)
        analysis_text = str(parsed.get("analysis") or "").strip() if isinstance(parsed, dict) else ""
    if not analysis_text:
        return None
    # Same element cap as every other capture branch; dumping cap.elements in full would bypass max_elements
    # exactly for non-vision main models. Dimensions are the backend's on this branch.
    view = SimpleNamespace(cap=cap, visible=cap.elements if visible_elements is None else visible_elements,
                           total=len(cap.elements), width=cap.width, height=cap.height, truncated=truncated_elements,
                           elements_file=elements_file, screenshot_path=screenshot_path, bounds_scale=None)
    return _text_capture_payload(view, summary, {"vision_analysis": analysis_text,
                                                 "vision_analysis_routed_via": "auxiliary.vision"})

# ── Availability check (used by the tool registry check_fn) ─────────────────
def check_computer_use_requirements() -> bool:
    """macOS/Windows/Linux + cua-driver binary (or env override), or — on a Mac without cua-driver — the native
    backend, which needs nothing installed. ``computer_use.enabled: false`` switches the tool off. `hermes
    computer-use doctor` names blocked checks."""
    if sys.platform not in ("darwin", "win32", "linux"):
        return False
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        if ((load_config() or {}).get("computer_use") or {}).get("enabled") is False:
            return False
    from tools.computer_use.cua_backend import cua_driver_binary_available  # via cua_backend: tests patch it there
    if cua_driver_binary_available():
        return True
    if sys.platform == "darwin" and _select_backend_name() in {"macos", "macos-native", "native"}:
        from tools.computer_use.macos_native_backend import native_backend_available
        return native_backend_available()
    return False

def get_computer_use_schema() -> Dict[str, Any]:
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA
    return COMPUTER_USE_SCHEMA


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import struct  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
