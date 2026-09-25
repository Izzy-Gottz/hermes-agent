"""Per-process unlock state for external password managers.

An unlock is a session token minted by the manager's CLI from the master
password (``op signin --raw`` / ``bw unlock --raw``). The token lives in
process memory only, keyed by backend, and expires after an idle TTL or an
explicit lock. The master password itself is consumed by the CLI call and
dropped; nothing is written to disk or env.

The surface owns the prompt: ``set_unlock_prompt_callback`` is installed by
the CLI panel / TUI gateway bridge for the current thread, exactly like the
sudo-password callback. Headless contexts (cron, webhook, api_server,
single-query) install none and the vault stays locked — the same posture
approvals take where nobody can answer.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

_IDLE_TTL_S = 30 * 60

_lock = threading.Lock()
_sessions: Dict[tuple[str, str], tuple[str, float]] = {}   # (profile home, backend) → (token, last_used)
_callback_tls = threading.local()

UnlockPrompt = Callable[[str, str], str]  # (backend_name, display_name) -> master password ("" = cancelled)
# (origin, site label) -> {"identifier": str, "password": str} or None when the user declines. The
# surface owns the masked fields; the tool stores the answer in the local vault and fills at once.
SaveLoginPrompt = Callable[[str, str], Optional[Dict[str, str]]]


def set_unlock_prompt_callback(cb: Optional[UnlockPrompt]) -> None:
    """Register the current surface's masked master-password prompt (per-thread slot)."""
    _callback_tls.prompt = cb


def get_unlock_prompt_callback() -> Optional[UnlockPrompt]:
    return getattr(_callback_tls, "prompt", None)


# (site, hint) -> the one-time code the user reads off their phone/email/app, "" when declined.
CodePrompt = Callable[[str, str], str]


def set_code_prompt_callback(cb: Optional[CodePrompt]) -> None:
    """Register the surface's "enter the code {site} sent you" prompt, per thread."""
    _callback_tls.code = cb


def get_code_prompt_callback() -> Optional[CodePrompt]:
    return getattr(_callback_tls, "code", None)


def set_save_login_prompt_callback(cb: Optional[SaveLoginPrompt]) -> None:
    """Register the surface's "save this login" prompt (identifier + masked password), per thread."""
    _callback_tls.save_login = cb


def get_save_login_prompt_callback() -> Optional[SaveLoginPrompt]:
    return getattr(_callback_tls, "save_login", None)


def _key(backend: str) -> tuple[str, str]:
    # Tokens are profile-scoped: a Desktop gateway hosts several profiles in one process and
    # profile B must never reuse (or lock) profile A's manager session.
    from hermes_constants import get_hermes_home
    return (str(get_hermes_home()), backend)


# Lock generation per key: ``lock()`` bumps it, and an unlock that started before the bump must
# not commit its token afterwards (a slow `bw unlock` child would otherwise silently undo an
# acknowledged Lock).
_generation: Dict[tuple[str, str], int] = {}
# Which gateway session performed the unlock; the token is released when THAT session ends,
# not when any sibling session in the profile is torn down.
_owner_session: Dict[tuple[str, str], Optional[str]] = {}
_current_session_tls = threading.local()


def set_current_session_id(session_id: Optional[str]) -> None:
    """Gateway surfaces bind the session running on this thread so an unlock records its owner."""
    _current_session_tls.sid = session_id


def _live(backend: str, *, touch: bool) -> Optional[str]:
    key = _key(backend)
    with _lock:
        entry = _sessions.get(key)
        if entry is None:
            return None
        token, last = entry
        if time.monotonic() - last > _IDLE_TTL_S:
            del _sessions[key]
            return None
        if touch:
            _sessions[key] = (token, time.monotonic())
        return token


def get_session_token(backend: str) -> Optional[str]:
    """Token for a real manager call; refreshes the idle timer."""
    return _live(backend, touch=True)


def begin_unlock(backend: str) -> int:
    """Snapshot the lock generation before spawning the manager CLI; pass it to ``store_session_token``."""
    with _lock:
        return _generation.get(_key(backend), 0)


def store_session_token(backend: str, token: str, generation: Optional[int] = None) -> bool:
    """Commit an unlock. Returns False (and drops the token) when a Lock happened since ``begin_unlock``."""
    key = _key(backend)
    with _lock:
        if generation is not None and generation != _generation.get(key, 0):
            return False
        _sessions[key] = (token, time.monotonic())
        _owner_session[key] = getattr(_current_session_tls, "sid", None)
        return True


def lock(backend: Optional[str] = None) -> None:
    """Forget the current profile's session for one backend (or all of them when None)."""
    home = _key("")[0]
    with _lock:
        # Bump the generation for every key the lock names (not only the ones holding a token):
        # an unlock that is still running for this backend must see the lock when it returns.
        keys = {k for k in list(_sessions) + list(_generation) if k[0] == home and (backend is None or k[1] == backend)}
        if backend is not None:
            keys.add((home, backend))
        for key in keys:
            _forget(key)


def release_session(session_id: str) -> None:
    """A gateway session ended: drop only the tokens that session unlocked."""
    with _lock:
        for key in [k for k, sid in _owner_session.items() if sid == session_id]:
            _forget(key)


def _forget(key: tuple[str, str]) -> None:
    _sessions.pop(key, None)
    _owner_session.pop(key, None)
    _generation[key] = _generation.get(key, 0) + 1


def lock_all_profiles() -> None:
    """Process shutdown: drop every token."""
    with _lock:
        for key in list(_sessions):
            _forget(key)


def is_unlocked(backend: str) -> bool:
    """Status probe: does NOT extend the idle TTL (only real manager calls do)."""
    return _live(backend, touch=False) is not None


_CALLBACK_GETTERS = {"unlock": lambda: get_unlock_prompt_callback(), "login": lambda: get_save_login_prompt_callback(),
                     "code": lambda: get_code_prompt_callback()}


def _in_process_prompt_ok(kind: Optional[str] = None) -> bool:
    """A surface in THIS process installed the prompt for ``kind`` (any of them when None), and the
    context is one a human can answer (not cron, webhook, api_server or -q)."""
    from tools.approval_context import (
        _is_cron_approval_context,
        _is_single_query_approval_context,
        _is_unattended_platform_approval_context,
    )
    if _is_cron_approval_context() or _is_unattended_platform_approval_context() or _is_single_query_approval_context():
        return False
    kinds = [kind] if kind in _CALLBACK_GETTERS else list(_CALLBACK_GETTERS)
    return any(_CALLBACK_GETTERS[k]() is not None for k in kinds)


def can_prompt_here(kind: Optional[str] = None) -> bool:
    """Whether the person can be asked for a secret here: an in-process surface prompt for ``kind``
    (``unlock`` / ``login`` / ``code``; any when None), or the host's own prompt program
    (agent/vault_backends/secret_prompt.py) with the person live at this Mac. False where no human can
    answer (cron, a relayed message, a turn from the person's phone)."""
    if _in_process_prompt_ok(kind):
        return True
    from agent.vault_backends.secret_prompt import external_prompt_ready
    return external_prompt_ready()[0]


def _external(kind: str):
    from agent.vault_backends.secret_prompt import external_prompt_ready, run_prompt
    ready, why = external_prompt_ready()
    return (run_prompt if ready else None), why


def _site(origin: str) -> str:
    return (origin or "").split("://", 1)[-1]


def resolve_save_login_prompt(*, label: str = "", username: str = "", message: str = "") -> tuple:
    """``(prompt, why)``: a ``SaveLoginPrompt`` (origin, host) -> {"identifier", "password"} | None,
    or ``(None, why nobody can be asked)``. The surface's own prompt first, then the host's program."""
    cb = get_save_login_prompt_callback()
    if cb is not None and can_prompt_here("login"):
        return cb, ""
    run, why = _external("login")
    if run is None:
        return None, why

    def ask(origin: str, host: str) -> Optional[Dict[str, str]]:
        answer = run("login", origin=origin, label=label or host or _site(origin), username=username,
                     message=message or (f"Type your login for {host or _site(origin)}. It is saved straight "
                                         "into your saved passwords; the assistant never sees it."))
        if not answer.get("ok"):
            return None
        out = {"identifier": answer.pop("username"), "password": answer.pop("password")}
        answer.clear()
        return out
    return ask, ""


def resolve_code_prompt(*, message: str = "") -> tuple:
    """``(prompt, why)``: a ``CodePrompt`` (site, hint) -> code ("" = declined), or ``(None, why)``."""
    cb = get_code_prompt_callback()
    if cb is not None and can_prompt_here("code"):
        return cb, ""
    run, why = _external("code")
    if run is None:
        return None, why

    def ask(site: str, hint: str) -> str:
        answer = run("code", origin=f"https://{site}" if site and "://" not in site else site, label=site,
                     message=message or hint or f"Type the code {site} sent you. It goes straight into the page; the assistant never sees it.")
        return str(answer.get("code") or "") if answer.get("ok") else ""
    return ask, ""


def resolve_unlock_prompt() -> tuple:
    """``(prompt, why)``: an ``UnlockPrompt`` (backend, display_name) -> master password, or ``(None, why)``."""
    cb = get_unlock_prompt_callback()
    if cb is not None and can_prompt_here("unlock"):
        return cb, ""
    run, why = _external("unlock")
    if run is None:
        return None, why

    def ask(backend: str, display_name: str) -> str:
        answer = run("unlock", label=display_name,
                     message=f"Unlock {display_name} so your saved logins can be used. The assistant never sees the password.")
        return str(answer.pop("password", "") or "") if answer.get("ok") else ""
    return ask, ""
