"""The host's own secret prompt: a program Hermes runs to ask the person for a login, a code or a
master password where no in-process prompt exists.

Why: in Moe every chat reaches Hermes as ``api_server`` and the tools run inside the hermes-tools
MCP server, a subprocess spawned for the claude-code runtime. Nothing in that process can draw a
window, so the in-process prompt callbacks (``unlock.set_*_prompt_callback``, installed only by the
CLI and the TUI gateway) are never there, and every save-login / code / unlock prompt answered
``prompt_unavailable``. The host can draw one: it names the program in ``HERMES_SECRET_PROMPT_CMD``.

The contract, kept deliberately small (the host implements the other side exactly):

* Hermes runs the program by its absolute path, with NO arguments, and writes ONE JSON object to its
  stdin, then closes it::

      {"kind": "login" | "code" | "unlock",
       "origin": "https://example.com",       # "" for unlock
       "label": "example.com",                 # what to call the site / the manager
       "username": "me@example.com",           # optional prefill (login only)
       "message": "…",                         # one or two plain sentences for the person
       "fields": ["username", "password"] | ["code"] | ["password"]}

* The program writes ONE JSON object to stdout and exits::

      {"ok": true, "username": "…", "password": "…"}    # kind=login, exit 0
      {"ok": true, "code": "…"}                          # kind=code,  exit 0
      {"ok": true, "password": "…"}                      # kind=unlock, exit 0
      {"ok": false, "reason": "cancelled" | "timeout"}   # exit 1

* Hermes waits at most :data:`TIMEOUT_S` (300 s), then kills it: ``timeout``. Anything else --
  non-zero exit with ``ok: true``, output that is not one JSON object, a missing field -- is
  ``bad_output`` and nothing is stored.

Secrets travel only on the program's stdout, into this process's memory. They are never put in
argv or the environment (the request carries none), never logged (not stdout, not stderr, not the
exception text of a failed parse), and the caller drops them as soon as the vault has them.

Whether the program may run at all is not this module's call alone: :func:`external_prompt_ready`
asks :func:`tools.browser_chrome_extension.at_this_mac` -- a live turn the person started at this
Mac, with the Mac in use. A cron job, a relayed message, a turn from the person's phone: no window.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_CMD = "HERMES_SECRET_PROMPT_CMD"
TIMEOUT_S = 300.0

KINDS = {"login": ["username", "password"], "code": ["code"], "unlock": ["password"]}


def prompt_command() -> Optional[str]:
    """The configured program, when it is an absolute path to an executable file."""
    cmd = (os.environ.get(ENV_CMD) or "").strip()
    if not cmd or not os.path.isabs(cmd) or not os.path.isfile(cmd) or not os.access(cmd, os.X_OK):
        return None
    return cmd


def external_prompt_ready() -> Tuple[bool, str]:
    """``(True, why)`` when the host's prompt may be put in front of the person now."""
    if prompt_command() is None:
        return False, "this host has no secret prompt"
    try:
        from tools.browser_chrome_extension import at_this_mac
        return at_this_mac()
    except Exception as exc:  # pragma: no cover -- defensive: unknown presence is not presence
        logger.debug("secret prompt: presence unknown (%s)", type(exc).__name__)
        return False, "whether the person is at the Mac could not be established"


def _child_env() -> Dict[str, str]:
    # The request carries no secret, and the environment passed on carries none either: the parent's
    # environment minus nothing secret-bearing that Hermes itself set. Inherit it as is so the host's
    # program finds its own home; Hermes never ADDS a secret to it.
    return dict(os.environ)


def run_prompt(kind: str, *, origin: str = "", label: str = "", username: str = "", message: str = "",
               timeout: float = TIMEOUT_S) -> Dict[str, Any]:
    """Run the host's prompt for ``kind``. Returns the program's answer when it is well formed
    (``{"ok": True, ...fields}``) or ``{"ok": False, "reason": ...}``. Never raises, never logs values."""
    cmd = prompt_command()
    if cmd is None:
        return {"ok": False, "reason": "unavailable"}
    if kind not in KINDS:
        return {"ok": False, "reason": "bad_request"}
    request: Dict[str, Any] = {"kind": kind, "origin": origin or "", "label": label or "", "message": message or "",
                               "fields": list(KINDS[kind])}
    if username and kind == "login":
        request["username"] = username
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no arguments at all
            [cmd], input=json.dumps(request), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, env=_child_env())
    except subprocess.TimeoutExpired:
        logger.info("secret prompt: %s timed out after %.0fs", kind, timeout)
        return {"ok": False, "reason": "timeout"}
    except OSError as exc:
        logger.info("secret prompt: could not start (%s)", type(exc).__name__)
        return {"ok": False, "reason": "unavailable"}
    answer, exited_ok = _parse(proc.stdout), proc.returncode == 0
    del proc  # stdout and stderr both may hold the secret: gone before anything else happens
    if answer is None:
        logger.info("secret prompt: %s answered with something that is not one JSON object", kind)
        return {"ok": False, "reason": "bad_output"}
    if answer.get("ok") is True and not exited_ok:
        answer.clear()
        logger.info("secret prompt: %s said ok but exited non-zero", kind)
        return {"ok": False, "reason": "bad_output"}
    result = _validate(kind, answer)
    if not result.get("ok"):
        logger.info("secret prompt: %s not answered (%s)", kind, result.get("reason"))
    return result


def _parse(stdout: Optional[str]) -> Optional[Dict[str, Any]]:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _validate(kind: str, answer: Dict[str, Any]) -> Dict[str, Any]:
    if answer.get("ok") is not True:
        reason = answer.get("reason")
        answer.clear()
        return {"ok": False, "reason": reason if reason in ("cancelled", "timeout") else "cancelled"}
    wanted: List[str] = KINDS[kind]
    out: Dict[str, Any] = {"ok": True}
    for key in wanted:
        value = answer.get(key)
        if not isinstance(value, str) or not (value if key == "password" else value.strip()):
            answer.clear()
            return {"ok": False, "reason": "bad_output"}
        out[key] = value.strip() if key != "password" else value
    answer.clear()
    return out
