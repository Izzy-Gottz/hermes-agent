"""(fork) Run model-written code as a second uid: ``terminal.run_as_user``.

Off by default (empty), so a Mac and upstream behave exactly as before. When set (Memoe's cloud machine
sets ``sandbox``), every command the terminal tool runs, every file-tool operation (the backend then
declares ``uses_host_filesystem = False``, so read_file / write_file go through it), every background
process, and every browser_exec CLI runs as that uid instead of the gateway's:

* through ``sudo -n -u <user> -- /usr/bin/env -i <allowlisted env> <argv>``. The machine's sudoers lets
  the gateway's uid run anything as that user and nothing as anyone else, so the launcher is usable only
  by the gateway, and the target uid has no sudo rights of its own;
* with a clean environment built from an ALLOWLIST (never the gateway's environment minus a blocklist:
  a blocklist is what let ``MOE_MACHINE_SECRET`` reach a terminal child), and that user's own HOME and
  TMPDIR;
* failing CLOSED: a configured user that does not resolve, or no sudo, raises :class:`RunAsUnavailable`
  and nothing runs. It never falls back to running as the gateway.

A different uid also means the gateway cannot signal those processes (``kill(2)`` needs a matching uid).
Measured on Memoe's cloud machine: SIGKILL to the group kills only sudo and orphans the command, so
every kill path also signals the group AS the target user (:func:`kill_group`).
"""
from __future__ import annotations

import logging
import os
import pwd
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

SUDO_CANDIDATES = ("/usr/bin/sudo", "/bin/sudo")
ENV_BIN = "/usr/bin/env"
KILL_BIN = "/bin/kill"

#: Exact names a sandboxed child may inherit when present. Nothing that can carry a credential.
_ALLOW_EXACT = frozenset({"LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "COLUMNS", "LINES",
                          "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "NO_COLOR", "FORCE_COLOR",
                          "GIT_PAGER", "PAGER", "NO_PROXY", "no_proxy", "ANONYMIZED_TELEMETRY",
                          "DISPLAY", "PYTHONPATH"})
#: Prefixes a caller may pass through (the browser harness reads BU_* / BH_*; LC_* is locale).
_ALLOW_PREFIX = ("LC_", "BU_", "BH_")
#: A name that looks like it carries a secret is dropped even when its prefix is allowed.
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASS|CREDENTIAL|AUTH|COOKIE|DSN|PRIVATE|BEARER|SESSION_ID)",
                          re.IGNORECASE)


class RunAsUnavailable(RuntimeError):
    """``terminal.run_as_user`` is set but cannot be honoured; nothing may run."""


@dataclass(frozen=True)
class Target:
    user: str
    uid: int
    gid: int
    home: str
    sudo: str

    @property
    def tmp(self) -> str:
        return os.path.join(self.home, ".tmp")


def configured_user() -> str:
    """``terminal.run_as_user`` (bridged to ``TERMINAL_RUN_AS_USER``), stripped; "" when off."""
    from tools.terminal_scope import terminal_env
    return (terminal_env("TERMINAL_RUN_AS_USER", "") or "").strip()


def _sudo() -> Optional[str]:
    for path in SUDO_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return shutil.which("sudo")


def resolve(user: Optional[str] = None) -> Optional[Target]:
    """The configured target, or None when the feature is off. Raises :class:`RunAsUnavailable`
    when it is on and unusable — the caller must not run anything then."""
    name = configured_user() if user is None else user.strip()
    if not name:
        return None
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        raise RunAsUnavailable(f"terminal.run_as_user is {name!r}, and no such user exists on this machine. "
                               "Nothing was run.") from None
    if pw.pw_uid == 0 or pw.pw_uid == os.getuid():
        raise RunAsUnavailable(f"terminal.run_as_user {name!r} is uid {pw.pw_uid}: it must be a separate, "
                               "unprivileged user, not root and not the gateway's own. Nothing was run.")
    sudo = _sudo()
    if not sudo:
        raise RunAsUnavailable("terminal.run_as_user is set, and sudo is not installed. Nothing was run.")
    return Target(user=name, uid=pw.pw_uid, gid=pw.pw_gid, home=pw.pw_dir or "/", sudo=sudo)


def _safe_path(path: str) -> str:
    """The gateway's PATH minus anything under HERMES_HOME (the target cannot read it, and a path
    there would only name the gateway's private tree), plus the FHS dirs."""
    from hermes_constants import get_hermes_home
    try:
        private = str(get_hermes_home()).rstrip("/") + "/"
    except Exception:
        private = "\0"
    keep = [p for p in (path or "").split(os.pathsep) if p and not (p + "/").startswith(private)]
    for d in ("/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin"):
        if d not in keep:
            keep.append(d)
    return os.pathsep.join(keep)


def allowed_name(name: str) -> bool:
    if not isinstance(name, str) or not name or "=" in name or _SECRET_NAME.search(name):
        return False
    return name in _ALLOW_EXACT or name.startswith(_ALLOW_PREFIX)


def clean_env(target: Target, extra: Optional[Mapping[str, str]] = None,
              base: Optional[Mapping[str, str]] = None) -> dict:
    """The whole environment a sandboxed child gets: an allowlist over ``base`` (default os.environ)
    and ``extra``, then the target's own identity. ``extra`` wins over ``base``; identity wins over both."""
    base = os.environ if base is None else base
    env: dict = {}
    for source in (base, extra or {}):
        for k, v in source.items():
            if allowed_name(k) and isinstance(v, str) and "\0" not in v:
                env[k] = v
    env.update({"HOME": target.home, "USER": target.user, "LOGNAME": target.user, "SHELL": "/bin/bash",
                "TMPDIR": target.tmp, "XDG_CACHE_HOME": os.path.join(target.home, ".cache"),
                "XDG_CONFIG_HOME": os.path.join(target.home, ".config"),
                "PATH": _safe_path((extra or {}).get("PATH") or base.get("PATH", ""))})
    return env


def argv(target: Target, cmd: Sequence[str], env: Mapping[str, str]) -> list:
    """``sudo -n -u <user> -- env -i K=V… <cmd>``. ``-n``: never prompt (a missing rule fails at once);
    ``env -i``: sudo's own env_reset is not relied on to decide what the child sees."""
    return [target.sudo, "-n", "-u", target.user, "--", ENV_BIN, "-i",
            *(f"{k}={v}" for k, v in sorted(env.items())), *cmd]


def run(target: Target, cmd: Sequence[str], *, timeout: float = 15, input: Optional[str] = None,
        extra_env: Optional[Mapping[str, str]] = None) -> subprocess.CompletedProcess:
    """One short helper command as the target (mkdir, rm, the group kill)."""
    return subprocess.run(argv(target, cmd, clean_env(target, extra_env)), input=input, capture_output=True,
                          text=True, timeout=timeout, cwd="/", stdin=None if input is not None else subprocess.DEVNULL)


def ensure_dirs(target: Target, paths: Iterable[str]) -> None:
    """``mkdir -p -m 0700`` each path AS the target (the gateway cannot write into its home)."""
    paths = [p for p in paths if p]
    if not paths:
        return
    r = run(target, ["/bin/mkdir", "-p", "-m", "0700", "--", *paths])
    if r.returncode != 0:
        raise RunAsUnavailable(f"could not prepare {target.user}'s directories: {r.stderr.strip()[:200]}")


def kill_group(target: Target, pgid: Optional[int], sig: int = signal.SIGKILL) -> bool:
    """Signal process group ``pgid`` AS the target user (its members are the target's; the gateway's
    own ``killpg`` reaches only sudo). Never raises; True when the kill command succeeded."""
    if not pgid or pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        name = signal.Signals(sig).name.removeprefix("SIG")
        r = run(target, [KILL_BIN, "-s", name, "--", f"-{int(pgid)}"], timeout=5)
        return r.returncode == 0
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("run_as kill_group(%s, %s) failed: %s", pgid, sig, exc)
        return False


def within(path: str, root: str) -> bool:
    """``path``'s real location is inside ``root``'s (symlinks resolved on both sides)."""
    try:
        real, base = os.path.realpath(path), os.path.realpath(root)
    except OSError:
        return False
    return real == base or real.startswith(base.rstrip("/") + "/")
