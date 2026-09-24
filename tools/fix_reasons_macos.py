"""macOS producers for ``tools.fix_reasons``: the evidence that says WHICH fixable cause a failure is.

Slice 3 of docs/FIXABLE-FAILURES.md (Moe repo). Each function here reads one kind of real evidence and
answers with a ``FixMessage`` (the person-facing sentence plus the contract fields), or ``None`` when the
evidence does not prove the cause. ``None`` is the common answer and the safe one: a failure that is
not proven fixable keeps its own words, because a Fix card for the wrong cause is how a person grants
the wrong app, or turns on a permission that was never the problem.

- ``tcc_files``: macOS privacy protection on Desktop, Documents, Downloads and iCloud Drive. Only
  ``EPERM`` ("Operation not permitted") is TCC. ``EACCES`` ("Permission denied") is ownership or mode
  and never gets a code. An EPERM on a file is not proof either (``chflags uchg`` gives EPERM too), so
  the proof is opening the protected folder itself: macOS refuses the listing of a TCC-protected folder
  with EPERM, and file flags never block listing a folder.
- ``tcc_automation``: an Apple event refused with -1743 (errAEEventNotPermitted: the person said
  Don't Allow, or turned it off) or -1744 (errAEEventWouldRequireUserConsent: macOS has to ask first
  and could not ask this time).
- ``tcc_driver_accessibility`` / ``tcc_driver_screen`` / ``driver_not_running``: cua-driver's own
  words, taken from the 0.28.2 binary (``strings /Applications/CuaDriver.app/Contents/MacOS/cua-driver``).
"""

from __future__ import annotations

import errno
import os
import re
import sys
from typing import Any, Iterable, Optional, Tuple

from tools.fix_reasons import (
    DRIVER_NOT_RUNNING, TCC_AUTOMATION, TCC_DRIVER_ACCESSIBILITY, TCC_DRIVER_SCREEN, TCC_FILES,
    FixMessage, fix_message, host_app_name,
)

# ── tcc_files ────────────────────────────────────────────────────────────────

#: (the name a person knows the folder by, its path under $HOME, the System Settings anchor).
#: Desktop, Documents and Downloads each have their own toggle under Privacy & Security › Files &
#: Folders, which is the narrowest grant that fixes the failure (and the one macOS itself asks for).
#: iCloud Drive has no reliable Files & Folders row of its own, so it names Full Disk Access.
PROTECTED_FOLDERS: Tuple[Tuple[str, str, str], ...] = (
    ("Desktop", "Desktop", "Privacy_FilesAndFolders"),
    ("Documents", "Documents", "Privacy_FilesAndFolders"),
    ("Downloads", "Downloads", "Privacy_FilesAndFolders"),
    ("iCloud Drive", os.path.join("Library", "Mobile Documents"), "Privacy_AllFiles"),
)

#: How every grant message ends. Nothing watches for the grant yet, so promising to carry on would let
#: the turn end on a promise; the person asks again. Slice 4 (GrantWatch) changes this one line to
#: "then I'll carry on", once the app watches the grant and resumes the task.
THEN = "then ask me to try again"

_PANE_WORDS = {
    "Privacy_FilesAndFolders": "Files & Folders",
    "Privacy_AllFiles": "Full Disk Access",
}


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _absolute(path: str, cwd: Optional[str]) -> str:
    p = os.path.expanduser(str(path))
    if not os.path.isabs(p):
        p = os.path.join(cwd or os.path.expanduser("~"), p)
    return os.path.normpath(p)  # never realpath: resolving would itself touch the protected folder


def protected_folder(path: Any, cwd: Optional[str] = None) -> Optional[Tuple[str, str, str]]:
    """``(folder name, folder root, pane)`` when ``path`` is inside a TCC-protected home folder."""
    if not path or not isinstance(path, (str, os.PathLike)):
        return None
    p = _absolute(os.fspath(path), cwd)
    home = os.path.normpath(os.path.expanduser("~"))
    for name, rel, pane in PROTECTED_FOLDERS:
        root = os.path.join(home, rel)
        if p == root or p.startswith(root + os.sep):
            return name, root, pane
    return None


def _probe_dirs(root: str, path: str) -> Iterable[str]:
    """The folder itself, then (for iCloud Drive) the container under it: iCloud Drive's protection sits
    on ``Mobile Documents/com~apple~CloudDocs`` and its siblings, not on the parent."""
    yield root
    rest = os.path.relpath(path, root)
    first = rest.split(os.sep, 1)[0]
    if first not in (".", "..", "") and root.endswith("Mobile Documents"):
        yield os.path.join(root, first)


def tcc_denied_folder(path: Any, cwd: Optional[str] = None) -> Optional[Tuple[str, str, str]]:
    """macOS only: ``(folder, root, pane)`` when privacy protection refuses THIS process the protected
    folder that holds ``path``. The proof is opening the folder: EPERM there is TCC. EACCES (a mode or
    ownership problem), a missing folder, or a folder that opens are all ``None``."""
    if not _is_darwin():
        return None
    hit = protected_folder(path, cwd)
    if hit is None:
        return None
    for d in _probe_dirs(hit[1], _absolute(os.fspath(path), cwd)):
        try:
            with os.scandir(d) as it:
                next(it, None)
        except OSError as e:
            if e.errno == errno.EPERM:
                return hit
            return None
    return None


def files_denied(folder: str, path: str, pane: str) -> FixMessage:
    """The ``tcc_files`` error for a path inside ``folder``."""
    app = host_app_name()
    where = _PANE_WORDS.get(pane, "Full Disk Access")
    turn_on = (f"turn on {folder} for {app} in System Settings › Privacy & Security › {where}"
               if pane == "Privacy_FilesAndFolders"
               else f"turn {app} on in System Settings › Privacy & Security › {where}")
    return fix_message(
        f"{_cap(app)} isn't allowed into your {folder} folder, so it couldn't open "
        f"{os.path.basename(path.rstrip(os.sep)) or folder}. macOS is blocking it, not the file itself, "
        f"so sudo or changing the file's permissions won't help. To fix it, {turn_on}, {THEN}.",
        TCC_FILES, pane=pane, subject=folder, retry=True, path=path)


def files_denied_for(path: Any, cwd: Optional[str] = None) -> Optional[FixMessage]:
    """``tcc_files`` for ``path`` when macOS is refusing its protected folder, else ``None``."""
    hit = tcc_denied_folder(path, cwd)
    return files_denied(hit[0], _absolute(os.fspath(path), cwd), hit[2]) if hit else None


def files_denied_from_exc(exc: BaseException, path: Any = None, cwd: Optional[str] = None) -> Optional[FixMessage]:
    """``tcc_files`` for an exception raised by a file operation: only an EPERM, and only when the folder
    probe confirms it (an immutable file on the Desktop is EPERM too, and is not a permission)."""
    if not isinstance(exc, OSError) or exc.errno != errno.EPERM:
        return None
    return files_denied_for(getattr(exc, "filename", None) or path, cwd)


_EPERM_TEXT = re.compile(r"operation not permitted", re.I)
_QUOTED = re.compile(r"'([^'\n]+)'|\"([^\"\n]+)\"|‘([^’\n]+)’|`([^`\n]+)'")


def _path_candidates(line: str) -> Iterable[str]:
    for m in _QUOTED.finditer(line):
        yield next(g for g in m.groups() if g)
    # "ls: Desktop: Operation not permitted", "cat: /Users/a/Documents/x: Operation not permitted",
    # "zsh: operation not permitted: ./x", "find: /Users/a/Library/Mobile Documents: Operation not permitted"
    for field in line.split(": "):
        field = field.strip().rstrip(".,;")
        if field and not _EPERM_TEXT.search(field) and "[Errno" not in field:
            yield field


_SEPARATOR_CHARS = frozenset(";&|\n()")
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "|&", ";;", "(", ")", "()", "{", "}"})


def _command_tokens(command: str) -> list:
    """Shell words and operators of ``command`` (quotes resolved), or [] if it cannot be tokenised."""
    import shlex
    try:
        # newline is a command separator OUTSIDE quotes only: it is punctuation, not whitespace, so
        # a quoted ``sh -c "…\n…"`` string keeps its own lines for the recursion to split
        lex = shlex.shlex(command or "", posix=True, punctuation_chars="();<>|&\n")
        lex.whitespace = " \t\r"
        lex.whitespace_split = True
        return list(lex)
    except ValueError:
        return []


def _segments(command: str) -> list:
    """The simple commands of ``command``, split on ``; && || | &`` and grouping, each as argv."""
    segs, cur = [], []
    for tok in _command_tokens(command):
        if tok in _SEPARATORS or (tok and set(tok) <= _SEPARATOR_CHARS):
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        segs.append(cur)
    return segs


def _command_touches(command: str, cand: str, cwd: Optional[str]) -> bool:
    """Whether the COMMAND itself operates on ``cand``: one of its words IS that path (after ~ and cwd).
    A path that only appears in output (an old log, a test's expected text), or only as part of a
    longer argument (``~/Documents/build.log`` is not ``~/Documents``), is not evidence, and must
    never cause the first touch of a protected folder."""
    if not command:
        return False
    target = _absolute(cand, cwd).rstrip(os.sep)
    return any(_absolute(tok, cwd).rstrip(os.sep) == target
               for seg in _segments(command) for tok in seg if tok and not tok.startswith("-"))


def files_denied_in_text(text: str, cwd: Optional[str] = None, command: str = "") -> Optional[FixMessage]:
    """``tcc_files`` from a failed command's output: a line saying "Operation not permitted" that names
    a path inside a protected folder which the command itself addressed, confirmed by the folder probe
    AFTER that failure (the command touched the folder first, never the probe)."""
    if not _is_darwin() or not text or not _EPERM_TEXT.search(text):
        return None
    for line in text.splitlines():
        if not _EPERM_TEXT.search(line):
            continue
        for cand in _path_candidates(line):
            if (protected_folder(cand, cwd) and _command_touches(command, cand, cwd)
                    and (fix := files_denied_for(cand, cwd)) is not None):
                return fix
    return None


# ── tcc_automation ───────────────────────────────────────────────────────────

# osascript: "execution error: Not authorized to send Apple events to Notes. (-1743)"
# JXA:       "execution error: Error: Error: Not authorized to send Apple events to Notes. (-1743)"
_AE_REFUSED = re.compile(
    r"Not authori[sz]ed to send Apple events to (?P<app>[^\n]+?)\.?\s*(?:\((?P<num>-174[34])\)|$)", re.M)
_AE_NUMBER = re.compile(r"\((?P<num>-174[34])\)")
_TELL_APP_BARE = re.compile(r"""tell\s+application\s+([A-Za-z][\w.]*)\b""", re.I)
_TELL_APP = re.compile(r"""tell\s+application\s+(?:id\s+)?["“]([^"”]+)["”]|Application\(\s*['"]([^'"]+)['"]\s*\)""", re.I)


_WRAPPERS = frozenset({"env", "sudo", "time", "command", "exec", "nohup", "caffeinate"})
# wrapper flags that take a value (``sudo -u bob``, ``env -u VAR``)
_WRAPPER_VALUE_FLAGS = {"sudo": {"-u", "-g", "-p", "-C", "-h", "-U", "-r", "-t", "-D"},
                        "env": {"-u", "-C", "-P", "-S"}}
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh"})
_REDIRECTS = frozenset({"<", ">", ">>", "<<", "<<<", ">&", "<&", "&>", "&>>", ">|", "2>", "2>>", "1>"})
_SCRIPT_FILE = re.compile(r"\.(?:scpt|scptd|applescript)$")
_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_]\w*)\2([^\n]*)\n(.*?)\n[ \t]*\3[ \t]*(?=\n|$)", re.S)
_HEREDOC_TOKEN = re.compile(r"^__HERMES_HEREDOC_(\d+)__$")


def _lift_heredocs(command: str) -> Tuple[str, list]:
    """``command`` with each heredoc body replaced by a placeholder word, and the bodies. Newlines
    separate commands, which would otherwise shred a heredoc into commands."""
    bodies: list = []

    def sub(m):
        bodies.append(m.group(5))
        return f"<< __HERMES_HEREDOC_{len(bodies) - 1}__{m.group(4)}"
    return _HEREDOC.sub(sub, command or ""), bodies


def _unwrap(argv: list) -> list:
    """argv without leading ``VAR=x`` assignments and env/sudo/time/... wrappers (with their flags)."""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if "=" in tok and not tok.startswith(("-", "/", ".")):
            i += 1  # FOO=1 osascript ...
            continue
        name = os.path.basename(tok)
        if name not in _WRAPPERS:
            break
        i += 1
        takes_value = _WRAPPER_VALUE_FLAGS.get(name, set())
        while i < len(argv) and (argv[i].startswith("-") or ("=" in argv[i] and name == "env")):
            i += 2 if argv[i] in takes_value else 1
    return argv[i:]


def _osascript_invocation(argv: list, bodies: list = (), depth: int = 0
                          ) -> Optional[Tuple[list, Optional[str], str]]:
    """``(argv, script path or None, script text)`` when this simple command RUNS osascript (argv[0]
    after unwrapping), executes a script file directly, or is ``bash/sh/zsh -c '…'`` whose own last
    command does; else None. Redirections are not arguments: ``osascript <<EOF`` takes its script
    from the heredoc body, ``osascript < f.applescript`` from the file."""
    rest = _unwrap(argv)
    if not rest:
        return None
    exe = os.path.basename(rest[0])
    if exe in _SHELLS and depth < 3:
        for k, tok in enumerate(rest[1:-1], start=1):
            if not tok.startswith("-"):
                return None  # a script name: every -c after it is the script's argument, not the shell's
            if not tok.startswith("--") and "c" in tok[1:]:
                return _invocation_of(rest[k + 1], depth + 1)
        return None
    if exe != "osascript":
        return (rest, rest[0], "") if _SCRIPT_FILE.search(rest[0]) else None
    script_file, text, j, args = None, "", 0, rest[1:]
    while j < len(args):
        tok = args[j]
        if tok in _REDIRECTS:
            target = args[j + 1] if j + 1 < len(args) else ""
            if tok == "<<" and (m := _HEREDOC_TOKEN.match(target)) and int(m.group(1)) < len(bodies):
                text += bodies[int(m.group(1))]
            elif tok == "<<<":
                text += target
            elif tok == "<" and target:
                script_file = target
            j += 2
            continue
        if tok in ("-e", "-l", "-s"):
            if tok == "-e" and j + 1 < len(args):
                text += " " + args[j + 1]
            j += 2
            continue
        if not tok.startswith("-") and script_file is None:
            script_file = tok
            break
        j += 1
    return rest, script_file, text


def _invocation_of(command: str, depth: int = 0) -> Optional[Tuple[list, Optional[str], str]]:
    """The osascript invocation that is ``command``'s LAST simple command, or None. A ``sh -c``
    string is its own command: its heredocs are lifted when the recursion reaches it."""
    lifted, bodies = _lift_heredocs(command)
    segs = _segments(lifted)
    return _osascript_invocation(segs[-1], bodies, depth) if segs else None


def _script_targets(inv: Tuple[list, Optional[str], str], cwd: Optional[str]) -> set:
    """The apps the script tells, from its ``-e`` / heredoc text or its readable plain-text file."""
    _argv, script_file, text = inv
    if script_file and not text.strip():
        try:
            with open(_absolute(script_file, cwd), "rb") as fh:
                raw = fh.read(1 << 20)
            text = raw.decode("utf-8") if b"\0" not in raw else ""  # a compiled .scpt is binary
        except (OSError, UnicodeDecodeError):
            text = ""
    told = {next(g for g in m.groups() if g) for m in _TELL_APP.finditer(text)}
    # A script that went through ``bash -c "…"`` reaches osascript with its inner quotes eaten by
    # the outer shell: ``tell application Notes``. Still a named target.
    told |= {m.group(1) for m in _TELL_APP_BARE.finditer(text)}
    return told


# Every AppleScript error osascript prints: "<pos>: execution error: <message> (<number>)", and a
# compile failure, "<pos>: syntax error: <message> (-2741)".
_EXEC_ERROR = re.compile(r"(?:execution|syntax) error: [^\n]*\((-?\d+)\)")


def automation_denied_in_command(command: str, output: str, cwd: Optional[str] = None) -> Optional[FixMessage]:
    """``tcc_automation`` for a failed terminal command, only on this evidence:

    - the command's LAST simple command runs osascript (its argv[0] after env/sudo/time wrappers,
      also inside ``bash -c '…'``; not a word in a grep pattern or a filename) or executes a script
      file, so the failing exit status is that invocation's own;
    - the output has exactly ONE of macOS's full "Not authorized to send Apple events to <App>" lines
      (the terminal merges stdout and stderr, so an old refusal printed by an earlier ``cat`` cannot
      be told apart from a new one; two lines is ambiguity, not evidence), and no OTHER osascript
      ``execution error`` line: a -1728 beside an old -1743 means the -1743 is not this run's;
    - the script's text is readable (``-e``, a heredoc, a plain-text file) and the refused app is one
      it tells (``tell application "X"`` / JXA ``Application("X")``). A script that names no app —
      ``osascript -e 'error "Not authorized to send Apple events to X." number -1743'`` — can print
      the refusal for any X it likes, and a genuinely signed card would then name it.
    """
    inv = _invocation_of(command)
    if inv is None:
        return None
    text = output or ""
    refusals = list(_AE_REFUSED.finditer(text))
    if len(refusals) != 1:
        return None
    if any("Not authori" not in m.group(0) for m in _EXEC_ERROR.finditer(text)):
        return None
    targets = _script_targets(inv, cwd)
    if refusals[0].group("app").strip() not in targets:
        return None
    # -1744 ("macOS needed to ask and couldn't") from the model's own script is not shown: the
    # app confirms a card with macOS, and "not yet asked" is true of every app the person has
    # never automated — a script telling any such app would get a card. Hermes's own scripts
    # (the native backend) still report it.
    if (refusals[0].group("num") or "-1743") == "-1744":
        return None
    # Targets were checked above against the script itself (a file's text, too).
    return automation_denied_in_text(refusals[0].group(0))


def automation_denied_in_text(text: str, script: str = "") -> Optional[FixMessage]:
    """``tcc_automation`` from osascript / JXA stderr, the target app parsed out of the message (or, when
    macOS gives only the number, out of the script's ``tell application "X"``).

    -1743 is a refusal the person can turn around in Settings › Automation. -1744 means macOS needed to
    ask and could not (a background process, or the dialog could not be shown); there is no switch to
    flip yet, so the words say "try again while you're at your Mac", and ``consent`` tells the host
    which of the two it is. Both are the same grant, the same owner and the same pane, so both are
    ``tcc_automation`` rather than one of them being dropped."""
    if not text or not _is_darwin():
        return None
    target, num = None, None
    m = _AE_REFUSED.search(text)
    if m:
        target, num = m.group("app").strip(), m.group("num") or "-1743"
    elif (n := _AE_NUMBER.search(text)) is not None:
        num = n.group("num")
    else:
        return None
    if not target and script and (t := _TELL_APP.search(script)) is not None:
        target = next(g for g in t.groups() if g)
    if script:
        # With the script in hand, the refused app must be one it tells: its own text can raise
        # a -1743 naming any app at all.
        told = {next(g for g in m.groups() if g) for m in _TELL_APP.finditer(script)}
        if not target or target not in told:
            return None
    target = target or "that app"
    app = host_app_name()
    if num == "-1744":
        msg = (f"macOS needs to ask you before {app} can control {target}, and it couldn't ask just now. "
               f"Try again while you're at your Mac and choose Allow when macOS asks. If it doesn't ask, turn on "
               f"{target} under {app} in System Settings › Privacy & Security › Automation, {THEN}.")
        consent = "not_asked"
    else:
        msg = (f"{_cap(app)} isn't allowed to control {target}. To fix it, turn on {target} under {app} in "
               f"System Settings › Privacy & Security › Automation, {THEN}.")
        consent = "denied"
    return fix_message(msg, TCC_AUTOMATION, subject=target, retry=True, consent=consent, ae_error=int(num))


# ── cua-driver: its grants, and whether it is running ────────────────────────

DRIVER_NAME = "CuaDriver"

# Verbatim from the cua-driver 0.28.2 binary. The refusal codes sit in a compiled match table as
# 16-byte heads and tails ("accessibility_pe" + "permission_denied", "tcc_permission_d" +
# "ermission_denied"); "screen_recording_permission_denied" is also stored whole.
_DRIVER_AX_CODES = frozenset({"accessibility_permission_denied"})
_DRIVER_SCREEN_CODES = frozenset({"screen_recording_permission_denied"})
_DRIVER_EITHER_CODES = frozenset({"tcc_permission_denied", "permissions_pending"})
_DRIVER_AX_TEXT = (
    "Accessibility permission not granted",
    "Accessibility is NOT granted",
    "AX is not trusted",
)
_DRIVER_SCREEN_TEXT = (
    "Screen Recording permission not granted",
    "Screen Recording is NOT granted",
)
_DRIVER_EITHER_TEXT = (
    # "permissions_pending: macOS Accessibility or Screen Recording permission is still pending; no
    # action started, retry after the permission gate completes"
    "permissions_pending",
    # "[cua-driver] desktop tool calls remain gated; grant Accessibility and Screen Recording
    # permissions, then restart the daemon."
    "desktop tool calls remain gated",
)
# "Cua Driver daemon is not running." (its CLI), "embedded cua-driver daemon is not running" and
# "the machine-wide cua-driver daemon is not running" (Hermes).
_DRIVER_DOWN_TEXT = ("daemon is not running",)


def _has(text: str, needles: Iterable[str]) -> bool:
    return any(n in text for n in needles)


def driver_grant_message(code: str, *, both: bool = False) -> FixMessage:
    """The words for a missing CuaDriver grant. ``both``: the driver did not say which of its two.
    ``restart: "driver"``: the grant only applies to a freshly launched driver (cua-driver: "grant ...
    then restart the daemon"); the computer_use boundary drops the session's backend so the next call
    launches one, which is what keeps ``retry`` true."""
    app = host_app_name()
    helper = f"{DRIVER_NAME}, the helper {app} uses to see and use your screen,"
    if both:
        return fix_message(
            f"{helper} is still waiting for macOS permission. Turn {DRIVER_NAME} on in System Settings › "
            f"Privacy & Security › Accessibility, and in Screen & System Audio Recording, {THEN}. "
            f"This is {DRIVER_NAME}'s own switch, not {app}'s.",
            TCC_DRIVER_ACCESSIBILITY, subject=DRIVER_NAME, retry=True, restart="driver",
            also_pane="Privacy_ScreenCapture")
    if code == TCC_DRIVER_SCREEN:
        return fix_message(
            f"{helper} isn't allowed to see your screen. Turn {DRIVER_NAME} on in System Settings › Privacy & "
            f"Security › Screen & System Audio Recording, {THEN}. This is {DRIVER_NAME}'s own "
            f"switch, not {app}'s.",
            TCC_DRIVER_SCREEN, subject=DRIVER_NAME, retry=True, restart="driver")
    return fix_message(
        f"{helper} isn't allowed to control your Mac. Turn {DRIVER_NAME} on in System Settings › Privacy & "
        f"Security › Accessibility, {THEN}. This is {DRIVER_NAME}'s own switch, not {app}'s.",
        TCC_DRIVER_ACCESSIBILITY, subject=DRIVER_NAME, retry=True, restart="driver")


def driver_not_running_message(detail: Optional[str] = None, *, reset: bool = False) -> FixMessage:
    """``driver_not_running``. ``reset``: a backend with its OWN (embedded) daemon was actually dropped,
    so the next call launches a fresh helper and the identical call is expected to work (``retry``).
    Anything else (nothing was released, or the machine-wide daemon Hermes does not own) is said
    plainly, with ``retry`` false: running the call again changes nothing until the helper runs."""
    app = host_app_name()
    extra = {"detail": detail} if detail else {}
    if reset:
        return fix_message(
            f"The screen helper ({DRIVER_NAME}) had stopped. I've reset it, so it starts again on the next "
            f"try. If it keeps stopping, quitting and reopening {app} restarts it.",
            DRIVER_NOT_RUNNING, subject=DRIVER_NAME, retry=True, **extra)
    return fix_message(
        f"The screen helper ({DRIVER_NAME}) isn't running, so I can't see or use your screen right now. "
        f"Quitting and reopening {app} starts it again; {THEN}.",
        DRIVER_NOT_RUNNING, subject=DRIVER_NAME, retry=False, **extra)


def driver_fix_in(text: str = "", driver_code: Optional[str] = None) -> Optional[FixMessage]:
    """The fixable cause in a cua-driver failure (its message and/or its refusal code), else ``None``.
    macOS only: these are TCC grants and a LaunchServices-launched helper."""
    if not _is_darwin():
        return None
    text = text or ""
    code = driver_code if isinstance(driver_code, str) else ""
    if _has(text, _DRIVER_DOWN_TEXT):
        return driver_not_running_message(text.strip()[:300] or None)
    ax = code in _DRIVER_AX_CODES or _has(text, _DRIVER_AX_TEXT)
    screen = code in _DRIVER_SCREEN_CODES or _has(text, _DRIVER_SCREEN_TEXT)
    either = code in _DRIVER_EITHER_CODES or _has(text, _DRIVER_EITHER_TEXT)
    if either or (ax and screen):
        return driver_grant_message(TCC_DRIVER_ACCESSIBILITY, both=True)
    if ax:
        return driver_grant_message(TCC_DRIVER_ACCESSIBILITY)
    if screen:
        return driver_grant_message(TCC_DRIVER_SCREEN)
    return None
