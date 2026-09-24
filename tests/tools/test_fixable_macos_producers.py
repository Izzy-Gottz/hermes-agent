"""Slice 3 of docs/FIXABLE-FAILURES.md (Moe repo): more tools say WHICH fixable cause a failure is.

Each producer is driven by the real condition, or by the real words of the thing that fails:

- tcc_files: ``PermissionError(errno.EPERM, "Operation not permitted")`` from opening the protected
  folder, which is what macOS raises; and two REAL non-TCC failures that must not get the code, a
  ``chmod 000`` folder (EACCES) and a ``chflags uchg`` file (a real EPERM that is not privacy).
- tcc_automation: osascript's stderr, verbatim.
- tcc_driver_* / driver_not_running: cua-driver 0.28.2's strings, from its binary
  (``strings /Applications/CuaDriver.app/Contents/MacOS/cua-driver``).
- window_other_space: the measured Spaces payload shape from tests/computer_use/test_cua_spaces.py.

No test touches a real protected folder, a real app, or the real driver: $HOME is a temp dir.
"""

from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import fix_reasons as fr
from tools import fix_reasons_macos as frm

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="chflags/TCC semantics are macOS")

CONTRACT = {"error", "code", "owner", "pane", "subject", "retry"}

# osascript, verbatim (the -1743 line is the one in the plan; JXA prefixes "Error: Error: ").
AE_1743 = "execution error: Not authorized to send Apple events to Notes. (-1743)"
AE_1743_JXA = "execution error: Error: Error: Not authorized to send Apple events to System Events. (-1743)"
# -1744 is NOT measured: no Mac here returns it on demand. This is osascript's generic shape for an error
# with no message of its own; the producer keys on the number and takes the app from the script.
AE_1744 = "execution error: An error of type -1744 has occurred. (-1744)"

# cua-driver 0.28.2, verbatim from the binary.
DRV_PENDING = ("permissions_pending: macOS Accessibility or Screen Recording permission is still pending; "
               "no action started, retry after the permission gate completes")
DRV_AX = "Accessibility is NOT granted for this process."
DRV_SCREEN = "Screen Recording permission not granted"
DRV_GATED = ("[cua-driver] desktop tool calls remain gated; grant Accessibility and Screen Recording "
             "permissions, then restart the daemon.")
DRV_DOWN = "Cua Driver daemon is not running.\nStart it first with: cua-driver serve"


def _contract(out):
    data = json.loads(out) if isinstance(out, str) else out
    assert CONTRACT <= set(data), data
    return data


# ── tcc_files ────────────────────────────────────────────────────────────────

@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake $HOME with the protected folders, and switches that make macOS refuse exactly the way it
    does: ``deny(folder)`` makes opening the folder raise PermissionError(EPERM), ``deny_file(path)``
    makes opening that file raise it. ``probes`` records every open of a protected folder, so a test
    can prove a folder was never touched before the operation itself failed."""
    import builtins
    h = tmp_path / "home"
    for d in ("Desktop", "Documents", "Downloads", "Library/Mobile Documents/com~apple~CloudDocs"):
        (h / d).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
    real_scandir, real_open = os.scandir, builtins.open
    state = SimpleNamespace(folders=set(), files=set(), probes=[])

    def scandir(p="."):
        norm = os.path.normpath(os.fspath(p))
        if norm.startswith(str(h) + os.sep):
            state.probes.append(norm)
        if norm in state.folders:
            raise PermissionError(errno.EPERM, "Operation not permitted", os.fspath(p))
        return real_scandir(p)

    def fake_open(file, *a, **k):
        if isinstance(file, (str, os.PathLike)) and os.path.normpath(os.fspath(file)) in state.files:
            raise PermissionError(errno.EPERM, "Operation not permitted", os.fspath(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(builtins, "open", fake_open)
    _state[str(h)] = state
    return h


_state: dict = {}


@pytest.fixture
def deny(home):
    """``deny("Desktop")``: macOS refuses this process that folder."""
    return lambda rel: _state[str(home)].folders.add(os.path.normpath(str(home / rel)))


@pytest.fixture
def deny_file(home):
    """``deny_file(path)``: opening this file raises EPERM, as it does inside a refused folder."""
    return lambda p: _state[str(home)].files.add(os.path.normpath(str(p)))


@pytest.fixture
def probes(home):
    return _state[str(home)].probes


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


@pytest.fixture
def uchg():
    """REAL EPERM that is not privacy: ``chflags uchg``. Always undone, or pytest can't clean up."""
    flagged = []

    def flag(p):
        subprocess.run(["chflags", "uchg", str(p)], check=True)
        flagged.append(p)
    yield flag
    for p in reversed(flagged):
        subprocess.run(["chflags", "nouchg", str(p)], check=False)


class TestTccFiles:
    def test_read_file_in_a_refused_folder(self, home, darwin, deny, deny_file):
        f = home / "Desktop" / "notes.txt"
        f.write_text("hi\n")
        deny("Desktop")
        deny_file(f)
        from tools.file_tools import read_file_tool
        out = _contract(read_file_tool(str(f)))
        assert out["code"] == "tcc_files" and out["owner"] == "app"
        assert out["pane"] == "Privacy_FilesAndFolders" and out["subject"] == "Desktop"
        assert out["retry"] is True and out["path"] == str(f)
        assert "Memoe isn't allowed into your Desktop folder" in out["error"]
        assert out["error"].endswith("then I'll carry on.")
        assert "content" not in out

    def test_native_eperm_is_not_misreported_as_the_shell(self, home, darwin, deny_file):
        """Without the folder refusing, it is not TCC, but the read still says what happened instead of
        the shell fallback's "Terminal environment unavailable … Retry shortly"."""
        f = home / "Desktop" / "notes.txt"
        f.write_text("hi\n")
        deny_file(f)
        from tools.file_tools import read_file_tool
        out = json.loads(read_file_tool(str(f)))
        assert "code" not in out and "Operation not permitted" in out["error"], out
        assert "Terminal environment unavailable" not in out["error"]

    def test_a_failure_without_eperm_never_probes(self, home, darwin, deny, probes):
        """Desktop refused, one file on it granted, and a patch that fails for its own reason ("Could not
        find a match"): not privacy, no code, and the folder is never opened. (Built by the review:
        without the EPERM-evidence gate this probed and reported tcc_files.)"""
        f = home / "Desktop" / "granted.py"
        f.write_text("x = 1\n")
        deny("Desktop")
        from tools.file_tools import patch_tool
        out = json.loads(patch_tool("replace", str(f), "nothing like this", "y"))
        assert "Could not find" in out["error"] and "code" not in out, out
        assert probes == [], probes

    def test_a_file_granted_on_its_own_is_never_refused(self, home, darwin, deny, probes):
        """The folder refuses, the file itself opens (granted individually, com.apple.macl): the read
        works, and the folder is never probed because nothing failed."""
        f = home / "Downloads" / "granted.txt"
        f.write_text("ok\n")
        deny("Downloads")
        from tools.file_tools import read_file_tool, write_file_tool
        out = json.loads(read_file_tool(str(f)))
        assert "code" not in out and "ok" in out["content"], out
        assert probes == [], f"probed before any failure: {probes}"

    @darwin_only
    def test_write_and_patch_in_a_refused_folder(self, home, darwin, deny, uchg):
        """The write REALLY fails with EPERM (the folder is uchg, so nothing can be created in it); the
        folder refusing EPERM afterwards is what makes it privacy protection."""
        (home / "Documents" / "old.txt").write_text("a\n")
        uchg(home / "Documents")
        deny("Documents")
        from tools.file_tools import patch_tool, write_file_tool
        out = _contract(write_file_tool(str(home / "Documents" / "new.txt"), "x"))
        assert (out["code"], out["subject"]) == ("tcc_files", "Documents")
        assert not (home / "Documents" / "new.txt").exists()
        out = _contract(patch_tool("replace", str(home / "Documents" / "old.txt"), "a", "b"))
        assert out["code"] == "tcc_files"
        assert (home / "Documents" / "old.txt").read_text() == "a\n"

    def test_icloud_drive_names_full_disk_access(self, home, darwin, deny, deny_file):
        deny("Library/Mobile Documents/com~apple~CloudDocs")
        path = home / "Library/Mobile Documents/com~apple~CloudDocs/plan.txt"
        path.write_text("p\n")
        deny_file(path)
        from tools.file_tools import read_file_tool
        out = _contract(read_file_tool(str(path)))
        assert (out["subject"], out["pane"]) == ("iCloud Drive", "Privacy_AllFiles")
        assert "Full Disk Access" in out["error"]

    @darwin_only
    def test_chmod_000_folder_is_eacces_not_tcc(self, home):
        """REAL: a folder with mode 000 refuses with EACCES. That is ownership/mode, never a grant."""
        (home / "Downloads" / "a.txt").write_text("x")
        os.chmod(home / "Downloads", 0)
        try:
            with pytest.raises(PermissionError) as e:
                os.listdir(home / "Downloads")
            assert e.value.errno == errno.EACCES  # measured, not assumed
            assert frm.tcc_denied_folder(str(home / "Downloads" / "a.txt")) is None
            from tools.file_tools import read_file_tool
            out = json.loads(read_file_tool(str(home / "Downloads" / "a.txt")))
            assert "error" in out and "code" not in out, out
        finally:
            os.chmod(home / "Downloads", 0o755)

    @darwin_only
    def test_immutable_file_is_a_real_eperm_and_not_tcc(self, home, uchg):
        """REAL: ``chflags uchg`` makes writes fail with EPERM "Operation not permitted", the same errno
        TCC uses. The folder still opens, so it is not privacy protection and gets no code."""
        f = home / "Desktop" / "locked.txt"
        f.write_text("x")
        uchg(f)
        with pytest.raises(PermissionError) as e:
            open(f, "w")
        assert e.value.errno == errno.EPERM
        assert frm.files_denied_from_exc(e.value) is None
        from tools.file_tools import write_file_tool
        out = json.loads(write_file_tool(str(f), "y"))
        assert out.get("error") and out.get("code") is None, out
        assert f.read_text() == "x"

    def test_eperm_exception_is_confirmed_on_the_folder(self, home, darwin, deny):
        deny("Desktop")
        exc = PermissionError(errno.EPERM, "Operation not permitted", str(home / "Desktop" / "x"))
        fix = frm.files_denied_from_exc(exc)
        assert fr.fields_of(fix)["code"] == "tcc_files"
        # EACCES with the same path is never TCC, even when the folder is refused
        assert frm.files_denied_from_exc(PermissionError(errno.EACCES, "Permission denied",
                                                         str(home / "Desktop" / "x"))) is None
        from tools.file_tools import _file_exception_error
        assert json.loads(_file_exception_error(exc, None, "default"))["code"] == "tcc_files"

    def test_outside_protected_folders_and_off_macos(self, home, darwin, monkeypatch, tmp_path, deny):
        deny("Desktop")
        assert frm.tcc_denied_folder(str(tmp_path / "elsewhere.txt")) is None
        monkeypatch.setattr(sys, "platform", "linux")
        assert frm.tcc_denied_folder(str(home / "Desktop" / "x")) is None


# ── terminal: hint, code, and the cwd fallback ──────────────────────────────

def _finalize(command, output, rc=1, env_type="local", cwd=None):
    from tools.terminal_tool_result import finalize_foreground_result
    return json.loads(finalize_foreground_result(
        command=command, result={"output": output, "returncode": rc}, env=SimpleNamespace(cwd=cwd),
        env_type=env_type, effective_task_id="t", task_id="t", session_id="s", session_key="k",
        workdir=None, command_cwd=cwd, approval_note=None))


class TestTerminal:
    def test_eperm_hint_is_not_sudo(self, darwin):
        """Privacy protection is claimed only when the failing path IS the protected folder."""
        from tools.terminal_hints import annotate_failure
        hint = annotate_failure("ls ~/Desktop", 1, "ls: /Users/a/Desktop: Operation not permitted")
        assert "privacy protection blocked access to the Desktop folder" in hint and "sudo" in hint
        assert "escalate to sudo" not in hint  # the EACCES hint it used to fall through to
        # EACCES keeps its own hint
        assert "ownership/mode" in annotate_failure("touch /etc/x", 1, "touch: /etc/x: Permission denied")

    @pytest.mark.parametrize("command, output", [
        ("cd ~/Documents/proj && kill -9 1", "kill: 1: Operation not permitted"),       # a process, no folder
        ("rm -f ~/Desktop/locked.txt", "rm: /Users/a/Desktop/locked.txt: Operation not permitted"),  # a file
        ("cat old.log", "ls: /Users/a/Desktop: Operation not permitted"),                # only quoted
    ])
    def test_eperm_hint_is_neutral_without_evidence(self, darwin, command, output):
        from tools.terminal_hints import annotate_failure
        hint = annotate_failure(command, 1, output)
        assert hint.startswith("Operation not permitted: macOS may be protecting this") and "sudo won't help" in hint
        assert "privacy protection blocked" not in hint

    def test_no_automation_hint_from_text_alone(self, darwin):
        """The -1743 hint is gone: the result's own code carries it, on evidence, or nothing does."""
        from tools.terminal_hints import annotate_failure
        hint = annotate_failure("osascript -e 'tell application \"Notes\" to get name'", 1, "0:5: " + AE_1743)
        assert hint is None or "Automation" not in hint, hint

    def test_osascript_refusal_carries_the_code(self, darwin, monkeypatch):
        monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
        out = _contract(_finalize("osascript -e 'tell application \"Notes\" to get name of every note'",
                                  "0:34: " + AE_1743))
        assert out["exit_code"] == 1  # the command's own exit is untouched
        assert (out["code"], out["owner"], out["pane"], out["subject"]) == (
            "tcc_automation", "app", "Privacy_Automation", "Notes")
        assert out["consent"] == "denied" and out["retry"] is True
        assert "Memoe isn't allowed to control Notes" in out["error"]
        assert AE_1743 in out["output"]

    # Output that only QUOTES a refusal. Each was reproduced by the review as a false tcc_automation.
    QUOTED_REFUSALS = {
        "grep of a header, then false": (
            "grep -n errAEEventNotPermitted AE.h; false",
            "AE.h:88:  errAEEventNotPermitted = -1743, /* (-1743) */\n" + "0:34: " + AE_1743),
        "cat of an old log, then exit 1": ("cat ~/logs/notes-sync.log; exit 1", "2026-09-01 12:00 " + AE_1743),
        "failing pytest quoting the line": (
            "python -m pytest tests/test_notes.py -q",
            f"E       AssertionError: assert 'ok' == '{AE_1743}'\nFAILED tests/test_notes.py::test_x"),
        # second review, (a)-(d): each got tcc_automation / Notes before
        "(a) osascript fails for Finder (-1728), then cat prints an old Notes refusal": (
            "osascript -e 'tell application \"Finder\" to get name of window 1'; cat sync.log",
            "0:40: execution error: Finder got an error: Can't get window 1. (-1728)\n" + AE_1743),
        "(b) osascript succeeds, then an old refusal is printed and exit 1": (
            "osascript -e 'tell application \"Notes\" to get name'; echo \"" + AE_1743 + "\"; exit 1",
            "Notes\n" + AE_1743),
        "(c) grep for the word osascript, then false": (
            "grep -rn osascript ~/logs | tail; false", "~/logs/a.log:3: 0:34: " + AE_1743),
        "(d) .applescript inside a log's filename": (
            "grep -n Notes ~/logs/notes.applescript.log", "12: 0:34: " + AE_1743),
        "log show filtering on osascript": (
            "log show --last 1h --predicate 'process == \"osascript\"'", "osascript: " + AE_1743),
        "the refused app is not the one the script tells": (
            "osascript -e 'tell application \"Finder\" to get name'", "0:34: " + AE_1743),
        "two refusal lines are ambiguity, not evidence": (
            "osascript -e 'tell application \"Notes\" to get name'", "0:34: " + AE_1743 + "\n0:34: " + AE_1743),
        "the number without macOS's full line": (
            "osascript -e 'tell application \"Notes\" to get name'", "0:34: execution error: Notes got an error: (-1743)"),
    }

    @pytest.mark.parametrize("case", list(QUOTED_REFUSALS))
    def test_quoted_refusal_is_not_automation(self, darwin, case):
        command, output = self.QUOTED_REFUSALS[case]
        out = _finalize(command, output)
        assert "code" not in out and out["error"] is None, out
        assert "Automation" not in (out.get("hint") or "")

    def test_jxa_and_script_files_count(self, darwin):
        assert _finalize("osascript -l JavaScript -e 'Application(\"Notes\").notes()'",
                         "0:1: " + AE_1743).get("code") == "tcc_automation"
        # A script file counts when its text tells the refused app; one that cannot be read (or a
        # compiled .scpt) names no target, so it is no evidence of which app refused.
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "sync.applescript"), "w") as fh:
                fh.write('tell application "Notes" to get name of every note\n')
            assert _finalize("/usr/bin/osascript sync.applescript", "0:1: " + AE_1743,
                             cwd=d).get("code") == "tcc_automation"
        assert "code" not in _finalize("osascript ./missing.scpt", "0:1: " + AE_1743)

    def test_a_folder_named_only_inside_a_longer_argument(self, home, darwin, deny, probes):
        """``tail ~/Documents/build.log`` is not an operation on ~/Documents, whatever its output lists."""
        deny("Documents")
        out = _finalize("tail ~/Documents/build.log", f"ls: {home}/Documents: Operation not permitted", cwd=str(home))
        assert "code" not in out and out["error"] is None and probes == []
        assert out["hint"].startswith("Operation not permitted: macOS may be protecting this")

    def test_protected_folder_in_output(self, home, darwin, deny):
        deny("Desktop")
        out = _contract(_finalize("ls Desktop", "ls: Desktop: Operation not permitted", cwd=str(home)))
        assert (out["code"], out["subject"]) == ("tcc_files", "Desktop")

    def test_a_log_line_never_probes_a_folder(self, home, darwin, deny, probes):
        """The command never addressed ~/Desktop: its output alone must not make Hermes open it."""
        deny("Desktop")
        out = _finalize("cat old.log; exit 1", f"ls: {home}/Desktop: Operation not permitted", cwd=str(home))
        assert "code" not in out and out["error"] is None
        assert probes == [], probes

    def test_success_and_containers_get_no_code(self, home, darwin, deny):
        deny("Desktop")
        assert "code" not in _finalize("osascript x", AE_1743, rc=0)
        assert "code" not in _finalize("osascript x", AE_1743, env_type="docker")
        assert "code" not in _finalize("ls Desktop", "ls: Desktop: Operation not permitted",
                                       env_type="docker", cwd=str(home))

    @darwin_only
    def test_real_eperm_that_is_not_tcc_gets_a_neutral_hint_and_no_code(self, home, uchg):
        """REAL: rm of a uchg file on the Desktop is EPERM. It is not privacy, and nothing may say so."""
        f = home / "Desktop" / "locked.txt"
        f.write_text("x")
        uchg(f)
        proc = subprocess.run(["/bin/rm", "-f", str(f)], capture_output=True, text=True)
        assert proc.returncode != 0 and "Operation not permitted" in proc.stderr  # real
        out = _finalize(f"rm -f {f}", proc.stderr, rc=proc.returncode, cwd=str(home))
        assert "code" not in out and out["error"] is None
        assert out["hint"].startswith("Operation not permitted: macOS may be protecting this")

    def test_cwd_fallback_is_surfaced_once(self, home, darwin, monkeypatch, deny):
        """``os.getcwd()`` in a refused folder raises EPERM; the fallback used to be silent."""
        from tools import terminal_tool_config as cfg
        deny("Documents")
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setenv("PWD", str(home / "Documents" / "proj"))
        monkeypatch.setattr(cfg, "_cwd_fallback", {})
        monkeypatch.setattr(os, "getcwd", lambda: (_ for _ in ()).throw(
            PermissionError(errno.EPERM, "Operation not permitted")))
        assert cfg._safe_getcwd() == str(home)
        out = _finalize("pwd", str(home), rc=0)
        note = out["cwd_fallback"]
        assert note["from"] == str(home / "Documents" / "proj") and note["to"] == str(home)
        assert note["fix"]["code"] == "tcc_files" and note["fix"]["subject"] == "Documents"
        assert "code" not in out  # the command itself succeeded
        cfg._safe_getcwd()
        assert "cwd_fallback" not in _finalize("pwd", str(home), rc=0)  # once, not every command


# ── tcc_automation ───────────────────────────────────────────────────────────

class TestAutomation:
    @pytest.mark.parametrize("stderr, target", [(AE_1743, "Notes"), (AE_1743_JXA, "System Events")])
    def test_parses_the_target_app(self, darwin, stderr, target):
        fields = fr.fields_of(frm.automation_denied_in_text(stderr))
        assert (fields["code"], fields["subject"], fields["consent"]) == ("tcc_automation", target, "denied")

    def test_1744_is_the_same_grant_not_yet_asked(self, darwin):
        fix = frm.automation_denied_in_text(AE_1744, 'tell application "Reminders" to get name')
        fields = fr.fields_of(fix)
        assert (fields["code"], fields["subject"], fields["consent"]) == ("tcc_automation", "Reminders", "not_asked")
        assert "Try again while you're at your Mac" in fix

    def test_other_osascript_errors_are_not_automation(self, darwin):
        assert frm.automation_denied_in_text("execution error: Can't get application \"Nope\". (-1728)") is None

    def test_native_backend_raises_it(self, darwin, monkeypatch):
        from tools.computer_use import macos_native_backend as nb
        monkeypatch.setattr(nb.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="0:28: " + AE_1743_JXA))
        backend = nb.MacNativeBackend()
        with pytest.raises(fr.FixableError) as e:
            backend._applescript('tell application "System Events" to get name of every process')
        assert fr.fields_of(e.value)["subject"] == "System Events"
        with pytest.raises(fr.FixableError):
            backend._jxa("Application('System Events').processes()")


# ── computer_use: driver grants, driver down, other Space ────────────────────

class _FakeBackend:
    def __init__(self, *, raises=None, result=None):
        self.raises, self.result = raises, result

    def capture(self, **kw):
        if self.raises:
            raise self.raises
        return self.result


@pytest.fixture
def cu(monkeypatch):
    from tools.computer_use import tool as cu_tool
    released = []
    state = SimpleNamespace(tool=cu_tool, released=released, backend=None, releases=True)
    monkeypatch.setattr(cu_tool, "_get_backend", lambda session_id="": state.backend)
    monkeypatch.setattr(cu_tool, "release_computer_use_session",
                        lambda sid: released.append(sid) or state.releases)
    # a cached backend for session sid that owns its (embedded) daemon, as Moe's does
    state.cache_embedded = lambda sid: monkeypatch.setitem(
        cu_tool._backends, sid, SimpleNamespace(_embedded_daemon=object()))
    monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
    state.run = lambda args, **kw: json.loads(cu_tool.handle_computer_use(args, **kw))
    return state


class TestComputerUse:
    @pytest.mark.parametrize("words, code, pane, both", [
        (DRV_AX, "tcc_driver_accessibility", "Privacy_Accessibility", False),
        (DRV_SCREEN, "tcc_driver_screen", "Privacy_ScreenCapture", False),
        (DRV_PENDING, "tcc_driver_accessibility", "Privacy_Accessibility", True),
        (DRV_GATED, "tcc_driver_accessibility", "Privacy_Accessibility", True),
    ])
    def test_driver_grant_words_through_the_tool(self, cu, words, code, pane, both):
        # the exact wrapping capture uses: RuntimeError(f"cua-driver get_window_state failed: {msg}")
        cu.backend = _FakeBackend(raises=RuntimeError(f"cua-driver get_window_state failed: {words}"))
        out = _contract(cu.run({"action": "capture"}, session_id="s1"))
        assert (out["code"], out["owner"], out["pane"], out["subject"]) == (code, "driver", pane, "CuaDriver")
        assert out["retry"] is True and words in out["detail"] and out["restart"] == "driver"
        assert ("also_pane" in out) is both
        assert out["error"].count("then I'll carry on") == 1
        assert "CuaDriver's own switch, not Memoe's" in out["error"]
        assert cu.released == ["s1"]  # relaunched on the next call, so the new grant applies

    def test_driver_refusal_code_on_an_action_result(self, cu, monkeypatch):
        from tools.computer_use.backend import ActionResult
        from tools.computer_use import tool as cu_tool
        res = ActionResult(ok=False, action="click", message="", code="screen_recording_permission_denied")
        out = _contract(cu_tool._with_fixable_cause(cu_tool._text_response(res), "s2"))
        assert out["code"] == "tcc_driver_screen" and out["driver_code"] == "screen_recording_permission_denied"
        assert out["verdict"]["decision"] == "escalate"  # the rest of the payload is kept

    def test_success_text_is_never_read(self, cu):
        from tools.computer_use import tool as cu_tool
        ok = json.dumps({"ok": True, "action": "click", "message": DRV_AX})
        assert cu_tool._with_fixable_cause(ok, "s") == ok
        assert cu.released == []

    @pytest.mark.parametrize("embedded, releases, reset", [
        (True, True, True),     # Moe's route: its own daemon, dropped -> the next call relaunches it
        (False, True, False),   # machine-wide daemon: dropping the backend never restarts it
        (True, False, False),   # nothing was cached to drop
    ])
    def test_embedded_daemon_down(self, cu, embedded, releases, reset):
        from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon
        daemon = object.__new__(_EmbeddedCuaDaemon)
        daemon._running = False
        with pytest.raises(fr.FixableError) as e:
            daemon.proxy_invocation()
        # as it reaches the tool: wrapped by the session's setup, cause chain intact
        wrapped = RuntimeError(f"cua-driver session setup failed: {e.value}")
        wrapped.__cause__ = e.value
        cu.backend, cu.releases = _FakeBackend(raises=wrapped), releases
        if embedded:
            cu.cache_embedded("s3")
        out = _contract(cu.run({"action": "capture"}, session_id="s3"))
        assert (out["code"], out["owner"], out["pane"], out["retry"]) == ("driver_not_running", None, None, reset)
        assert ("I've reset it" in out["error"]) is reset
        if not reset:
            assert "isn't running" in out["error"] and "then ask me to try again" in out["error"]
        assert "embedded cua-driver daemon is not running" in out["detail"]
        assert cu.released == ["s3"]

    def test_daemon_down_during_an_action(self, cu):
        """cua_backend._action turns a call's exception into an ActionResult; the code must survive it."""
        from tools.computer_use.cua_backend import CuaDriverBackend
        from tools.computer_use import tool as cu_tool
        from tools.fix_reasons_macos import driver_not_running_message

        class _Dead:
            def call_tool(self, name, args):
                raise fr.as_error(driver_not_running_message("embedded cua-driver daemon is not running"))

            def supports_capability(self, *a, **k):
                return False
        backend = object.__new__(CuaDriverBackend)
        backend._session, backend._session_id, backend._snapshot_tokens = _Dead(), "hermes-x", {}
        res = backend._action("click", {"x": 1, "y": 2})
        out = _contract(cu_tool._with_fixable_cause(cu_tool._text_response(res), "s4"))
        assert out["code"] == "driver_not_running" and out["ok"] is False
        assert cu.released == ["s4"]

    def test_driver_cli_words_for_daemon_down(self, cu):
        """The machine-wide daemon's own CLI words, no embedded backend cached: said plainly."""
        cu.backend = _FakeBackend(raises=RuntimeError(DRV_DOWN))
        out = _contract(cu.run({"action": "capture"}))
        assert out["code"] == "driver_not_running" and out["retry"] is False
        assert "I've reset it" not in out["error"]

    def test_off_macos_nothing_is_read(self, cu, monkeypatch):
        from tools.computer_use import tool as cu_tool
        monkeypatch.setattr(sys, "platform", "linux")
        failed = json.dumps({"ok": False, "action": "click", "error": DRV_AX})
        assert cu_tool._with_fixable_cause(failed, "s") == failed
        assert frm.driver_fix_in(DRV_AX) is None

    def test_backend_that_cannot_start(self, cu, monkeypatch):
        from tools.computer_use import tool as cu_tool

        def boom(session_id=""):
            raise RuntimeError(f"embedded cua-driver exited during startup: {DRV_GATED}")
        monkeypatch.setattr(cu_tool, "_get_backend", boom)
        out = _contract(cu.run({"action": "capture"}))
        assert out["code"] == "tcc_driver_accessibility" and out["also_pane"] == "Privacy_ScreenCapture"


def _space_backend():
    from tools.computer_use.cua_backend import CuaDriverBackend, _ingest_windows
    raw = [{"window_id": 143441, "pid": 11, "app_name": "Terminal", "title": "", "z_index": 30,
            "is_on_screen": True, "on_current_space": True, "space_ids": [2915]},
           {"window_id": 122663, "pid": 7253, "app_name": "Calendar", "title": "", "z_index": 5,
            "is_on_screen": False, "on_current_space": False, "space_ids": [1]}]

    class _B(CuaDriverBackend):
        def __init__(self):
            self._active_pid = self._active_window_id = None
            self._last_target = None

        def _clear_active_target(self):
            self._active_pid = self._active_window_id = None

        def _load_windows_all_spaces_checked(self):
            return _ingest_windows(raw), False

        def _load_windows(self):
            return _ingest_windows([w for w in raw if w["on_current_space"] is True])

        # bring_to_front is the driver's; record it rather than raise a real window
        _session = SimpleNamespace(_has_tool=lambda name: True)
        raised: list = []

        def bring_to_front(self, *, pid, window_id=None):
            from tools.computer_use.backend import ActionResult
            self.raised.append((pid, window_id))
            return ActionResult(ok=True, action="bring_to_front", message="raised")

        def list_apps(self):
            return []
    return _B()


class TestOtherSpace:
    def test_capture_keeps_the_text_and_adds_the_code(self):
        from tools.computer_use.tool import _capture_response
        backend = _space_backend()
        cap = backend._resolve_capture_windows("som", "Calendar", None, None)
        out = _contract(_capture_response(cap))
        assert (out["code"], out["owner"], out["pane"], out["subject"]) == ("window_other_space", None, None, "Calendar")
        assert out["retry"] is True and out["window_id"] == 122663 and out["space_ids"] == [1]
        assert "IS running, on another macOS Space" in out["window_title"]  # the model's text, kept
        assert "another desktop" in out["error"]

    def test_focus_app(self):
        from tools.computer_use.tool import _text_response
        res = _space_backend().focus_app("Calendar")
        out = _contract(_text_response(res))
        assert out["ok"] is False and out["code"] == "window_other_space" and out["subject"] == "Calendar"
        assert out["message"].startswith("No on-screen window found for app 'Calendar'.")

    def test_raise_window_reaches_the_off_space_window(self):
        """What the off-Space advice says to do now works: no window_other_space loop."""
        backend = _space_backend()
        backend.raised = []
        res = backend.focus_app("Calendar", raise_window=True)
        assert res.ok is True and res.action == "focus_app" and res.fix is None
        assert backend.raised == [(7253, 122663)]
        assert (backend._active_pid, backend._active_window_id) == (7253, 122663)

    def test_not_running_is_not_other_space(self):
        res = _space_backend().focus_app("Pages")
        assert res.fix is None and res.message == "No on-screen window found for app 'Pages'."


# ── the shapes a real refusal comes in, and the one that only looks like one ───────────────

_NOTES_SCRIPT = 'tell application "Notes" to get name of every note'
_OTHER_ERROR = "0:20: execution error: Notes got an error: Can't get foo. (-1728)"


class TestAutomationShapes:
    POSITIVE = {
        "heredoc, quoted delimiter": f"osascript <<'EOF'\n{_NOTES_SCRIPT}\nEOF",
        "heredoc, bare delimiter, piped": f"osascript <<EOF 2>&1\n{_NOTES_SCRIPT}\nEOF",
        "heredoc, <<- with tabs": f"osascript <<-END\n\t{_NOTES_SCRIPT}\n\tEND",
        "bash -c": f"bash -c \"osascript -e '{_NOTES_SCRIPT}'\"",
        "zsh -lc": f"zsh -lc \"osascript -e '{_NOTES_SCRIPT}'\"",
        "sh -c with a heredoc inside": f"sh -c \"osascript <<EOF\n{_NOTES_SCRIPT}\nEOF\"",
        "sudo -u bob": f"sudo -u bob osascript -e '{_NOTES_SCRIPT}'",
        "env -u VAR": f"env -u LANG FOO=1 osascript -e '{_NOTES_SCRIPT}'",
        "here-string": f"osascript <<< '{_NOTES_SCRIPT}'",
    }
    NEGATIVE = {
        "heredoc that tells another app": "osascript <<EOF\ntell application \"Finder\" to get name\nEOF",
        "bash -c whose last command is false": f"bash -c \"osascript -e '{_NOTES_SCRIPT}'; false\"",
        "a heredoc fed to cat, not osascript": f"cat <<EOF\n{_NOTES_SCRIPT}\nEOF",
        "sudo -u bob grep osascript": "sudo -u bob grep -rn osascript /var/log",
        "bash -c without osascript": "bash -c 'grep -n osascript ~/logs/a.log'",
        "a newline, then false": f"osascript -e '{_NOTES_SCRIPT}'\nfalse",
        "bash with a script whose own argument is -c": f"bash run.sh -c \"osascript -e '{_NOTES_SCRIPT}'\"",
    }

    @pytest.mark.parametrize("case", list(POSITIVE))
    def test_real_refusal_shapes(self, darwin, case):
        out = _contract(_finalize(self.POSITIVE[case], "0:34: " + AE_1743))
        assert (out["code"], out["subject"]) == ("tcc_automation", "Notes"), case

    @pytest.mark.parametrize("case", list(NEGATIVE))
    def test_look_alike_shapes(self, darwin, case):
        out = _finalize(self.NEGATIVE[case], "0:34: " + AE_1743)
        assert "code" not in out and out["error"] is None, (case, out)

    def test_an_old_refusal_beside_this_runs_real_error(self, darwin):
        """The review's last false positive: the log holds an old Notes -1743, this run's error is -1728."""
        command = "cat old.log; osascript -e 'tell application \"Notes\" to get foo'"
        out = _finalize(command, "2026-09-01 " + AE_1743 + "\n" + _OTHER_ERROR)
        assert "code" not in out and out["error"] is None, out
        # a compile failure this run is an error too
        syntax = "0:12: syntax error: Expected end of line but found identifier. (-2741)"
        bad = "cat old.log; osascript -e 'tell application \"Notes\" to get bad syntax here'"
        out = _finalize(bad, "2026-09-01 " + AE_1743 + "\n" + syntax)
        assert "code" not in out and out["error"] is None, out
        # and the same command with only the refusal still counts
        assert _finalize(command, "0:34: " + AE_1743).get("code") == "tcc_automation"

    @darwin_only
    def test_a_real_heredoc_through_the_terminal(self, monkeypatch, tmp_path):
        """A REAL command in the local shell: ``osascript`` is a function that reads its script from the
        heredoc on stdin and prints macOS's refusal, so no Apple event is sent."""
        from tools.terminal_tool import terminal_tool
        monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        command = ("osascript() { cat >/dev/null; echo \"0:34: " + AE_1743 + "\" >&2; return 1; }\n"
                   f"osascript <<'EOF'\n{_NOTES_SCRIPT}\nEOF")
        out = _contract(json.loads(terminal_tool(command)))
        assert (out["code"], out["subject"], out["exit_code"]) == ("tcc_automation", "Notes", 1)


# ── every route to the model keeps the contract (as slice 1's 12-route test) ─────────────────
# One real failure per producer, sent through each way a terminal or computer_use result reaches a
# model: the tool function, the registry dispatcher, and the claude-code bridge Moe uses (the
# hermes-tools MCP server's per-tool handler: handle_function_call then to_mcp_content).

# A REAL command, run by the real terminal tool in the local environment: ``osascript`` is a shell
# function here printing macOS's exact refusal, so no Apple event is ever sent to a real app.
_OSASCRIPT_CMD = ("osascript() { echo \"0:34: " + AE_1743 + "\" >&2; return 1; }; "
                  "osascript -e 'tell application \"Notes\" to get name of every note'")


def _via_registry(name, args):
    import tools.terminal_tool  # noqa: F401 — registers terminal
    import tools.computer_use.tool  # noqa: F401 — registers computer_use
    from tools.registry import registry
    return registry.dispatch(name, args)


def _via_claude_code_bridge(name, args, monkeypatch):
    import mcp.server as mcp_server
    import model_tools
    from agent.transports import hermes_tools_mcp_server as m

    class _RecordingServer:
        def __init__(self, name, instructions=None):
            self.tools = {}

        def add_tool(self, fn, name=None, description=None, meta=None, structured_output=None):
            self.tools[name or fn.__name__] = fn

    monkeypatch.setattr(mcp_server, "MCPServer", _RecordingServer)
    monkeypatch.setattr(m, "discover_external_mcp_servers", lambda: [])
    monkeypatch.setattr(m, "tools_to_offer", lambda *a, **k: [name])
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda *a, **k: [
        {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {
            k: {"type": "string"} for k in args}}}}])
    server = m._build_server("claude-code")
    out = server.tools[name](**args)
    assert isinstance(out, str), f"the bridge must hand the child text, got {type(out)}"
    return out


def _terminal_direct(mp):
    from tools.terminal_tool import terminal_tool
    return terminal_tool(_OSASCRIPT_CMD)


TERMINAL_ROUTES = {
    "terminal_tool": _terminal_direct,
    "registry/terminal": lambda mp: _via_registry("terminal", {"command": _OSASCRIPT_CMD}),
    "claude-code-bridge/terminal": lambda mp: _via_claude_code_bridge("terminal", {"command": _OSASCRIPT_CMD}, mp),
}

CU_ROUTES = {
    "handle_computer_use": lambda mp: __import__("tools.computer_use.tool", fromlist=["x"]).handle_computer_use(
        {"action": "capture"}),
    "registry/computer_use": lambda mp: _via_registry("computer_use", {"action": "capture"}),
    "claude-code-bridge/computer_use": lambda mp: _via_claude_code_bridge("computer_use", {"action": "capture"}, mp),
}


@darwin_only
@pytest.mark.parametrize("route", list(TERMINAL_ROUTES))
def test_every_terminal_route_delivers_the_contract(route, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    out = json.loads(TERMINAL_ROUTES[route](monkeypatch))
    assert {k: out.get(k) for k in ("code", "owner", "pane", "subject", "retry", "consent")} == {
        "code": "tcc_automation", "owner": "app", "pane": "Privacy_Automation", "subject": "Notes",
        "retry": True, "consent": "denied"}, f"{route} dropped the contract: {out}"
    assert out["exit_code"] == 1 and out["error"].startswith("Memoe isn't allowed to control Notes")


@pytest.mark.parametrize("route", list(CU_ROUTES))
def test_every_computer_use_route_delivers_the_contract(route, cu, monkeypatch):
    cu.backend = _FakeBackend(raises=RuntimeError(f"cua-driver get_window_state failed: {DRV_SCREEN}"))
    out = json.loads(CU_ROUTES[route](monkeypatch))
    assert {k: out.get(k) for k in ("code", "owner", "pane", "subject", "retry", "restart")} == {
        "code": "tcc_driver_screen", "owner": "driver", "pane": "Privacy_ScreenCapture", "subject": "CuaDriver",
        "retry": True, "restart": "driver"}, f"{route} dropped the contract: {out}"


def test_a_script_that_names_no_app_cannot_name_one_on_a_card(monkeypatch):
    """``osascript -e 'error "Not authorized to send Apple events to X." number -1743'`` prints a real
    refusal line for any X it chooses; with no ``tell application`` in the script there is no
    evidence it was X that refused, so no tcc_automation — and one that tells a different app is
    refused too. A script that really tells the refused app still gets its fix."""
    import tools.fix_reasons_macos as frm
    monkeypatch.setattr(frm, "_is_darwin", lambda: True)
    line = "0:67: execution error: Not authorized to send Apple events to Terminal. (-1743)"
    forged = "osascript -e 'error \"Not authorized to send Apple events to Terminal.\" number -1743'"
    assert frm.automation_denied_in_command(forged, line) is None
    other = "osascript -e 'tell application \"Notes\" to error \"Not authorized to send Apple events to Terminal.\" number -1743'"
    assert frm.automation_denied_in_command(other, line) is None
    real = "osascript -e 'tell application \"Terminal\" to get name of front window'"
    assert frm.automation_denied_in_command(real, line) is not None
    # The native backend's own script: the same rule.
    assert frm.automation_denied_in_text(line, 'error "x" number -1743') is None
    assert frm.automation_denied_in_text(line, 'tell application "Terminal" to activate') is not None


def test_a_1744_from_the_models_own_script_makes_no_card(monkeypatch):
    """-1744 is "not yet asked", which is true of any app never automated: from a terminal
    osascript (the model's script) it makes no card; Hermes's own script still reports it."""
    import tools.fix_reasons_macos as frm
    monkeypatch.setattr(frm, "_is_darwin", lambda: True)
    line = "0:40: execution error: Not authorized to send Apple events to Notes. (-1744)"
    script = "osascript -e 'tell application \"Notes\" to get name of every note'"
    assert frm.automation_denied_in_command(script, line) is None
    assert frm.automation_denied_in_text(line, 'tell application "Notes" to activate') is not None
