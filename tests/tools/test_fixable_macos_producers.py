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
    """A fake $HOME with the protected folders, and a switch that makes macOS refuse one of them
    exactly the way it does: opening the folder raises PermissionError(EPERM)."""
    h = tmp_path / "home"
    for d in ("Desktop", "Documents", "Downloads", "Library/Mobile Documents/com~apple~CloudDocs"):
        (h / d).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
    real_scandir, denied = os.scandir, set()

    def scandir(p="."):
        if os.path.normpath(os.fspath(p)) in denied:
            raise PermissionError(errno.EPERM, "Operation not permitted", os.fspath(p))
        return real_scandir(p)

    monkeypatch.setattr(os, "scandir", scandir)
    _denied[str(h)] = denied
    return h


_denied: dict = {}


@pytest.fixture
def deny(home):
    """``deny("Desktop")``: macOS refuses this process that folder."""
    return lambda rel: _denied[str(home)].add(os.path.normpath(str(home / rel)))


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


class TestTccFiles:
    def test_read_file_in_a_refused_folder(self, home, darwin, deny):
        (home / "Desktop" / "notes.txt").write_text("hi\n")
        deny("Desktop")
        from tools.file_tools import read_file_tool
        out = _contract(read_file_tool(str(home / "Desktop" / "notes.txt")))
        assert out["code"] == "tcc_files" and out["owner"] == "app"
        assert out["pane"] == "Privacy_FilesAndFolders" and out["subject"] == "Desktop"
        assert out["retry"] is True and out["path"] == str(home / "Desktop" / "notes.txt")
        assert "Memoe isn't allowed into your Desktop folder" in out["error"]
        assert "sudo" in out["error"]  # says sudo will NOT help, which the old hint told it to try
        assert "content" not in out

    def test_write_file_and_patch_in_a_refused_folder(self, home, darwin, deny):
        deny("Documents")
        from tools.file_tools import patch_tool, write_file_tool
        out = _contract(write_file_tool(str(home / "Documents" / "new.txt"), "x"))
        assert (out["code"], out["subject"]) == ("tcc_files", "Documents")
        assert not (home / "Documents" / "new.txt").exists()
        out = _contract(patch_tool("replace", str(home / "Documents" / "new.txt"), "a", "b"))
        assert out["code"] == "tcc_files"

    def test_icloud_drive_names_full_disk_access(self, home, darwin, deny):
        deny("Library/Mobile Documents/com~apple~CloudDocs")
        path = home / "Library/Mobile Documents/com~apple~CloudDocs/plan.txt"
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
            assert "code" not in out, out
        finally:
            os.chmod(home / "Downloads", 0o755)

    @darwin_only
    def test_immutable_file_is_a_real_eperm_and_not_tcc(self, home):
        """REAL: ``chflags uchg`` makes writes fail with EPERM "Operation not permitted", the same errno
        TCC uses. The folder still opens, so it is not privacy protection and gets no code."""
        f = home / "Desktop" / "locked.txt"
        f.write_text("x")
        subprocess.run(["chflags", "uchg", str(f)], check=True)
        try:
            with pytest.raises(PermissionError) as e:
                open(f, "w")
            assert e.value.errno == errno.EPERM
            assert frm.files_denied_from_exc(e.value) is None
            from tools.file_tools import write_file_tool
            out = json.loads(write_file_tool(str(f), "y"))
            assert out.get("code") is None, out
            assert f.read_text() == "x"
        finally:
            subprocess.run(["chflags", "nouchg", str(f)], check=True)

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
        from tools.terminal_hints import annotate_failure
        hint = annotate_failure("ls ~/Desktop", 1, "ls: /Users/a/Desktop: Operation not permitted")
        assert "privacy protection" in hint and "Desktop" in hint and "sudo" in hint
        assert "escalate to sudo" not in hint  # the EACCES hint it used to fall through to
        # EACCES keeps its own hint
        assert "ownership/mode" in annotate_failure("touch /etc/x", 1, "touch: /etc/x: Permission denied")

    def test_automation_hint(self, darwin):
        from tools.terminal_hints import annotate_failure
        hint = annotate_failure("osascript -e 'tell application \"Notes\" to get name'", 1, "0:5: " + AE_1743)
        assert "Automation" in hint and "Notes" in hint

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

    def test_protected_folder_in_output(self, home, darwin, deny):
        deny("Desktop")
        out = _contract(_finalize("ls Desktop", "ls: Desktop: Operation not permitted", cwd=str(home)))
        assert (out["code"], out["subject"]) == ("tcc_files", "Desktop")

    def test_success_and_containers_get_no_code(self, home, darwin, deny):
        deny("Desktop")
        assert "code" not in _finalize("osascript x", AE_1743, rc=0)
        assert "code" not in _finalize("osascript x", AE_1743, env_type="docker")
        assert "code" not in _finalize("ls Desktop", "ls: Desktop: Operation not permitted",
                                       env_type="docker", cwd=str(home))

    @darwin_only
    def test_real_eperm_that_is_not_tcc_gets_the_hint_but_no_code(self, home):
        f = home / "Desktop" / "locked.txt"
        f.write_text("x")
        subprocess.run(["chflags", "uchg", str(f)], check=True)
        try:
            proc = subprocess.run(["/bin/rm", "-f", str(f)], capture_output=True, text=True)
            assert proc.returncode != 0 and "Operation not permitted" in proc.stderr  # real
            out = _finalize(f"rm -f {f}", proc.stderr, rc=proc.returncode, cwd=str(home))
            assert "code" not in out and "privacy protection" in out["hint"]
        finally:
            subprocess.run(["chflags", "nouchg", str(f)], check=True)

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
    state = SimpleNamespace(tool=cu_tool, released=released, backend=None)
    monkeypatch.setattr(cu_tool, "_get_backend", lambda session_id="": state.backend)
    monkeypatch.setattr(cu_tool, "release_computer_use_session", lambda sid: released.append(sid) or True)
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
        assert out["retry"] is True and words in out["detail"]
        assert ("also_pane" in out) is both
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

    def test_embedded_daemon_down(self, cu):
        from tools.computer_use.cua_backend_daemon import _EmbeddedCuaDaemon
        daemon = object.__new__(_EmbeddedCuaDaemon)
        daemon._running = False
        with pytest.raises(fr.FixableError) as e:
            daemon.proxy_invocation()
        # as it reaches the tool: wrapped by the session's setup, cause chain intact
        wrapped = RuntimeError(f"cua-driver session setup failed: {e.value}")
        wrapped.__cause__ = e.value
        cu.backend = _FakeBackend(raises=wrapped)
        out = _contract(cu.run({"action": "capture"}, session_id="s3"))
        assert (out["code"], out["owner"], out["pane"], out["retry"]) == ("driver_not_running", None, None, True)
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
        cu.backend = _FakeBackend(raises=RuntimeError(DRV_DOWN))
        assert _contract(cu.run({"action": "capture"}))["code"] == "driver_not_running"

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

    def test_not_running_is_not_other_space(self):
        res = _space_backend().focus_app("Pages")
        assert res.fix is None and res.message == "No on-screen window found for app 'Pages'."
