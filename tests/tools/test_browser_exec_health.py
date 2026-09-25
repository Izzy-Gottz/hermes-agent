"""Ticket #16: browser_exec against a page that answers slowly (tools/browser_exec_health.py).

The harness tests run the REAL browser_harness (the uv tool Hermes installs) against a fake
daemon on a temp ``BH_RUNTIME_DIR`` socket, so no Chrome is needed: the fake answers every
request at once except page-scoped ``Runtime.evaluate`` calls marked SLOW, which it answers late,
exactly as the ticket's daemon log shows (``Received duplicate response``). Skipped when the
harness is not installed.
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest

from tools import browser_exec_health as health

_HARNESS_BIN = Path.home() / ".local/share/uv/tools/browser-use/bin/browser-harness"
needs_harness = pytest.mark.skipif(not _HARNESS_BIN.exists(), reason="browser_harness (uv tool browser-use) not installed")


class FakeDaemon:
    """One-request-per-connection JSON-line server speaking the harness daemon's IPC protocol."""

    def __init__(self, slow_s=7.0, dialog=None, url="https://slow.example/checkout"):
        # AF_UNIX paths are capped at 104 bytes on macOS: pytest's tmp_path is too long.
        self.dir = tempfile.mkdtemp(prefix="bh", dir="/tmp")
        self.sock_path = os.path.join(self.dir, "bu.sock")
        self.slow_s, self.dialog, self.url = slow_s, dialog, url
        self.requests, self.closed = [], []
        self.pages = ["T-person"]          # a tab Hermes never opened
        self.attached = "T-person"
        self._seq = 0
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.sock_path)
        self._srv.listen(64)
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    return
                data += chunk
            req = json.loads(data)
            with self._lock:
                self.requests.append(req)
            resp = self._answer(req)
            try:
                conn.sendall((json.dumps(resp) + "\n").encode())
            except OSError:
                pass  # the client gave up: the late reply the real daemon logs as a duplicate
        finally:
            conn.close()

    def _answer(self, req):
        meta, method, params = req.get("meta"), req.get("method"), req.get("params") or {}
        if meta == "ping":
            return {"pong": True, "pid": os.getpid(), "browser_kind": "cdp"}
        if meta == "current_tab":
            return {"targetId": self.attached, "url": self.url, "title": "Checkout"}
        if meta == "pending_dialog":
            return {"dialog": self.dialog}
        if meta == "set_session":
            self.attached = req.get("target_id") or self.attached
            return {"session_id": req.get("session_id")}
        if meta:
            return {}
        if method == "Target.getTargets":
            return {"result": {"targetInfos": [{"targetId": t, "type": "page", "url": "about:blank", "title": ""}
                                               for t in self.pages]}}
        if method == "Target.createTarget":
            with self._lock:
                self._seq += 1
                tid = f"T-{self._seq}"
                self.pages.append(tid)
            return {"result": {"targetId": tid}}
        if method == "Target.closeTarget":
            with self._lock:
                self.closed.append(params["targetId"])
                self.pages.remove(params["targetId"])
            return {"result": {"success": True}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "S-" + params["targetId"]}}
        if method == "Runtime.evaluate" and "SLOW" in str(params.get("expression")):
            time.sleep(self.slow_s)
            return {"result": {"result": {"type": "string", "value": "late but fine"}}}
        return {"result": {"result": {"type": "undefined"}}}

    def close(self):
        self._stop = True
        self._srv.close()
        shutil.rmtree(self.dir, ignore_errors=True)


@pytest.fixture
def daemon():
    made = []

    def make(**kw):
        d = FakeDaemon(**kw)
        made.append(d)
        return d
    yield make
    for d in made:
        d.close()


def _run_harness(d, code, *, patched=True, deadline_in=None, tab_cap=None, ledger=None, timeout=120):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BU_", "BH_", "HERMES_BU_", "PYTHON"))}
    env.update({
        "BH_RUNTIME_DIR": d.dir, "BH_TMP_DIR": d.dir, "BH_AGENT_WORKSPACE": os.path.join(d.dir, "ws"),
        "BU_NAME": "default", "BU_CDP_WS": "ws://127.0.0.1:9/devtools/browser/fake",
        "BH_UPDATE_CHECK": "0", "BH_TELEMETRY": "0", "ANONYMIZED_TELEMETRY": "false", "BH_TAB_MARKER": "1",
    })
    if deadline_in is not None:
        env[health.ENV_DEADLINE] = f"{time.time() + deadline_in:.3f}"
    if tab_cap is not None:
        env[health.ENV_TAB_CAP] = str(tab_cap)
    if ledger is not None:
        env[health.ENV_TAB_LEDGER] = str(ledger)
    script = (health.harness_patch_preamble() if patched else "") + code
    started = time.monotonic()
    proc = subprocess.run([str(_HARNESS_BIN)], input=script, capture_output=True, text=True, env=env, timeout=timeout)
    return proc, time.monotonic() - started


@needs_harness
class TestSlowPage:
    def test_unpatched_harness_gives_up_after_5s_blaming_the_daemon(self, daemon):
        """The ticket, reproduced: the page answers in 7 s, the harness has already given up."""
        d = daemon(slow_s=7.0)
        proc, _ = _run_harness(d, 'print(js("\'SLOW\'"))', patched=False)
        assert proc.returncode != 0
        assert "timed out" in proc.stderr and "late but fine" not in proc.stdout

    def test_patched_harness_waits_for_a_page_that_answers_in_7s(self, daemon):
        d = daemon(slow_s=7.0)
        proc, _ = _run_harness(d, 'print(js("\'SLOW\'"))', deadline_in=110)
        assert proc.returncode == 0, proc.stderr
        assert "late but fine" in proc.stdout

    def test_the_default_is_patched_in_the_function_not_just_the_constant(self, daemon):
        """cdp()'s _response_timeout default is bound at def time: raw cdp() must get the budget too."""
        d = daemon(slow_s=7.0)
        proc, _ = _run_harness(d, 'print(cdp("Runtime.evaluate", expression="\'SLOW\'")["result"]["value"])',
                               deadline_in=110)
        assert proc.returncode == 0, proc.stderr
        assert "late but fine" in proc.stdout

    def test_budget_is_capped_by_the_calls_deadline_and_the_error_is_honest(self, daemon):
        d = daemon(slow_s=40.0)
        proc, took = _run_harness(d, 'print(js("\'SLOW\'"))', deadline_in=9)  # budget = 9 - 2 = ~7 s
        assert proc.returncode != 0
        assert took < 30, "the deadline must cap the 30 s budget"
        err = proc.stderr
        assert "The page https://slow.example/checkout did not answer Runtime.evaluate in" in err
        assert "The browser itself is fine" in err
        assert "waiting for the daemon" not in err, "the old text blamed the daemon, which was fine"
        assert "PageDidNotAnswer" in err
        assert "new_tab(url)" in err

    def test_error_names_an_open_dialog(self, daemon):
        d = daemon(slow_s=40.0, dialog={"type": "confirm", "message": "Leave this page?"})
        proc, _ = _run_harness(d, 'print(js("\'SLOW\'"))', deadline_in=8)
        assert proc.returncode != 0
        assert "A confirm dialog is open on it ('Leave this page?')" in proc.stderr
        assert "Page.handleJavaScriptDialog" in proc.stderr

    def test_timeout_is_still_catchable_as_the_harness_raised_it(self, daemon):
        """Model code written against the harness (except TimeoutError / RuntimeError) keeps working."""
        d = daemon(slow_s=40.0)
        code = ("try:\n    js(\"'SLOW'\")\nexcept TimeoutError as e:\n    print('TIMEOUT', isinstance(e, RuntimeError))\n")
        proc, _ = _run_harness(d, code, deadline_in=7)
        assert proc.returncode == 0, proc.stderr
        assert "TIMEOUT True" in proc.stdout


@needs_harness
class TestTabCap:
    def test_oldest_hermes_tabs_closed_never_the_attached_or_the_persons(self, daemon, tmp_path):
        d = daemon()
        ledger = tmp_path / "tabs.json"
        # nine new_tab()s (the "fresh tab to reset" habit), then go back to the first one
        code = "ids = [new_tab() for _ in range(9)]\nswitch_tab(ids[0])\nprint(ids)\n"
        proc, _ = _run_harness(d, code, deadline_in=110, tab_cap=6, ledger=ledger)
        assert proc.returncode == 0, proc.stderr
        # 9 opened, cap 6: three go, oldest first, skipping T-1 because it is attached
        assert d.closed == ["T-2", "T-3", "T-4"]
        assert "T-person" in d.pages and "T-1" in d.pages
        assert [t["id"] for t in json.loads(ledger.read_text())] == ["T-1", "T-5", "T-6", "T-7", "T-8", "T-9"]
        assert "closed 3 old tab(s)" in proc.stderr

    def test_cap_holds_across_calls(self, daemon, tmp_path):
        d = daemon()
        ledger = tmp_path / "tabs.json"
        for _ in range(3):
            proc, _ = _run_harness(d, "new_tab()\nnew_tab()\nnew_tab()\n", deadline_in=110, tab_cap=6, ledger=ledger)
            assert proc.returncode == 0, proc.stderr
        assert d.closed == ["T-1", "T-2", "T-3"]

    def test_no_cap_without_the_env(self, daemon, tmp_path):
        """The Chrome lane (the person's own browser) gets no cap and no ledger."""
        d = daemon()
        proc, _ = _run_harness(d, "for _ in range(9): new_tab()\n", deadline_in=110)
        assert proc.returncode == 0, proc.stderr
        assert d.closed == []


class TestExecEnv:
    def test_own_lane_gets_deadline_cap_and_ledger(self):
        env = {}
        before = time.time()
        health.exec_env(env, timeout_s=300, session="r7k2", own_lane=True)
        assert before + 299 < float(env[health.ENV_DEADLINE]) < time.time() + 301
        assert env[health.ENV_TAB_CAP] == str(health.TAB_CAP)
        assert "tabs-r7k2-" in env[health.ENV_TAB_LEDGER]
        other = {"BU_CDP_URL": "http://127.0.0.1:9333"}
        health.exec_env(other, timeout_s=300, session="r7k2", own_lane=True)
        assert other[health.ENV_TAB_LEDGER] != env[health.ENV_TAB_LEDGER], "one ledger per browser"

    def test_chrome_lane_gets_only_the_deadline(self):
        env = {}
        health.exec_env(env, timeout_s=60, session="", own_lane=False)
        assert set(env) == {health.ENV_DEADLINE}

    def test_preamble_is_one_line_and_compiles(self):
        pre = health.harness_patch_preamble()
        assert pre.count("\n") == 1
        compile(pre + "print(1)\n", "<t>", "exec")

    def test_browser_exec_prepends_the_patch_and_sets_the_env(self, tmp_path, monkeypatch):
        import tools.browser_use_cli as bu_cli
        from tools import browser_tool_session as bt_session, browser_supervisor
        monkeypatch.setattr(bt_session, "_run_browser_command",
                            lambda *a, **k: {"success": True, "data": {"cdpUrl": "ws://127.0.0.1:47000/x"}})
        monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY",
                            type("R", (), {"get_or_start": lambda self, **k: None, "get": lambda self, t: None})())
        cli = tmp_path / "browser-use"
        cli.write_text('#!/bin/sh\ncode=$(cat)\necho "${code%%(*}"\necho "cap:$HERMES_BU_TAB_CAP deadline:$HERMES_BU_DEADLINE"\n')
        cli.chmod(0o755)
        monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])
        out = json.loads(bu_cli.browser_exec("print(1)"))["output"]
        assert out.startswith("exec\n")
        assert "cap:6 deadline:" in out


class TestDialogs:
    def test_must_respond_becomes_auto_dismiss_without_browser_dialog(self, monkeypatch):
        monkeypatch.setattr(health, "_dialog_tool_available", lambda: False)
        assert health.exec_dialog_policy("must_respond") == "auto_dismiss"
        assert health.exec_dialog_policy("auto_accept") == "auto_accept"

    def test_must_respond_kept_when_the_model_can_answer(self, monkeypatch):
        monkeypatch.setattr(health, "_dialog_tool_available", lambda: True)
        assert health.exec_dialog_policy("must_respond") == "must_respond"

    def test_browser_use_mode_really_has_no_dialog_tool(self, monkeypatch):
        """The premise: in Browser Use mode browser_dialog's check_fn is False."""
        # a static CDP URL is what normally offers browser_dialog; Browser Use mode takes it away
        monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override_raw", lambda: "http://127.0.0.1:9222")
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", lambda: False)
        monkeypatch.setattr("tools.browser_tool._is_browser_use_cli_mode", lambda: False)
        assert health._dialog_tool_available() is True
        monkeypatch.setattr("tools.browser_tool._is_browser_use_cli_mode", lambda: True)
        assert health._dialog_tool_available() is False

    def _supervisor(self, policy):
        from tools.browser_supervisor import CDPSupervisor
        return CDPSupervisor(task_id="t16", cdp_url="ws://127.0.0.1:9/x", dialog_policy=policy, dialog_timeout_s=300)

    def test_attach_passes_the_exec_policy_and_corrects_a_running_supervisor(self, monkeypatch):
        import tools.browser_use_cli as bu_cli
        from tools import browser_supervisor, browser_tool_cdp
        running = self._supervisor("must_respond")
        seen = {}

        class _Reg:
            def get_or_start(self, task_id, cdp_url, **kw):
                seen.update(kw)
                return running
        monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Reg())
        monkeypatch.setattr(browser_tool_cdp, "_get_dialog_policy_config", lambda: ("must_respond", 300.0))
        monkeypatch.setattr(browser_tool_cdp, "_resolve_cdp_override", lambda u: u)
        monkeypatch.setattr(health, "_dialog_tool_available", lambda: False)
        bu_cli._attach_vault_supervisor({"BU_CDP_URL": "http://127.0.0.1:9222"}, "t16")
        assert seen["dialog_policy"] == "auto_dismiss"
        assert running.dialog_policy == "auto_dismiss"

    def test_a_bridge_dialog_is_released_at_once_and_reported(self, monkeypatch):
        """End to end on the supervisor: the page's confirm() is answered now, not in 300 s, and the
        browser_exec result says so."""
        from tools import browser_supervisor
        monkeypatch.setattr(health, "_dialog_tool_available", lambda: False)
        sup = self._supervisor(health.exec_dialog_policy("must_respond"))
        responded = []

        async def respond(dialog, *, accept, prompt_text):
            responded.append((dialog.type, accept))
        monkeypatch.setattr(sup, "_respond_quiet", respond)
        started = time.time()

        async def page_calls_confirm():
            await sup._on_fetch_paused({"requestId": "r1", "request": {
                "url": "http://hermes-dialog-bridge.invalid/?kind=confirm&message=Delete%3F&default_prompt="}}, "S1")
            await asyncio.sleep(0)
        asyncio.run(page_calls_confirm())
        assert responded == [("confirm", False)]
        assert sup.snapshot().pending_dialogs == ()

        monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", type("R", (), {"get": lambda self, t: sup})())
        result = {"success": True}
        health.annotate(result, task_id="t16", started=started, unresumed_before=None)
        assert result["dialogs"]["closed"][0]["type"] == "confirm"
        assert result["dialogs"]["closed"][0]["closed_by"] == "auto_policy"
        assert "dismissed automatically" in result["dialog_note"]

    def test_an_open_dialog_is_reported_too(self, monkeypatch):
        from tools import browser_supervisor
        sup = self._supervisor("must_respond")

        async def page_calls_alert():
            await sup._on_fetch_paused({"requestId": "r1", "request": {
                "url": "http://hermes-dialog-bridge.invalid/?kind=alert&message=Hi"}}, "S1")
            for h in sup._dialog_watchdogs.values():
                h.cancel()
        asyncio.run(page_calls_alert())
        monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", type("R", (), {"get": lambda self, t: sup})())
        result = {"success": False}
        health.annotate(result, task_id="t16", started=time.time(), unresumed_before=None)
        assert result["dialogs"]["open"][0]["type"] == "alert"


class TestFidelity:
    def test_a_jump_in_unresumed_is_named_in_a_failed_result(self, monkeypatch):
        from tools import browser_tool_fidelity as fid
        counts = iter([15, 18])
        monkeypatch.setattr(fid, "read_statuses", lambda: [{"keepers": [{"unresumed": next(counts)}]}])
        before = health.fidelity_unresumed()
        result = {"success": False}
        health.annotate(result, task_id=None, started=time.time(), unresumed_before=before)
        assert result["paused_tabs"] == 3
        assert "browser-fidelity-keeper.log" in result["paused_tabs_note"]

    def test_no_note_when_nothing_changed_or_on_success(self, monkeypatch):
        from tools import browser_tool_fidelity as fid
        monkeypatch.setattr(fid, "read_statuses", lambda: [{"keepers": [{"unresumed": 15}]}])
        for success in (False, True):
            result = {"success": success}
            health.annotate(result, task_id=None, started=time.time(), unresumed_before=15 if not success else 1)
            assert "paused_tabs" not in result

    def test_keeper_stderr_reaches_a_log_file(self, tmp_path, monkeypatch):
        """A keeper whose browser is not there logs why and exits; that line used to go to /dev/null."""
        from tools import browser_tool_fidelity as fid
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with socket.socket() as s:  # a port nothing listens on
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        kp = fid.KeeperProcess(port, {"brand": "Google Chrome", "ua_suffix": None,
                                      "full_version": "153.0.8010.53", "major": 153})
        try:
            kp.proc.wait(120)
        finally:
            kp.kill()
        log = (tmp_path / "logs" / fid.KEEPER_LOG_NAME).read_text()
        assert f"fidelity keeper for port {port} ended" in log
        assert "keeper[" in log
