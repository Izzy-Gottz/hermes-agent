"""Handing the page to the person: detection, browser_handoff's routes, and grounded words (slice D).

Measured 2026-09-24/25 (Moe transcript 42bec772, read-only): a Google sign-in in Moe's headless
browser printed "Use your passkey to confirm it’s really you" at
accounts.google.com/v3/signin/challenge/pk/presend, and Moe told the owner "Google's asking you to
approve this sign-in on your phone" -- nothing had been sent to a phone. The words came from the
vault tool's no_code_field hint ("tell the user to complete it on their device") and a skill.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tools import browser_chrome_extension as lane
from tools import browser_person_step as ps
from tools import browser_use_cli as bu

#: browser_exec stdout from the transcript, verbatim (trimmed of the account list).
GOOGLE_PASSKEY_STDOUT = ("clicked\nLoading\nSign in with Google\nChoose an account\nto continue to TinyLaunch\n"
                         "Use your passkey to confirm it’s really you\n"
                         "https://accounts.google.com/v3/signin/challenge/pk/presend?TL=ADG-GRTx&app_domain=x")
#: What Moe told the owner, verbatim: the sentence no tool result grounded.
INVENTED = "Passkey confirmation needed again — Google's asking you to approve this sign-in on your phone, same as yesterday."


def _bind(platform="api_server", origin="person", cron=""):
    from gateway.session_context import set_session_vars, set_turn_origin
    return set_session_vars(platform=platform, cron_session=cron), set_turn_origin(origin)


def _unbind(bound):
    from gateway.session_context import clear_session_vars, reset_turn_origin
    tokens, origin_token = bound
    reset_turn_origin(origin_token)
    clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def _live_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    lane.reset_sticky_lanes()
    bound = _bind()
    yield
    _unbind(bound)
    lane.reset_sticky_lanes()


# ---- what the page is ------------------------------------------------------------------------

class TestDetection:
    def test_the_measured_google_passkey_page_is_a_passkey_step(self):
        step = ps.classify_text(GOOGLE_PASSKEY_STDOUT)
        assert step == {"kind": "passkey", "page_says": "Use your passkey to confirm it’s really you"}

    @pytest.mark.parametrize("text,kind", [
        ("Check your phone\nGoogle sent a notification to your Pixel", "device_prompt"),
        ("Tap Yes on your phone to sign in", "device_prompt"),
        ("Insert your security key and touch it", "passkey"),
        ("Please verify you are human", "captcha"),
        ("Verify it's you\nTo help keep your account safe", "identity_check"),
        ("Confirm your identity to continue", "identity_check"),
    ])
    def test_kinds(self, text, kind):
        assert ps.classify_text(text)["kind"] == kind

    @pytest.mark.parametrize("text", [
        "Welcome back! Your dashboard is ready.",
        # A cookie banner that says "your device" is not a device prompt.
        "cookies can be stored or read on your device to recognise it each time it connects",
        "",
    ])
    def test_ordinary_pages_are_nothing(self, text):
        assert ps.classify_text(text) is None

    def test_only_a_device_prompt_names_a_device(self):
        """The passkey page says nothing about a phone: page_says must not either."""
        step = ps.classify_text(GOOGLE_PASSKEY_STDOUT)
        assert step["kind"] != "device_prompt" and "phone" not in step["page_says"].lower()

    @pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
    @pytest.mark.parametrize("body,frames,expect", [
        ("Sign in\nUse your passkey to confirm it’s really you\nTry another way", [], "passkey"),
        ("Log in", [("https://www.google.com/recaptcha/api2/anchor?k=x&size=invisible", 256, 60)], None),
        ("Log in", [("https://www.google.com/recaptcha/api2/anchor?k=x&size=normal", 304, 78)], "captcha"),
        ("Log in", [("https://challenges.cloudflare.com/cdn-cgi/challenge-platform/x", 300, 65)], "captcha"),
        ("Hello", [], None),
    ])
    def test_the_page_probe_runs_the_same_patterns_in_javascript(self, tmp_path, body, frames, expect):
        """PROBE_JS executed by a real JS engine against a minimal page: the shared patterns compile
        and match there too (a Python-only regex feature would fail here, not in the owner's browser)."""
        script = tmp_path / "probe.js"
        script.write_text(
            "const frames = %s.map(([src, w, h]) => ({src, getBoundingClientRect: () => ({width: w, height: h})}));\n"
            "globalThis.location = {href: 'https://accounts.example/x'};\n"
            "globalThis.document = {title: 't', body: {innerText: %s},\n"
            "  querySelectorAll: (sel) => sel === 'iframe' ? frames : []};\n"
            "globalThis.getComputedStyle = () => ({visibility: 'visible', display: 'block'});\n"
            "console.log(JSON.stringify(%s));\n" % (json.dumps(frames), json.dumps(body), ps.probe_js()))
        out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        got = ps.parse_probe(json.loads(out.stdout))
        assert (got or {}).get("kind") == expect


# ---- browser_exec says so -------------------------------------------------------------------------

@pytest.fixture()
def cli(monkeypatch):
    calls = {}

    def fake_run(cmd, code, env, timeout):
        calls.update(cmd=cmd, code=code, env=dict(env))
        return subprocess.CompletedProcess(cmd, 0, calls.get("stdout", "ok\n"), "")

    monkeypatch.setattr(bu, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", fake_run)
    monkeypatch.setattr(bu, "_route_backend", lambda env, session, task_id, local: None)
    monkeypatch.setattr(bu, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu, "_workspace_dir", lambda task_id: None)
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {})
    return calls


class TestBrowserExec:
    def test_a_passkey_page_on_a_live_turn_points_to_browser_handoff(self, cli):
        cli["stdout"] = GOOGLE_PASSKEY_STDOUT
        out = json.loads(bu.browser_exec('click_at_xy(1, 2)\nprint(js("document.body.innerText"))', task_id="t"))
        assert out["needs_person"]["kind"] == "passkey"
        assert out["needs_person"]["page_says"] == "Use your passkey to confirm it’s really you"
        assert "browser_handoff" in out["hint"]
        assert "phone" not in out["hint"].replace("their phone or another device", "")  # the rule, not a claim

    def test_on_a_scheduled_job_it_is_reported_not_handed_over(self, cli):
        _unbind(_bind())
        bound = _bind(platform="cron", cron="1")
        try:
            cli["stdout"] = GOOGLE_PASSKEY_STDOUT
            out = json.loads(bu.browser_exec('print(1)', task_id="t"))
        finally:
            _unbind(bound)
        assert out["needs_person"]["code"] == "person_needed"
        assert "report" in out["hint"].lower() and "browser_handoff(" not in out["hint"]

    def test_an_ordinary_page_carries_nothing(self, cli):
        out = json.loads(bu.browser_exec('print(page_info())', task_id="t"))
        assert "needs_person" not in out and "hint" not in out

    def test_the_page_itself_is_probed_when_the_output_says_nothing(self, cli, monkeypatch):
        monkeypatch.setattr(bu, "_route_backend",
                            lambda env, session, task_id, local: env.update(BU_CDP_URL="http://127.0.0.1:9333") or None)
        seen = []
        monkeypatch.setattr(ps, "probe_pages", lambda cdp: seen.append(cdp) or
                            {"kind": "captcha", "page_says": "", "url": "https://shop.example/", "title": "x"})
        out = json.loads(bu.browser_exec('print("done")', task_id="t"))
        assert seen == ["http://127.0.0.1:9333"] and out["needs_person"]["kind"] == "captcha"

    def test_the_schema_teaches_the_handoff_and_the_rule(self):
        desc = bu._HELPERS_DIGEST
        assert "browser_handoff" in desc and "unless the page itself says so" in desc


# ---- browser_handoff ----------------------------------------------------------------------------------

@pytest.fixture()
def handoff(monkeypatch):
    from tools import browser_handoff_tool as bh
    from tools import browser_tool_real_profile as rp
    state = {"shown": {"ok": False, "why": "Moe's own browser is not running"}, "pages": [], "opened": [],
             "chrome": [], "bridge": None}
    monkeypatch.setattr(rp, "show_to_person", lambda url="": state.setdefault("show_calls", []).append(url) or state["shown"])
    monkeypatch.setattr(rp, "driven_pages", lambda: state["pages"])
    monkeypatch.setattr(rp, "hand_back", lambda: {"url": "https://accounts.google.com/after", "title": "Done"})
    monkeypatch.setattr(bh, "_open_default_browser", lambda url: state["opened"].append(url) or None)
    monkeypatch.setattr(bh, "_open_in_chrome_group", lambda bridge, url: state["chrome"].append(url) or None)
    monkeypatch.setattr(lane, "read_bridge", lambda *a, **k: state["bridge"])
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {"chrome_extension": {"enabled": True}})
    monkeypatch.setattr("tools.browser_tool.evaluate_url_safety", lambda url: None, raising=False)
    bh._handoffs.clear()
    return bh, state


class TestHandoff:
    def test_moe_s_own_browser_is_shown_first(self, handoff):
        bh, state = handoff
        state["shown"] = {"ok": True, "url": "https://accounts.google.com/pk", "title": "Sign in", "relaunched": True,
                          "front": True, "form_state_lost": True, "closed_tabs": 0}
        out = json.loads(bh.browser_handoff(reason="confirm the Google passkey", task_id="t"))
        assert out["route"] == "driven_browser" and out["front"] is True
        assert "lost" in out["form_state"]                      # the reload is said, not hidden
        assert "END YOUR TURN" in out["next"] and "unless page_says" in out["next"]
        assert state["opened"] == [] and state["chrome"] == []

    def test_then_the_extension_lane(self, handoff):
        bh, state = handoff
        state["pages"] = [{"id": "1", "url": "https://tinylaunch.com/login", "title": "", "ws": ""}]
        state["shown"] = {"ok": False, "why": "could not relaunch", "url": "https://tinylaunch.com/login"}
        state["bridge"] = {"ws_url": "ws://127.0.0.1:1/devtools/browser/x"}
        out = json.loads(bh.browser_handoff(reason="approve the sign-in", task_id="t"))
        assert out["route"] == "chrome" and state["chrome"] == ["https://tinylaunch.com/login"]
        assert "did not carry over" in out["session"] and out["front"] is False

    def test_last_the_default_browser_with_the_session_caveat(self, handoff):
        bh, state = handoff
        out = json.loads(bh.browser_handoff(reason="solve the CAPTCHA", url="https://shop.example/checkout", task_id="t"))
        assert out["route"] == "default_browser" and state["opened"] == ["https://shop.example/checkout"]
        assert "does NOT carry over" in out["session"]

    def test_a_task_already_in_chrome_stays_there(self, handoff):
        bh, state = handoff
        lane.record_lane("t", "s1", lane.LANE_CHROME)
        out = json.loads(bh.browser_handoff(reason="approve it", task_id="t"))
        assert out["route"] == "chrome" and out["already_open"] is True
        assert state.get("show_calls") is None and state["chrome"] == [] and state["opened"] == []

    def test_nothing_to_show_is_an_error_not_a_guess(self, handoff):
        bh, _ = handoff
        out = json.loads(bh.browser_handoff(reason="approve it", task_id="t"))
        assert "error" in out and "no page open" in out["error"]

    def test_a_turn_nobody_is_at_opens_nothing_and_says_person_needed(self, handoff, monkeypatch):
        bh, state = handoff
        from tools.fix_reasons import host_fields
        monkeypatch.setenv("HERMES_KANBAN_TASK", "k1")
        state["pages"] = [{"id": "1", "url": "https://accounts.google.com/pk", "title": "", "ws": ""}]
        raw = bh.browser_handoff(reason="confirm the passkey", task_id="t")
        out = json.loads(raw)
        assert out["code"] == "person_needed" and out["retry"] is False and out["subject"] == "accounts.google.com"
        assert state.get("show_calls") is None and state["opened"] == [] and state["chrome"] == []
        assert "nothing was opened" in out["error"] and "unless page_says" in out["error"]
        assert host_fields(raw)["code"] == "person_needed"      # a genuine, signed fix for the host

    def test_done_reads_where_the_page_is_now(self, handoff):
        bh, state = handoff
        state["shown"] = {"ok": True, "url": "https://accounts.google.com/pk", "title": "", "front": True}
        bh.browser_handoff(reason="confirm", task_id="t")
        out = json.loads(bh.browser_handoff(done=True, task_id="t"))
        assert out["route"] == "driven_browser" and out["url"] == "https://accounts.google.com/after"

    def test_a_non_web_url_is_refused(self, handoff):
        bh, state = handoff
        out = json.loads(bh.browser_handoff(reason="x", url="file:///etc/passwd", task_id="t"))
        assert "error" in out and state["opened"] == []

    def test_it_is_registered_in_the_browser_toolset(self):
        import toolsets
        from tools import browser_handoff_tool  # noqa: F401 (registers)
        from tools.registry import registry
        assert "browser_handoff" in toolsets.TOOLSETS["browser"]["tools"]
        assert registry.get_entry("browser_handoff").toolset == "browser"


# ---- showing Moe's own browser -----------------------------------------------------------------------

class TestShowToPerson:
    @pytest.fixture()
    def rp(self, monkeypatch):
        from tools import browser_tool_real_profile as rp
        _bt = rp._origin()
        calls = {"cdp": [], "launched_headless": [], "cookies_in": None, "terminated": 0, "front": []}
        monkeypatch.setitem(_bt._real_profile_cdp_cache, "cdp", "http://127.0.0.1:9100")
        monkeypatch.setattr(rp, "_cdp_http_ready", lambda cdp: True)
        monkeypatch.setattr(rp, "driven_pages", lambda: [
            {"id": "A", "url": "https://accounts.google.com/v3/signin/challenge/pk/presend", "title": "Sign in", "ws": ""},
            {"id": "B", "url": "https://tinylaunch.com/", "title": "TinyLaunch", "ws": ""}])
        monkeypatch.setattr("hermes_cli.browser_connect.detect_default_chromium", lambda: "chrome")
        monkeypatch.setattr("hermes_cli.browser_connect.real_profile_copy_dir", lambda b: "/tmp/copy")
        monkeypatch.setattr(rp, "driven_browser_executable",
                            lambda: "/x/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing")
        monkeypatch.setattr(rp, "_cdp_on_data_dir", lambda cdp, d: True)
        monkeypatch.setattr(rp, "_persons_identity", lambda b: None)
        monkeypatch.setattr(rp, "_agent_browser_close_session", lambda name: None)
        monkeypatch.setattr(rp, "_await_holders_gone", lambda d, wait=10.0: None)
        monkeypatch.setattr(rp, "_terminate_real_profile_chrome",
                            lambda: calls.__setitem__("terminated", calls["terminated"] + 1))

        def fake_cdp(port, method, params=None, timeout=20.0):
            calls["cdp"].append((port, method, params))
            if method == "Storage.getCookies":
                return {"cookies": [{"name": "SID", "value": "v", "domain": ".google.com"}]}
            if method == "Target.createTarget":
                return {"targetId": "NEW"}
            if method == "Target.getTargets":
                return {"targetInfos": [{"type": "page", "targetId": "NEW"}, {"type": "page", "targetId": "BLANK"}]}
            return {}
        monkeypatch.setattr(rp, "_cdp_call", fake_cdp)

        def fake_launch(binary, copy_dir, identity=None):
            calls["launched_headless"].append(rp._driven_browser_headless())
            return 9200, None
        monkeypatch.setattr(rp, "_launch_driven_browser", fake_launch)
        monkeypatch.setattr(rp, "_import_cookies_into_driven_browser",
                            lambda port, cookies: calls.__setitem__("cookies_in", (port, cookies)))
        monkeypatch.setattr(rp, "_attach_agent_browser_to_real_profile", lambda port, d: (f"http://127.0.0.1:{port}", None))
        monkeypatch.setattr(rp, "_bring_to_front", lambda port, tid: calls["front"].append((port, tid)) or True)
        monkeypatch.setitem(rp._shown_to_person, "on", False)
        monkeypatch.setitem(rp._launched_headless, "headless", True)
        return rp, calls

    def test_a_headless_browser_is_relaunched_headed_with_the_same_sign_ins_on_the_same_page(self, rp):
        rp, calls = rp
        out = rp.show_to_person()
        assert out["ok"] and out["relaunched"] and out["form_state_lost"] and out["front"]
        assert calls["launched_headless"] == [False]                     # the relaunch is a window
        assert calls["cookies_in"] == (9200, [{"name": "SID", "value": "v", "domain": ".google.com"}])
        assert (9200, "Target.createTarget", {"url": "https://accounts.google.com/v3/signin/challenge/pk/presend"}) in calls["cdp"]
        assert (9200, "Target.closeTarget", {"targetId": "BLANK"}) in calls["cdp"]
        assert calls["front"] == [(9200, "NEW")]
        assert rp._origin()._real_profile_cdp_cache["cdp"] == "http://127.0.0.1:9200"
        assert out["closed_tabs"] == 1                                    # the other tab is said, not hidden

    def test_the_named_tab_is_the_one_shown_and_an_unknown_one_is_refused(self, rp):
        rp, calls = rp
        assert rp.show_to_person("https://tinylaunch.com/")["url"] == "https://tinylaunch.com/"
        out = rp.show_to_person("https://elsewhere.example/")
        assert not out["ok"] and "not open" in out["why"]

    def test_an_already_headed_browser_is_raised_not_relaunched(self, rp, monkeypatch):
        rp, calls = rp
        monkeypatch.setitem(rp._launched_headless, "headless", False)
        out = rp.show_to_person()
        assert out["ok"] and not out["relaunched"] and not out["form_state_lost"]
        assert calls["launched_headless"] == [] and calls["terminated"] == 0
        assert calls["front"] == [(9100, "A")]

    def test_while_shown_a_relaunch_stays_headed_and_hand_back_restores_headless(self, rp, monkeypatch):
        rp, _ = rp
        monkeypatch.setattr(rp._cloud, "_is_headed_mode", lambda: False)
        rp.show_to_person()
        assert rp._driven_browser_headless() is False
        rp.hand_back()
        assert rp._driven_browser_headless() is True

    @pytest.mark.parametrize("cmdlines,headless", [
        (["/x/Chrome for Testing --user-data-dir=/tmp/copy --headless=new", "/x/Helper --type=renderer"], True),
        (["/x/Chrome for Testing --user-data-dir=/tmp/copy --no-startup-window"], False),
        ([], True),                                   # nothing found: hidden, never "it's in front of you"
    ])
    def test_a_browser_another_process_launched_is_judged_by_its_command_line(self, rp, monkeypatch, cmdlines, headless):
        rp, _ = rp
        monkeypatch.setitem(rp._launched_headless, "headless", None)

        class P:
            def __init__(self, c):
                self.c = c

            def cmdline(self):
                return self.c.split()
        monkeypatch.setattr(rp, "_browsers_on_data_dir", lambda d: [P(c) for c in cmdlines])
        assert rp._is_headless_now("/tmp/copy") is headless

    def test_a_browser_that_is_not_ours_is_never_relaunched(self, rp, monkeypatch):
        rp, calls = rp
        monkeypatch.setattr(rp, "_cdp_on_data_dir", lambda cdp, d: False)
        out = rp.show_to_person()
        assert not out["ok"] and calls["terminated"] == 0 and calls["launched_headless"] == []

    def test_a_failed_relaunch_says_the_browser_is_gone(self, rp, monkeypatch):
        rp, calls = rp
        monkeypatch.setattr(rp, "_launch_driven_browser", lambda *a, **k: (None, "did not start"))
        out = rp.show_to_person()
        assert not out["ok"] and out["browser_gone"] and rp._shown_to_person["on"] is False


# ---- grounded words ------------------------------------------------------------------------------------

class TestGroundedWords:
    def test_no_tool_text_tells_the_model_to_send_the_person_to_their_device(self):
        from tools import browser_handoff_tool as bh
        from tools import browser_vault_tool as vt
        import model_tools
        texts = [vt.BROWSER_VAULT_ENTER_CODE_SCHEMA["description"], bh.BROWSER_HANDOFF_SCHEMA["description"],
                 bu._HELPERS_DIGEST, model_tools._VAULT_NO_PASSWORD_NOTE, open(vt.__file__).read()]
        for t in texts:
            assert "complete it on their device" not in t
            assert "approve it on their device" not in t

    def test_the_input_tool_note_routes_a_code_the_person_says_to_the_vault(self):
        """The note on the input tool used to forbid accepting a code "even if the user shows it"."""
        import model_tools
        assert "source='person'" in model_tools._VAULT_NO_PASSWORD_NOTE
