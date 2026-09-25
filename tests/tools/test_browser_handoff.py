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
import time

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


def _stamp(path, active=True, age=5.0):
    path.write_text(json.dumps({"active": active, "at": time.time() - age, "idle": 3}))


@pytest.fixture(autouse=True)
def _never_launch_or_raise_anything(monkeypatch):
    """No test here may start a browser or raise a window on the Mac running it (see 8641d9e17b)."""
    real_popen = subprocess.Popen

    def guarded(argv, *a, **k):
        exe = str(argv[0] if isinstance(argv, (list, tuple)) else argv).lower()
        if "chrom" in exe or exe.endswith(("osascript", "/open")):
            raise AssertionError(f"test tried to run {exe}; stub it")
        return real_popen(argv, *a, **k)
    monkeypatch.setattr(subprocess, "Popen", guarded)


@pytest.fixture(autouse=True)
def _live_turn_at_the_mac(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    stamp = tmp_path / "presence.json"
    _stamp(stamp)
    monkeypatch.setenv("HERMES_PRESENCE_FILE", str(stamp))
    lane.reset_sticky_lanes()
    bound = _bind()
    yield stamp
    _unbind(bound)
    lane.reset_sticky_lanes()


# Page text as a person reads it (innerText), for pages that must NOT be a person step. Modelled on
# the reviewer's measurements, 2026-09-25: a login page that OFFERS a passkey, and articles about them.
GITHUB_LOGIN_TEXT = ("Skip to content\nSign in to GitHub\nUsername or email address\nPassword\nForgot password?\n"
                     "Sign in\nor\nSign in with a passkey\nNew to GitHub? Create an account")
GOOGLE_HELP_TEXT = ("Sign in with a passkey instead of a password - Google Account Help\n"
                    + "With a passkey you can sign in with your fingerprint, face or screen lock. "
                    "Use your passkey to sign in to your Google Account on any device. " * 12)
WIKIPEDIA_TEXT = ("Passkey\nFrom Wikipedia, the free encyclopedia\nA passkey is a FIDO credential... "
                  + "Security key, Touch ID, Windows Hello and phones can hold passkeys. " * 40)


# ---- what the page is ------------------------------------------------------------------------

class TestDetection:
    def test_the_measured_google_passkey_page_is_a_passkey_step(self):
        step = ps.classify_page(GOOGLE_PASSKEY_STDOUT)
        assert step == {"kind": "passkey", "page_says": "Use your passkey to confirm it’s really you"}

    @pytest.mark.parametrize("text,kind", [
        ("Check your phone\nGoogle sent a notification to your Pixel", "device_prompt"),
        ("Tap Yes on your phone to sign in", "device_prompt"),
        ("Insert your security key and touch it", "passkey"),
        ("Please verify you are human", "captcha"),
        ("Verify it's you\nTo help keep your account safe", "identity_check"),
        ("Confirm your identity to continue", "identity_check"),
    ])
    def test_imperative_kinds_on_a_short_page_with_nothing_to_fill(self, text, kind):
        assert ps.classify_page(text)["kind"] == kind

    @pytest.mark.parametrize("text,inputs", [
        (GITHUB_LOGIN_TEXT, 2),           # a passkey is one option beside a password form
        (GOOGLE_HELP_TEXT, 0),            # an article: long, and not asking for anything
        (WIKIPEDIA_TEXT, 1),              # the search box aside, long
        ("Welcome back! Your dashboard is ready.", 0),
        ("cookies can be stored or read on your device to recognise it each time it connects", 0),
        ("", 0),
    ])
    def test_pages_that_only_mention_it_are_nothing(self, text, inputs):
        assert ps.classify_page(text, fillable_inputs=inputs) is None

    def test_a_webauthn_request_in_flight_is_a_passkey_step_whatever_the_words(self):
        assert ps.classify_page("Continue", fillable_inputs=3, webauthn_pending=True)["kind"] == "passkey"

    def test_output_only_decides_whether_to_look(self):
        assert ps.suggests_person_step(GOOGLE_PASSKEY_STDOUT)
        assert ps.suggests_person_step("https://accounts.google.com/v3/signin/challenge/pk")
        assert not ps.suggests_person_step("{'url': 'https://example.com/', 'title': 'Example'}\nok")

    def test_only_a_device_prompt_names_a_device(self):
        step = ps.classify_page(GOOGLE_PASSKEY_STDOUT)
        assert step["kind"] != "device_prompt" and "phone" not in step["page_says"].lower()

    def test_the_fidelity_keeper_installs_the_webauthn_hook_before_any_page_script(self):
        from tools.browser_tool_fidelity import commands_for_attached_target
        cmds = commands_for_attached_target({"targetInfo": {"type": "page"}}, "UA", {})
        methods = [m for m, _ in cmds]
        i = methods.index("Page.addScriptToEvaluateOnNewDocument")
        assert i < methods.index("Runtime.runIfWaitingForDebugger")
        assert cmds[i][1]["source"] == ps.WEBAUTHN_HOOK_JS and cmds[i][1]["runImmediately"] is True


def _find_node():
    import os
    import pwd
    home = pwd.getpwuid(os.getuid()).pw_dir  # not $HOME: other suites point it at a temp dir
    for cand in (shutil.which("node"), os.path.join(home, ".local/bin/node"), os.path.join(home, ".hermes/node/bin/node"),
                 "/opt/homebrew/bin/node", "/usr/local/bin/node"):
        if cand and os.path.exists(cand):
            return cand
    return None


NODE = _find_node()


def _run_js(tmp_path, body, *, frames=(), inputs=(), calls=()):
    """PROBE_JS (after WEBAUTHN_HOOK_JS and the page's own navigator.credentials calls) in node,
    against a minimal page. The shared patterns compile and match in a real JS engine."""
    script = tmp_path / "probe.js"
    script.write_text(
        "globalThis.window = globalThis;\n"
        "class CredentialsContainer { get(o) { return new Promise(() => {}); } create(o) { return new Promise(() => {}); } }\n"
        "globalThis.CredentialsContainer = CredentialsContainer;\n"
        "const creds = new CredentialsContainer();\n"
        "const rect = (w, h) => () => ({width: w, height: h});\n"
        "const frames = %s.map(([src, w, h]) => ({src, getBoundingClientRect: rect(w, h)}));\n"
        "const inputs = %s.map((t) => ({type: t, getBoundingClientRect: rect(200, 30)}));\n"
        "globalThis.location = {href: 'https://accounts.example/x'};\n"
        "globalThis.document = {title: 't', body: {innerText: %s},\n"
        "  querySelectorAll: (sel) => sel === 'iframe' ? frames : (sel.startsWith('input') ? inputs : [])};\n"
        "globalThis.getComputedStyle = () => ({visibility: 'visible', display: 'block'});\n"
        "%s\n"
        "for (const c of %s) creds[c[0]](c[1]);\n"
        "console.log(JSON.stringify(%s));\n"
        % (json.dumps(list(frames)), json.dumps(list(inputs)), json.dumps(body), ps.WEBAUTHN_HOOK_JS,
           json.dumps(list(calls)), ps.probe_js()))
    out = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return ps.parse_probe(json.loads(out.stdout))


@pytest.mark.skipif(NODE is None, reason="node is not installed")
class TestTheProbeInJavaScript:
    def test_the_google_passkey_page(self, tmp_path):
        got = _run_js(tmp_path, "Sign in\nUse your passkey to confirm it’s really you\nTry another way")
        assert got["kind"] == "passkey" and got["why"] == "imperative"

    def test_github_login_offering_a_passkey_is_not(self, tmp_path):
        assert _run_js(tmp_path, GITHUB_LOGIN_TEXT, inputs=["text", "password"],
                       calls=[["get", {"publicKey": {}, "mediation": "conditional"}]]) is None

    @pytest.mark.parametrize("text", [GOOGLE_HELP_TEXT, WIKIPEDIA_TEXT])
    def test_articles_are_not(self, tmp_path, text):
        assert _run_js(tmp_path, text, inputs=["search"]) is None

    def test_a_webauthn_get_in_flight_is_a_passkey_step(self, tmp_path):
        got = _run_js(tmp_path, "Continue", inputs=["text"], calls=[["get", {"publicKey": {"challenge": 1}}]])
        assert got["kind"] == "passkey" and got["why"] == "webauthn"

    @pytest.mark.parametrize("frames,expect", [
        ([("https://www.google.com/recaptcha/api2/anchor?k=x&size=invisible", 256, 60)], None),
        ([("https://www.google.com/recaptcha/api2/anchor?k=x&size=normal", 304, 78)], "captcha"),
        ([("https://challenges.cloudflare.com/cdn-cgi/challenge-platform/x", 300, 65)], "captcha"),
    ])
    def test_only_a_visible_challenge_is_a_captcha(self, tmp_path, frames, expect):
        assert (_run_js(tmp_path, "Log in", frames=frames, inputs=["text", "password"]) or {}).get("kind") == expect


# ---- at the Mac, or not ----------------------------------------------------------------------------

class TestAtThisMac:
    def test_the_app_s_own_turn_with_a_fresh_stamp_is_here(self):
        assert lane.at_this_mac()[0] is True

    @pytest.mark.parametrize("active,age", [(False, 5), (True, 600)])
    def test_locked_idle_asleep_or_stale_is_not(self, _live_turn_at_the_mac, active, age):
        _stamp(_live_turn_at_the_mac, active=active, age=age)
        assert lane.at_this_mac()[0] is False

    def test_no_stamp_on_a_host_that_writes_one_is_not(self, _live_turn_at_the_mac):
        _live_turn_at_the_mac.unlink()
        assert lane.at_this_mac()[0] is False

    def test_the_owner_texting_from_their_phone_is_live_but_not_here(self):
        bound = _bind(platform="telegram", origin="")
        try:
            p = lane.turn_presence()
            here, why = lane.at_this_mac(p)
        finally:
            _unbind(bound)
        assert p["live"] is True and p["surface"] == "chat" and here is False and "telegram" in why


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


def _probe_returns(monkeypatch, cli, result):
    monkeypatch.setattr(bu, "_route_backend",
                        lambda env, session, task_id, local: env.update(BU_CDP_URL="http://127.0.0.1:9333") or None)
    seen = []
    monkeypatch.setattr(ps, "probe_active_page", lambda cdp, **k: seen.append(cdp) or result)
    return seen


PASSKEY_PROBE = {"kind": "passkey", "page_says": "Use your passkey to confirm it’s really you", "why": "imperative",
                 "url": "https://accounts.google.com/v3/signin/challenge/pk/presend", "title": "Sign in"}


class TestBrowserExec:
    def test_a_passkey_page_at_the_mac_points_to_browser_handoff(self, cli, monkeypatch):
        seen = _probe_returns(monkeypatch, cli, PASSKEY_PROBE)
        cli["stdout"] = GOOGLE_PASSKEY_STDOUT
        out = json.loads(bu.browser_exec('click_at_xy(1, 2)\nprint(js("document.body.innerText"))', task_id="t"))
        assert seen == ["http://127.0.0.1:9333"]
        assert out["needs_person"]["kind"] == "passkey"
        assert out["needs_person"]["page_says"] == "Use your passkey to confirm it’s really you"
        assert "browser_handoff" in out["hint"]
        assert "phone" not in out["hint"].replace("their phone or another device", "")  # the rule, not a claim

    def test_the_output_alone_is_never_the_verdict(self, cli, monkeypatch):
        """github.com/login prints "Sign in with a passkey"; the page itself says there is a password form."""
        _probe_returns(monkeypatch, cli, None)
        cli["stdout"] = GITHUB_LOGIN_TEXT
        out = json.loads(bu.browser_exec('print(js("document.body.innerText"))', task_id="t"))
        assert "needs_person" not in out

    def test_a_quiet_output_is_not_probed_at_all(self, cli, monkeypatch):
        seen = _probe_returns(monkeypatch, cli, PASSKEY_PROBE)
        cli["stdout"] = "{'url': 'https://example.com/', 'title': 'Example'}\nsaved"
        out = json.loads(bu.browser_exec('print(page_info())', task_id="t"))
        assert seen == [] and "needs_person" not in out

    def test_on_a_scheduled_job_it_is_reported_with_words_for_the_owner(self, cli, monkeypatch):
        _probe_returns(monkeypatch, cli, PASSKEY_PROBE)
        bound = _bind(platform="cron", cron="1")
        try:
            cli["stdout"] = GOOGLE_PASSKEY_STDOUT
            out = json.loads(bu.browser_exec('print(1)', task_id="t"))
        finally:
            _unbind(bound)
        assert out["needs_person"]["code"] == "person_needed"
        assert "reach_owner(text)" in out["hint"] and "browser_handoff(" not in out["hint"]
        assert "accounts.google.com" in out["needs_person"]["tell_owner"]

    def test_the_owner_texting_from_away_gets_words_not_a_window(self, cli, monkeypatch):
        _probe_returns(monkeypatch, cli, PASSKEY_PROBE)
        bound = _bind(platform="whatsapp", origin="")
        try:
            cli["stdout"] = GOOGLE_PASSKEY_STDOUT
            out = json.loads(bu.browser_exec('print(1)', task_id="t"))
        finally:
            _unbind(bound)
        assert out["needs_person"]["code"] == "person_needed" and "in your reply" in out["hint"]
        assert "browser_handoff(" not in out["hint"]

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
    monkeypatch.setattr(rp, "show_to_person",
                        lambda url="", task="": state.setdefault("show_calls", []).append((url, task)) or state["shown"])
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
                          "front": True, "form_state_lost": True, "reopened": 2}
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
        assert "reach_owner(text)" in out["error"] and out["tell_owner"]
        assert host_fields(raw)["code"] == "person_needed"      # a genuine, signed fix for the host

    @pytest.mark.parametrize("platform", ["telegram", "whatsapp", "slack", "discord"])
    def test_the_owner_writing_from_away_gets_no_window_and_words_to_pass_on(self, handoff, platform):
        bh, state = handoff
        state["pages"] = [{"id": "1", "url": "https://accounts.google.com/pk", "title": "", "ws": ""}]
        bound = _bind(platform=platform, origin="")
        try:
            out = json.loads(bh.browser_handoff(reason="confirm the passkey", task_id="t"))
        finally:
            _unbind(bound)
        assert out["code"] == "person_needed" and "in your reply" in out["error"]
        assert state.get("show_calls") is None and state["opened"] == [] and state["chrome"] == []
        assert "accounts.google.com/pk" in out["tell_owner"] and "confirm the passkey" in out["tell_owner"]
        assert "phone" not in out["tell_owner"]

    def test_a_locked_mac_gets_no_window(self, handoff, _live_turn_at_the_mac):
        bh, state = handoff
        _stamp(_live_turn_at_the_mac, active=False)
        state["shown"] = {"ok": True, "url": "https://x.example/", "title": "", "front": True}
        out = json.loads(bh.browser_handoff(reason="confirm", url="https://x.example/", task_id="t"))
        assert out["code"] == "person_needed" and state.get("show_calls") is None

    def test_the_task_is_passed_so_other_conversations_are_protected(self, handoff):
        bh, state = handoff
        state["shown"] = {"ok": True, "url": "https://x.example/", "title": "", "front": True}
        bh.browser_handoff(reason="confirm", task_id="conv-7")
        assert state["show_calls"] == [("", "conv-7")]

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
        calls = {"cdp": [], "launched_headless": [], "cookies_in": None, "terminated": 0, "front": [], "made": 0}
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
        monkeypatch.setattr(rp, "_other_work", lambda task: [])
        monkeypatch.setattr(rp, "_main_browser_pid", lambda d: 4242)
        monkeypatch.setattr(rp, "_terminate_real_profile_chrome",
                            lambda: calls.__setitem__("terminated", calls["terminated"] + 1))

        def fake_cdp(port, method, params=None, timeout=20.0):
            calls["cdp"].append((port, method, params))
            if method == "Storage.getCookies":
                return {"cookies": [{"name": "SID", "value": "v", "domain": ".google.com"}]}
            if method == "Target.createTarget":
                calls["made"] += 1
                return {"targetId": f"NEW{calls['made']}"}
            if method == "Target.getTargets":
                return {"targetInfos": [{"type": "page", "targetId": "BLANK"}]}
            return {}
        monkeypatch.setattr(rp, "_cdp_call", fake_cdp)

        def fake_launch(binary, copy_dir, identity=None):
            calls["launched_headless"].append(rp._driven_browser_headless())
            return 9200, None
        monkeypatch.setattr(rp, "_launch_driven_browser", fake_launch)
        monkeypatch.setattr(rp, "_import_cookies_into_driven_browser",
                            lambda port, cookies: calls.__setitem__("cookies_in", (port, cookies)))
        monkeypatch.setattr(rp, "_attach_agent_browser_to_real_profile", lambda port, d: (f"http://127.0.0.1:{port}", None))
        monkeypatch.setattr(rp, "_bring_to_front", lambda port, tid, pid: calls["front"].append((port, tid, pid)) or True)
        monkeypatch.setitem(rp._shown_to_person, "on", False)
        monkeypatch.setitem(rp._launched_headless, "headless", True)
        return rp, calls

    def test_a_headless_browser_is_relaunched_headed_with_every_tab_and_the_same_sign_ins(self, rp):
        rp, calls = rp
        out = rp.show_to_person()
        assert out["ok"] and out["relaunched"] and out["form_state_lost"] and out["front"]
        assert calls["launched_headless"] == [False]                     # the relaunch is a window
        assert calls["cookies_in"] == (9200, [{"name": "SID", "value": "v", "domain": ".google.com"}])
        made = [p["url"] for port, m, p in calls["cdp"] if m == "Target.createTarget"]
        # every tab back, the chosen one last so it is in front
        assert made == ["https://tinylaunch.com/", "https://accounts.google.com/v3/signin/challenge/pk/presend"]
        assert (9200, "Target.closeTarget", {"targetId": "BLANK"}) in calls["cdp"]
        assert calls["front"] == [(9200, "NEW2", 4242)]
        assert out["reopened"] == 2 and out["tabs_lost"] == 0
        assert rp._origin()._real_profile_cdp_cache["cdp"] == "http://127.0.0.1:9200"

    def test_another_conversation_using_the_browser_is_never_cut_off(self, rp, monkeypatch):
        rp, calls = rp
        monkeypatch.setattr(rp, "_other_work", lambda task: ["1 other conversation(s) used it in the last few minutes"])
        out = rp.show_to_person(task_id="t")
        assert not out["ok"] and out["busy"] and "other conversation" in out["why"]
        assert calls["terminated"] == 0 and calls["launched_headless"] == []

    def test_other_work_is_read_from_this_process_s_lanes(self, monkeypatch):
        from tools import browser_tool_real_profile as rp
        lane.record_lane("mine", "", lane.LANE_OWN)
        assert rp._other_work("mine") == []
        lane.record_lane("theirs", "s2", lane.LANE_OWN)
        assert any("other conversation" in r for r in rp._other_work("mine"))

    def test_a_headless_browser_another_process_launched_is_not_restarted_from_here(self, rp, monkeypatch):
        rp, calls = rp
        monkeypatch.setitem(rp._launched_headless, "headless", None)
        monkeypatch.setattr(rp, "_is_headless_now", lambda d: True)
        out = rp.show_to_person()
        assert not out["ok"] and out["busy"] and "another part of Moe" in out["why"]
        assert calls["terminated"] == 0 and calls["launched_headless"] == []

    def test_a_headed_browser_another_process_launched_is_raised_in_place(self, rp, monkeypatch):
        rp, calls = rp
        monkeypatch.setitem(rp._launched_headless, "headless", None)
        monkeypatch.setattr(rp, "_is_headless_now", lambda d: False)
        out = rp.show_to_person()
        assert out["ok"] and not out["relaunched"] and calls["front"] == [(9100, "A", 4242)]
        assert calls["terminated"] == 0

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
        assert calls["front"] == [(9100, "A", 4242)]

    def test_while_shown_a_relaunch_stays_headed_and_hand_back_restores_headless(self, rp, monkeypatch):
        rp, _ = rp
        monkeypatch.setattr(rp._cloud, "_is_headed_mode", lambda: False)
        rp.show_to_person()
        assert rp._driven_browser_headless() is False
        rp.hand_back()
        assert rp._driven_browser_headless() is True

    def test_the_idle_reaper_never_closes_the_window_the_person_is_using(self, rp, monkeypatch):
        rp, calls = rp
        _bt = rp._origin()
        monkeypatch.setattr(rp, "_real_profile_in_use", lambda: False)
        rp.show_to_person()
        long_after = time.time() + _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT + 60
        before = calls["terminated"]
        assert rp.release_if_idle(now=long_after) is False and calls["terminated"] == before
        rp.hand_back()                                   # done: the idle clock starts from now
        assert rp.release_if_idle(now=time.time() + 5) is False
        assert rp.release_if_idle(now=time.time() + _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT + 60) is True

    def test_a_window_nobody_hands_back_is_released_after_the_ceiling(self, rp, monkeypatch):
        rp, _ = rp
        monkeypatch.setattr(rp, "_real_profile_in_use", lambda: False)
        rp.show_to_person()
        assert rp.release_if_idle(now=time.time() + rp.SHOWN_MAX_SECONDS + 60) is True

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


class TestRaiseByPid:
    def test_the_running_process_is_raised_by_its_pid_never_by_bundle(self, monkeypatch):
        from tools import browser_tool_real_profile as rp
        ran = []
        monkeypatch.setattr(rp.sys, "platform", "darwin")
        monkeypatch.setattr(rp.subprocess, "run", lambda argv, **k: ran.append(argv) or
                            subprocess.CompletedProcess(argv, 0, "true\n", ""))
        assert rp._activate_pid(4242) is True
        assert ran[0][:3] == ["/usr/bin/osascript", "-l", "JavaScript"] and "(4242)" in ran[0][-1]
        assert not any("/usr/bin/open" in a for argv in ran for a in argv)

    def test_no_pid_raises_nothing(self, monkeypatch):
        from tools import browser_tool_real_profile as rp
        monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: pytest.fail("ran something"))
        assert rp._activate_pid(0) is False


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
