"""Handing the page to the person: detection, browser_handoff's routes, and grounded words (slice D).

Measured 2026-09-24/25 (Moe transcript 42bec772, read-only): a Google sign-in in Moe's headless
browser printed "Use your passkey to confirm it’s really you" at
accounts.google.com/v3/signin/challenge/pk/presend, and Moe told the owner "Google's asking you to
approve this sign-in on your phone" -- nothing had been sent to a phone. The words came from the
vault tool's no_code_field hint ("tell the user to complete it on their device") and a skill.
"""

from __future__ import annotations

import json
import os
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

GOOGLE_CTX = {"title": "Sign in - Google Accounts", "buttons": ["Try another way"]}


class TestDetection:
    def test_the_measured_google_passkey_page_is_a_passkey_step(self):
        step = ps.classify_page(GOOGLE_PASSKEY_STDOUT, **GOOGLE_CTX)
        assert step == {"kind": "passkey", "page_says": "Use your passkey to confirm it’s really you"}

    @pytest.mark.parametrize("text,kind,ctx", [
        ("Check your phone\nGoogle sent a notification to your Pixel", "device_prompt", {"title": "2-Step Verification"}),
        ("Tap Yes on your phone to sign in", "device_prompt", {"buttons": ["Try another way"]}),
        ("Insert your security key and touch it", "passkey", {"title": "Security key"}),
        ("Please verify you are human", "captcha", {"buttons": ["Continue"]}),
        ("Verify it's you\nTo help keep your account safe", "identity_check", {"headings": ["Verify it's you"]}),
        ("Confirm your identity to continue", "identity_check", {"title": "Confirm your identity"}),
        # Common phrasings (review round 3)
        ("Use your face, fingerprint, PIN or security key", "passkey", {"title": "Sign in to your account"}),
        ("Use Touch ID or your passkey", "passkey", {"buttons": ["Use another method"]}),
        ("Authenticate with a passkey", "passkey", {"headings": ["Two-factor authentication"]}),
        ("A verification code has been sent to your Apple devices", "device_prompt", {"title": "Apple Account"}),
        ("Allow this sign in on your iPhone", "device_prompt", {"headings": ["Sign in with Apple"]}),
        ("Utilisez votre clé d'accès pour vous connecter", "passkey", {"title": "Connexion"}),
        ("Confirmez qu'il s'agit bien de vous", "identity_check", {"headings": ["Confirmez qu'il s'agit bien de vous"]}),
        ("Verwenden Sie Ihren Passkey", "passkey", {"buttons": ["Andere Methode"]}),
        ("Bestätigen Sie, dass Sie es sind", "identity_check", {"headings": ["Bestätigen Sie, dass Sie es sind"]}),
        ("Usa tu llave de acceso", "passkey", {"title": "Iniciar sesión"}),
        ("Verifica tu identidad", "identity_check", {"headings": ["Verifica tu identidad"]}),
        ("השתמש במפתח הגישה שלך", "passkey", {"title": "התחברות"}),
        ("אימות זהות", "identity_check", {"headings": ["אימות זהות"]}),
    ])
    def test_imperative_kinds_in_a_challenge_context(self, text, kind, ctx):
        assert (ps.classify_page(text, **ctx) or {}).get("kind") == kind

    @pytest.mark.parametrize("text,kw", [
        (GITHUB_LOGIN_TEXT, {"fillable_inputs": 2, "title": "Sign in to GitHub"}),  # one option beside a form
        (GOOGLE_HELP_TEXT, {"title": "Sign in with a passkey - Google Account Help"}),  # an article: long
        (WIKIPEDIA_TEXT, {"fillable_inputs": 0, "title": "Passkey - Wikipedia"}),
        # A short help snippet: the words and a sign-in title, but article structure
        ("Use your passkey to sign in to your Google Account.", {"title": "Sign in with a passkey", "article": True}),
        # The words with no challenge context at all
        ("Use your passkey to confirm it's you", {}),
        # "Verify your identity" on a cart page: a Continue button is not an identity-check context
        ("Verify your identity\nYour cart (2 items)", {"title": "Your cart - Shop", "buttons": ["Continue"]}),
        # Review, final round: generic context words are not an identity-check title
        ("Verify your identity", {"title": "Security settings", "headings": ["Account security"]}),
        ("Confirm your identity", {"headings": ["Confirm your order"], "buttons": ["Continue"]}),
        # The reviewer's short help page: a heading, one paragraph, a Continue button
        ("Sign in with a passkey\nWith a passkey you can sign in with your fingerprint, face or screen lock. "
         "Use your passkey to sign in to your account on any of your devices.\nContinue",
         {"headings": ["Sign in with a passkey"], "buttons": ["Continue"]}),
        ("Welcome back! Your dashboard is ready.", {"title": "Sign in"}),
        ("", {}),
    ])
    def test_pages_that_only_mention_it_are_nothing(self, text, kw):
        assert ps.classify_page(text, **kw) is None

    def test_output_only_decides_whether_to_look(self):
        assert ps.suggests_person_step(GOOGLE_PASSKEY_STDOUT)
        assert ps.suggests_person_step("https://accounts.google.com/v3/signin/challenge/pk")
        assert not ps.suggests_person_step("{'url': 'https://example.com/', 'title': 'Example'}\nok")

    def test_only_a_device_prompt_names_a_device(self):
        step = ps.classify_page(GOOGLE_PASSKEY_STDOUT, **GOOGLE_CTX)
        assert step["kind"] != "device_prompt" and "phone" not in step["page_says"].lower()

    def test_detection_is_documented_as_a_hint_and_the_rule_is_handoff(self):
        from tools import browser_handoff_tool as bh
        assert "HINT" in ps.__doc__ and "English first" in ps.__doc__
        d = bh.BROWSER_HANDOFF_SCHEMA["description"]
        assert "a check only the person can complete" in d and "only a hint" in d
        assert "whether or not a result flags needs_person" in bu._HELPERS_DIGEST

    def test_nothing_is_injected_into_pages(self):
        """The WebAuthn hook was dropped (a Proxy is not native to toString; see browser_person_step):
        the fidelity keeper must not bring a page script back."""
        from tools.browser_tool_fidelity import commands_for_attached_target
        for kind in ("page", "iframe", "webview"):
            methods = [m for m, _ in commands_for_attached_target({"targetInfo": {"type": kind}}, "UA", {})]
            assert "Page.addScriptToEvaluateOnNewDocument" not in methods
        assert not hasattr(ps, "WEBAUTHN_HOOK_JS") and "Symbol.for" not in ps.probe_js()


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

_DOM = """
const S = %s;
const withStyle = (o, style) => Object.assign(o, {nodeType: 1, style: style || {}});
const rectOf = ([x, y, w, h]) => () => ({left: x, top: y, width: w, height: h, right: x + w, bottom: y + h});
const mkInput = (i) => {
  const parent = withStyle({parentElement: null, getAttribute: (n) => (n === "aria-hidden" && i.parentAriaHidden) ? "true" : null,
                            getBoundingClientRect: rectOf(i.parentRect || [0, 0, 800, 600])}, i.parentStyle);
  return withStyle({parentElement: parent, tagName: (i.tag || "INPUT").toUpperCase(), type: i.type || "text",
    name: i.name || "", id: i.id || "", placeholder: i.placeholder || "", disabled: !!i.disabled, readOnly: !!i.readOnly,
    getAttribute: (n) => n === "aria-hidden" ? (i.ariaHidden ? "true" : null) : (n === "role" ? (i.role || null) : null),
    getBoundingClientRect: rectOf(i.rect || [10, 10, 200, 30])}, i.style);
};
const frames = (S.frames || []).map(([src, w, h]) => withStyle({src, getBoundingClientRect: rectOf([0, 0, w, h])}, {}));
const inputs = (S.inputs || []).map(mkInput);
const paras = (S.paragraphs || []).map((p) => typeof p === "string" ? {textContent: p}
  : {textContent: p.text, closest: (sel) => (p.footer && /footer/.test(sel)) ? {} : null});
const heads = (S.headings || []).map((t) => ({textContent: t}));
const buttons = (S.buttons || []).map((t) => ({innerText: t, textContent: t}));
const articles = S.article ? [{}] : [];
globalThis.window = globalThis; globalThis.scrollX = 0; globalThis.scrollY = 0; globalThis.innerWidth = 1280;
globalThis.location = {href: "https://accounts.example/x"};
globalThis.document = {title: S.title || "", body: {innerText: S.body}, documentElement: {scrollWidth: 1280},
  querySelectorAll: (sel) => sel === "iframe" ? frames : sel.startsWith("input") ? inputs : sel === "p" ? paras
    : sel === "article" ? articles : sel.startsWith("h1") ? heads : sel.startsWith("button") ? buttons : []};
globalThis.getComputedStyle = (n) => Object.assign({visibility: "visible", display: "block", opacity: "1",
  clip: "auto", clipPath: "none"}, n.style || {});
console.log(JSON.stringify(%s));
"""


def _run_js(tmp_path, body, **page):
    """PROBE_JS in node against a small page model (real enough for its rules: rects, computed styles,
    parents, attributes): the shared patterns and the visibility rules run in a real JS engine."""
    script = tmp_path / "probe.js"
    script.write_text(_DOM % (json.dumps({"body": body, **page}), ps.probe_js()))
    out = subprocess.run([NODE, str(script)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return ps.parse_probe(json.loads(out.stdout))


GOOGLE_PK_BODY = "Sign in\nUse your passkey to confirm it’s really you\nTry another way"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
class TestTheProbeInJavaScript:
    def test_the_google_passkey_page(self, tmp_path):
        got = _run_js(tmp_path, GOOGLE_PK_BODY, **GOOGLE_CTX)
        assert got["kind"] == "passkey" and got["why"] == "imperative"

    @pytest.mark.parametrize("hidden", [
        {"type": "email", "rect": [-5000, 10, 300, 40]},                         # off screen (Google's email field)
        {"type": "email", "style": {"opacity": "0"}},                              # transparent
        {"type": "text", "style": {"clip": "rect(0px, 0px, 0px, 0px)"}},           # clipped away
        {"type": "text", "rect": [10, 10, 1, 1]},                                  # sr-only 1x1
        {"type": "text", "parentStyle": {"display": "none"}},                      # hidden parent
        {"type": "text", "parentStyle": {"visibility": "hidden"}},                 # hidden parent
        {"type": "text", "parentAriaHidden": True},                                # aria-hidden parent
        {"type": "text", "ariaHidden": True},
        {"type": "hidden"},
        {"type": "text", "disabled": True},
        {"type": "text", "readOnly": True},
        {"tag": "select", "type": "select-one", "name": "hl"},                     # the language picker
        {"type": "text", "name": "q", "placeholder": "Search"},                    # a search box
        {"type": "text", "role": "searchbox"},
        {"type": "email", "rect": [5000, 10, 300, 40]},                          # off screen to the right
        {"type": "text", "parentStyle": {"overflow": "hidden"}, "parentRect": [0, 0, 800, 0]},  # 0-height clip box
    ])
    def test_fields_not_really_on_screen_do_not_stop_it(self, tmp_path, hidden):
        """Review round 3, measured: each of these left the old "rect > 0" rule counting a field. Revert
        to `visible(el) || rect > 0` and the transparent, clipped and hidden-parent cases go red."""
        got = _run_js(tmp_path, GOOGLE_PK_BODY, inputs=[hidden], **GOOGLE_CTX)
        assert got and got["kind"] == "passkey", hidden

    def test_a_real_field_on_screen_is_another_way_forward(self, tmp_path):
        assert _run_js(tmp_path, GOOGLE_PK_BODY, inputs=[{"type": "password"}], **GOOGLE_CTX) is None

    def test_github_login_offering_a_passkey_is_not(self, tmp_path):
        assert _run_js(tmp_path, GITHUB_LOGIN_TEXT, inputs=[{"type": "text", "name": "login"}, {"type": "password"}],
                       title="Sign in to GitHub", buttons=["Sign in", "Sign in with a passkey"]) is None

    @pytest.mark.parametrize("text", [GOOGLE_HELP_TEXT, WIKIPEDIA_TEXT])
    def test_articles_are_not(self, tmp_path, text):
        assert _run_js(tmp_path, text, inputs=[{"type": "search"}], title="Sign in with a passkey") is None

    def test_a_short_help_page_with_article_structure_is_not(self, tmp_path):
        para = "With a passkey you can sign in with your fingerprint, face or screen lock instead of a password. " * 2
        assert _run_js(tmp_path, "Use your passkey to sign in", title="Sign in with a passkey - Help",
                       buttons=["Continue"], paragraphs=[para, para, para]) is None
        assert _run_js(tmp_path, "Use your passkey to sign in", title="Sign in with a passkey - Help",
                       article=True) is None

    def test_footer_paragraphs_do_not_make_a_challenge_page_an_article(self, tmp_path):
        foot = {"text": "Privacy Policy. Terms of Service. Help. This site is protected and the Privacy Policy "
                        "and Terms of Service apply.", "footer": True}
        got = _run_js(tmp_path, GOOGLE_PK_BODY, paragraphs=[foot, foot, foot], **GOOGLE_CTX)
        assert got and got["kind"] == "passkey"

    @pytest.mark.parametrize("body,ctx", [
        ("Verify your identity", {"title": "Security settings", "headings": ["Account security"]}),
        ("Confirm your identity", {"headings": ["Confirm your order"], "buttons": ["Continue"]}),
        ("Sign in with a passkey\nWith a passkey you can sign in with your fingerprint, face or screen lock. Use your "
         "passkey to sign in to your account on any of your devices.\nContinue",
         {"headings": ["Sign in with a passkey"], "buttons": ["Continue"]}),
    ])
    def test_the_final_round_s_three_false_positives_stay_quiet(self, tmp_path, body, ctx):
        assert _run_js(tmp_path, body, **ctx) is None

    def test_the_words_alone_without_a_challenge_context_are_not(self, tmp_path):
        assert _run_js(tmp_path, "Use your passkey to confirm it's you", title="My notes") is None

    def test_verify_your_identity_on_a_cart_page_is_not(self, tmp_path):
        assert _run_js(tmp_path, "Verify your identity\nYour cart", title="Your cart", buttons=["Continue"]) is None

    @pytest.mark.parametrize("body,ctx,kind", [
        ("Use your face, fingerprint, PIN or security key", {"title": "Sign in to your account"}, "passkey"),
        ("Authenticate with a passkey", {"headings": ["Two-factor authentication"]}, "passkey"),
        ("Utilisez votre clé d'accès", {"title": "Connexion"}, "passkey"),
        ("השתמש במפתח הגישה שלך", {"title": "התחברות"}, "passkey"),
        ("Verifica tu identidad", {"headings": ["Verifica tu identidad"]}, "identity_check"),
    ])
    def test_common_and_other_language_phrasings(self, tmp_path, body, ctx, kind):
        assert (_run_js(tmp_path, body, **ctx) or {}).get("kind") == kind

    @pytest.mark.parametrize("frames,expect", [
        ([("https://www.google.com/recaptcha/api2/anchor?k=x&size=invisible", 256, 60)], None),
        ([("https://www.google.com/recaptcha/api2/anchor?k=x&size=normal", 304, 78)], "captcha"),
        ([("https://challenges.cloudflare.com/cdn-cgi/challenge-platform/x", 300, 65)], "captcha"),
    ])
    def test_only_a_visible_challenge_is_a_captcha(self, tmp_path, frames, expect):
        got = _run_js(tmp_path, "Log in", frames=frames, inputs=[{"type": "text"}, {"type": "password"}])
        assert (got or {}).get("kind") == expect


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

    OK = {"active": False, "locked": False, "asleep": False, "display_asleep": False}

    @pytest.mark.parametrize("stamp,started_ago,here", [
        ({**OK, "idle": 120}, 90, True),                        # asked 90 s ago, touched 2 min ago: waiting
        ({**OK, "idle": 150}, 90, True),                        # the minute of slack
        ({**OK, "idle": 151}, 90, False),                       # not touched around when they asked
        ({**OK, "idle": 200}, 179, True),
        ({**OK, "idle": 200}, 181, False),                      # 180 s is the whole grace
        ({**OK, "idle": 5}, None, False),                       # start unknown: no grace (fail closed)
        ({**OK, "locked": True, "idle": 5}, 30, False),         # locked: never
        ({**OK, "asleep": True, "idle": 5}, 30, False),         # asleep: never
        ({**OK, "display_asleep": True, "idle": 5}, 30, False),  # the display asleep: away
        ({"active": False, "locked": False, "asleep": False, "idle": 5}, 30, False),  # does not say display: absent
        ({"active": True, "display_asleep": True, "idle": 5}, 30, False),             # active but display off: away
        ({"active": True, "idle": 5}, None, True),              # active is here, whatever the turn
    ])
    def test_the_reviewer_s_rule_for_a_person_waiting_at_the_mac(self, _live_turn_at_the_mac, stamp, started_ago, here):
        _live_turn_at_the_mac.write_text(json.dumps({**stamp, "at": time.time() - 5}))
        presence = {"live": True, "why": "", "surface": "local"}
        if started_ago is not None:
            presence["turn_started_at"] = time.time() - started_ago
        assert lane.at_this_mac(presence)[0] is here
        assert (lane.LOCAL_TURN_GRACE_SECONDS, lane.LOCAL_TURN_TOUCH_SLACK_SECONDS) == (180, 60)

    def test_the_grace_needs_a_fresh_stamp(self, _live_turn_at_the_mac):
        _live_turn_at_the_mac.write_text(json.dumps({**self.OK, "idle": 1, "at": time.time() - 600}))
        assert lane.at_this_mac({"live": True, "surface": "local", "turn_started_at": time.time()})[0] is False

    def test_the_native_runtime_stamps_its_turn_start(self, monkeypatch):
        from agent import conversation_loop as cl
        from gateway.session_context import get_turn_started_at
        monkeypatch.setattr(cl, "_run_conversation_turn", lambda *a, **k: {"seen": get_turn_started_at()})
        monkeypatch.setattr("agent.turn_context.export_current_turn_boundary", lambda agent, result, msg: result)
        before = time.time()
        out = cl.run_conversation(object(), "hi")
        assert before <= out["seen"] <= time.time()
        assert before <= lane.local_turn_presence()["turn_started_at"] <= time.time()

    def test_a_local_turn_with_no_stamped_start_carries_none(self):
        import contextvars
        ctx = contextvars.Context()   # a turn that never went through run_conversation
        got = ctx.run(lambda: (_bind(), lane.local_turn_presence())[1])
        assert got["live"] is True and "turn_started_at" not in got

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

    def test_a_captcha_the_ladder_already_handled_is_not_said_twice(self, cli, monkeypatch):
        seen = _probe_returns(monkeypatch, cli, {"kind": "captcha", "page_says": "", "why": "challenge_frame",
                                                 "url": "https://shop.example/", "title": "x"})

        def ladder(result, *a, **k):
            result["captcha"] = {"outcome": "needs_person", "next": "Call browser_handoff(...)"}
            return "needs_person"
        monkeypatch.setattr(bu, "_captcha_ladder", ladder)
        cli["stdout"] = "Verify you are human"
        out = json.loads(bu.browser_exec('print(js("document.body.innerText"))', task_id="t"))
        assert out["captcha"]["outcome"] == "needs_person" and "needs_person" not in out and seen == []

    def test_after_a_captcha_the_ladder_passed_a_passkey_still_gets_our_note(self, cli, monkeypatch):
        _probe_returns(monkeypatch, cli, PASSKEY_PROBE)

        def ladder(result, *a, **k):
            result["captcha"] = {"outcome": "passed", "next": "carry on"}
            return "passed"
        monkeypatch.setattr(bu, "_captcha_ladder", ladder)
        cli["stdout"] = GOOGLE_PASSKEY_STDOUT
        out = json.loads(bu.browser_exec('print(js("document.body.innerText"))', task_id="t"))
        assert out["captcha"]["outcome"] == "passed" and out["needs_person"]["kind"] == "passkey"

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

    def test_a_window_macos_did_not_bring_forward_is_said_honestly(self, handoff):
        bh, state = handoff
        state["shown"] = {"ok": True, "url": "https://accounts.google.com/pk", "title": "", "relaunched": False,
                          "front": False, "form_state_lost": False, "reopened": 0}
        out = json.loads(bh.browser_handoff(reason="confirm", task_id="t"))
        assert out["front"] is False
        assert "click Moe's browser in the Dock" in out["next"] and "Never say it is in front" in out["next"]
        assert "in front of them (macOS confirmed" not in out["next"]

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

    def test_asked_from_the_mac_but_gone_goes_to_their_channels(self, handoff, _live_turn_at_the_mac):
        bh, state = handoff
        _stamp(_live_turn_at_the_mac, active=False)
        state["pages"] = [{"id": "1", "url": "https://accounts.google.com/pk", "title": "", "ws": ""}]
        out = json.loads(bh.browser_handoff(reason="confirm the passkey", task_id="t"))
        assert out["code"] == "person_needed" and "reach_owner(text)" in out["error"]
        assert "waiting in Moe's browser on your Mac" in out["tell_owner"] and state.get("show_calls") is None

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

    def test_other_processes_claims_are_their_own_files(self, tmp_path, monkeypatch):
        from tools import browser_tool_real_profile as rp
        d = tmp_path / "claims"
        monkeypatch.setattr(rp, "_claims_dir", lambda: str(d))
        window = rp._origin().BROWSER_SESSION_INACTIVITY_TIMEOUT
        rp.claim_driven_browser()
        assert rp.other_live_claims(window) == []                     # our own claim is never "other work"
        other = subprocess.Popen(["/bin/sleep", "30"])
        try:
            (d / str(other.pid)).write_text("")
            rp.claim_driven_browser()                                  # our re-claim touches only our file
            assert rp.other_live_claims(window) == [other.pid]
            assert any("other part" in r for r in rp._other_work("t"))
            old = time.time() - window - 5                             # outside the window: not live work
            os.utime(d / str(other.pid), (old, old))
            assert rp.other_live_claims(window) == []
            now = time.time()
            os.utime(d / str(other.pid), (now - window + 5, now - window + 5))
            assert rp.other_live_claims(window) == [other.pid]         # just inside it: live work
        finally:
            other.kill()
            other.wait()
        assert rp.other_live_claims(window) == [] and not (d / str(other.pid)).exists()   # dead: pruned

    def test_every_acquire_claims(self, tmp_path, monkeypatch):
        from tools import browser_tool_real_profile as rp
        src = open(rp.__file__).read()
        acquire = src[src.index("def _real_profile_cdp() -> tuple:"):]
        assert acquire.index("claim_driven_browser()") < acquire.index('cached = _bt._real_profile_cdp_cache.get("cdp")')

    def test_other_work_is_read_from_this_process_s_lanes(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tools.browser_tool_real_profile._claims_dir", lambda: str(tmp_path / "none"))
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

    def test_a_window_nobody_hands_back_is_released_after_the_two_hour_ceiling(self, rp, monkeypatch):
        rp, _ = rp
        assert rp.SHOWN_MAX_SECONDS == 2 * 3600
        monkeypatch.setattr(rp, "_real_profile_in_use", lambda: False)
        rp.show_to_person()
        since = rp._shown_to_person["since"]
        assert rp.release_if_idle(now=since + rp.SHOWN_MAX_SECONDS - 5) is False
        assert rp.release_if_idle(now=since + rp.SHOWN_MAX_SECONDS + 5) is True

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
                            subprocess.CompletedProcess(argv, 0, "front\n", ""))
        assert rp._activate_pid(4242) is True
        assert ran[0][:3] == ["/usr/bin/osascript", "-l", "JavaScript"] and "(4242)" in ran[0][-1]
        assert "frontmostApplication" in ran[0][-1] and "== 4242" in ran[0][-1] and "delay(0.3)" in ran[0][-1]
        assert not any("/usr/bin/open" in a for argv in ran for a in argv)

    @pytest.mark.parametrize("said", ["not_front", "no_app", "true", ""])
    def test_only_a_confirmed_front_is_front(self, monkeypatch, said):
        from tools import browser_tool_real_profile as rp
        monkeypatch.setattr(rp.sys, "platform", "darwin")
        monkeypatch.setattr(rp.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, said + "\n", ""))
        assert rp._activate_pid(4242) is False

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


def test_the_bridge_says_when_the_live_turn_began(monkeypatch):
    """at_this_mac's grace needs the turn's start: the agent answers it with the presence."""
    import contextvars
    import types
    from agent import claude_code_runtime as rt
    from agent.transports.hermes_tool_bridge import TURN_PRESENCE_QUERY
    agent = types.SimpleNamespace()
    agent._turn_context = contextvars.copy_context()
    dispatch = rt.make_tool_bridge_dispatch(agent)
    seen = {}

    def body(agent, **_kw):
        _kw["_on_locked"]()
        seen["during"] = json.loads(dispatch(TURN_PRESENCE_QUERY, {}))
        return {}

    monkeypatch.setattr(rt, "_run_claude_code_turn_body", body)
    before = time.time()
    rt.run_claude_code_turn(agent, user_message="hi", original_user_message="hi", messages=[], effective_task_id="t")
    assert seen["during"]["live"] is True and before <= seen["during"]["turn_started_at"] <= time.time()
    assert seen["during"]["surface"] == "local"


def test_asked_from_the_mac_but_gone_reaches_their_channels_from_exec_and_the_ladder(cli, monkeypatch,
                                                                                     _live_turn_at_the_mac):
    """The person asked from the Mac, then walked away: the reply lands on an island nobody watches, so
    the words go to reach_owner (their phone and chats), from browser_exec's note and the ladder alike."""
    from tools import browser_captcha_ladder as bl
    _stamp(_live_turn_at_the_mac, active=False)
    _probe_returns(monkeypatch, cli, PASSKEY_PROBE)
    cli["stdout"] = GOOGLE_PASSKEY_STDOUT
    out = json.loads(bu.browser_exec('print(1)', task_id="t"))
    assert out["needs_person"]["code"] == "person_needed" and "reach_owner(text)" in out["hint"]
    assert "browser_handoff(" not in out["hint"]
    o = bl.Outcome(kind="recaptcha_v2", host="forms.example", outcome=bl.NEEDS_PERSON, reason="an image puzzle", tier="C")
    step = bl.next_step(o, {"live": True, "why": "", "surface": "local"})
    assert "reach_owner" in step and "waiting in Moe's browser on their Mac" in step and "browser_handoff" not in step
