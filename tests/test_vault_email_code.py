"""A site that emails a sign-in code: find the field, read the mail, enter the code (Moe ticket #13).

On 2026-09-24 a site emailed the owner a 6-digit code; Moe could read his Gmail and still stopped,
twice -- the code field (a bare <input id="code"> under "6-digit code (check your email)") was not a
code field to the classifier, and the code-entry tool had no source but an authenticator seed and
asking the person. These tests pin the engine behaviour for every site, not that one.
"""

import json
import re
from unittest.mock import patch

import pytest

from agent import vault_code_sources as sources
from agent.vault_code_sources import MailMessage, extract_code, find_code, site_label
from agent.vault_login_classifier import LoginControl, classify_otp_controls

NOW = 1_790_280_000.0


@pytest.fixture(autouse=True)
def _no_sources():
    saved = list(sources._sources)
    sources._sources.clear()
    yield
    sources._sources[:] = saved


def _ctl(name="", label="", context="", type_="text", maxlen=None, autocomplete=""):
    return LoginControl(autocomplete, 0, 3, label, name, type_, maxlen, context)


class TestTheField:
    def test_the_tickets_field_is_a_code_field(self):
        """<input type="text" id="code"> with the words beside it, no <label>, no autocomplete."""
        got = classify_otp_controls([_ctl(name=" code", context="6-digit code (check your email)")])
        assert len(got) == 1 and got[0].token == "one-time-code"

    @pytest.mark.parametrize("name,label,context", [
        ("", "", "Enter the code we sent to you"),
        ("pin", "", ""),
        ("login_code", "", ""),
        ("", "Code", ""),
        ("", "", "Enter your 6 digit code"),
    ])
    def test_plain_code_and_pin_fields_are_code_fields(self, name, label, context):
        assert classify_otp_controls([_ctl(name=name, label=label, context=context)])

    @pytest.mark.parametrize("name,label,context", [
        ("promo_code", "", ""),
        ("", "Coupon code", ""),
        ("zip", "ZIP code", ""),
        ("", "", "Postal code"),
        ("", "Gift card code", ""),
        ("referral", "Referral code", ""),
        ("", "Country code", "+1"),
        ("q", "Search", "Search the docs"),
    ])
    def test_codes_that_are_not_a_sign_in_are_not(self, name, label, context):
        assert classify_otp_controls([_ctl(name=name, label=label, context=context)]) == []

    def test_a_long_field_or_a_password_is_not_a_plain_code_field(self):
        assert classify_otp_controls([_ctl(name="code", maxlen=200)]) == []
        assert classify_otp_controls([_ctl(name="code", type_="password")]) == []

    def test_the_stronger_readings_still_win(self):
        got = classify_otp_controls([_ctl(name="otp"), _ctl(autocomplete="one-time-code")])
        assert sorted(c.score for c in got) == [70, 100]

    def test_the_page_script_reports_nearby_text(self):
        from agent.vault_login_classifier import build_inspection_js
        js = build_inspection_js("n")
        assert "context" in js and "innerText" in js


class TestTheMessage:
    @pytest.mark.parametrize("text,code", [
        ("Your TinyLaunch login code is 482913. It expires in 10 minutes.", "482913"),
        ("Use 123-456 to sign in to Acme", "123456"),
        ("Verification code: K7Q2PX", "K7Q2PX"),
        ("Hi Sruly, here is your one-time code\n\n 9041 \n\nThanks, 2026 Acme Inc.", "9041"),
        ("Order #88231 shipped. Your sign-in code is 552019", "552019"),
    ])
    def test_the_code_is_found(self, text, code):
        assert extract_code(text) == code

    @pytest.mark.parametrize("text", [
        "Welcome to TinyLaunch! Launch day is 2026. See you there.",
        "Your invoice for $4999 is ready",
        "Call us at +1 555 0100 anytime",
        "",
    ])
    def test_no_code_is_invented(self, text):
        assert extract_code(text) is None

    @pytest.mark.parametrize("origin,label", [
        ("https://www.tinylaunch.com/login", "tinylaunch"),
        ("https://app.acme.io", "acme"),
        ("https://accounts.example.co.uk", "example"),
    ])
    def test_the_site_name(self, origin, label):
        assert site_label(origin) == label


def _source(messages, calls=None):
    def fn(label, since):
        if calls is not None:
            calls.append((label, since))
        return messages() if callable(messages) else messages
    return fn


class TestFindCode:
    def test_the_fresh_message_from_the_site(self):
        sources.register_code_source("mail", _source([
            MailMessage("noreply@other.com", "Your code", "Your code is 111111", NOW - 30, "a@x.com"),
            MailMessage("hello@tinylaunch.com", "TinyLaunch login code", "Your code is 482913", NOW - 60, "a@x.com"),
        ]))
        code, where = find_code("https://tinylaunch.com/login", wait=0, now=lambda: NOW)
        assert (code, where) == ("482913", "a@x.com")

    def test_the_site_named_in_the_text_counts_when_the_sender_is_a_mail_service(self):
        sources.register_code_source("mail", _source([
            MailMessage("bounce@sendgrid.net", "Sign in", "Sign in to TinyLaunch with code 700155", NOW - 5)]))
        assert find_code("https://tinylaunch.com", wait=0, now=lambda: NOW)[0] == "700155"

    def test_a_stale_message_is_not_used(self):
        sources.register_code_source("mail", _source([
            MailMessage("hello@tinylaunch.com", "code", "Your code is 482913", NOW - 3600)]))
        code, why = find_code("https://tinylaunch.com", wait=0, now=lambda: NOW)
        assert code is None and "tinylaunch" in why

    def test_another_sites_code_is_never_used(self):
        sources.register_code_source("mail", _source([
            MailMessage("security@bank.com", "Your code", "Your bank code is 123456", NOW - 10)]))
        assert find_code("https://tinylaunch.com", wait=0, now=lambda: NOW)[0] is None

    def test_the_newest_of_two_wins(self):
        sources.register_code_source("mail", _source([
            MailMessage("a@tinylaunch.com", "code", "Your code is 111111", NOW - 200),
            MailMessage("a@tinylaunch.com", "code", "Your code is 222222", NOW - 20)]))
        assert find_code("https://tinylaunch.com", wait=0, now=lambda: NOW)[0] == "222222"

    def test_the_sites_own_mail_beats_a_newer_one_that_only_mentions_it(self):
        """Measured on the owner's inbox: Google's security notice naming the site, with a
        number in it, beside the site's own code mail."""
        sources.register_code_source("mail", _source([
            MailMessage("TinyLaunch <hello@tinylaunch.com>", "Your login code", "Your code is 482913", NOW - 120),
            MailMessage("Google <no-reply@accounts.google.com>", "Security alert",
                        "New sign-in to TinyLaunch. Verify with code 7731", NOW - 10)]))
        assert find_code("https://tinylaunch.com", wait=0, now=lambda: NOW)[0] == "482913"

    def test_it_waits_for_the_email_to_land(self):
        clock = [NOW]
        inbox = []
        sources.register_code_source("mail", _source(lambda: list(inbox)))

        def sleep(s):
            clock[0] += s
            if clock[0] >= NOW + 15:
                inbox.append(MailMessage("a@tinylaunch.com", "code", "Your code is 333444", clock[0]))

        code, _ = find_code("https://tinylaunch.com", wait=60, poll=5, now=lambda: clock[0], sleep=sleep)
        assert code == "333444" and NOW + 15 <= clock[0] < NOW + 60

    def test_a_broken_source_does_not_stop_another(self):
        def broken(label, since):
            raise RuntimeError("offline")
        sources.register_code_source("broken", broken)
        sources.register_code_source("mail", _source([MailMessage("a@tinylaunch.com", "c", "code 909090", NOW)]))
        assert find_code("https://tinylaunch.com", wait=0, now=lambda: NOW)[0] == "909090"

    def test_no_source_says_so(self):
        assert find_code("https://tinylaunch.com", wait=0) == (None, "no mailbox is connected")


def _enter(controls, presence_live=True, prompt=None):
    """Run browser_vault_enter_code against a fake page; returns (result, filled expr, asked)."""
    from agent.vault_backends import unlock as unlock_mod
    from tools import browser_vault_tool

    asked, seen = [], {}
    unlock_mod.set_code_prompt_callback(lambda site, hint: asked.append(site) or (prompt or ""))

    def fake_eval(t, e):
        return {"success": True, "result": json.dumps(controls) if "querySelectorAll" in e else "https://tinylaunch.com/login"}

    def fake_secret(t, e):
        seen["expr"] = e
        return {"success": True, "result": json.dumps({"filled": 1})}

    try:
        with patch("agent.vault_backends.unlock.can_prompt_here", return_value=True), \
             patch("tools.browser_chrome_extension.unattended_turn",
                   return_value=None if presence_live else "a scheduled job"), \
             patch.object(sources, "WAIT_SECONDS", 0), \
             patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_secret):
            raw = browser_vault_tool.browser_vault_enter_code(task_id="t")
    finally:
        unlock_mod.set_code_prompt_callback(None)
    return raw, seen.get("expr", ""), asked


TICKET_PAGE = [{"index": 0, "type": "text", "name": " code", "label": "", "autocomplete": "",
                "context": "6-digit code (check your email)"}]


class TestEnterCode:
    def test_the_ticket_end_to_end_the_code_comes_from_the_mail(self):
        sources.register_code_source("gmail", _source(lambda: [
            MailMessage("hello@tinylaunch.com", "Your TinyLaunch code", "Your login code is 482913",
                        __import__("time").time() - 30, "srulynj@gmail.com")]))
        raw, expr, asked = _enter(TICKET_PAGE)
        out = json.loads(raw)
        assert out["success"] and out["source"] == "email (srulynj@gmail.com)"
        assert asked == [], "nobody was asked"
        assert re.search(r'"value": "482913"', expr), "the code went into the page"
        assert "482913" not in raw, "and never to the model"

    def test_an_unattended_turn_does_not_read_the_mail(self):
        calls = []
        sources.register_code_source("gmail", _source([], calls))
        raw, _, asked = _enter(TICKET_PAGE, presence_live=False, prompt="111222")
        assert calls == [], "no mailbox was read on a turn nobody started"
        assert json.loads(raw)["source"] == "user" and asked

    def test_no_mail_falls_back_to_asking(self):
        sources.register_code_source("gmail", _source([]))
        raw, _, asked = _enter(TICKET_PAGE, prompt="424242")
        out = json.loads(raw)
        assert out["success"] and out["source"] == "user" and asked
