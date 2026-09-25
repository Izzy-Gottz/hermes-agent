"""The code the person says, the field it goes in, and what no_code_field tells the model (slice D).

Measured 2026-09-24/25 (Moe transcript 42bec772, read-only): the owner read a TinyLaunch sign-in code
aloud and Moe refused to type it (the tool said never, "even when the user shows you one"); and when
the page had no code field (a Google passkey), the tool told the model to "tell the user to complete
it on their device" -- which it turned into "approve it on your phone".
"""

import json
import re
from unittest.mock import patch

import pytest

from agent import vault_code_sources as sources
from agent.vault_login_classifier import LoginControl, classify_otp_controls

TICKET_PAGE = [{"index": 0, "type": "text", "name": " code", "label": "", "autocomplete": "",
                "context": "6-digit code (check your email)"}]
PASSKEY_PROBE = {"kind": "passkey", "page_says": "Use your passkey to confirm it’s really you",
                 "url": "https://accounts.google.com/v3/signin/challenge/pk/presend", "title": "Sign in"}


@pytest.fixture(autouse=True)
def _no_sources():
    saved = list(sources._sources)
    sources._sources.clear()
    yield
    sources._sources[:] = saved


def _enter(controls, *, live=True, code="", source="", prompt=None, probe=None, can_prompt=True):
    from agent.vault_backends import unlock as unlock_mod
    from tools import browser_vault_tool

    asked, seen = [], {}
    unlock_mod.set_code_prompt_callback(lambda site, hint: asked.append(site) or (prompt or ""))

    def fake_eval(t, e):
        if "querySelectorAll(\"input, select\")" in e:
            return {"success": True, "result": json.dumps(controls)}
        if "innerText" in e and "page_says" in e:
            return {"success": True, "result": json.dumps(probe) if probe else None}
        return {"success": True, "result": "https://tinylaunch.com/login"}

    def fake_secret(t, e):
        seen["expr"] = e
        return {"success": True, "result": json.dumps({"filled": 1})}

    try:
        with patch("agent.vault_backends.unlock.can_prompt_here", return_value=can_prompt), \
             patch("tools.browser_chrome_extension.unattended_turn",
                   return_value=None if live else "a scheduled job"), \
             patch.object(sources, "WAIT_SECONDS", 0), \
             patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None), \
             patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval), \
             patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_secret):
            raw = browser_vault_tool.browser_vault_enter_code(task_id="t", code=code, source=source)
    finally:
        unlock_mod.set_code_prompt_callback(None)
    return raw, seen.get("expr", ""), asked


class TestTheCodeThePersonSays:
    def test_a_code_the_person_read_out_is_typed_into_the_code_field(self):
        raw, expr, asked = _enter(TICKET_PAGE, code="482 913", source="person")
        out = json.loads(raw)
        assert out["success"] and out["source"] == "person" and asked == []
        assert re.search(r'"value": "482913"', expr)

    def test_without_source_person_nothing_is_typed(self):
        raw, expr, _ = _enter(TICKET_PAGE, code="482913")
        assert json.loads(raw)["error_type"] == "code_source" and expr == ""

    def test_on_a_turn_nobody_is_at_there_is_no_person_to_have_said_it(self):
        raw, expr, _ = _enter(TICKET_PAGE, code="482913", source="person", live=False)
        assert json.loads(raw)["error_type"] == "code_source" and expr == ""

    @pytest.mark.parametrize("bad", ["hunter2 is my password", "12", "x" * 40, "<script>"])
    def test_only_something_shaped_like_a_code(self, bad):
        raw, expr, _ = _enter(TICKET_PAGE, code=bad, source="person")
        assert json.loads(raw)["error_type"] == "code_format" and expr == ""

    def test_no_code_field_no_code_is_typed(self):
        raw, expr, _ = _enter([], code="482913", source="person")
        assert json.loads(raw)["error_type"] == "no_code_field" and expr == ""


class TestNoCodeField:
    def test_a_passkey_page_is_said_from_the_page_and_handed_over(self):
        raw, _, _ = _enter([], probe=PASSKEY_PROBE)
        out = json.loads(raw)
        assert out["error_type"] == "no_code_field" and out["needs_person"]["kind"] == "passkey"
        assert out["needs_person"]["page_says"] == "Use your passkey to confirm it’s really you"
        assert "browser_handoff" in out["error"]

    def test_no_device_is_invented(self):
        for probe in (None, PASSKEY_PROBE):
            err = json.loads(_enter([], probe=probe)[0])["error"]
            assert "on their device" not in err and "your phone" not in err
            assert "when a tool result shows the site said so" in err
        err = json.loads(_enter([])[0])["error"]
        assert "inbox" in err and "browser_handoff" in err


class TestUnattendedCode:
    def test_a_scheduled_job_gets_a_person_needed_result_and_the_mail_stays_unread(self):
        from tools.fix_reasons import host_fields
        calls = []
        sources.register_code_source("gmail", lambda label, since: calls.append(label) or [])
        raw, expr, _ = _enter(TICKET_PAGE, live=False, can_prompt=False)
        out = json.loads(raw)
        assert calls == [] and expr == ""
        assert out["code"] == "person_needed" and out["retry"] is False and out["subject"] == "tinylaunch.com"
        assert host_fields(raw)["code"] == "person_needed"


def _ctl(name="", label="", context="", type_="text", maxlen=None, autocomplete=""):
    return LoginControl(autocomplete, 0, 3, label, name, type_, maxlen, context)


class TestTheFieldByItsName:
    @pytest.mark.parametrize("name", [
        "otpCode", "verificationCode", " code", "code ", "otp", "verification", "emailcode", "mfa-pin",
        "twoFactorCode", "one-time-code", "loginCode",
    ])
    def test_code_fields_named_the_ways_markup_names_them(self, name):
        got = classify_otp_controls([_ctl(name=name)])
        assert got and got[0].token == "one-time-code", name

    @pytest.mark.parametrize("name", ["promoCode", "zipCode", "countryCode", "couponCode", "giftCode", "username"])
    def test_other_codes_stay_other(self, name):
        assert classify_otp_controls([_ctl(name=name)]) == []
