"""The fixable-cause vocabulary and the dict contract the host app parses (docs/FIXABLE-FAILURES.md)."""

import json

import pytest

from tools import fix_reasons as fr

# The plan's table, verbatim. Adding a code means adding it here AND teaching the app about it.
PLAN_CODES = {
    "tcc_app_data", "tcc_files", "tcc_automation", "tcc_driver_accessibility", "tcc_driver_screen",
    "tcc_microphone", "tcc_calendar", "tcc_reminders", "tcc_contacts", "driver_not_running",
    "window_other_space", "mcp_needs_reauth", "mcp_not_connected", "claude_login_expired",
    "claude_usage_cap", "browser_missing", "profile_locked",
}


def test_vocabulary_is_the_plan_table():
    assert fr.CODES == PLAN_CODES
    assert isinstance(fr.CODES, frozenset)


def test_every_tcc_code_names_its_grant():
    tcc = {c for c in fr.CODES if c.startswith("tcc_")}
    assert set(fr.GRANTS) == tcc
    for code, (owner, pane) in fr.GRANTS.items():
        assert owner in fr.OWNERS and pane in fr.PANES
    assert fr.GRANTS["tcc_driver_accessibility"][0] == "driver"
    assert fr.GRANTS["tcc_driver_screen"][0] == "driver"


def test_fix_error_is_the_contract_shape():
    out = json.loads(fr.fix_error("Memoe isn't allowed to read Google Chrome's data", "tcc_app_data",
                                  owner="app", pane="Privacy_AllFiles", subject="Google Chrome"))
    assert out == {"error": "Memoe isn't allowed to read Google Chrome's data", "code": "tcc_app_data",
                   "owner": "app", "pane": "Privacy_AllFiles", "subject": "Google Chrome", "retry": True}


def test_every_contract_key_is_always_present():
    out = json.loads(fr.fix_error("sign in again", "mcp_needs_reauth", retry=False))
    assert set(out) == {"error", "code", "owner", "pane", "subject", "retry"}
    assert out["owner"] is None and out["pane"] is None and out["retry"] is False


@pytest.mark.parametrize("kw, match", [
    ({"code": "profile_lockd"}, "unknown fix code"),
    ({"code": "tcc_app_data", "owner": "driver"}, "owner"),
    ({"code": "mcp_needs_reauth", "owner": "someone"}, "unknown owner"),
    ({"code": "tcc_files", "pane": "Privacy_Everything"}, "unknown pane"),
    ({"code": "tcc_files", "error": "x"}, "contract keys"),
])
def test_unknown_values_raise(kw, match):
    code = kw.pop("code")
    with pytest.raises(ValueError, match=match):
        fr.fix_fields(code, **kw)


def test_fix_message_survives_string_plumbing():
    msg = fr.fix_message("why and what fixes it", "tcc_files", subject="Desktop")
    assert isinstance(msg, str) and msg == "why and what fixes it"
    assert fr.fields_of(msg)["pane"] == "Privacy_AllFiles"
    assert fr.fields_of("plain string") == {}
    exc = fr.as_error(msg)
    assert isinstance(exc, fr.FixableError) and isinstance(exc, RuntimeError)
    assert fr.fields_of(exc)["code"] == "tcc_files"
    assert type(fr.as_error("plain")) is RuntimeError
    assert json.loads(fr.as_tool_error(msg))["code"] == "tcc_files"
    assert json.loads(fr.as_tool_error("plain")) == {"error": "plain"}


def test_host_app_name(monkeypatch):
    monkeypatch.delenv("HERMES_HOST_APP_NAME", raising=False)
    assert fr.host_app_name() == "the app"
    monkeypatch.setenv("HERMES_HOST_APP_NAME", "Memoe")
    assert fr.host_app_name() == "Memoe"
