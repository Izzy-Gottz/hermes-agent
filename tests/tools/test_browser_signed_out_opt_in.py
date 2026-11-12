"""Ask, then continue (owner decision, 2026-11-12).

When the person's signed-in browser profile cannot load, browser_exec neither falls back to a
signed-out browser on its own (upstream since 80f4f989d3a) nor stops dead (the fork before this):
it returns the fixable reason and a question for the person, and only a retry carrying
``allow_signed_out=true`` -- the person's yes -- runs signed out, for that route only, saying so on
every result.

These drive the REAL ``_route_backend`` / ``_route_exec`` / ``browser_exec``; only the profile
loader, the backend below the decision, and the CLI process are replaced. ``_route_backend``
merges to upstream's silent fallback without a textual conflict, so each of the two behaviours
was checked by reverting it and watching the matching test go red.
"""
import json
import subprocess

import pytest

from tools import browser_use_cli as bu
from tools.fix_reasons import PROFILE_LOCKED, fix_message

REASON = "Chrome is holding its profile, so Moe could not copy your sign-ins."


@pytest.fixture()
def world(monkeypatch):
    """A Mac with real-profile consent on, no cloud provider, and a profile that will not load."""
    state = {"runs": [], "backends": [], "profile_err": fix_message(REASON, PROFILE_LOCKED), "profile_calls": 0}

    def real_profile_cdp():
        state["profile_calls"] += 1
        return None, state["profile_err"]

    def backend(env, task_id, session_name="", force_local_sidecar=False):
        state["backends"].append({"force_local_sidecar": force_local_sidecar, "session": session_name})
        env.setdefault("BU_CDP_URL", "http://127.0.0.1:9444")  # like the real one: a resolved endpoint wins
        return None

    def fake_run(cmd, code, env, timeout):
        state["runs"].append(dict(env))
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr(bu, "_real_profile_consented", lambda: True)
    monkeypatch.setattr("tools.browser_tool_cdp._get_cdp_override_raw", lambda: "")
    monkeypatch.setattr("tools.browser_tool_cloud._get_cloud_provider", lambda: None)
    monkeypatch.setattr("tools.browser_tool_real_profile._real_profile_cdp", real_profile_cdp)
    monkeypatch.setattr(bu, "_resolve_backend_cdp", backend)
    monkeypatch.setattr(bu, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", fake_run)
    monkeypatch.setattr(bu, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu, "_workspace_dir", lambda task_id: None)
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {})
    monkeypatch.setattr(bu, "_real_profile_fallback", {})
    monkeypatch.setattr(bu, "_real_profile_note_delivered", set())
    monkeypatch.setattr(bu, "_exec_route_is_sidecar", {})
    return state


def _exec(**kw):
    return json.loads(bu.browser_exec("print(1)", task_id="task-A", **kw))


def test_a_profile_that_cannot_load_never_reaches_a_signed_out_browser_without_the_opt_in(world):
    """THE GUARD. No browser, no CLI run, nothing remembered -- and the answer is the question."""
    out = _exec()

    assert world["backends"] == [], "a backend was resolved: the call fell back without the person's yes"
    assert world["runs"] == [], "the code ran in a browser nobody agreed to"
    assert bu._real_profile_fallback == {}
    assert "success" not in out or out["success"] is False
    # (a) the fixable reason, exactly as the fork reports it
    assert out["error"] == REASON and out["code"] == PROFILE_LOCKED
    # (b) it can continue signed out, if the person agrees, and how the model asks
    assert out["signed_in"] is False and out["signed_out_available"] is True
    assert out["ask_person"] == bu.SIGNED_OUT_QUESTION
    assert "still continue" in out["ask_person"]
    assert "allow_signed_out=true" in out["hint"] and "not yours" in out["hint"]
    assert "Only if THEY say yes" in out["hint"]


def test_asking_twice_without_a_yes_changes_nothing(world):
    _exec()
    out = _exec()
    assert out["signed_out_available"] is True and world["runs"] == [] and world["backends"] == []


def test_a_plain_reason_with_no_fix_code_still_asks(world):
    world["profile_err"] = "could not read the 'chrome' profile's login data"
    out = _exec()
    assert out["error"].startswith("could not read") and out["signed_out_available"] is True
    assert "code" not in out and world["runs"] == []


def test_the_persons_yes_runs_signed_out_and_every_result_says_so(world):
    """THE OPT-IN. The retry runs on the local sidecar (which never loads the profile); later calls of
    the same task stay there without the flag, never try the profile again mid-task, and EVERY result
    carries signed_out -- so nothing done there can be reported as done as the person."""
    assert _exec()["signed_out_available"] is True

    first = _exec(allow_signed_out=True)
    second = _exec()                       # same task, no flag: the yes is remembered for this route
    third = _exec()

    assert [r["success"] for r in (first, second, third)] == [True, True, True]
    assert len(world["runs"]) == 3
    assert [b["force_local_sidecar"] for b in world["backends"]] == [True, True, True]
    assert all(env["BU_NAME"].startswith("_local_") for env in world["runs"])
    for r in (first, second, third):
        assert r["signed_out"] is True
        assert "signed in as NOBODY" in r["signed_out_note"] and "never say it did" in r["signed_out_note"]
    assert REASON in first["real_profile_unavailable"]          # the reason, once
    assert "real_profile_unavailable" not in second
    assert world["profile_calls"] == 2, "the profile was tried again after the person said continue"


def test_the_yes_is_for_one_task_never_a_default(world):
    """Another task, or another named session, on the same Mac asks again."""
    assert _exec(allow_signed_out=True)["signed_out"] is True

    other_task = json.loads(bu.browser_exec("print(1)", task_id="task-B"))
    other_session = _exec(session="s2")

    assert other_task["signed_out_available"] is True and "signed_out" not in other_task
    assert other_session["signed_out_available"] is True and "signed_out" not in other_session
    assert len(world["runs"]) == 1


def test_a_profile_that_loads_is_never_marked_signed_out(world, monkeypatch):
    monkeypatch.setattr("tools.browser_tool_real_profile._real_profile_cdp",
                        lambda: ("http://127.0.0.1:9555", None))
    out = _exec(allow_signed_out=True)     # the flag alone signs nobody out
    assert out["success"] is True and "signed_out" not in out
    assert [b["force_local_sidecar"] for b in world["backends"]] == [False]
    assert world["runs"][0]["BU_CDP_URL"] == "http://127.0.0.1:9555"


def test_the_model_only_sees_the_argument_where_a_signed_in_profile_exists(monkeypatch):
    monkeypatch.setattr(bu, "_cloud_backend_configured", lambda: False)
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {})
    monkeypatch.setattr(bu, "_real_profile_consented", lambda: True)
    prop = bu._dynamic_schema_overrides()["parameters"]["properties"]["allow_signed_out"]
    assert prop["default"] is False and "Never set it on your own judgment" in prop["description"]
    monkeypatch.setattr(bu, "_real_profile_consented", lambda: False)
    assert "allow_signed_out" not in (bu._dynamic_schema_overrides().get("parameters") or {}).get("properties", {})


def test_only_a_literal_true_from_the_model_counts(world):
    """A model that writes "yes" or 1 has not passed the person's answer through the schema."""
    handler = bu.registry.get_entry("browser_exec").handler
    for sloppy in ("true", "yes", 1):
        out = json.loads(handler({"code": "print(1)", "allow_signed_out": sloppy}, task_id="task-A"))
        assert out["signed_out_available"] is True
    assert world["runs"] == []
    assert json.loads(handler({"code": "print(1)", "allow_signed_out": True}, task_id="task-A"))["signed_out"] is True
