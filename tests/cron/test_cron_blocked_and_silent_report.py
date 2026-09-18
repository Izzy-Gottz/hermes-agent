"""A cron job that could not do its work is never "ok", and its report is never swallowed.

Moe ticket #5, 2026-09-18: a daily X follow-and-comment job found the browser
signed out, wrote four paragraphs saying so — and ended the response with
``[SILENT]``. The last-line silence rule suppressed the delivery, the run was
booked ``completed``, ``last_status`` read ``ok``, and the person learned the
next day from X itself that nothing had happened.

Two rules fix it, each tested from ``run_one_job`` down to the platform sender
(the same seam ``test_cron_failure_deliver.py`` uses):

* a silence marker beside a *report* is a report: delivered, marker stripped;
* a response opening with ``[BLOCKED]`` is booked ``blocked`` (a failure,
  streak counted, status never ``ok``) and the agent's own words are delivered.
"""

import pytest

import cron.scheduler as s
from gateway.response_filters import (
    autonomous_silence_report,
    blocked_reason,
    is_autonomous_silence_response,
    strip_blocked_marker,
)

# ~/.hermes/cron/output/711b145c4ed2/2026-09-18_09-44-08.md, the "## Response" section verbatim.
TICKET_5_RESPONSE = (
    "**Login not active.** Your X/Twitter session is not logged in on the browser. "
    "The Moe growth pass requires accessing your real account to follow users and comment. \n"
    "\n"
    "To proceed with the daily growth automation, your X login needs to be active in the "
    "browser (local session). I'm stopping here without guessing at credentials, as per your "
    "instructions.\n"
    "\n"
    "[SILENT]\n"
)


# --- the matcher -------------------------------------------------------------------------------

def test_marker_beside_a_report_is_not_silence():
    assert not is_autonomous_silence_response(TICKET_5_RESPONSE)
    report = autonomous_silence_report(TICKET_5_RESPONSE)
    assert report is not None
    assert "Login not active" in report
    assert "[SILENT]" not in report


def test_marker_with_a_short_note_is_still_silence():
    """Upstream's tolerated forms keep working — a one-line note rides along with the marker."""
    for text in (
        "[SILENT]",
        "[SILENT]\n\nNothing new this tick.",
        "2 deals filtered\n\n[SILENT]",
        "[SILENT] No changes detected",
    ):
        assert is_autonomous_silence_response(text), text
        assert autonomous_silence_report(text) is None, text


def test_no_marker_means_no_report_to_strip():
    assert autonomous_silence_report("Found 3 backpacks under $40.") is None
    assert autonomous_silence_report("") is None


def test_blocked_reason_is_the_first_line_or_the_next():
    assert blocked_reason("[BLOCKED] not signed in to X\nI opened x.com and got the login page.") == "not signed in to X"
    assert blocked_reason("[BLOCKED]\nnot signed in to X") == "not signed in to X"
    assert blocked_reason("[blocked]: 2FA needed") == "2FA needed"
    assert blocked_reason("[BLOCKED]") == "blocked"
    assert blocked_reason("Done. Nothing was [BLOCKED] today.") is None
    assert blocked_reason("") is None
    assert strip_blocked_marker("[BLOCKED] not signed in to X\nDetails.") == "not signed in to X\nDetails."


# --- the scheduler, end to end ----------------------------------------------------------------

@pytest.fixture
def run_env(monkeypatch, tmp_path):
    """Drive run_one_job with the REAL delivery path down to a fake sender; bookkeeping recorded."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "platforms:\n  slack:\n    enabled: true\n    token: xoxb-test\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    send_calls = []

    async def fake_sender(pconfig, chat_id, message, *, thread_id=None,
                          media_files=None, force_document=False, caption=None):
        send_calls.append({"chat_id": chat_id, "message": message})
        return {"success": True, "chat_id": chat_id, "message_id": "1.2"}

    import gateway.platform_registry as reg
    import hermes_cli.plugins as hp

    entry = reg.platform_registry.get("slack")
    if entry is None:
        hp.discover_plugins()
        entry = reg.platform_registry.get("slack")
    if entry is None:
        pytest.skip("slack platform entry not registered")
    monkeypatch.setattr(entry, "standalone_sender_fn", fake_sender)
    monkeypatch.setattr(hp, "discover_plugins", lambda *a, **k: None)

    state = {"send": send_calls, "marked": [], "finished": []}
    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": "exec-t"})
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: {})
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **kw: state["marked"].append((a, kw)) or True)
    monkeypatch.setattr(s, "finish_execution", lambda *a, **kw: state["finished"].append((a, kw)))
    monkeypatch.setattr(s, "_upsert_incident_for_failure", lambda *_a, **_kw: (False, None))
    monkeypatch.setattr(s, "load_config", lambda: {})
    return state


def _agent_says(final):
    def _fake(job, **_kw):
        return (True, "raw output", final, None)
    return _fake


JOB = {"id": "711b", "name": "X daily growth", "deliver": "slack:D0MAIN"}


def test_ticket_5_response_is_delivered_without_the_marker(run_env, monkeypatch):
    monkeypatch.setattr(s, "run_job", _agent_says(TICKET_5_RESPONSE))

    s.run_one_job(dict(JOB))

    assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]
    msg = run_env["send"][0]["message"]
    assert "Login not active" in msg
    assert "[SILENT]" not in msg


def test_bare_silent_still_delivers_nothing(run_env, monkeypatch):
    monkeypatch.setattr(s, "run_job", _agent_says("[SILENT]"))

    s.run_one_job(dict(JOB))

    assert run_env["send"] == []
    (args, kw), = run_env["marked"]
    assert args[1] is True and kw.get("status") is None


def test_blocked_is_booked_blocked_and_delivered_in_the_agents_words(run_env, monkeypatch):
    monkeypatch.setattr(
        s, "run_job",
        _agent_says("[BLOCKED] not signed in to X\nI opened x.com/notifications and got the login page."))

    s.run_one_job(dict(JOB))

    # Delivered, every run, in the agent's words, marker gone.
    assert [c["chat_id"] for c in run_env["send"]] == ["D0MAIN"]
    msg = run_env["send"][0]["message"]
    assert "not signed in to X\nI opened x.com/notifications" in msg
    assert "[BLOCKED]" not in msg
    # Booked as a failure with status "blocked" — never ok, streak counted by mark_job_run.
    (args, kw), = run_env["marked"]
    assert args[0] == "711b" and args[1] is False
    assert args[2] == "blocked: not signed in to X"
    assert kw.get("status") == "blocked"
    # The execution row says so too.
    (fargs, fkw), = run_env["finished"]
    assert fkw.get("success") is False
    assert fkw.get("error") == "blocked: not signed in to X"


def test_blocked_survives_a_stapled_silent(run_env, monkeypatch):
    """The two mistakes together: the model says it is blocked AND appends [SILENT]."""
    monkeypatch.setattr(
        s, "run_job", _agent_says("[BLOCKED] 2FA verification required\nStopped.\n\n[SILENT]"))

    s.run_one_job(dict(JOB))

    assert len(run_env["send"]) == 1
    (args, kw), = run_env["marked"]
    assert args[1] is False and kw.get("status") == "blocked"


def test_a_real_report_is_untouched(run_env, monkeypatch):
    monkeypatch.setattr(s, "run_job", _agent_says("Followed 24 accounts, left 3 comments."))

    s.run_one_job(dict(JOB))

    assert "Followed 24 accounts, left 3 comments." in run_env["send"][0]["message"]
    (args, kw), = run_env["marked"]
    assert args[1] is True and kw.get("status") is None


def test_the_preamble_tells_the_model_about_blocked():
    from cron.scheduler_prompt import _CRON_HINT
    assert "[BLOCKED]" in _CRON_HINT
    assert "Never report success for work that did not happen" in _CRON_HINT
