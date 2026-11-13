"""The ``pre_cron_run`` plugin hook: a host may hear "not yours" before a fire.

Two hosts can hold one job (a laptop and an always-on machine). A plugin that
asks its own arbiter answers ``{"action": "skip", "reason": ...}``; the fire
then ends silent before any gate, script or agent runs. Everything else —
no plugin, a crash, another shape — runs the job: a claim that cannot be
answered must never lose a reminder.
"""

from __future__ import annotations

import pytest

import cron.scheduler as sched
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager
from hermes_cli.plugins_dispatch import _HOOK_TIMEOUT_BOUNDED_HOOKS


JOB = {"id": "a1b2c3d4e5f6", "name": "Morning digest", "no_agent": True, "script": "digest.py"}


@pytest.fixture
def ran(monkeypatch):
    calls = []
    monkeypatch.setattr(sched, "_run_no_agent_job",
                        lambda job, job_id, job_name, cancel_event: calls.append(job_id) or
                        (True, "doc", "said", None))
    skipped = []
    monkeypatch.setattr(sched, "note_cron_skipped", lambda job: skipped.append(job.get("id")))
    return calls, skipped


@pytest.fixture
def hook(monkeypatch):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    seen = []

    def install(answer):
        def callback(**kw):
            seen.append(kw)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        mgr._hooks.setdefault("pre_cron_run", []).append(callback)
    try:
        yield install, seen
    finally:
        mgr._hooks = saved


def test_a_skip_ends_the_fire_silent_before_anything_runs(ran, hook):
    calls, skipped = ran
    install, seen = hook
    install({"action": "skip", "reason": "the cloud computer has this run"})
    early, prompt = sched._prepare_job_prompt(dict(JOB), JOB["id"], JOB["name"], None, None)
    assert prompt is None and calls == []
    ok, doc, final, error = early
    assert ok is True and error is None and final == sched.SILENT_MARKER
    assert "Not run here: the cloud computer has this run" in doc
    assert skipped == [JOB["id"]]
    assert seen[0]["job_id"] == JOB["id"] and seen[0]["job"]["name"] == "Morning digest"


@pytest.mark.parametrize("answer", [None, {"action": "run"}, "skip", RuntimeError("edge down")])
def test_anything_but_a_skip_runs_the_job(ran, hook, answer):
    calls, skipped = ran
    install, _ = hook
    install(answer)
    early, _ = sched._prepare_job_prompt(dict(JOB), JOB["id"], JOB["name"], None, None)
    assert calls == [JOB["id"]] and skipped == []
    assert early == (True, "doc", "said", None)


def test_with_no_plugin_nothing_changes(ran):
    calls, _ = ran
    sched._prepare_job_prompt(dict(JOB), JOB["id"], JOB["name"], None, None)
    assert calls == [JOB["id"]]


def test_the_hook_is_declared_and_bounded():
    assert "pre_cron_run" in VALID_HOOKS
    assert "pre_cron_run" in _HOOK_TIMEOUT_BOUNDED_HOOKS
