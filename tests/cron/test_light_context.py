"""`light_context: true` on a cron job runs its agent without memory or
context files — SOUL.md and the skills index still load.

The scheduler hardcoded skip_memory=False and skip_context_files=(no
workdir) for every job. A small clockwork job — a check, a classifier, a
ping — paid for the person's whole MEMORY.md / USER.md every tick. This is
the opt-out, per job, and these prove the flag reaches AIAgent's kwargs
rather than merely being stored.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from cron.jobs import create_job, get_job, update_job


@pytest.fixture(autouse=True)
def _cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")


def _build(job, workdir=None):
    """Upstream moved the AIAgent construction into ``_construct_cron_agent``;
    it is the one place the flag can reach the agent, so it is pinned there."""
    from types import SimpleNamespace
    from cron.scheduler import _construct_cron_agent
    agent_cls = MagicMock()
    setup = SimpleNamespace(model="m", runtime={"api_key": "k", "provider": "openrouter"},
                            prefill_messages=None, max_iterations=5, reasoning_config=None,
                            fallback_model=None, credential_pool=None)
    _construct_cron_agent(agent_cls, job, {}, setup, workdir=workdir, session_id="s", session_db=None)
    return agent_cls


# ── the flag reaches the agent ────────────────────────────────────────────

def test_a_light_job_skips_memory_and_context_files(tmp_path):
    job = {"id": "light-job", "name": "t", "prompt": "ping", "light_context": True}
    agent_cls = _build(job, job.get("workdir"))
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["skip_memory"] is True
    assert kwargs["skip_context_files"] is True
    assert kwargs["load_soul_identity"] is True, "SOUL still loads"


def test_a_light_job_with_a_workdir_still_skips_context_files(tmp_path):
    workdir = tmp_path / "repo"
    workdir.mkdir()
    job = {"id": "light-wd", "name": "t", "prompt": "ping",
           "light_context": True, "workdir": str(workdir)}
    agent_cls = _build(job, job.get("workdir"))
    assert agent_cls.call_args.kwargs["skip_context_files"] is True


def test_an_ordinary_job_is_unchanged(tmp_path):
    """The default every existing job relies on: memory on, context files
    only with a workdir."""
    job = {"id": "plain", "name": "t", "prompt": "ping"}
    agent_cls = _build(job, job.get("workdir"))
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["skip_memory"] is False
    assert kwargs["skip_context_files"] is True  # no workdir

    workdir = tmp_path / "repo"
    workdir.mkdir()
    job = {"id": "plain-wd", "name": "t", "prompt": "ping", "workdir": str(workdir)}
    agent_cls = _build(job, job.get("workdir"))
    kwargs = agent_cls.call_args.kwargs
    assert kwargs["skip_memory"] is False
    assert kwargs["skip_context_files"] is False


def test_a_false_flag_is_the_default(tmp_path):
    job = {"id": "off", "name": "t", "prompt": "ping", "light_context": False}
    agent_cls = _build(job, job.get("workdir"))
    assert agent_cls.call_args.kwargs["skip_memory"] is False


# ── stored by create_job / update_job ─────────────────────────────────────

def test_create_job_stores_the_flag_only_when_true():
    on = create_job("ping", "1h", name="on", light_context=True)
    off = create_job("ping", "1h", name="off", light_context=False)
    absent = create_job("ping", "1h", name="absent")
    assert get_job(on["id"])["light_context"] is True
    assert "light_context" not in get_job(off["id"])
    assert "light_context" not in get_job(absent["id"])


def test_update_job_turns_the_flag_on_and_off():
    job = create_job("ping", "1h", name="j")
    update_job(job["id"], {"light_context": True})
    assert get_job(job["id"])["light_context"] is True
    update_job(job["id"], {"light_context": "yes"})
    assert get_job(job["id"])["light_context"] is True
    update_job(job["id"], {"light_context": False})
    assert "light_context" not in get_job(job["id"]), "off looks like never-set"
