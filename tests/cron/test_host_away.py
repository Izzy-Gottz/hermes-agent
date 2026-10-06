"""While the host app says it is away (`cron/AWAY`), this host's ticker fires nothing; gateway turns
are not this switch's business (Memoe, 2026-10-06: darkwake runs raced the cloud's claims)."""
from unittest.mock import patch

from cron import scheduler


def test_away_sentinel_skips_dispatch_and_its_removal_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    (tmp_path / "cron").mkdir()
    reaped = []
    monkeypatch.setattr(scheduler, "_maybe_reap_dead_owners", lambda: reaped.append(1))

    (tmp_path / "cron" / "AWAY").write_text("")
    assert scheduler.tick(verbose=False) == 0
    assert reaped == []  # returned before any dispatch work

    (tmp_path / "cron" / "AWAY").unlink()
    with patch.object(scheduler, "get_due_jobs", return_value=[]):
        scheduler.tick(verbose=False)
    assert reaped == [1]  # the tick went on past the gate


def test_no_cron_dir_is_not_away(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: tmp_path)
    assert scheduler._host_away() is False
