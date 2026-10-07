"""Real-profile snapshot errors name the auth database they could not read and why (#111647).

Fork behaviour (Moe, ticket #13 / 47aa2b1191): a running Chrome holds ``Login Data`` /
``Login Data For Account`` / ``Web Data`` in EXCLUSIVE locking mode, and the snapshot copies them
anyway with a verified immutable read, so a locked login DB is never a failure. Only the cookie
jar reaches the driven browser, so only an unreadable jar fails the launch, closed, and its
message names the database and the reason -- never a bare "N database(s) unavailable" count, and
never "close your browser" (it is open because the person is using it).
"""
import json
import os
import sqlite3

import pytest

import hermes_cli.browser_connect as bc

_LOCKED = ("Login Data", "Login Data For Account", "Web Data")


def _fake_profile(root):
    (root / "Default" / "Network").mkdir(parents=True)
    (root / "Local State").write_text(json.dumps({"profile": {"last_used": "Default"}}))
    (root / "Default" / "Preferences").write_text("{}")
    for name in ("Cookies", *_LOCKED):
        con = sqlite3.connect(root / "Default" / name)
        con.execute("create table t(x)")
        con.execute("insert into t values(1)")
        con.commit()
        con.close()


def test_mirror_profile_auth_copies_exclusively_locked_login_dbs(tmp_path, monkeypatch):
    """Exclusive writers on the three login/autofill DBs: each is still copied (immutable read),
    and the result is the upstream ``{name: reason}`` shape -- empty, because nothing failed."""
    root, dst = tmp_path / "real", tmp_path / "copy"
    _fake_profile(root)
    monkeypatch.setattr(bc, "_AUTH_BACKUP_DEADLINE_S", 0.3)  # keep the test fast; 5 s in prod
    holders = []
    for name in _LOCKED:
        h = sqlite3.connect(root / "Default" / name)
        h.execute("begin exclusive")
        holders.append(h)
    try:
        failed = bc._mirror_profile_auth(str(root), str(dst), "Default")
    finally:
        for h in holders:
            h.rollback()
            h.close()
    assert failed == {}
    for name in ("Cookies", *_LOCKED):
        assert os.path.isfile(dst / "Default" / name)
    assert bc._mirror_profile_auth(str(root), str(dst), "Default") == {}


def test_mirror_profile_auth_reports_reason_by_name(tmp_path, monkeypatch):
    root, dst = tmp_path / "real", tmp_path / "copy"
    _fake_profile(root)
    monkeypatch.setattr(bc, "_copy_auth_file",
                        lambda src, d: "file is not a database" if os.path.basename(src) == "Web Data" else None)
    assert bc._mirror_profile_auth(str(root), str(dst), "Default") == {"Web Data": "file is not a database"}


@pytest.mark.parametrize("reason", [bc._AUTH_DB_LOCKED, "file is not a database"])
def test_unreadable_cookie_jar_fails_closed_naming_it_and_why(tmp_path, monkeypatch, reason):
    root = tmp_path / "real"
    _fake_profile(root)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    monkeypatch.setattr(bc, "_copy_auth_file",
                        lambda src, d: reason if os.path.basename(src) == "Cookies" else None)
    dst, err = bc.snapshot_real_profile("chrome", src=str(root))
    assert dst is None and err
    assert "cookie jar" in err and "Cookies" in err and reason in err, err
    assert "database(s) unavailable" not in err
    assert "close" not in err.lower()
    for name in _LOCKED:
        assert name not in err  # only the database that actually failed is named


def test_unreadable_login_dbs_do_not_fail_the_snapshot(tmp_path, monkeypatch):
    """The hand-over discards Login Data / Web Data whether or not they copied, so their failure
    changes nothing the person gets and never blocks the launch."""
    root = tmp_path / "real"
    _fake_profile(root)
    monkeypatch.setattr(bc, "get_hermes_home", lambda: tmp_path / "hh")
    monkeypatch.setattr(bc, "_copy_auth_file",
                        lambda src, d: bc._AUTH_DB_LOCKED if os.path.basename(src) in _LOCKED else None)
    dst, err = bc.snapshot_real_profile("chrome", src=str(root))
    assert err is None and dst
