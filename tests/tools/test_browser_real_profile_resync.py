"""Ticket #13: the agent browses on the person's real profile while their own Chrome is open.

What failed on 2026-09-24 (measured, not assumed): the driven browser stalled for a few seconds
(harness "Input.dispatchKeyEvent timed out after 5s waiting for the daemon"). The next acquire's
one- and two-second DevTools probes read the stalled-but-alive browser as gone, so the per-launch
re-sync ran UNDER it. The copy-browser holds its own Login Data / Login Data For Account / Web Data
in SQLite exclusive locking mode, the destination probe refused all three, and the call failed with
"3 database(s) unavailable. Close chrome and retry" — three times in four minutes, while a signed-in
complete snapshot sat right there. These tests use real SQLite locks held by a second connection,
the way Chrome holds them, not mocks of the copy.
"""

import contextlib
import json
import os
import sqlite3
import time
from unittest.mock import Mock, patch

import pytest

import hermes_cli.browser_connect as bc
import tests.tools.test_browser_real_profile as base
from tests.tools.test_browser_real_profile import bt_real_profile

AUTH_DBS = ("Login Data", "Login Data For Account", "Web Data")


def _db(path, value):
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("create table if not exists marker(value)")
        conn.execute("delete from marker")
        conn.execute("insert into marker values(?)", (value,))


def _read(path):
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return conn.execute("select value from marker").fetchone()[0]


def _profile(root):
    (root / "Default" / "Network").mkdir(parents=True)
    (root / "Local State").write_text(json.dumps({"profile": {"last_used": "Default"}}))
    _db(root / "Default" / "Network" / "Cookies", "cookies-v1")
    for name in AUTH_DBS:
        _db(root / "Default" / name, f"{name}-v1")
    (root / "Default" / "Preferences").write_text("{}")
    return root


def _hold(path, *, write):
    """A connection holding ``path`` the way Chrome holds these DBs: EXCLUSIVE locking mode, after a
    read (SHARED, kept) or after a write (EXCLUSIVE, kept)."""
    conn = sqlite3.connect(path, isolation_level=None, timeout=0)
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    if write:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("insert into marker values('driven-browser-write')")
        conn.execute("COMMIT")
    else:
        conn.execute("select count(*) from marker").fetchone()
    return conn


@pytest.fixture
def first_snapshot(tmp_path, monkeypatch):
    src = _profile(tmp_path / "real")
    home = tmp_path / "hh"
    monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
    getattr(bc, "_stale_logged", {}).clear()  # getattr: so a revert fails on behaviour, not a name
    dst, err = bc.snapshot_real_profile("chrome", src=str(src))
    assert err is None and dst
    getattr(bc, "pop_snapshot_note", lambda b: None)("chrome")
    return src, dst


class TestResyncUnderALiveCopyBrowser:
    def test_copy_held_by_a_browser_that_wrote_it_keeps_the_last_complete_snapshot(self, first_snapshot, caplog):
        """The incident: all three auth DBs of the COPY held (written) by the live driven browser.
        Before the fix: "(3 database(s) unavailable). Close chrome and retry". Now: the complete,
        signed-in snapshot is used, the model gets a one-line note, and nobody is told to close
        Chrome."""
        src, dst = first_snapshot
        for name in AUTH_DBS:
            _db(src / "Default" / name, f"{name}-v2")  # the person's Chrome moved on
        holders = [_hold(os.path.join(dst, "Default", name), write=True) for name in AUTH_DBS]
        try:
            with caplog.at_level("WARNING", logger=bc.logger.name):
                got, err = bc.snapshot_real_profile("chrome", src=str(src))
                again, err2 = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            for h in holders:
                h.close()
        assert err is None and got == dst, err
        assert err2 is None and again == dst
        note = bc.pop_snapshot_note("chrome")
        taken = time.strftime("%H:%M", time.localtime(os.path.getmtime(os.path.join(dst, bc._SNAPSHOT_DONE_MARKER))))
        assert note == f"using your sign-ins from {taken}; a newer copy couldn't be read while Chrome was writing"
        assert "close" not in note.lower()
        stale_lines = [r for r in caplog.records if "complete snapshot from" in r.getMessage()]
        assert len(stale_lines) == 1, "logged once, not once per launch"
        for name in AUTH_DBS:  # untouched: still the signed-in v1 copy, plus the holder's own write
            assert _read(os.path.join(dst, "Default", name)) == f"{name}-v1"
        # Cookies are not held and still refresh.
        assert _read(os.path.join(dst, "Default", "Network", "Cookies")) == "cookies-v1"

    def test_a_db_the_browser_has_only_read_is_not_replaced_under_it(self, first_snapshot):
        """SHARED only (Chrome read its Login Data, never wrote it): the old BEGIN IMMEDIATE probe
        passed, and the copy unlinked the file out from under the live browser (measured against a
        real Chrome for Testing on the copy dir). The file the browser has open must stay the file."""
        src, dst = first_snapshot
        target = os.path.join(dst, "Default", "Login Data")
        inode = os.stat(target).st_ino
        holder = _hold(target, write=False)
        try:
            got, err = bc.snapshot_real_profile("chrome", src=str(src))
            assert os.stat(target).st_ino == inode
        finally:
            holder.close()
        assert err is None and got == dst
        assert bc.pop_snapshot_note("chrome")

    def test_a_clean_resync_leaves_no_note_and_refreshes(self, first_snapshot):
        src, dst = first_snapshot
        _db(src / "Default" / "Login Data", "Login Data-v2")
        got, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None and got == dst
        assert bc.pop_snapshot_note("chrome") is None
        assert _read(os.path.join(dst, "Default", "Login Data")) == "Login Data-v2"


class TestFirstSnapshotStillFails:
    def test_first_snapshot_that_cannot_read_login_data_fails_closed(self, tmp_path, monkeypatch):
        """Nothing complete to fall back on: the person's Chrome holds Web Data and every lock-free
        read of it comes out torn. Fail, as before — a silently signed-out session is worse than an
        error. Still no "close Chrome"."""
        src = _profile(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        web = src / "Default" / "Web Data"
        with contextlib.closing(sqlite3.connect(web)) as conn, conn:
            conn.executemany("insert into marker values(?)", [("x" * 500,)] * 200)
        holder = _hold(web, write=True)
        try:
            with open(web, "r+b") as fh:  # mid-write, as a lock-free reader sees it
                fh.seek(4096 * 2 + 16)
                fh.write(os.urandom(2048))
            dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            holder.close()
        assert dst is None
        assert "1 database(s) unavailable" in err and "close" not in err.lower()
        assert bc.pop_snapshot_note("chrome") is None
        assert not (home / "browser-profile" / "chrome" / bc._SNAPSHOT_DONE_MARKER).exists()

    def test_complete_snapshot_of_another_profile_is_not_a_fallback(self, first_snapshot):
        src, dst = first_snapshot
        with open(os.path.join(dst, bc._SNAPSHOT_DONE_MARKER), "w") as fh:
            fh.write("Profile 6")  # the person has since switched profiles
        holder = _hold(os.path.join(dst, "Default", "Web Data"), write=True)
        try:
            got, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            holder.close()
        assert got is None and "unavailable" in err


class TestCopyOfADbTheBrowserHolds:
    def test_locked_source_is_read_without_spending_the_deadline(self, tmp_path):
        """Chrome's own Login Data is locked for as long as Chrome runs. The old path spent the whole
        five-second backup deadline per DB finding that out — 15 s per launch, measured against a
        live Chrome for Testing — before the immutable read that always followed."""
        src, dst = tmp_path / "Login Data", tmp_path / "out" / "Login Data"
        _db(src, "committed")
        holder = _hold(src, write=True)
        try:
            started = time.monotonic()
            assert bc._copy_auth_file(str(src), str(dst)) is True
            elapsed = time.monotonic() - started
        finally:
            holder.close()
        assert elapsed < 2.0, f"took {elapsed:.1f}s"
        assert _read(dst) in ("committed", "driven-browser-write")

    def test_a_torn_immutable_read_is_rejected_and_the_old_copy_kept(self, tmp_path):
        """A lock-free read of a DB Chrome is writing can be torn: measured 11 of 20 corrupt copies
        under a steady writer, every one reported as a good copy. A torn read must never replace a
        good copy."""
        src, dst = tmp_path / "Web Data", tmp_path / "out" / "Web Data"
        dst.parent.mkdir()
        _db(dst, "good-old-copy")
        with contextlib.closing(sqlite3.connect(src)) as conn, conn:
            conn.execute("create table marker(value)")
            conn.executemany("insert into marker values(?)", [("x" * 500,)] * 200)
        holder = _hold(src, write=True)
        try:
            with open(src, "r+b") as fh:  # what a reader sees mid-write: a page half overwritten
                fh.seek(4096 * 2 + 16)
                fh.write(os.urandom(2048))
            assert bc._copy_auth_file(str(src), str(dst)) is False
        finally:
            holder.close()
        assert _read(dst) == "good-old-copy"
        assert not os.path.exists(str(dst) + bc._SIDE_COPY_SUFFIX)

    def test_a_slow_backup_past_the_deadline_falls_back_to_the_lock_free_read(self, tmp_path, monkeypatch):
        """A locking backup that cannot finish (the source's write lock held, but not in a way the
        instant probe sees) is bounded by the deadline, then read lock-free — and the result is a
        whole copy of the committed state."""
        src, dst = tmp_path / "Login Data", tmp_path / "out" / "Login Data"
        _db(src, "committed")
        monkeypatch.setattr(bc, "_AUTH_DB_BACKUP_SECONDS", 0.5)
        monkeypatch.setattr(bc, "_source_held_exclusively", lambda uri: False)
        holder = sqlite3.connect(src, isolation_level=None, timeout=0)
        holder.execute("BEGIN EXCLUSIVE")  # normal locking mode: released at COMMIT/ROLLBACK
        try:
            started = time.monotonic()
            assert bc._copy_auth_file(str(src), str(dst)) is True
            elapsed = time.monotonic() - started
        finally:
            holder.rollback()
            holder.close()
        assert 0.5 <= elapsed < 3.0, f"took {elapsed:.1f}s"
        assert _read(dst) == "committed"


class TestAcquireNeverOverlaysALiveBrowser:
    """``_real_profile_cdp``: a live browser on the copy dir that does not answer DevTools in time
    is waited for, never snapshotted under. Uses the launch harness of the main suite."""

    def setup_method(self):
        h = base.TestRealProfileCdpLaunch()
        self._run, self._reset = h._run, h._reset

    def _live(self):
        holder = Mock(pid=4242)
        holder.ppid.return_value = os.getpid()  # its Hermes is alive: not an orphan
        return holder

    def test_stalled_live_browser_is_not_overlaid(self, tmp_path):
        self._reset()
        snapshot = Mock(return_value=(str(tmp_path), None))
        cdp, err = self._run(tmp_path, extra_patches=[
            patch.object(bt_real_profile, "_BUSY_BROWSER_WAIT_S", 0.6),
            patch.object(bt_real_profile, "_browsers_on_data_dir", return_value=[self._live()]),
            patch.object(bt_real_profile, "_surviving_chrome_cdp", return_value=None),
            patch("hermes_cli.browser_connect.snapshot_real_profile", snapshot)])
        assert cdp is None
        snapshot.assert_not_called()
        assert "still running" in err and "close" not in err.lower()

    def test_stalled_live_browser_that_answers_again_is_reattached(self, tmp_path):
        self._reset()
        snapshot = Mock(return_value=(str(tmp_path), None))
        answers = iter([None, None, "http://127.0.0.1:41000"])
        attach = Mock(return_value=("http://127.0.0.1:41000", None))
        cdp, err = self._run(tmp_path, extra_patches=[
            patch.object(bt_real_profile, "_browsers_on_data_dir", return_value=[self._live()]),
            patch.object(bt_real_profile, "_surviving_chrome_cdp", side_effect=lambda d: next(answers)),
            patch.object(bt_real_profile, "_attach_agent_browser_to_real_profile", attach),
            patch("tools.browser_tool_fidelity.ensure_keeper", return_value=None),
            patch("hermes_cli.browser_connect.snapshot_real_profile", snapshot)])
        assert err is None and cdp == "http://127.0.0.1:41000"
        snapshot.assert_not_called()
        attach.assert_called_once()

    def test_stale_snapshot_note_reaches_the_model_once(self, tmp_path):
        self._reset()
        bt_real_profile._pending_notes.clear()
        bc._snapshot_notes["chrome"] = "using your sign-ins from 12:39; a newer copy couldn't be read while Chrome was writing"
        cdp, err = self._run(tmp_path, extra_patches=[
            patch("hermes_cli.browser_connect.snapshot_real_profile",
                  side_effect=lambda b: (str(tmp_path), None))])
        assert err is None and cdp
        assert bt_real_profile.take_real_profile_note().startswith("using your sign-ins from 12:39")
        assert bt_real_profile.take_real_profile_note() is None
        self._reset()
