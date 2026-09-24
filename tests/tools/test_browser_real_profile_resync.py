"""Ticket #13: the agent browses on the person's real profile while their own Chrome is open.

What failed on 2026-09-24 (measured, not assumed): the driven browser stalled for a few seconds
(harness "Input.dispatchKeyEvent timed out after 5s waiting for the daemon"). The next acquire's
one- and two-second DevTools probes read the stalled-but-alive browser as gone, so the per-launch
re-sync ran UNDER it. The copy-browser holds its own Login Data / Login Data For Account / Web Data
in SQLite exclusive locking mode, the destination probe refused all three, and the call failed with
"3 database(s) unavailable. Close chrome and retry" — three times in four minutes. Those three DBs
never reach the driven browser anyway: the hand-over reads the COOKIE JAR with the person's own
browser and then deletes the copy's Cookies / Login Data / Web Data
(``_forget_keychain_bound_auth_files``); the driven browser recreates its own. So the fixtures run
that real sequence, and only the jar decides. Real SQLite locks held by a second connection, the
way Chrome holds them, not mocks of the copy.
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


JAR = os.path.join("Network", "Cookies")


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
def handed_over(tmp_path, monkeypatch):
    """The production sequence: first snapshot, the hand-over's forget, then the driven browser
    recreating its own auth DBs (sealed to its own mock keychain) in the copy."""
    src = _profile(tmp_path / "real")
    home = tmp_path / "hh"
    monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
    dst, err = bc.snapshot_real_profile("chrome", src=str(src))
    assert err is None and dst
    bt_real_profile._forget_keychain_bound_auth_files(dst)
    for rel in (JAR, *AUTH_DBS):
        assert not os.path.exists(os.path.join(dst, "Default", rel)), f"hand-over kept {rel}"
        _db(os.path.join(dst, "Default", rel), f"DRIVEN-{rel}")
    return src, dst


class TestResyncUnderALiveCopyBrowser:
    def test_copy_browser_holding_its_login_data_does_not_fail_the_launch(self, handed_over, caplog):
        """The incident's three DBs, write-held by the driven browser. Before: "(3 database(s)
        unavailable). Close chrome and retry". They are the driven browser's own files and the
        hand-over discards them, so the re-sync goes on — and claims nothing about them."""
        src, dst = handed_over
        _db(src / "Default" / JAR, "cookies-v2")  # the person signed in somewhere new
        holders = [_hold(os.path.join(dst, "Default", name), write=True) for name in AUTH_DBS]
        try:
            with caplog.at_level("INFO", logger=bc.logger.name):
                got, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            for h in holders:
                h.close()
        assert err is None and got == dst, err
        assert _read(os.path.join(dst, "Default", JAR)) == "cookies-v2", "the jar is the person's, fresh"
        for name in AUTH_DBS:
            assert _read(os.path.join(dst, "Default", name)) == f"DRIVEN-{name}"  # untouched
        assert any("hand-over discards them" in r.getMessage() for r in caplog.records)
        assert not hasattr(bc, "pop_snapshot_note"), "no note may claim stale sign-ins"

    def test_an_unreadable_cookie_jar_fails_closed_without_close_chrome(self, handed_over):
        """The jar is the only thing that signs the driven browser in, and after the hand-over
        there is no older jar of the person's to fall back on (the copy's is the driven browser's
        own). Unreadable -> fail closed, honestly."""
        src, dst = handed_over
        holder = _hold(os.path.join(dst, "Default", JAR), write=True)
        try:
            got, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            holder.close()
        assert got is None
        assert "cookie jar" in err and "close" not in err.lower()
        assert _read(os.path.join(dst, "Default", JAR)) == f"DRIVEN-{JAR}"

    def test_a_db_the_browser_has_only_read_is_not_replaced_under_it(self, handed_over):
        """SHARED only (Chrome read its Login Data, never wrote it): the old BEGIN IMMEDIATE probe
        passed, and the copy unlinked the file out from under the live browser (measured against a
        real Chrome for Testing on the copy dir). The file the browser has open must stay the file."""
        src, dst = handed_over
        target = os.path.join(dst, "Default", "Login Data")
        inode = os.stat(target).st_ino
        holder = _hold(target, write=False)
        try:
            got, err = bc.snapshot_real_profile("chrome", src=str(src))
            assert os.stat(target).st_ino == inode
        finally:
            holder.close()
        assert err is None and got == dst

    def test_a_clean_resync_refreshes_the_jar(self, handed_over):
        src, dst = handed_over
        _db(src / "Default" / JAR, "cookies-v2")
        got, err = bc.snapshot_real_profile("chrome", src=str(src))
        assert err is None and got == dst
        assert _read(os.path.join(dst, "Default", JAR)) == "cookies-v2"


def _torn(path):
    """Fill ``path``'s marker table, hold it as Chrome does, and tear a page — what a lock-free
    reader sees mid-write. Returns the holder (close it)."""
    with contextlib.closing(sqlite3.connect(path)) as conn, conn:
        conn.executemany("insert into marker values(?)", [("x" * 500,)] * 200)
    holder = _hold(path, write=True)
    with open(path, "r+b") as fh:
        fh.seek(4096 * 2 + 16)
        fh.write(os.urandom(2048))
    return holder


class TestFirstSnapshot:
    def test_first_snapshot_with_an_unreadable_jar_fails_closed(self, tmp_path, monkeypatch):
        """Nothing to fall back on: the person's Chrome holds its jar and every lock-free read of it
        comes out torn. Fail — a silently signed-out session is worse than an error. No "close"."""
        src = _profile(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        holder = _torn(src / "Default" / JAR)
        try:
            dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            holder.close()
        assert dst is None
        assert "cookie jar" in err and "close" not in err.lower()
        assert not (home / "browser-profile" / "chrome" / bc._SNAPSHOT_DONE_MARKER).exists()

    def test_first_snapshot_with_unreadable_web_data_still_launches(self, tmp_path, monkeypatch):
        src = _profile(tmp_path / "real")
        home = tmp_path / "hh"
        monkeypatch.setattr(bc, "get_hermes_home", lambda: home)
        holder = _torn(src / "Default" / "Web Data")
        try:
            dst, err = bc.snapshot_real_profile("chrome", src=str(src))
        finally:
            holder.close()
        assert err is None and dst
        assert _read(os.path.join(dst, "Default", JAR)) == "cookies-v1"


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


    def test_a_source_that_changed_during_a_whole_looking_copy_is_read_again(self, tmp_path):
        """A lock-free copy can pass the integrity check and still be a mix of before and after (a
        commit landing between pages). The source changing underneath is the tell: read again."""
        src, side = tmp_path / "Web Data", tmp_path / "out.side"
        _db(src, "before")
        attempts = []

        def backup(query):
            with contextlib.closing(sqlite3.connect(src.resolve().as_uri() + query, uri=True)) as a, \
                    contextlib.closing(sqlite3.connect(side)) as b:
                a.backup(b)
            attempts.append(query)
            if len(attempts) == 1:  # the browser commits while the first copy is being taken
                time.sleep(0.01)
                _db(src, "after")

        bc._immutable_copy(str(src), str(side), backup)
        assert len(attempts) == 2
        assert _read(side) == "after"

    def test_the_copy_is_swapped_in_whole_never_written_into_the_old_file(self, tmp_path):
        """A reader of the previous copy (an fd opened before the re-sync) must keep seeing that
        complete file: the new one arrives by rename, never by writing over the old bytes."""
        src, dst = tmp_path / "Web Data", tmp_path / "out" / "Web Data"
        dst.parent.mkdir()
        _db(dst, "old-copy")
        with contextlib.closing(sqlite3.connect(src)) as conn, conn:
            conn.execute("create table marker(value)")
            conn.executemany("insert into marker values(?)", [("new" * 300,)] * 50)
        old_bytes = dst.read_bytes()
        with open(dst, "rb") as reader:
            assert bc._copy_auth_file(str(src), str(dst)) is True
            reader.seek(0)
            assert reader.read() == old_bytes
        assert _read(dst) == "new" * 300

    def test_a_browser_that_opens_the_copy_after_the_probe_is_not_swapped_under(self, tmp_path, monkeypatch):
        """The probe and the swap are two steps; a browser opening the destination between them
        must still block the swap (the swap holds the destination's lock itself)."""
        src, dst = tmp_path / "Web Data", tmp_path / "out" / "Web Data"
        dst.parent.mkdir()
        _db(src, "person")
        _db(dst, "in-use")
        monkeypatch.setattr(bc, "_db_in_use", lambda path: False)  # it opened after the probe
        holder = _hold(dst, write=True)
        try:
            assert bc._copy_auth_file(str(src), str(dst)) is False
            inode = os.stat(dst).st_ino
        finally:
            holder.close()
        assert os.stat(dst).st_ino == inode and _read(dst) == "in-use"
        assert not os.path.exists(str(dst) + bc._SIDE_COPY_SUFFIX)


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

    def test_our_own_wedged_browser_is_restarted_not_waited_on_forever(self, tmp_path):
        """OURS (launched by this Hermes) and still silent after the wait: terminate it and launch
        afresh, or every later call pays the same wait until Hermes restarts."""
        self._reset()
        ours = self._live()
        alive = {"v": True}
        terminated = []
        snapshot = Mock(return_value=(str(tmp_path), None))
        launches = []
        cdp, err = self._run(tmp_path, launches=launches, extra_patches=[
            patch.object(bt_real_profile, "_BUSY_BROWSER_WAIT_S", 0.3),
            patch.object(bt_real_profile, "_own_browser_pids", side_effect=lambda: {4242} if alive["v"] else set()),
            patch.object(bt_real_profile, "_browsers_on_data_dir", side_effect=lambda d: [ours] if alive["v"] else []),
            patch.object(bt_real_profile, "_surviving_chrome_cdp", return_value=None),
            patch.object(bt_real_profile, "_terminate_real_profile_chrome",
                         side_effect=lambda: (terminated.append(4242), alive.update(v=False))),
            patch("hermes_cli.browser_connect.snapshot_real_profile", snapshot)])
        assert terminated, "our wedged browser was not terminated"
        snapshot.assert_called_once()
        assert err is None and cdp and [a[0] for a in launches] == [base.TestRealProfileCdpLaunch.PERSON, base.TestRealProfileCdpLaunch.DRIVEN]
        # The model is told its pages are gone, once, so it checks before resubmitting a form.
        assert bt_real_profile.take_restart_note() == bt_real_profile.RESTART_NOTE
        assert bt_real_profile.take_restart_note() is None
        self._reset()

    def test_a_wedged_browser_that_is_not_ours_is_never_terminated(self, tmp_path):
        self._reset()
        terminate = Mock()
        snapshot = Mock(return_value=(str(tmp_path), None))
        cdp, err = self._run(tmp_path, extra_patches=[
            patch.object(bt_real_profile, "_BUSY_BROWSER_WAIT_S", 0.3),
            patch.object(bt_real_profile, "_own_browser_pids", return_value=set()),
            patch.object(bt_real_profile, "_browsers_on_data_dir", return_value=[self._live()]),
            patch.object(bt_real_profile, "_surviving_chrome_cdp", return_value=None),
            patch.object(bt_real_profile, "_terminate_real_profile_chrome", terminate),
            patch("hermes_cli.browser_connect.snapshot_real_profile", snapshot)])
        terminate.assert_not_called()
        snapshot.assert_not_called()
        assert bt_real_profile.take_restart_note() is None
        assert cdp is None and "still running" in err


def test_browser_exec_result_carries_the_restart_note_once(tmp_path, monkeypatch):
    """The note reaches the model in the browser_exec result right after the restart, and only then."""
    import json as _json
    from tests.tools.test_browser_use_cli import _fake_cli
    from tools import browser_use_cli as bu_cli
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"browser": {"backend": "browser-use"}})
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [_fake_cli(tmp_path, 'cat > /dev/null\necho ok\n')])
    bt_real_profile._restart_notes[:] = [bt_real_profile.RESTART_NOTE]
    first = _json.loads(bu_cli.browser_exec("print(1)"))
    second = _json.loads(bu_cli.browser_exec("print(1)"))
    assert first.get("note") == bt_real_profile.RESTART_NOTE
    assert "resubmitting" in first["note"]
    assert "note" not in second

