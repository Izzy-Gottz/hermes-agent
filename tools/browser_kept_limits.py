"""The kept browser on a cloud computer must never take the machine down with it.

``browser.kept_profile`` is the lane a cloud computer (Memoe Away) runs: one Chromium, its own
profile, on a small machine whose other tenant is the gateway that answers the person. A browser
that grows until the kernel's OOM killer chooses has, on the machine it was measured on, chosen
the gateway (exit 137 mid-reply). So on this lane, and only on it:

* **The browser is the thing the kernel kills first.** Its main process is marked
  ``oom_score_adj = 1000`` straight after it starts; every process Chromium forks inherits it, and
  Chromium cannot lower it again without ``CAP_SYS_RESOURCE``. Under pressure the browser goes and
  the conversation stays; the next browser call starts a new one on the same kept profile.
* **It is launched small**: a renderer-process ceiling, no GPU process, no extensions, no
  ``/dev/shm`` reliance, a V8 heap ceiling per renderer (:data:`KEPT_FLAGS`).
* **Tabs are capped** (``browser.max_tabs``): pages beyond the cap are closed, oldest first, by the
  cleanup thread that already releases the browser after ``inactivity_timeout``.
* **A memory floor** (``browser.memory_floor_mb``): when the machine's ``MemAvailable`` falls under
  it, extra tabs go first and, if that is not enough, the browser is released early.

On a Mac (no kept profile) none of this runs: :func:`active` is False and every function here is a
no-op, so the person's own browsing lane is untouched.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

#: Chromium switches for the kept lane, on top of the shared set. Each is a documented switch;
#: none weakens the sandbox or site isolation (the profile holds the person's sign-ins).
KEPT_FLAGS = (
    "--renderer-process-limit=3",
    "--disable-gpu",
    "--disable-extensions",
    "--disable-dev-shm-usage",
    "--js-flags=--max-old-space-size=384",
)

#: The value written to the browser's ``/proc/<pid>/oom_score_adj``: the kernel's maximum.
OOM_SCORE_ADJ = 1000


def _cfg(key: str, default, parse):
    from tools.browser_tool_origin import origin_module
    return origin_module()._browser_cfg(key, default, parse, key + " from config")


def active() -> bool:
    """True on the kept-profile lane of a Linux host — a cloud computer — and nowhere else."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        from tools import browser_tool_cloud as cloud
        return bool(cloud._use_real_profile() and cloud._use_kept_profile())
    except Exception:  # noqa: BLE001 - a config we cannot read is not the cloud lane
        return False


def _int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def max_tabs() -> int:
    """``browser.max_tabs``; 0 (absent, invalid) is no cap."""
    return _cfg("max_tabs", 0, _int)


def memory_floor_mb() -> int:
    """``browser.memory_floor_mb``; 0 (absent, invalid) is no floor."""
    return _cfg("memory_floor_mb", 0, _int)


def kept_flags() -> tuple:
    return KEPT_FLAGS if active() else ()


def mark_expendable(pid: int, proc_root: str = "/proc") -> bool:
    """Make ``pid`` (the kept browser's main process) the OOM killer's first choice."""
    if not active():
        return False
    try:
        with open(os.path.join(proc_root, str(int(pid)), "oom_score_adj"), "w") as fh:
            fh.write(str(OOM_SCORE_ADJ))
        return True
    except OSError as exc:
        logger.warning("kept browser: could not mark pid %s expendable to the OOM killer: %s", pid, exc)
        return False


def _tree(data_dir: str, proc_root: str) -> List[int]:
    """Every process of the browser bound to ``data_dir``: the ones whose command line names it,
    and everything descended from them (Chromium's zygotes and renderers do not repeat it)."""
    want = f"--user-data-dir={data_dir}".encode()
    parent = {}
    roots = []
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, entry, "cmdline"), "rb") as fh:
                argv = fh.read().split(b"\0")
            with open(os.path.join(proc_root, entry, "stat"), "rb") as fh:
                stat = fh.read()
            parent[int(entry)] = int(stat[stat.rindex(b")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if want in argv:
            roots.append(int(entry))
    found, frontier = set(roots), list(roots)
    while frontier:
        pid = frontier.pop()
        for child, ppid in parent.items():
            if ppid == pid and child not in found:
                found.add(child)
                frontier.append(child)
    return sorted(found)


def mark_tree(data_dir: Optional[str], proc_root: str = "/proc") -> int:
    """Raise every process of the kept browser to :data:`OOM_SCORE_ADJ`. Chromium resets its main
    process after start-up (measured: written 1000 at spawn, read back 0 once the browser was
    attached), so this runs again after the attach and on every cleanup tick. Returns how many
    were raised."""
    if not active() or not data_dir:
        return 0
    raised = 0
    for pid in _tree(data_dir, proc_root):
        path = os.path.join(proc_root, str(pid), "oom_score_adj")
        try:
            with open(path) as fh:
                if int(fh.read().strip() or 0) >= OOM_SCORE_ADJ:
                    continue
            with open(path, "w") as fh:
                fh.write(str(OOM_SCORE_ADJ))
            raised += 1
        except (OSError, ValueError):
            continue
    return raised


def mem_available_mb(meminfo: str = "/proc/meminfo") -> Optional[int]:
    try:
        with open(meminfo, encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _pages(port: int) -> List[dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=3) as r:  # noqa: S310 - loopback
        return [t for t in json.loads(r.read().decode("utf-8")) if t.get("type") == "page"]


def _close(port: int, target_id: str) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/close/{target_id}", timeout=3):  # noqa: S310
        pass


def trim_tabs(port: int, keep: int, *, pages: Callable[[int], List[dict]] = _pages,
              close: Callable[[int, str], None] = _close) -> int:
    """Close page targets beyond ``keep``, oldest first. Chromium's ``/json/list`` lists the most
    recently created target first, so the tail is the oldest. Returns how many were closed."""
    if keep <= 0:
        return 0
    try:
        listed = pages(port)
    except Exception as exc:  # noqa: BLE001 - the browser went between ticks
        logger.debug("kept browser: could not list pages: %s", exc)
        return 0
    closed = 0
    for target in listed[keep:]:
        try:
            close(port, str(target.get("id") or ""))
            closed += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("kept browser: could not close a page: %s", exc)
    if closed:
        logger.info("kept browser: closed %d page(s) over the cap of %d", closed, keep)
    return closed


def enforce(port: Optional[int], release: Callable[[], None], *,
            available: Callable[[], Optional[int]] = mem_available_mb,
            trim: Callable[[int, int], int] = trim_tabs, data_dir: Optional[str] = None,
            mark: Callable[[Optional[str]], int] = mark_tree) -> str:
    """One cleanup tick on the kept lane: keep the browser the OOM killer's first choice, cap the
    tabs, then hold the memory floor. Returns what it did (``""``, ``"trimmed"``, ``"released"``)
    for the log and the tests."""
    if not active() or port is None:
        return ""
    mark(data_dir)
    did = ""
    cap = max_tabs()
    if cap and trim(port, cap):
        did = "trimmed"
    floor = memory_floor_mb()
    if floor:
        free = available()
        if free is not None and free < floor:
            if trim(port, 1):
                did = "trimmed"
                free = available()
            if free is not None and free < floor:
                logger.warning("kept browser: %s MB available is under the %s MB floor; releasing the browser",
                               free, floor)
                release()
                return "released"
    return did
