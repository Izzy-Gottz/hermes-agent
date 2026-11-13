"""Frozen old-updater compatibility stub (tests/compat/old_updater_surface.json, ``bare`` list).

The extraction-era home of ``warm_agent_browser_npx_cache``: a pre-PM ``hermes update`` still running
mid-swap may import it from the NEW tree. agent-browser is gone.

(fork) Also the host-browser facts a host app that seeds Hermes (Memoe) reads by these names, and the
driven-browser lane (``tools/browser_tool_real_profile.py``) uses: where packaged Chrome for Testing
builds live (``_chromium_search_roots``), and the "still downloading its browser" marker the host
leaves while it fetches PM's Chromium (``_browser_pending``). Upstream has no slot for either; the
names predate the rewire and Memoe's own suites read them here.
"""

import os
import sys
from typing import List


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Frozen old-updater surface: nothing is warmed (there is no agent-browser to warm); always False."""
    return False


def _chromium_search_roots() -> List[str]:
    """(fork) Packaged Chromium / Chrome for Testing scan roots: ``PLAYWRIGHT_BROWSERS_PATH``, then
    ``~/.agent-browser/browsers`` (agent-browser's own download dir, kept: a Mac that has one keeps its
    Chrome for Testing), then the per-OS Playwright default cache. PM's pinned Chromium is resolved by
    ``hermes_cli.browser_runtime`` and comes after these (``driven_browser_executable``)."""
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    home = os.path.expanduser("~")
    roots: List[str] = [env_path] if env_path and env_path != "0" else []
    roots.append(os.path.join(home, ".agent-browser", "browsers"))
    roots.append(os.path.join(home, ".cache", "ms-playwright"))
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Caches", "ms-playwright"))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        roots.append(os.path.join(local, "ms-playwright"))
    return roots


# (fork) The host installs PM's pinned Chromium itself, outside Hermes, while lazy installs stay off
# (Memoe ships its seed without the 330 MB browser and fetches it on first launch). While that
# download is outstanding the host leaves this marker in PM's store root, and browser_exec says so in
# plain words instead of failing with "install Chromium" -- there is nothing for anyone to install.
BROWSER_PENDING_MARKER = ".browser-pending"


def _browser_pending() -> bool:
    """The host says its pinned Chromium is still on its way (``<store>/.browser-pending``)."""
    try:
        from pm.paths import store_root
        return (store_root() / BROWSER_PENDING_MARKER).is_file()
    except Exception:  # an unresolvable store is not a pending browser
        return False


def browser_pending_message() -> str:
    """What a browser tool answers while the host's browser download is outstanding."""
    from tools.fix_reasons import host_app_name
    name = host_app_name()
    name = name[:1].upper() + name[1:]
    return (f"{name} is still downloading its browser (about 330 MB). "
            "Try again in a minute; nothing else is needed.")
