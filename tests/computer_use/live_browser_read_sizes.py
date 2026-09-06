"""Opt-in: measure what `browser_read` actually costs, against a real page.

The numbers in `schema.py`, the skill and `test_cua_browser_read.py` all come
from this script. It exists because the published guidance for this is wrong
in the direction that matters: Playwright MCP's "a snapshot is 200-400 tokens
against 3,000-5,000 for a screenshot" is true of a small page and off by more
than an order of magnitude on a real one. A closed loop of invented constants
agrees with itself; this measures the artefact instead.

It never touches the person's own browser. It launches a driver-owned
`isolated_new` Chromium, which lives and dies with the driver session, and
navigates it to a public page.

    .venv/bin/python tests/computer_use/live_browser_read_sizes.py [URL]

Measured 2026-09-06, macOS, cua-driver 0.23.2, on
https://en.wikipedia.org/wiki/Accessibility :

    default (viewport-first)   9,337 B   ~2,334 tokens   1.6 s
    query='search'             3,805 B   ~  951 tokens   1.6 s
    include_text              30,405 B   ~7,601 tokens   1.8 s
    full                      63,309 B  ~15,827 tokens   1.6 s

against ~1,568 image tokens and ~2.2 s for a `som` capture of the same
window. Which is the whole design argument: this tool wins on ACTING, not on
reading, so the cheap shape is the default and `full` has to be asked for.
"""

import json
import sys
import time

from tools.computer_use import tool as T

URL = sys.argv[1] if len(sys.argv) > 1 else \
    "https://en.wikipedia.org/wiki/Accessibility"
SESSION = "live-browser-read-sizes"


def main() -> int:
    backend = T._get_backend(session_id=SESSION)
    prep = backend.call_tool(
        "browser_prepare",
        {"allow_launch": True, "profile": {"mode": "isolated_new"}},
        timeout=60.0)
    body = prep.get("structuredContent") or {}
    pid = body.get("prepared_pid")
    if not pid:
        print("could not launch an isolated browser:",
              json.dumps(body)[:300])
        return 2
    time.sleep(1.5)
    windows = [w for w in backend.list_windows() if w.get("pid") == pid]
    if not windows:
        print("the isolated browser has no window yet")
        return 2
    bound = backend.browser_bind(pid=pid, window_id=windows[0]["window_id"])
    if bound.get("status") != "ok":
        print("bind refused:", json.dumps(bound)[:300])
        return 2
    target, tab = backend.browser_targets()
    backend.call_tool("browser_navigate",
                      {"target_id": target, "tab_id": tab, "url": URL},
                      timeout=60.0)
    # Wait for the page rather than guessing at a sleep. The first attempt at
    # this used a flat 2.5 s and measured a half-loaded page: 330 tokens one
    # run and 2,338 the next, from the same URL. The second attempt polled the
    # snapshot every second and lost the CDP attachment ("cannot attach to tab
    # target") — hammering the endpoint is its own failure. So: a few spaced
    # reads through the public path, stopping as soon as the page has refs.
    ready = False
    for _ in range(6):
        time.sleep(2.0)
        probe = json.loads(T.handle_computer_use(
            {"action": "browser_read", "query": "the"}, session_id=SESSION))
        if probe.get("error"):
            continue
        if probe.get("refs") or (probe.get("outline") or ""):
            ready = True
            break
    if not ready:
        print("warning: the page never became readable; sizes below may be "
              "of a blank tab\n")

    print(f"{URL}\n")
    print(f"{'shape':26s} {'ms':>7s} {'bytes':>9s} {'~tokens':>9s} {'refs':>5s}")
    for label, args in (
        ("default", {}),
        ("query='search'", {"query": "search"}),
        ("include_text", {"include_text": True}),
        ("full", {"full": True}),
    ):
        start = time.perf_counter()
        out = T.handle_computer_use({"action": "browser_read", **args},
                                    session_id=SESSION)
        ms = (time.perf_counter() - start) * 1000
        try:
            parsed = json.loads(out)
        except ValueError:
            parsed = {}
        if parsed.get("error"):
            print(f"{label:26s} {ms:7.0f}   refused: {parsed['error'][:60]}")
            continue
        print(f"{label:26s} {ms:7.0f} {len(out):9d} {len(out) // 4:9d} "
              f"{len(parsed.get('refs') or []):5d}")
    backend.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
