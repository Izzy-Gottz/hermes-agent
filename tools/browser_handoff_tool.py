"""``browser_handoff``: put the page in front of the person, wait for them, carry on where they left it.

Moe's own browser runs out of sight (tools/browser_tool_real_profile.py). Some steps only the person
can take -- a passkey, a CAPTCHA, an identity check, "open that so I can approve it" -- and on
2026-09-24 there was no way to show them the page: Moe gave a text link at best, and told the owner
to "approve it on your phone" when nothing had been sent to a phone. This tool is the way.

Routes, first that works:

1. ``driven_browser`` -- Moe's own browser, shown: relaunched headed on the SAME profile copy with the
   same cookies, the tab reopened at its URL, raised in front. A reload loses typed-but-unsaved form
   state, and the result says so (``form_state_lost``).
2. ``chrome`` -- the extension lane is connected: the URL opens in Moe's own tab group in the
   person's Chrome (signed in as them; Moe's own browser's page state does not carry over). The
   extension never focuses a window by design, so the tab is there but not raised (``front: false``).
3. ``default_browser`` -- the URL opens in the person's default browser, with the explicit note that
   Moe's session does not carry over.

A window is put up only for a person AT THIS MAC (``at_this_mac``: a live turn from the machine's own
surfaces and a fresh "in use" presence stamp). The owner texting from their phone is live but not here,
and a turn nobody is at (cron, a helper, other people's words) is neither: both get the ``person_needed``
fixable result (tools/fix_reasons.py) with ``tell_owner``, the words to pass on, instead of a window on
an empty Mac or an invented device prompt.

After the call the turn ENDS: the model tells the person what to do in one line and waits. When they
say they are done, ``browser_handoff(done=true)`` reads where the page is now; the next launch of
Moe's browser is out of sight again.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ROUTE_DRIVEN = "driven_browser"
ROUTE_CHROME = "chrome"
ROUTE_DEFAULT = "default_browser"

_lock = threading.Lock()
#: task_id -> the last hand-off: {"route", "url", "at"}.
_handoffs: Dict[str, Dict[str, Any]] = {}

_WAIT = ("Tell the person in one short line what to do there ({reason}) and that you will carry on when they say "
         "they are done. Then END YOUR TURN and wait -- do not keep working, do not poll the page. When they say "
         "done, call browser_handoff(done=true).")
_GROUNDED = ("Say only what the page shows: never that a code or a prompt was sent to their phone or another "
             "device unless page_says (or another tool result) shows the site said so.")


def _check_available() -> bool:
    try:
        from tools.browser_tool_install import check_browser_requirements
        from tools.browser_use_cli import is_browser_use_cli_mode
        return bool(is_browser_use_cli_mode() or check_browser_requirements())
    except Exception:
        return False


def _safe_url(url: str) -> Optional[str]:
    """An error when ``url`` may not be opened for the person (not http(s), or the navigation policy refuses it)."""
    if not url.startswith(("http://", "https://")):
        return "Only an http(s) page can be handed to the person."
    try:
        from tools.browser_tool import evaluate_url_safety
        err = evaluate_url_safety(url)
    except Exception:
        err = None
    return (err or {}).get("error") if err else None


def _presence() -> dict:
    from tools.browser_chrome_extension import turn_presence
    return turn_presence()


def owner_message(reason: str, url: str, step: Optional[dict] = None) -> str:
    """Plain words for the owner about a step that waits for them: what, where, and what happens
    next. Only what the page showed -- never that anything was sent to a device."""
    from tools.browser_person_step import host_of
    host = host_of(url) or "A site"
    says = (step or {}).get("page_says") or ""
    line = f"{host} needs you for one step: {reason}"
    if says:
        line += f' (the page says: "{says}")'
    line += ". It's waiting in Moe's browser on your Mac"
    if url:
        line += f" ({url})"
    return line + ". Tell me when you're at the Mac and I'll bring it up in front of you."


def _needs_person(reason: str, url: str, presence: dict, why: str, step: Optional[dict] = None) -> str:
    """The step waits for a person who is not at this Mac: nothing is opened on its screen.

    * A live turn from a chat app (the owner texting from their phone): the reply IS the message --
      the model says ``tell_owner`` there.
    * A turn nobody is at (a scheduled job, a helper): ``tell_owner`` goes to the owner through
      ``reach_owner(text)`` when the runtime offers it (Memoe; slice A), and is the job's reported
      outcome either way.
    """
    from tools.browser_person_step import describe, host_of
    from tools.fix_reasons import PERSON_NEEDED, fix_error
    host = host_of(url)
    what = describe(step) if step else reason
    tell = owner_message(reason, url, step)
    if presence.get("live"):
        where = presence.get("platform") or "a chat"
        next_step = (f"They are writing from {where.replace('_', ' ')}, away from the Mac: tell them tell_owner in your "
                     "reply, then stop -- do not keep working the page.")
    else:
        next_step = ("Nobody is at the Mac: pass tell_owner to reach_owner(text) if you have that tool, and report it "
                     "as this job's outcome -- the step is waiting for them.")
    msg = (f"{host or 'This page'} needs the person for this step: {what}. This is {why}, so nothing was opened on "
           f"the Mac's screen. {next_step} " + _GROUNDED)
    extra: Dict[str, Any] = {"tell_owner": tell, "reason": reason}
    if url:
        extra["url"] = url
    if step:
        extra["step"] = step.get("kind", "")
    return fix_error(msg, PERSON_NEEDED, subject=host or None, retry=False, **extra)


def _open_in_chrome_group(bridge: dict, url: str) -> Optional[str]:
    """Open ``url`` in Moe's tab group through the extension's relay; error string or None."""
    from websockets.sync.client import connect
    try:
        with connect(bridge["ws_url"], open_timeout=5, close_timeout=2) as ws:
            ws.send(json.dumps({"id": 1, "method": "Target.createTarget", "params": {"url": url}}))
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                msg = json.loads(ws.recv(timeout=max(0.1, deadline - time.monotonic())))
                if msg.get("id") != 1:
                    continue
                if "error" in msg:
                    return str((msg["error"] or {}).get("message") or msg["error"])
                return None
    except Exception as exc:
        return str(exc)
    return "the extension did not answer"


def _open_default_browser(url: str) -> Optional[str]:
    try:
        if sys.platform == "darwin":
            proc = subprocess.run(["/usr/bin/open", url], capture_output=True, text=True, timeout=15,
                                  stdin=subprocess.DEVNULL)
            return None if proc.returncode == 0 else (proc.stderr or f"open exited {proc.returncode}").strip()
        import webbrowser
        return None if webbrowser.open(url) else "no browser could be opened"
    except Exception as exc:
        return str(exc)


def _record(task_id: str, route: str, url: str) -> None:
    with _lock:
        _handoffs[task_id] = {"route": route, "url": url, "at": time.time()}


def _hand_back(task_id: str) -> str:
    from tools.registry import tool_result
    with _lock:
        last = _handoffs.pop(task_id, None)
    route = (last or {}).get("route") or ROUTE_DRIVEN
    out: Dict[str, Any] = {"success": True, "route": route}
    if route == ROUTE_DRIVEN:
        from tools import browser_tool_real_profile as rp
        out.update(rp.hand_back())
        out["next"] = ("Carry on with browser_exec in the same session: it drives this same window. Read the page "
                       "first (page_info()) -- the person may have moved it on. Moe's browser goes back out of sight "
                       "the next time it starts.")
    elif route == ROUTE_CHROME:
        out["next"] = ("Carry on with browser_exec(where=\"chrome\"): the page is in Moe's tab group in the person's "
                       "Chrome. Read it first; they may have moved it on.")
    else:
        out["next"] = ("The person worked in their own browser, which Moe cannot read. Ask what they finished if it "
                       "matters, or continue in Moe's browser; its session did not change.")
    if last and last.get("url") and not out.get("url"):
        out["handed_url"] = last["url"]
    return tool_result(out)


def browser_handoff(reason: str = "", url: str = "", resume_hint: str = "", done: bool = False,
                    task_id: Optional[str] = None) -> str:
    from tools.registry import tool_error, tool_result
    task = task_id or "default"
    if done:
        return _hand_back(task)
    reason = " ".join(str(reason or "").split())[:200]
    if not reason:
        return tool_error("Say why the person needs the page (reason), e.g. \"approve the Google passkey\".")
    url = str(url or "").strip()
    if url and (err := _safe_url(url)):
        return tool_error(err)

    from tools import browser_chrome_extension as chrome_lane
    from tools import browser_tool_real_profile as rp

    current = url
    if not current:
        pages = rp.driven_pages()
        current = pages[0]["url"] if pages else ""

    presence = _presence()
    here, why = chrome_lane.at_this_mac(presence)
    if not here:
        step = None
        try:
            from tools.browser_person_step import probe_active_page
            cached = rp._origin()._real_profile_cdp_cache.get("cdp")
            step = probe_active_page(cached) if cached else None
        except Exception:
            step = None
        return _needs_person(reason, current, presence, why or "a turn with no person present", step)

    base: Dict[str, Any] = {"success": True, "reason": reason}
    if resume_hint:
        base["resume_hint"] = str(resume_hint)[:300]

    # 0. The work is already in the person's Chrome (the last call ran in Moe's tab group there).
    if chrome_lane.last_lane(task) == chrome_lane.LANE_CHROME:
        _record(task, ROUTE_CHROME, url)
        return tool_result({**base, "route": ROUTE_CHROME, "url": url or None, "front": False, "already_open": True,
                            "session": "the person's own Chrome, where this task has been working all along",
                            "next": ("The page is already in the person's Chrome, in the \"Moe\" tab group (Moe's own "
                                     "window there, which is not raised on its own). " + _WAIT.format(reason=reason)
                                     + " " + _GROUNDED)})

    # 1. Moe's own browser, shown. The URL the model names only picks the tab; it is never a new site.
    shown = rp.show_to_person(url, task)
    if shown.get("ok"):
        _record(task, ROUTE_DRIVEN, shown.get("url", ""))
        out = {**base, "route": ROUTE_DRIVEN, "url": shown.get("url"), "title": shown.get("title"),
               "front": bool(shown.get("front")), "relaunched": bool(shown.get("relaunched"))}
        if shown.get("form_state_lost"):
            out["form_state"] = ("lost: Moe's browser had to restart in a window, so the page was reloaded at its "
                                 "address. Sign-ins came across; anything typed on the page that the site had not "
                                 "saved is gone. Say so if it matters, and re-enter it after they are done.")
        if shown.get("tabs_lost"):
            out["other_tabs"] = f"{shown['tabs_lost']} other tab(s) of Moe's browser did not reopen."
        if out["front"]:
            where = "The page is open in Moe's browser window, in front of them (macOS confirmed it)."
        else:
            where = ("The page is open in Moe's browser window, but macOS did not bring it forward: tell them \"It's "
                     "open, but macOS didn't bring it forward -- click Moe's browser in the Dock.\" Never say it is "
                     "in front of them.")
        out["next"] = f"{where} " + _WAIT.format(reason=reason) + " " + _GROUNDED
        return tool_result(out)
    target = current or str(shown.get("url") or "")
    if not target:
        return tool_error("There is no page to hand over: Moe's own browser has no page open and no url was given. "
                          "Open the page first with browser_exec, or pass url.")
    if (err := _safe_url(target)):
        return tool_error(err)

    # 2. Moe's tab group in the person's Chrome.
    browser_cfg = {}
    try:
        from tools.browser_use_cli import _read_browser_cfg
        browser_cfg = _read_browser_cfg()
    except Exception:
        pass
    bridge = chrome_lane.read_bridge() if chrome_lane.lane_enabled(browser_cfg) else None
    if bridge is not None:
        err = _open_in_chrome_group(bridge, target)
        if err is None:
            _record(task, ROUTE_CHROME, target)
            return tool_result({**base, "route": ROUTE_CHROME, "url": target, "front": False,
                                "session": ("the person's own Chrome, signed in as them; what Moe's own browser had "
                                            "on this page (a half-done sign-in, a form) did not carry over"),
                                "next": ("It is open in the person's Chrome, in the \"Moe\" tab group (Moe's own "
                                         "window there, which is not raised on its own). " + _WAIT.format(reason=reason)
                                         + " " + _GROUNDED),
                                **({"driven_browser": shown.get("why")} if shown.get("why") else {})})
        logger.info("handoff: the Chrome lane could not open the page: %s", err)

    # 3. The person's default browser.
    err = _open_default_browser(target)
    if err:
        return tool_error(f"Could not show the page to the person ({err}). Give them the address instead: {target}")
    _record(task, ROUTE_DEFAULT, target)
    return tool_result({**base, "route": ROUTE_DEFAULT, "url": target,
                        "session": ("the person's default browser: Moe's own session (its sign-in, anything typed on "
                                    "the page) does NOT carry over, so they may have to sign in there themselves"),
                        "next": ("It is open in the person's default browser. Tell them Moe's session did not come "
                                 "with it. " + _WAIT.format(reason=reason) + " " + _GROUNDED),
                        **({"driven_browser": shown.get("why")} if shown.get("why") else {})})


BROWSER_HANDOFF_SCHEMA = {
    "name": "browser_handoff",
    "description": (
        "Show the person the page Moe is working on, so they can do a step only they can do -- a passkey or "
        "security key, a CAPTCHA, an identity check, an approval -- or whenever they ask to see it ('open that so "
        "I can approve it'). Moe's own browser runs out of sight; this brings it up in a window in front of them "
        "on the same tab with the same sign-ins (or, if that cannot be done, opens the page in their Chrome via "
        "Moe's tab group, else in their default browser -- the result's route and session say which, and what "
        "carried over). Then tell them in one line what to do and END YOUR TURN: they say when they are done, and "
        "you call browser_handoff(done=true) and carry on in the same browser_exec session. Use it when a result "
        "has needs_person, or browser_vault_enter_code finds no code field. A window is only ever put up for a "
        "person at this Mac: when they are writing from their phone, or nobody is there (a scheduled job), it opens "
        "nothing and returns code person_needed with tell_owner -- say it in your reply, or pass it to "
        "reach_owner(text) on a job, and stop. "
        "Only tell the person that something was sent, or is waiting on their phone or another device, when a "
        "tool result shows the site said so (page_says)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "What the person needs to do there, in plain words "
                                                        "(e.g. \"confirm the Google passkey\")."},
            "url": {"type": "string", "description": "Optional: which open page (its address), when Moe's browser "
                                                     "has several, or the page to open when Moe's browser has none."},
            "resume_hint": {"type": "string", "description": "Optional: what you will do once they are done."},
            "done": {"type": "boolean", "default": False,
                     "description": "True when the person says they have finished: reads where the page is now."},
        },
        "required": [],
    },
}


def _handle(args: Dict[str, Any], **kwargs) -> str:
    return browser_handoff(reason=str(args.get("reason") or ""), url=str(args.get("url") or ""),
                           resume_hint=str(args.get("resume_hint") or ""), done=bool(args.get("done", False)),
                           task_id=kwargs.get("task_id"))


from tools.registry import no_cache_check_fn, registry  # noqa: E402

registry.register(
    name="browser_handoff",
    toolset="browser",
    schema=BROWSER_HANDOFF_SCHEMA,
    handler=_handle,
    check_fn=no_cache_check_fn(_check_available),
    emoji="🤝",
)
