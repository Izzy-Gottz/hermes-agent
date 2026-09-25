"""Keeping ``browser_exec`` honest and bounded when the page, not the browser, is the problem.

Ticket #16 (2026-09-25): through a ~90 minute browser_exec session every call raised
``Runtime.evaluate timed out after 5s waiting for the daemon`` eight to ten times, and the last
one never recovered. The daemon was alive throughout: its log shows the late replies arriving
(``Received duplicate response``). What was slow was the page's renderer, on a Mac under heavy
load, in a browser that had accumulated 40 tabs. Four things made that worse than it had to be,
and this module holds the fix for each:

* The harness (browser_harness 0.1.13) gives every IPC round trip a fixed 5 s, bound at def time
  (``helpers._send(req, response_timeout=DEFAULT_IPC_RESPONSE_TIMEOUT_SECONDS)`` and ``cdp(...,
  _response_timeout=...)``) with no override, and its error blames "the daemon".
  :data:`HARNESS_PATCH_SOURCE` runs inside the harness process before the model's code: calls that
  used the default get up to :data:`PAGE_BUDGET_S` (never past the call's own deadline), and on
  expiry the error says which page did not answer, whether a dialog is open on it, that the browser
  itself still answers, and what to do.
* ``new_tab()`` never closes anything. The same patch records every target the harness creates
  and, when the process exits, closes the oldest beyond :data:`TAB_CAP` — never the attached tab,
  never a tab Hermes did not open, and only in Moe's own browser.
* The CDP supervisor's dialog bridge holds a page's JS thread for up to 300 s waiting for
  ``browser_dialog``, a tool the model does not have in Browser Use mode.
  :func:`exec_dialog_policy` dismisses those at once instead, and :func:`dialog_report` puts
  what happened in the browser_exec result.
* The fidelity keeper's ``unresumed`` counter (a new tab left paused at start) was visible only
  in a status file. :func:`fidelity_unresumed` lets a failed result say it went up.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Env contract with HARNESS_PATCH_SOURCE (set by browser_exec, read inside the harness process).
ENV_DEADLINE = "HERMES_BU_DEADLINE"      # epoch seconds the whole call must finish by
ENV_TAB_CAP = "HERMES_BU_TAB_CAP"        # keep at most this many Hermes-opened tabs (unset = no cap)
ENV_TAB_LEDGER = "HERMES_BU_TAB_LEDGER"  # JSON file listing the tabs the harness opened, oldest first
ENV_CURRENT_TAB = "HERMES_BU_CURRENT_TAB"  # JSON file the harness writes on exit: the tab it is attached to

PAGE_BUDGET_S = 30.0
TAB_CAP = 6

# Runs inside the harness process (its own Python, no Hermes on the path), in the module
# namespace the model's code executes in. Patches helpers' module globals, which every helper
# (cdp, js, page_info, new_tab, switch_tab...) resolves at call time.
HARNESS_PATCH_SOURCE = r'''
def _hermes_patch_harness():
    import atexit, json, os, sys, time
    try:
        from browser_harness import helpers as _h
        _orig_send, _orig_eval = _h._send, _h._runtime_evaluate
        _default, _base_timeout = _h.DEFAULT_IPC_RESPONSE_TIMEOUT_SECONDS, _h._IPCResponseTimeout
    except Exception:
        return  # a harness without these seams runs unpatched rather than not at all
    if getattr(_h, "_hermes_patched", False):
        return
    try:
        _deadline = float(os.environ.get("HERMES_BU_DEADLINE") or 0)
    except ValueError:
        _deadline = 0.0
    _ledger_path = os.environ.get("HERMES_BU_TAB_LEDGER") or ""
    _marker = "\U0001F434"  # the harness's cosmetic tab-title marker: best effort, never worth a wait

    class PageDidNotAnswer(_base_timeout, RuntimeError):
        """A TimeoutError like the harness's own, and a RuntimeError like _runtime_evaluate's."""

    def _budget():
        budget = PAGE_BUDGET_S
        if _deadline:
            budget = min(budget, _deadline - time.time() - 2.0)
        return max(1.0, budget)

    def _quick(req):
        try:
            return _orig_send(req, response_timeout=3.0)
        except Exception:
            return None

    def _explain(req, waited):
        what = req.get("method") or req.get("meta") or "a request"
        tab = _quick({"meta": "current_tab"})
        pending = _quick({"meta": "pending_dialog"})
        if tab is None and pending is None:
            return (f"Neither the page nor the browser answered {what} in {waited:g}s. The Mac is probably "
                    "under heavy load, or the browser is stuck. Wait a little and call again; if it keeps "
                    "happening, report it rather than retrying in a loop.")
        if not req.get("method") or what.startswith(("Target.", "Browser.")):  # the page was never asked
            return (f"The browser took longer than {waited:g}s to answer {what}, but it is answering now: the "
                    "Mac is probably under heavy load. Call again.")
        url = (tab or {}).get("url") or "the attached tab"
        msg = (f"The page {url} did not answer {what} in {waited:g}s. The browser itself is fine (it answered "
               "straight away); the page is busy, has a dialog open, or the Mac is under heavy load.")
        dialog = (pending or {}).get("dialog")
        if dialog:
            text = str(dialog.get("message") or "")[:200]
            msg += (f" A {dialog.get('type') or 'JavaScript'} dialog is open on it ({text!r}), and the page is "
                    "frozen until it is answered: cdp('Page.handleJavaScriptDialog', accept=False) dismisses it.")
        else:
            msg += (" Wait a few seconds and try again. If it stays silent, close_tab() and open the page again "
                    "with new_tab(url); do not keep opening tabs on top of a stuck one.")
        return msg

    def _record_tab(target_id):
        if not (_ledger_path and target_id):
            return
        try:
            try:
                with open(_ledger_path, encoding="utf-8") as fh:
                    tabs = json.load(fh)
            except (OSError, ValueError):
                tabs = []
            tabs = [t for t in tabs if isinstance(t, dict) and t.get("id") != target_id]
            tabs.append({"id": target_id, "at": time.time()})
            tmp = _ledger_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(tabs[-200:], fh)
            os.replace(tmp, _ledger_path)
        except OSError:
            pass

    def _send(req, response_timeout=_default):
        cosmetic = _marker in str((req.get("params") or {}).get("expression") or "")
        if response_timeout == _default and not cosmetic:
            response_timeout = _budget()
        try:
            r = _orig_send(req, response_timeout=response_timeout)
        except PageDidNotAnswer:
            raise
        except _base_timeout:
            if cosmetic:
                raise
            raise PageDidNotAnswer(_explain(req, response_timeout)) from None
        if req.get("method") == "Target.createTarget":
            _record_tab((r.get("result") or {}).get("targetId"))
        return r

    def _runtime_evaluate(expression, session_id=None, await_promise=False):
        try:
            return _orig_eval(expression, session_id=session_id, await_promise=await_promise)
        except PageDidNotAnswer:
            raise
        except RuntimeError as e:
            if isinstance(e.__cause__, PageDidNotAnswer):
                raise PageDidNotAnswer(f"{e.__cause__} (expression: {_h._js_snippet(expression)})") from None
            raise

    def _cap_tabs():
        try:
            cap = int(os.environ.get("HERMES_BU_TAB_CAP") or 0)
        except ValueError:
            cap = 0
        if cap <= 0 or not _ledger_path:
            return
        try:
            with open(_ledger_path, encoding="utf-8") as fh:
                tabs = [t for t in json.load(fh) if isinstance(t, dict) and t.get("id")]
        except (OSError, ValueError):
            return
        try:
            infos = _orig_send({"method": "Target.getTargets", "params": {}}, response_timeout=5.0)
            live = {t.get("targetId") for t in infos["result"]["targetInfos"] if t.get("type") == "page"}
            attached = _orig_send({"meta": "current_tab"}, response_timeout=5.0).get("targetId")
        except Exception:
            return  # never close anything without knowing which tab is in use
        tabs = [t for t in tabs if t["id"] in live]
        excess = len(tabs) - cap
        closed = []
        for t in list(tabs):
            if excess <= 0:
                break
            if t["id"] == attached:
                continue
            try:
                _orig_send({"method": "Target.closeTarget", "params": {"targetId": t["id"]}}, response_timeout=5.0)
            except Exception:
                break
            closed.append(t["id"])
            excess -= 1
        tabs = [t for t in tabs if t["id"] not in closed]
        try:
            tmp = _ledger_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(tabs, fh)
            os.replace(tmp, _ledger_path)
        except OSError:
            pass
        if closed:
            print(f"hermes: closed {len(closed)} old tab(s) Hermes had opened (keeps the newest {cap})",
                  file=sys.stderr)

    def _record_current_tab():
        # The tab this call left the harness attached to: the page the model is working on. The vault
        # binds a login to THIS tab rather than to whichever tab happens to hold a password field
        # (Moe ticket #17). Daemon-local meta call: no page is asked, so a busy page cannot stall it.
        path = os.environ.get("HERMES_BU_CURRENT_TAB") or ""
        if not path:
            return
        try:
            tab = _orig_send({"meta": "current_tab"}, response_timeout=3.0) or {}
            record = {"targetId": tab.get("targetId") or "", "url": tab.get("url") or "", "at": time.time()}
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            os.replace(tmp, path)
        except Exception:
            pass

    _h._send, _h._runtime_evaluate = _send, _runtime_evaluate
    _h.PageDidNotAnswer = PageDidNotAnswer
    _h._hermes_patched = True
    atexit.register(_cap_tabs)
    atexit.register(_record_current_tab)  # atexit is LIFO: recorded before any tab is closed
_hermes_patch_harness()
del _hermes_patch_harness
'''.replace("PAGE_BUDGET_S", repr(PAGE_BUDGET_S))


def harness_patch_preamble() -> str:
    """One line to put in front of the model's code, so its traceback line numbers move by one only."""
    return f"exec(compile({HARNESS_PATCH_SOURCE!r}, '<hermes-harness-patch>', 'exec'))\n"


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name or "default"))[:80] or "default"


def current_tab_path(task_id: Optional[str]) -> Optional[str]:
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home() / "cache" / "browser-use" / f"current-tab-{_safe(task_id or 'default')}.json")
    except Exception:
        return None


#: How long the harness's record of its tab is believed: a record from an old call may name a tab
#: the model has long since left.
CURRENT_TAB_FRESH_SECONDS = 30 * 60


def current_tab(task_id: Optional[str], *, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """``{"targetId", "url", "at"}``: the tab the last browser_exec call of ``task_id`` left the
    harness attached to, in Moe's own browser. None when unknown or stale."""
    path = current_tab_path(task_id)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    now = time.time() if now is None else now
    if not isinstance(data, dict) or not data.get("targetId"):
        return None
    at = data.get("at")
    if not isinstance(at, (int, float)) or not (0 <= now - float(at) <= CURRENT_TAB_FRESH_SECONDS):
        return None
    return data


def exec_env(env: dict, *, timeout_s: float, session: str, own_lane: bool, task_id: Optional[str] = None) -> None:
    """The env the patch reads: the call's deadline always; the tab cap, its ledger and the
    current-tab record only in Moe's own browser (the person's Chrome is theirs, whatever Hermes
    opened there, and the vault never fills there)."""
    env[ENV_DEADLINE] = f"{time.time() + float(timeout_s):.3f}"
    if not own_lane:
        return
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "cache" / "browser-use"
        d.mkdir(parents=True, exist_ok=True)
        path = current_tab_path(task_id)
        if path:
            env[ENV_CURRENT_TAB] = path
        # One ledger per daemon name AND browser: two tasks' default sessions can drive two browsers.
        browser = hashlib.sha1(str(env.get("BU_CDP_WS") or env.get("BU_CDP_URL") or "").encode()).hexdigest()[:10]
        env[ENV_TAB_LEDGER] = str(d / f"tabs-{session or 'default'}-{browser}.json")
        env[ENV_TAB_CAP] = str(TAB_CAP)
    except Exception as e:
        logger.debug("browser_exec tab ledger unavailable: %s", e)


def _dialog_tool_available() -> bool:
    try:
        from tools.browser_dialog_tool import _browser_dialog_check
        return bool(_browser_dialog_check())
    except Exception:
        return False


def exec_dialog_policy(policy: str) -> str:
    """The supervisor policy for a browser_exec browser. ``must_respond`` holds the page's JS thread
    until ``browser_dialog`` answers; when the model has no such tool that is a 300 s freeze nobody
    can end, so the dialog is dismissed at once and reported in the result instead."""
    from tools.browser_supervisor_dialogs import DIALOG_POLICY_AUTO_DISMISS, DIALOG_POLICY_MUST_RESPOND
    if policy == DIALOG_POLICY_MUST_RESPOND and not _dialog_tool_available():
        return DIALOG_POLICY_AUTO_DISMISS
    return policy


def dialog_report(task_id: Optional[str], since: float) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Dialogs the supervisor saw during this call: still open, or closed since ``since``."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY
        sup = SUPERVISOR_REGISTRY.get(task_id or "default")
        if sup is None:
            return None
        snap = sup.snapshot()
        pending = [d.to_dict() for d in snap.pending_dialogs]
        closed = [d.to_dict() for d in snap.recent_dialogs if d.closed_at >= since - 1]
    except Exception as e:
        logger.debug("browser_exec dialog report failed: %s", e)
        return None
    out = {k: v for k, v in (("open", pending), ("closed", closed)) if v}
    return out or None


def fidelity_unresumed() -> Optional[int]:
    """Targets the fidelity keeper gave up resuming, summed over every live keeper; None if unknown."""
    try:
        from tools.browser_tool_fidelity import read_statuses
        return sum(int(k.get("unresumed") or 0) for data in read_statuses() for k in (data.get("keepers") or []))
    except Exception as e:
        logger.debug("fidelity status unavailable: %s", e)
        return None


def annotate(result: Dict[str, Any], *, task_id: Optional[str], started: float,
             unresumed_before: Optional[int]) -> None:
    """Add what the harness cannot see to a browser_exec result: the supervisor's dialogs, and on a
    failure, new tabs the fidelity keeper left paused."""
    dialogs = dialog_report(task_id, started)
    if dialogs:
        result["dialogs"] = dialogs
        if dialogs.get("closed"):
            result["dialog_note"] = ("The page showed a dialog during this call and it was dismissed "
                                     "automatically (confirm() returned false, prompt() returned null). "
                                     "If the page needed a yes, it did not get one.")
    if result.get("success") is False and unresumed_before is not None:
        after = fidelity_unresumed()
        if after is not None and after > unresumed_before:
            result["paused_tabs"] = after - unresumed_before
            result["paused_tabs_note"] = (f"{after - unresumed_before} new tab(s) in this browser stayed paused at "
                                          "start during this call (the fidelity keeper could not resume them), so "
                                          "a page opened now may never answer. Retrying in a moment usually works; "
                                          "the keeper's log is logs/browser-fidelity-keeper.log.")
