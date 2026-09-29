"""What the model is told about background helpers it (or its owner) started, read from the durable ledger.

Moe, 2026-09-29. Seven background helpers (``deleg_fe9575cd``) finished at 16:32; a
``deliver_detached_completion`` plugin took the completion for its own inbox, the inbox was never read, and
nothing reached the model. Later, in a new conversation, ``delegate_task(action="list")`` read only the in-memory
live registry — count 0, "children that already finished have delivered their results" — and the model told the
owner it had never started any. The work (52 KB of it) sat in ``async_delegations`` the whole time.

Three things read that ledger here:

* ``turn_preamble`` — the next client turn of the originating session hands over any result a plugin took but
  the model has not seen (after a grace window, so the plugin's own push can land first), and a fresh or
  respawned conversation is told about recent helpers from another conversation of the same owner;
* ``list_view`` — ``delegate_task(action="list")`` reports this conversation's recent finished helpers and the
  owner's from other conversations, with whether each result actually reached the model;
* ``result_view`` — ``delegate_task(action="result")`` reads one of them in full.

"Reached the model" is ``delivery_state == 'delivered'``; ``plugin`` means a plugin has it and the model does
not. Everything here is best-effort: a ledger that cannot be read yields nothing, never a failed turn.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

RECENT_WINDOW_SECONDS = 24 * 60 * 60
# How long a plugin that took a completion gets to put it in front of the model itself before the next client
# turn hands it over instead. Moe's app runs the inbox turn when it is next idle — usually seconds.
PLUGIN_GRACE_SECONDS = 180.0
_RESULT_TEXT_CAP = 60_000
_GOAL_CAP = 160
_SUMMARY_CAP = 200
_MAX_PATHS = 8
_MAX_ROWS = 50

# The header every async completion notification starts with (tools/process_registry_notifications.py).
MARKER_RE = re.compile(r"\[ASYNC DELEGATION (?:BATCH )?COMPLETE — (deleg_[A-Za-z0-9_-]+)\]")
_PATH_RE = re.compile(r"(?<![\w/.~])((?:~|/(?:Users|home|tmp|private|var|opt|Volumes|mnt|data|srv))/[^\s`'\"<>()\[\]{},;]+)")
_LIVE = ("running", "finalizing", "stalling")


def _ledger_rows(*, since: float, session_ids: Sequence[str], owner_key: str) -> List[Dict[str, Any]]:
    ids = [s for s in dict.fromkeys(str(x or "").strip() for x in session_ids) if s]
    owner_key = str(owner_key or "").strip()
    if not ids and not owner_key:
        return []
    from tools.async_delegation import _DB_LOCK, _transaction
    clauses, params = [], []
    if ids:
        marks = ",".join("?" * len(ids))
        clauses.append(f"parent_session_id IN ({marks}) OR origin_session_id IN ({marks})")
        params += ids + ids
    if owner_key:
        clauses.append("origin_session = ?")
        params.append(owner_key)
    sql = (f"""SELECT delegation_id, origin_session, origin_session_id, parent_session_id, state, dispatched_at,
                      completed_at, delivery_state, delivered_via, task_json, event_json, result_json
               FROM async_delegations WHERE ({' OR '.join(clauses)})
                 AND COALESCE(completed_at, dispatched_at) >= ?
               ORDER BY dispatched_at DESC LIMIT {_MAX_ROWS}""")
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(sql, (*params, since)).fetchall()
    keys = ("delegation_id", "origin_session", "origin_session_id", "parent_session_id", "state", "dispatched_at",
            "completed_at", "delivery_state", "delivered_via", "task_json", "event_json", "result_json")
    out = []
    for row in rows:
        rec = dict(zip(keys, row))
        in_this = bool(ids) and (rec["parent_session_id"] in ids or rec["origin_session_id"] in ids)
        if not in_this and not (owner_key and rec["origin_session"] == owner_key):
            continue
        rec["this_conversation"] = in_this
        out.append(rec)
    return out


def _loads(raw: Optional[str]) -> Any:
    try:
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _clip(text: Any, cap: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _when(ts: Optional[float]) -> Optional[str]:
    if not isinstance(ts, (int, float)):
        return None
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _goals(rec: Dict[str, Any]) -> List[str]:
    task = _loads(rec.get("task_json")) or {}
    event = _loads(rec.get("event_json")) or {}
    goals = task.get("goals") or event.get("goals")
    if isinstance(goals, list) and goals:
        return [str(g) for g in goals]
    goal = task.get("goal") or event.get("goal")
    return [str(goal)] if goal else []


def _results(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = _loads(rec.get("result_json")) or {}
    if isinstance(result.get("results"), list):
        return [r for r in result["results"] if isinstance(r, dict)]
    return [result] if result else []


def _paths(texts: Iterable[str]) -> List[str]:
    found: List[str] = []
    for text in texts:
        for match in _PATH_RE.findall(str(text or "")):
            path = match.rstrip(".:!?")
            if path not in found:
                found.append(path)
            if len(found) >= _MAX_PATHS:
                return found
    return found


def reached_model(rec: Dict[str, Any]) -> bool:
    return rec.get("delivery_state") == "delivered"


def _view(rec: Dict[str, Any]) -> Dict[str, Any]:
    goals = _goals(rec)
    results = _results(rec)
    tasks = []
    for i, r in enumerate(results):
        idx = r.get("task_index", i)
        goal = goals[idx] if isinstance(idx, int) and 0 <= idx < len(goals) else ""
        summary = next((ln for ln in str(r.get("summary") or r.get("error") or "").splitlines() if ln.strip()), "")
        tasks.append({"goal": _clip(goal, _GOAL_CAP), "status": r.get("status"),
                      "summary_line": _clip(summary, _SUMMARY_CAP)})
    live = rec.get("state") in _LIVE
    view = {
        "delegation_id": rec["delegation_id"],
        "state": rec.get("state"),
        "goals": [_clip(g, _GOAL_CAP) for g in goals],
        "started_at": _when(rec.get("dispatched_at")),
        "completed_at": None if live else _when(rec.get("completed_at")),
        "delivery_state": rec.get("delivery_state"),
        "reached_model": reached_model(rec),
        "tasks": tasks,
        "result_paths": _paths(list(goals) + [str(r.get("summary") or "") for r in results]),
    }
    if not rec.get("this_conversation"):
        view["from_another_conversation"] = True
    if not live and not view["reached_model"]:
        view["how_to_read"] = f"delegate_task(action='result', subagent_id='{rec['delegation_id']}')"
    return view


def list_view(session_ids: Sequence[str], owner_key: str, *, now: Optional[float] = None) -> Dict[str, List[Dict]]:
    """``{"finished": [...this conversation...], "other_conversations": [...same owner, last 24 h...]}``.
    Live children of this conversation are the live registry's to report; running rows from OTHER conversations
    are included here because nothing else can show them."""
    now = time.time() if now is None else now
    try:
        rows = _ledger_rows(since=now - RECENT_WINDOW_SECONDS, session_ids=session_ids, owner_key=owner_key)
    except Exception:
        logger.debug("delegation recall: ledger unreadable for list", exc_info=True)
        return {"finished": [], "other_conversations": []}
    finished = [_view(r) for r in rows if r["this_conversation"] and r.get("state") not in _LIVE]
    others = [_view(r) for r in rows if not r["this_conversation"]]
    return {"finished": finished, "other_conversations": others}


def result_view(delegation_id: str, session_ids: Sequence[str], owner_key: str) -> Optional[Dict[str, Any]]:
    """The full completion text of one finished helper this conversation (or its owner) started, or None.
    Reading it counts as the model having it."""
    try:
        rows = _ledger_rows(since=0.0, session_ids=session_ids, owner_key=owner_key)
    except Exception:
        logger.debug("delegation recall: ledger unreadable for result", exc_info=True)
        return None
    rec = next((r for r in rows if r["delegation_id"] == delegation_id), None)
    if rec is None:
        return None
    if rec.get("state") in _LIVE:
        return {"delegation_id": delegation_id, "state": rec.get("state"),
                "note": "Still running; its result arrives when it finishes."}
    text = _completion_text(rec)
    if not text:
        return None
    if not reached_model(rec):
        _mark(delegation_id, "result-read")
    return {"delegation_id": delegation_id, "state": rec.get("state"), "completed_at": _when(rec.get("completed_at")),
            "from_another_conversation": not rec["this_conversation"], "result": text}


def _completion_text(rec: Dict[str, Any]) -> str:
    event = _loads(rec.get("event_json"))
    if not isinstance(event, dict):
        return ""
    try:
        from tools.process_registry_notifications import format_process_notification
        text = format_process_notification(event) or ""
    except Exception:
        logger.debug("delegation recall: could not format %s", rec.get("delegation_id"), exc_info=True)
        return ""
    if len(text) > _RESULT_TEXT_CAP:
        text = text[:_RESULT_TEXT_CAP] + "\n[… truncated; the full result is in the files named above …]"
    return text


def _mark(delegation_id: str, via: str) -> bool:
    try:
        from tools.async_delegation import mark_seen_by_model
        return mark_seen_by_model(delegation_id, via)
    except Exception:
        logger.debug("delegation recall: could not mark %s seen", delegation_id, exc_info=True)
        return False


def _helpers_count(rec: Dict[str, Any]) -> int:
    return max(1, len(_goals(rec)))


def _other_conversation_line(rows: List[Dict[str, Any]]) -> str:
    n = sum(_helpers_count(r) for r in rows)
    items = []
    for rec in rows:
        goals = _goals(rec)
        task = _loads(rec.get("task_json")) or {}
        title = _clip(task.get("goal") or (goals[0] if goals else "a background task"), _GOAL_CAP)
        if rec.get("state") in _LIVE:
            items.append(f"{title} — still running (started {_when(rec.get('dispatched_at'))})")
            continue
        paths = _paths(goals + [str(r.get("summary") or "") for r in _results(rec)])
        where = (", ".join(paths[:4]) + "; full result: " if paths else "")
        items.append(f"{title} — finished at {_when(rec.get('completed_at'))}; results: {where}"
                     f"delegate_task(action='result', subagent_id='{rec['delegation_id']}')")
    noun = "helper" if n == 1 else "helpers"
    return f"You have {n} background {noun} from a recent conversation: " + "; ".join(items) + "."


def turn_preamble(session_ids: Sequence[str], owner_key: str, user_text: str, *, fresh: bool,
                  now: Optional[float] = None, grace: float = PLUGIN_GRACE_SECONDS) -> str:
    """Background-helper context to put in front of this turn's user message, or ``""``.

    * A marker for one of this conversation's helpers in ``user_text`` is the plugin's own delivery arriving:
      mark it seen. If the fallback already handed it over, say so, so it is not reported twice.
    * Any other result of this conversation that a plugin took and the model never saw, older than ``grace``,
      is handed over in full — exactly once (the ledger transition is the lock).
    * ``fresh`` (a new or respawned model process): one line about the owner's helpers from other
      conversations in the last 24 h that are running or whose results never reached the model.
    """
    now = time.time() if now is None else now
    try:
        rows = _ledger_rows(since=now - RECENT_WINDOW_SECONDS, session_ids=session_ids, owner_key=owner_key)
    except Exception:
        logger.debug("delegation recall: ledger unreadable for preamble", exc_info=True)
        return ""
    arriving = set(MARKER_RE.findall(str(user_text or "")))
    parts: List[str] = []
    for rec in rows:
        did = rec["delegation_id"]
        if did not in arriving:
            continue
        if rec.get("delivery_state") == "plugin":
            _mark(did, "plugin-turn")
        elif rec.get("delivered_via") in ("fallback", "result-read"):
            parts.append(f"(Hermes note, not from the user: the helper result {did} below was already given to you "
                         "on an earlier turn. If you have told the person, do not repeat it; at most acknowledge it.)")
    for rec in rows:
        did = rec["delegation_id"]
        if (not rec["this_conversation"] or did in arriving or rec.get("delivery_state") != "plugin"
                or (rec.get("completed_at") or 0) > now - grace):
            continue
        text = _completion_text(rec)
        if text and _mark(did, "fallback"):
            parts.append(
                "(Hermes note, not from the user: a background helper you started earlier in this conversation "
                f"finished at {_when(rec.get('completed_at'))}, and its result never reached you. It is below. Tell "
                "the person the outcome briefly when it fits; do not start the job again.)\n" + text)
    if fresh:
        others = [r for r in rows if not r["this_conversation"] and r["delegation_id"] not in arriving
                  and (r.get("state") in _LIVE or not reached_model(r))]
        if others:
            parts.append("(Hermes note, not from the user: " + _other_conversation_line(others) + ")")
    return "\n\n".join(parts)
