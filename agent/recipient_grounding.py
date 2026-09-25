"""Where a recipient came from: who said what, and what the lookups returned, on disk.

A send gate can say *what* a call does, but not *who asked for it*. On
2026-09-25 a cron job told "send ONE WhatsApp message to Yisrael" (the owner)
called ``whatsapp_send(to="Moshe Finkelman")`` — a name that existed only in the
memory block of the system prompt. The pre_tool_call hook (a separate process;
under ``claude_code`` a child of the CLI's MCP server, with no conversation) had
no way to know that nobody had asked for that person.

This module writes down, per conversation, what the gate needs to answer that —
including WHO said each thing, because "a user-role message" is not "the
person":

* ``words`` — one entry per thing said, each with an ``origin``:

  - ``person``: a message a human sent in this conversation, with its ``sender``
    (``local`` for the Mac's own surfaces, or the chat platform and user id).
    Moe's gate decides whether that sender is the OWNER (tools/recipients.py);
    only the owner's words can ground a recipient.
  - ``model``: text a model wrote that now reads like an instruction — a cron
    job's prompt (with its skills and ``context_from``), a delegated goal. It
    can only add refusals and mismatches, never ground.
  - ``other``: a turn nobody can vouch for (a webhook, the app's background
    notes, an unknown origin).

  Compaction summaries, injected ``<memory-context>`` blocks and replayed
  history are never read back as words: person words ACCUMULATE here, one
  turn at a time, as each turn starts.
* A job's or a delegated helper's person words are those of the turn that
  CREATED it (``grounding_words`` on the job; inherited by the child agent), so
  a job the model wrote grounds only what the person said when asking for it.
* ``results`` — what tools returned, with the call's arguments, so the gate can
  tell a real lookup (contacts, chats, mail search) from a web page or a grep.
  The memory tools and reads of the memory files are never recorded.

The reader is Moe's ``tools/recipients.py``; the contract is this JSON shape,
:func:`ledger_path` and the ``.enabled`` marker. With the marker present, a
missing or unreadable ledger is a refusal on Moe's side — so the marker is
written only by the code that writes ledgers.

**Keys.** :func:`current_key`: ``HERMES_GROUNDING_KEY`` when set (the
``claude_code`` runtime mints one per session and hands it to the MCP server;
the codex runtime hands the session id to its MCP server the same way), else
the Hermes session id — which the native loop's tool subprocesses also carry,
as ``HERMES_SESSION_ID``.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

GROUNDING_KEY_ENV = "HERMES_GROUNDING_KEY"
LEDGER_DIRNAME = "grounding"
ENABLED_MARKER = ".enabled"
VERSION = 2

MAX_WORDS = 80
MAX_WORD_CHARS = 8000
MAX_RESULTS = 80
MAX_RESULT_CHARS = 64 * 1024
MAX_ARGS_CHARS = 2000
STALE_SECONDS = 2 * 24 * 3600

#: Tools whose results are memory, not lookups.
MEMORY_TOOLS = frozenset({"memory", "session_search"})
_MEMORY_PATH = re.compile(r"(^|/)(memories/|USER\.md$|MEMORY\.md$|SOUL\.md$)", re.I)
#: A fenced memory block — closed, or left open to the end of the text.
_MEMORY_FENCE = re.compile(
    r"<\s*memory-context\s*>(?:[\s\S]*?<\s*/\s*memory-context\s*>|[\s\S]*\Z)", re.I)

#: The person words of the turn now running (set by :func:`record_turn`), so a
#: job or a helper created during it can carry them. None = no turn running.
_TURN_PERSON_WORDS: contextvars.ContextVar = contextvars.ContextVar(
    "hermes_grounding_turn_person_words", default=None)


# ── keys and paths ───────────────────────────────────────────────────────────

def current_key(session_id: Optional[str] = "", env: Optional[dict] = None) -> str:
    env = os.environ if env is None else env
    key = str(env.get(GROUNDING_KEY_ENV) or "").strip()
    return key or str(session_id or "").strip()


def ledger_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:  # pragma: no cover
        home = Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    return home / LEDGER_DIRNAME


def ledger_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return ledger_dir() / f"{digest}.json"


def _ensure_enabled(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    marker = directory / ENABLED_MARKER
    if not marker.exists():
        fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write("recipient grounding ledgers are written here (version %d)\n" % VERSION)


# ── who is speaking ──────────────────────────────────────────────────────────

_LOCAL_SURFACES = frozenset({"cli", "tui", "desktop", "local"})
_CHAT_PLATFORMS = frozenset({
    "telegram", "whatsapp", "whatsapp_cloud", "signal", "discord", "slack", "matrix",
    "mattermost", "bluebubbles", "imessage", "sms", "email",
})


def turn_sender() -> Dict[str, Any]:
    """Who sent the message that started the turn now running.

    ``{"kind": "local"}`` — the person at this machine (CLI, desktop, the Memoe
    app's own person-started turns). ``{"kind": "chat", platform, user_id, …}`` —
    somebody on a chat platform (a relayed message included); whether that is
    the OWNER is Moe's call, against its owner record. ``job``, ``helper``,
    ``background`` and ``unknown`` are nobody whose words ground a recipient."""
    try:
        from gateway.session_context import (
            TURN_ORIGIN_BACKGROUND, TURN_ORIGIN_PERSON, get_session_env, get_turn_origin,
        )
    except Exception:
        return {"kind": "unknown"}

    def env(name: str) -> str:
        try:
            return str(get_session_env(name, "") or "").strip()
        except Exception:
            return ""

    if env("HERMES_CRON_SESSION").lower() in ("1", "true", "yes"):
        return {"kind": "job"}
    try:
        from agent.delegation_context import is_delegated_child_context
        if is_delegated_child_context():
            return {"kind": "helper"}
    except Exception:
        pass
    if os.environ.get("HERMES_KANBAN_TASK"):
        return {"kind": "background"}
    platform = env("HERMES_SESSION_PLATFORM").lower()
    source = env("HERMES_SESSION_SOURCE").lower()
    if platform == "api_server":
        origin = get_turn_origin()
        if origin == TURN_ORIGIN_PERSON:
            return {"kind": "local", "platform": "api_server"}
        if origin == TURN_ORIGIN_BACKGROUND:
            return {"kind": "background", "platform": "api_server"}
        return {"kind": "unknown", "platform": "api_server"}
    if platform in _LOCAL_SURFACES or (not platform and source in _LOCAL_SURFACES):
        return {"kind": "local", "platform": platform or source}
    if platform in _CHAT_PLATFORMS:
        return {"kind": "chat", "platform": platform,
                "user_id": env("HERMES_SESSION_USER_ID"),
                "user_id_alt": env("HERMES_SESSION_USER_ID_ALT"),
                "chat_id": env("HERMES_SESSION_CHAT_ID"),
                "chat_type": env("HERMES_SESSION_CHAT_TYPE")}
    if not platform and not source:
        # No gateway session at all: the CLI's one-shot, a script driving
        # AIAgent directly — a person at this machine.
        return {"kind": "local", "platform": ""}
    return {"kind": "unknown", "platform": platform or source}


# ── text ─────────────────────────────────────────────────────────────────────

def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in (None, "text", "input_text"):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def clean(text: Any) -> str:
    """Plain text with injected memory blocks removed — closed or not."""
    return _MEMORY_FENCE.sub(" ", _text_of(text)).strip()[:MAX_WORD_CHARS]


def person_words_now() -> List[Dict[str, Any]]:
    """The person words of the turn now running (empty outside a turn)."""
    value = _TURN_PERSON_WORDS.get()
    return [dict(w) for w in value] if isinstance(value, list) else []


def in_turn() -> bool:
    return _TURN_PERSON_WORDS.get() is not None


def words_for_new_task(prompt: Any = None) -> List[Dict[str, Any]]:
    """``grounding_words`` for a job or goal being created now.

    Inside a turn: that turn's person words — the model writes the job, the
    person's words are what may ground it. Outside any turn (the CLI, the
    dashboard, an app writing a job for the person) a person at this machine
    typed the prompt, and it is recorded as theirs."""
    if in_turn():
        return person_words_now()
    text = clean(prompt)
    return [{"text": text, "origin": "person", "sender": {"kind": "local"}}] if text else []


# ── the ledger ───────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _update(key: str, mutate) -> None:
    if not key:
        return
    path = ledger_path(key)
    _ensure_enabled(path.parent)
    lock_path = str(path) + ".lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        data = _load(path)
        if data.get("version") != VERSION:
            data = {"version": VERSION, "words": [], "results": []}
        mutate(data)
        data["updated"] = time.time()
        tmp_fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ledger.")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    finally:
        os.close(fd)


def _same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a.get("text") == b.get("text") and a.get("origin") == b.get("origin")


def turn_words(agent: Any, current: Any) -> List[Dict[str, Any]]:
    """This turn's word entries, each with its origin."""
    override = getattr(agent, "_grounding_override", None)
    if isinstance(override, dict):
        out = [dict(w) for w in override.get("person") or []
               if isinstance(w, dict) and w.get("text") and w.get("origin") == "person"]
        for text in override.get("model") or []:
            t = clean(text)
            if t:
                out.append({"text": t, "origin": "model"})
        return out
    text = clean(current)
    if not text:
        return []
    sender = turn_sender()
    kind = sender.get("kind")
    origin = "person" if kind in ("local", "chat") else ("model" if kind in ("job", "helper") else "other")
    return [{"text": text, "origin": origin, "sender": sender}]


def record_turn(agent: Any, keys: Iterable[str], current: Any) -> None:
    """File this turn's words under every key the gate may look it up by, and
    make its person words available to anything the turn creates. Person words
    accumulate across the conversation; everything else is this turn's only.
    Never raises."""
    try:
        new = turn_words(agent, current)
        keys = [k for k in dict.fromkeys(keys) if k]
        persons: List[Dict[str, Any]] = []
        for key in keys:
            prior = _load(ledger_path(key))
            if prior.get("version") == VERSION:
                for w in prior.get("words") or []:
                    if isinstance(w, dict) and w.get("origin") == "person" \
                            and not any(_same(w, s) for s in persons):
                        persons.append(w)
        for w in new:
            if w.get("origin") == "person" and not any(_same(w, s) for s in persons):
                persons.append(w)
        persons = persons[-MAX_WORDS:]
        words = persons + [w for w in new if w.get("origin") != "person"]
        _TURN_PERSON_WORDS.set(persons)
        try:
            agent._grounding_person_words = persons
        except Exception:
            pass
        current_entry = new[-1] if new else None

        def _set(data: dict) -> None:
            data["words"] = words
            data["current"] = current_entry

        for key in keys:
            _update(key, _set)
        _prune_sometimes()
    except Exception:
        return


def is_memory_read(tool_name: str, args: Any) -> bool:
    if tool_name in MEMORY_TOOLS:
        return True
    if isinstance(args, dict):
        for k in ("path", "file", "file_path", "filename"):
            v = args.get(k)
            if isinstance(v, str) and _MEMORY_PATH.search(v):
                return True
    return False


def record_result(key: str, tool_name: str, args: Any, result: Any) -> None:
    """Append one tool result, with the arguments it was called with. Never raises."""
    try:
        if not key or not tool_name or is_memory_read(tool_name, args):
            return
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        text = (text or "").strip()
        if not text:
            return
        try:
            args_text = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False, default=str)
        except Exception:
            args_text = "{}"

        def _add(data: dict) -> None:
            rows = data.get("results") if isinstance(data.get("results"), list) else []
            rows.append({"tool": tool_name, "args": args_text[:MAX_ARGS_CHARS],
                         "text": text[:MAX_RESULT_CHARS], "at": time.time()})
            data["results"] = rows[-MAX_RESULTS:]

        _update(key, _add)
    except Exception:
        return


_last_prune = 0.0


def _prune_sometimes() -> None:
    global _last_prune
    now = time.time()
    if now - _last_prune < 3600:
        return
    _last_prune = now
    try:
        for entry in ledger_dir().iterdir():
            if entry.name == ENABLED_MARKER:
                continue
            try:
                if now - entry.stat().st_mtime > STALE_SECONDS:
                    entry.unlink()
            except OSError:
                continue
    except OSError:
        return
