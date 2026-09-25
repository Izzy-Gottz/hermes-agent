"""Where a recipient came from: the words and the lookups of this conversation, on disk.

A send gate can say *what* a call does, but not *who asked for it*. On
2026-09-25 a cron job told "send ONE WhatsApp message to Yisrael" (the owner)
called ``whatsapp_send(to="Moshe Finkelman")`` — the owner's grandfather, the
only WhatsApp contact named in the memory block of the system prompt. No
lookup preceded it; the name existed nowhere but in memory. The pre_tool_call
hook saw a well-formed send to a real chat and had no way to know that nobody
had asked for that person.

This module records the two things a recipient may legitimately come from,
per conversation, so the hook (a separate process — under the ``claude_code``
runtime it is spawned by the CLI's MCP server, which has no agent and no
conversation) can check it:

* ``words``   — what the person (or the job) said: every ``user`` message of the
  conversation, the current one last, with fenced ``<memory-context>`` blocks
  removed. Replaced at the start of every turn from the agent's own history.
* ``results`` — what tools returned during the conversation (contacts, chats,
  search results, files), appended as each call finishes. Results of the
  ``memory`` tool, and reads of the memory files themselves, are NOT recorded:
  memory is context, never a source of recipients.

The reader is Moe's ``tools/recipients.py``; the contract is this file's JSON
shape and :func:`ledger_path`. Nothing here decides anything — it only writes
down what happened, best effort, and never raises into the turn.

**Keys.** A ledger is keyed by :func:`current_key`: the ``HERMES_GROUNDING_KEY``
environment variable when set (the ``claude_code`` runtime mints one per
``ClaudeCodeSession`` and hands it to the MCP server, whose tool calls carry no
session id), else the Hermes session id. The hook computes the same key from
the same two sources.

**Privacy.** The file is 0600 under ``$HERMES_HOME/grounding``, beside the
transcripts Hermes already keeps; results are truncated, old ledgers are
pruned after :data:`STALE_SECONDS`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

GROUNDING_KEY_ENV = "HERMES_GROUNDING_KEY"
LEDGER_DIRNAME = "grounding"
VERSION = 1

#: How much of the conversation is kept. The words are short; a lookup result
#: can be long, so each is truncated and only the most recent are kept.
MAX_WORDS = 60
MAX_WORD_CHARS = 8000
MAX_RESULTS = 80
MAX_RESULT_CHARS = 64 * 1024
STALE_SECONDS = 2 * 24 * 3600

#: Tools whose results are memory, not lookups. A recipient found only here
#: was picked from memory — exactly what grounding exists to refuse.
MEMORY_TOOLS = frozenset({"memory"})
#: Paths whose contents are the memory store, whatever tool reads them.
_MEMORY_PATH = re.compile(r"(^|/)(memories/|USER\.md$|MEMORY\.md$)", re.I)
_MEMORY_FENCE = re.compile(r"<\s*memory-context\s*>[\s\S]*?<\s*/\s*memory-context\s*>", re.I)


def current_key(session_id: Optional[str] = "", env: Optional[dict] = None) -> str:
    env = os.environ if env is None else env
    key = str(env.get(GROUNDING_KEY_ENV) or "").strip()
    return key or str(session_id or "").strip()


def ledger_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:  # pragma: no cover — only without hermes_constants
        home = Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes"))
    return home / LEDGER_DIRNAME


def ledger_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return ledger_dir() / f"{digest}.json"


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


def person_words(messages: Iterable[Any], current: Any = None) -> List[str]:
    """The ``user`` turns of a conversation as plain text, oldest first, the
    current one last, with injected memory blocks removed."""
    out: List[str] = []
    for m in messages or ():
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        text = _MEMORY_FENCE.sub(" ", _text_of(m.get("content"))).strip()
        if text:
            out.append(text[:MAX_WORD_CHARS])
    cur = _MEMORY_FENCE.sub(" ", _text_of(current)).strip() if current is not None else ""
    if cur and (not out or out[-1] != cur[:MAX_WORD_CHARS]):
        out.append(cur[:MAX_WORD_CHARS])
    return out[-MAX_WORDS:]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
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


def record_words(key: str, messages: Iterable[Any], current: Any = None) -> None:
    """Replace the conversation's words with ``messages``' user turns. Never raises."""
    try:
        words = person_words(messages, current)
        if not key or not words:
            return

        def _set(data: dict) -> None:
            data["words"] = words
            data["current"] = words[-1]

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
    """Append one tool result to the conversation's lookups. Never raises."""
    try:
        if not key or not tool_name or is_memory_read(tool_name, args):
            return
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        text = (text or "").strip()
        if not text:
            return

        def _add(data: dict) -> None:
            rows = data.get("results") if isinstance(data.get("results"), list) else []
            rows.append({"tool": tool_name, "text": text[:MAX_RESULT_CHARS], "at": time.time()})
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
            try:
                if now - entry.stat().st_mtime > STALE_SECONDS:
                    entry.unlink()
            except OSError:
                continue
    except OSError:
        return
