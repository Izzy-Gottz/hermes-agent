"""Write/delete guards for ``skill_manage``. Every guard returns ``None`` when the
operation may proceed, else a refusal (error dict or message). Origin-owned state
(``_find_skill``, ``_skills_dir``) is reached lazily via ``tools.skill_manager_tool``
so test patches keep working."""

import contextvars as _ctxvars
import logging
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("tools.skill_manager_tool")


def _refusal(message: str, **extra: Any) -> Dict[str, Any]:
    return {"success": False, "error": message, **extra}


def _is_background_review() -> bool:
    """True inside the autonomous curator review fork; False on any lookup failure."""
    try:
        from tools.skill_provenance import is_background_review
        return bool(is_background_review())
    except Exception:
        return False


def _resolved_str(path: Path) -> str:
    with suppress(Exception):
        return str(path.resolve())
    return str(path)


class _BackgroundReviewReadMarks:
    """Read marks shared by copied tool contexts within one review run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths: set[str] = set()

    def add(self, path: str) -> None:
        with self._lock:
            self._paths.add(path)

    def contains(self, path: str) -> bool:
        with self._lock:
            return path in self._paths


_background_review_read_paths: "_ctxvars.ContextVar[Optional[_BackgroundReviewReadMarks]]" = (
    _ctxvars.ContextVar("background_review_read_paths", default=None))


def mark_background_review_skill_read(path: Path) -> None:
    """Record that the active background-review fork has read a skill file. The fork must not
    patch content it only inferred from the transcript: skill_view/read_file call this, and
    the write guards require the mark."""
    if not _is_background_review():
        return
    if (marks := _background_review_read_paths.get()) is None:
        _background_review_read_paths.set(marks := _BackgroundReviewReadMarks())
    marks.add(_resolved_str(path))


def _background_review_has_read(path: Path) -> bool:
    marks = _background_review_read_paths.get()
    return marks is not None and marks.contains(_resolved_str(path))


def _reset_background_review_read_marks() -> None:
    """Start a fresh, isolated read set for the current review context."""
    _background_review_read_paths.set(_BackgroundReviewReadMarks())


def _resolved_roots(skill_path: Path):
    """``(resolved skill_path, [(root, resolved_root), ...])`` over every resolvable skills root."""
    from agent.skill_utils import get_all_skills_dirs
    try:
        resolved = skill_path.resolve()
    except OSError:
        resolved = skill_path
    roots = []
    for root in get_all_skills_dirs():
        with suppress(OSError):
            roots.append((root, root.resolve()))
    return resolved, roots


def _containing_skills_root(skill_path: Path) -> Path:
    """Skills root (local or external_dirs) containing ``skill_path``; local dir if none match."""
    from tools import skill_manager_tool as _smt
    resolved, roots = _resolved_roots(skill_path)
    return next((root for root, r in roots if resolved.is_relative_to(r)), _smt._skills_dir())


def _is_path_redirect(path: Path) -> bool:
    """Symlink or (Windows 3.12+) junction — either lets a poisoned tree redirect rmtree outside."""
    try:
        return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
    except OSError:
        return False


def _validate_delete_target(skill_dir: Path) -> Optional[str]:
    """Last-line guard before rmtree: even a poisoned tree must never delete (1) a path outside
    every known skills root, (2) a skills root itself, (3) a symlink/junction (rmtree follows it).

    ``_find_skill`` already restricts ``skill_dir`` to a real ``SKILL.md`` parent discovered by walking the
    skills roots, so the agent cannot inject an arbitrary path the way Kilo Code's HTTP endpoint could
    (their issue 11227: a built-in-skill sentinel resolved to the server cwd and a recursive delete wiped
    the user's entire working directory). This is the matching defense-in-depth for our agent-facing
    ``skill_manage`` delete path: even if discovery or a poisoned tree hands us a bad directory, never
    recursively delete See #11227.
    """
    if _is_path_redirect(skill_dir):
        return (f"Refusing to delete '{skill_dir}': the skill directory is a "
                f"symlink/junction. Remove the link target manually if intended.")
    try:
        skill_dir.resolve()
    except OSError as exc:
        return f"Refusing to delete '{skill_dir}': could not resolve path ({exc})."
    resolved, roots = _resolved_roots(skill_dir)
    for _root, root in roots:
        if resolved == root:
            return (f"Refusing to delete '{skill_dir}': resolves to the skills root "
                    f"itself, which would remove every installed skill.")
        if resolved.is_relative_to(root):
            return None
    return f"Refusing to delete '{skill_dir}': path does not resolve inside any known skills root."


def _is_pinned(name: str, what: str) -> Optional[bool]:
    """skill_usage pinned flag; None (logged at debug) when the record is unreadable."""
    try:
        from tools import skill_usage
        return bool(skill_usage.get_record(name).get("pinned"))
    except Exception:
        logger.debug("%s lookup failed for %s", what, name, exc_info=True)
        return None


def _pinned_guard(name: str) -> Optional[str]:
    """Refusal message if *name* is pinned or essential, else None. Pin only guards DELETION;
    patches/edits stay allowed. ESSENTIAL_SKILLS are permanently pinned (the system prompt
    references them). Best-effort: an unreadable sidecar lets the delete through."""
    try:
        from agent.skill_utils import ESSENTIAL_SKILLS
        if name in ESSENTIAL_SKILLS:
            return (
                f"Skill '{name}' is essential to Hermes (the agent's own "
                f"operating manual referenced by the system prompt) and "
                f"cannot be deleted. Patches and edits are still allowed.")
    except Exception:
        logger.debug("essential-guard lookup failed for %s", name, exc_info=True)
    if _is_pinned(name, "pinned-guard"):
        return (
            f"Skill '{name}' is pinned and cannot be deleted by skill_manage. Ask the user to "
            f"run `hermes curator unpin {name}` if they want to delete it. Patches and edits "
            f"are allowed on pinned skills; only deletion is blocked.")
    return None


def _background_review_delete_guard(name: str, skill_dir: Path) -> Optional[Dict[str, Any]]:
    """Refuse autonomous deletes of anything but curator-owned sediment. Content writes are not
    gated by ownership: the review fork exists to improve every skill it learns from, and every
    write is ledgered and reversible, while an archive removes a skill the user relies on."""
    if not _is_background_review():
        return None
    refuse = "Refusing background curator delete for"
    if _is_pinned(name, "pinned skill guard"):
        return _refusal(
            f"{refuse} pinned skill '{name}': pinned skills "
            f"are off-limits to autonomous maintenance. Ask the user to run `hermes curator "
            f"unpin {name}` if they want it changed.")
    try:
        from agent.skill_utils import is_external_skill_path
        if is_external_skill_path(skill_dir):
            return _refusal(
                f"{refuse} skill '{name}': the skill lives in skills.external_dirs, which are "
                f"externally owned and read-only to autonomous curation.")
    except Exception:
        logger.debug("external skill guard lookup failed for %s", name, exc_info=True)
    try:
        from tools import skill_usage
        for predicate, label in (
            (skill_usage.is_protected_builtin, "protected built-in"),
            (skill_usage.is_hub_installed, "hub-installed"),
            (skill_usage.is_bundled, "bundled")):
            if predicate(name):
                return _refusal(f"{refuse} {label} skill '{name}'.")
        # Not curator-managed (no `created_by: "agent"`) => user-owned, never archived
        # autonomously. A MISSING record and `created_by: null` must resolve identically:
        # keying on presence let the guard's own bookkeeping flip the verdict (#67140).
        usage_rec = skill_usage.load_usage().get(name)
        if not skill_usage._is_curator_managed_record(usage_rec):
            _detail = (f"created_by={usage_rec.get('created_by')!r}" if isinstance(usage_rec, dict)
                       else "no usage record")
            return _refusal(
                f"{refuse} skill '{name}': the skill is not "
                f"curator-managed ({_detail}). Only curator-managed skills may be archived "
                f"autonomously. Run `hermes curator adopt {name}` to opt it in.")
    except Exception:
        logger.warning("owned skill guard lookup failed for %s", name, exc_info=True)
        return _refusal(
            f"{refuse} skill '{name}': agent ownership could not "
            f"be verified because the provenance record is unavailable or unreadable.")
    return None


def _background_review_read_before_write_guard(
    name: str, target: Path, action: str, file_label: str) -> Optional[Dict[str, Any]]:
    """Require review forks to load the exact target before mutating it."""
    if not _is_background_review() or _background_review_has_read(target):
        return None
    return _refusal(
        f"Refusing background curator {action} for skill '{name}': the current {file_label} "
        f"content has not been loaded in this review turn. Call skill_view(name) for SKILL.md, or "
        f"skill_view(name, file_path=...) for a supporting file, then retry the write using the "
        f"content just returned.",
        _read_before_write_required=True)


def _background_review_preflight(action: str, name: str) -> Optional[Dict[str, Any]]:
    if action != "delete":
        return None
    from tools import skill_manager_tool as _smt
    existing = _smt._find_skill(name)
    return _background_review_delete_guard(name, existing["path"]) if existing else None


def _curator_consolidation_delete_guard(
    name: str, absorbed_into: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fail closed on unverified deletes during the curator consolidation pass. The fork's only
    legitimate delete is a consolidation declared via ``absorbed_into=<umbrella>`` (existence
    validated in ``_delete_skill``); the deterministic inactivity prune never calls skill_manage,
    so a bare delete here can only be the LLM pass pruning without evidence.

    A delete with no forwarding target — ``absorbed_into`` omitted (``None``) or empty (``""``) — is the
    fail-open behavior reported in #29912: the consolidation pass archived whole clusters of active skills
    with zero verified consolidations (``consolidated_this_run == 0``), leaving active automations pointing
    at names that no longer resolve. Refuse it; keep the skill active.
    """
    if not _is_background_review() or (isinstance(absorbed_into, str) and absorbed_into.strip()):
        return None
    return _refusal(
        f"Refusing background curator delete of skill '{name}': the consolidation pass may only "
        f"archive a skill it has absorbed into an umbrella. Pass absorbed_into=<umbrella> (the "
        f"umbrella must already exist) to record a verified consolidation. Pruning a skill with no "
        f"forwarding target is not permitted here — the deterministic inactivity prune handles "
        f"staleness archival separately. Keeping '{name}' active.",
        _fail_closed=True)


# ── Claims a skill should not keep unmarked ────────────────────────────────────
#
# A skill is read on every later turn, for every site. On 2026-09-24 the background review wrote into
# a launch skill that a passkey challenge "on the user's own device" is expected, "so ... wait for the
# user to approve it on their device" -- and Moe then told its owner to approve things on his phone
# that no site had sent to any phone. Whether a step waits on a device is a fact about ONE page at ONE
# moment, known only from what that page said (tools/browser_person_step.py: page_says).
#
# What this does, and what it does not:
# * Only writes the BACKGROUND REVIEW / curator makes are looked at. A skill the person dictates is
#   theirs, word for word.
# * It never refuses. A sentence the write ADDS that says a step waits on the person's phone or
#   device gets a visible marker after it, and the tool result carries a warning, so the reviewer can
#   reword it -- the rest of the lesson (often true and useful) is kept.
# * A sentence that QUOTES a page ("the page says \"check your phone\"") is left alone: that is a
#   grounded report, not a claim.
# * It is a word-level check. A reworded claim ("the owner's handset", "their iPad") passes, and a
#   true general line that happens to pair a device word with an approval word gets a marker. It is a
#   nudge at the one place such lines are written, not a proof; the grounding rule in the tool
#   descriptions and the soul is what the model is held to.

import re as _re
import threading as _threading

_DEVICE_WORDS = _re.compile(
    r"\b(?:on|from|with|via|using)\s+(?:the\s+|their\s+|your\s+|his\s+|her\s+)?"
    r"(?:(?:user|person|owner|customer|human)(?:'s|s'|s)?\s+)?(?:own\s+|other\s+|mobile\s+)?"
    r"(?:phone|device|iphone|android|mobile)\b", _re.I)
_APPROVAL_WORDS = _re.compile(r"\b(?:approv\w*|confirm\w*|tap\w*|wait\w*|accept\w*|allow\w*|complet\w*|finish\w*|"
                              r"authori[sz]\w*|respond\w*)\b", _re.I)
_SENTENCE_SPLIT = _re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTED = _re.compile(r"\"[^\"]{3,}\"|“[^”]{3,}”|'[^']{6,}'")
#: No colon (YAML reads "key: value" in a flow context), nothing a Markdown renderer interprets.
DEVICE_CLAIM_MARK = "[unverified]"

_FRONTMATTER = _re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", _re.S)
_FENCE = _re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", _re.S | _re.M)


def _segments(text: str) -> list:
    """``[(is_prose, chunk)]`` covering ``text``: YAML frontmatter and fenced code blocks are not prose,
    and are never read for claims or marked."""
    spans = []
    fm = _FRONTMATTER.match(text or "")
    if fm:
        spans.append((fm.start(), fm.end()))
    for m in _FENCE.finditer(text or ""):
        if not fm or m.start() >= fm.end():
            spans.append((m.start(), m.end()))
    out, pos = [], 0
    for a, b in sorted(spans):
        if a > pos:
            out.append((True, text[pos:a]))
        out.append((False, text[a:b]))
        pos = b
    if pos < len(text or ""):
        out.append((True, text[pos:]))
    return out


def _prose(text: str) -> str:
    return "\n".join(chunk for is_prose, chunk in _segments(text) if is_prose)


def device_approval_claims(text: str) -> list:
    """The sentences of ``text``'s prose that say a step waits on the person's phone or device, except
    ones that quote a page's own words about it. Frontmatter and fenced code are not read."""
    out = []
    for sentence in _SENTENCE_SPLIT.split(_prose(text or "")):
        s = " ".join(sentence.split())
        if not s or DEVICE_CLAIM_MARK in s or not (_DEVICE_WORDS.search(s) and _APPROVAL_WORDS.search(s)):
            continue
        if any(_DEVICE_WORDS.search(q.group(0)) or _re.search(r"phone|device", q.group(0), _re.I)
               for q in _QUOTED.finditer(s)):
            continue  # quotes what a page said: grounded
        out.append(s)
    return out


def annotate_device_claims(old: Optional[str], new: str) -> tuple:
    """``(content, added)``: ``new`` with :data:`DEVICE_CLAIM_MARK` after each device-approval sentence
    that ``old`` did not already hold. Sentences already in the skill are left exactly as they are."""
    before = set(device_approval_claims(old or ""))
    added = [s for s in device_approval_claims(new) if s not in before]
    segments = _segments(new)
    for sentence in added:
        pattern = _re.compile(r"\s+".join(_re.escape(w) for w in sentence.split()))
        for i, (is_prose, chunk) in enumerate(segments):
            if is_prose and pattern.search(chunk):
                segments[i] = (True, pattern.sub(lambda m: m.group(0) + " " + DEVICE_CLAIM_MARK, chunk, count=1))
                break
    return "".join(chunk for _, chunk in segments), added


_claim_warnings = _threading.local()


def review_device_claims(old: Optional[str], new: str, label: str) -> str:
    """For a background-review write: the content to write, with any new device claim marked, and a
    warning parked for :func:`take_device_claim_warning`. Any other write passes through unchanged."""
    if not _is_background_review():
        return new
    content, added = annotate_device_claims(old, new)
    if added:
        quote = added[0] if len(added[0]) <= 220 else added[0][:217] + "..."
        _claim_warnings.value = (
            f"{label}: this adds a claim that a step waits on the person's phone or device (\"{quote}\"); it was "
            "kept, and marked [unverified] (true only when the page itself says so). Whether a site sent anything to a device is only known from that page, at "
            "that moment. Consider rewording it as the general rule: when a page needs the person, call "
            "browser_handoff, and say only what the page itself says.")
    return content


def take_device_claim_warning() -> Optional[str]:
    warning = getattr(_claim_warnings, "value", None)
    _claim_warnings.value = None
    return warning
