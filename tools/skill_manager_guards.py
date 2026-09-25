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


def _background_review_write_guard(
    name: str, skill_dir: Path, action: str) -> Optional[Dict[str, Any]]:
    """Refuse autonomous curator writes to anything but curator-owned sediment. The review fork
    has no user in the loop, so it is also blocked on pinned/external/bundled/hub skills."""
    if not _is_background_review():
        return None
    refuse = f"Refusing background curator {action} for"
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
        # Not curator-managed (no `created_by: "agent"`) => user-owned. A MISSING
        # record and an explicit `created_by: null` must resolve IDENTICALLY (keying
        # on presence made the policy depend on the guard's own side effect: the
        # first write created a null record, the next identical write was refused).
        usage_rec = skill_usage.load_usage().get(name)
        # Skills that are not curator-managed are off-limits to autonomous curation. This prevents the LLM
        # consolidation pass from mutating skills the user owns (manually authored, URL-installed, or
        # created by a foreground `skill_manage(create)` at the user's request), which lack the `created_by:
        # "agent"` marker. Keying on `isinstance(usage_rec, dict)` made the policy depend on the guard's own
        # side effect: a local skill with no telemetry record passed, the successful write called
        # bump_patch() which created a `created_by: null` record, and the very same write was refused from
        # then on. "Allowed exactly once" is not a policy — it is a race with our own bookkeeping. Fail
        # closed for both shapes; `hermes curator adopt <name>` is the supported way in. See #67140.
        if not skill_usage._is_curator_managed_record(usage_rec):
            _detail = (f"created_by={usage_rec.get('created_by')!r}" if isinstance(usage_rec, dict)
                       else "no usage record")
            return _refusal(
                f"{refuse} skill '{name}': the skill is not "
                f"curator-managed ({_detail}). User-owned skills are off-limits to autonomous "
                f"curation. Run `hermes curator adopt {name}` to opt it in.")
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
    if action not in {"edit", "patch", "delete", "write_file", "remove_file"}:
        return None
    from tools import skill_manager_tool as _smt
    existing = _smt._find_skill(name)
    return _background_review_write_guard(name, existing["path"], action) if existing else None


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


def _is_org_mirror(skill_path: Path) -> bool:
    from agent.skill_utils import is_org_mirror_path
    from tools import skill_manager_tool as _smt
    return is_org_mirror_path(skill_path, _smt._skills_dir())


def _maybe_auto_propose_org_edit(name: str, skill_path: Path) -> Optional[str]:
    """Submit an org-skill edit upstream when `sync.org_auto_propose` is on. Returns a note for
    the tool result or None; never raises (the edit is saved locally and can be proposed later)."""
    try:
        from tools import skills_sync_client as ssc
        if not _is_org_mirror(skill_path):
            return None
        if not ssc.sync_org_auto_propose():
            return (
                f"This skill is shared by your organisation. Your edit is "
                f"saved locally and will not be overwritten by org updates. "
                f"Run `hermes sync propose {name}` to share it back.")
        from tools.skills_sync_client_org import propose_skill
        result = propose_skill(name)
        if result.get("proposal_pending"):
            return (
                f"Auto-proposed to your organisation as proposal "
                f"#{result.get('proposal_id')} (pending admin review).")
        return "Auto-proposed to your organisation (merged into the shared set)."
    except Exception as e:
        logger.debug("auto-propose skipped for %s: %s", name, e)
        return (
            f"Edit saved locally. Could not submit it to your organisation "
            f"right now — run `hermes sync propose {name}` to retry.")


def _org_mirror_write_guard(name: str, skill_path: Path, action: str) -> Optional[Dict[str, Any]]:
    """Org-shared skills are EDITABLE IN PLACE — this only blocks deletion. Edits land in the
    mirror, survive the next org pull (baseline sidecar in skills_sync_client) and reach the org
    via `hermes sync propose`. Deletion stays refused: the mirror is a view of org HEAD, so a
    local delete just comes back, and removing for everyone is an admin action."""
    if action not in {"delete", "remove_file"}:
        return None
    try:
        if _is_org_mirror(skill_path):
            return _refusal(
                f"Cannot {action} '{name}' locally: it is shared by your organisation, so a local "
                f"delete would just come back on the next sync. Ask an org admin to remove it for "
                f"everyone. (Editing it IS allowed — your changes are kept and can be proposed "
                f"back with `hermes sync propose {name}`.)")
    except Exception:
        logger.debug("org mirror guard lookup failed for %s", name, exc_info=True)
    return None


# ── Claims a skill must not keep ─────────────────────────────────────────────
#
# A skill is read on every later turn, for every site. On 2026-09-24 the background review wrote into
# a launch skill that a passkey challenge "on the user's own device" is expected, "so ... wait for the
# user to approve it on their device" -- and Moe then told its owner to approve things on his phone
# that no site had sent to any phone. Whether a step waits on a device is a fact about ONE page at ONE
# moment, known only from what that page said (tools/browser_person_step.py: page_says); a skill that
# states it as procedure is a standing false claim. So a write that ADDS such a sentence is refused,
# whoever makes it; a sentence already there is left for the person to remove (their skills are
# theirs), and edits that keep it pass.

import re as _re

_DEVICE_WORDS = _re.compile(
    r"\b(?:on|from|with|via|using)\s+(?:the\s+|their\s+|your\s+|his\s+|her\s+)?"
    r"(?:(?:user|person|owner|customer|human)(?:'s|s'|s)?\s+)?(?:own\s+|other\s+|mobile\s+)?"
    r"(?:phone|device|iphone|android|mobile)\b", _re.I)
_APPROVAL_WORDS = _re.compile(r"\b(?:approv\w*|confirm\w*|tap\w*|wait\w*|accept\w*|allow\w*|complet\w*|finish\w*|"
                              r"authori[sz]\w*|respond\w*)\b", _re.I)
_SENTENCE_SPLIT = _re.compile(r"(?<=[.!?])\s+|\n+")


def device_approval_claims(text: str) -> list:
    """The sentences of ``text`` that say a step waits on the person's phone or device."""
    out = []
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        s = " ".join(sentence.split())
        if s and _DEVICE_WORDS.search(s) and _APPROVAL_WORDS.search(s):
            out.append(s)
    return out


def _device_claim_write_guard(old: Optional[str], new: str, label: str) -> Optional[Dict[str, Any]]:
    """Refuse a write that adds a device-approval claim ``old`` did not already hold."""
    before = set(device_approval_claims(old or ""))
    added = [s for s in device_approval_claims(new) if s not in before]
    if not added:
        return None
    quote = added[0] if len(added[0]) <= 220 else added[0][:217] + "..."
    return _refusal(
        f"Refused: this write to {label} adds a claim that a step waits on the person's phone or device "
        f"(\"{quote}\"). A skill is read for every later site and turn, but whether a site sent anything to a "
        "device is only known from that page, at that moment. Write the general rule instead: when a page needs "
        "the person (a passkey, a security key, a CAPTCHA, an approval), call browser_handoff so they can see it, "
        "and tell them only what the page itself says.",
        error_type="ungrounded_device_claim")
