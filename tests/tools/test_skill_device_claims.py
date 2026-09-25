"""A background-review write that adds "it waits on their phone/device" is marked and warned (slice D).

On 2026-09-24 the background review wrote into ~/.hermes/skills/product-launch-social-media/SKILL.md
(then line 118, now 128; read-only here) the sentence quoted below, and Moe then told its owner to approve things
on his phone that no site had sent to any phone.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

import json

from tools.skill_manager_guards import DEVICE_CLAIM_MARK, annotate_device_claims, device_approval_claims
from tools.skill_manager_tool import _create_skill, _edit_skill, _patch_skill, _write_file, skill_manage

#: Verbatim from the live skill (measured 2026-09-25).
POISON = ("That is expected, not a new failure — there is nothing to click through it (`browser_vault_enter_code` "
          "returns `no_code_field` for this case), so say so plainly and wait for the user to approve it on their "
          "device before resuming.")

SKILL = """\
---
name: launch-kit
description: Submitting a product to launch directories.
---

# Launch kit

Read the post-login page before re-filling anything.
"""


@contextmanager
def _skills(tmp_path):
    with patch("tools.skill_manager_tool.SKILLS_DIR", tmp_path), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[tmp_path]):
        yield


def test_the_measured_sentence_is_a_device_claim():
    assert device_approval_claims(POISON) == [POISON]


@pytest.mark.parametrize("text", [
    "Tell the user to approve the sign-in on their phone.",
    "Wait for the person to confirm on their own device.",
    "The owner taps approve on the user's phone, then the page moves on.",
])
def test_other_wordings_are_too(text):
    assert device_approval_claims(text)


@pytest.mark.parametrize("text", [
    "When a page needs the person, call browser_handoff so they can see it.",
    "Cookies can be stored on your device.",                        # no approval in it
    "Approve the draft before posting.",                            # no device in it
])
def test_ordinary_sentences_are_not(text):
    assert device_approval_claims(text) == []


@contextmanager
def _review(on=True):
    """The background review, writing skills it manages (the ownership guards are not under test)."""
    with patch("tools.skill_manager_guards._is_background_review", return_value=on), \
         patch("tools.skill_manager_tool._background_review_write_guard", return_value=None), \
         patch("tools.skill_manager_tool._background_review_read_before_write_guard", return_value=None):
        yield


def _read(path):
    """The review has read the file it changes (the read-before-write guard requires it)."""
    from tools.skill_manager_guards import _reset_background_review_read_marks, mark_background_review_skill_read
    _reset_background_review_read_marks()
    mark_background_review_skill_read(path)


def test_a_quoted_page_line_is_not_a_claim():
    """Without the exemption this sentence IS matched ("on your phone" + "Tap"/"approve")."""
    quoted = 'When the page says "Tap Yes on your phone to approve", tell them exactly that.'
    unquoted = "When the page asks, Tap Yes on your phone to approve, and tell them exactly that."
    assert device_approval_claims(unquoted)                   # the words alone are a claim
    assert device_approval_claims(quoted) == []               # quoting the page is a grounded report
    content, added = annotate_device_claims(None, SKILL + quoted + "\n")
    assert added == [] and DEVICE_CLAIM_MARK not in content


def test_the_mark_has_no_colon_and_the_skill_still_parses():
    import yaml
    assert DEVICE_CLAIM_MARK == "[unverified]" and ":" not in DEVICE_CLAIM_MARK
    fm_claim = SKILL.replace("description: Submitting a product to launch directories.",
                             "description: Wait for the user to approve it on their phone first.")
    fence = "\n```yaml\nnote: wait for the user to approve it on their device\n```\n"
    content, added = annotate_device_claims(None, fm_claim + fence + "\n" + POISON + "\n")
    assert added == [POISON]                                  # the prose line only
    assert "approve it on their phone first.\n" in content    # frontmatter untouched
    assert "approve it on their device\n```" in content        # the fence untouched
    front = content.split("---")[1]
    assert yaml.safe_load(front)["description"].endswith("phone first.")
    block = content.split("```yaml\n")[1].split("```")[0]
    assert yaml.safe_load(block) == {"note": "wait for the user to approve it on their device"}
    assert POISON + " " + DEVICE_CLAIM_MARK in content


def test_the_background_review_adding_the_claim_keeps_it_marked_and_warns(tmp_path):
    with _skills(tmp_path), _review():
        out = json.loads(skill_manage(action="create", name="launch-kit", content=SKILL + "\n" + POISON + "\n"))
    assert out["success"] is True and "unverified" in out["warning"] and "browser_handoff" in out["warning"]
    text = (tmp_path / "launch-kit" / "SKILL.md").read_text()
    assert POISON + " " + DEVICE_CLAIM_MARK in text


def test_a_review_patch_that_adds_it_is_marked_not_refused(tmp_path):
    with _skills(tmp_path):
        assert _create_skill("launch-kit", SKILL)["success"]
        with _review():
            _read(tmp_path / "launch-kit" / "SKILL.md")
            out = _patch_skill("launch-kit", "Read the post-login page before re-filling anything.",
                               "Read the post-login page before re-filling anything. " + POISON)
        assert out["success"] is True
        assert DEVICE_CLAIM_MARK in (tmp_path / "launch-kit" / "SKILL.md").read_text()


def test_a_review_supporting_file_is_held_to_the_same_rule(tmp_path):
    with _skills(tmp_path):
        assert _create_skill("launch-kit", SKILL)["success"]
        with _review():
            out = _write_file("launch-kit", "references/google.md", "Google sign-in\n\n" + POISON + "\n")
        assert out["success"] is True
        assert DEVICE_CLAIM_MARK in (tmp_path / "launch-kit" / "references" / "google.md").read_text()


def test_what_the_owner_dictates_is_written_word_for_word(tmp_path):
    with _skills(tmp_path), _review(on=False):
        out = _create_skill("launch-kit", SKILL + "\n" + POISON + "\n")
    assert out["success"] is True
    assert DEVICE_CLAIM_MARK not in (tmp_path / "launch-kit" / "SKILL.md").read_text()


def test_an_edit_that_keeps_an_existing_sentence_leaves_it_alone(tmp_path):
    """The claim already in someone's skill is theirs to remove; unrelated edits do not touch it."""
    with _skills(tmp_path), _review():
        (tmp_path / "launch-kit").mkdir()
        (tmp_path / "launch-kit" / "SKILL.md").write_text(SKILL + "\n" + POISON + "\n")
        _read(tmp_path / "launch-kit" / "SKILL.md")
        out = _edit_skill("launch-kit", SKILL.replace("Launch kit", "Launch kit (v2)") + "\n" + POISON + "\n")
    assert out["success"] is True
    assert DEVICE_CLAIM_MARK not in (tmp_path / "launch-kit" / "SKILL.md").read_text()


def test_marking_is_idempotent(tmp_path):
    content, added = annotate_device_claims(None, SKILL + POISON)
    again, added2 = annotate_device_claims(content, content)
    assert added and added2 == [] and again == content
