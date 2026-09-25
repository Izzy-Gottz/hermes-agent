"""A skill may not add a claim that a step waits on the person's phone or device (slice D).

On 2026-09-24 the background review wrote into ~/.hermes/skills/product-launch-social-media/SKILL.md
(line 118, read-only here) the sentence quoted below, and Moe then told its owner to approve things
on his phone that no site had sent to any phone.
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from tools.skill_manager_guards import device_approval_claims
from tools.skill_manager_tool import _create_skill, _edit_skill, _patch_skill, _write_file

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


def test_creating_a_skill_with_the_claim_is_refused(tmp_path):
    with _skills(tmp_path):
        out = _create_skill("launch-kit", SKILL + "\n" + POISON + "\n")
    assert out["success"] is False and out["error_type"] == "ungrounded_device_claim"
    assert "browser_handoff" in out["error"]
    assert not (tmp_path / "launch-kit").exists()


def test_a_patch_that_adds_it_is_refused_and_the_file_is_unchanged(tmp_path):
    with _skills(tmp_path):
        assert _create_skill("launch-kit", SKILL)["success"]
        out = _patch_skill("launch-kit", "Read the post-login page before re-filling anything.",
                           "Read the post-login page before re-filling anything. " + POISON)
        assert out["success"] is False and out["error_type"] == "ungrounded_device_claim"
        assert (tmp_path / "launch-kit" / "SKILL.md").read_text() == SKILL


def test_a_supporting_file_is_held_to_the_same_rule(tmp_path):
    with _skills(tmp_path):
        assert _create_skill("launch-kit", SKILL)["success"]
        out = _write_file("launch-kit", "references/google.md", "Google sign-in\n\n" + POISON + "\n")
    assert out["success"] is False and out["error_type"] == "ungrounded_device_claim"


def test_an_edit_that_keeps_an_existing_sentence_passes(tmp_path):
    """The claim already in someone's skill is theirs to remove; unrelated edits are not blocked by it."""
    with _skills(tmp_path):
        (tmp_path / "launch-kit").mkdir()
        (tmp_path / "launch-kit" / "SKILL.md").write_text(SKILL + "\n" + POISON + "\n")
        out = _edit_skill("launch-kit", SKILL.replace("Launch kit", "Launch kit (v2)") + "\n" + POISON + "\n")
    assert out["success"] is True


def test_an_edit_that_removes_it_passes(tmp_path):
    with _skills(tmp_path):
        (tmp_path / "launch-kit").mkdir()
        (tmp_path / "launch-kit" / "SKILL.md").write_text(SKILL + "\n" + POISON + "\n")
        out = _edit_skill("launch-kit", SKILL + "\nWhen a page needs the person, call browser_handoff.\n")
    assert out["success"] is True
