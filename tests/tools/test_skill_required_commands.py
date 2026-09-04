"""A skill that needs a binary this machine lacks is not ready to run.

`skill_view` returned ``required_commands: []`` and
``missing_required_commands: []`` as literals, always — while
``cron/scheduler.py::_preflight_check_skills`` was already written to read
the second one and name the missing binary in its refusal. So the consumer
existed, the producer always said "nothing missing", and the check could not
fail: a cron job attached to such a skill was dispatched, spent an LLM call,
and died at the shell.

Measured on one real install before the fix: 14 skills declare a required
command, 11 of those commands are absent, and 7 of those skills are enabled.

Where the refusal lives is a deliberate split, and it cost a real test to
learn: `skill_view` REPORTS a missing binary and does not relabel the skill,
because interactively there is someone who can install it -- and because
`agentmail` declares a command while keeping its API key optional precisely
so the CLI self-signup path stays open. Blocking there flipped that skill to
setup_needed for a reason its own test was never about. A scheduled run has
nobody to ask, so cron preflight refuses on the command alone.

Two things these tests hold down that are easy to get wrong:

- **The negative control.** A test that only ever asserts "missing" passes
  just as well against a probe that finds nothing at all. Every absence below
  is paired with a present binary resolved through the same code path.
- **PATH.** A launchd- or gateway-spawned Hermes inherits
  /usr/bin:/bin:/usr/sbin:/sbin — no Homebrew, no ~/.local/bin. Resolving
  against THAT would report half a developer's machine missing and, now that
  a missing command blocks a job, would refuse jobs that work by hand.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from tools import skills_tool


def _write_skill(root: Path, name: str, frontmatter_lines) -> Path:
    d = root / "testing" / name
    d.mkdir(parents=True, exist_ok=True)
    body = "---\n" + "\n".join(frontmatter_lines) + "\n---\n\n# " + name + "\n\nDo the thing.\n"
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return d / "SKILL.md"


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: root)
    return root


def _view(name):
    return json.loads(skills_tool.skill_view(name))


class TestResolution:
    """The probe itself, before anything is wired to it."""

    def test_a_present_command_resolves(self):
        assert skills_tool._resolve_required_command("sh") is not None

    def test_an_absent_command_does_not(self):
        assert skills_tool._resolve_required_command(
            "definitely-not-a-real-binary-9f3a2b") is None

    def test_a_stray_space_does_not_make_a_binary_missing(self):
        """`shutil.which(" sh ")` is None. One space in frontmatter would,
        without the strip, report a present binary as absent — and now that a
        missing binary blocks a job, refuse a job that works."""
        assert skills_tool._resolve_required_command("  sh  ") is not None

    def test_the_empty_string_is_not_a_command(self):
        assert skills_tool._resolve_required_command("") is None
        assert skills_tool._resolve_required_command("   ") is None

    def test_a_thin_PATH_still_finds_the_usual_local_bins(self, tmp_path, monkeypatch):
        """The launchd case: a binary the user's shell finds and PATH does not.

        Without the widened search this returns None, a job that works by
        hand is refused, and the guard fires on the wrong person.
        """
        fake_brew = tmp_path / "brew-bin"
        fake_brew.mkdir()
        exe = fake_brew / "songsee"
        exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)

        monkeypatch.setattr(skills_tool, "_COMMAND_SEARCH_EXTRA_DIRS",
                            (str(fake_brew),))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        assert skills_tool._resolve_required_command("songsee") == str(exe)

    def test_the_widening_is_not_what_finds_everything(self, monkeypatch):
        """Negative control for the test above: the extra dirs are not magic."""
        monkeypatch.setattr(skills_tool, "_COMMAND_SEARCH_EXTRA_DIRS", ())
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        assert skills_tool._resolve_required_command("sh") is not None
        assert skills_tool._resolve_required_command("songsee") is None

    def test_missing_is_a_subset_and_keeps_its_order(self):
        got = skills_tool._missing_required_commands(
            ["sh", "nope-a-9f3a2b", "ls", "nope-b-9f3a2b"])
        assert got == ["nope-a-9f3a2b", "nope-b-9f3a2b"]


class TestSkillView:
    """What `skill_view` now reports, and what it blocks."""

    def test_a_declared_present_command_is_reported_and_does_not_block(
        self, skills_root
    ):
        _write_skill(skills_root, "shellish", [
            "name: shellish",
            "description: uses sh",
            "prerequisites:",
            "  commands: [sh]",
        ])
        got = _view("shellish")
        assert got["success"] is True
        assert got["required_commands"] == ["sh"]
        assert got["missing_required_commands"] == []
        assert got["setup_needed"] is False
        assert got["readiness_status"] == "available"

    def test_a_missing_command_is_named_but_does_not_relabel_the_skill(
        self, skills_root
    ):
        """Reported, not blocked — interactively there is someone to ask.

        The one real skill that declares a command shows why: `agentmail`
        keeps its API key optional precisely so the CLI self-signup path
        stays open, and `test_agentmail_key_is_optional_so_self_signup_still
        _works` asserts it is not setup_needed. A missing binary must not
        relabel it out from under that. The refusal lives in cron preflight,
        where a run is spent instead of a person being asked.
        """
        _write_skill(skills_root, "songseeish", [
            "name: songseeish",
            "description: shells out to a binary nobody has",
            "prerequisites:",
            "  commands: [definitely-not-a-real-binary-9f3a2b]",
        ])
        got = _view("songseeish")
        assert got["required_commands"] == [
            "definitely-not-a-real-binary-9f3a2b"]
        assert got["missing_required_commands"] == [
            "definitely-not-a-real-binary-9f3a2b"]
        assert got["setup_needed"] is False
        assert got["readiness_status"] == "available"

    def test_the_setup_note_names_the_binary_when_a_note_is_written(
        self, skills_root
    ):
        """A note is written for the missing env var — and once one is being
        written, "missing env $X" alone would hide the other half. "missing
        required prerequisites" is true, useless, and unfindable."""
        _write_skill(skills_root, "notey", [
            "name: notey",
            "description: x",
            "prerequisites:",
            "  commands: [definitely-not-a-real-binary-9f3a2b]",
            "required_environment_variables:",
            "  - name: NOTEY_KEY_9F3A2B",
            "    prompt: a key",
        ])
        got = _view("notey")
        assert got["setup_needed"] is True, "the env var is what blocks"
        note = got["setup_note"]
        assert "NOTEY_KEY_9F3A2B" in note
        assert "definitely-not-a-real-binary-9f3a2b" in note, \
            "the binary is named too, or half the reason stays hidden"

    def test_a_skill_declaring_nothing_is_unaffected(self, skills_root):
        _write_skill(skills_root, "plain", [
            "name: plain",
            "description: needs nothing",
        ])
        got = _view("plain")
        assert got["required_commands"] == []
        assert got["missing_required_commands"] == []
        assert got["setup_needed"] is False

    def test_present_and_absent_together(self, skills_root):
        """Only the absent one is named — the list is not all-or-nothing."""
        _write_skill(skills_root, "mixed", [
            "name: mixed",
            "description: x",
            "prerequisites:",
            "  commands: [sh, definitely-not-a-real-binary-9f3a2b]",
        ])
        got = _view("mixed")
        assert got["required_commands"] == [
            "sh", "definitely-not-a-real-binary-9f3a2b"]
        assert got["missing_required_commands"] == [
            "definitely-not-a-real-binary-9f3a2b"]

    def test_a_single_command_as_a_bare_string_is_accepted(self, skills_root):
        """`commands: jq` rather than `commands: [jq]` — people write both."""
        _write_skill(skills_root, "bare", [
            "name: bare",
            "description: x",
            "prerequisites:",
            "  commands: definitely-not-a-real-binary-9f3a2b",
        ])
        got = _view("bare")
        assert got["missing_required_commands"] == [
            "definitely-not-a-real-binary-9f3a2b"]


class TestCronPreflight:
    """The consumer that was written against a field that was always empty."""

    def test_a_missing_binary_refuses_the_job_on_its_own(self, skills_root):
        """No other requirement is missing here — the binary alone must block.

        This is the whole point of the change: the skill reports
        ``setup_needed: False`` (nothing to ask a person for), and the job is
        still refused, because a scheduled run has nobody to ask and would
        spend an LLM call before dying at the shell.
        """
        from cron import scheduler

        _write_skill(skills_root, "cronish", [
            "name: cronish",
            "description: x",
            "prerequisites:",
            "  commands: [definitely-not-a-real-binary-9f3a2b]",
        ])
        assert _view("cronish")["setup_needed"] is False
        reason = scheduler._preflight_check_skills({"skills": ["cronish"]})
        assert reason is not None, \
            "a job that cannot possibly succeed was dispatched anyway"
        assert "definitely-not-a-real-binary-9f3a2b" in reason
        assert "cronish" in reason

    def test_a_runnable_skill_is_not_refused(self, skills_root):
        """The control: preflight must not have started refusing everything."""
        from cron import scheduler

        _write_skill(skills_root, "fine", [
            "name: fine",
            "description: x",
            "prerequisites:",
            "  commands: [sh]",
        ])
        assert scheduler._preflight_check_skills({"skills": ["fine"]}) is None
