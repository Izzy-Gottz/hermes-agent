#!/usr/bin/env python3
"""Prove these tests can fail. Break one property at a time; expect red.

    python tests/tools/mutation_guard.py            # every group
    python tests/tools/mutation_guard.py composio   # one group

A check that cannot fail is worse than no check: it reports PASS forever, and
this repo has now found eleven of them — two in the very changes these tests
cover (`skill_view` hard-coded `missing_required_commands` to `[]` while cron
preflight was already reading it; the schema-cache writer read `inputSchema`
off a model that had renamed the field, writing 1,994 empty schemas). Green
tests are not evidence that a test works. Only a mutation the tests catch is.

So every entry below names **the tests it must break**. The run fails if any
of them stays green, and it fails just as loudly if an anchor stops matching —
a mutation whose anchor has drifted is silently testing nothing, which is the
same disease one level up.

Deliberately NOT a pytest file (`mutate`/`mutation` prefix, not `test_`): it
rewrites source files on disk and must never run inside a normal collection.
It restores every file it touches, including on exception.

Run it with the interpreter that can import the tree, e.g.

    PYTHONPATH=/path/to/pytest-libs \\
      ~/.hermes/hermes-agent/.venv/bin/python tests/tools/mutation_guard.py

(On the Mac these changes were written on, `uv sync` cannot reach pypi while
`pip` can, so pytest lives in a --target directory on PYTHONPATH rather than
in the venv. See HERMES-SURFACES.md §0's handoff notes.)
"""

import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MCP = ROOT / "tools/mcp_tool.py"
SKILLS = ROOT / "tools/skills_tool.py"
CRON = ROOT / "cron/scheduler.py"

COMPOSIO_TESTS = ["tests/tools/test_mcp_composio_defaults.py"]
SKILL_TESTS = ["tests/tools/test_skill_required_commands.py",
               # The regression this change caused once and must not again:
               # blocking in skill_view flipped `agentmail` to setup_needed.
               "tests/tools/test_skill_env_passthrough.py"]
CACHE_TESTS = ["tests/tools/test_mcp_schema_cache_field_rename.py"]


# (file, name, anchor, replacement, [substrings of tests that MUST go red])
GROUPS = {
    "composio": (COMPOSIO_TESTS, [
        (MCP, "rule 1: the caller's own value stops mattering",
         "    if any(param in arguments for param in table):\n        return arguments, []",
         "    if False:\n        return arguments, []",
         ["naming_one_lever_stands_the_whole_table_down",
          "explicit_false_is_left_alone",
          "explicit_inner_value_stands_the_table_down"]),

        (MCP, "an absent `arguments` key is skipped instead of filled",
         "    if arguments is None:\n        arguments = {}",
         "    if arguments is None:\n        pass",
         ["entry_with_no_arguments_at_all_is_filled",
          "null_arguments_value_is_filled_too"]),

        (MCP, "a malformed `arguments` value is reinterpreted",
         "    if not isinstance(arguments, dict):\n        return arguments, []",
         "    if not isinstance(arguments, dict):\n        arguments = {}",
         ["arguments_of_the_wrong_type_are_passed_through"]),

        (MCP, "rule 2: the Composio check is dropped",
         "    if not isinstance(args, dict) or not _is_composio_server(server):",
         "    if not isinstance(args, dict):",
         ["same_tool_name_on_another_server", "server_with_no_url_at_all",
          "lookalike_host_is_not_composio"]),

        (MCP, "rule 2: the host is substring-matched again",
         '    return any(host == h or host.endswith("." + h) for h in _COMPOSIO_HOSTS)',
         '    return any(h in url.lower() for h in _COMPOSIO_HOSTS)',
         ["lookalike_host_is_not_composio"]),

        (MCP, "rule 3: the schema is ignored",
         "        if _composio_declares(server, key, param) is False:",
         "        if False:",
         ["withdrawn_parameter_is_not_posted"]),

        (MCP, "rule 3 inverted: no schema reads as a veto",
         "        if _composio_declares(server, key, param) is False:",
         "        if _composio_declares(server, key, param) is not True:",
         ["undiscovered_tool_still_gets_the_default"]),

        (MCP, "rule 4: the disclosure is dropped from the payload",
         '                if _defaults_note:\n                    payload["_hermes"] = _defaults_note',
         '                if False:\n                    payload["_hermes"] = _defaults_note',
         ["result_names_what_was_applied",
          "structured_content_survives_alongside_the_note"]),

        (MCP, "rule 4b: the note no longer forces a payload",
         "            if structured is not None or meta is not None or _defaults_note:",
         "            if structured is not None or meta is not None:",
         ["result_names_what_was_applied"]),

        (MCP, "rule 4c: a tool ERROR loses the note (the retry loop)",
         '                        (error_text or "MCP tool returned an error")\n'
         '                        + (("\\n\\n" + _defaults_note) if _defaults_note else "")',
         '                        (error_text or "MCP tool returned an error")',
         ["tool_error_carries_the_note"]),

        (MCP, "rule 4d: a transport failure loses the note",
         '                f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"\n'
         '                + (("\\n\\n" + _defaults_note) if _defaults_note else "")',
         '                f"MCP call failed: {type(exc).__name__}: {_exc_str(exc)}"',
         ["transport_exception_carries_the_note"]),

        (MCP, "the note is appended to every failure, applied or not",
         '                + (("\\n\\n" + _defaults_note) if _defaults_note else "")\n            ))',
         '                + "\\n\\nHermes filled in something"\n            ))',
         ["failure_with_no_defaults_applied_says_nothing"]),

        (MCP, "the multiplexer path is skipped",
         '    if str(tool_name or "").upper() == _COMPOSIO_MULTIPLEXER:',
         '    if False:',
         ["inner_arguments_are_filled", "only_the_matching_entry_changes",
          "entry_with_no_arguments_at_all_is_filled"]),

        (MCP, "the direct path is skipped",
         "    new_args, kvs = _composio_default_one(server, tool_name, args)",
         "    new_args, kvs = args, []",
         ["omitted_parameters_are_filled", "result_names_what_was_applied"]),

        (MCP, "the caller's dict is mutated in place",
         "    out = dict(arguments)\n    out.update(filled)",
         "    out = arguments\n    out.update(filled)",
         ["callers_dict_is_not_mutated", "callers_nested_dicts_are_not_mutated"]),

        (MCP, "a tool outside the table gets defaults anyway",
         "    table = _COMPOSIO_CHEAP_DEFAULTS.get(key)\n    if not table:",
         '    table = _COMPOSIO_CHEAP_DEFAULTS.get(key) or {"verbose": False}\n    if not table:',
         ["tool_not_in_the_table_is_untouched", "no_note_when_nothing_was_filled"]),
    ]),

    "skills": (SKILL_TESTS, [
        (SKILLS, "the declared list goes back to a literal []",
         '            "required_commands": required_commands,',
         '            "required_commands": [],',
         ["declared_present_command_is_reported", "missing_command_is_named",
          "present_and_absent_together"]),

        (SKILLS, "the missing list goes back to a literal [] (the original bug)",
         '            "missing_required_commands": missing_required_commands,',
         '            "missing_required_commands": [],',
         ["missing_command_is_named", "present_and_absent_together",
          "missing_binary_refuses_the_job_on_its_own"]),

        (SKILLS, "a missing binary relabels the skill after all",
         "        missing_required_commands = _missing_required_commands(required_commands)",
         "        missing_required_commands = _missing_required_commands(required_commands)\n"
         "        if missing_required_commands:\n            setup_needed = True",
         ["missing_command_is_named_but_does_not_relabel",
          "agentmail_key_is_optional"]),

        (SKILLS, "the note stops naming the binary",
         "            ] + [\n                f\"command '{name}'\" for name in missing_required_commands\n            ]",
         "            ]",
         ["setup_note_names_the_binary"]),

        (SKILLS, "the search stops widening past PATH (the launchd case)",
         "    for extra in _COMMAND_SEARCH_EXTRA_DIRS:\n"
         "        expanded = os.path.expanduser(extra)\n"
         "        if expanded not in parts:\n"
         "            parts.append(expanded)",
         "    pass",
         ["thin_PATH_still_finds_the_usual_local_bins"]),

        (SKILLS, "everything is reported missing",
         "    return [c for c in commands if _resolve_required_command(c) is None]",
         "    return list(commands)",
         ["declared_present_command_is_reported", "present_and_absent_together",
          "missing_is_a_subset", "runnable_skill_is_not_refused"]),

        (SKILLS, "nothing is ever reported missing",
         "    return [c for c in commands if _resolve_required_command(c) is None]",
         "    return []",
         ["missing_command_is_named", "missing_is_a_subset",
          "missing_binary_refuses_the_job_on_its_own"]),

        (SKILLS, "the name is no longer stripped",
         '    name = str(command or "").strip()',
         '    name = str(command or "")',
         ["stray_space_does_not_make_a_binary_missing"]),

        (CRON, "preflight goes back to blocking only on setup_needed",
         '            or payload.get("readiness_status") == "setup_needed"\n'
         "            or missing_commands\n        ):",
         '            or payload.get("readiness_status") == "setup_needed"\n        ):',
         ["missing_binary_refuses_the_job_on_its_own"]),

        (CRON, "preflight refuses every job",
         '        missing_commands = payload.get("missing_required_commands") or []',
         '        missing_commands = ["anything"]',
         ["runnable_skill_is_not_refused"]),

        (CRON, "the refusal stops naming the binary",
         '            missing += [f"command \'{name}\'" for name in missing_commands]',
         '            missing += []',
         ["missing_binary_refuses_the_job_on_its_own"]),
    ]),

    "schema-cache": (CACHE_TESTS, [
        (MCP, "the writer reads the renamed field with getattr again",
         '                schema_obj = mcp_field(mcp_tool, "input_schema", "inputSchema")',
         '                schema_obj = getattr(mcp_tool, "inputSchema", None)',
         ["written_cache_entry_keeps_the_parameters"]),
    ]),
}


def run(tests, selector=None):
    cmd = [sys.executable, "-m", "pytest"] + tests + ["-q", "-p", "no:cacheprovider"]
    if selector:
        cmd += ["-k", selector]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def last_summary(res):
    lines = [l for l in res.stdout.splitlines()
             if "passed" in l or "failed" in l or "error" in l]
    return lines[-1].strip() if lines else "?"


def main(argv):
    wanted = argv[1:] or list(GROUPS)
    unknown = [g for g in wanted if g not in GROUPS]
    if unknown:
        print("unknown group(s): %s — have: %s" % (
            ", ".join(unknown), ", ".join(GROUPS)))
        return 2

    failures = []
    for group in wanted:
        tests, mutations = GROUPS[group]
        print("── %s ──" % group)
        base = run(tests)
        if base.returncode != 0:
            print("  BASELINE IS RED — fix that before trusting anything below")
            print(base.stdout[-3000:])
            failures.append("%s: baseline red" % group)
            continue
        print("  baseline: %s" % last_summary(base))

        originals = {}
        try:
            for path, name, old, new, must_break in mutations:
                src = originals.setdefault(
                    path, io.open(path, encoding="utf-8").read())
                if src.count(old) != 1:
                    # An anchor that no longer matches is a mutation testing
                    # nothing — the same failure mode one level up.
                    print("  ANCHOR   %s (matched %d times)" % (name, src.count(old)))
                    failures.append("%s / %s: anchor matched %d times"
                                    % (group, name, src.count(old)))
                    continue
                io.open(path, "w", encoding="utf-8").write(
                    src.replace(old, new, 1))
                res = run(tests, " or ".join(must_break))
                io.open(path, "w", encoding="utf-8").write(src)
                broke = res.returncode != 0
                print("  %-8s %-56s (%s)" % (
                    "BROKE" if broke else "SURVIVED", name, last_summary(res)))
                if not broke:
                    failures.append("%s / %s: the tests stayed green"
                                    % (group, name))
        finally:
            for path, src in originals.items():
                io.open(path, "w", encoding="utf-8").write(src)

        after = run(tests)
        print("  restored: %s" % last_summary(after))
        if after.returncode != 0:
            failures.append("%s: files did not restore cleanly" % group)
        print()

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("every mutation was caught by the tests that name it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
