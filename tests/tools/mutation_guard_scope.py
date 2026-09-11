#!/usr/bin/env python3
"""Re-prove that the profile-scoping tests can fail.

Every rule in the scoping change is broken here, one at a time, and the
suite must go red for each. A rule nobody can break on demand is a rule
nobody has tested — this repo has found eleven checks that could not fail,
including one in this very area (the profile-isolation suite's own comment
promised a sweep test that did not exist).

Not named ``test_*``: pytest must not collect it. Run it directly.

    PYTHONPATH="$SCRATCH/pylibs" ~/.hermes/hermes-agent/.venv/bin/python \\
        tests/tools/mutation_guard_scope.py

Paths come from ``__file__`` and the interpreter from ``sys.executable``, so
it runs wherever the tree is checked out. Every file is restored in a
``finally``; a drifted anchor is a loud failure, not a silent skip, because
that is exactly how a harness like this rots into one more check that
cannot fail.
"""
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MCP = os.path.join(ROOT, "tools", "mcp_tool.py")
REG = os.path.join(ROOT, "tools", "registry.py")
TS = os.path.join(ROOT, "toolsets.py")
PLUGINS = os.path.join(ROOT, "hermes_cli", "plugins.py")
ISO_TEST = os.path.join(ROOT, "tests", "tools", "test_mcp_profile_isolation.py")

SCOPE_TESTS = "tests/tools/test_mcp_registry_scope.py"
# The partition-free registry rules (alias overlay, membership memo) moved here on the
# 0.21.1 port; the MCP-side entries below still name SCOPE_TESTS and are superseded.
ALIAS_TESTS = "tests/tools/test_registry_scoped_aliases.py"
SCOPE_TEST_FILE = os.path.join(
    ROOT, "tests", "tools", "test_mcp_registry_scope.py"
)
MCP_TESTS = "tests/tools/test_mcp_tool.py"
ISO_TESTS = "tests/tools/test_mcp_profile_isolation.py"
TOOLSET_TESTS = "tests/test_toolsets.py"
PLUGIN_TESTS = "tests/hermes_cli/test_plugins.py"

# (group, description, file, old, new, test target[, occurrences])
MUTATIONS = [
    # -- the key translation ------------------------------------------
    (
        "scope-map",
        "the empty root key is registered as scope '' instead of global",
        MCP,
        '''    if key == _ROOT_PROFILE_KEY:
        return None
''',
        '''''',
        SCOPE_TESTS,
    ),
    (
        "scope-map",
        "the unscoped key is collapsed to global instead of failing closed",
        MCP,
        '''    if key == _ROOT_PROFILE_KEY:
        return None
''',
        '''    if key in (_ROOT_PROFILE_KEY, _UNSCOPED_PROFILE_KEY):
        return None
''',
        SCOPE_TESTS,
    ),
    (
        "scope-map",
        "every profile collapses to the global scope (the original bug)",
        MCP,
        '''            "a profile scope (see gateway.run._profile_runtime_scope)."
        )
    return key''',
        '''            "a profile scope (see gateway.run._profile_runtime_scope)."
        )
    return None''',
        SCOPE_TESTS,
    ),
    # -- registration -------------------------------------------------
    (
        "register",
        "registration reads the ambient profile instead of the server's",
        MCP,
        '''    profile_key = server._profile_key
    scope = _mcp_registry_scope(profile_key)''',
        '''    profile_key = _mcp_profile_key()
    scope = _mcp_registry_scope(profile_key)''',
        SCOPE_TESTS,
    ),
    (
        "register",
        "registry.register() is called without a scope (eager path)",
        MCP,
        '''            description=candidate["schema"]["description"],
            scope=scope,
        )''',
        '''            description=candidate["schema"]["description"],
        )''',
        SCOPE_TESTS,
    ),
    (
        "register",
        "the post-register ownership check re-reads ambiently",
        MCP,
        '''        if registry.get_toolset_for_tool(
            registry_name, scope=scope
        ) != toolset_name:
            logger.error(''',
        '''        if registry.get_toolset_for_tool(registry_name) != toolset_name:
            logger.error(''',
        SCOPE_TESTS,
    ),
    (
        "register",
        "provenance is tracked against the ambient profile",
        MCP,
        '''        _track_mcp_tool_server(registry_name, name, profile_key)
        registered_names.append(registry_name)

    if registered_names:
        registry.register_toolset_alias(name, toolset_name, scope=scope)
        # Write-through (#56832)''',
        '''        _track_mcp_tool_server(registry_name, name)
        registered_names.append(registry_name)

    if registered_names:
        registry.register_toolset_alias(name, toolset_name, scope=scope)
        # Write-through (#56832)''',
        SCOPE_TESTS,
    ),
    (
        "register",
        "the provenance map goes back to a plain process-global dict",
        MCP,
        '''_mcp_tool_server_names: _ProfileScopedDict = _ProfileScopedDict()''',
        '''_mcp_tool_server_names: Dict[str, str] = {}''',
        SCOPE_TESTS + " " + ISO_TESTS,
    ),
    (
        "register-lazy",
        "the lazy schema-cache path registers without a scope",
        MCP,
        '''            description=schema["description"],
            scope=scope,
        )''',
        '''            description=schema["description"],
        )''',
        SCOPE_TESTS,
    ),
    # NOT a mutation: on the lazy path ``profile_key`` IS ``_mcp_profile_key()``,
    # read at function entry on the same thread, so passing it explicitly and
    # omitting it are the same call. Tried as a mutation and it survived, as
    # it must. Recorded here rather than deleted, so the next person does not
    # spend the same twenty minutes deciding whether it is a missing test.
    (
        "register-lazy",
        "the lazy path registers its server-name alias globally",
        MCP,
        '''        registry.register_toolset_alias(name, toolset_name, scope=scope)
        with _lock:
            _lazy_server_configs[name] = dict(config)''',
        '''        registry.register_toolset_alias(name, toolset_name)
        with _lock:
            _lazy_server_configs[name] = dict(config)''',
        SCOPE_TESTS,
    ),
    (
        "register-lazy",
        "the lazy path's utility tools register without a scope",
        MCP,
        '''            description=schema.get("description") or "",
            scope=scope,
        )''',
        '''            description=schema.get("description") or "",
        )''',
        SCOPE_TESTS,
    ),
    # -- teardown -----------------------------------------------------
    (
        "teardown",
        "MCP teardown deregisters without naming its scope",
        MCP,
        '''            registry.deregister(tool_name, scope=scope)
            _forget_mcp_tool_server(tool_name, self._profile_key)''',
        '''            registry.deregister(tool_name)
            _forget_mcp_tool_server(tool_name, self._profile_key)''',
        SCOPE_TESTS,
    ),
    (
        "teardown",
        "MCP teardown uses the ambient profile instead of the server's",
        MCP,
        '''        scope = _mcp_registry_scope(self._profile_key)
        for tool_name in list(getattr(self, "_registered_tool_names", [])):''',
        '''        scope = _mcp_registry_scope()
        for tool_name in list(getattr(self, "_registered_tool_names", [])):''',
        SCOPE_TESTS,
    ),
    (
        "teardown",
        "deregister() ignores the scope it was given",
        REG,
        '''            effective_scope = scope if scope is not None else caller_scope''',
        '''            effective_scope = caller_scope''',
        SCOPE_TESTS,
    ),
    (
        "teardown",
        "a scoped teardown pops the process-global toolset check again",
        REG,
        '''                if effective_scope is None:
                    self._toolset_checks.pop(entry.toolset, None)''',
        '''                self._toolset_checks.pop(entry.toolset, None)''',
        SCOPE_TESTS,
    ),
    (
        "teardown",
        "a scoped teardown drops aliases from the global partition",
        REG,
        '''                self._drop_aliases_for_toolset(entry.toolset, effective_scope)''',
        '''                self._drop_aliases_for_toolset(entry.toolset, None)''',
        SCOPE_TESTS,
    ),
    # -- the alias map ------------------------------------------------
    (
        "alias",
        "MCP registers its server-name alias globally (eager path)",
        MCP,
        '''        registry.register_toolset_alias(name, toolset_name, scope=scope)
        # Write-through (#56832)''',
        '''        registry.register_toolset_alias(name, toolset_name)
        # Write-through (#56832)''',
        SCOPE_TESTS,
    ),
    (
        "alias",
        "the scoped alias overlay is never consulted",
        REG,
        '''        merged = dict(self._toolset_aliases)
        merged.update(self._scoped_toolset_aliases.get(active_scope, {}))
        return merged''',
        '''        merged = dict(self._toolset_aliases)
        return merged''',
        ALIAS_TESTS,
    ),
    (
        "alias",
        "a scoped alias is written to the global map",
        REG,
        '''            target = (
                self._toolset_aliases
                if scope is None
                else self._scoped_toolset_aliases.setdefault(scope, {})
            )''',
        '''            target = self._toolset_aliases''',
        ALIAS_TESTS,
    ),
    # -- the toolset-membership memo ----------------------------------
    (
        "memo",
        "the membership memo drops the scope from its lookup AND write key",
        TS,
        '''*_registry_generation(), _registry_scope_key())''',
        '''*_registry_generation(), None)''',
        ALIAS_TESTS,
    ),
    # ("the memo write key disagrees with the lookup key" is gone: since the
    # 0.21.1 port the write reuses the lookup's ``memo_key`` object, so there
    # is no second key to drift.)
    # -- what the skeptical review forced -----------------------------
    (
        "review",
        "a plugin may name another profile's scope on deregister",
        REG,
        '''            if (
                scope is not None
                and caller_owner is not None
                and scope != caller_scope
            ):
                raise PermissionError(
                    f"Plugin module {caller_mod!r} (scope {caller_scope!r}) "
                    f"cannot deregister from another profile's scope "
                    f"{scope!r}."
                )
''',
        '''''',
        SCOPE_TESTS,
    ),
    (
        "review",
        "an empty scope stays a write-only partition",
        REG,
        '''        return None if scope == "" else scope''',
        '''        return scope''',
        SCOPE_TESTS,
    ),
    (
        "review",
        "restore_registration drops aliases from the scope only",
        REG,
        '''                    self._drop_aliases_for_toolset(toolset, None)
                    if scope is not None:
                        self._drop_aliases_for_toolset(toolset, scope)''',
        '''                    self._drop_aliases_for_toolset(toolset, scope)''',
        SCOPE_TESTS,
    ),
    (
        "review",
        "the alias collision check consults the ambient overlay again",
        REG,
        '''            existing = target.get(alias)''',
        '''            existing = self._merged_aliases(scope).get(alias)''',
        ALIAS_TESTS,
    ),
    (
        "review",
        "trust metadata is recorded against the ambient profile",
        MCP,
        '''    _record_tool_trust_metadata(name, config, server._tools, profile_key)''',
        '''    _record_tool_trust_metadata(name, config, server._tools)''',
        SCOPE_TESTS,
    ),
    (
        "review",
        "an unscoped registration goes back to being silent",
        MCP,
        '''        logger.error(
            "MCP registration has no profile scope while multiplexing is "
            "active. These tools will be registered into an isolated "
            "partition and advertised to NO profile. Run discovery inside "
            "a profile scope (see gateway.run._profile_runtime_scope)."
        )
''',
        '''''',
        SCOPE_TESTS,
    ),
    (
        "review",
        "the plugin zombie sweep goes back to being a silent no-op",
        PLUGINS,
        '''                            tool_registry.deregister(
                                tool_name, scope=self.scope_key
                            )''',
        '''                            tool_registry.deregister(tool_name)''',
        PLUGIN_TESTS,
    ),
    # -- the fixture itself -------------------------------------------
    (
        "fixture",
        "the scope suite restores partitions by hand instead of enumerating",
        SCOPE_TEST_FILE,
        '''    return {
        attr: getattr(mcp, attr)
        for attr in dir(mcp)
        if isinstance(
            getattr(mcp, attr), (mcp._ProfileScopedDict, mcp._ProfileScopedSet)
        )
    }''',
        '''    return {
        attr: getattr(mcp, attr)
        for attr in ("_servers", "_mcp_tool_server_names")
    }''',
        SCOPE_TESTS + " " + MCP_TESTS,
    ),
    # -- the sweep ----------------------------------------------------
    (
        "sweep",
        "a partitioned registry is left out of the isolation fixture's list",
        ISO_TEST,
        '''    "_mcp_tool_server_names",
]''',
        ''']''',
        ISO_TESTS,
    ),
]


def run(paths):
    env = dict(os.environ)
    return subprocess.run(
        [sys.executable, "-m", "pytest", *paths.split(), "-x", "-q",
         "-p", "no:cacheprovider"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )


def main():
    baseline = run(" ".join(sorted({m[5] for m in MUTATIONS})))
    if baseline.returncode != 0:
        print("BASELINE IS RED — fix that before trusting any mutation below.")
        print(baseline.stdout[-4000:])
        return 1
    print(f"baseline green ({len(MUTATIONS)} mutations to run)\n")

    survived, caught, broken = [], 0, []
    for mutation in MUTATIONS:
        group, desc, path, old, new, target = mutation[:6]
        occurrences = mutation[6] if len(mutation) > 6 else 1
        original = io.open(path, encoding="utf-8").read()
        if original.count(old) != occurrences:
            broken.append((group, desc, path, original.count(old)))
            print(f"  ANCHOR DRIFT  [{group}] {desc}")
            continue
        try:
            io.open(path, "w", encoding="utf-8").write(
                original.replace(old, new)
            )
            result = run(target)
        finally:
            io.open(path, "w", encoding="utf-8").write(original)
        if result.returncode == 0:
            survived.append((group, desc))
            print(f"  SURVIVED      [{group}] {desc}")
        else:
            caught += 1
            print(f"  caught        [{group}] {desc}")

    print(f"\n{caught}/{len(MUTATIONS)} caught")
    if broken:
        print(f"\n{len(broken)} ANCHOR(S) DRIFTED — the guard is stale, not passing:")
        for group, desc, path, n in broken:
            print(f"  [{group}] {desc}  ({os.path.relpath(path, ROOT)}: {n} matches)")
    if survived:
        print(f"\n{len(survived)} MUTATION(S) SURVIVED — untested rules:")
        for group, desc in survived:
            print(f"  [{group}] {desc}")
    return 1 if (survived or broken) else 0


if __name__ == "__main__":
    sys.exit(main())
