"""MCP tool REGISTRATION is profile-scoped (HERMES-SURFACES §7, item 4b).

Execution has been profile-partitioned since ``380e4caba7`` — ``_servers``,
the connect/error/breaker maps and the lazy-config cache all live in
per-profile partitions. The tool NAMES AND SCHEMAS the model is shown were
not: ``tools/mcp_tool.py`` registered into ``tools/registry.py`` with no
``scope=``, so a second profile's ``mcp__github__*`` entries overwrote the
first's advertised description while dispatch went to the right server. That
commit's own message said so and left it open.

Two things these tests exist to keep honest, both named in the plan as ways
the fix could LOOK done without being done:

* **Teardown must be scoped too.** ``registry.deregister`` picks its
  partition from the CALLER when no scope is given, and ``mcp_tool`` is not
  a plugin module — so a scoped registration paired with an unscoped
  deregistration is a silent no-op that advertises a dead server's tools
  forever. Tested with two scopes alive at once, not two registrations.
* **Schemas, not just dispatch.** If profile B's description can still reach
  profile A, the model reads the wrong tool and the bug survives the fix.

And the direction of failure matters. ``ToolRegistry.current_scope_key()`` is
``hermes_home_key()`` *ungated* while ``_mcp_profile_key()`` returns ``""``
whenever multiplexing is off; mapping one onto the other wrongly makes tools
VANISH, which reads as a broken server rather than as a leak. That mapping
lives in ``_mcp_registry_scope`` and is tested first.
"""
import contextlib

import pytest

import agent.secret_scope as secret_scope
import toolsets
import tools.mcp_tool as mcp
from hermes_constants import (
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.registry import registry


@contextlib.contextmanager
def _profile(home):
    """Mimic ``gateway.run._profile_runtime_scope`` for one profile.

    The same two seams the multiplexed inbound path installs: the
    context-local HERMES_HOME override and the profile's secret scope.
    """
    home_token = set_hermes_home_override(str(home))
    secret_token = secret_scope.set_secret_scope(
        {"GITHUB_TOKEN": f"token-for-{home.name}"}
    )
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


# The MCP-side half of these checks drove the fork's per-profile PARTITIONS
# (``_mcp_profile_key``, ``server._profile_key``, ``_register_server_tools``
# addressing a partition). That design was superseded on the 0.21.1 port by
# upstream's scope-key model (see tests/tools/test_mcp_profile_isolation.py);
# the registry-only checks below still run against the ported registry.
_NO_PARTITIONS = not hasattr(mcp, "_ProfileScopedDict")
_SUPERSEDED = pytest.mark.skipif(
    _NO_PARTITIONS, reason="per-profile MCP partitions superseded by upstream's scope-key design")


def _mcp_partitioned():
    """Every profile-partitioned registry in ``tools.mcp_tool``, by name.

    Asked of the module rather than listed by hand. The hand-written list in
    ``test_mcp_profile_isolation.py`` went stale the moment a registry was
    added, and this fixture made the same mistake on its first draft: it
    saved four things by name, missed ``_lazy_server_tool_names``, and left
    a phantom lazy server behind that failed
    ``TestMCPSelectiveToolLoading::test_enabled_false_skips_connection_attempt``
    two files later. Enumerating is the only version that stays true.
    """
    # The per-profile partitions were superseded on the 0.21.1 port (see
    # tests/tools/test_mcp_profile_isolation.py); with them gone there is
    # nothing MCP-side to save, and the registry-level checks below still run.
    kinds = tuple(k for k in (getattr(mcp, "_ProfileScopedDict", None),
                              getattr(mcp, "_ProfileScopedSet", None)) if k is not None)
    if not kinds:
        return {}
    return {
        attr: getattr(mcp, attr)
        for attr in dir(mcp)
        if isinstance(getattr(mcp, attr), kinds)
    }


@pytest.fixture(autouse=True)
def _clean_registries():
    """Restore both the MCP partitions and the process-wide tool registry.

    ``tools.registry.registry`` is a singleton shared with every other test
    in the suite, so anything registered here must be removed again —
    including from the scoped partitions and the alias overlay, which no
    other fixture knows about.
    """
    was_multiplex = secret_scope.is_multiplex_active()
    saved_tools = dict(registry._tools)
    saved_scoped = {k: dict(v) for k, v in registry._scoped_tools.items()}
    saved_checks = dict(registry._toolset_checks)
    saved_aliases = dict(registry._toolset_aliases)
    saved_scoped_aliases = {
        k: dict(v) for k, v in registry._scoped_toolset_aliases.items()
    }
    saved_memo = dict(toolsets._resolve_toolset_memo)
    saved_mcp = {
        name: {
            key: (dict(part) if hasattr(part, "items") else set(part))
            for key, part in container.partitions().items()
        }
        for name, container in _mcp_partitioned().items()
    }
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(was_multiplex)
        registry._tools.clear()
        registry._tools.update(saved_tools)
        registry._scoped_tools.clear()
        registry._scoped_tools.update(saved_scoped)
        registry._toolset_checks.clear()
        registry._toolset_checks.update(saved_checks)
        registry._toolset_aliases.clear()
        registry._toolset_aliases.update(saved_aliases)
        registry._scoped_toolset_aliases.clear()
        registry._scoped_toolset_aliases.update(saved_scoped_aliases)
        toolsets._resolve_toolset_memo.clear()
        toolsets._resolve_toolset_memo.update(saved_memo)
        for name, container in _mcp_partitioned().items():
            for part in container.partitions().values():
                part.clear()
            for key, value in saved_mcp.get(name, {}).items():
                container.for_key(key).update(value)


@pytest.fixture
def homes(tmp_path):
    a = tmp_path / "profile-a"
    b = tmp_path / "profile-b"
    a.mkdir()
    b.mkdir()
    return a, b


def _tool(name, description):
    """A REAL ``mcp.types.Tool``.

    Deliberately not a duck-typed double. ``_convert_mcp_schema`` reads the
    input schema through ``mcp_field(tool, "input_schema", "inputSchema")``
    because mcp 2.0 renamed the field and kept camelCase only as a
    serialization alias — a stand-in carrying a plain ``inputSchema``
    attribute passes against code that cannot read the real object.
    """
    from mcp.types import Tool

    return Tool(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}},
    )


class _Caps:
    """``initialize_result.capabilities`` advertising resources and prompts.

    Without this ``_select_utility_schemas`` takes its legacy fallback and
    rejects all four utility stubs, so the eager path's utility registration
    — which has its own ``registry.register`` call — is never entered and the
    test silently covers half the function.
    """

    resources = object()
    prompts = object()


class _Init:
    capabilities = _Caps()


class _FakeServer:
    """Stand-in for a connected ``MCPServerTask`` (no subprocess, no I/O)."""

    def __init__(self, name, profile_key, tools, utilities=False):
        self.name = name
        self._profile_key = profile_key
        self._tools = list(tools)
        self.session = object()
        self.tool_timeout = 5.0
        self.initialize_result = _Init() if utilities else None
        self._registered_tool_names = []

    def _is_recycled_stdio(self):
        return False

    # The REAL teardown, borrowed rather than reimplemented. A fake copy of
    # ``_deregister_tools`` would pass whatever the copy did, which is the
    # check-that-cannot-fail shape: it is precisely this method's choice of
    # partition that is under test.
    _deregister_tools = mcp.MCPServerTask._deregister_tools


def _register(server, config=None):
    server._registered_tool_names = mcp._register_server_tools(
        server.name, server, config or {}
    )
    return server._registered_tool_names


# ---------------------------------------------------------------------------
# The key translation — where a wrong answer makes tools vanish
# ---------------------------------------------------------------------------

@_SUPERSEDED
def test_root_key_becomes_global_scope_not_the_empty_scope(homes):
    """Multiplex OFF ⇒ ``None`` (process-global), never the ``""`` partition.

    ``_mcp_profile_key`` answers ``""`` here on purpose, so that the many
    single-profile paths which install a HERMES_HOME override (plugins, the
    ``mcp_startup`` discovery thread, ``bot_mode_probe``, the desktop
    backend) do not each strand a server behind its own key. But the tool
    registry's own key is ``hermes_home_key()`` UNGATED — it never answers
    ``""`` — so registering into scope ``""`` would put every MCP tool in a
    partition no reader can name. Every Mac would advertise zero MCP tools.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(False)

    assert mcp._mcp_registry_scope() is None
    with _profile(home_a):
        assert mcp._mcp_profile_key() == mcp._ROOT_PROFILE_KEY
        assert mcp._mcp_registry_scope() is None
    with _profile(home_b):
        assert mcp._mcp_registry_scope() is None


def test_profile_key_becomes_the_registry_scope_under_multiplex(homes):
    """Multiplex ON ⇒ the scope is the profile home, the registry's own key."""
    home_a, _ = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        assert mcp._mcp_registry_scope() == hermes_home_key(str(home_a))
        # The two notions of "who is this" agree here, and that agreement is
        # the whole reason a scoped registration is readable by the turn.
        assert mcp._mcp_registry_scope() == registry.current_scope_key()


@_SUPERSEDED
def test_unscoped_key_is_not_collapsed_to_global():
    """Multiplex ON with no scope installed ⇒ an unreadable partition.

    Fail-closed on purpose. The alternative — registering globally — hands
    one tenant's tool list to every other, which is the failure the whole
    partitioning exists to prevent. Invisible tools are the cheaper bug.
    """
    secret_scope.set_multiplex_active(True)

    assert mcp._mcp_registry_scope() == mcp._UNSCOPED_PROFILE_KEY


# ---------------------------------------------------------------------------
# Single profile: byte-identical to before
# ---------------------------------------------------------------------------

@_SUPERSEDED
def test_single_profile_registration_stays_process_global(homes):
    """The property every ordinary Mac depends on, asserted structurally."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(False)

    with _profile(home_a):
        server = _FakeServer("github", mcp._mcp_profile_key(), [_tool("list_repos", "List repos.")])
        names = _register(server)

    assert names == ["mcp__github__list_repos"]
    # In the global map, not in any scoped partition.
    assert "mcp__github__list_repos" in registry._tools
    assert not any(
        "mcp__github__list_repos" in part
        for part in registry._scoped_tools.values()
    )
    # And visible from a different override, exactly as before.
    with _profile(home_b):
        assert registry.get_entry("mcp__github__list_repos") is not None


# ---------------------------------------------------------------------------
# Two profiles: names, schemas and provenance stay apart
# ---------------------------------------------------------------------------

@_SUPERSEDED
def test_two_profiles_see_their_own_schema_not_each_others(homes):
    """The bug in one assertion: same tool name, two descriptions.

    Scoping dispatch while leaving the advertised schema shared would pass a
    "does the call reach the right server" test and still show the model the
    other tenant's description.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        ))
    with _profile(home_b):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "B's repositories.")]
        ))

    with _profile(home_a):
        entry = registry.get_entry("mcp__github__list_repos")
        assert entry is not None
        assert entry.schema["description"] == "A's repositories."
    with _profile(home_b):
        entry = registry.get_entry("mcp__github__list_repos")
        assert entry is not None
        assert entry.schema["description"] == "B's repositories."


@_SUPERSEDED
def test_a_profile_without_the_server_sees_no_tool(homes):
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        ))

    with _profile(home_b):
        assert registry.get_entry("mcp__github__list_repos") is None
        assert mcp.get_registered_mcp_server_names() == set()
        assert mcp.has_registered_mcp_tools() is False
    with _profile(home_a):
        assert mcp.get_registered_mcp_server_names() == {"github"}
        assert mcp.has_registered_mcp_tools() is True


@_SUPERSEDED
def test_registration_follows_the_server_not_the_ambient_context(homes):
    """``server._profile_key``, not whoever happens to be on the thread.

    ``MCPServerTask.run()`` is long-lived and re-publishes its tools after
    every reconnect, carrying the context it was CREATED in. Registering
    against the ambient key would file those tools under whichever profile's
    turn happened to drive the reconnect.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        owner_key = mcp._mcp_profile_key()
    server = _FakeServer("github", owner_key, [_tool("list_repos", "A's repositories.")])

    # Register while profile B is the ambient one.
    with _profile(home_b):
        names = _register(server)

    # The returned list matters as much as the registry: the post-register
    # ownership verification re-reads the registry, and if THAT read is
    # ambient it sees an empty partition, calls the registration rejected,
    # and drops the tool from the result while leaving the entry behind.
    assert names == ["mcp__github__list_repos"]
    with _profile(home_a):
        assert registry.get_entry("mcp__github__list_repos") is not None
        assert dict(mcp._mcp_tool_server_names) == {
            "mcp__github__list_repos": "github"
        }
    with _profile(home_b):
        assert registry.get_entry("mcp__github__list_repos") is None
        assert dict(mcp._mcp_tool_server_names) == {}


# ---------------------------------------------------------------------------
# Teardown — two scopes ALIVE at once, not two registrations
# ---------------------------------------------------------------------------

@_SUPERSEDED
def test_teardown_removes_only_its_own_profiles_tools(homes):
    """Profile A's shutdown must not unregister profile B's LIVE tool."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        server_a = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        )
        _register(server_a)
    with _profile(home_b):
        server_b = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "B's repositories.")]
        )
        _register(server_b)

    server_a._deregister_tools()

    with _profile(home_a):
        assert registry.get_entry("mcp__github__list_repos") is None
        assert mcp.get_registered_mcp_server_names() == set()
    with _profile(home_b):
        entry = registry.get_entry("mcp__github__list_repos")
        assert entry is not None, "profile A's teardown removed profile B's live tool"
        assert entry.schema["description"] == "B's repositories."
        assert mcp.get_registered_mcp_server_names() == {"github"}


@_SUPERSEDED
def test_teardown_actually_removes_the_tool_in_its_own_profile(homes):
    """The other direction: a scoped teardown must not be a silent no-op.

    Without ``deregister(scope=)`` this is exactly what happened — the tools
    went into a profile partition and teardown looked for them in
    ``_tools``, found nothing, and returned, leaving a dead server's tools
    advertised for the life of the process.
    """
    home_a, _ = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        server = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        )
        _register(server)
        assert registry.get_entry("mcp__github__list_repos") is not None

    server._deregister_tools()

    with _profile(home_a):
        assert registry.get_entry("mcp__github__list_repos") is None
        assert mcp.has_registered_mcp_tools() is False
    assert server._registered_tool_names == []


@_SUPERSEDED
def test_unscoped_deregister_cannot_reach_a_scoped_tool(homes):
    """Why ``scope=`` had to be added rather than relying on the caller.

    ``deregister`` derives its partition from the calling MODULE's plugin
    ownership, and ``tools.mcp_tool`` is not a plugin — so the caller-derived
    scope is always ``None``. This pins that seam: the plain call is a no-op
    against a scoped entry, which is precisely why the MCP paths must name
    their scope.
    """
    home_a, _ = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        server = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        )
        _register(server)
        registry.deregister("mcp__github__list_repos")
        assert registry.get_entry("mcp__github__list_repos") is not None

        registry.deregister(
            "mcp__github__list_repos", scope=mcp._mcp_registry_scope()
        )
        assert registry.get_entry("mcp__github__list_repos") is None


@_SUPERSEDED
def test_single_profile_teardown_is_unchanged(homes):
    """Multiplex off: register and tear down through the global map."""
    home_a, _ = homes
    secret_scope.set_multiplex_active(False)

    with _profile(home_a):
        server = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "Repos.")]
        )
        _register(server)
        assert "mcp__github__list_repos" in registry._tools
        server._deregister_tools()
        assert "mcp__github__list_repos" not in registry._tools
        assert registry.get_entry("mcp__github__list_repos") is None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@_SUPERSEDED
def test_provenance_is_partitioned(homes):
    """``_mcp_tool_server_names`` is what policy reads, so it partitions too.

    ``is_mcp_tool_parallel_safe`` and the capability-aware prompt both read
    it. A shared map means every profile sees every profile's entries, so
    ``get_registered_mcp_server_names()`` — which builds the capability note
    in the system prompt — tells one tenant which servers another has
    connected.

    (An earlier draft of this docstring claimed the failure was profile B
    "retargeting" profile A's provenance for a colliding registry name.
    That is not what this test shows and it is not the cheap failure: a
    registry-name collision needs both profiles to have server names that
    normalize together, whereas simple co-presence needs nothing at all.)
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        _register(_FakeServer(
            "docs", mcp._mcp_profile_key(), [_tool("search", "Search A's docs.")]
        ))
    with _profile(home_b):
        _register(_FakeServer(
            "wiki", mcp._mcp_profile_key(), [_tool("search", "Search B's wiki.")]
        ))

    # Whole-dict equality, not membership: a shared map puts BOTH entries in
    # both views, and `"mcp__docs__search" in ...` would pass against it.
    with _profile(home_a):
        assert dict(mcp._mcp_tool_server_names) == {"mcp__docs__search": "docs"}
        assert mcp.get_registered_mcp_server_names() == {"docs"}
    with _profile(home_b):
        assert dict(mcp._mcp_tool_server_names) == {"mcp__wiki__search": "wiki"}
        assert mcp.get_registered_mcp_server_names() == {"wiki"}


# ---------------------------------------------------------------------------
# The two caches that would have kept names crossing anyway
# ---------------------------------------------------------------------------

@_SUPERSEDED
def test_server_name_does_not_become_a_toolset_in_another_profile(homes):
    """An MCP server's raw name is registered as a toolset ALIAS.

    Scoping the tools while leaving the alias map global still answers "does
    a toolset called ``github`` exist?" with yes for every tenant — which
    discloses what another person has connected even though the tools
    themselves are gone.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        ))

    with _profile(home_a):
        assert registry.get_toolset_alias_target("github") == "mcp-github"
        assert toolsets.validate_toolset("github") is True
    with _profile(home_b):
        assert registry.get_toolset_alias_target("github") is None
        assert "github" not in registry.get_registered_toolset_aliases()
        assert toolsets.validate_toolset("github") is False


@_SUPERSEDED
def test_single_profile_alias_is_still_global(homes):
    """Multiplex off: the alias lands in the global map, as before."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(False)

    with _profile(home_a):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "Repos.")]
        ))

    assert registry._toolset_aliases.get("github") == "mcp-github"
    assert not registry._scoped_toolset_aliases
    with _profile(home_b):
        assert registry.get_toolset_alias_target("github") == "mcp-github"


@_SUPERSEDED
def test_toolset_membership_memo_does_not_serve_another_profile(homes):
    """``toolsets._resolve_toolset_memo`` is keyed on the registry scope.

    It memoizes ``registry.get_tool_names_for_toolset`` — read ambiently —
    against ``(name, include_registry, id(registry), generation)``. With the
    generation unchanged, which is the steady state, profile A's answer was
    returned verbatim to profile B. Reading A's list FIRST is the point of
    this test: it is the warm cache that leaks, not the cold one.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        ))
        assert toolsets.resolve_toolset("mcp-github") == ["mcp__github__list_repos"]

    with _profile(home_b):
        assert toolsets.resolve_toolset("mcp-github") == []


@_SUPERSEDED
def test_scoped_teardown_never_pops_a_global_toolset_check(homes):
    """``deregister(scope=...)`` must not mutate ``_toolset_checks``.

    That map is process-global and NOT partitioned, and ``register`` already
    refuses to write to it from a scoped registration
    (``if scope is None and check_fn ...``). Deregistration had no matching
    gate, so a scoped teardown could remove a check only a global
    registration could have created.

    Stated honestly: no realistic single-process sequence reaches that today,
    because a scoped registration never creates the entry in the first place
    and a surviving global tool keeps ``toolset_still_exists`` true. This
    asserts the gate itself rather than pretending to reproduce a live bug —
    the asymmetry is the defect, and it is the kind that only bites once
    something else changes.
    """
    home_a, _ = homes
    secret_scope.set_multiplex_active(True)

    sentinel = lambda: True
    registry._toolset_checks["mcp-github"] = sentinel

    with _profile(home_a):
        server = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        )
        _register(server)

    server._deregister_tools()

    assert registry._toolset_checks.get("mcp-github") is sentinel


@_SUPERSEDED
def test_global_teardown_still_drops_its_toolset_check(homes):
    """And the gate must not break the unscoped path it wraps."""
    home_a, _ = homes
    secret_scope.set_multiplex_active(False)

    with _profile(home_a):
        server = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "Repos.")]
        )
        _register(server)
        assert "mcp-github" in registry._toolset_checks
        server._deregister_tools()
        assert "mcp-github" not in registry._toolset_checks
        assert registry.get_toolset_alias_target("github") is None


@_SUPERSEDED
def test_scoped_teardown_leaves_another_profiles_alias_alone(homes):
    """A's shutdown must not drop the alias B's server still owns."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        server_a = _FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        )
        _register(server_a)
    with _profile(home_b):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "B's repositories.")]
        ))

    server_a._deregister_tools()

    with _profile(home_a):
        assert registry.get_toolset_alias_target("github") is None
    with _profile(home_b):
        assert registry.get_toolset_alias_target("github") == "mcp-github"


# ---------------------------------------------------------------------------
# The lazy path: registered from the on-disk schema cache, nothing spawned
# ---------------------------------------------------------------------------

def _cache_entry(tool_name, description):
    """A cached manifest carrying a real tool AND a utility tool.

    The utility half matters: resource/prompt stubs are registered by a
    second loop with its own ``registry.register`` call, and a manifest with
    an empty ``utility_tools`` list never enters it — so the test would
    silently cover half the path.
    """
    return {
        "tools": [{
            "name": tool_name,
            "description": description,
            "inputSchema": {"type": "object", "properties": {}},
        }],
        "utility_tools": [{
            "handler_key": "list_resources",
            "schema": {
                "name": "mcp__github__list_resources",
                "description": "List resources.",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    }


@_SUPERSEDED
def test_lazy_cache_registration_is_scoped_too(homes):
    """``_register_from_cache_sync`` has no server object to ask.

    Lazy startup (#56832) registers a server's tools from the on-disk
    manifest without spawning anything, so there is no ``MCPServerTask`` and
    no ``_profile_key``. It takes the ambient profile instead — correct,
    because its caller runs on the turn's own thread inside the profile
    scope and reads ``_servers`` ambiently a few lines earlier. Scoping the
    eager path and forgetting this one would leave every lazy server's tools
    crossing profiles, which is the whole point on a Mac where lazy is what
    avoids spawning a process per profile.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        names = mcp._register_from_cache_sync(
            "github", {"command": "npx"}, _cache_entry("list_repos", "A's repositories.")
        )
    assert names == [
        "mcp__github__list_repos", "mcp__github__list_resources",
    ]

    with _profile(home_a):
        entry = registry.get_entry("mcp__github__list_repos")
        assert entry is not None
        assert entry.schema["description"] == "A's repositories."
        assert registry.get_toolset_alias_target("github") == "mcp-github"
        assert registry.get_entry("mcp__github__list_resources") is not None
        assert dict(mcp._mcp_tool_server_names) == {
            "mcp__github__list_repos": "github",
            "mcp__github__list_resources": "github",
        }
    with _profile(home_b):
        assert registry.get_entry("mcp__github__list_repos") is None
        assert registry.get_entry("mcp__github__list_resources") is None
        assert registry.get_toolset_alias_target("github") is None
        assert dict(mcp._mcp_tool_server_names) == {}


@_SUPERSEDED
def test_lazy_cache_registration_is_global_without_multiplexing(homes):
    """And unchanged on every ordinary Mac."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(False)

    with _profile(home_a):
        mcp._register_from_cache_sync(
            "github", {"command": "npx"}, _cache_entry("list_repos", "Repos.")
        )

    assert "mcp__github__list_repos" in registry._tools
    assert not registry._scoped_toolset_aliases
    with _profile(home_b):
        assert registry.get_entry("mcp__github__list_repos") is not None


# ---------------------------------------------------------------------------
# What the skeptical review of the first draft turned up
# ---------------------------------------------------------------------------

@_SUPERSEDED
def test_the_eager_utility_branch_is_scoped_too(homes):
    """Resource/prompt stubs register through their own ``register`` call.

    A server advertising the ``resources``/``prompts`` capabilities gets four
    generated utility tools, registered in a second loop. With
    ``initialize_result = None`` that loop is never entered, so the first
    draft of this file left those ``scope=`` sites untested.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        names = _register(_FakeServer(
            "github",
            mcp._mcp_profile_key(),
            [_tool("list_repos", "A's repositories.")],
            utilities=True,
        ))

    assert "mcp__github__list_resources" in names
    assert "mcp__github__list_prompts" in names
    with _profile(home_a):
        assert registry.get_entry("mcp__github__list_resources") is not None
    with _profile(home_b):
        assert registry.get_entry("mcp__github__list_resources") is None
        assert registry.get_entry("mcp__github__list_prompts") is None


@_SUPERSEDED
def test_trust_metadata_lands_in_the_same_profile_as_the_tools(homes):
    """Trust is read at call time and fails OPEN, so a split is dangerous.

    ``_trust_gate_check`` reads ``_server_trust_levels.get(name, _TRUST_FULL)``
    under the turn's own key. Registration takes its key from the server;
    if the trust write took its key from the thread instead, an ``untrusted``
    server's write-capable tools would resolve to the FULL default in the
    profile that actually has them — no elicitation prompt, nothing logged.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        owner_key = mcp._mcp_profile_key()
    server = _FakeServer(
        "github", owner_key, [_tool("delete_repo", "Delete a repository.")]
    )

    # Registered while profile B is the ambient one, as a reconnecting
    # long-lived server task can be.
    with _profile(home_b):
        _register(server, {"trust": "untrusted"})

    with _profile(home_a):
        assert mcp._server_trust_levels.get("github") == mcp._TRUST_UNTRUSTED
    with _profile(home_b):
        assert mcp._server_trust_levels.get("github") is None


@_SUPERSEDED
def test_unscoped_registration_says_so_loudly(homes, caplog):
    """The unscoped partition is invisible, so it must never be silent.

    It is reachable: a multiplexed gateway that runs discovery outside a
    profile scope registers every tool where no reader can name it, and the
    symptom is "this gateway shows no MCP tools" with nothing in the log.
    """
    import logging

    secret_scope.set_multiplex_active(True)
    with caplog.at_level(logging.ERROR, logger="tools.mcp_tool"):
        assert mcp._mcp_registry_scope() == mcp._UNSCOPED_PROFILE_KEY
    assert any(
        "advertised to NO profile" in record.getMessage()
        for record in caplog.records
    )


# --- tools/registry.py's own seams ----------------------------------------

@_SUPERSEDED
def test_a_plugin_cannot_deregister_from_another_profiles_scope(homes):
    """``scope=`` must not become a privilege-widening parameter.

    ``deregister`` skips its ownership checks entirely for ``mcp-*``
    toolsets, because dynamic discovery legitimately nukes and repaves its
    own tools. Meeting the new ``scope=`` parameter, that exemption let a
    plugin loaded in one profile delete every MCP tool another tenant was
    advertising — with the FIRST profile's operator opt-in governing the
    deletion, and nothing logged. On the parent commit this was unreachable:
    with no ``scope`` parameter a plugin was pinned to its own partition.

    The call has to come from a frame whose ``__name__`` is the plugin's:
    ``deregister`` takes only a tool name, so ``_caller_module`` reads
    ``sys._getframe(2).f_globals["__name__"]`` and nothing else identifies
    the caller. Setting ``__module__`` on a function defined here would not
    do it — that is not what the frame carries.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_b):
        victim_key = mcp._mcp_profile_key()
        _register(_FakeServer(
            "github", victim_key, [_tool("list_repos", "B's repositories.")]
        ))
    with _profile(home_a):
        scope_a = mcp._mcp_profile_key()

    module_name = "hermes_plugins.probe_scope_widening"
    plugin_globals = {"__name__": module_name, "registry": registry}
    exec(
        "def attempt(name, scope):\n"
        "    registry.deregister(name, scope=scope)\n",
        plugin_globals,
    )
    registry._plugin_module_scopes.setdefault(module_name, set()).add(scope_a)
    try:
        with pytest.raises(PermissionError):
            plugin_globals["attempt"]("mcp__github__list_repos", victim_key)
        # Its own partition is still allowed.
        plugin_globals["attempt"]("mcp__github__list_repos", scope_a)
    finally:
        registry._plugin_module_scopes.pop(module_name, None)

    with _profile(home_b):
        assert registry.get_entry("mcp__github__list_repos") is not None


def test_empty_scope_is_normalized_to_global():
    """``scope=""`` was a slot that could not read back its own write.

    Writes select their partition with ``is not None`` while reads fall back
    with ``scope or current_scope_key()``, so the empty string was a real
    partition for one and ambient for the other. And ``_ROOT_PROFILE_KEY``
    IS the empty string — the only thing standing between it and every
    ordinary Mac was one ``if`` in a helper two calls away.
    """
    secret_scope.set_multiplex_active(False)

    registry.register(
        name="_probe_empty_scope",
        toolset="mcp-probe",
        schema={"name": "_probe_empty_scope", "description": "", "parameters": {}},
        handler=lambda **kw: "",
        scope="",
    )
    assert "_probe_empty_scope" in registry._tools
    assert "" not in registry._scoped_tools
    assert registry.get_entry("_probe_empty_scope") is not None

    registry.deregister("_probe_empty_scope", scope="")
    assert registry.get_entry("_probe_empty_scope") is None


def test_restore_registration_drops_an_orphaned_global_alias():
    """Plugins register aliases globally while their tools are always scoped.

    So a partition-local drop found nothing and the alias outlived every
    tool it named: ``validate_toolset`` kept answering True for a toolset
    with zero tools, forever, after an unload or force reload.
    """
    secret_scope.set_multiplex_active(False)
    scope = registry.current_scope_key()

    registry.register(
        name="_probe_plugin_tool",
        toolset="_probe_pset",
        schema={"name": "_probe_plugin_tool", "description": "", "parameters": {}},
        handler=lambda **kw: "",
        scope=scope,
    )
    registry.register_toolset_alias("_probe_pset_alias", "_probe_pset")
    current = registry.snapshot_registration("_probe_plugin_tool", scope=scope)

    assert registry.restore_registration(
        "_probe_plugin_tool", current, None, scope=scope
    )
    assert registry.get_toolset_alias_target("_probe_pset_alias") is None


@_SUPERSEDED
def test_a_global_alias_registration_ignores_a_profiles_overlay(homes, caplog):
    """The collision warning must name the partition it is writing.

    ``_merged_aliases(None)`` falls back to the AMBIENT profile's overlay, so
    a global registration reported a collision against whichever tenant was
    on the thread — a false claim (the scoped entry still wins the merged
    read afterwards) that also prints another person's MCP server name into
    a shared log.
    """
    import logging

    home_a, _ = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        _register(_FakeServer(
            "github", mcp._mcp_profile_key(), [_tool("list_repos", "A's repositories.")]
        ))

        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            registry.register_toolset_alias("github", "hermes-github")
        assert not [
            r for r in caplog.records if "alias collision" in r.getMessage()
        ]
        # And the profile's own alias is untouched.
        assert registry.get_toolset_alias_target("github") == "mcp-github"
    assert registry._toolset_aliases.get("github") == "hermes-github"
