"""The registry's per-profile toolset-alias overlay, on its own.

An MCP server's raw name is registered as an alias for its ``mcp-{name}``
toolset. Left process-global, that alias would keep leaking WHICH SERVERS
another tenant has connected even once their tools are scoped — the alias
resolves, and the name is the disclosure. These checks used to ride inside
tests/tools/test_mcp_registry_scope.py through the fork's MCP partitions;
that design was superseded on the 0.21.1 port, so the registry rules it
relied on are pinned here without any MCP state at all.
"""
import logging

import pytest

from tools.registry import registry


@pytest.fixture(autouse=True)
def _restore_aliases():
    saved_global = dict(registry._toolset_aliases)
    saved_scoped = {k: dict(v) for k, v in registry._scoped_toolset_aliases.items()}
    try:
        yield
    finally:
        with registry._lock:
            registry._toolset_aliases.clear()
            registry._toolset_aliases.update(saved_global)
            registry._scoped_toolset_aliases.clear()
            registry._scoped_toolset_aliases.update(saved_scoped)


def test_a_scoped_alias_resolves_in_its_own_scope_and_nowhere_else():
    registry.register_toolset_alias("_probe_srv", "mcp-_probe_srv", scope="/profile/a")
    assert registry.get_toolset_alias_target("_probe_srv", scope="/profile/a") == "mcp-_probe_srv"
    assert registry.get_toolset_alias_target("_probe_srv", scope="/profile/b") is None
    assert registry.get_registered_toolset_aliases(scope="/profile/b").get("_probe_srv") is None


def test_a_scoped_alias_never_lands_in_the_global_map():
    registry.register_toolset_alias("_probe_srv", "mcp-_probe_srv", scope="/profile/a")
    assert "_probe_srv" not in registry._toolset_aliases
    assert registry._scoped_toolset_aliases["/profile/a"]["_probe_srv"] == "mcp-_probe_srv"


def test_the_empty_scope_is_the_global_map():
    registry.register_toolset_alias("_probe_srv", "mcp-_probe_srv", scope="")
    assert registry._toolset_aliases["_probe_srv"] == "mcp-_probe_srv"
    assert "" not in registry._scoped_toolset_aliases


def test_a_global_registration_ignores_a_profiles_overlay(caplog, monkeypatch):
    """The collision warning must name the partition it is writing.

    ``_merged_aliases(None)`` falls back to the AMBIENT profile's overlay, so a
    global registration reported a collision against whichever tenant was on
    the thread — a false claim (the scoped entry still wins the merged read
    afterwards) that also prints another person's MCP server name into a
    shared log.
    """
    monkeypatch.setattr(type(registry), "current_scope_key", staticmethod(lambda: "/profile/a"))
    registry.register_toolset_alias("_probe_srv", "mcp-_probe_srv", scope="/profile/a")
    with caplog.at_level(logging.WARNING, logger="tools.registry"):
        registry.register_toolset_alias("_probe_srv", "hermes-_probe_srv")
    assert not [r for r in caplog.records if "alias collision" in r.getMessage()]
    # The profile's own alias still wins the merged read...
    assert registry.get_toolset_alias_target("_probe_srv") == "mcp-_probe_srv"
    # ...and the global map holds the global one.
    assert registry._toolset_aliases["_probe_srv"] == "hermes-_probe_srv"


def test_dropping_a_toolsets_aliases_is_per_partition():
    registry.register_toolset_alias("_probe_srv", "mcp-_probe_srv", scope="/profile/a")
    registry.register_toolset_alias("_probe_srv", "mcp-_probe_srv")
    with registry._lock:
        registry._drop_aliases_for_toolset("mcp-_probe_srv", "/profile/a")
    assert registry.get_toolset_alias_target("_probe_srv", scope="/profile/a") == "mcp-_probe_srv", \
        "the global alias still resolves through the merged read"
    assert "/profile/a" not in registry._scoped_toolset_aliases
    assert registry._toolset_aliases["_probe_srv"] == "mcp-_probe_srv"


def test_toolset_membership_memo_does_not_serve_another_profile(monkeypatch):
    """The membership memo is keyed on the registry scope as well.

    Without it this memo was the hole underneath profile-scoped registration:
    the value it caches is registry-derived toolset MEMBERSHIP, read
    ambiently, so with the generation unchanged profile A's tool names were
    served straight to profile B.
    """
    import toolsets

    def _probe_handler(args):
        return "ok"

    scope_holder = {"key": "/profile/a"}
    monkeypatch.setattr(type(registry), "current_scope_key",
                        staticmethod(lambda: scope_holder["key"]))
    registry.register(
        name="_probe_scoped_tool", toolset="mcp-_probe_scoped", schema={
            "name": "_probe_scoped_tool", "description": "probe",
            "parameters": {"type": "object", "properties": {}}},
        handler=_probe_handler, scope="/profile/a")
    try:
        toolsets._resolve_toolset_memo.clear()
        assert toolsets.resolve_toolset("mcp-_probe_scoped") == ["_probe_scoped_tool"]
        # Same registry generation, other profile: the memo must not answer for A.
        scope_holder["key"] = "/profile/b"
        assert "_probe_scoped_tool" not in toolsets.resolve_toolset("mcp-_probe_scoped")
    finally:
        registry.deregister("_probe_scoped_tool", scope="/profile/a")
        toolsets._resolve_toolset_memo.clear()
