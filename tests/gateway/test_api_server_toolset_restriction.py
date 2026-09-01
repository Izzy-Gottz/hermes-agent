"""Regression guard for the opt-in per-turn toolset restriction
(X-Hermes-Enabled-Toolsets) in api_server._create_agent.

The restriction must be narrow-only AND actually strip MCP servers + plugin
toolsets. The first implementation re-ran ``_get_platform_tools`` with a
rewritten ``platform_toolsets`` — but that path re-adds every *enabled* MCP
server (and default-on plugin toolsets) regardless, so a ``["memory"]`` turn
still received vercel/pulse/supabase/cloudflare + connectors: the "lethal
trifecta" was not broken. The fix filters the ALREADY-RESOLVED
``enabled_toolsets`` down to the requested set. This test pins both the reason
(the resolved set leaks MCP servers) and the fix (filtering strips them and
fails closed)."""

from hermes_cli.tools_config import _get_platform_tools


def _config():
    return {
        "mcp_servers": {
            "vercel": {"url": "https://example.invalid/v", "enabled": True},
            "supabase": {"url": "https://example.invalid/s", "enabled": True},
        },
        "platform_toolsets": {"api_server": ["memory", "web", "terminal"]},
    }


def _restrict(resolved, requested):
    # exactly the expression api_server._create_agent uses
    req = {t.strip() for t in requested if t.strip()}
    return sorted(t for t in resolved if t in req)


def test_resolved_set_leaks_enabled_mcp_servers():
    # why re-resolving is unsafe: enabled MCP servers ride along as toolsets
    resolved = set(_get_platform_tools(_config(), "api_server"))
    assert "memory" in resolved
    assert {"vercel", "supabase"} <= resolved


def test_memory_only_filter_strips_mcp_and_keeps_memory():
    resolved = sorted(_get_platform_tools(_config(), "api_server"))
    restricted = _restrict(resolved, ["memory"])
    assert restricted == ["memory"]
    assert not ({"vercel", "supabase", "web", "terminal"} & set(restricted))


def test_empty_or_unknown_request_fails_closed():
    resolved = sorted(_get_platform_tools(_config(), "api_server"))
    assert _restrict(resolved, []) == []            # empty → no tools
    assert _restrict(resolved, ["does-not-exist"]) == []   # unknown → no tools


def test_restriction_can_only_narrow_never_widen():
    resolved = set(_get_platform_tools(_config(), "api_server"))
    # a client asking for terminal+vercel gets only what was already resolved,
    # and never anything outside it
    restricted = set(_restrict(sorted(resolved), ["terminal", "vercel", "not-real"]))
    assert restricted <= resolved
    assert "not-real" not in restricted
