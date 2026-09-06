"""The api_server must hit the tool-definition cache across requests.

Measured on 2026-09-06 (agent.log, claude-code runtime, ~2000 lazy MCP tools
registered from the schema cache): every ``/v1/chat/completions`` request
rebuilt the full tool surface twice — once in ``AIAgent.__init__`` and once
in the between-turns MCP refresh — ~120 ms each, 1,008 ``tool_search
activated`` lines for 613 turns. ``model_tools.get_tool_definitions`` already
memoises on (scope, toolsets, registry generation, config mtime, ...), but
``registry.check_fn_cache_scope`` returned ``CHECK_FN_CACHE_BYPASS`` for
every API request because ``_bind_api_server_session`` binds a session id, a
browser-control principal and a transport family unconditionally, and the
bypass treated that identity as "a browser controller may be attached". With
``browser.extension_control.enabled`` off nothing request-bound can change,
so the bypass now requires the feature flag.

These tests pin, against a counting stand-in for the expensive assembly:

- (a) two api_server-shaped requests with the same key assemble ONCE, and
  the bypass still holds when browser control is actually enabled;
- (b) every invalidation input — registry (de)registration / alias
  (what an MCP reload, plugin load or skill-driven tool registration does),
  the config file changing, the explicit ``_clear_tool_defs_cache`` hook, a
  different toolset selection — forces one rebuild;
- (c) different keys never share an entry (toolsets, profile scope).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

import model_tools
from tools.registry import CHECK_FN_CACHE_BYPASS, check_fn_cache_scope, registry


_FAKE_DEFS = [
    {"type": "function", "function": {"name": "fake_tool", "parameters": {"type": "object", "properties": {}}}},
]


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Empty cache, a private config file for the mtime fingerprint, a
    counting assembly stub, and browser control OFF (the shipped default)."""
    model_tools._tool_defs_cache.clear()

    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  default: fake\n")
    import hermes_cli.config as _cfg_mod
    monkeypatch.setattr(_cfg_mod, "get_config_path", lambda: cfg)

    calls = []

    def _fake_compute(enabled, disabled, quiet, skip_tool_search_assembly=False):
        calls.append((tuple(enabled or ()), tuple(disabled or ()), skip_tool_search_assembly))
        return [dict(d) for d in _FAKE_DEFS]

    monkeypatch.setattr(model_tools, "_compute_tool_definitions", _fake_compute)
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: False
    )
    yield calls, cfg
    model_tools._tool_defs_cache.clear()


@pytest.fixture
def api_server_identity():
    """What ``ApiServerAdapter._bind_api_server_session`` binds per request."""
    from gateway.session_context import clear_session_vars, set_session_vars

    def _bind(session_id: str = "sess-1"):
        return set_session_vars(
            platform="api_server",
            session_id=session_id,
            browser_control_principal="principal:default:0123456789abcdef",
            browser_control_transport_family="local-api",
            async_delivery=False,
            cron_session="",
        )

    bound = []
    yield lambda sid="sess-1": bound.append(_bind(sid)) or bound[-1]
    for tokens in reversed(bound):
        clear_session_vars(tokens)


def _call(toolsets=("web", "terminal")):
    return model_tools.get_tool_definitions(enabled_toolsets=list(toolsets), quiet_mode=True)


# ── (a) same key assembles once ─────────────────────────────────────────────


def test_two_api_server_requests_assemble_once(_isolated_cache, api_server_identity):
    calls, _ = _isolated_cache
    api_server_identity("sess-1")
    assert check_fn_cache_scope() != CHECK_FN_CACHE_BYPASS

    first = _call()
    second = _call()

    assert len(calls) == 1, f"assembly ran {len(calls)}x for one key: {calls}"
    assert first == second == _FAKE_DEFS
    assert len(model_tools._tool_defs_cache) == 1


def test_agent_init_and_between_turns_refresh_share_one_assembly(_isolated_cache, api_server_identity):
    """The two per-turn callers (agent_init, refresh_agent_mcp_tools) pass the
    same toolsets with quiet_mode=True; the second must be a hit."""
    calls, _ = _isolated_cache
    api_server_identity("sess-1")
    _call(("web", "terminal", "browser"))
    # refresh_agent_mcp_tools reads agent.enabled_toolsets — same list object
    # semantics, possibly a different order.
    _call(("browser", "terminal", "web"))
    assert len(calls) == 1


def test_different_sessions_share_when_browser_control_off(_isolated_cache, api_server_identity):
    """Session identity is not part of the key: with the feature off it cannot
    influence which tools exist, so request N+1 hits request N's entry."""
    calls, _ = _isolated_cache
    api_server_identity("sess-1")
    _call()
    api_server_identity("sess-2")
    _call()
    assert len(calls) == 1


def test_bypass_still_holds_when_browser_control_enabled(_isolated_cache, api_server_identity, monkeypatch):
    """The safety property the bypass exists for is untouched: a real
    browser-control deployment keeps per-request evaluation and caches nothing."""
    calls, _ = _isolated_cache
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )
    api_server_identity("sess-1")
    assert check_fn_cache_scope() == CHECK_FN_CACHE_BYPASS

    _call()
    _call()

    assert len(calls) == 2
    assert model_tools._tool_defs_cache == {}


def test_miss_and_bypass_are_logged_once_per_assembly(_isolated_cache, api_server_identity, monkeypatch, caplog):
    calls, _ = _isolated_cache
    api_server_identity("sess-1")
    with caplog.at_level(logging.INFO, logger="model_tools"):
        _call()
        _call()
    lines = [r.getMessage() for r in caplog.records if "tool definitions assembled" in r.getMessage()]
    assert len(lines) == 1 and lines[0].endswith("cache miss)"), lines

    caplog.clear()
    monkeypatch.setattr(
        "gateway.browser_control_broker.browser_control_enabled", lambda: True
    )
    with caplog.at_level(logging.INFO, logger="model_tools"):
        _call()
    lines = [r.getMessage() for r in caplog.records if "tool definitions assembled" in r.getMessage()]
    assert len(lines) == 1 and lines[0].endswith("cache bypassed)"), lines


# ── (b) every invalidation input forces exactly one rebuild ─────────────────


def _register_fake(name="cache_probe_tool", toolset="cache-probe"):
    registry.register(
        name=name,
        toolset=toolset,
        schema={"name": name, "description": "", "parameters": {"type": "object", "properties": {}}},
        handler=lambda *a, **k: None,
    )


@pytest.fixture
def probe_tool():
    _register_fake()
    try:
        yield "cache_probe_tool"
    finally:
        try:
            registry.deregister("cache_probe_tool")
        except Exception:
            pass


def test_registry_register_invalidates(_isolated_cache, api_server_identity):
    """An MCP reload / plugin load lands in registry.register(); the
    generation bump must miss the old entry."""
    calls, _ = _isolated_cache
    api_server_identity()
    _call()
    gen_before = registry._generation
    _register_fake()
    try:
        assert registry._generation > gen_before
        _call()
        assert len(calls) == 2
        _call()
        assert len(calls) == 2, "the rebuilt entry must itself be reused"
    finally:
        registry.deregister("cache_probe_tool")


def test_registry_deregister_invalidates(_isolated_cache, api_server_identity, probe_tool):
    calls, _ = _isolated_cache
    api_server_identity()
    _call()
    registry.deregister(probe_tool)
    _call()
    assert len(calls) == 2


def test_toolset_alias_invalidates(_isolated_cache, api_server_identity, probe_tool):
    calls, _ = _isolated_cache
    api_server_identity()
    _call()
    registry.register_toolset_alias("cache-probe-alias", "cache-probe")
    _call()
    assert len(calls) == 2


def test_config_file_change_invalidates(_isolated_cache, api_server_identity):
    """A config reload is a rewritten config.yaml; the key carries its
    (mtime_ns, size) fingerprint."""
    calls, cfg = _isolated_cache
    api_server_identity()
    _call()
    cfg.write_text("model:\n  default: fake\nagent:\n  x: 1\n")
    st = cfg.stat()
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _call()
    assert len(calls) == 2


def test_explicit_clear_hook_invalidates(_isolated_cache, api_server_identity):
    calls, _ = _isolated_cache
    api_server_identity()
    _call()
    model_tools._clear_tool_defs_cache()
    _call()
    assert len(calls) == 2


def test_toolset_narrowing_invalidates(_isolated_cache, api_server_identity):
    """restrict_toolsets narrows enabled_toolsets per request; a narrowed
    request must never be served the wide entry."""
    calls, _ = _isolated_cache
    api_server_identity()
    _call(("web", "terminal", "browser"))
    _call(("web",))
    assert len(calls) == 2
    assert calls[1][0] == ("web",)


# ── (c) different keys never share ──────────────────────────────────────────


def test_distinct_toolsets_get_distinct_entries(_isolated_cache, api_server_identity):
    calls, _ = _isolated_cache
    api_server_identity()
    _call(("web",))
    _call(("terminal",))
    _call(("web",))
    _call(("terminal",))
    assert len(calls) == 2
    assert len(model_tools._tool_defs_cache) == 2


def test_distinct_profile_scopes_get_distinct_entries(_isolated_cache, api_server_identity, monkeypatch):
    """Multiplex: two profiles with the same toolsets must not alias — the
    profile scope (and the registry scope key) are both in the key."""
    calls, _ = _isolated_cache
    api_server_identity()
    scope = {"v": "/profiles/a"}
    monkeypatch.setattr(model_tools, "check_fn_cache_scope", lambda: scope["v"])
    _call()
    scope["v"] = "/profiles/b"
    _call()
    assert len(calls) == 2
    assert len(model_tools._tool_defs_cache) == 2
    scope["v"] = "/profiles/a"
    _call()
    assert len(calls) == 2, "returning to profile A must hit A's entry, not rebuild"


def test_skip_tool_search_assembly_is_its_own_key(_isolated_cache, api_server_identity):
    """The tool_search bridge asks for the un-collapsed catalog; it must not be
    handed the collapsed one (or vice versa)."""
    calls, _ = _isolated_cache
    api_server_identity()
    _call()
    model_tools.get_tool_definitions(
        enabled_toolsets=["web", "terminal"], quiet_mode=True, skip_tool_search_assembly=True
    )
    assert len(calls) == 2
    assert calls[1][2] is True
