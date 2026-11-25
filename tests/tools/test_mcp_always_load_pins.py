"""A server's pinned tools stay out of ``tool_search`` on Hermes's own loop.

Every MCP tool is deferrable, and tool_search matches words literally, so a
tool the person connected on purpose could sit one search away and the model
would say the service "isn't connected". ``mcp_servers.<name>.always_load``
(and a tool's own ``_meta`` alwaysLoad mark, which Claude Code also reads)
pins tools out of the deferred half -- within a per-server token ceiling.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import tool_pins
from tools.mcp_tool import MCPServerTask
from tools.mcp_tool_registration import (
    _cached_tools, _deregister_mcp_tool_all_scopes, _register_from_cache_sync, _register_server_tools)
from tools.tool_search import (
    BRIDGE_TOOL_NAMES, ToolSearchConfig, assemble_tool_defs, is_deferrable_tool_name, resolve_underlying_call)


def _tool(name, desc="", meta=None, props=None):
    return SimpleNamespace(name=name, description=desc or f"{name} does a thing",
                           inputSchema={"type": "object", "properties": props or {}},
                           meta=meta)


def _defs(names):
    from tools.registry import registry
    out = []
    for n in names:
        entry = registry.get_entry(n) if hasattr(registry, "get_entry") else None
        schema = getattr(entry, "schema", None) or {"name": n, "description": n,
                                                     "parameters": {"type": "object", "properties": {}}}
        out.append({"type": "function", "function": schema})
    return out


@pytest.fixture
def server_factory():
    made = []

    def make(name, tools, config):
        server = MCPServerTask(name)
        server._tools = tools
        server.session = MagicMock()
        registered = _register_server_tools(name, server, config)
        made.append((server, registered))
        return registered

    yield make
    for server, registered in made:
        for n in registered:
            _deregister_mcp_tool_all_scopes(server, n)


def _assemble(names):
    return {td["function"]["name"] for td in assemble_tool_defs(
        _defs(names) + [{"type": "function", "function": {"name": "terminal", "description": "x",
                                                          "parameters": {"type": "object", "properties": {}}}}],
        context_length=200_000, config=ToolSearchConfig.from_raw({"enabled": "on"})).tool_defs}


def _by_suffix(registered, suffix):
    [hit] = [n for n in registered if n.endswith(suffix)]
    return hit


def test_config_list_pins_named_tools_and_defers_the_rest(server_factory):
    registered = server_factory("pinprobe_a", [_tool("GMAIL_FETCH_EMAILS"), _tool("GOOGLECALENDAR_FIND_EVENT"),
                                               _tool("SOMETHING_ELSE")],
                                {"always_load": ["GMAIL_FETCH_EMAILS", "GOOGLECALENDAR_*"]})
    gmail = _by_suffix(registered, "GMAIL_FETCH_EMAILS")
    cal = _by_suffix(registered, "GOOGLECALENDAR_FIND_EVENT")
    other = _by_suffix(registered, "SOMETHING_ELSE")

    seen = _assemble(registered)
    assert gmail in seen and cal in seen, "pinned tools must be in the model's list, not behind tool_search"
    assert other not in seen, "an unpinned MCP tool still defers"
    assert BRIDGE_TOOL_NAMES <= seen
    assert not is_deferrable_tool_name(gmail)
    assert is_deferrable_tool_name(other)


def test_without_config_nothing_is_pinned(server_factory):
    registered = server_factory("pinprobe_b", [_tool("GMAIL_FETCH_EMAILS")], {})
    assert not (set(registered) & _assemble(registered))


def test_servers_own_meta_mark_pins_and_false_overrides_it(server_factory):
    marked = {tool_pins.ALWAYS_LOAD_META_KEY: True}
    registered = server_factory("pinprobe_c", [_tool("MARKED", meta=marked), _tool("PLAIN")], {})
    assert _by_suffix(registered, "MARKED") in _assemble(registered)
    assert _by_suffix(registered, "PLAIN") not in _assemble(registered)

    refused = server_factory("pinprobe_d", [_tool("MARKED", meta=marked)], {"always_load": False})
    assert not (set(refused) & _assemble(refused))


def test_true_pins_every_tool(server_factory):
    registered = server_factory("pinprobe_e", [_tool("ONE"), _tool("TWO")], {"always_load": True})
    listed = {_by_suffix(registered, "ONE"), _by_suffix(registered, "TWO")}
    seen = _assemble(registered)
    assert listed <= seen
    # Generated resource/prompt utilities are not tools the server listed; they stay deferred.
    assert not ((set(registered) - listed) & seen)


def test_token_ceiling_bounds_what_one_server_can_pin(server_factory):
    big = {f"p{i}": {"type": "string", "description": "x" * 200} for i in range(10)}  # ~600 tokens each
    tools = [_tool(f"BIG_{i}", props=big) for i in range(6)]
    registered = server_factory("pinprobe_f", tools, {"always_load": True, "always_load_max_tokens": 1500})
    pinned = set(registered) & tool_pins.pinned()
    assert 1 <= len(pinned) <= 2, pinned
    assert set(registered) - pinned, "tools past the ceiling stay deferred"
    seen = _assemble(registered)
    assert pinned <= seen and not ((set(registered) - pinned) & seen)


def test_re_registration_unpins_and_deregistration_forgets(server_factory):
    registered = server_factory("pinprobe_g", [_tool("FLIP")], {"always_load": ["FLIP"]})
    name = registered[0]
    assert tool_pins.is_pinned(name)
    server_factory("pinprobe_g", [_tool("FLIP")], {})
    assert not tool_pins.is_pinned(name), "a config that drops the pin must unpin on refresh"


def test_bridge_and_list_agree_a_pinned_tool_is_direct(server_factory):
    registered = server_factory("pinprobe_h", [_tool("DIRECT")], {"always_load": ["DIRECT"]})
    name, _args, err = resolve_underlying_call({"calls": [{"name": registered[0], "arguments": {}}]},
                                               session_tool_names=set(registered))
    # tool_call refuses a tool that is already in the model's list and says to call it directly:
    # the predicate is one source, so the catalog, the bridge and the list agree.
    assert name is None and err


def test_schema_cache_round_trip_keeps_the_mark():
    [stand_in] = _cached_tools([{"name": "X", "inputSchema": {}, "_meta": {tool_pins.ALWAYS_LOAD_META_KEY: True}}])
    from tools.mcp_tool_registration import _tool_marks_always_load
    assert _tool_marks_always_load(stand_in)
    [plain] = _cached_tools([{"name": "Y", "inputSchema": {}}])
    assert not _tool_marks_always_load(plain)


def test_lazy_cache_start_honours_the_config_list():
    entry = {"tools": [{"name": "GMAIL_FETCH_EMAILS", "description": "d", "inputSchema": {}},
                       {"name": "OTHER", "description": "d", "inputSchema": {}}], "utility_tools": []}
    registered = _register_from_cache_sync("pinprobe_lazy", {"always_load": ["GMAIL_FETCH_EMAILS"]}, entry)
    try:
        pinned = set(registered) & tool_pins.pinned()
        assert [n for n in pinned] == [_by_suffix(registered, "GMAIL_FETCH_EMAILS")]
    finally:
        from tools.mcp_tool_common import _core
        from tools.mcp_tool_scope import _server_key
        for n in registered:
            _deregister_mcp_tool_all_scopes("pinprobe_lazy", n)
        key = _server_key("pinprobe_lazy")
        with _core._lock:
            for table in (_core._lazy_server_configs, _core._lazy_server_fingerprints,
                          _core._lazy_server_tool_names):
                table.pop(key, None)
