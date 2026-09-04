"""The schema-cache writer read a field the pinned SDK does not have.

``_register_server_tools`` built its cache payload with
``getattr(mcp_tool, "inputSchema", None)``. mcp 2.0 renamed every model field
to snake_case and kept the camelCase spelling only as a *serialization*
alias — and pydantic aliases do not apply to attribute access — so that call
returned the default instead of raising. Measured on the live cache before
the fix: **all 1,994 cached tools were written with ``inputSchema: {}``**, so
a server registered lazily from cache offered the model tools with no
parameters at all, and nothing anywhere reported an error.

This is the same trap already documented in ``mcp_tool.py`` for
``readOnlyHint``, where it recorded every MCP tool on one machine as
write-capable. A rename that reads as a default is silent by construction.

**These tests use a real ``mcp.types.Tool``, deliberately.** A hand-rolled
double with an ``inputSchema`` attribute passes against the broken code — it
is the exact shape the bug cannot see — so a fake here would be a test that
cannot fail. The first test below asserts the SDK's shape directly, so if a
future mcp release brings the camelCase attribute back, it says so instead of
quietly making the rest meaningless.
"""

import json

import pytest

from tools import mcp_tool

mcp_types = pytest.importorskip("mcp.types")


SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}, "verbose": {"type": "boolean"}},
    "required": ["query"],
}


def _real_tool():
    return mcp_types.Tool(name="GMAIL_FETCH_EMAILS", description="d",
                          inputSchema=SCHEMA)


def test_the_pinned_sdk_really_does_hide_the_camelcase_attribute():
    """The premise, asserted rather than assumed.

    If this ever fails, the bug below is no longer reachable and this file's
    other assertions stop meaning what they say.
    """
    tool = _real_tool()
    assert getattr(tool, "inputSchema", None) is None, \
        "the camelCase attribute is back — re-read why this file exists"
    assert tool.input_schema == SCHEMA


def test_mcp_field_reads_it_across_the_rename():
    assert mcp_tool.mcp_field(_real_tool(), "input_schema", "inputSchema") == SCHEMA


def test_the_written_cache_entry_keeps_the_parameters(monkeypatch, tmp_path):
    """End to end: what actually lands on disk for a real SDK tool."""
    from tools import mcp_schema_cache as msc

    monkeypatch.setattr(msc, "_cache_path", lambda: tmp_path / "cache.json")

    written = {}

    def _capture(server, fingerprint, tools, utility_tools=None, **kw):
        written["tools"] = tools

    monkeypatch.setattr(msc, "write_cache_entry", _capture)

    server = _FakeServer([_real_tool()])
    mcp_tool._register_server_tools("srv", server, {"url": "https://x/mcp"})

    assert "tools" in written, "the write-through never ran"
    entry = next(t for t in written["tools"] if t["name"] == "GMAIL_FETCH_EMAILS")
    assert entry["inputSchema"] == SCHEMA, \
        "a lazily-registered server would offer this tool with no parameters"
    # The on-disk key stays camelCase — that is the file format the reader
    # expects, and changing it would strand every existing cache file.
    assert "input_schema" not in entry
    assert json.dumps(entry)  # serializable, as the cache requires


class _FakeServer:
    """Only what `_register_server_tools` touches."""

    def __init__(self, tools):
        self._tools = list(tools)
        self.name = "srv"
        self.session = object()
        self.tool_timeout = 30.0
        self._config = {"url": "https://x/mcp"}
        self.initialize_result = None
