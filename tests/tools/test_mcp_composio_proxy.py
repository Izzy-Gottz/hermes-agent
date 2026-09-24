"""Composio tools by name on a multi-account session (tools/mcp_composio_proxy.py).

Moe ticket #12, 2026-09-24: two Gmail accounts connected, a mail to send from
one of them, and no route. The session preloaded named tools, which Composio
will not do on a session that holds several accounts of one service; and the
model, which could see GMAIL_FETCH_EMAILS but not GMAIL_SEND_EMAIL, called
the latter three ways and was sent in a circle.

Every assertion reads what reached ``session.call_tool``. The fixture below
is the live answer to COMPOSIO_GET_TOOL_SCHEMAS measured that day, trimmed
to the fields the code reads: note ``successful: false`` and ``success:
false`` -- one unknown slug in the request marks the whole answer failed
while every known schema is still in it.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool
from tools import mcp_tool_handlers
from tools import mcp_composio_proxy as proxy
from tools import tool_search

COMPOSIO_URL = "https://backend.composio.dev/tool_router/trs_fake/mcp"

LIVE_SCHEMAS = json.dumps({
    "data": {
        "success": False,
        "tool_schemas": {
            "GMAIL_SEND_EMAIL": {
                "toolkit": "GMAIL", "tool_slug": "GMAIL_SEND_EMAIL",
                "description": "Sends an email via Gmail API using the authenticated user's Google profile display name.",
                "input_schema": {"type": "object", "properties": {
                    "attachment": {"type": "object"}, "bcc": {"type": "array"}, "body": {"type": "string"},
                    "cc": {"type": "array"}, "extra_recipients": {"type": "array"},
                    "from_email": {"type": "string"}, "is_html": {"type": "boolean"},
                    "recipient_email": {"type": "string"}, "subject": {"type": "string"},
                    "user_id": {"type": "string", "default": "me"}}},
            },
            "GMAIL_FETCH_EMAILS": {
                "toolkit": "GMAIL", "tool_slug": "GMAIL_FETCH_EMAILS",
                "description": "Fetches a list of email messages from a Gmail account.",
                "input_schema": {"type": "object", "properties": {
                    "query": {"type": "string"}, "user_id": {"type": "string", "default": "me"},
                    "verbose": {"type": "boolean"}, "include_payload": {"type": "boolean"},
                    "max_results": {"type": "integer"}}},
            },
        },
        "not_found": ["NOT_A_REAL_TOOL_X"],
        "not_found_message": "The following tool slugs were not found: NOT_A_REAL_TOOL_X.",
    },
    "error": None, "successful": False,
})

META = ["COMPOSIO_GET_TOOL_SCHEMAS", "COMPOSIO_MANAGE_CONNECTIONS",
        "COMPOSIO_MULTI_EXECUTE_TOOL", "COMPOSIO_SEARCH_TOOLS"]


class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _Result:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.isError = False
        self.structuredContent = None


def _tool(name, props=None):
    return SimpleNamespace(name=name, description="", inputSchema={"type": "object", "properties": props or {}})


def _server(url=COMPOSIO_URL, slugs=("GMAIL_SEND_EMAIL", "GMAIL_FETCH_EMAILS", "NOT_A_REAL_TOOL_X"),
            listed=None):
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=_Result(LIVE_SCHEMAS))
    cfg = {"url": url}
    if slugs is not None:
        cfg["composio_tools"] = list(slugs)
        cfg["composio_read_only"] = ["GMAIL_FETCH_EMAILS"]
    return SimpleNamespace(session=session, _rpc_lock=None, _config=cfg,
                           _tools=list(listed if listed is not None else [_tool(n) for n in META]))


def _augment(server):
    server._tools = asyncio.new_event_loop().run_until_complete(proxy.augment(server, server._tools))
    server.session.call_tool.reset_mock()
    server.session.call_tool.return_value = _Result('{"data":{"results":[]},"successful":true}')
    return server


def _run(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def go():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(go())
    finally:
        loop.close()


def _call(server, tool_name, args, name="composio"):
    with patch.dict(mcp_tool._servers, {name: server}), \
         patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=_run), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True):
        return mcp_tool_handlers._make_tool_handler(name, tool_name, 30.0)(args)


def _sent(server):
    server.session.call_tool.assert_awaited_once()
    a = server.session.call_tool.await_args
    return a.args[0], a.kwargs["arguments"]


class TestDiscovery:
    def test_configured_slugs_join_the_list_with_an_account_parameter(self):
        s = _augment(_server())
        names = [t.name for t in s._tools]
        assert names[:4] == META, "the server's own tools come first, untouched"
        assert "GMAIL_SEND_EMAIL" in names and "GMAIL_FETCH_EMAILS" in names
        send = next(t for t in s._tools if t.name == "GMAIL_SEND_EMAIL")
        props = send.inputSchema["properties"]
        assert "account" in props and "recipient_email" in props
        assert "ONLY parameter that chooses the account" in props["account"]["description"]

    def test_reads_are_labelled_reads_and_everything_else_is_not(self):
        """A preloaded tool carried Composio's readOnlyHint; Moe's send gate
        reads it. Without it every Gmail read would ask as if it could send."""
        from tools.mcp_tool_registration import _annotation_read_only_hint
        s = _augment(_server())
        by = {t.name: t for t in s._tools}
        assert _annotation_read_only_hint(by["GMAIL_FETCH_EMAILS"]) is True
        assert _annotation_read_only_hint(by["GMAIL_SEND_EMAIL"]) is False

    def test_a_failed_answer_still_yields_every_schema_it_carries(self):
        """``successful: false`` because one slug was unknown -- not because
        the known ones are missing. Reading the flag would lose them all."""
        s = _augment(_server())
        assert "NOT_A_REAL_TOOL_X" not in [t.name for t in s._tools], "no schema, no tool"
        assert sum(t.name.startswith("GMAIL_") for t in s._tools) == 2

    def test_the_schemas_are_asked_for_once_with_the_missing_slugs(self):
        s = _server()
        asyncio.new_event_loop().run_until_complete(proxy.augment(s, s._tools))
        s.session.call_tool.assert_awaited_once_with(
            "COMPOSIO_GET_TOOL_SCHEMAS",
            arguments={"tool_slugs": ["GMAIL_SEND_EMAIL", "GMAIL_FETCH_EMAILS", "NOT_A_REAL_TOOL_X"]})

    def test_a_slug_the_server_lists_itself_is_left_to_the_server(self):
        listed = [_tool(n) for n in META] + [_tool("GMAIL_FETCH_EMAILS", {"query": {}})]
        s = _augment(_server(listed=listed))
        fetch = [t for t in s._tools if t.name == "GMAIL_FETCH_EMAILS"]
        assert len(fetch) == 1 and "account" not in fetch[0].inputSchema["properties"]

    @pytest.mark.parametrize("case", ["no-config", "foreign-host", "no-multiplexer"])
    def test_nothing_is_added_where_it_should_not_be(self, case):
        if case == "no-config":
            s = _server(slugs=None)
        elif case == "foreign-host":
            s = _server(url="https://composio.dev.attacker.example/mcp")
        else:
            s = _server(listed=[_tool("COMPOSIO_GET_TOOL_SCHEMAS")])
        before = list(s._tools)
        after = asyncio.new_event_loop().run_until_complete(proxy.augment(s, s._tools))
        assert after == before
        s.session.call_tool.assert_not_awaited()

    def test_a_network_failure_leaves_the_servers_own_list(self):
        s = _server()
        s.session.call_tool = AsyncMock(side_effect=OSError("offline"))
        after = asyncio.new_event_loop().run_until_complete(proxy.augment(s, s._tools))
        assert [t.name for t in after] == META


class TestCalls:
    def test_a_send_from_a_named_account_goes_through_the_multiplexer(self):
        """The ticket's call, as it should have gone."""
        s = _augment(_server())
        _call(s, "GMAIL_SEND_EMAIL", {"recipient_email": "a@example.com", "subject": "hi",
                                      "body": "x", "account": "srulynj@gmail.com"})
        name, args = _sent(s)
        assert name == "COMPOSIO_MULTI_EXECUTE_TOOL"
        assert args["tools"] == [{"tool_slug": "GMAIL_SEND_EMAIL", "account": "srulynj@gmail.com",
                                  "arguments": {"recipient_email": "a@example.com", "subject": "hi", "body": "x"}}]

    def test_no_account_means_the_default_and_no_account_key(self):
        s = _augment(_server())
        _call(s, "GMAIL_SEND_EMAIL", {"recipient_email": "a@example.com"})
        _, args = _sent(s)
        assert "account" not in args["tools"][0]

    def test_the_cheap_defaults_still_reach_a_muxed_fetch(self):
        s = _augment(_server())
        _call(s, "GMAIL_FETCH_EMAILS", {"query": "in:inbox", "account": "sruly@hqpulse.ai"})
        _, args = _sent(s)
        entry = args["tools"][0]
        assert entry["account"] == "sruly@hqpulse.ai"
        assert entry["arguments"]["verbose"] is False and entry["arguments"]["include_payload"] is False

    def test_a_server_that_lists_the_tool_itself_is_called_directly(self):
        listed = [_tool(n) for n in META] + [_tool("GMAIL_SEND_EMAIL")]
        s = _augment(_server(listed=listed))
        _call(s, "GMAIL_SEND_EMAIL", {"recipient_email": "a@example.com"})
        name, args = _sent(s)
        assert name == "GMAIL_SEND_EMAIL" and args == {"recipient_email": "a@example.com"}

    def test_an_unconfigured_tool_is_called_by_its_own_name(self):
        s = _augment(_server())
        _call(s, "COMPOSIO_SEARCH_TOOLS", {"queries": []})
        name, _ = _sent(s)
        assert name == "COMPOSIO_SEARCH_TOOLS"

    def test_registered_from_cache_before_discovery_still_routes(self):
        """A lazily registered server has no native list yet; the multiplexer
        reaches every slug, so that is where the call goes."""
        s = _server()
        s.session.call_tool.return_value = _Result("{}")
        _call(s, "GMAIL_SEND_EMAIL", {"account": "srulynj@gmail.com"})
        name, args = _sent(s)
        assert name == "COMPOSIO_MULTI_EXECUTE_TOOL" and args["tools"][0]["account"] == "srulynj@gmail.com"

    def test_the_callers_dict_is_not_mutated(self):
        s = _augment(_server())
        original = {"recipient_email": "a@example.com", "account": "srulynj@gmail.com"}
        _call(s, "GMAIL_SEND_EMAIL", original)
        assert original == {"recipient_email": "a@example.com", "account": "srulynj@gmail.com"}


class TestRouteHint:
    def test_an_unlisted_composio_action_names_the_multiplexer(self):
        with patch("tools.registry.registry.get_all_tool_names",
                   return_value=["mcp__composio__COMPOSIO_MULTI_EXECUTE_TOOL"]):
            hint = tool_search._composio_route_hint("mcp__composio__GMAIL_SEND_EMAIL")
        assert "mcp__composio__COMPOSIO_MULTI_EXECUTE_TOOL" in hint
        assert '"tool_slug": "GMAIL_SEND_EMAIL"' in hint and "account" in hint
        assert "call it directly" not in hint, "the advice that sent the model in a circle"

    def test_no_hint_without_a_multiplexer_or_for_other_names(self):
        with patch("tools.registry.registry.get_all_tool_names", return_value=[]):
            assert tool_search._composio_route_hint("mcp__composio__GMAIL_SEND_EMAIL") == ""
        with patch("tools.registry.registry.get_all_tool_names",
                   return_value=["mcp__composio__COMPOSIO_MULTI_EXECUTE_TOOL"]):
            assert tool_search._composio_route_hint("mcp__github__create_issue") == ""
