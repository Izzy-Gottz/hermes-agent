"""Composio's expensive defaults, filled in at the outbound MCP seam.

A handful of Composio tools default to returning far more data than any
question needs — ``GMAIL_FETCH_EMAILS`` measured at 1,674,724 B for 40
messages with both flags omitted against 33,690 B with both set false, a
49.7x span, while neither parameter's description mentions bytes at all. A
result that size is spilled to a file and the model is handed a path, so the
cost is not the bytes: it is that the answer stops being in the conversation.

Every assertion below reads the arguments that reached ``session.call_tool``,
not what the handler printed. The four properties under test, in the order
they can go wrong:

1. It fills only what the caller OMITTED. A default that overrides is not a
   default, and a model that asked for the body must get the body.
2. It fires on Composio's server and NOT on a foreign server that happens to
   have a tool of the same name. Inheriting a broker's identity from a tool
   name alone is a bug this codebase has already shipped once.
3. It reaches BOTH invocation paths — the tool called directly and the same
   tool reached through ``COMPOSIO_MULTI_EXECUTE_TOOL``. A default landing on
   one path only would make the cost depend on which wrapper the model picked.
4. It SAYS SO in the result. A silently cheapened result is unrecoverable:
   the model cannot ask for what it does not know it lost.

And one guard that is easy to write so it cannot fail: the live schema is
consulted, so a renamed parameter stops the table rather than posting an
unknown key — asserted in both directions (declared → applied, withdrawn →
skipped, unknown tool → applied, because "I have no schema" is not "the
parameter is gone").
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool


COMPOSIO_URL = "https://backend.composio.dev/tool_router/sess_fake/mcp"


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class _FakeCallToolResult:
    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


class _FakeTool:
    """A discovered MCP tool, shaped like the SDK model the code reads."""

    def __init__(self, name: str, properties):
        self.name = name
        self.inputSchema = {"type": "object", "properties": dict(properties)}


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _install_lock_and_run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_install_lock_and_run())
    finally:
        loop.close()


# The two parameters the table fills, as Composio advertises them today.
GMAIL_SCHEMA = {
    "query": {"type": "string"},
    "verbose": {"type": "boolean"},
    "include_payload": {"type": "boolean"},
    "max_results": {"type": "integer"},
}


def _server(url=COMPOSIO_URL, tools=None):
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=_FakeCallToolResult(content=[_FakeContentBlock("ok")])
    )
    return SimpleNamespace(
        session=session,
        _rpc_lock=None,
        _config={"url": url},
        _tools=list(tools if tools is not None
                    else [_FakeTool("GMAIL_FETCH_EMAILS", GMAIL_SCHEMA)]),
    )


@pytest.fixture
def composio():
    """A connected Composio session with GMAIL_FETCH_EMAILS discovered."""
    server = _server()
    with patch.dict(mcp_tool._servers, {"composio": server}), \
         patch("tools.mcp_tool._run_on_mcp_loop",
               side_effect=_fake_run_on_mcp_loop), \
         patch.dict(mcp_tool._server_error_counts, {}, clear=True):
        yield server


def _sent(server):
    """The arguments that actually reached the wire."""
    server.session.call_tool.assert_awaited_once()
    return server.session.call_tool.await_args.kwargs["arguments"]


def _call(server, tool_name, args, name="composio"):
    handler = mcp_tool._make_tool_handler(name, tool_name, 30.0)
    return handler(args)


class TestDirectPath:
    """The tool called by its own name."""

    def test_omitted_parameters_are_filled(self, composio):
        _call(composio, "GMAIL_FETCH_EMAILS", {"query": "in:inbox"})
        sent = _sent(composio)
        assert sent["verbose"] is False
        assert sent["include_payload"] is False
        assert sent["query"] == "in:inbox", "the caller's own argument survived"

    def test_naming_one_lever_stands_the_whole_table_down(self, composio):
        """The escape hatch has to work in ONE move, or it is not an escape.

        The note tells the model "pass verbose=true for full text". If the
        other flag were still injected on that retry, the model would get the
        same advice back a second time over a result it had just tried to
        widen — advice it had already followed. So one named lever means the
        caller is steering this tool's verbosity, and the table stands aside.
        """
        raw = _call(composio, "GMAIL_FETCH_EMAILS",
                    {"query": "x", "verbose": True})
        assert _sent(composio) == {"query": "x", "verbose": True}
        assert "_hermes" not in json.loads(raw), \
            "nothing was filled, so there is nothing to say"

    def test_explicit_false_is_left_alone_too(self, composio):
        """Same value, but the caller chose it — nothing is 'filled in'."""
        raw = _call(composio, "GMAIL_FETCH_EMAILS",
                    {"verbose": False, "include_payload": False})
        assert _sent(composio) == {"verbose": False, "include_payload": False}
        assert "_hermes" not in json.loads(raw), \
            "nothing was filled, so there is nothing to disclose"

    def test_a_tool_not_in_the_table_is_untouched(self, composio):
        _call(composio, "GMAIL_SEND_EMAIL", {"to": "x@example.com"})
        assert _sent(composio) == {"to": "x@example.com"}

    def test_the_callers_dict_is_not_mutated(self, composio):
        original = {"query": "in:inbox"}
        _call(composio, "GMAIL_FETCH_EMAILS", original)
        assert original == {"query": "in:inbox"}


class TestForeignServer:
    """A tool name is not an identity. Only Composio's URL is."""

    def test_same_tool_name_on_another_server_is_untouched(self):
        """The bug this guards: one broker's rules applied to another's tool."""
        other = _server(url="https://mcp.example.com/v1")
        with patch.dict(mcp_tool._servers, {"other": other}), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   side_effect=_fake_run_on_mcp_loop), \
             patch.dict(mcp_tool._server_error_counts, {}, clear=True):
            _call(other, "GMAIL_FETCH_EMAILS", {"query": "x"}, name="other")
        assert _sent(other) == {"query": "x"}

    @pytest.mark.parametrize("url", [
        "https://composio.dev.attacker.example/mcp",
        "https://notcomposio.dev/mcp",
        "https://backend.composio.dev.evil.test/mcp",
        "https://mcp.example.com/?ref=backend.composio.dev",
    ])
    def test_a_lookalike_host_is_not_composio(self, url):
        """Substring-matching a URL matches all four of these. Parsing does not."""
        other = _server(url=url)
        with patch.dict(mcp_tool._servers, {"other": other}), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   side_effect=_fake_run_on_mcp_loop), \
             patch.dict(mcp_tool._server_error_counts, {}, clear=True):
            _call(other, "GMAIL_FETCH_EMAILS", {"query": "x"}, name="other")
        assert _sent(other) == {"query": "x"}

    @pytest.mark.parametrize("url", [
        "https://backend.composio.dev/tool_router/sess_fake/mcp",
        "https://composio.dev/mcp",
        "HTTPS://BACKEND.COMPOSIO.DEV/mcp",
    ])
    def test_the_real_host_and_its_subdomains_do_apply(self, url):
        """The negative above must not have been bought by matching nothing."""
        srv = _server(url=url)
        with patch.dict(mcp_tool._servers, {"c": srv}), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   side_effect=_fake_run_on_mcp_loop), \
             patch.dict(mcp_tool._server_error_counts, {}, clear=True):
            _call(srv, "GMAIL_FETCH_EMAILS", {"query": "x"}, name="c")
        assert _sent(srv)["verbose"] is False

    def test_a_server_with_no_url_at_all_is_untouched(self):
        """stdio servers have no url key — that must not raise, or apply."""
        stdio = _server(url=None)
        stdio._config = {"command": "npx", "args": ["some-server"]}
        with patch.dict(mcp_tool._servers, {"stdio": stdio}), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   side_effect=_fake_run_on_mcp_loop), \
             patch.dict(mcp_tool._server_error_counts, {}, clear=True):
            _call(stdio, "GMAIL_FETCH_EMAILS", {"query": "x"}, name="stdio")
        assert _sent(stdio) == {"query": "x"}


class TestMultiplexerPath:
    """The same tool reached through COMPOSIO_MULTI_EXECUTE_TOOL."""

    def test_inner_arguments_are_filled(self, composio):
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", {
            "tools": [
                {"tool_slug": "GMAIL_FETCH_EMAILS",
                 "arguments": {"query": "in:inbox"}},
            ],
        })
        entry = _sent(composio)["tools"][0]
        assert entry["tool_slug"] == "GMAIL_FETCH_EMAILS", "envelope preserved"
        assert entry["arguments"] == {
            "query": "in:inbox", "verbose": False, "include_payload": False,
        }

    def test_only_the_matching_entry_changes(self, composio):
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", {
            "tools": [
                {"tool_slug": "GOOGLECALENDAR_EVENTS_LIST",
                 "arguments": {"calendar_id": "primary"}},
                {"tool_slug": "GMAIL_FETCH_EMAILS", "arguments": {}},
            ],
        })
        tools = _sent(composio)["tools"]
        assert tools[0]["arguments"] == {"calendar_id": "primary"}
        assert tools[1]["arguments"] == {
            "verbose": False, "include_payload": False,
        }

    def test_an_explicit_inner_value_stands_the_table_down(self, composio):
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", {
            "tools": [{"tool_slug": "GMAIL_FETCH_EMAILS",
                       "arguments": {"verbose": True}}],
        })
        assert _sent(composio)["tools"][0]["arguments"] == {"verbose": True}

    def test_an_entry_with_no_arguments_at_all_is_filled(self, composio):
        """The cheapest call to write is the most expensive one to answer.

        `{"tool_slug": "GMAIL_FETCH_EMAILS"}` — no arguments key — is the bare
        call a model reaches for first, and it is the 1.6 MB one. Reading it
        as "not a dict, leave it alone" would skip the exact shape the table
        exists for.
        """
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", {
            "tools": [{"tool_slug": "GMAIL_FETCH_EMAILS"}],
        })
        assert _sent(composio)["tools"][0]["arguments"] == {
            "verbose": False, "include_payload": False,
        }

    def test_a_null_arguments_value_is_filled_too(self, composio):
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", {
            "tools": [{"tool_slug": "GMAIL_FETCH_EMAILS", "arguments": None}],
        })
        assert _sent(composio)["tools"][0]["arguments"] == {
            "verbose": False, "include_payload": False,
        }

    def test_arguments_of_the_wrong_type_are_passed_through(self, composio):
        """Never reinterpret a caller's malformed value — let the server say so."""
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", {
            "tools": [{"tool_slug": "GMAIL_FETCH_EMAILS",
                       "arguments": '{"query": "x"}'}],
        })
        assert _sent(composio)["tools"][0]["arguments"] == '{"query": "x"}'

    def test_a_malformed_envelope_is_passed_through_unharmed(self, composio):
        """Never turn a server's validation error into ours."""
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", {"tools": "not-a-list"})
        assert _sent(composio) == {"tools": "not-a-list"}

    def test_the_callers_nested_dicts_are_not_mutated(self, composio):
        inner = {"query": "in:inbox"}
        original = {"tools": [{"tool_slug": "GMAIL_FETCH_EMAILS",
                               "arguments": inner}]}
        _call(composio, "COMPOSIO_MULTI_EXECUTE_TOOL", original)
        assert inner == {"query": "in:inbox"}


class TestSchemaGuard:
    """The table applies only what the live schema still declares."""

    def test_a_withdrawn_parameter_is_not_posted(self):
        """Composio renames `verbose` → the table stops, it does not guess."""
        server = _server(tools=[_FakeTool("GMAIL_FETCH_EMAILS", {
            "query": {"type": "string"},
            "include_payload": {"type": "boolean"},
        })])
        with patch.dict(mcp_tool._servers, {"composio": server}), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   side_effect=_fake_run_on_mcp_loop), \
             patch.dict(mcp_tool._server_error_counts, {}, clear=True):
            _call(server, "GMAIL_FETCH_EMAILS", {"query": "x"})
        sent = _sent(server)
        assert "verbose" not in sent, "the schema no longer declares it"
        assert sent["include_payload"] is False, "the survivor still applies"

    def test_an_undiscovered_tool_still_gets_the_default(self):
        """The multiplexer reaches tools that were never preloaded.

        No schema means no opinion, not a veto — otherwise the guard would
        silently disable the table on the path it matters most.
        """
        server = _server(tools=[])
        with patch.dict(mcp_tool._servers, {"composio": server}), \
             patch("tools.mcp_tool._run_on_mcp_loop",
                   side_effect=_fake_run_on_mcp_loop), \
             patch.dict(mcp_tool._server_error_counts, {}, clear=True):
            _call(server, "COMPOSIO_MULTI_EXECUTE_TOOL", {
                "tools": [{"tool_slug": "GMAIL_FETCH_EMAILS",
                           "arguments": {}}],
            })
        assert _sent(server)["tools"][0]["arguments"] == {
            "verbose": False, "include_payload": False,
        }


class TestDisclosure:
    """What was filled in reaches the model, or it cannot recover."""

    def test_the_result_names_what_was_applied(self, composio):
        raw = _call(composio, "GMAIL_FETCH_EMAILS", {"query": "x"})
        note = json.loads(raw)["_hermes"]
        assert "GMAIL_FETCH_EMAILS" in note
        assert "verbose=false" in note
        assert "include_payload=false" in note
        assert "verbose=true" in note, "and how to get the bodies back"

    def test_the_payload_still_carries_the_result(self, composio):
        raw = _call(composio, "GMAIL_FETCH_EMAILS", {"query": "x"})
        assert json.loads(raw)["result"] == "ok"

    def test_no_note_when_nothing_was_filled(self, composio):
        raw = _call(composio, "GMAIL_SEND_EMAIL", {"to": "x@example.com"})
        assert json.loads(raw) == {"result": "ok"}

    def test_structured_content_survives_alongside_the_note(self, composio):
        composio.session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("ok")],
                structuredContent={"messages": []},
            )
        )
        raw = _call(composio, "GMAIL_FETCH_EMAILS", {"query": "x"})
        payload = json.loads(raw)
        assert payload["structuredContent"] == {"messages": []}
        assert "_hermes" in payload


class TestDisclosureSurvivesFailure:
    """The note matters MOST when the call fails.

    If Composio ever rejects a key this table injected, the model sees an
    error naming a parameter it did not pass. It cannot remove what it does
    not know it sent, so it retries the identical call — forever. Every
    return path has to carry the note, not just the happy one.
    """

    def test_a_tool_error_carries_the_note(self, composio):
        composio.session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock(
                    "Invalid request: unexpected parameter 'include_payload'"
                )],
                is_error=True,
            )
        )
        raw = _call(composio, "GMAIL_FETCH_EMAILS", {"query": "x"})
        err = json.loads(raw)["error"]
        assert "include_payload" in err, "the server's own words survive"
        assert "Hermes filled in" in err, \
            "and the model is told who put that parameter there"

    def test_a_transport_exception_carries_the_note(self, composio):
        composio.session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        raw = _call(composio, "GMAIL_FETCH_EMAILS", {"query": "x"})
        err = json.loads(raw)["error"]
        assert "boom" in err
        assert "Hermes filled in" in err

    def test_a_failure_with_no_defaults_applied_says_nothing(self, composio):
        """The negative control: the note is not simply always appended."""
        composio.session.call_tool = AsyncMock(side_effect=RuntimeError("boom"))
        raw = _call(composio, "GMAIL_SEND_EMAIL", {"to": "x@example.com"})
        assert "Hermes filled in" not in json.loads(raw)["error"]
