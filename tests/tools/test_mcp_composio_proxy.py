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


# ── which account: an address resolves to the id Composio can match ─────────
#
# 2026-10-07, measured live: three Gmail accounts on one multi-account session,
# one with no alias. ``account: "yisrael@claimoe.ai"`` came back ``No account
# found matching``; ``account: "ca_1r8-AuqvZ_oB"`` (that mailbox) answered its
# profile. Composio matches an id or an alias, and nothing gave that one an
# alias. The fixtures are the live shapes, trimmed to the fields read.

USER = "moe-0123456789abcdef0123"
ACCOUNTS = [  # newest first is NOT the order the API returns; sorted by created_at
    {"id": "ca_S1ES5lxCn2qH", "alias": "sruly@hqpulse.ai", "user_id": USER, "status": "ACTIVE",
     "toolkit": {"slug": "gmail"}, "created_at": "2026-09-20T10:00:00.000Z"},
    {"id": "ca_1r8-AuqvZ_oB", "alias": None, "user_id": USER, "status": "ACTIVE",
     "toolkit": {"slug": "gmail"}, "created_at": "2026-10-07T10:35:58.451Z"},
    {"id": "ca_7NIHiXGSiOUe", "alias": "srulynj@gmail.com", "user_id": USER, "status": "ACTIVE",
     "toolkit": {"slug": "gmail"}, "created_at": "2026-09-01T10:00:00.000Z"},
    # Someone else's, on a project-wide list: never a choice for this person.
    {"id": "ca_OTHERUSER0001", "alias": "stranger@example.com", "user_id": "moe-someoneelse", "status": "ACTIVE",
     "toolkit": {"slug": "gmail"}, "created_at": "2026-10-01T10:00:00.000Z"},
]
PROFILE = {"ca_1r8-AuqvZ_oB": "yisrael@claimoe.ai", "ca_S1ES5lxCn2qH": "sruly@hqpulse.ai",
           "ca_7NIHiXGSiOUe": "srulynj@gmail.com"}


class _FakeComposio:
    """Composio's REST API as this module uses it: the session, the accounts
    list, and the alias PATCH. Records every request."""

    def __init__(self, accounts=ACCOUNTS, readable=True):
        self.accounts = [dict(a) for a in accounts]
        self.readable = readable
        self.requests = []

    async def __call__(self, server, method, path, body=None):
        self.requests.append((method, path, body))
        if not self.readable:
            return None
        if method == "GET" and path.startswith("/api/v3.1/tool_router/session/"):
            return {"session_id": "trs_fake", "config": {"user_id": USER}}
        if method == "GET" and path.startswith("/api/v3/connected_accounts?"):
            assert "user_ids=" + USER in path and "statuses=ACTIVE" in path
            return {"items": [dict(a) for a in self.accounts], "next_cursor": None}
        if method == "PATCH":
            return {"ok": True}
        return None

    def lists(self):
        return sum(1 for m, p, _ in self.requests if p.startswith("/api/v3/connected_accounts?"))

    def patches(self):
        return [(p, b) for m, p, b in self.requests if m == "PATCH"]


def _profile_answer(account):
    return _Result(json.dumps({"successful": True, "data": {"results": [{"response": {
        "successful": True, "data": {"emailAddress": PROFILE[account]}},
        "tool_slug": "GMAIL_GET_PROFILE", "index": 0}]}}))


def _composio_session(server):
    """call_tool that answers GET_TOOL_SCHEMAS and a muxed GMAIL_GET_PROFILE."""
    async def call_tool(name, arguments=None):
        if name == "COMPOSIO_GET_TOOL_SCHEMAS":
            return _Result(LIVE_SCHEMAS)
        entry = (arguments or {}).get("tools", [{}])[0]
        if name == "COMPOSIO_MULTI_EXECUTE_TOOL" and entry.get("tool_slug") == "GMAIL_GET_PROFILE":
            return _profile_answer(entry["account"])
        return _Result('{"data":{"results":[]},"successful":true}')
    server.session.call_tool = AsyncMock(side_effect=call_tool)
    return server


@pytest.fixture(autouse=True)
def _no_real_composio(tmp_path, monkeypatch):
    """No test reaches the network or the real HERMES_HOME. By default the
    accounts cannot be read, which is the behaviour the tests above pin."""
    monkeypatch.setattr(proxy, "_rest", _FakeComposio(readable=False))
    monkeypatch.setattr(proxy, "_names_path", lambda: tmp_path / "cache" / "composio-account-names.json")


def _augment_with_accounts(monkeypatch, fake=None):
    fake = fake or _FakeComposio()
    monkeypatch.setattr(proxy, "_rest", fake)
    s = _composio_session(_server())
    s._tools = asyncio.new_event_loop().run_until_complete(proxy.augment(s, s._tools))
    return s, fake


def _sent_after(server, n_before):
    calls = server.session.call_tool.await_args_list[n_before:]
    assert len(calls) == 1, calls
    return calls[0].args[0], calls[0].kwargs["arguments"]


class TestAccountChoice:
    def test_the_unnamed_account_is_learned_from_its_profile_and_listed(self, monkeypatch, tmp_path):
        s, fake = _augment_with_accounts(monkeypatch)
        asked = [c.kwargs["arguments"]["tools"][0]["account"] for c in s.session.call_tool.await_args_list
                 if c.args[0] == "COMPOSIO_MULTI_EXECUTE_TOOL"]
        assert asked == ["ca_1r8-AuqvZ_oB"], "only the account with no alias is asked who it is"
        send = next(t for t in s._tools if t.name == "GMAIL_SEND_EMAIL")
        desc = send.inputSchema["properties"]["account"]["description"]
        for who in ("yisrael@claimoe.ai", "sruly@hqpulse.ai", "srulynj@gmail.com"):
            assert '"%s"' % who in desc
        assert "stranger@example.com" not in desc, "another user's account is never a choice"
        saved = json.loads((tmp_path / "cache" / "composio-account-names.json").read_text())
        assert saved == {"ca_1r8-AuqvZ_oB": "yisrael@claimoe.ai"}
        assert fake.patches() == [("/api/v3/connected_accounts/ca_1r8-AuqvZ_oB", {"alias": "yisrael@claimoe.ai"})], \
            "an alias is given only where there was none"

    def test_naming_the_unnamed_accounts_address_sends_its_id(self, monkeypatch):
        """The 2026-10-07 call, as it should have gone."""
        s, _ = _augment_with_accounts(monkeypatch)
        n = len(s.session.call_tool.await_args_list)
        out = _call(s, "GMAIL_SEND_EMAIL", {"recipient_email": "a@example.com", "subject": "hi",
                                            "body": "x", "account": "yisrael@claimoe.ai"})
        name, args = _sent_after(s, n)
        assert name == "COMPOSIO_MULTI_EXECUTE_TOOL"
        assert args["tools"][0]["account"] == "ca_1r8-AuqvZ_oB"
        assert args["tools"][0]["arguments"] == {"recipient_email": "a@example.com", "subject": "hi", "body": "x"}
        assert "error" not in json.loads(out)

    @pytest.mark.parametrize("named,expected", [
        ("SRULY@HQPULSE.AI", "ca_S1ES5lxCn2qH"),     # any case
        ("ca_7NIHiXGSiOUe", "ca_7NIHiXGSiOUe"),      # the id itself
        ("claimoe", "ca_1r8-AuqvZ_oB"),              # a name that is part of exactly one address
    ])
    def test_an_address_id_or_unique_name_resolves(self, monkeypatch, named, expected):
        s, _ = _augment_with_accounts(monkeypatch)
        n = len(s.session.call_tool.await_args_list)
        _call(s, "GMAIL_FETCH_EMAILS", {"query": "in:inbox", "account": named})
        _, args = _sent_after(s, n)
        assert args["tools"][0]["account"] == expected

    @pytest.mark.parametrize("named", ["nobody@example.com", "stranger@example.com", "sruly"])
    def test_an_unknown_or_ambiguous_account_is_refused_with_the_choices(self, monkeypatch, named):
        """Never passed on, never the default. ``sruly`` is part of two
        addresses, so it names neither."""
        s, fake = _augment_with_accounts(monkeypatch)
        n = len(s.session.call_tool.await_args_list)
        out = json.loads(_call(s, "GMAIL_SEND_EMAIL", {"recipient_email": "a@example.com", "account": named}))
        assert s.session.call_tool.await_args_list[n:] == [], "nothing reaches Composio"
        msg = out["error"]
        assert named in msg and "Nothing was sent" in msg
        for who in ("yisrael@claimoe.ai", "sruly@hqpulse.ai", "srulynj@gmail.com"):
            assert who in msg

    def test_a_miss_rereads_the_list_once_so_a_new_account_is_found(self, monkeypatch):
        fake = _FakeComposio()
        s, _ = _augment_with_accounts(monkeypatch, fake)
        s._composio_accounts = (s._composio_accounts[0] - 60, s._composio_accounts[1])  # past the throttle
        fake.accounts.append({"id": "ca_NEWNEWNEW001", "alias": "new@example.com", "user_id": USER,
                              "status": "ACTIVE", "toolkit": {"slug": "gmail"},
                              "created_at": "2026-10-07T12:00:00.000Z"})
        n = len(s.session.call_tool.await_args_list)
        _call(s, "GMAIL_SEND_EMAIL", {"recipient_email": "a@example.com", "account": "new@example.com"})
        _, args = _sent_after(s, n)
        assert args["tools"][0]["account"] == "ca_NEWNEWNEW001"

    def test_a_learned_address_is_never_asked_for_again(self, monkeypatch):
        _augment_with_accounts(monkeypatch)
        s2, _ = _augment_with_accounts(monkeypatch)  # a fresh server, same Hermes home
        assert not [c for c in s2.session.call_tool.await_args_list if c.args[0] == "COMPOSIO_MULTI_EXECUTE_TOOL"]
        desc = next(t for t in s2._tools if t.name == "GMAIL_SEND_EMAIL").inputSchema["properties"]["account"]
        assert '"yisrael@claimoe.ai"' in desc["description"]

    def test_the_model_calling_the_multiplexer_itself_is_resolved_too(self, monkeypatch):
        s, _ = _augment_with_accounts(monkeypatch)
        n = len(s.session.call_tool.await_args_list)
        _call(s, "COMPOSIO_MULTI_EXECUTE_TOOL", {"tools": [
            {"tool_slug": "GMAIL_SEND_EMAIL", "arguments": {}, "account": "yisrael@claimoe.ai"}]})
        _, args = _sent_after(s, n)
        assert args["tools"][0]["account"] == "ca_1r8-AuqvZ_oB"

    def test_unreadable_accounts_leave_the_call_as_it_was(self, monkeypatch):
        """Composio then refuses an unknown name itself; it does not default."""
        s = _augment(_server())
        _call(s, "GMAIL_SEND_EMAIL", {"account": "yisrael@claimoe.ai"})
        _, args = _sent(s)
        assert args["tools"][0]["account"] == "yisrael@claimoe.ai"
