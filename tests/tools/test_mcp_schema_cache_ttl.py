"""SEP-2549 schema-cache TTL expiry (tools/mcp_schema_cache.py)."""

import time

import pytest

from tools import mcp_schema_cache as sc


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "_cache_path", lambda: tmp_path / "cache.json")
    yield


def test_entry_without_ttl_never_expires():
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}])
    assert sc.get_cached_entry("srv", "fp") is not None


def test_entry_within_ttl_served():
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    entry = sc.get_cached_entry("srv", "fp")
    assert entry is not None
    assert entry["ttl_ms"] == 60_000
    assert "written_at" in entry


def test_entry_past_ttl_is_a_miss(monkeypatch):
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=1_000)
    real_time = time.time
    monkeypatch.setattr(sc.time, "time", lambda: real_time() + 2.0)
    assert sc.get_cached_entry("srv", "fp") is None


def test_ttl_rewrite_advances_written_at():
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    first = sc.get_cached_entry("srv", "fp")["written_at"]
    time.sleep(0.01)
    # Identical payload would previously short-circuit; TTL'd entries must
    # rewrite so written_at advances on every live reconfirmation.
    sc.write_cache_entry("srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000)
    second = sc.get_cached_entry("srv", "fp")["written_at"]
    assert second > first


def test_cache_scope_round_trips():
    sc.write_cache_entry(
        "srv", "fp", tools=[{"name": "t"}], ttl_ms=60_000, cache_scope="private"
    )
    assert sc.get_cached_entry("srv", "fp")["cache_scope"] == "private"


class TestTheSDKDefaultIsNotAServerHint:
    """``ListToolsResult.ttl_ms`` defaults to 0 in mcp 2.x.

    Recording that default as a real SEP-2549 hint stamped every cache entry
    "expired the moment it was written", so the schema cache never hit once
    and every process start re-discovered every server live. Measured on a
    four-server config: 3.32 s of live discovery per process start that
    should have been — and now is — 0.58 s.
    """

    def test_a_default_ttl_is_not_recorded(self):
        import mcp.types as mt

        from tools.mcp_tool import mcp_field_was_set

        assert mt.ListToolsResult(tools=[]).ttl_ms == 0, (
            "this test exists because the SDK default is 0; if that changed, "
            "re-read the reasoning rather than deleting the guard"
        )
        assert not mcp_field_was_set(mt.ListToolsResult(tools=[]), "ttl_ms", "ttlMs")

    def test_a_server_that_really_sends_zero_is_honoured(self):
        import mcp.types as mt

        from tools.mcp_tool import mcp_field_was_set

        sent = mt.ListToolsResult(tools=[], ttl_ms=0)
        assert mcp_field_was_set(sent, "ttl_ms", "ttlMs")
        sc.write_cache_entry("zero", "fp", tools=[{"name": "t"}], ttl_ms=0)
        assert sc.get_cached_entry("zero", "fp") is None

    def test_a_real_ttl_is_still_recorded(self):
        import mcp.types as mt

        from tools.mcp_tool import mcp_field_was_set

        assert mcp_field_was_set(
            mt.ListToolsResult(tools=[], ttl_ms=60_000), "ttl_ms", "ttlMs"
        )

    def test_a_non_pydantic_result_falls_back_to_the_old_read(self):
        """mcp 1.x models and test doubles have no model_fields_set."""
        from tools.mcp_tool import mcp_field_was_set

        class Old:
            ttl_ms = 60_000

        class Bare:
            pass

        assert mcp_field_was_set(Old(), "ttl_ms", "ttlMs")
        assert not mcp_field_was_set(Bare(), "ttl_ms", "ttlMs")

    def test_the_capture_path_ignores_the_default(self):
        """End to end through the paginator's cache-hint capture."""
        import asyncio

        import mcp.types as mt

        from tools.mcp_tool import _paginate_full_list

        async def only_page(**kw):
            return mt.ListToolsResult(tools=[])

        meta: dict = {}
        asyncio.run(_paginate_full_list(only_page, "tools", "srv", cache_meta_out=meta))
        assert meta == {}, "the SDK's default leaked into the cache hints: %r" % (meta,)

        async def real_hint(**kw):
            return mt.ListToolsResult(tools=[], ttl_ms=45_000)

        meta = {}
        asyncio.run(_paginate_full_list(real_hint, "tools", "srv", cache_meta_out=meta))
        assert meta.get("ttl_ms") == 45_000
