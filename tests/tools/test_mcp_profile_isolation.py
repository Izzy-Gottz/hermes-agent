"""Cross-profile isolation contract for the MCP server registry.

A multiplexing gateway serves many profiles from ONE process. Every
inbound turn runs inside ``gateway.run._profile_runtime_scope``, which
installs that profile's ``HERMES_HOME`` override and its ``.env`` secret
scope. The MCP registry, however, used to be keyed by the *bare* server
name, so two profiles that each configure a server called ``github``
shared one subprocess -- started with whichever profile connected first,
holding that profile's credentials. Profile B's turns then reached
profile A's GitHub account.

These tests pin the fix: every module-level MCP registry structure is
partitioned by the active profile, derived from the SAME scope the rest
of the multiplexed path uses (``get_hermes_home_override()`` gated on
``agent.secret_scope.is_multiplex_active()``), and collapses to exactly
one partition -- today's behaviour -- when multiplexing is off.
"""

import contextlib

import pytest

import agent.secret_scope as secret_scope
import tools.mcp_tool as mcp
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


# The module-level registries that must be profile-partitioned. Kept as a
# list so a newly added registry is caught by the sweep test below.
_SCOPED_REGISTRY_NAMES = [
    "_servers",
    "_server_connecting",
    "_server_connect_errors",
    "_lazy_server_configs",
    "_lazy_server_fingerprints",
    "_lazy_server_tool_names",
    "_server_connect_retry_after",
    "_server_connect_failures",
    "_server_error_counts",
    "_server_breaker_opened_at",
    "_server_trust_levels",
    "_tool_read_only_hints",
    "_parallel_safe_servers",
]


@pytest.fixture(autouse=True)
def _clean_mcp_registries():
    """Snapshot every scoped registry and restore it afterwards."""
    saved = {}
    for name in _SCOPED_REGISTRY_NAMES:
        container = getattr(mcp, name)
        saved[name] = _snapshot(container)
        _wipe(container)
    was_multiplex = secret_scope.is_multiplex_active()
    yield
    secret_scope.set_multiplex_active(was_multiplex)
    for name in _SCOPED_REGISTRY_NAMES:
        container = getattr(mcp, name)
        _wipe(container)
        _restore(container, saved[name])


def _partitions(container):
    getter = getattr(container, "partitions", None)
    return getter() if callable(getter) else {"": container}


def _snapshot(container):
    return {
        key: (dict(part) if hasattr(part, "items") else set(part))
        for key, part in _partitions(container).items()
    }


def _wipe(container):
    for part in list(_partitions(container).values()):
        part.clear()


def _restore(container, snapshot):
    parts = _partitions(container)
    for key, value in snapshot.items():
        part = parts.get(key)
        if part is None:
            continue
        part.update(value)


@contextlib.contextmanager
def _profile(home):
    """Mimic ``gateway.run._profile_runtime_scope`` for one profile.

    Installs the same two seams the multiplexed inbound path installs: the
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


class _FakeServer:
    """Stand-in for a connected ``MCPServerTask`` (no subprocess, no I/O)."""

    def __init__(self, name, token):
        self.name = name
        self.token = token
        self.session = object()
        self._registered_tool_names = [f"mcp__{name}__do_thing"]
        self._tools = []
        self._sampling = None
        self.shutdown_calls = 0

    def _is_recycled_stdio(self):
        return False

    async def shutdown(self):
        self.shutdown_calls += 1
        self.session = None


@pytest.fixture
def homes(tmp_path):
    a = tmp_path / "profile-a"
    b = tmp_path / "profile-b"
    a.mkdir()
    b.mkdir()
    return a, b


# ── the leak ────────────────────────────────────────────────────────────


def test_two_profiles_same_server_name_do_not_share_a_connection(homes):
    """Profile B must never be handed profile A's live MCP server."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    server_a = _FakeServer("github", token="token-A")
    with _profile(home_a):
        mcp._servers["github"] = server_a
        assert mcp._get_connected_server_for_call("github") is server_a

    with _profile(home_b):
        leaked = mcp._get_connected_server_for_call("github")

    assert leaked is not server_a, (
        "profile B was handed profile A's 'github' MCP connection -- that "
        "subprocess holds profile A's credentials"
    )
    assert leaked is None


def test_lazy_config_cache_does_not_hand_b_profile_as_credentials(homes):
    """``_lazy_server_configs`` holds ALREADY-INTERPOLATED secrets.

    ``_load_mcp_config`` expands ``${GITHUB_TOKEN}`` through the active
    profile's secret scope before the config is cached, so a bare-name
    lazy cache literally stores profile A's token and would spawn profile
    B's first ``github`` call with it.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        mcp._lazy_server_configs["github"] = {
            "command": "npx",
            "env": {"GITHUB_TOKEN": "token-A"},
        }

    with _profile(home_b):
        leaked = mcp._lazy_server_configs.get("github")

    assert leaked is None, (
        "profile B's lazy start would spawn 'github' with profile A's "
        f"credentials: {leaked}"
    )


def test_registration_gate_does_not_skip_a_second_profile(homes):
    """``register_mcp_servers`` skips names already in the registry.

    With a bare-name registry that skip fires for profile B, so B never
    gets its own connection at all -- it just inherits A's.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        mcp._servers["github"] = _FakeServer("github", token="token-A")
        mcp._server_connecting.add("slack")
        mcp._lazy_server_configs["linear"] = {"command": "npx"}

    with _profile(home_b):
        assert "github" not in mcp._servers
        assert "slack" not in mcp._server_connecting
        assert "linear" not in mcp._lazy_server_configs


def test_connect_error_and_cooldown_state_is_per_profile(homes):
    """A failure in profile A must not blank out / block profile B."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        mcp._server_connect_errors["github"] = "bad token"
        mcp._record_connect_failure("github")
        assert mcp._connect_cooldown_active("github")

    with _profile(home_b):
        assert mcp._server_connect_errors.get("github") is None
        assert not mcp._connect_cooldown_active("github")


def test_trust_tier_is_per_profile(homes):
    """Profile A marking a server ``trust: full`` must not un-gate B's."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        mcp._server_trust_levels["github"] = mcp._TRUST_FULL
        mcp._tool_read_only_hints["github"] = {"mcp__github__push": True}

    with _profile(home_b):
        # B has not classified 'github' yet: it must NOT inherit A's
        # "full trust" tier or A's readOnlyHint exemptions, both of which
        # short-circuit the dangerous-call approval gate.
        assert mcp._server_trust_levels.get("github") is None
        assert mcp._tool_read_only_hints.get("github") is None
        mcp._server_trust_levels["github"] = mcp._TRUST_UNTRUSTED

    with _profile(home_a):
        assert mcp._server_trust_levels["github"] == mcp._TRUST_FULL


def test_circuit_breaker_counts_are_per_profile(homes):
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        for _ in range(mcp._CIRCUIT_BREAKER_THRESHOLD):
            mcp._bump_server_error("github")

    with _profile(home_b):
        assert mcp._server_error_counts.get("github", 0) == 0


def test_parallel_safe_flag_is_per_profile(homes):
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        mcp._parallel_safe_servers.add("github")
        assert "github" in mcp._parallel_safe_servers

    with _profile(home_b):
        assert "github" not in mcp._parallel_safe_servers


# ── single-profile behaviour must not change ────────────────────────────


def test_single_profile_gateway_is_unpartitioned(homes):
    """Multiplexing OFF: one partition, exactly today's behaviour.

    Even a HERMES_HOME override (plugins, ``mcp_startup`` discovery
    threads, ``bot_mode_probe``, the desktop backend) must NOT split the
    registry when the process is not a multiplexer -- otherwise a server
    started under an override would be looked up under a different key
    and silently orphaned.
    """
    home_a, home_b = homes
    secret_scope.set_multiplex_active(False)

    server = _FakeServer("github", token="token-A")
    with _profile(home_a):
        mcp._servers["github"] = server
        mcp._lazy_server_configs["linear"] = {"command": "npx"}

    with _profile(home_b):
        assert mcp._get_connected_server_for_call("github") is server
        assert mcp._lazy_server_configs.get("linear") == {"command": "npx"}

    # And with no scope at all.
    assert mcp._servers.get("github") is server
    # Exactly one populated partition -- the legacy root one.
    populated = [k for k, v in _partitions(mcp._servers).items() if v]
    assert populated == [mcp._ROOT_PROFILE_KEY]


def test_unscoped_multiplex_read_does_not_alias_a_profile(homes):
    """Fail-closed: an unscoped read under multiplexing gets its own slot.

    ``agent.secret_scope.get_secret`` raises on an unscoped read while
    multiplexing is on. The registry cannot raise (it is consulted from
    status/banner paths), so it does the next-safest thing: an unscoped
    caller gets a dedicated partition that can never alias a real
    profile's servers.
    """
    home_a, _ = homes
    secret_scope.set_multiplex_active(True)

    server = _FakeServer("github", token="token-A")
    with _profile(home_a):
        mcp._servers["github"] = server

    assert mcp._servers.get("github") is None


# ── teardown ────────────────────────────────────────────────────────────


def test_profile_teardown_leaves_other_profiles_running(homes):
    """``shutdown_mcp_servers(profile_only=True)`` reaps only this profile."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    server_a = _FakeServer("github", token="token-A")
    server_b = _FakeServer("github", token="token-B")
    with _profile(home_a):
        mcp._servers["github"] = server_a
    with _profile(home_b):
        mcp._servers["github"] = server_b

    mcp._ensure_mcp_loop()
    try:
        with _profile(home_b):
            mcp.shutdown_mcp_servers(profile_only=True)
            assert mcp._servers.get("github") is None

        assert server_b.shutdown_calls == 1
        assert server_a.shutdown_calls == 0
        with _profile(home_a):
            assert mcp._servers.get("github") is server_a
    finally:
        mcp._stop_mcp_loop()


def test_full_shutdown_reaps_every_profile(homes):
    """The process-exit path stays process-wide (unchanged default)."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    server_a = _FakeServer("github", token="token-A")
    server_b = _FakeServer("github", token="token-B")
    with _profile(home_a):
        mcp._servers["github"] = server_a
    with _profile(home_b):
        mcp._servers["github"] = server_b

    mcp._ensure_mcp_loop()
    mcp.shutdown_mcp_servers()

    assert server_a.shutdown_calls == 1
    assert server_b.shutdown_calls == 1
    with _profile(home_a):
        assert mcp._servers.get("github") is None
    with _profile(home_b):
        assert mcp._servers.get("github") is None


def test_idle_loop_check_sees_every_profile(homes):
    """Another profile's live server must keep the shared MCP loop alive."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        mcp._servers["github"] = _FakeServer("github", token="token-A")

    mcp._ensure_mcp_loop()
    try:
        with _profile(home_b):
            assert mcp._stop_mcp_loop_if_idle() is False
            assert mcp._mcp_loop is not None
    finally:
        with _profile(home_a):
            mcp._servers.clear()
        mcp._stop_mcp_loop()


# ── lifecycle keys stay consistent ──────────────────────────────────────


def test_reconnect_targets_only_the_calling_profile(homes):
    """``reconnect_mcp_server`` must not reach into another profile."""
    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        mcp._servers["github"] = _FakeServer("github", token="token-A")

    with _profile(home_b):
        assert mcp.reconnect_mcp_server("github") is False


def test_server_task_remembers_its_owning_profile(homes):
    """A long-lived server task addresses the partition it was born in.

    ``MCPServerTask.run()`` re-enters the registry to self-evict and to
    republish tools after a reconnect. It must always mean ITS OWN
    profile, never whatever context the loop task happens to carry.
    """
    from hermes_constants import hermes_home_key

    home_a, home_b = homes
    secret_scope.set_multiplex_active(True)

    with _profile(home_a):
        task_a = mcp.MCPServerTask("github")
    with _profile(home_b):
        task_b = mcp.MCPServerTask("github")

    assert task_a._profile_key == hermes_home_key(home_a)
    assert task_b._profile_key == hermes_home_key(home_b)
    assert task_a._profile_key != task_b._profile_key


def test_server_task_profile_key_is_root_without_multiplexing(homes):
    """Single-profile: the owning key is the one legacy partition."""
    home_a, _ = homes
    secret_scope.set_multiplex_active(False)

    with _profile(home_a):
        task = mcp.MCPServerTask("github")

    assert task._profile_key == mcp._ROOT_PROFILE_KEY
