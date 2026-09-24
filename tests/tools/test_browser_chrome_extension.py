"""The Chrome-extension lane of browser_exec: lane choice, bot-wall detection, the bridge handshake,
the one-step host install, and the tool mapping (what the CLI is actually handed)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tools import browser_chrome_extension as lane
from tools import browser_use_cli as bu

ON = {"chrome_extension": {"enabled": True}}
REPO = Path(__file__).resolve().parents[2]

# Page text measured 2026-09-23 (browser_exec page_info + innerText), not invented:
SKYSCANNER_PX = ("{'url': 'https://www.skyscanner.com/sttc/px/captcha-v2/index.html?url=L3RyYW5zcG9ydC9m', "
                 "'title': 'Skyscanner'}\nAre you a person or a robot?\nPlease don’t take this personally")
PRODUCTHUNT_CF = "{'url': 'https://www.producthunt.com/products/moe-2', 'title': '\U0001F434 Just a moment...'}"
SANNYSOFT_OK = "{'url': 'https://bot.sannysoft.com/', 'title': 'Antibot'}\nIntoli.com tests + additions\nTest Name\tResult"
TRIP_OK = "{'url': 'https://www.trip.com/flights/', 'title': 'Cheap Flights, Airline Tickets & Plane Tickets | Trip.com'}"


def _bind(platform="api_server", origin="person", cron=""):
    """Bind a turn's session context the way the gateway does (contextvars, not env)."""
    from gateway.session_context import set_session_vars, set_turn_origin
    tokens = set_session_vars(platform=platform, cron_session=cron)
    return tokens, set_turn_origin(origin)


def _unbind(bound):
    from gateway.session_context import clear_session_vars, reset_turn_origin
    tokens, origin_token = bound
    reset_turn_origin(origin_token)
    clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MCP_TOOL_PROFILE", raising=False)
    lane.reset_sticky_lanes()
    bound = _bind()  # by default: the person's own live turn in the Memoe app
    yield
    _unbind(bound)
    lane.reset_sticky_lanes()


def choose(where="", code="", cfg=ON, connected=True, session="", task="t"):
    return lane.choose_lane(where, code, task_id=task, session=session, browser_cfg=cfg, connected=connected)[0]


# ---- lane choice -------------------------------------------------------------------

def test_explicit_where_is_final_whatever_the_url():
    assert choose("chrome", 'new_tab("https://example.com")') == "chrome"
    assert choose("own", 'new_tab("https://www.skyscanner.com/")') == "own"


def test_disabled_lane_never_routes():
    assert choose("", 'new_tab("https://www.skyscanner.com/")', cfg={}) == "own"


def test_blocking_host_goes_to_chrome_only_when_connected_and_auto():
    code = 'new_tab("https://www.skyscanner.com/transport/flights/nyca/lond/")'
    assert choose("", code) == "chrome"
    assert choose("", code, connected=False) == "own"
    assert choose("", code, cfg={"chrome_extension": {"enabled": True, "auto_route": False}}) == "own"


def test_host_matching_is_by_domain_not_substring():
    assert choose("", 'goto_url("https://mobile.x.com/home")') == "chrome"
    assert choose("", 'goto_url("https://netflix.com/")') == "own"      # ends in "x.com" as text only
    assert choose("", 'goto_url("https://trip.com.evil.example/")') == "own"


def test_configured_and_learned_hosts_route():
    assert choose("", 'goto_url("https://www.kayak.com/")') == "own"
    assert choose("", 'goto_url("https://www.kayak.com/")', cfg={"chrome_extension": {"enabled": True, "sites": ["kayak.com"]}}) == "chrome"
    lane.remember_blocking_host("www.example-shop.test")
    assert choose("", 'goto_url("https://www.example-shop.test/cart")') == "chrome"


def test_learned_blockers_expire(tmp_path):
    t0 = 1_800_000_000.0
    lane.remember_blocking_host("www.example-shop.test", now=t0)
    assert lane.learned_blocking_hosts(now=t0 + lane.LEARNED_TTL_S - 60) == ["www.example-shop.test"]
    assert lane.learned_blocking_hosts(now=t0 + lane.LEARNED_TTL_S + 60) == []
    (tmp_path / "chrome-bridge" / "blocked-hosts.json").write_text('["legacy.example"]')  # old list format
    assert lane.learned_blocking_hosts() == []


@pytest.mark.parametrize("platform,origin,cron,why", [
    ("api_server", "person", "1", "a scheduled job"),
    ("webhook", "", "", "a webhook turn"),
    ("msgraph_webhook", "", "", "a msgraph webhook turn"),
    ("api_server", "background", "", "a background note from the app"),
    ("api_server", "", "", "an API turn whose client did not say a person started it"),
    ("", "", "", "a turn whose origin could not be established"),
])
def test_only_a_persons_live_turn_auto_routes_to_their_chrome(platform, origin, cron, why):
    code = 'new_tab("https://www.skyscanner.com/")'
    assert choose("", code) == "chrome"   # the fixture's live person turn
    bound = _bind(platform, origin, cron)
    try:
        assert lane.local_turn_presence() == {"live": False, "why": why}
        assert choose("", code) == "own"
    finally:
        _unbind(bound)


@pytest.mark.parametrize("platform", ["cli", "telegram"])
def test_a_person_at_the_cli_or_on_a_chat_channel_is_live(platform):
    bound = _bind(platform, "", "")
    try:
        assert lane.local_turn_presence()["live"] is True
    finally:
        _unbind(bound)


def test_single_query_marker_is_not_absence(monkeypatch):
    # The claude-code MCP server sets it on every turn; it means "no approval prompt", not "nobody here".
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    assert lane.local_turn_presence()["live"] is True


def test_kanban_worker_is_not_live(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    assert lane.local_turn_presence()["live"] is False


def test_inside_the_mcp_server_with_no_bridge_nothing_is_live(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "claude-code")
    monkeypatch.delenv("HERMES_TOOL_BRIDGE_SOCKET", raising=False)
    assert lane.turn_presence()["live"] is False     # even though this thread's context says "person"


def test_a_call_without_a_url_stays_in_its_sessions_lane():
    assert choose("", "print(page_info())", session="s1") == "own"
    lane.record_lane("t", "s1", "chrome")
    assert choose("", "print(page_info())", session="s1") == "chrome"
    assert choose("", "print(page_info())", session="other") == "own"


# ---- bot walls -------------------------------------------------------------------------

@pytest.mark.parametrize("text,label", [(SKYSCANNER_PX, "perimeterx"), (PRODUCTHUNT_CF, "cloudflare"),
                                        (SANNYSOFT_OK, None), (TRIP_OK, None), ("", None)])
def test_detect_block_on_measured_pages(text, label):
    assert lane.detect_block(text) == label


# ---- the bridge handshake file -------------------------------------------------------------

def _write_bridge(tmp_path, **over):
    d = tmp_path / "chrome-bridge"
    d.mkdir(exist_ok=True)
    data = {"ws_url": "ws://127.0.0.1:5555/devtools/browser/abc", "pid": os.getpid(),
            "extension_origin": f"chrome-extension://{lane.EXTENSION_ID}/"}
    data.update(over)
    p = d / "bridge.json"
    p.write_text(json.dumps(data))
    os.chmod(p, 0o600)
    return p


def test_read_bridge_accepts_only_a_private_live_loopback_bridge_from_our_extension(tmp_path):
    p = _write_bridge(tmp_path)
    assert lane.read_bridge()["ws_url"].startswith("ws://127.0.0.1:")
    os.chmod(p, 0o644)
    assert lane.read_bridge() is None                                  # readable by others
    _write_bridge(tmp_path, ws_url="ws://10.0.0.5:5555/devtools/browser/abc")
    assert lane.read_bridge() is None                                  # not loopback
    _write_bridge(tmp_path, extension_origin="chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/")
    assert lane.read_bridge() is None                                  # some other extension's host
    dead = subprocess.Popen(["true"])
    dead.wait()
    _write_bridge(tmp_path, pid=dead.pid)
    assert lane.read_bridge() is None                                  # host exited


# ---- install --------------------------------------------------------------------------------

def test_extension_id_is_the_one_its_manifest_key_produces():
    manifest = json.loads((REPO / "extensions" / "chrome" / "manifest.json").read_text())
    digest = hashlib.sha256(base64.b64decode(manifest["key"])).hexdigest()[:32]
    assert "".join(chr(ord("a") + int(c, 16)) for c in digest) == lane.EXTENSION_ID


def test_manifest_asks_only_for_the_permissions_it_uses():
    manifest = json.loads((REPO / "extensions" / "chrome" / "manifest.json").read_text())
    assert manifest["manifest_version"] == 3
    assert sorted(manifest["permissions"]) == ["debugger", "nativeMessaging", "storage", "tabGroups"]
    assert "host_permissions" not in manifest and "content_scripts" not in manifest


def test_install_writes_one_manifest_beside_the_profiles_and_nothing_inside_them(tmp_path):
    browser_dir = tmp_path / "Chrome"
    (browser_dir / "Default").mkdir(parents=True)
    (browser_dir / "Default" / "Preferences").write_text("{}")
    manifest, launcher = lane.install_native_host(str(browser_dir), hermes_home=str(tmp_path / "hh"), python="/usr/bin/python3")
    assert manifest == browser_dir / "NativeMessagingHosts" / "app.memoe.chrome_bridge.json"
    data = json.loads(manifest.read_text())
    assert data["allowed_origins"] == [f"chrome-extension://{lane.EXTENSION_ID}/"] and data["type"] == "stdio"
    assert data["path"] == str(launcher) and os.access(launcher, os.X_OK)
    assert oct(launcher.stat().st_mode & 0o777) == "0o700"
    assert sorted(p.name for p in browser_dir.iterdir()) == ["Default", "NativeMessagingHosts"]
    assert (browser_dir / "Default" / "Preferences").read_text() == "{}"
    assert "unset PYTHONPATH" in launcher.read_text() and "browser_chrome_bridge.py" in launcher.read_text()


# ---- tool mapping: what browser_exec hands the CLI ---------------------------------------------

@pytest.fixture()
def cli(monkeypatch):
    calls = {"route": 0}

    def fake_run(cmd, code, env, timeout):
        calls.update(cmd=cmd, code=code, env=dict(env))
        return subprocess.CompletedProcess(cmd, 0, calls.get("stdout", "ok\n"), "")

    def fake_route(env, session, task_id, local):
        calls["route"] += 1
        return None

    monkeypatch.setattr(bu, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(bu, "_run_cli_killing_process_group", fake_run)
    monkeypatch.setattr(bu, "_route_backend", fake_route)
    monkeypatch.setattr(bu, "_attach_vault_supervisor", lambda env, task_id: None)
    monkeypatch.setattr(bu, "_workspace_dir", lambda task_id: None)
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {"chrome_extension": {"enabled": True}})
    return calls


def test_where_chrome_points_the_harness_at_the_bridge_and_nowhere_else(tmp_path, cli):
    _write_bridge(tmp_path)
    out = json.loads(bu.browser_exec('new_tab("https://example.com")\nprint(page_info())', session="s1",
                                     task_id="t", where="chrome"))
    env = cli["env"]
    assert out["success"] and out["lane"] == "chrome"
    assert env["BU_CDP_WS"] == "ws://127.0.0.1:5555/devtools/browser/abc"
    assert env["BU_NAME"] == "chrome-s1"                  # its own daemon, never the own-browser one
    assert "BU_CDP_URL" not in env and "BU_BROWSER_ID" not in env
    assert cli["route"] == 0                              # the own-browser backend was never resolved
    assert "_hermes_ensure_own_tab" not in cli["code"]    # the relay's dedicated tab replaces the preamble


def test_where_chrome_without_the_extension_fails_closed_with_a_fix_code(cli):
    out = json.loads(bu.browser_exec('print(page_info())', task_id="t", where="chrome"))
    assert out["code"] == "browser_extension_missing" and out["owner"] == "app" and out["pane"] is None
    assert out["subject"] == "Memoe extension for Google Chrome" and out["retry"] is True
    assert out["extension_id"] == lane.EXTENSION_ID and out["native_host"] == "app.memoe.chrome_bridge"
    assert "cmd" not in cli and cli["route"] == 0         # nothing ran anywhere


def test_where_chrome_with_the_lane_off_runs_nothing(monkeypatch, tmp_path, cli):
    _write_bridge(tmp_path)
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {})
    out = json.loads(bu.browser_exec('print(1)', task_id="t", where="chrome"))
    assert "error" in out and "cmd" not in cli


def test_invalid_where_is_rejected(cli):
    assert "Invalid where" in json.loads(bu.browser_exec('print(1)', task_id="t", where="firefox"))["error"]


def test_a_bot_wall_in_the_own_lane_teaches_the_router_only_the_host_that_served_it(tmp_path, cli):
    _write_bridge(tmp_path)
    cli["stdout"] = SKYSCANNER_PX.replace("www.skyscanner.com", "www.skyscanner.co.il")
    code = 'new_tab("https://www.skyscanner.co.il/transport/flights/")\nfetch_other("https://api.innocent.example/")'
    out = json.loads(bu.browser_exec(code, task_id="t", where="own"))
    assert out["lane"] == "own" and out["blocked_by"] == "perimeterx" and 'where="chrome"' in out["hint"]
    assert cli["route"] == 1
    assert lane.learned_blocking_hosts() == ["www.skyscanner.co.il"]   # not api.innocent.example from the code
    assert choose("", 'goto_url("https://www.skyscanner.co.il/")') == "chrome"


@pytest.mark.parametrize("stdout", [
    "{'url': 'https://blog.example/post', 'title': 'How reCAPTCHA works'}\nverify you are human, unusual traffic, akamai",
    "{'url': 'https://news.example/', 'title': 'News'}\nAre you a robot? A quiz.",
])
def test_words_on_a_normal_page_hint_but_never_teach(tmp_path, cli, stdout):
    _write_bridge(tmp_path)
    cli["stdout"] = stdout
    out = json.loads(bu.browser_exec('new_tab("https://blog.example/post")', task_id="t", where="own"))
    assert out.get("blocked_by")          # a hint for the model is fine
    assert lane.learned_blocking_hosts() == []


@pytest.mark.parametrize("text,wall", [
    (SKYSCANNER_PX, ("perimeterx", "www.skyscanner.com")),
    (PRODUCTHUNT_CF, ("cloudflare", "www.producthunt.com")),
    ("{'url': 'https://shop.example/', 'title': 'Access Denied'}\nReference #18.abc", ("akamai", "shop.example")),
    ('{"url":"https://a.example/?__cf_chl_tk=x","title":"a"}', ("cloudflare", "a.example")),
    (SANNYSOFT_OK, None), (TRIP_OK, None),
    ("see https://www.skyscanner.com/sttc/px/captcha-v2/ in the docs", None),  # no page_info: not the page itself
])
def test_wall_host_is_signature_grade(text, wall):
    assert lane.wall_host(text) == wall


def test_no_person_present_explicit_chrome_is_refused_and_the_block_is_reported(monkeypatch, tmp_path, cli):
    _write_bridge(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t1")
    out = json.loads(bu.browser_exec('print(1)', task_id="t", where="chrome"))
    assert "not the person's own live turn" in out["error"] and "cmd" not in cli
    cli["stdout"] = SKYSCANNER_PX
    out = json.loads(bu.browser_exec('new_tab("https://www.skyscanner.com/")', task_id="t"))
    assert out["lane"] == "own" and out["blocked_by"] == "perimeterx" and "Report the block" in out["hint"]


def test_auto_routing_reaches_the_bridge_for_a_known_blocker(tmp_path, cli):
    _write_bridge(tmp_path)
    out = json.loads(bu.browser_exec('new_tab("https://x.com/explore")', task_id="t"))
    assert out["lane"] == "chrome" and "x.com" in out["lane_reason"]
    assert cli["env"]["BU_CDP_WS"].startswith("ws://127.0.0.1:")


def test_where_is_in_the_schema_only_when_the_lane_is_on(monkeypatch):
    monkeypatch.setattr(bu, "_real_profile_consented", lambda: False)
    monkeypatch.setattr(bu, "_description_header", lambda: "")
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {})
    assert "where" not in (bu._dynamic_schema_overrides().get("parameters") or bu.BROWSER_EXEC_SCHEMA["parameters"])["properties"]
    monkeypatch.setattr(bu, "_read_browser_cfg", lambda: {"chrome_extension": {"enabled": True}})
    assert bu._dynamic_schema_overrides()["parameters"]["properties"]["where"]["enum"] == ["own", "chrome"]


def test_file_access_on_is_a_warning_for_the_app():
    assert lane.bridge_warnings({"extension": {"fileAccess": False}}) == []
    assert [w["code"] for w in lane.bridge_warnings({"extension": {"fileAccess": True}})] == ["file_access_on"]
