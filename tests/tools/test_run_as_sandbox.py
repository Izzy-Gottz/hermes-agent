"""(fork) terminal.run_as_user: model-written code runs as a second uid (Memoe Away's shell).

What these prove, each red on revert of the matching change:
* off (the default) builds the plain LocalEnvironment: a Mac and upstream are unchanged;
* on, the backend is SandboxedLocalEnvironment, declares uses_host_filesystem False (so read_file /
  write_file go through the sandboxed shell, never the gateway's own open()), and fails CLOSED when the
  user does not exist or sudo is missing: nothing ever falls back to running as the gateway;
* the child's environment is an ALLOWLIST: the gateway's model key, Composio key, ask key, machine
  secret and harness credential never reach it, whatever their names;
* every command, and every browser_exec CLI, is launched through ``sudo -n -u <user> -- env -i``;
* background=true goes through the sandboxed env, not spawn_local's own Popen as the gateway;
* the screenshot attach never takes a path from the browser's output under run_as.
The live proof (EACCES on the real files, egress refused) is deploy/away/check-sandbox.sh on the machine.
"""
from __future__ import annotations

import os
import pwd
import signal
import types

import pytest

from tools.environments import run_as


def _target(tmp_path):
    return run_as.Target(user="sandbox", uid=10001, gid=10001, home=str(tmp_path / "sbx"), sudo="/usr/bin/sudo")


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv("TERMINAL_RUN_AS_USER", raising=False)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("TERMINAL_RUN_AS_USER", "sandbox")


SECRETS = {"OPENAI_API_KEY": "sk-x", "ANTHROPIC_API_KEY": "sk-ant", "COMPOSIO_API_KEY": "ak_x",
           "MOE_ASK_KEY": "ask", "MOE_MACHINE_SECRET": "boot", "API_SERVER_KEY": "srv",
           "TELEGRAM_WEBHOOK_SECRET": "tg", "BROWSER_USE_API_KEY": "bu", "BU_API_KEY": "bu2",
           "BH_AUTH_TOKEN": "bh", "CLAUDE_CODE_OAUTH_TOKEN": "cc", "GATEWAY_RELAY_TOKEN": "relay"}
#: Not secret, but the gateway's private tree: never handed to the child either.
PRIVATE = {"HERMES_HOME": "/opt/data", "MOE_HOME": "/opt/data/moe-home"}


def test_off_by_default_resolves_nothing(off):
    assert run_as.configured_user() == ""
    assert run_as.resolve() is None


def test_off_builds_the_plain_local_backend(off, monkeypatch):
    import tools.terminal_tool_backends as tb
    built = {}
    monkeypatch.setattr(tb, "_LocalEnvironment", lambda **kw: built.setdefault("plain", kw) or "plain")
    tb._build_local_env(cwd="/tmp", timeout=5)
    assert "plain" in built


def test_on_with_a_missing_user_fails_closed(monkeypatch):
    monkeypatch.setenv("TERMINAL_RUN_AS_USER", "no-such-user-zz9")
    import tools.terminal_tool_backends as tb
    monkeypatch.setattr(tb, "_LocalEnvironment", lambda **kw: pytest.fail("fell back to the gateway's uid"))
    with pytest.raises(run_as.RunAsUnavailable):
        tb._build_local_env(cwd="/tmp", timeout=5)


def test_on_with_a_real_user_builds_the_sandboxed_backend(tmp_path, monkeypatch):
    import tools.terminal_tool_backends as tb
    import tools.environments.local as local
    t = _target(tmp_path)
    monkeypatch.setattr(run_as, "resolve", lambda user=None: t)
    monkeypatch.setattr(tb, "_LocalEnvironment", lambda **kw: pytest.fail("built the gateway-uid backend"))
    built = {}
    monkeypatch.setattr(local, "SandboxedLocalEnvironment", lambda **kw: built.setdefault("sbx", kw))
    tb._build_local_env(cwd="/tmp", timeout=5)
    assert built["sbx"]["target"] is t


def test_on_without_sudo_fails_closed(monkeypatch):
    me = pwd.getpwuid(os.getuid())
    other = next((p for p in pwd.getpwall() if p.pw_uid not in (0, os.getuid())), None)
    if other is None:
        pytest.skip("no second user on this host")
    monkeypatch.setattr(run_as, "_sudo", lambda: None)
    with pytest.raises(run_as.RunAsUnavailable, match="sudo"):
        run_as.resolve(other.pw_name)
    with pytest.raises(run_as.RunAsUnavailable, match="separate"):
        run_as.resolve(me.pw_name)


def test_bridge_maps_the_config_key():
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    assert TERMINAL_CONFIG_ENV_MAP["run_as_user"] == "TERMINAL_RUN_AS_USER"


def test_clean_env_is_an_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/opt/data")
    t = _target(tmp_path)
    base = {**SECRETS, **PRIVATE, "PATH": "/opt/data/bin:/usr/bin", "LANG": "C.UTF-8", "SOMETHING_NEW": "x"}
    env = run_as.clean_env(t, extra={"BU_NAME": "r7k2", "BH_TMP_DIR": "/x", "BU_CDP_URL": "http://127.0.0.1:9"},
                           base=base)
    leaked = sorted(k for k in env if k in SECRETS or k in PRIVATE or env[k] in SECRETS.values())
    assert leaked == []
    assert "SOMETHING_NEW" not in env  # unknown names are dropped, not passed
    assert env["HOME"] == t.home and env["TMPDIR"] == t.tmp and env["USER"] == "sandbox"
    assert env["BU_NAME"] == "r7k2" and env["BU_CDP_URL"].startswith("http://127.0.0.1")
    assert "/opt/data/bin" not in env["PATH"].split(":")


def test_argv_is_sudo_noninteractive_with_env_i(tmp_path):
    t = _target(tmp_path)
    a = run_as.argv(t, ["bash", "-c", "id"], {"HOME": t.home})
    assert a[:7] == ["/usr/bin/sudo", "-n", "-u", "sandbox", "--", "/usr/bin/env", "-i"]
    assert a[-3:] == ["bash", "-c", "id"] and f"HOME={t.home}" in a


def test_sandboxed_backend_routes_file_tools_through_the_shell():
    from tools.environments.base import env_uses_host_filesystem
    from tools.environments.local import LocalEnvironment, SandboxedLocalEnvironment
    assert LocalEnvironment.uses_host_filesystem is True
    fake = SandboxedLocalEnvironment.__new__(SandboxedLocalEnvironment)
    assert env_uses_host_filesystem(fake) is False


def _bare_env(tmp_path):
    from tools.environments.local import SandboxedLocalEnvironment
    env = SandboxedLocalEnvironment.__new__(SandboxedLocalEnvironment)
    env._run_as = _target(tmp_path)
    env.env, env.cwd = {}, env._run_as.home
    return env


def test_every_command_is_launched_as_the_target(tmp_path, monkeypatch):
    import tools.environments.local as local
    seen = {}

    class P:
        pid = 4242

    def fake_popen(args, **kw):
        seen["args"], seen["kw"] = args, kw
        return P()
    monkeypatch.setattr(local.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(local, "own_process_groups", lambda pids: None)
    monkeypatch.setattr(local.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(local.os, "environ", {**SECRETS, "PATH": "/usr/bin"})
    env = _bare_env(tmp_path)
    env._run_bash("cat /opt/data/.env")
    args = seen["args"]
    assert args[:5] == ["/usr/bin/sudo", "-n", "-u", "sandbox", "--"]
    assert args[-1] == "cat /opt/data/.env"
    assert not any(v in " ".join(args) for v in SECRETS.values() if len(v) > 3)
    assert seen["kw"]["cwd"] == "/" and set(seen["kw"]["env"]) == {"PATH"}


def test_the_claude_code_token_hermes_lets_through_never_reaches_the_sandbox(tmp_path, monkeypatch):
    """Measured on Memoe's cloud machine: CLAUDE_CODE_OAUTH_TOKEN reaches every plain terminal child,
    because local_env_policy deliberately does not block it (#55878). The control shows that route is
    real; the sandboxed launch must still carry no credential of any name."""
    import re
    import tools.environments.local as local
    token = "sk-ant-oat01-" + "q" * 40
    creds = {"CLAUDE_CODE_OAUTH_TOKEN": token, "MOE_MACHINE_SECRET": "m" * 40,
             "ANTHROPIC_API_KEY": "a" * 40, "COMPOSIO_API_KEY": "ak_" + "c" * 40}
    monkeypatch.setattr(local.os, "environ", {**creds, "PATH": "/usr/bin"})
    # Control: the plain run env Hermes builds DOES carry the token (so this test can fail).
    assert local._make_run_env({}).get("CLAUDE_CODE_OAUTH_TOKEN") == token
    seen = {}

    class P:
        pid = 4243

    monkeypatch.setattr(local.subprocess, "Popen", lambda args, **kw: (seen.update(args=args, kw=kw), P())[1])
    monkeypatch.setattr(local, "own_process_groups", lambda pids: None)
    monkeypatch.setattr(local.os, "getpgid", lambda pid: pid)
    env = _bare_env(tmp_path)
    env.env = dict(creds)          # and handed to the backend directly, as a config env would be
    env._run_bash("env")
    args = seen["args"]
    joined = " ".join(args)
    assert not any(v in joined for v in creds.values())
    start = args.index("-i") + 1
    names = [a.split("=", 1)[0] for a in args[start:] if "=" in a and not a.startswith("-")
             and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", a.split("=", 1)[0])]
    credential_like = [n for n in names if re.search(r"KEY|TOKEN|SECRET|PASS|AUTH|CREDENTIAL|COOKIE", n, re.I)]
    assert credential_like == [], credential_like
    assert set(seen["kw"]["env"]) == {"PATH"}


def test_kill_signals_the_group_as_the_target(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_as, "kill_group", lambda t, pgid, sig=signal.SIGKILL: calls.append((pgid, sig)))
    import tools.environments.local as local
    monkeypatch.setattr(local.LocalEnvironment, "_kill_process", lambda self, proc: None)
    env = _bare_env(tmp_path)
    env._kill_process(types.SimpleNamespace(pid=77, _hermes_pgid=77))
    assert (77, signal.SIGKILL) in calls


def test_background_goes_through_the_sandboxed_env(tmp_path):
    from tools.terminal_tool_background import _spawn
    env = _bare_env(tmp_path)
    reg = types.SimpleNamespace(
        spawn_local=lambda **kw: pytest.fail("background ran through spawn_local, as the gateway's uid"),
        spawn_via_env=lambda env, **kw: "via-env")
    out = _spawn(reg, env=env, env_type="local", command="cat /opt/data/.env", cwd=None,
                 effective_task_id="t", task_id="t", session_key="", effective_pty=False)
    assert out == "via-env"


def test_background_off_still_uses_spawn_local(off):
    from tools.terminal_tool_background import _spawn
    from tools.environments.local import LocalEnvironment
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.env = {}
    reg = types.SimpleNamespace(spawn_local=lambda **kw: "local", spawn_via_env=lambda env, **kw: "via-env")
    assert _spawn(reg, env=env, env_type="local", command="true", cwd=None, effective_task_id="t",
                  task_id="t", session_key="", effective_pty=False) == "local"


def test_yield_to_background_is_off_under_run_as(on):
    from tools.terminal_tool_background import yield_to_background_handler
    assert yield_to_background_handler(command="sleep 9", env_type="local", cwd=None, effective_task_id="t",
                                       task_id="t", session_key="") is None


def test_browser_cli_is_launched_as_the_target(tmp_path, monkeypatch):
    import tools.browser_use_cli as cli
    t = _target(tmp_path)
    monkeypatch.setattr(cli, "_run_as_target", lambda: t)
    seen = {}

    class P:
        pid, returncode = 99, 0

        def communicate(self, input=None, timeout=None):
            return "", ""
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **kw: (seen.update(cmd=cmd, kw=kw), P())[1])
    cli._run_cli_killing_process_group(["/opt/hermes/.venv/bin/python", "-m", "browser_harness.run"], "print(1)",
                                       {**SECRETS, "BU_NAME": "r7k2", "PATH": "/usr/bin"}, 10)
    cmd = seen["cmd"]
    assert cmd[:5] == ["/usr/bin/sudo", "-n", "-u", "sandbox", "--"]
    assert cmd[-3:] == ["/opt/hermes/.venv/bin/python", "-m", "browser_harness.run"]
    assert "BU_NAME=r7k2" in cmd
    assert not any(f"{k}=" in a for a in cmd for k in SECRETS)


def test_browser_base_env_carries_no_harness_credential(tmp_path, monkeypatch):
    import tools.browser_use_cli as cli
    import tools.browser_session as bs
    t = _target(tmp_path)
    monkeypatch.setattr(cli, "_run_as_target", lambda: t)
    monkeypatch.setattr(bs, "build_browser_env", lambda: {"PATH": "/usr/bin"})
    monkeypatch.setattr(cli, "_harness_credentials", lambda: {"BROWSER_USE_API_KEY": "bu"})
    env = cli._base_subprocess_env()
    assert "BROWSER_USE_API_KEY" not in env
    assert env["HOME"] == t.home and env["BH_TMP_DIR"].startswith(t.home)


def test_screenshot_path_from_output_is_never_read_under_run_as(tmp_path, monkeypatch):
    import tools.browser_use_cli as cli
    planted = tmp_path / "gateway-private.png"
    planted.write_bytes(b"\x89PNG\r\n\x1a\nsecret")
    monkeypatch.setattr(cli, "_run_as_target", lambda: _target(tmp_path))
    monkeypatch.setattr(cli, "_fetch_sandbox_screenshot", lambda t, since: None)
    assert cli._find_screenshot(f"saved {planted}", 0) is None


def test_screenshot_path_from_output_still_works_off(off, tmp_path, monkeypatch):
    import tools.browser_use_cli as cli
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(cli, "_run_as_target", lambda: None)
    assert cli._find_screenshot(f"saved {shot}", 0) == str(shot)
