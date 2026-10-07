"""A cua-driver installed before PM owned it is still the driver (fork).

On a Moe Mac cua-driver's own installer put /Applications/CuaDriver.app and linked
~/.local/bin/cua-driver to it, and the person granted THAT bundle Accessibility and Screen
Recording. Upstream's PM-only lookup ignored it: the first computer_use call then fetched
PM's (older) pin into the store — or, with lazy installs off, failed outright.
"""
import pytest


@pytest.fixture
def host(tmp_path, monkeypatch):
    from pm import paths

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    driver = tmp_path / ".local" / "bin" / "cua-driver"
    driver.parent.mkdir(parents=True)
    driver.write_text("#!/bin/sh\necho cua-driver 0.28.2\n")
    driver.chmod(0o755)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # a Dock-launched app's PATH: ~/.local/bin is not on it
    monkeypatch.setattr("tools.computer_use.cua_backend_driver._external_cua_driver_candidates",
                        lambda: ["cua-driver", str(driver)])
    return driver


def test_pre_pm_driver_resolves_when_pm_has_none(host):
    from tools.computer_use.cua_backend_driver import cua_driver_binary_available, resolve_cua_driver_cmd

    assert resolve_cua_driver_cmd() == str(host)
    assert cua_driver_binary_available()


def test_nothing_anywhere_is_still_none(host, monkeypatch):
    from tools.computer_use.cua_backend_driver import resolve_cua_driver_cmd

    host.unlink()
    assert resolve_cua_driver_cmd() is None


def test_backend_start_does_not_acquire_over_a_pre_pm_driver(host, monkeypatch):
    import pm
    from tools.computer_use import cua_backend

    def no_acquire(*a, **k):
        raise AssertionError("start() reached for PM over a driver that is already installed")

    monkeypatch.setattr(pm, "ensure", no_acquire)
    monkeypatch.setattr(cua_backend, "cua_driver_runtime_contract_status",
                        lambda *a, **k: {"ready": False, "reason": "stop here"})
    backend = cua_backend.CuaDriverBackend(permission_mode="standard")
    with pytest.raises(RuntimeError, match="stop here"):
        backend.start()


def test_explicit_install_keeps_a_pre_pm_driver_unless_upgrade(host, monkeypatch):
    import pm
    from hermes_cli import tools_config_cua as cfg

    calls = []
    monkeypatch.setattr(pm, "ensure", lambda name, **k: calls.append((name, k)))
    monkeypatch.setattr(cfg, "_cua_driver_contract_status", lambda binary: {"ready": False, "reason": "probe"})
    assert cfg.install_cua_driver(upgrade=False, show_installer_progress=False) is False  # failed on the contract
    assert calls == []  # ...but never acquired the pin beside the installed driver
    cfg.install_cua_driver(upgrade=True, show_installer_progress=False)
    assert calls == [("cua-driver", {"explicit": True})]
