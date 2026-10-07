"""Keep the fork's pre-PM cua-driver discovery off this suite's host.

``resolve_cua_driver_cmd`` falls back to a driver installed before PM owned it (PATH,
~/.local/bin, /Applications/CuaDriver.app). These tests pin PM-selection behaviour, so a
developer Mac that has CuaDriver.app installed must not answer for them. Tests of the
fallback itself patch ``_external_cua_driver_candidates`` back in.
"""
import pytest


@pytest.fixture(autouse=True)
def _no_host_cua_driver(monkeypatch):
    monkeypatch.setattr("tools.computer_use.cua_backend_driver._external_cua_driver_candidates", lambda: [])
