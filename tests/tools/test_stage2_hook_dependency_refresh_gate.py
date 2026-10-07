"""(fork) HERMES_SKIP_DEPENDENCY_REFRESH=1 skips stage2's per-boot dependency refresh.

Memoe's away image has its extras fixed at build time and lazy installs off; it must not
re-resolve anything from the network at boot. The block is run as stage2 runs it, with
``s6-setuidgid`` stubbed to record whether Python was asked to refresh.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STAGE2_HOOK = Path(__file__).resolve().parents[2] / "docker" / "stage2-hook.sh"


def _block() -> str:
    text = STAGE2_HOOK.read_text()
    start = text.index("# --- Refresh the dependency generation for this image ---")
    end = text.index("# auth.json: bootstrap from env on first boot only.", start)
    return text[start:end]


@pytest.mark.parametrize("skip", ["", "1"])
def test_the_refresh_runs_unless_skipped(tmp_path, skip):
    if shutil.which("sh") is None:
        pytest.skip("sh not available")
    calls = tmp_path / "calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "s6-setuidgid"  # a hyphenated name cannot be a POSIX sh function
    stub.write_text(f'#!/bin/sh\necho "$*" >> "{calls}"\n')
    stub.chmod(0o755)
    script = (
        "set -eu\n"
        f'PATH="{bin_dir}:$PATH"\n'
        f'INSTALL_DIR="{tmp_path}"\n'
        + (f"HERMES_SKIP_DEPENDENCY_REFRESH={skip}\n" if skip else "unset HERMES_SKIP_DEPENDENCY_REFRESH\n")
        + _block()
    )
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    ran = calls.exists() and "refresh_dependencies" in calls.read_text()
    if skip:
        assert not ran and "dependency refresh skipped" in out.stdout
    else:
        assert ran
