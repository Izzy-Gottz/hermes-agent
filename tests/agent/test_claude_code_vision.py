"""The Claude Code CLI as a vision lane.

Closes the hole that left a subscription user with no eyes at all: the
auxiliary client is HTTP-shaped, `claude-code-cli` has no HTTP client, so
every screenshot was routed to a third-party auxiliary provider — and when
that provider's balance ran out, `computer_use(action='capture')` returned

    402 - {'error': {'message': 'Insufficient Balance'}}

Verified live on the owner's Mac (2026-09-07) against a 5.8 MB screenshot:
a path answered in 7.8 s, a base64 data URL in 5.4 s, both correctly naming
the page. Those runs are what these fakes are shaped from.
"""

from __future__ import annotations

import base64
import os
import subprocess

import pytest

from agent import claude_code_vision as V


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()
DATA_URL = "data:image/png;base64," + PNG


class _Result:
    def __init__(self, code=0, out="a screenshot of a web page", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


@pytest.fixture
def spy(monkeypatch):
    """Capture the argv and stdin the CLI would receive."""
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["input"] = kw.get("input", "")
        seen["timeout"] = kw.get("timeout")
        return seen.get("result", _Result())

    monkeypatch.setattr(V.shutil, "which", lambda name: "/usr/local/bin/claude")
    monkeypatch.setattr(V.subprocess, "run", fake_run)
    return seen


# ── the two details that were learned by getting them wrong ───────────────

def test_the_prompt_goes_on_stdin(spy):
    """`claude -p "<prompt>"` was rejected: "Input must be provided either
    through stdin or as a prompt argument when using --print"."""
    V.describe_image(DATA_URL, "what is this?")
    assert "what is this?" in spy["input"]
    assert "what is this?" not in " ".join(spy["argv"])


def test_the_image_is_passed_as_a_path_for_the_read_tool(spy):
    """Mentioning an image without a readable path gets "I don't see an image
    in your message" — measured."""
    V.describe_image(DATA_URL, "q")
    assert "--allowedTools" in spy["argv"]
    assert spy["argv"][spy["argv"].index("--allowedTools") + 1] == "Read"
    assert "Read the image at" in spy["input"]


def test_a_path_with_spaces_is_quoted(spy, tmp_path):
    """Moe's own home is ~/Library/Application Support/Moe."""
    d = tmp_path / "Application Support"
    d.mkdir()
    shot = d / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    V.describe_image(str(shot), "q")
    assert '"%s"' % shot in spy["input"]


# ── inputs ────────────────────────────────────────────────────────────────

def test_a_local_path_is_used_as_is(spy, tmp_path):
    shot = tmp_path / "a.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    V.describe_image(str(shot), "q")
    assert str(shot) in spy["input"]


def test_a_data_url_is_decoded_to_a_real_file(spy, monkeypatch):
    """The aux lane hands over a data URL; the CLI's Read tool needs a file."""
    written = {}
    real = V._decode_to_file

    def wrapper(image, directory):
        path = real(image, directory)
        written["path"] = path
        written["bytes"] = os.path.getsize(path)
        return path

    monkeypatch.setattr(V, "_decode_to_file", wrapper)
    V.describe_image(DATA_URL, "q")
    assert written["bytes"] > 0
    assert written["path"].endswith(".png")


def test_the_temp_file_does_not_outlive_the_call(spy, monkeypatch):
    """A screenshot is the person's screen. It must not be left on disk."""
    kept = {}
    real = V._decode_to_file

    def wrapper(image, directory):
        path = real(image, directory)
        kept["path"] = path
        return path

    monkeypatch.setattr(V, "_decode_to_file", wrapper)
    V.describe_image(DATA_URL, "q")
    assert not os.path.exists(kept["path"])


def test_a_jpeg_data_url_keeps_its_extension(spy):
    V.describe_image("data:image/jpeg;base64," + PNG, "q")
    assert 'Read the image at "' in spy["input"]


# ── the model, and the default ────────────────────────────────────────────

def test_haiku_is_the_default(spy):
    V.describe_image(DATA_URL, "q")
    assert spy["argv"][spy["argv"].index("--model") + 1] == "haiku"


def test_the_model_can_be_overridden(spy):
    V.describe_image(DATA_URL, "q", model="sonnet")
    assert spy["argv"][spy["argv"].index("--model") + 1] == "sonnet"


def test_an_empty_question_still_asks_something(spy):
    V.describe_image(DATA_URL, "   ")
    assert "Describe this image." in spy["input"]


# ── falling through, never failing hard ───────────────────────────────────
#
# Every failure here raises ClaudeCodeVisionUnavailable so the caller can try
# the ordinary auxiliary chain. A hard error would trade one dead lane for
# another, which is the bug this module exists to fix.

def test_no_cli_is_unavailable_not_an_error(monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda name: None)
    with pytest.raises(V.ClaudeCodeVisionUnavailable):
        V.describe_image(DATA_URL, "q")


def test_a_nonzero_exit_is_unavailable(spy):
    spy["result"] = _Result(code=1, out="", err="not logged in")
    with pytest.raises(V.ClaudeCodeVisionUnavailable) as e:
        V.describe_image(DATA_URL, "q")
    assert "not logged in" in str(e.value)


def test_empty_output_is_unavailable(spy):
    spy["result"] = _Result(out="   ")
    with pytest.raises(V.ClaudeCodeVisionUnavailable):
        V.describe_image(DATA_URL, "q")


def test_a_timeout_is_unavailable(monkeypatch):
    monkeypatch.setattr(V.shutil, "which", lambda name: "/usr/local/bin/claude")

    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw.get("timeout", 1))

    monkeypatch.setattr(V.subprocess, "run", boom)
    with pytest.raises(V.ClaudeCodeVisionUnavailable) as e:
        V.describe_image(DATA_URL, "q", timeout=5)
    assert "5s" in str(e.value)


def test_a_missing_file_is_unavailable(spy):
    with pytest.raises(V.ClaudeCodeVisionUnavailable):
        V.describe_image("/no/such/file.png", "q")


def test_undecodable_data_is_unavailable(spy):
    with pytest.raises(V.ClaudeCodeVisionUnavailable):
        V.describe_image("data:image/png;base64,!!!not base64!!!", "q")


def test_an_empty_image_is_unavailable(spy):
    with pytest.raises(V.ClaudeCodeVisionUnavailable):
        V.describe_image("data:image/png;base64,", "q")


def test_the_answer_is_returned_stripped(spy):
    spy["result"] = _Result(out="  a login page\n\n")
    assert V.describe_image(DATA_URL, "q") == "a login page"
