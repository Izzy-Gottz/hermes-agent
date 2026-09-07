"""Describe an image using the user's Claude Code subscription, via the CLI.

WHY THIS EXISTS
---------------
A Claude Code subscription is an OAuth login to a *CLI*, not an API key. The
auxiliary client is built on HTTP clients, so `claude-code-cli` cannot serve
any auxiliary lane — `auxiliary_client`'s own header says so, and
configure-hermes repeats it. The consequence measured on a real Mac
(2026-09-07): every `computer_use(action='capture')` was routed to a
third-party auxiliary vision provider, that provider's balance was empty, and
the screenshot came back as

    402 - {'error': {'message': 'Insufficient Balance'}}

So the assistant could not see the screen at all, and the error named a
payment problem with a service the owner had not chosen and did not know was
in the path.

Reaching for the OAuth token to make an HTTP call is NOT the fix, and this
module deliberately does not do it. Hermes reads those credentials only behind
`HERMES_CLAUDE_CODE_CREDENTIALS`, which hosts set to `0`, because refreshing
that single-use token rotates it and logs every Claude Code session on the
machine out of its keychain pair — observed 2026-09-02.

Instead we spawn the CLI the same way the main conversation already does. No
token is read, nothing is refreshed, and the subscription serves the call.
Measured on the owner's Mac against a 5.8 MB screenshot:

    echo "Read the image at <path> and describe it." \\
        | claude -p --model haiku --allowedTools Read

    -> "A long-form Wikipedia article about disability and accessibility with
        multiple sections, images, and icons displayed in a vertical
        scrollable format."   (5.9 s)

Two details that are load-bearing, both learned by getting them wrong:

* The prompt must arrive on **stdin**. `claude -p "<prompt>"` with other flags
  returned `Input must be provided either through stdin or as a prompt
  argument when using --print`.
* The image must be a **path the CLI reads with its Read tool**, not inline
  data. Piping a prompt that merely mentions an image gets "I don't see an
  image in your message."
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

#: Small and cheap: this lane describes a screenshot, it does not reason about
#: one. The caller may override via ``auxiliary.vision.model``.
DEFAULT_MODEL = "haiku"

#: The CLI reads the file itself, so the only tool it needs is Read. Naming it
#: explicitly keeps this call from inheriting a permissive tool surface.
_ALLOWED_TOOLS = "Read"

_DATA_URL_RE = re.compile(r"^data:image/(?P<ext>[a-zA-Z0-9.+-]+);base64,(?P<b64>.+)$", re.S)

_EXT_FOR_SUBTYPE = {
    "jpeg": ".jpg", "jpg": ".jpg", "png": ".png",
    "gif": ".gif", "webp": ".webp",
}


class ClaudeCodeVisionUnavailable(RuntimeError):
    """The CLI is not usable for this call. Callers should fall through."""


def cli_available(command: str = "claude") -> bool:
    """True when the Claude Code CLI is on PATH."""
    return shutil.which(command) is not None


def _decode_to_file(image: str, directory: str) -> str:
    """Write *image* into *directory* and return the path.

    Accepts a local path (used as-is) or a ``data:image/...;base64,`` URL.
    """
    match = _DATA_URL_RE.match(image.strip())
    if match is None:
        if os.path.isfile(image):
            return image
        raise ClaudeCodeVisionUnavailable(
            "image must be a local file path or a data:image/...;base64 URL")
    subtype = match.group("ext").lower()
    suffix = _EXT_FOR_SUBTYPE.get(subtype, ".png")
    try:
        blob = base64.b64decode(match.group("b64"), validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ClaudeCodeVisionUnavailable("image data URL did not decode: %s" % exc)
    if not blob:
        raise ClaudeCodeVisionUnavailable("image data URL decoded to nothing")
    fd, path = tempfile.mkstemp(suffix=suffix, dir=directory)
    with os.fdopen(fd, "wb") as handle:
        handle.write(blob)
    return path


def describe_image(
    image: str,
    question: str,
    *,
    model: Optional[str] = None,
    timeout: float = 120.0,
    command: str = "claude",
    env: Optional[dict] = None,
) -> str:
    """Answer *question* about *image* using the Claude Code CLI.

    *image* is a local path or a base64 data URL. Raises
    ``ClaudeCodeVisionUnavailable`` when the CLI cannot serve the call, so the
    caller can fall through to the ordinary auxiliary chain rather than
    surfacing a hard failure.
    """
    if not cli_available(command):
        raise ClaudeCodeVisionUnavailable(
            "the `%s` CLI is not on PATH" % command)
    asked = (question or "").strip() or "Describe this image."

    with tempfile.TemporaryDirectory(prefix="hermes-cc-vision-") as workdir:
        path = _decode_to_file(image, workdir)
        # The path goes in the PROMPT because the CLI's Read tool is what
        # opens it. Quoted so a path with spaces survives — Moe's own home is
        # "~/Library/Application Support/Moe", which has one.
        prompt = (
            'Read the image at "%s" and answer this about it. '
            "Answer directly, with no preamble and no mention of having read "
            "a file.\n\n%s" % (path, asked)
        )
        argv = [
            command, "-p",
            "--model", model or DEFAULT_MODEL,
            "--allowedTools", _ALLOWED_TOOLS,
        ]
        try:
            proc = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env if env is not None else os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            raise ClaudeCodeVisionUnavailable(
                "the %s CLI did not answer within %.0fs" % (command, timeout))
        except OSError as exc:
            raise ClaudeCodeVisionUnavailable(
                "could not run the %s CLI: %s" % (command, exc))

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:300]
        raise ClaudeCodeVisionUnavailable(
            "the %s CLI exited %d: %s" % (command, proc.returncode, detail))
    answer = (proc.stdout or "").strip()
    if not answer:
        raise ClaudeCodeVisionUnavailable(
            "the %s CLI returned nothing" % command)
    return answer


__all__ = [
    "ClaudeCodeVisionUnavailable",
    "DEFAULT_MODEL",
    "cli_available",
    "describe_image",
]
