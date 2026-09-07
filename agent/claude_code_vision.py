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




# ── the call_llm seam ─────────────────────────────────────────────────────
#
# Everything vision funnels through `call_llm(task="vision", messages=[...])`,
# and every caller reads the answer as `response.choices[0].message.content`
# (see auxiliary_client.extract_content_or_reasoning). So the lane plugs in by
# returning that shape, and nothing downstream needs to know a CLI served it.


class _Message:
    __slots__ = ("content", "role", "reasoning", "reasoning_content",
                 "tool_calls", "refusal")

    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        # extract_content_or_reasoning reads these when content is empty.
        # Present and None, so a getattr() probe behaves like a real message
        # rather than raising.
        self.reasoning = None
        self.reasoning_content = None
        self.tool_calls = None
        self.refusal = None


class _Choice:
    __slots__ = ("message", "finish_reason", "index")

    def __init__(self, content: str):
        self.message = _Message(content)
        self.finish_reason = "stop"
        self.index = 0


class CLIVisionResponse:
    """An OpenAI-shaped response carrying one CLI answer.

    Usage is reported as zero rather than guessed: the subscription is not
    metered per token here, and inventing numbers would put fiction into cost
    accounting. A caller that sums usage sees zero, which is true.
    """

    __slots__ = ("choices", "model", "usage", "id", "object", "created")

    def __init__(self, content: str, model: str):
        self.choices = [_Choice(content)]
        self.model = model
        self.usage = None
        self.id = "claude-code-cli-vision"
        self.object = "chat.completion"
        self.created = 0


def image_and_question_from_messages(messages) -> "tuple[Optional[str], str]":
    """Pull the image and the asked question out of OpenAI-style messages.

    Returns ``(image, question)``; ``image`` is None when the messages carry
    none, which is the signal that this lane does not apply.
    """
    image: Optional[str] = None
    texts: list = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if message.get("role") != "system":
                texts.append(content)
            continue
        for part in content or []:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "text" and message.get("role") != "system":
                texts.append(str(part.get("text") or ""))
            elif kind == "image_url":
                url = (part.get("image_url") or {})
                if isinstance(url, dict):
                    url = url.get("url")
                # LAST image wins: the caller may retry a downscaled copy in
                # the same message list, and the newest is the one meant.
                if url:
                    image = str(url)
    question = "\n\n".join(t.strip() for t in texts if t and t.strip())
    return image, question


def serves_vision_for(provider: str) -> bool:
    """True when this lane should be tried for *provider*.

    Only for the CLI-backed brain, because that is the one the auxiliary
    client cannot serve at all. A provider with an HTTP client has a working
    chain already and must keep using it.
    """
    return str(provider or "").strip().lower() in {
        "claude-code-cli", "claude-code", "claude_code",
    }


def try_vision_call(
    provider: str,
    messages,
    *,
    model: Optional[str] = None,
    timeout: float = 120.0,
) -> Optional[CLIVisionResponse]:
    """Serve a vision call from the CLI, or return None to fall through.

    None — never an exception — for every "not applicable" and every failure,
    so a caller wraps this in nothing and loses no existing behaviour.
    """
    if not serves_vision_for(provider):
        return None
    try:
        image, question = image_and_question_from_messages(messages)
    except Exception:
        return None
    if not image:
        return None
    try:
        answer = describe_image(
            image, question, model=model, timeout=timeout)
    except ClaudeCodeVisionUnavailable as exc:
        logger.info("claude-code CLI vision unavailable, falling through: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("claude-code CLI vision raised unexpectedly: %s", exc)
        return None
    return CLIVisionResponse(answer, model or DEFAULT_MODEL)




# ── the text lanes ────────────────────────────────────────────────────────
#
# Vision was the loudest failure but not the only one. The same empty balance
# took compression, session_search and title_generation down with it, and on
# the owner's Mac the gateway logged, every turn:
#
#     Auxiliary title_generation: payment error on deepseek
#     credential pool: marking DEEPSEEK_API_KEY exhausted (status=402)
#
# Those lanes are text-only, so they need none of the image handling above —
# just the CLI, a prompt, and an answer. Same subscription, same reasoning:
# a brain that is a CLI login can serve its own side tasks.

#: The auxiliary tasks this lane will serve. Deliberately a small, named set
#: rather than "anything not vision": a task added upstream later should have
#: to be considered here, not silently inherit a lane nobody chose for it.
TEXT_TASKS = frozenset({
    "compression", "session_search", "title_generation", "reflection",
})


def prompt_from_messages(messages) -> str:
    """Flatten OpenAI-style messages into one prompt for the CLI.

    Roles are labelled rather than dropped: a compression task's system
    message carries the instruction and the user message carries the text, and
    concatenating them unlabelled turns two things into one ambiguous blob.
    """
    parts = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip().lower()
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        text = str(content or "").strip()
        if not text:
            continue
        if role == "system":
            parts.append("[instructions]\n" + text)
        else:
            parts.append(text)
    return "\n\n".join(parts)


def answer_text(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout: float = 120.0,
    command: str = "claude",
    env: Optional[dict] = None,
) -> str:
    """Answer a text-only prompt through the CLI."""
    if not cli_available(command):
        raise ClaudeCodeVisionUnavailable("the `%s` CLI is not on PATH" % command)
    body = (prompt or "").strip()
    if not body:
        raise ClaudeCodeVisionUnavailable("nothing to ask")
    # The CLI is a conversational assistant and answers like one. Measured, a
    # compression request came back as
    #
    #     "DeepSeek's depleted balance broke the screenshot feature...
    #
    #      What would you like me to help with?"
    #
    # That trailing offer would land inside a stored summary or a conversation
    # title. These lanes want the artefact and nothing else, so say so.
    body = (
        "You are performing an automated side task. Output ONLY the requested "
        "result: no preamble, no explanation of what you did, and no closing "
        "question or offer of further help.\n\n" + body
    )
    # No --allowedTools at all: these lanes summarise text that was handed to
    # them. A side task must not be able to read files or run anything.
    argv = [command, "-p", "--model", model or DEFAULT_MODEL]
    try:
        proc = subprocess.run(
            argv, input=body, capture_output=True, text=True, timeout=timeout,
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
        raise ClaudeCodeVisionUnavailable("the %s CLI returned nothing" % command)
    return answer


def try_text_call(
    provider: str,
    task: str,
    messages,
    *,
    model: Optional[str] = None,
    timeout: float = 120.0,
) -> Optional[CLIVisionResponse]:
    """Serve a text side task from the CLI, or return None to fall through."""
    if not serves_vision_for(provider):
        return None
    if str(task or "").strip().lower() not in TEXT_TASKS:
        return None
    try:
        prompt = prompt_from_messages(messages)
    except Exception:
        return None
    if not prompt:
        return None
    try:
        answer = answer_text(prompt, model=model, timeout=timeout)
    except ClaudeCodeVisionUnavailable as exc:
        logger.info("claude-code CLI text lane unavailable: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("claude-code CLI text lane raised: %s", exc)
        return None
    return CLIVisionResponse(answer, model or DEFAULT_MODEL)


__all__ = [
    "CLIVisionResponse",
    "ClaudeCodeVisionUnavailable",
    "DEFAULT_MODEL",
    "cli_available",
    "describe_image",
    "image_and_question_from_messages",
    "serves_vision_for",
    "TEXT_TASKS",
    "answer_text",
    "prompt_from_messages",
    "try_text_call",
    "try_vision_call",
]
