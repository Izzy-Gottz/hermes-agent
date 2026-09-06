"""Opening an inbound frame the connector sealed to this gateway's device.

**Why a gateway would ever do this.** Moe's connector (the "edge") is a hosted
service that receives a person's Telegram messages and hands them to whichever
of their runtimes is awake. The plaintext of those messages is not the edge's
to keep: it seals each one to the *device* it is about to hand it to, using
that device's X25519 public key, and holds no key that opens it. So the frame
arrives with a ``sealed`` block beside ``text``, and this module is what turns
that block back into words.

**Nothing happens here unless an operator asks for it.** Both paths below are
unset by default, and with them unset a ``sealed`` block is ignored and
``text`` is used — which is exactly what every gateway did before this file
existed. No upstream deployment changes behaviour.

  ``GATEWAY_RELAY_INBOUND_OPENER``  the file holding the opening rule
  ``GATEWAY_RELAY_INBOUND_KEYS``    this device's identity JSON

**The rule is loaded from a file rather than written here, and that is the
whole point.** The derivation (X25519 → HKDF-SHA256 → AES-GCM, with the
channel, message id and recipient device bound into ``info``) already exists in
three implementations that must agree byte for byte: the edge that seals,
Swift's ``InboundSeal``, and the Python opener that ships in the connector's
own payload. A fourth copy living here would agree with them until one of the
four was edited, and the failure would be an ``InvalidTag`` on somebody's real
message with nothing to say which side drifted. So the operator points this at
*their* copy of the rule and there is no second copy of the crypto.

The loaded file must expose ``open_sealed(sealed, *, ephemeral_public, channel,
message_id, to_device, device_id, agreement_private, agreement_public,
scheme)`` returning ``str``, and raise on anything it will not open. It is
loaded with no package around it, so it must not import its own siblings.

**What a message opening does and does not prove.** It proves the blob was
sealed by someone holding this device's public key, to this exact ``(channel,
message id, device)`` triple, and has not been altered since. The recipient's
public key is by design known to the connector, so this is confidentiality and
integrity — it is **not** proof that the connector sealed it and **not** proof
that a person sent it. Do not build authorization on a frame having opened.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
from importlib import util as importlib_util
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

#: The env var naming the file that holds the opening rule.
OPENER_ENV = "GATEWAY_RELAY_INBOUND_OPENER"
#: The env var naming this device's identity JSON: ``device``,
#: ``agreement_private`` and ``agreement_public``, each base64.
KEYS_ENV = "GATEWAY_RELAY_INBOUND_KEYS"


class SealError(Exception):
    """A frame carried a seal and this gateway could not turn it into words.

    Raised only where the caller has no plaintext to fall back on. While a
    connector sends ``text`` alongside ``sealed`` — the ordinary shape of a
    two-sided rollout — a failure here is logged and ``text`` is used instead;
    when ``text`` goes away, the same failure becomes this exception and the
    frame is not delivered. **That is the correct failure**: a message that
    cannot be opened must not become a message that was silently downgraded.
    """


# The loaded rule module and the mtime/size of the file it came from. Reloaded
# when that file changes, because the connector's payload is upgraded
# underneath a long-running gateway and a stale rule would fail every message
# with a tag error nobody could explain.
_rule_lock = threading.Lock()
_keys_lock = threading.Lock()
_rule: Optional[Tuple[Any, Tuple[str, float, int]]] = None
_keys: Optional[Tuple[Dict[str, Any], Tuple[str, float, int]]] = None

#: Set once a sealed frame has arrived at a gateway with no opener configured,
#: so that says itself once rather than once per message.
_warned_unconfigured = False


def _stamp(path: str) -> Tuple[str, float, int]:
    st = os.stat(path)
    return (path, st.st_mtime, st.st_size)


def _load_rule(path: str) -> Any:
    """Exec the opener file as a standalone module.

    ``sys.dont_write_bytecode`` is set across the exec deliberately: the file
    belongs to somebody else's tree — the connector's installed payload — and a
    ``__pycache__`` appearing in it is our droppings in their directory, which
    on at least one machine has also raced a concurrent upgrade of that very
    file.
    """
    spec = importlib_util.spec_from_file_location("_relay_inbound_rule", path)
    if spec is None or spec.loader is None:
        raise SealError("%s is not a Python file this gateway can load" % path)
    module = importlib_util.module_from_spec(spec)
    was = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    finally:
        sys.dont_write_bytecode = was
    if not callable(getattr(module, "open_sealed", None)):
        raise SealError("%s has no open_sealed(...) to call" % path)
    return module


def rule() -> Optional[Any]:
    """The loaded opening rule, or None when no operator configured one."""
    global _rule
    path = os.environ.get(OPENER_ENV, "").strip()
    if not path:
        return None
    with _rule_lock:
        if _rule is not None and _rule[1][0] == path:
            try:
                if _stamp(path) == _rule[1]:
                    return _rule[0]
            except OSError:
                # The file went away under us. Keep serving the rule already in
                # memory rather than failing every message: an upgrade that
                # replaces the payload is momentarily exactly this.
                return _rule[0]
        module = _load_rule(path)
        _rule = (module, _stamp(path))
        return module


def keys() -> Optional[Dict[str, Any]]:
    """This device's id and X25519 key material, or None when unconfigured.

    Returns the private half as raw bytes rather than an object so this module
    holds no opinion about the key type: the rule builds what it needs. Both
    halves are returned because the **public** one is HKDF salt — re-deriving
    it here would be a second place for it to be got wrong.
    """
    global _keys
    path = os.environ.get(KEYS_ENV, "").strip()
    if not path:
        return None
    with _keys_lock:
        if _keys is not None and _keys[1][0] == path:
            try:
                if _stamp(path) == _keys[1]:
                    return _keys[0]
            except OSError:
                return _keys[0]
        with open(path, "rb") as fh:
            obj = json.loads(fh.read().decode("utf-8"))
        if not isinstance(obj, dict):
            raise SealError("%s does not hold a device identity" % path)
        try:
            material = {
                "device_id": str(obj["device"]),
                "agreement_private": base64.b64decode(obj["agreement_private"], validate=True),
                "agreement_public": base64.b64decode(obj["agreement_public"], validate=True),
            }
        except (KeyError, ValueError, TypeError) as exc:
            raise SealError("%s is not a readable device identity: %s"
                            % (path, type(exc).__name__)) from exc
        if len(material["agreement_private"]) != 32 or len(material["agreement_public"]) != 32:
            raise SealError("%s holds an X25519 key that is not 32 bytes" % path)
        _keys = (material, _stamp(path))
        return material


def configured() -> bool:
    """Whether an operator asked this gateway to open seals at all."""
    return bool(os.environ.get(OPENER_ENV, "").strip()
                and os.environ.get(KEYS_ENV, "").strip())


def _open(sealed: Dict[str, Any]) -> str:
    """The words inside one ``sealed`` block, or raise ``SealError``."""
    module = rule()
    material = keys()
    if module is None or material is None:
        raise SealError("this gateway has no inbound opener configured (%s, %s)"
                        % (OPENER_ENV, KEYS_ENV))
    try:
        blob = base64.b64decode(sealed["ciphertext"], validate=True)
        ephemeral = base64.b64decode(sealed["ephemeralPublic"], validate=True)
        channel = str(sealed["channel"])
        message_id = str(sealed["messageId"])
        to_device = str(sealed["toDevice"])
    except (KeyError, ValueError, TypeError) as exc:
        raise SealError("that seal is not the shape this gateway reads (%s)"
                        % type(exc).__name__) from exc

    from cryptography.hazmat.primitives.asymmetric import x25519

    private = x25519.X25519PrivateKey.from_private_bytes(material["agreement_private"])
    kwargs: Dict[str, Any] = dict(
        ephemeral_public=ephemeral, channel=channel, message_id=message_id,
        to_device=to_device, device_id=material["device_id"],
        agreement_private=private, agreement_public=material["agreement_public"])
    scheme = sealed.get("scheme")
    if scheme is not None:
        # Passed through rather than defaulted. The scheme names the whole
        # derivation, so quietly substituting the one we happen to implement
        # for a missing or unknown one is how a future scheme gets opened as
        # this one and fails with a tag error that names no suspect.
        kwargs["scheme"] = str(scheme)
    try:
        text = module.open_sealed(blob, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the rule is someone else's file
        raise SealError("%s: %s" % (type(exc).__name__, exc)) from exc
    if not isinstance(text, str):
        raise SealError("the opener returned %s, not text" % type(text).__name__)
    return text


def resolve_text(raw: Dict[str, Any]) -> str:
    """The text of one inbound event — the seal's when there is one to open.

    The four cases, and each is a decision rather than an accident:

    * **no ``sealed``** — ``text``, unchanged. Every gateway before this file.
    * **``sealed``, opened** — the opened words, and ``text`` is ignored. Once
      this gateway can open a seal, the seal is what it believes; preferring
      ``text`` would make the whole path decorative.
    * **``sealed``, would not open, ``text`` present** — a warning naming the
      reason, then ``text``. This is the rollout window and it is deliberately
      loud: every one of these becomes a dropped message the day ``text``
      leaves the frame, so they must be visible *before* that day, not after.
    * **``sealed``, would not open, no ``text``** — ``SealError``. There is
      nothing honest left to deliver.
    """
    global _warned_unconfigured
    sealed = raw.get("sealed")
    has_text = isinstance(raw.get("text"), str)
    text = raw.get("text", "") if has_text else ""
    if not isinstance(sealed, dict) or not sealed:
        if sealed is not None and not has_text:
            raise SealError("that frame carries neither a readable seal nor text")
        return text

    if not configured():
        if not has_text:
            raise SealError(
                "that frame is sealed and this gateway has no opener configured "
                "(%s, %s)" % (OPENER_ENV, KEYS_ENV))
        if not _warned_unconfigured:
            _warned_unconfigured = True
            logger.warning(
                "relay inbound: frames arrive sealed but no opener is configured "
                "(%s, %s); falling back to plaintext. This gateway will stop "
                "receiving messages when the connector drops the plaintext.",
                OPENER_ENV, KEYS_ENV)
        return text

    try:
        opened = _open(sealed)
    except SealError as exc:
        if not has_text:
            raise
        # No content in the log line, from either side: the failure says which
        # frame and why, never what anybody wrote.
        logger.warning("relay inbound: a sealed frame would not open (%s); "
                       "using the plaintext this connector still sends", exc)
        return text
    if has_text and text != opened:
        # A canary for the rollout window and nothing more. Lengths only —
        # printing either version would put the person's words in a log on the
        # one path where we have just discovered we do not know which is real.
        logger.warning("relay inbound: the seal and the plaintext disagree "
                       "(%d vs %d characters); taking the seal",
                       len(opened), len(text))
    return opened
