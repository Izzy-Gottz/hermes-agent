"""Regression test for approval prompt credential redaction (issue #48456).

When a flagged command contains a credential-shaped value, the gateway approval
prompt must redact the credential from the command text before sending it to
the chat platform. Without this fix, the raw command (with the credential in
plaintext) is sent verbatim to Telegram/Discord/etc.

The redaction is wired through the module-level ``_redact_approval_command``
seam. These tests bind that seam -- the production wiring -- not just the
underlying ``redact_sensitive_text`` helper, so they fail if the redaction
call is removed from either approval path.

Credential fixtures are built at runtime from a benign prefix + a run of
``X`` characters (the same trick tests/agent/test_redact.py uses): they match
the redactor regexes so the assertions stay meaningful, but contain no real
or real-looking key, so secret scanners do not flag this file.
"""

from gateway.run import _redact_approval_command

# Synthetic, scanner-safe credential fixtures. Each matches its redactor
# regex (ghp_/sk-/JWT) but is unmistakably fake -- a run of X's, never a
# real or real-format key.
_FAKE_GHP = "ghp_" + "X" * 36
_FAKE_OPENAI = "sk-proj-" + "X" * 40
_FAKE_JWT = "eyJ" + "X" * 20 + "." + "eyJ" + "X" * 24 + "." + "X" * 30


class TestRedactApprovalCommand:
    """Contract for the approval-prompt redaction seam used by the gateway."""

    def test_redacts_github_pat(self):
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com/user"
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out
        # command structure preserved so the operator can still judge the action
        assert "curl" in out
        assert "github.com" in out

    def test_redacts_openai_key(self):
        raw = "export OPENAI_API_KEY=" + _FAKE_OPENAI + " && python s.py"
        out = _redact_approval_command(raw)
        assert _FAKE_OPENAI not in out
        assert "python s.py" in out

    def test_redacts_bearer_token(self):
        raw = "curl -H 'Authorization: Bearer " + _FAKE_JWT + "' https://api.example.com"
        out = _redact_approval_command(raw)
        assert _FAKE_JWT not in out


    def test_forces_redaction_even_when_disabled(self, monkeypatch):
        """force=True must redact even if security.redact_secrets is off -- the
        approval prompt is a hard secret-egress boundary regardless of config."""
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com"
        # With redaction globally disabled, the seam must STILL redact (force=True).
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out


class TestApprovalTextFallbackContract:
    def test_smart_deny_only_advertises_one_operation(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback(
            "rm -rf /", "dangerous deletion", "/",
            allow_permanent=False, smart_denied=True,
        )
        assert "`/approve`" in text
        assert "approve session" not in text
        assert "approve always" not in text

    def test_text_fallback_says_silence_means_no(self, monkeypatch):
        """Surfaces without buttons get the same deadline line as the button card."""
        from gateway.run import _format_exec_approval_fallback

        monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
        text = _format_exec_approval_fallback("rm -rf /", "recursive delete", "/")
        assert "recursive delete" in text
        assert "5 minutes" in text
        for step in ("`/approve`", "`/approve session`", "`/approve always`", "`/deny`"):
            assert step in text



# ── a plugin's approval: the whole question, under a heading that fits it ─────────────────────────

_LONG_SEND = ("Send this Gmail message?\n\nTO (checked): dad@example.com\n\nMESSAGE (as written, not checked):\n"
              + ("Every word of this has to be in front of the person who approves it. " * 80)
              + "\nTHE LAST LINE.")


def test_a_plugin_approval_carries_every_word_of_its_description():
    """The relay has no buttons, so this text IS the prompt on Telegram. Moe's send gate puts the
    recipient and the whole message in the description; cutting it would ask a person to approve
    words they were never shown."""
    from gateway.run import _format_exec_approval_fallback

    msg = _format_exec_approval_fallback("<gmail_send> (plugin approval rule)", _LONG_SEND, "/")
    assert _LONG_SEND in msg
    assert msg.index("THE LAST LINE.") > msg.index("dad@example.com")
    assert "Dangerous command" not in msg
    assert "/approve" in msg and "/deny" in msg
    assert "(plugin approval rule)" not in msg


def test_a_shell_command_keeps_its_own_prompt():
    from gateway.run import _format_exec_approval_fallback

    msg = _format_exec_approval_fallback("rm -rf /tmp/x", "recursive delete", "/")
    from gateway.platforms.base_exec_approval import EA_HEADER_TEXT
    assert msg.startswith(f"⚠️ **{EA_HEADER_TEXT}**")
    assert "`/approve always`" in msg
