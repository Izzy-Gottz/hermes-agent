"""A cron turn whose final answer is tool calls written out as text is a failed run, never a
delivered reply (Memoe, 2026-09-30: the hermes-tools MCP server timed out, and a marketing job's
"reply" was `<invoke name="skill_view">…` on the owner's phone)."""
import pytest

from cron import scheduler


class _Agent:
    @staticmethod
    def _format_turn_completion_explanation(*_a):
        return ""


def _final(text):
    return scheduler._final_response_from_result(
        {"final_response": text, "completed": True, "messages": []}, "j1", "job", _Agent)


def test_tool_calls_as_text_fail_the_run():
    for text in ('I\'ll start.\n<invoke name="skill_view">\n<parameter name="name">x</parameter>\n</invoke>',
                 "<function_calls>\n<invoke name=\"terminal\"></invoke>\n</function_calls>"):
        with pytest.raises(RuntimeError, match="tools did not load"):
            _final(text)


def test_an_ordinary_answer_is_unchanged():
    assert _final("Two posts went out today; nothing needs you.") == \
        "Two posts went out today; nothing needs you."
    # Angle brackets that are not tool calls are an answer like any other.
    assert _final("Use <b>bold</b> in the invite.") == "Use <b>bold</b> in the invite."
