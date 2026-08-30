"""Images across the hermes-tools MCP bridge.

A Hermes multimodal tool result (computer_use capture, vision tools) must
reach the CLI as an MCP ``image`` content block — not as a JSON string with
a base64 blob or a file path the model cannot open — and must be no larger
than 1568 px on its longest side.
"""

from __future__ import annotations

import base64
import io
import json

import pytest

from agent.transports import hermes_tools_mcp_server as srv


def _png_b64(w: int, h: int) -> str:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 90)).save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


def _multimodal(b64: str, mime: str = "image/png", text: str = "capture 1x1") -> dict:
    return {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
        "text_summary": text,
    }


def test_plain_string_results_pass_through():
    assert srv.to_mcp_content("hello") == "hello"
    assert json.loads(srv.to_mcp_content({"ok": True})) == {"ok": True}


def test_multimodal_becomes_text_plus_image_block():
    from mcp.types import ImageContent, TextContent

    blocks = srv.to_mcp_content(_multimodal(_png_b64(64, 48)))
    assert [type(b) for b in blocks] == [TextContent, ImageContent]
    assert blocks[0].text == "capture 1x1"
    assert blocks[1].type == "image" and blocks[1].mime_type == "image/png"
    assert base64.b64decode(blocks[1].data)[:8] == b"\x89PNG\r\n\x1a\n"


def test_oversized_image_is_downscaled_with_a_scale_note():
    from PIL import Image

    blocks = srv.to_mcp_content(_multimodal(_png_b64(2940, 1912)))
    im = Image.open(io.BytesIO(base64.b64decode(blocks[1].data)))
    assert im.size == (1568, 1020)
    note = blocks[-1].text
    assert "downscaled by 0.533" in note and "multiply pixel coordinates by 1.875" in note


def test_image_at_the_cap_is_untouched():
    b64 = _png_b64(1568, 700)
    blocks = srv.to_mcp_content(_multimodal(b64))
    assert len(blocks) == 2 and blocks[1].data == b64


def test_anthropic_style_source_block_is_accepted():
    from mcp.types import ImageContent

    b64 = _png_b64(8, 8)
    result = {
        "_multimodal": True,
        "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}],
        "text_summary": "shot",
    }
    blocks = srv.to_mcp_content(result)
    assert blocks[0].text == "shot" and isinstance(blocks[1], ImageContent)


def test_unreadable_image_data_passes_through_unchanged():
    blocks = srv.to_mcp_content(_multimodal("bm90IGEgcG5n", mime="image/jpeg"))
    assert blocks[1].mime_type == "image/jpeg" and blocks[1].data == "bm90IGEgcG5n"


def test_claude_code_profile_exposes_computer_use():
    assert "computer_use" in srv.exposed_tools_for_profile("claude-code")
    assert "computer_use" not in srv.exposed_tools_for_profile(None)


def test_registered_tools_do_not_validate_structured_output(monkeypatch):
    """The SDK must not wrap results in a ``{"result": str}`` output model,
    or a content-block list would be rejected (pydantic string_type)."""
    pytest.importorskip("mcp")
    captured = {}

    class FakeServer:
        def __init__(self, *a, **k):
            pass

        def add_tool(self, fn, name=None, description=None, **kw):
            captured[name] = kw

    monkeypatch.setattr("mcp.server.MCPServer", FakeServer)
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda quiet_mode=True: [{"type": "function", "function": {"name": "computer_use", "parameters": {}}}],
    )
    srv._build_server("claude-code")
    assert captured["computer_use"].get("structured_output") is False


def test_aux_vision_routing_is_off_under_claude_code_profile(monkeypatch):
    from tools.computer_use import tool

    monkeypatch.setenv("HERMES_MCP_TOOL_PROFILE", "claude-code")
    assert tool._should_route_through_aux_vision() is False
