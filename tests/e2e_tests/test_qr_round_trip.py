"""
End-to-end tests for QR generation through the MCP integration.

Two tests, covering the same pipeline at two different depths.

test_qr_round_trip_through_the_agent is the full one: a real LLM decides to
call the tool, the tool runs in a real MCP server subprocess, and the PNG it
writes is decoded again and compared against what was asked for. It covers
what Week 2 requires of an end-to-end test -- runtime tool discovery,
invocation, and result handling -- and it needs a token, so it skips
without one.

test_wifi_payload_survives_the_round_trip goes through everything except the
LLM: registry, client, subprocess, file, decode. It needs no token and no
network, so it runs everywhere, and it checks the thing a language model
cannot be relied on to exercise reproducibly -- that special characters in
a Wi-Fi SSID and password come back byte-identical after being escaped,
encoded into a QR symbol, written to a PNG, and decoded by an independent
implementation.

Decoding uses zxing-cpp, which is a different implementation from the
qrcode library that writes the images. That matters: comparing our encoder
against our own encoder would only prove it is self-consistent.

Run from the repository root:
    pytest tests/e2e_tests/test_qr_round_trip.py -v
"""

import os

import pytest

from config import Config, load_config
from context import Context
from llm_client import LLMClient
from loop import react_step
from permissions import auto_approve
from main import SYSTEM_PROMPT
from tool_registry import ToolRegistry

zxingcpp = pytest.importorskip(
    "zxingcpp", reason="zxing-cpp is required to decode the generated QR codes"
)
from PIL import Image  # noqa: E402  (imported after the decoder check on purpose)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ASSET_TAG = "INV-2026-XPS13-DXP1345927"

QR_PROMPT = (
    f"Generate a QR code named asset-tag.png that encodes exactly this "
    f"text and nothing else: {ASSET_TAG}"
)


def _decode(png_path):
    """Decodes a QR code PNG and returns its text, or None if unreadable."""
    result = zxingcpp.read_barcode(Image.open(png_path))
    return result.text if result else None


def _real_config(workspace):
    """Loads the project's config but points the workspace at a temp dir."""
    config_path = os.path.join(REPO_ROOT, "config.json")
    if not os.path.exists(config_path):
        pytest.skip("config.json not found at repo root -- cannot run E2E test")

    real = load_config(config_path)
    return Config(
        model=real.model,
        temperature=real.temperature,
        base_url=real.base_url,
        endpoint=real.endpoint,
        max_iterations=real.max_iterations,
        workspace_root=str(workspace),
        ltm_db_path=real.ltm_db_path,
        api_key=real.api_key,
        mcp_servers=real.mcp_servers,
        permissions=real.permissions,
    )


# ---------------------------------------------------------------------------
# The full path, including the LLM
# ---------------------------------------------------------------------------

def test_qr_round_trip_through_the_agent(tmp_path, shipped_policy):
    """The agent generates a QR code, and the code decodes to what was asked.

    Covers the three things Week 2 asks an end-to-end MCP test to show:
    the QR tools are discovered from a running server rather than hardcoded,
    the model selects and invokes one, and its result is handled well enough
    that the agent can answer from it.
    """
    if not os.environ.get("INNKUBE_TOKEN"):
        pytest.skip("INNKUBE_TOKEN not set -- skipping E2E test")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = _real_config(workspace)

    llm = LLMClient(config)
    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers=config.mcp_servers,
    )

    # --- Discovery: the tools came from the server, not from a literal ---
    tool_names = [s["function"]["name"] for s in tools.schemas()]
    assert "generate_qr_code" in tool_names, (
        f"QR tools were not discovered at runtime. Discovered: {tool_names}"
    )

    context = Context(SYSTEM_PROMPT)
    context.add_user_message(QR_PROMPT)
    answer = react_step(
        llm, context, tools, config.max_iterations,
        policy=shipped_policy, confirm=auto_approve,
    )

    # --- Invocation: a PNG actually exists, in the directory the server
    # chose rather than one the model named ---
    qr_dir = workspace / "qr_codes"
    assert qr_dir.is_dir(), (
        f"qr_codes/ was never created, so the tool was not invoked. "
        f"Agent said: {answer}"
    )

    pngs = sorted(qr_dir.glob("*.png"))
    assert pngs, f"No PNG was written. Agent said: {answer}"

    # --- Round trip: the image decodes back to the requested payload ---
    decoded = [_decode(p) for p in pngs]
    assert ASSET_TAG in decoded, (
        f"No generated QR code decoded to the requested payload.\n"
        f"  requested: {ASSET_TAG!r}\n"
        f"  decoded:   {decoded!r}\n"
        f"  agent said: {answer}"
    )

    # --- Result handling: the tool's observation reached the final answer ---
    assert answer, "Agent produced no final answer"
    assert "asset-tag" in answer.lower() or "qr" in answer.lower(), (
        f"Agent's answer does not reflect the tool result: {answer}"
    )


# ---------------------------------------------------------------------------
# The same pipeline without the LLM -- runs everywhere, no token needed
# ---------------------------------------------------------------------------

def test_wifi_payload_survives_the_round_trip(tmp_path):
    """Special characters in a Wi-Fi SSID and password survive escaping,
    encoding, writing and decoding.

    This is the assertion the escaping rules exist for, made against a
    decoder that knows nothing about how the payload was built. It runs
    through the real registry, the real client, and a real server
    subprocess -- only the model is left out, because a model cannot be
    relied on to reproduce an exact string with backslashes in it.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers={"recall": {"enabled": False}, "ocr": {"enabled": False},
                     "qr": {"enabled": True}},
    )

    ssid = "Cafe;Guest,WiFi"
    password = "p:a\\ss;word"

    result = tools.call("generate_qr_code", {
        "filename": "guest-wifi.png",
        "payload_type": "wifi",
        "ssid": ssid,
        "password": password,
        "auth_type": "WPA",
    })
    assert not result.startswith("ERROR"), result

    png = workspace / "qr_codes" / "guest-wifi.png"
    assert png.is_file(), f"PNG was not written. Tool said: {result}"

    decoded = _decode(png)
    assert decoded is not None, "The generated PNG could not be decoded"

    # Built independently of the server so the expectation is not simply
    # whatever the server happened to produce.
    expected = (
        r"WIFI:T:WPA;S:Cafe\;Guest\,WiFi;P:p\:a\\ss\;word;;"
    )
    assert decoded == expected, (
        f"Wi-Fi payload did not survive the round trip.\n"
        f"  expected: {expected!r}\n"
        f"  decoded:  {decoded!r}"
    )

    # And the secret did not leak into the string the agent would see.
    assert password not in result


def test_generated_code_is_not_merely_our_own_encoder_agreeing_with_itself(tmp_path):
    """Guards the premise of the tests above: the decoder is independent.

    If zxing-cpp were somehow decoding by re-encoding with the same library,
    a deliberately corrupted file would still 'decode'. It must not.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tools = ToolRegistry(
        workspace_root=str(workspace),
        receipt_memory_path=str(tmp_path / "memory.json"),
        mcp_servers={"recall": {"enabled": False}, "ocr": {"enabled": False},
                     "qr": {"enabled": True}},
    )
    tools.call("generate_qr_code", {
        "filename": "plain.png", "payload_type": "text", "text": "hello"})

    png = workspace / "qr_codes" / "plain.png"
    assert _decode(png) == "hello"

    # Replace the image with white noise of the same size; it must fail to
    # decode rather than returning the previous payload.
    corrupted = workspace / "qr_codes" / "corrupted.png"
    Image.new("RGB", Image.open(png).size, "white").save(corrupted)
    assert _decode(corrupted) is None
