"""
Unit tests for the QR MCP server (mcp_servers/qr/qr_server.py).

Two things are being checked here, and they are quite different in
character.

The first is the server's contract: every tool returns a string and never
raises, because an exception inside a stdio subprocess reaches the agent as
an opaque transport failure rather than as something it can read and
correct.

The second is containment. The tool takes a bare filename and joins it to
one fixed directory, so the tests push a range of path-shaped filenames at
it -- traversal, absolute paths, both separators, encoded forms -- and
assert that nothing is written outside that directory. These are the tests
that matter if the argument ever reaches the server from a model rather
than from a person.

The output directory is redirected to a tmp_path for every test, so no test
writes into the real workspace.
"""

import importlib

import pytest

import mcp_servers.qr.qr_server as qr_server


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Points the server's fixed output directory at a temp directory.

    OUTPUT_DIR is read at import time, so it is patched on the already
    imported module rather than through the environment.
    """
    output_dir = tmp_path / "qr_codes"
    monkeypatch.setattr(qr_server, "OUTPUT_DIR", output_dir)
    return qr_server


@pytest.fixture
def output_dir(server):
    return server.OUTPUT_DIR


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_generates_a_png_for_plain_text(server, output_dir):
    result = server.generate_qr_code("note.png", payload_type="text", text="hello")
    assert not result.startswith("ERROR")
    assert (output_dir / "note.png").is_file()


def test_written_file_is_actually_a_png(server, output_dir):
    """Checks the PNG magic number rather than trusting the extension."""
    server.generate_qr_code("note.png", payload_type="text", text="hello")
    header = (output_dir / "note.png").read_bytes()[:8]
    assert header == b"\x89PNG\r\n\x1a\n"


def test_extension_is_added_when_omitted(server, output_dir):
    result = server.generate_qr_code("note", payload_type="text", text="hello")
    assert not result.startswith("ERROR")
    assert (output_dir / "note.png").is_file()


def test_output_directory_is_created_on_demand(server, output_dir):
    assert not output_dir.exists()
    server.generate_qr_code("note.png", payload_type="text", text="hi")
    assert output_dir.is_dir()


def test_confirmation_names_the_file(server):
    result = server.generate_qr_code("label.png", payload_type="text", text="hi")
    assert "label.png" in result


@pytest.mark.parametrize("payload_type,fields", [
    ("text", {"text": "plain text"}),
    ("url", {"url": "https://example.com"}),
    ("wifi", {"ssid": "MyNet", "password": "hunter2"}),
    ("vcard", {"first_name": "Ada", "last_name": "Lovelace"}),
])
def test_every_payload_type_produces_a_file(server, output_dir, payload_type, fields):
    result = server.generate_qr_code("out.png", payload_type=payload_type, **fields)
    assert not result.startswith("ERROR"), result
    assert (output_dir / "out.png").is_file()


# ---------------------------------------------------------------------------
# Containment -- the filename must not be able to express a location
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("attempt", [
    "../escape.png",
    "../../escape.png",
    "../../../../../../tmp/escape.png",
    "subdir/escape.png",
    "/etc/escape.png",
    "/tmp/escape.png",
    "..\\escape.png",
    "subdir\\escape.png",
    "C:\\Windows\\escape.png",
    "....//escape.png",
    "foo/../../escape.png",
])
def test_rejects_anything_path_shaped(server, attempt):
    result = server.generate_qr_code(attempt, payload_type="text", text="hi")
    assert result.startswith("ERROR"), f"{attempt!r} was not rejected"


def test_rejection_writes_nothing_anywhere(server, output_dir, tmp_path):
    """A rejected filename must not leave a file behind, in the output
    directory or beside it."""
    server.generate_qr_code("../escape.png", payload_type="text", text="hi")
    assert not (tmp_path / "escape.png").exists()
    assert not output_dir.exists() or list(output_dir.iterdir()) == []


def test_traversal_is_rejected_before_the_directory_is_created(server, output_dir):
    """Validation happens before any filesystem work, so a bad filename
    does not even create the output directory as a side effect."""
    server.generate_qr_code("../escape.png", payload_type="text", text="hi")
    assert not output_dir.exists()


@pytest.mark.parametrize("attempt", [
    "",
    "   ",
    ".",
    "..",
    ".hidden.png",
    "x" * 200 + ".png",
])
def test_rejects_malformed_filenames(server, attempt):
    result = server.generate_qr_code(attempt, payload_type="text", text="hi")
    assert result.startswith("ERROR"), f"{attempt!r} was not rejected"


@pytest.mark.parametrize("attempt", [
    "code;rm -rf.png",
    "code$(whoami).png",
    "code\x00.png",
    "code\n.png",
    "code|pipe.png",
    "café.png",
])
def test_rejects_characters_outside_the_allowlist(server, attempt):
    """The allowlist is letters, digits, spaces, dots, hyphens and
    underscores. Anything else is refused -- including accented characters,
    which are harmless but not worth the encoding questions."""
    result = server.generate_qr_code(attempt, payload_type="text", text="hi")
    assert result.startswith("ERROR"), f"{attempt!r} was not rejected"


def test_rejects_a_non_png_extension(server):
    result = server.generate_qr_code("note.txt", payload_type="text", text="hi")
    assert result.startswith("ERROR")
    assert ".png" in result


@pytest.mark.parametrize("accepted", [
    "simple.png",
    "with-hyphen.png",
    "with_underscore.png",
    "with space.png",
    "MixedCase.png",
    "digits123.png",
    "no-extension",
])
def test_accepts_ordinary_filenames(server, output_dir, accepted):
    result = server.generate_qr_code(accepted, payload_type="text", text="hi")
    assert not result.startswith("ERROR"), f"{accepted!r} was rejected: {result}"


# ---------------------------------------------------------------------------
# Error handling -- strings, never exceptions
# ---------------------------------------------------------------------------

def test_unknown_payload_type_returns_error_string(server):
    result = server.generate_qr_code("out.png", payload_type="morse", text="hi")
    assert result.startswith("ERROR")
    assert "morse" in result


def test_payload_validation_error_is_returned_as_a_string(server):
    """A ValueError out of the payload builders must be converted, not
    propagated."""
    result = server.generate_qr_code("out.png", payload_type="wifi", ssid="")
    assert result.startswith("ERROR")
    assert "ssid" in result


def test_rejects_dangerous_url_scheme(server):
    result = server.generate_qr_code(
        "out.png", payload_type="url", url="javascript:alert(1)"
    )
    assert result.startswith("ERROR")


def test_bad_error_correction_level_returns_error_string(server):
    result = server.generate_qr_code(
        "out.png", payload_type="text", text="hi", error_correction="Z"
    )
    assert result.startswith("ERROR")


def test_oversized_payload_returns_error_string(server):
    result = server.generate_qr_code(
        "out.png", payload_type="text", text="x" * 5000
    )
    assert result.startswith("ERROR")
    assert "too large" in result


def test_unwritable_output_directory_returns_error_string(server, tmp_path, monkeypatch):
    """An OSError from the filesystem is reported, not raised."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("this is a file, so mkdir beneath it must fail")
    monkeypatch.setattr(server, "OUTPUT_DIR", blocked / "qr_codes")

    result = server.generate_qr_code("out.png", payload_type="text", text="hi")
    assert result.startswith("ERROR")


def test_no_tool_call_ever_raises(server):
    """Sweeps a range of bad inputs and asserts every one returns a string.
    The contract is that the agent always gets something readable back."""
    bad_inputs = [
        {"filename": "../x.png", "payload_type": "text", "text": "hi"},
        {"filename": "", "payload_type": "text", "text": "hi"},
        {"filename": "ok.png", "payload_type": "nonsense"},
        {"filename": "ok.png", "payload_type": "wifi"},
        {"filename": "ok.png", "payload_type": "url", "url": "not a url"},
        {"filename": "ok.png", "payload_type": "vcard"},
        {"filename": "ok.png", "payload_type": "text", "text": ""},
        {"filename": "ok.png", "payload_type": "text", "text": "hi", "box_size": 0},
    ]
    for kwargs in bad_inputs:
        result = server.generate_qr_code(**kwargs)
        assert isinstance(result, str), kwargs
        assert result.startswith("ERROR"), kwargs


# ---------------------------------------------------------------------------
# Secrets must not be echoed back into the conversation
# ---------------------------------------------------------------------------

def test_wifi_password_is_not_echoed_in_the_confirmation(server):
    """The confirmation goes into the agent's context and from there into
    its long-term memory, so the payload is described rather than quoted."""
    result = server.generate_qr_code(
        "wifi.png", payload_type="wifi", ssid="GuestNet", password="s3cr3t-pw"
    )
    assert not result.startswith("ERROR")
    assert "s3cr3t-pw" not in result


def test_vcard_contents_are_not_echoed_in_the_confirmation(server):
    result = server.generate_qr_code(
        "card.png", payload_type="vcard", first_name="Ada", phone="+44 20 7946 0000"
    )
    assert not result.startswith("ERROR")
    assert "+44 20 7946 0000" not in result


# ---------------------------------------------------------------------------
# Field isolation between payload types
# ---------------------------------------------------------------------------

def test_fields_for_another_payload_type_are_ignored(server, output_dir):
    """Supplying vCard fields with payload_type='text' must not smuggle them
    into the payload -- only the fields declared for the chosen type are
    forwarded."""
    result = server.generate_qr_code(
        "out.png",
        payload_type="text",
        text="just this",
        ssid="SomeNetwork",
        password="leaked",
        first_name="Ada",
    )
    assert not result.startswith("ERROR")
    assert "SomeNetwork" not in result
    assert "leaked" not in result


# ---------------------------------------------------------------------------
# Discovery tool
# ---------------------------------------------------------------------------

def test_list_payload_types_names_every_type(server):
    result = server.qr_list_payload_types()
    for payload_type in ("text", "url", "wifi", "vcard"):
        assert payload_type in result


# ---------------------------------------------------------------------------
# Module-level wiring
# ---------------------------------------------------------------------------

def test_output_dir_honours_the_environment_variable(monkeypatch, tmp_path):
    """The client uses QR_OUTPUT_DIR to point the server at the agent's
    configured workspace, so it has to be read at import time."""
    custom = tmp_path / "elsewhere"
    monkeypatch.setenv("QR_OUTPUT_DIR", str(custom))
    reloaded = importlib.reload(qr_server)
    try:
        assert reloaded.OUTPUT_DIR == custom
    finally:
        # Restore the unpatched module for any test that runs afterwards.
        monkeypatch.delenv("QR_OUTPUT_DIR")
        importlib.reload(qr_server)


def test_default_output_dir_is_inside_the_workspace():
    assert qr_server.DEFAULT_OUTPUT_DIR.name == "qr_codes"
    assert qr_server.DEFAULT_OUTPUT_DIR.parent.name == "workspace"


def test_tools_are_registered_with_fastmcp():
    """Guards against the decorator being dropped in a refactor -- the tools
    exist as far as MCP is concerned, not merely as Python functions."""
    assert qr_server.mcp.name == "qr"
