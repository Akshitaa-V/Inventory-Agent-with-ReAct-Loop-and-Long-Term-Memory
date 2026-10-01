"""
qr_server.py — self-hosted MCP server that generates QR code PNGs.

Same shape as recall_server.py: a FastMCP server over stdio, exposing a
small number of bounded tools, each returning an error *string* rather than
raising. An exception here would die inside a subprocess and reach the
agent as an opaque MCP transport failure, with nothing it could act on; a
returned string becomes an observation it can read and correct.

Tools exposed:
  - generate_qr_code(filename, payload_type, ...) -> confirmation or ERROR
  - qr_list_payload_types() -> the payload types and their fields

The capability is bounded in two ways. It only ever writes PNG files, and
it only writes them into one fixed directory: the tool takes a bare
filename, never a path, so there is no argument through which a caller can
express a location. See _resolve_output_path for the reasoning.

Run standalone for a manual check:
    python mcp_servers/qr/qr_server.py
(it idles waiting for a client on stdio -- normally it is launched by an
MCP client rather than run directly by a person)
"""

import os
import re
import sys
from pathlib import Path

# When this file is launched directly as a script -- which is how an MCP
# client spawns it, and how the MCP Inspector runs it -- Python puts THIS
# directory on sys.path, not the repository root, so the absolute imports
# below would fail. Adding the repository root explicitly makes the module
# work both ways: spawned as a script, and imported as
# mcp_servers.qr.qr_server by the tests.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from mcp_servers.qr.payloads import build_payload
from mcp_servers.qr.render import ERROR_CORRECTION_LEVELS, generate_qr

# Where generated codes are written. Every QR code this server produces
# lands here and nowhere else. The environment variable exists so the
# client can point the server at the agent's configured workspace, which
# is not knowable from inside this file; the default keeps the server
# usable standalone (with the Inspector, say) with no setup.
#
# Derived from __file__ rather than the working directory, so it resolves
# the same whether the server is spawned from the repository root, from
# elsewhere, or from inside the container.
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "workspace" / "qr_codes"
OUTPUT_DIR = Path(os.environ.get("QR_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)

# Filenames are restricted to an allowlist rather than a blocklist: letters,
# digits, spaces, dot, hyphen and underscore. A blocklist of "bad"
# characters is the wrong shape for this problem -- it has to anticipate
# every dangerous character, whereas an allowlist only has to name the safe
# ones.
SAFE_FILENAME_PATTERN = re.compile(r"^[A-Za-z0-9 ._-]+$")

MAX_FILENAME_LENGTH = 100

# Which flat tool arguments belong to which payload type. Declared here so
# that fields meant for one type cannot silently leak into another -- if a
# caller supplies vCard fields alongside payload_type="wifi", those fields
# are simply not forwarded, and the mismatch is reported.
PAYLOAD_FIELDS = {
    "wifi": ("ssid", "password", "auth_type", "hidden"),
    "url": ("url",),
    "vcard": ("first_name", "last_name", "organization", "title", "phone", "email", "url"),
    "text": ("text",),
}

mcp = FastMCP("qr")


def _resolve_output_path(filename: str) -> Path:
    """Validates a bare filename and joins it to the fixed output directory.

    The tool deliberately accepts a filename, not a path. Accepting a path
    would mean the server had to decide which paths are acceptable, which is
    the containment problem that '../' traversal exists to defeat; accepting
    a filename means there is no syntax for expressing a location at all,
    and containment follows from the argument's shape rather than from a
    check that has to be got right.

    Both separators are rejected, not just the platform's own: a payload
    crafted with backslashes would otherwise pass on Linux and then mean
    something different if the same name were ever used on Windows.

    Raises:
        ValueError: If the filename is empty, too long, contains a path
            separator or '..', or uses characters outside the allowlist.
    """
    filename = filename.strip()

    if not filename:
        raise ValueError("filename cannot be empty.")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(
            f"filename is too long ({len(filename)} characters, "
            f"maximum is {MAX_FILENAME_LENGTH})."
        )

    if "/" in filename or "\\" in filename:
        raise ValueError(
            f"filename must be a bare filename, not a path: {filename!r} "
            f"contains a path separator. Files are always written to the "
            f"QR output directory."
        )

    if ".." in filename:
        raise ValueError(
            f"filename must not contain '..': {filename!r}."
        )

    # Rejected explicitly rather than left to the pattern, so the message
    # says what is actually wrong.
    if filename.startswith("."):
        raise ValueError(
            f"filename must not start with a dot: {filename!r}."
        )

    if not SAFE_FILENAME_PATTERN.match(filename):
        raise ValueError(
            f"filename contains unsupported characters: {filename!r}. "
            f"Use letters, digits, spaces, dots, hyphens and underscores."
        )

    # The server only ever produces PNGs, so the extension is settled here
    # rather than trusted from the caller: a missing one is added, and a
    # different one is a mistake worth reporting rather than silently
    # renaming.
    suffix = Path(filename).suffix.lower()
    if not suffix:
        filename += ".png"
    elif suffix != ".png":
        raise ValueError(
            f"filename must end in .png (or omit the extension): {filename!r}."
        )

    resolved = (OUTPUT_DIR / filename).resolve()

    # Belt-and-braces. The checks above should already make this impossible,
    # so if it ever fires it means one of them has a hole -- which is
    # exactly when a second, independent check is worth having.
    output_root = OUTPUT_DIR.resolve()
    if output_root != resolved.parent:
        raise ValueError(
            f"refusing to write outside the QR output directory: {filename!r}."
        )

    return resolved


def _describe_payload(payload_type: str, payload: str) -> str:
    """A short, safe description of what was encoded, for the confirmation.

    Wi-Fi payloads are described without their contents on purpose: echoing
    the payload back would copy the network password into the conversation
    transcript and from there into the agent's long-term memory, for no
    benefit -- the caller already knows what it asked for.
    """
    if payload_type == "wifi":
        return "a Wi-Fi network configuration"
    if payload_type == "vcard":
        return "a contact card"
    if len(payload) > 80:
        return f"{payload[:80]}..."
    return payload


@mcp.tool()
def generate_qr_code(
    filename: str,
    payload_type: str = "text",
    text: str = "",
    url: str = "",
    ssid: str = "",
    password: str = "",
    auth_type: str = "WPA",
    hidden: bool = False,
    first_name: str = "",
    last_name: str = "",
    organization: str = "",
    title: str = "",
    phone: str = "",
    email: str = "",
    error_correction: str = "M",
    box_size: int = 10,
) -> str:
    """Generates a QR code image and saves it as a PNG.

    Useful for producing a scannable label for an inventory item, a link, a
    contact card, or Wi-Fi credentials for a guest network.

    Args:
        filename: Name for the PNG, WITHOUT any directory -- for example
            "printer-label.png". Must not contain '/' or '..'; the file is
            always written to the QR output directory. The .png extension
            is added if omitted.
        payload_type: What kind of thing to encode. One of:
            "text"  - any plain text (uses: text)
            "url"   - a web address (uses: url)
            "wifi"  - Wi-Fi credentials (uses: ssid, password, auth_type, hidden)
            "vcard" - a contact card (uses: first_name, last_name,
                      organization, title, phone, email, url)
        text: The text to encode, for payload_type "text".
        url: The web address, for payload_type "url" or "vcard". Must be
            http or https.
        ssid: Wi-Fi network name, for payload_type "wifi".
        password: Wi-Fi password. Omit for an open network.
        auth_type: "WPA" (correct for WPA2 and WPA3 too), "WEP", or
            "nopass" for an open network.
        hidden: True if the Wi-Fi network does not broadcast its name.
        first_name: Given name, for payload_type "vcard".
        last_name: Family name, for payload_type "vcard".
        organization: Company or organisation, for payload_type "vcard".
        title: Job title, for payload_type "vcard".
        phone: Telephone number, for payload_type "vcard".
        email: Email address, for payload_type "vcard".
        error_correction: "L", "M", "Q" or "H". Higher levels stay
            scannable when the printed code is damaged or partly obscured,
            at the cost of a physically larger code. Defaults to "M"; use
            "H" for a label that will be handled or may get scuffed.
        box_size: Pixels per module, controlling the image resolution.
            Defaults to 10.

    Returns:
        A confirmation naming the file that was written, or a string
        beginning with "ERROR:" describing what was wrong. Never raises.
    """
    normalised_type = payload_type.strip().lower()
    if normalised_type not in PAYLOAD_FIELDS:
        return (
            f"ERROR: unknown payload_type {payload_type!r}. "
            f"Expected one of: {', '.join(sorted(PAYLOAD_FIELDS))}."
        )

    try:
        output_path = _resolve_output_path(filename)
    except ValueError as e:
        return f"ERROR: {e}"

    # Only the fields that belong to this payload type are forwarded, so a
    # stray argument for a different type cannot end up in the payload.
    supplied = {
        "text": text,
        "url": url,
        "ssid": ssid,
        "password": password,
        "auth_type": auth_type,
        "hidden": hidden,
        "first_name": first_name,
        "last_name": last_name,
        "organization": organization,
        "title": title,
        "phone": phone,
        "email": email,
    }
    fields = {name: supplied[name] for name in PAYLOAD_FIELDS[normalised_type]}

    try:
        payload = build_payload(normalised_type, **fields)
    except ValueError as e:
        return f"ERROR: {e}"

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"ERROR: could not create the QR output directory ({e})."

    try:
        generate_qr(
            payload,
            str(output_path),
            error_correction=error_correction,
            box_size=box_size,
        )
    except ValueError as e:
        return f"ERROR: {e}"
    except OSError as e:
        return f"ERROR: could not write {output_path.name} ({e})."

    return (
        f"Wrote QR code to {output_path.name} in the QR output directory "
        f"({OUTPUT_DIR}). It encodes {_describe_payload(normalised_type, payload)}."
    )


@mcp.tool()
def qr_list_payload_types() -> str:
    """Lists the QR payload types this server can generate and the fields
    each one uses.

    Returns:
        A short description of each payload type. Never raises.
    """
    lines = ["This server can generate the following QR payload types:"]
    for name in sorted(PAYLOAD_FIELDS):
        fields = ", ".join(PAYLOAD_FIELDS[name])
        lines.append(f"  - {name}: uses {fields}")
    lines.append(
        f"Error-correction levels: {', '.join(sorted(ERROR_CORRECTION_LEVELS))}."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
