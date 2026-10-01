"""
render.py — turns a string into a QR code PNG.

This is the image half of QR generation; payloads.py is the format half.
Moved here from the repository-root prototype so that it ships inside the
container: the Dockerfile copies agent/, mcp_servers/, config.json and
samples/, so a module left at the repository root would be missing from the
image at runtime.

Knows nothing about MCP, the agent, or where files are allowed to be
written -- it writes to the path it is given, and the layer above decides
what path that may be.
"""

import qrcode
from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H,
)

# The four error-correction levels the QR standard defines, keyed by the
# letter people actually use to refer to them. Higher levels survive more
# damage to the printed code, but spend part of the symbol's capacity on
# recovery data -- so the same string needs a physically bigger QR code at
# H than at L. M is the usual default and what this module uses.
ERROR_CORRECTION_LEVELS = {
    "L": ERROR_CORRECT_L,  # recovers ~7% of the symbol
    "M": ERROR_CORRECT_M,  # ~15%
    "Q": ERROR_CORRECT_Q,  # ~25%
    "H": ERROR_CORRECT_H,  # ~30%
}

# A version-40 symbol (the largest the standard allows) holds at most 2953
# bytes, and only at error-correction level L. Checked up front so an
# oversized string produces a clear message here rather than a
# DataOverflowError from inside the library.
MAX_BYTES = 2953


def generate_qr(
    data: str,
    output_path: str,
    error_correction: str = "M",
    box_size: int = 10,
    border: int = 4,
) -> str:
    """Generates a QR code PNG from a string and writes it to output_path.

    Args:
        data: The text to encode -- a URL, an ID, or any other string.
        output_path: Where to write the PNG.
        error_correction: One of "L", "M", "Q", "H" (see the table above).
        box_size: Pixels per module (a "module" is one black or white square
            of the QR grid). This is what controls the output resolution;
            the module *count* is fixed by how much data there is.
        border: Width of the white quiet zone, counted in modules. The QR
            specification requires at least 4; scanners rely on it to find
            the symbol's edges, so anything smaller risks not scanning.

    Returns:
        The path that was written.

    Raises:
        ValueError: If the data is empty, too large to encode, or the
            arguments are out of range.
    """
    if not data:
        raise ValueError("data cannot be empty -- there is nothing to encode.")

    # Capacity is defined in bytes, not characters: a non-ASCII character
    # can take several bytes once encoded, so len(data) would undercount.
    byte_length = len(data.encode("utf-8"))
    if byte_length > MAX_BYTES:
        raise ValueError(
            f"data is too large to encode: {byte_length} bytes, "
            f"maximum is {MAX_BYTES}."
        )

    level = error_correction.upper()
    if level not in ERROR_CORRECTION_LEVELS:
        raise ValueError(
            f"unknown error-correction level {error_correction!r}. "
            f"Expected one of: {', '.join(ERROR_CORRECTION_LEVELS)}."
        )

    if box_size < 1:
        raise ValueError("box_size must be at least 1.")
    if border < 4:
        raise ValueError("border must be at least 4 (QR specification minimum).")

    # version=None combined with fit=True below lets the library choose the
    # smallest symbol version (1-40) that the data actually fits into,
    # instead of hardcoding a size that is either too small to hold the
    # data or needlessly large.
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECTION_LEVELS[level],
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    # make_image() hands back a wrapper around a Pillow image, which is why
    # Pillow is a real dependency here and not just an optional extra.
    image = qr.make_image(fill_color="black", back_color="white")
    image.save(output_path)

    return output_path
