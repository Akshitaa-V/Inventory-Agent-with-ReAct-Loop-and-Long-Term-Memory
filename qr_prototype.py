"""
qr_prototype.py — command-line front end for QR code generation.

The generation logic itself now lives in mcp_servers/qr/render.py, and the
payload formats in mcp_servers/qr/payloads.py; this file is only the
terminal interface to them. It was the standalone step-one prototype, and
is kept because being able to produce a QR code by hand, without the agent
or an MCP client in the way, is the quickest way to check that a change to
the rendering or payload code did what was intended.

Run it directly:
    python qr_prototype.py "https://example.com" out.png
    python qr_prototype.py "https://example.com" out.png --ec H --box-size 12

Requires the `qrcode` library (and Pillow, which it uses to write the PNG):
    pip install qrcode
"""

import argparse
import sys

from mcp_servers.qr.render import ERROR_CORRECTION_LEVELS, generate_qr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a QR code PNG from a string."
    )
    parser.add_argument("data", help="The text or URL to encode.")
    parser.add_argument("output_path", help="Where to write the PNG.")
    parser.add_argument(
        "--ec",
        default="M",
        choices=sorted(ERROR_CORRECTION_LEVELS),
        help="Error-correction level (default: M).",
    )
    parser.add_argument(
        "--box-size", type=int, default=10, help="Pixels per module (default: 10)."
    )
    parser.add_argument(
        "--border", type=int, default=4, help="Quiet-zone width in modules (default: 4)."
    )
    args = parser.parse_args()

    try:
        path = generate_qr(
            args.data,
            args.output_path,
            error_correction=args.ec,
            box_size=args.box_size,
            border=args.border,
        )
    except ValueError as e:
        # Argument problems are the user's to fix, so report them plainly on
        # stderr and exit non-zero rather than showing a traceback.
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"Error: could not write {args.output_path} ({e})", file=sys.stderr)
        return 1

    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
