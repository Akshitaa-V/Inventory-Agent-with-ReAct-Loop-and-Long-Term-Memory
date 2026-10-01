"""
ocr_server.py — self-hosted MCP server wrapping Tesseract OCR.

Same pattern as recall_server.py and qr_server.py: a standalone MCP server (stdio transport)
that exposes a small set of tools. The agent's ToolRegistry talks to this
over stdio via ocr_mcp_client.py.

Tools exposed:
  - ocr_extract_text(image_path, lang="eng") -> plain text extracted from an image
  - ocr_extract_text_from_pdf(pdf_path, lang="eng") -> per-page text from a scanned PDF
  - ocr_list_languages() -> languages installed in this Tesseract build

Run standalone for a quick manual check:
    python3 ocr_server.py
(it will just idle waiting for a client on stdio — normally it's launched
by the MCP client, not run directly by a person)
"""

import sys
import shutil
from pathlib import Path

import pytesseract
from PIL import Image

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Startup checks — fail loudly and immediately if Tesseract isn't reachable,
# same "fail fast, don't fail silently mid-agent-run" spirit as the rest of
# the agent's startup path.
# ---------------------------------------------------------------------------
if shutil.which("tesseract") is None:
    print(
        "FATAL: tesseract binary not found on PATH. Install it first "
        "(e.g. `apt-get install tesseract-ocr`) before starting this server.",
        file=sys.stderr,
    )
    sys.exit(1)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

mcp = FastMCP("ocr")


def _validate_path(path_str: str) -> Path:
    """Resolve and sanity-check a path before touching the filesystem.

    Mirrors the path-containment fix made in tool_registry.py — reject
    anything that isn't a real, existing file rather than letting a bad
    path silently propagate into pytesseract's error messages.
    """
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"Not a file: {p}")
    return p


@mcp.tool()
def ocr_extract_text(image_path: str, lang: str = "eng") -> str:
    """Extract text from an image file using Tesseract OCR.

    Args:
        image_path: Path to an image file (png, jpg, jpeg, tif, tiff, bmp, webp).
        lang: Tesseract language code(s), e.g. "eng" or "eng+deu". Defaults to "eng".

    Returns:
        The extracted text, or a clear error message string if extraction failed.
    """
    try:
        p = _validate_path(image_path)
    except (FileNotFoundError, IsADirectoryError) as e:
        return f"ERROR: {e}"

    if p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        return (
            f"ERROR: unsupported image extension '{p.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
        )

    try:
        image = Image.open(p)
        text = pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractError as e:
        return f"ERROR: Tesseract failed on '{p.name}': {e}"
    except Exception as e:
        return f"ERROR: could not process '{p.name}': {e}"

    text = text.strip()
    if not text:
        return f"(no text detected in {p.name})"
    return text


@mcp.tool()
def ocr_extract_text_from_pdf(pdf_path: str, lang: str = "eng", max_pages: int = 20) -> str:
    """Extract text from a scanned (image-based) PDF using Tesseract OCR.

    Rasterizes each page and OCRs it — use this for scanned receipts saved
    as PDF rather than a photo. Requires poppler-utils (pdftoppm) on PATH.

    Args:
        pdf_path: Path to the PDF file.
        lang: Tesseract language code(s), e.g. "eng" or "eng+deu". Defaults to "eng".
        max_pages: Safety cap on how many pages to OCR (default 20).

    Returns:
        Text extracted from each page, separated by page markers, or an
        error message string if extraction failed.
    """
    try:
        p = _validate_path(pdf_path)
    except (FileNotFoundError, IsADirectoryError) as e:
        return f"ERROR: {e}"

    if p.suffix.lower() != ".pdf":
        return f"ERROR: '{p.name}' is not a .pdf file"

    if shutil.which("pdftoppm") is None:
        return (
            "ERROR: poppler-utils not installed (pdftoppm not found on PATH). "
            "Install it (e.g. `apt-get install poppler-utils`) to OCR PDFs."
        )

    try:
        from pdf2image import convert_from_path
    except ImportError:
        return "ERROR: pdf2image not installed. Run: pip install pdf2image"

    try:
        pages = convert_from_path(str(p))
    except Exception as e:
        return f"ERROR: could not rasterize '{p.name}': {e}"

    if len(pages) > max_pages:
        pages = pages[:max_pages]

    chunks = []
    for i, page_image in enumerate(pages, start=1):
        try:
            page_text = pytesseract.image_to_string(page_image, lang=lang).strip()
        except pytesseract.TesseractError as e:
            page_text = f"(OCR failed on this page: {e})"
        chunks.append(f"--- Page {i} ---\n{page_text or '(no text detected)'}")

    return "\n\n".join(chunks)


@mcp.tool()
def ocr_list_languages() -> str:
    """List the OCR languages available in this Tesseract installation."""
    try:
        langs = pytesseract.get_languages(config="")
    except Exception as e:
        return f"ERROR: could not list languages: {e}"
    return ", ".join(sorted(langs)) if langs else "(no languages reported)"


if __name__ == "__main__":
    mcp.run(transport="stdio")
