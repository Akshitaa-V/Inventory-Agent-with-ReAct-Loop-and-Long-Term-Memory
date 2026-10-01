"""
recall_server.py — self-hosted MCP server wrapping the US Consumer Product
Safety Commission's (CPSC) public recall database (saferproducts.gov).

Free, official, keyless REST API. This replaces the Google Drive MCP server
in this project: unlike Drive, it needs no per-user OAuth login, so every
tester sees the exact same, correct results regardless of whose machine
runs the code -- and it's directly relevant to the inventory use case: an
item tracked for insurance purposes may have an active safety recall the
owner should know about.

Tool exposed:
  - check_product_recall(product_name, max_results=5) -> formatted text
    summary of matching recalls, or a clear "no recalls found" message.

This is a bounded, single-purpose tool: it only ever queries the fixed
CPSC recall endpoint with a product-name filter -- it is not a generic
HTTP-request tool and cannot be used to reach arbitrary URLs.
"""

import requests

from mcp.server.fastmcp import FastMCP

CPSC_RECALL_ENDPOINT = "https://www.saferproducts.gov/RestWebServices/Recall"
REQUEST_TIMEOUT_SECONDS = 10

mcp = FastMCP("recall")


def _format_recall(record: dict) -> str:
    """Turns one raw CPSC recall record into a short, readable block --
    the raw API response is deeply nested and far too verbose (full legal
    description, every retailer, every image) to hand back as-is."""
    title = record.get("Title", "Untitled recall")
    date = (record.get("RecallDate") or "")[:10]  # YYYY-MM-DD, drop time
    recall_number = record.get("RecallNumber", "unknown")
    url = record.get("URL", "")

    hazards = record.get("Hazards") or []
    hazard_text = hazards[0].get("Name", "") if hazards else "Not specified"
    if len(hazard_text) > 200:
        hazard_text = hazard_text[:200].rsplit(" ", 1)[0] + "..."

    remedies = record.get("Remedies") or []
    remedy_text = remedies[0].get("Name", "") if remedies else "Not specified"
    if len(remedy_text) > 200:
        remedy_text = remedy_text[:200].rsplit(" ", 1)[0] + "..."

    return (
        f"- {title} (Recall #{recall_number}, {date})\n"
        f"  Hazard: {hazard_text}\n"
        f"  Remedy: {remedy_text}\n"
        f"  Details: {url}"
    )


@mcp.tool()
def check_product_recall(product_name: str, max_results: int = 5) -> str:
    """Checks the official US CPSC (Consumer Product Safety Commission)
    recall database for safety recalls matching a product name. Useful for
    checking whether an inventory item -- especially medical equipment,
    appliances, or children's products -- has an active safety recall.

    Args:
        product_name: The product name or keyword to search for
            (e.g. "blender", "baby monitor", "space heater").
        max_results: Maximum number of matching recalls to return
            (default 5, clamped to 1-10).

    Returns:
        A formatted summary of matching recalls, or a clear message if
        none are found. Returns an error message string (never raises)
        if the product name is invalid or the CPSC service is unreachable.
    """
    product_name = (product_name or "").strip()
    if not product_name:
        return "ERROR: product_name cannot be empty."
    if len(product_name) > 200:
        return "ERROR: product_name is too long (max 200 characters)."

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        return "ERROR: max_results must be an integer."
    max_results = max(1, min(max_results, 10))

    try:
        response = requests.get(
            CPSC_RECALL_ENDPOINT,
            params={"format": "json", "ProductName": product_name},
            headers={
                "User-Agent": "InventoryAgentBot/1.0 (student project; "
                               "AI Engineering Lab; contact: group-2)",
                "Accept": "application/json",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "ERROR: CPSC recall database timed out. Try again shortly."
    except requests.exceptions.RequestException as e:
        return f"ERROR: could not reach CPSC recall database ({e})."

    try:
        records = response.json()
    except ValueError:
        return "ERROR: CPSC recall database returned an unreadable response."

    if not records:
        return f"No CPSC recalls found matching '{product_name}'."

    shown = records[:max_results]
    blocks = [_format_recall(r) for r in shown]
    header = f"Found {len(records)} recall(s) matching '{product_name}'"
    if len(records) > max_results:
        header += f" (showing {max_results})"
    header += ":"

    return header + "\n\n" + "\n\n".join(blocks)


if __name__ == "__main__":
    mcp.run(transport="stdio")