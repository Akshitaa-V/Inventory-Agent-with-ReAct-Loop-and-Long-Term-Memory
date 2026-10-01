"""
payloads.py — builders that turn structured fields into the exact strings
that go inside a QR code.

A QR code is just a container for bytes; what makes a scanned code join a
Wi-Fi network or create a contact is that the decoded text follows a format
the scanner recognises. This module owns those formats and nothing else --
no image generation, no MCP, no filesystem. Each builder takes structured
fields, validates them, and returns the string to encode, raising ValueError
on bad input.

Keeping this separate from image generation means the formats can be tested
by string comparison, which is the only way to check escaping rules
meaningfully -- rendering to a PNG and decoding it again would test the
image pipeline, not the payload.

Format references:
  - Wi-Fi:  ZXing "Barcode Contents" wiki, WIFI: URI scheme.
            https://github.com/zxing/zxing/wiki/Barcode-Contents
            This is a de facto standard, not an RFC -- see WIFI_ESCAPE_CHARS.
  - vCard:  RFC 2426 (vCard 3.0). Escaping rules in section 5, the
            structured N property in section 3.1.2, required FN in 3.1.1.
            https://www.rfc-editor.org/rfc/rfc2426
  - URL:    RFC 3986 for the general syntax; only the scheme is constrained
            here (see build_url for why).
"""

from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Wi-Fi
# ---------------------------------------------------------------------------

# The WIFI: format uses ';' to separate fields and ':' to separate a field's
# tag from its value, so any of those characters appearing *inside* an SSID
# or password would otherwise be read as structure. The ZXing wiki specifies
# escaping '\', ';', ',' and ':' with a backslash, as in MECARD encoding;
# '"' is included because a value that begins with a quote is otherwise
# treated as a quoted literal by some parsers.
#
# Order matters when escaping: the backslash must be replaced first, or the
# backslashes introduced by the later replacements would themselves be
# escaped a second time.
WIFI_ESCAPE_CHARS = ["\\", ";", ",", ":", '"']

WIFI_AUTH_TYPES = {
    "WPA": "WPA",      # covers WPA, WPA2 and WPA3-Personal; scanners treat them alike
    "WEP": "WEP",
    "NOPASS": "nopass",  # open network -- the password field is omitted entirely
}


def _escape_wifi(value: str) -> str:
    """Backslash-escapes the characters that are structural in a WIFI: URI."""
    for char in WIFI_ESCAPE_CHARS:
        value = value.replace(char, "\\" + char)
    return value


def build_wifi(
    ssid: str,
    password: str = "",
    auth_type: str = "WPA",
    hidden: bool = False,
) -> str:
    """Builds a WIFI: payload that a phone camera can use to join a network.

    Args:
        ssid: The network name. Must not be empty.
        password: The network password. Required unless auth_type is
            "nopass", in which case it must be empty.
        auth_type: "WPA" (also correct for WPA2/WPA3), "WEP", or "nopass"
            for an open network. Case-insensitive.
        hidden: True if the network does not broadcast its SSID. Only
            emitted when true, since H:false is the default anyway.

    Returns:
        A string of the form WIFI:T:WPA;S:name;P:secret;; -- note the
        double semicolon, which terminates the URI.

    Raises:
        ValueError: If the SSID is empty, the auth type is unknown, or the
            password is inconsistent with the auth type.
    """
    if not ssid:
        raise ValueError("ssid cannot be empty.")

    normalised = auth_type.strip().upper()
    if normalised not in WIFI_AUTH_TYPES:
        raise ValueError(
            f"unknown auth_type {auth_type!r}. "
            f"Expected one of: WPA, WEP, nopass."
        )
    auth = WIFI_AUTH_TYPES[normalised]

    if auth == "nopass":
        if password:
            raise ValueError(
                "an open network (auth_type='nopass') must not have a password."
            )
    elif not password:
        raise ValueError(
            f"a {auth} network requires a password. "
            f"Use auth_type='nopass' for an open network."
        )

    parts = [f"WIFI:T:{auth}", f"S:{_escape_wifi(ssid)}"]
    if auth != "nopass":
        parts.append(f"P:{_escape_wifi(password)}")
    if hidden:
        parts.append("H:true")

    # Each field is terminated by ';', and the URI as a whole by a further
    # ';' -- which is why a well-formed payload ends in ';;'.
    return ";".join(parts) + ";;"


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

# Restricted to the two schemes a QR code is normally expected to carry.
# Allowing arbitrary schemes would turn this into a way to emit things like
# javascript: or file: URIs, which is a meaningfully different capability
# from "encode a web address".
URL_ALLOWED_SCHEMES = {"http", "https"}


def build_url(url: str) -> str:
    """Validates a web address and returns it unchanged.

    A URL needs no special encoding to sit inside a QR code -- scanners
    recognise it by its scheme. This builder exists for the validation, so
    that a typo produces an error here rather than an unscannable code or
    one that silently opens the wrong thing.

    Args:
        url: An absolute http or https URL.

    Returns:
        The URL, stripped of surrounding whitespace.

    Raises:
        ValueError: If the URL has no scheme, an unsupported scheme, or no
            host component.
    """
    url = url.strip()
    if not url:
        raise ValueError("url cannot be empty.")

    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError(
            f"url must include a scheme, e.g. https://{url} rather than {url}."
        )
    if parsed.scheme.lower() not in URL_ALLOWED_SCHEMES:
        raise ValueError(
            f"unsupported URL scheme {parsed.scheme!r}. "
            f"Expected one of: {', '.join(sorted(URL_ALLOWED_SCHEMES))}."
        )
    if not parsed.netloc:
        raise ValueError(f"url is missing a host: {url!r}.")

    return url


# ---------------------------------------------------------------------------
# vCard
# ---------------------------------------------------------------------------

# RFC 2426 section 5: within a text value, backslash, comma and semicolon
# are escaped with a backslash, and a line break becomes the two characters
# '\' and 'n'. Colon is NOT escaped in vCard, unlike in the Wi-Fi format --
# a difference worth keeping in mind, since the two look similar.
#
# As with Wi-Fi, the backslash is replaced first so that the backslashes
# introduced below are not escaped again.
VCARD_ESCAPES = [
    ("\\", "\\\\"),
    (",", "\\,"),
    (";", "\\;"),
    ("\n", "\\n"),
    ("\r", ""),  # bare CR carries no meaning once CRLF pairs are normalised
]

# RFC 2426 section 2.4.2: content lines are delimited by CRLF, not LF.
VCARD_LINE_ENDING = "\r\n"


def _escape_vcard(value: str) -> str:
    """Escapes a vCard text value per RFC 2426 section 5."""
    for char, replacement in VCARD_ESCAPES:
        value = value.replace(char, replacement)
    return value


def build_vcard(
    first_name: str = "",
    last_name: str = "",
    organization: str = "",
    title: str = "",
    phone: str = "",
    email: str = "",
    url: str = "",
) -> str:
    """Builds a vCard 3.0 payload that scanners turn into a contact entry.

    Field order is not decorative. RFC 2426 requires that BEGIN:VCARD is the
    first line, that VERSION comes immediately after it, and that END:VCARD
    is the last; and the N property (section 3.1.2) is a structured value
    whose five components must appear in the fixed order
    Family;Given;Additional;Prefix;Suffix. Getting that order wrong does not
    produce an error, it produces a contact with the names in the wrong
    fields -- which is why it is asserted explicitly in the tests.

    Args:
        first_name: Given name.
        last_name: Family name.
        organization: Company or organisation.
        title: Job title.
        phone: Telephone number.
        email: Email address.
        url: Associated web address.

    Returns:
        A CRLF-delimited vCard 3.0 string.

    Raises:
        ValueError: If neither a first nor a last name is supplied. FN is a
            required property in vCard 3.0 (section 3.1.1) and is derived
            from the two name parts here, so at least one must be present.
    """
    if not first_name and not last_name:
        raise ValueError(
            "at least one of first_name or last_name is required -- "
            "vCard 3.0 requires a formatted name (FN)."
        )

    lines = ["BEGIN:VCARD", "VERSION:3.0"]

    # N: the five structured components, in the order RFC 2426 fixes. Each
    # component is escaped individually, so that a semicolon inside a name
    # cannot be mistaken for a component separator.
    name_components = [
        _escape_vcard(last_name),   # Family
        _escape_vcard(first_name),  # Given
        "",                         # Additional (middle names)
        "",                         # Prefix (e.g. Dr.)
        "",                         # Suffix (e.g. Jr.)
    ]
    lines.append("N:" + ";".join(name_components))

    # FN is the human-readable display name and is required. Built from the
    # parts rather than asking for it separately, so the two can never
    # disagree.
    formatted_name = " ".join(part for part in (first_name, last_name) if part)
    lines.append("FN:" + _escape_vcard(formatted_name))

    # Everything below is optional and emitted only when supplied, so a
    # sparse contact does not carry a run of empty properties.
    if organization:
        lines.append("ORG:" + _escape_vcard(organization))
    if title:
        lines.append("TITLE:" + _escape_vcard(title))
    if phone:
        lines.append("TEL;TYPE=CELL:" + _escape_vcard(phone))
    if email:
        lines.append("EMAIL;TYPE=INTERNET:" + _escape_vcard(email))
    if url:
        lines.append("URL:" + _escape_vcard(url))

    lines.append("END:VCARD")

    return VCARD_LINE_ENDING.join(lines)


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def build_text(text: str) -> str:
    """Returns arbitrary text unchanged, after checking it is not empty.

    There is no format to apply: a QR code containing text that matches no
    known scheme is displayed as-is by scanners. The builder exists so that
    every payload type reaches the caller through the same interface, and so
    that the empty-string case is rejected in one place.

    Args:
        text: Any non-empty string.

    Returns:
        The text, unchanged -- including any leading or trailing whitespace,
        which is preserved deliberately since it may be significant.

    Raises:
        ValueError: If the text is empty or entirely whitespace.
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty.")
    return text


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Maps a payload type name to its builder. Used by the MCP layer later so
# that adding a payload type is a change in this module alone.
BUILDERS = {
    "wifi": build_wifi,
    "url": build_url,
    "vcard": build_vcard,
    "text": build_text,
}


def build_payload(payload_type: str, **fields) -> str:
    """Builds a payload of the named type from keyword fields.

    Args:
        payload_type: One of "wifi", "url", "vcard", "text".
        **fields: The arguments for that type's builder.

    Returns:
        The string to encode in a QR code.

    Raises:
        ValueError: If the type is unknown, a required field is missing, or
            an unexpected field is supplied.
    """
    normalised = payload_type.strip().lower()
    if normalised not in BUILDERS:
        raise ValueError(
            f"unknown payload_type {payload_type!r}. "
            f"Expected one of: {', '.join(sorted(BUILDERS))}."
        )

    try:
        return BUILDERS[normalised](**fields)
    except TypeError as e:
        # A wrong or missing keyword reaches us as TypeError; converted here
        # so that every failure out of this module is a ValueError, which is
        # what the layers above are written to expect.
        raise ValueError(f"invalid fields for payload_type {normalised!r}: {e}") from e
