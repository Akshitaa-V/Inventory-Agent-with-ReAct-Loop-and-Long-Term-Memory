"""
Unit tests for the QR payload builders (mcp_servers/qr/payloads.py).

These are pure string tests on purpose. The payload formats are the part of
QR generation that is easy to get subtly wrong and impossible to notice by
looking at a rendered code -- a mis-escaped semicolon produces a QR code
that scans perfectly and then joins the wrong network, and a vCard with its
N components in the wrong order produces a contact whose first and last
names are swapped. Asserting on the exact string is the only way to catch
either.

No network, no filesystem, no image generation is involved.
"""

import pytest

from mcp_servers.qr.payloads import (
    build_payload,
    build_text,
    build_url,
    build_vcard,
    build_wifi,
)


# ---------------------------------------------------------------------------
# Wi-Fi -- basic structure
# ---------------------------------------------------------------------------

def test_wifi_basic_structure():
    assert build_wifi("MyNetwork", "hunter2") == "WIFI:T:WPA;S:MyNetwork;P:hunter2;;"


def test_wifi_terminates_with_double_semicolon():
    """The trailing ';;' terminates the URI -- scanners rely on it."""
    assert build_wifi("net", "pw").endswith(";;")


def test_wifi_wep_auth_type():
    assert build_wifi("net", "pw", auth_type="WEP") == "WIFI:T:WEP;S:net;P:pw;;"


def test_wifi_auth_type_is_case_insensitive():
    assert build_wifi("net", "pw", auth_type="wpa") == build_wifi("net", "pw", auth_type="WPA")


def test_wifi_open_network_omits_password_field_entirely():
    """An open network must not carry an empty P: field -- the field is
    dropped, not left blank."""
    result = build_wifi("FreeWiFi", auth_type="nopass")
    assert result == "WIFI:T:nopass;S:FreeWiFi;;"
    assert "P:" not in result


def test_wifi_hidden_network_adds_flag():
    assert build_wifi("net", "pw", hidden=True) == "WIFI:T:WPA;S:net;P:pw;H:true;;"


def test_wifi_visible_network_omits_hidden_flag():
    """H:false is the default, so it is not emitted at all."""
    assert "H:" not in build_wifi("net", "pw", hidden=False)


# ---------------------------------------------------------------------------
# Wi-Fi -- escaping. This is the section that matters.
# ---------------------------------------------------------------------------

def test_wifi_escapes_semicolon_in_ssid():
    """An unescaped ';' would terminate the S: field early, so everything
    after it would be parsed as a new field."""
    assert build_wifi("my;network", "pw") == "WIFI:T:WPA;S:my\\;network;P:pw;;"


def test_wifi_escapes_semicolon_in_password():
    assert build_wifi("net", "pass;word") == "WIFI:T:WPA;S:net;P:pass\\;word;;"


def test_wifi_escapes_colon():
    """':' separates a field tag from its value."""
    assert build_wifi("net:work", "pw") == "WIFI:T:WPA;S:net\\:work;P:pw;;"


def test_wifi_escapes_comma():
    assert build_wifi("net,work", "pw") == "WIFI:T:WPA;S:net\\,work;P:pw;;"


def test_wifi_escapes_double_quote():
    assert build_wifi('net"work', "pw") == 'WIFI:T:WPA;S:net\\"work;P:pw;;'


def test_wifi_escapes_backslash():
    assert build_wifi("net\\work", "pw") == "WIFI:T:WPA;S:net\\\\work;P:pw;;"


def test_wifi_backslash_is_escaped_before_other_characters():
    """Ordering regression test.

    If ';' were escaped before '\\', the backslash introduced by that first
    replacement would itself be escaped by the second, turning 'a;b' into
    'a\\\\;b' -- an escaped backslash followed by a *bare* semicolon, which
    terminates the field. The backslash must be replaced first.
    """
    result = build_wifi("a\\b;c", "pw")
    assert result == "WIFI:T:WPA;S:a\\\\b\\;c;P:pw;;"
    # The literal backslash is doubled and the semicolon is singly escaped.
    assert "S:a\\\\b\\;c;" in result


def test_wifi_zxing_wiki_example():
    """The worked example from the ZXing 'Barcode Contents' wiki: an SSID of
    foo;bar\\baz encodes as S:foo\\;bar\\\\baz."""
    result = build_wifi("foo;bar\\baz", "pw")
    assert "S:foo\\;bar\\\\baz;" in result


def test_wifi_escapes_every_special_character_at_once():
    result = build_wifi("a;b,c:d\\e\"f", "pw")
    assert result == 'WIFI:T:WPA;S:a\\;b\\,c\\:d\\\\e\\"f;P:pw;;'


def test_wifi_leaves_ordinary_characters_untouched():
    """Escaping must not touch anything outside the five special characters,
    including spaces, unicode and characters that are special elsewhere."""
    assert build_wifi("Café Wifi-2.4GHz (guest)", "pw") == \
        "WIFI:T:WPA;S:Café Wifi-2.4GHz (guest);P:pw;;"


# ---------------------------------------------------------------------------
# Wi-Fi -- validation
# ---------------------------------------------------------------------------

def test_wifi_rejects_empty_ssid():
    with pytest.raises(ValueError, match="ssid cannot be empty"):
        build_wifi("", "pw")


def test_wifi_rejects_unknown_auth_type():
    with pytest.raises(ValueError, match="unknown auth_type"):
        build_wifi("net", "pw", auth_type="WPA3-Enterprise-Plus")


def test_wifi_rejects_password_on_open_network():
    with pytest.raises(ValueError, match="must not have a password"):
        build_wifi("net", "pw", auth_type="nopass")


def test_wifi_rejects_missing_password_on_secured_network():
    with pytest.raises(ValueError, match="requires a password"):
        build_wifi("net", "", auth_type="WPA")


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

def test_url_passes_through_https():
    assert build_url("https://example.com/path?q=1") == "https://example.com/path?q=1"


def test_url_passes_through_http():
    assert build_url("http://example.com") == "http://example.com"


def test_url_strips_surrounding_whitespace():
    assert build_url("  https://example.com  ") == "https://example.com"


def test_url_rejects_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        build_url("")


def test_url_rejects_missing_scheme():
    with pytest.raises(ValueError, match="must include a scheme"):
        build_url("example.com")


def test_url_rejects_missing_host():
    with pytest.raises(ValueError, match="missing a host"):
        build_url("https://")


@pytest.mark.parametrize("dangerous", [
    "javascript:alert(1)",
    "file:///etc/passwd",
    "ftp://example.com",
    "data:text/html,<script>alert(1)</script>",
])
def test_url_rejects_non_web_schemes(dangerous):
    """Only http and https are accepted -- a QR code is scanned by tapping,
    so the scheme decides what the phone does with it."""
    with pytest.raises(ValueError):
        build_url(dangerous)


# ---------------------------------------------------------------------------
# vCard -- structure and required ordering
# ---------------------------------------------------------------------------

def _vcard_lines(vcard):
    return vcard.split("\r\n")


def test_vcard_uses_crlf_line_endings():
    """RFC 2426 section 2.4.2 delimits content lines with CRLF."""
    vcard = build_vcard(first_name="Ada", last_name="Lovelace")
    assert "\r\n" in vcard
    assert "\n" not in vcard.replace("\r\n", "")


def test_vcard_begins_and_ends_correctly():
    lines = _vcard_lines(build_vcard(first_name="Ada", last_name="Lovelace"))
    assert lines[0] == "BEGIN:VCARD"
    assert lines[-1] == "END:VCARD"


def test_vcard_version_immediately_follows_begin():
    """RFC 2426 requires VERSION to come directly after BEGIN:VCARD, not
    merely somewhere in the card."""
    lines = _vcard_lines(build_vcard(first_name="Ada"))
    assert lines[1] == "VERSION:3.0"


def test_vcard_n_component_order_is_family_then_given():
    """N is Family;Given;Additional;Prefix;Suffix (RFC 2426 section 3.1.2).
    Reversing the first two produces a contact with swapped names and no
    error anywhere."""
    vcard = build_vcard(first_name="Ada", last_name="Lovelace")
    assert "N:Lovelace;Ada;;;" in vcard


def test_vcard_n_has_all_five_components():
    vcard = build_vcard(first_name="Ada", last_name="Lovelace")
    n_line = next(l for l in _vcard_lines(vcard) if l.startswith("N:"))
    assert n_line.count(";") == 4  # five components, four separators


def test_vcard_fn_is_derived_from_both_names():
    assert "FN:Ada Lovelace" in build_vcard(first_name="Ada", last_name="Lovelace")


def test_vcard_fn_with_only_first_name_has_no_stray_space():
    assert "FN:Ada\r\n" in build_vcard(first_name="Ada")


def test_vcard_optional_fields_are_emitted_when_supplied():
    vcard = build_vcard(
        first_name="Ada",
        last_name="Lovelace",
        organization="Analytical Engines Ltd",
        title="Mathematician",
        phone="+44 20 7946 0000",
        email="ada@example.com",
        url="https://example.com",
    )
    assert "ORG:Analytical Engines Ltd" in vcard
    assert "TITLE:Mathematician" in vcard
    assert "TEL;TYPE=CELL:+44 20 7946 0000" in vcard
    assert "EMAIL;TYPE=INTERNET:ada@example.com" in vcard
    assert "URL:https://example.com" in vcard


def test_vcard_omits_optional_fields_that_were_not_supplied():
    vcard = build_vcard(first_name="Ada", last_name="Lovelace")
    for absent in ("ORG:", "TITLE:", "TEL", "EMAIL", "URL:"):
        assert absent not in vcard


# ---------------------------------------------------------------------------
# vCard -- escaping
# ---------------------------------------------------------------------------

def test_vcard_escapes_semicolon_in_a_name_component():
    """An unescaped ';' inside a component would be read as the separator
    to the next component, shifting every following field."""
    vcard = build_vcard(first_name="Ada", last_name="Lovelace;King")
    assert "N:Lovelace\\;King;Ada;;;" in vcard


def test_vcard_escapes_comma():
    vcard = build_vcard(first_name="Ada", organization="Engines, Ltd")
    assert "ORG:Engines\\, Ltd" in vcard


def test_vcard_escapes_backslash():
    vcard = build_vcard(first_name="Ada", organization="A\\B")
    assert "ORG:A\\\\B" in vcard


def test_vcard_backslash_is_escaped_before_other_characters():
    """Same ordering hazard as the Wi-Fi builder."""
    vcard = build_vcard(first_name="Ada", organization="a\\b;c")
    assert "ORG:a\\\\b\\;c" in vcard


def test_vcard_converts_newline_to_escaped_n():
    """A literal newline would end the content line and corrupt the card;
    RFC 2426 section 5 represents it as the two characters '\\' and 'n'."""
    vcard = build_vcard(first_name="Ada", organization="Line1\nLine2")
    assert "ORG:Line1\\nLine2" in vcard
    # The card still has exactly the lines it should -- the newline did not
    # create a new content line.
    assert len(_vcard_lines(vcard)) == 6  # BEGIN, VERSION, N, FN, ORG, END


def test_vcard_does_not_escape_colon():
    """Unlike the Wi-Fi format, vCard does not escape ':' in values."""
    vcard = build_vcard(first_name="Ada", url="https://example.com")
    assert "URL:https://example.com" in vcard
    assert "\\:" not in vcard


def test_vcard_rejects_missing_name():
    with pytest.raises(ValueError, match="first_name or last_name"):
        build_vcard(organization="Engines Ltd")


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def test_text_passes_through_unchanged():
    assert build_text("hello world") == "hello world"


def test_text_preserves_special_characters_without_escaping():
    """Plain text has no format, so nothing is escaped."""
    assert build_text("a;b,c:d\\e") == "a;b,c:d\\e"


def test_text_preserves_surrounding_whitespace():
    assert build_text("  padded  ") == "  padded  "


def test_text_rejects_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        build_text("")


def test_text_rejects_whitespace_only():
    with pytest.raises(ValueError, match="cannot be empty"):
        build_text("   \n\t  ")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_build_payload_dispatches_to_each_builder():
    assert build_payload("wifi", ssid="net", password="pw") == build_wifi("net", "pw")
    assert build_payload("url", url="https://example.com") == "https://example.com"
    assert build_payload("text", text="hi") == "hi"
    assert build_payload("vcard", first_name="Ada").startswith("BEGIN:VCARD")


def test_build_payload_type_is_case_and_whitespace_insensitive():
    assert build_payload("  WiFi  ", ssid="net", password="pw") == build_wifi("net", "pw")


def test_build_payload_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown payload_type"):
        build_payload("morse", text="hi")


def test_build_payload_converts_bad_keywords_to_valueerror():
    """A wrong keyword arrives as TypeError; every failure out of this
    module should surface as ValueError so the callers above need to handle
    only one type."""
    with pytest.raises(ValueError, match="invalid fields"):
        build_payload("wifi", not_a_real_field="x")


def test_build_payload_propagates_builder_validation_errors():
    with pytest.raises(ValueError, match="ssid cannot be empty"):
        build_payload("wifi", ssid="", password="pw")
