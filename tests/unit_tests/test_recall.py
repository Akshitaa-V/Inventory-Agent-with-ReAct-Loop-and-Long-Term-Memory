"""
Unit tests for the CPSC recall MCP tool (mcp_servers/recall/recall_server.py).

Network calls are mocked so these tests run offline and deterministically --
they don't depend on the live CPSC service being reachable or returning
any particular data at test time.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mcp_servers" / "recall"))
from recall_server import check_product_recall, _format_recall


SAMPLE_RECORD = {
    "RecallID": 10919,
    "RecallNumber": "26691",
    "RecallDate": "2026-08-13T00:00:00",
    "Title": "Deli Jerry Fuel Containers Recalled Due to Risk of Serious Injury",
    "URL": "https://www.cpsc.gov/Recalls/2026/Deli-Jerry-Fuel-Containers",
    "Hazards": [{"Name": "The recalled fuel containers violate the requirement for closures."}],
    "Remedies": [{"Name": "Consumers should stop using the recalled fuel containers immediately."}],
}


def test_rejects_empty_product_name():
    assert "ERROR" in check_product_recall("", 5)


def test_rejects_whitespace_only_product_name():
    assert "ERROR" in check_product_recall("   ", 5)


def test_rejects_overly_long_product_name():
    result = check_product_recall("x" * 300, 5)
    assert "ERROR" in result
    assert "too long" in result


def test_rejects_non_integer_max_results():
    assert "ERROR" in check_product_recall("blender", "not_a_number")


def test_clamps_max_results_to_valid_range():
    with patch("recall_server.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = [SAMPLE_RECORD] * 20
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = check_product_recall("blender", max_results=999)
        # Should be clamped to 10, not literally return 20 blocks
        assert result.count("Recall #") == 10


def test_no_recalls_found_returns_clear_message():
    with patch("recall_server.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = check_product_recall("a very obscure item unlikely to be recalled")
        assert "No CPSC recalls found" in result


def test_successful_match_includes_key_fields():
    with patch("recall_server.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = [SAMPLE_RECORD]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = check_product_recall("fuel container")
        assert "Deli Jerry Fuel Containers" in result
        assert "26691" in result
        assert "2026-08-13" in result
        assert "closures" in result
        assert "cpsc.gov" in result


def test_network_timeout_returns_clean_error_not_exception():
    import requests
    with patch("recall_server.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout()
        result = check_product_recall("blender")
        assert "ERROR" in result
        assert "timed out" in result


def test_connection_failure_returns_clean_error_not_exception():
    import requests
    with patch("recall_server.requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("no route to host")
        result = check_product_recall("blender")
        assert "ERROR" in result
        assert "could not reach" in result


def test_malformed_json_response_returns_clean_error():
    with patch("recall_server.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("not valid json")
        mock_get.return_value = mock_response

        result = check_product_recall("blender")
        assert "ERROR" in result
        assert "unreadable" in result


def test_format_recall_truncates_long_hazard_text():
    long_record = dict(SAMPLE_RECORD)
    long_record["Hazards"] = [{"Name": "x" * 500}]
    formatted = _format_recall(long_record)
    # Should be truncated, not the full 500 chars
    hazard_line = [l for l in formatted.splitlines() if l.strip().startswith("Hazard:")][0]
    assert len(hazard_line) < 250


def test_format_recall_handles_missing_hazards_gracefully():
    record = dict(SAMPLE_RECORD)
    record["Hazards"] = []
    record["Remedies"] = []
    formatted = _format_recall(record)
    assert "Not specified" in formatted
