"""
tests/test_convert_mt5_csv_to_json.py

Unit tests for MT5 CSV to QUANTLAB structured JSON converter.
"""

import json
from pathlib import Path
import pytest

from tools.convert_mt5_csv_to_json import (
    ValidationError,
    build_json_payload,
    detect_columns,
    run_integrity_check,
    validate_and_convert_records,
)


@pytest.fixture
def sample_mt5_rows():
    return [
        {
            "<DateTime>": "2024.01.01 00:00:00",
            "<Open>": "100.5",
            "<High>": "105.0",
            "<Low>": "99.0",
            "<Close>": "103.0",
            "<TickVolume>": "1",
            "<RealVolume>": "5000",
            "<Spread>": "0",
        },
        {
            "<DateTime>": "2024.01.02 00:00:00",
            "<Open>": "103.0",
            "<High>": "106.0",
            "<Low>": "101.0",
            "<Close>": "104.5",
            "<TickVolume>": "1",
            "<RealVolume>": "6000",
            "<Spread>": "0",
        },
    ]


def test_detect_columns_with_angle_brackets():
    header = ["<DateTime>", "<Open>", "<High>", "<Low>", "<Close>", "<TickVolume>", "<RealVolume>", "<Spread>"]
    mapping = detect_columns(header)
    assert mapping["datetime"] == "<DateTime>"
    assert mapping["open"] == "<Open>"


def test_detect_columns_without_angle_brackets():
    header = ["DateTime", "Open", "High", "Low", "Close", "TickVolume", "RealVolume", "Spread"]
    mapping = detect_columns(header)
    assert mapping["datetime"] == "DateTime"
    assert mapping["open"] == "Open"


def test_detect_columns_missing():
    header = ["DateTime", "Open", "High"]
    with pytest.raises(ValidationError):
        detect_columns(header)


def test_validate_and_convert_records(sample_mt5_rows):
    header = list(sample_mt5_rows[0].keys())
    col_map = detect_columns(header)
    records, stats = validate_and_convert_records(sample_mt5_rows, col_map)

    assert len(records) == 2
    assert stats["num_input"] == 2
    assert stats["num_json"] == 2
    assert stats["first_datetime"] == "2024.01.01 00:00:00"
    assert records[0]["open"] == 100.5
    assert records[0]["tick_volume"] == 1
    assert records[0]["real_volume"] == 5000


def test_validate_and_convert_records_invalid_high(sample_mt5_rows):
    sample_mt5_rows[0]["<High>"] = "95.0"  # High < Open
    header = list(sample_mt5_rows[0].keys())
    col_map = detect_columns(header)
    with pytest.raises(ValidationError) as exc:
        validate_and_convert_records(sample_mt5_rows, col_map)
    assert "violates High >= max" in str(exc.value)


def test_validate_and_convert_records_duplicates(sample_mt5_rows):
    sample_mt5_rows[1]["<DateTime>"] = sample_mt5_rows[0]["<DateTime>"]
    header = list(sample_mt5_rows[0].keys())
    col_map = detect_columns(header)
    with pytest.raises(ValidationError) as exc:
        validate_and_convert_records(sample_mt5_rows, col_map)
    assert "Duplicate timestamp" in str(exc.value)


def test_build_json_payload():
    records = [{"datetime": "2024.01.01 00:00:00", "open": 100.0}]
    payload = build_json_payload("QUANTLAB_RELIANCE", records)
    assert payload["symbol"] == "QUANTLAB_RELIANCE"
    assert payload["description"] == "Reliance Industries - QUANTLAB Validation"
    assert payload["timeframe"] == "D1"
    assert payload["records"] == records


def test_roundtrip_integrity(tmp_path, sample_mt5_rows):
    csv_file = tmp_path / "test.csv"
    json_file = tmp_path / "test.json"

    # Write CSV
    header = list(sample_mt5_rows[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in sample_mt5_rows:
            f.write(",".join(str(r[c]) for c in header) + "\n")

    col_map = detect_columns(header)
    records, stats = validate_and_convert_records(sample_mt5_rows, col_map)
    payload = build_json_payload("QUANTLAB_TEST", records)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    assert run_integrity_check(csv_file, json_file, col_map) is True
