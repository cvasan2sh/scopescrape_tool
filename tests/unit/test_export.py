"""Tests for export formatters."""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow.parquet as pq
import pytest

from scopescrape.export.csv_exporter import CSVExporter
from scopescrape.export.json_exporter import JSONExporter
from scopescrape.export.parquet_exporter import ParquetExporter
from scopescrape.export.airtable_exporter import AirtableExporter


@pytest.fixture
def results_list(sample_scored_result):
    return [sample_scored_result]


class TestJSONExporter:
    def test_export_creates_file(self, sample_config, results_list, tmp_path):
        out = tmp_path / "results.json"
        exporter = JSONExporter(sample_config)
        exporter.export(results_list, out)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["result_count"] == 1
        assert data["results"][0]["post_id"] == "t3_abc123"

    def test_export_empty_results(self, sample_config, tmp_path):
        out = tmp_path / "empty.json"
        exporter = JSONExporter(sample_config)
        exporter.export([], out)

        data = json.loads(out.read_text())
        assert data["result_count"] == 0
        assert data["results"] == []


class TestCSVExporter:
    def test_export_creates_file(self, sample_config, results_list, tmp_path):
        out = tmp_path / "results.csv"
        exporter = CSVExporter(sample_config)
        exporter.export(results_list, out)

        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["post_id"] == "t3_abc123"
        assert float(rows[0]["composite_score"]) == 7.83

    def test_export_has_all_columns(self, sample_config, results_list, tmp_path):
        out = tmp_path / "results.csv"
        exporter = CSVExporter(sample_config)
        exporter.export(results_list, out)

        with open(out) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        assert "composite_score" in headers
        assert "signal_count" in headers
        assert "entities" in headers


class TestParquetExporter:
    def test_export_creates_file(self, sample_config, results_list, tmp_path):
        out = tmp_path / "results.parquet"
        exporter = ParquetExporter(sample_config)
        exporter.export(results_list, out)

        assert out.exists()
        table = pq.read_table(str(out))
        assert table.num_rows == 1
        assert "composite_score" in table.column_names

    def test_export_empty_results(self, sample_config, tmp_path):
        out = tmp_path / "empty.parquet"
        exporter = ParquetExporter(sample_config)
        exporter.export([], out)
        assert not out.exists()  # should skip writing


class TestAirtableExporter:
    @pytest.fixture
    def airtable_config(self, sample_config):
        """Config with Airtable settings."""
        config = sample_config.copy()
        config["airtable"] = {
            "api_key": "pat_test_key_123",
            "base_id": "appTest123",
            "scans_table_id": "tblScans123",
            "pain_points_table_id": "tblPainPoints456",
            "signals_table_id": "tblSignals789",
        }
        return config

    def test_init_with_missing_config(self, sample_config):
        """Should raise ValueError if Airtable config is incomplete."""
        with pytest.raises(ValueError, match="requires AIRTABLE_API_KEY"):
            AirtableExporter(sample_config)

    def test_init_from_env_var(self, sample_config, monkeypatch):
        """Should read API key from environment variable."""
        config = sample_config.copy()
        config["airtable"] = {
            "base_id": "appTest123",
            "scans_table_id": "tblScans123",
            "pain_points_table_id": "tblPainPoints456",
            "signals_table_id": "tblSignals789",
        }
        monkeypatch.setenv("AIRTABLE_API_KEY", "pat_env_key_xyz")

        exporter = AirtableExporter(config)
        assert exporter.api_key == "pat_env_key_xyz"

    def test_export_not_implemented(self, airtable_config, results_list):
        """Should raise NotImplementedError for standard export method."""
        exporter = AirtableExporter(airtable_config)
        with pytest.raises(NotImplementedError, match="export_to_airtable"):
            exporter.export(results_list, Path("dummy.txt"))

    def test_export_to_airtable_success(self, airtable_config, results_list):
        """Should successfully export results to Airtable."""
        # Create a mock for requests module
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"records": [{"id": "rec_scan_123"}]}
        mock_requests.post.return_value = mock_response

        exporter = AirtableExporter(airtable_config)
        scan_metadata = {
            "scan_id": "scan_001",
            "platforms": ["reddit"],
            "subreddits": ["r/saas"],
            "keywords": ["frustrated"],
            "time_range": "week",
            "min_score": 5.0,
        }

        with patch("scopescrape.export.airtable_exporter.AirtableExporter._create_scan_record") as mock_scan:
            with patch("scopescrape.export.airtable_exporter.AirtableExporter._create_pain_points_batch") as mock_pain:
                with patch("scopescrape.export.airtable_exporter.AirtableExporter._create_signals_batch") as mock_sig:
                    mock_scan.return_value = "rec_scan_123"
                    mock_pain.return_value = {"t3_abc123": "rec_pain_123"}
                    mock_sig.return_value = 1

                    exporter.export_to_airtable(results_list, scan_metadata)

                    # Verify methods were called
                    mock_scan.assert_called_once()
                    mock_pain.assert_called_once()
                    mock_sig.assert_called()

    def test_export_to_airtable_empty_results(self, airtable_config):
        """Should handle empty results gracefully."""
        exporter = AirtableExporter(airtable_config)
        scan_metadata = {
            "platforms": ["reddit"],
            "subreddits": ["r/test"],
            "keywords": ["test"],
            "time_range": "week",
            "min_score": 5.0,
        }

        # Should complete without error
        exporter.export_to_airtable([], scan_metadata)

    def test_create_scan_record(self, airtable_config):
        """Should create a Scan record with correct fields."""
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"records": [{"id": "rec_scan_001"}]}
        mock_requests.post.return_value = mock_response

        exporter = AirtableExporter(airtable_config)
        scan_metadata = {
            "scan_id": "scan_abc",
            "platforms": ["reddit", "hn"],
            "subreddits": ["r/startup", "r/saas"],
            "keywords": ["bug", "frustration"],
            "time_range": "month",
            "min_score": 6.0,
        }

        result = exporter._create_scan_record(scan_metadata, 5, mock_requests)

        assert result == "rec_scan_001"
        # Verify the POST call
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert "appTest123/tblScans123" in call_args[0][0]

    def test_create_pain_points_batch(self, airtable_config, results_list):
        """Should create Pain Point records linked to a Scan."""
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "records": [{"id": "rec_pain_001"}, {"id": "rec_pain_002"}]
        }
        mock_requests.post.return_value = mock_response

        exporter = AirtableExporter(airtable_config)
        pain_point_ids = exporter._create_pain_points_batch(results_list, "rec_scan_001", mock_requests)

        assert results_list[0].post_id in pain_point_ids
        assert pain_point_ids[results_list[0].post_id] == "rec_pain_001"

    def test_create_signals_batch(self, airtable_config, sample_pain_point):
        """Should create Signal records linked to a Pain Point."""
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "records": [{"id": "rec_signal_001"}, {"id": "rec_signal_002"}]
        }
        mock_requests.post.return_value = mock_response

        exporter = AirtableExporter(airtable_config)
        created = exporter._create_signals_batch([sample_pain_point], "rec_pain_001", mock_requests)

        assert created == 2  # 2 signals created in mock response
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert "appTest123/tblSignals789" in call_args[0][0]
