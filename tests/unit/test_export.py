"""Tests for export formatters."""

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scopescrape.export.csv_exporter import CSVExporter
from scopescrape.export.json_exporter import JSONExporter
from scopescrape.export.parquet_exporter import ParquetExporter


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
