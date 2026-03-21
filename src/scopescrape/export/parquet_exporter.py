"""Parquet exporter for scan results (columnar, analytics-friendly).

Requires the optional 'parquet' extra: pip install scopescrape[parquet]
"""

from __future__ import annotations

from pathlib import Path

from scopescrape.export.base import BaseExporter
from scopescrape.log import get_logger
from scopescrape.models import ScoredResult

logger = get_logger(__name__)


class ParquetExporter(BaseExporter):
    """Export results as an Apache Parquet file."""

    def export(self, results: list[ScoredResult], output_file: Path):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.error(
                "pyarrow is not installed. Install it with: pip install scopescrape[parquet]"
            )
            raise SystemExit(
                "Parquet export requires pyarrow. Install with: pip install scopescrape[parquet]"
            )

        output_file = Path(output_file)

        if not results:
            logger.warning("No results to export to Parquet")
            return

        rows = [r.to_dict() for r in results]
        # Flatten entities and signal_phrases for columnar storage
        for row in rows:
            row["entities"] = ", ".join(row.get("entities", []))
            row["top_signal"] = (
                row["signal_phrases"][0]["phrase"] if row.get("signal_phrases") else ""
            )
            row["signal_phrases"] = str(row.get("signal_phrases", []))

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(output_file))

        logger.info(f"Parquet export: {len(results)} results -> {output_file}")
