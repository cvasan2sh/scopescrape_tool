"""CSV exporter for scan results."""

from __future__ import annotations

import csv
from pathlib import Path

from scopescrape.export.base import BaseExporter
from scopescrape.log import get_logger
from scopescrape.models import ScoredResult

logger = get_logger(__name__)

CSV_COLUMNS = [
    "post_id", "platform", "source", "title", "body_excerpt", "author",
    "url", "created_at", "composite_score", "frequency_score",
    "intensity_score", "specificity_score", "recency_score",
    "sentiment_score", "signal_count", "entities", "text_length",
]


class CSVExporter(BaseExporter):
    """Export results as a flat CSV file (one row per scored post)."""

    def export(self, results: list[ScoredResult], output_file: Path):
        output_file = Path(output_file)

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()

            for result in results:
                row = result.to_dict()
                # Flatten entities list to comma-separated string
                row["entities"] = ", ".join(row.get("entities", []))
                writer.writerow(row)

        logger.info(f"CSV export: {len(results)} results -> {output_file}")
