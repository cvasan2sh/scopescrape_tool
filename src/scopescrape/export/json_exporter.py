"""JSON exporter for scan results."""

from __future__ import annotations

import json
from pathlib import Path

from scopescrape.export.base import BaseExporter
from scopescrape.log import get_logger
from scopescrape.models import ScoredResult

logger = get_logger(__name__)


class JSONExporter(BaseExporter):
    """Export results as a JSON file."""

    def export(self, results: list[ScoredResult], output_file: Path):
        output_file = Path(output_file)
        data = {
            "scopescrape_version": "0.1.0",
            "result_count": len(results),
            "results": [r.to_dict() for r in results],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"JSON export: {len(results)} results -> {output_file}")
