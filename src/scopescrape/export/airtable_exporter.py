"""Airtable exporter for scan results.

Exports scored results to an Airtable base using the REST API.
Requires AIRTABLE_API_KEY environment variable and base/table IDs in config.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from scopescrape.export.base import BaseExporter
from scopescrape.log import get_logger
from scopescrape.models import ScoredResult

logger = get_logger(__name__)


class AirtableExporter(BaseExporter):
    """Export results to an Airtable base using the REST API."""

    RATE_LIMIT_DELAY = 0.2  # 5 req/sec = 200ms between requests
    BATCH_SIZE = 10  # Airtable batch limit

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = os.environ.get("AIRTABLE_API_KEY") or config.get("airtable", {}).get("api_key", "")
        self.base_id = config.get("airtable", {}).get("base_id", "")
        self.scans_table_id = config.get("airtable", {}).get("scans_table_id", "")
        self.pain_points_table_id = config.get("airtable", {}).get("pain_points_table_id", "")
        self.signals_table_id = config.get("airtable", {}).get("signals_table_id", "")

        if not all([self.api_key, self.base_id, self.scans_table_id, self.pain_points_table_id, self.signals_table_id]):
            raise ValueError(
                "Airtable exporter requires AIRTABLE_API_KEY env var and "
                "base_id, scans_table_id, pain_points_table_id, signals_table_id in config"
            )

    def export(self, results: list[ScoredResult], output_file: Path):
        """Standard export interface (ignores output_file for Airtable).

        For Airtable, use export_to_airtable() instead to provide scan_metadata.
        """
        raise NotImplementedError(
            "Use export_to_airtable(results, scan_metadata) instead. "
            "output_file parameter is not used for Airtable exports."
        )

    def export_to_airtable(self, results: list[ScoredResult], scan_metadata: dict):
        """Export results to Airtable with full schema support.

        Args:
            results: List of ScoredResult objects to export.
            scan_metadata: Dict with 'platform', 'subreddits', 'keywords',
                          'time_range', 'min_score', etc.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library is required for Airtable export")

        if not results:
            logger.info("No results to export to Airtable")
            return

        # Step 1: Create Scan record
        scan_record_id = self._create_scan_record(scan_metadata, len(results), requests)

        if not scan_record_id:
            logger.error("Failed to create Scan record in Airtable")
            return

        logger.info(f"Created Scan record: {scan_record_id}")

        # Step 2: Create Pain Point records in batches
        pain_point_ids = {}  # maps post_id -> record_id

        for i in range(0, len(results), self.BATCH_SIZE):
            batch = results[i : i + self.BATCH_SIZE]
            batch_ids = self._create_pain_points_batch(batch, scan_record_id, requests)
            pain_point_ids.update(batch_ids)
            time.sleep(self.RATE_LIMIT_DELAY)

        logger.info(f"Created {len(pain_point_ids)} Pain Point records")

        # Step 3: Create Signal records in batches, linked to parent Pain Points
        signal_count = 0
        for result in results:
            if result.post_id in pain_point_ids:
                pain_point_record_id = pain_point_ids[result.post_id]

                # Create signals for this pain point
                for i in range(0, len(result.signal_phrases), self.BATCH_SIZE):
                    batch = result.signal_phrases[i : i + self.BATCH_SIZE]
                    created = self._create_signals_batch(batch, pain_point_record_id, requests)
                    signal_count += created
                    time.sleep(self.RATE_LIMIT_DELAY)

        logger.info(f"Created {signal_count} Signal records")
        logger.info(f"Airtable export complete: {len(results)} pain points, {signal_count} signals")

    def _create_scan_record(self, scan_metadata: dict, result_count: int, requests) -> Optional[str]:
        """Create a Scan record and return its ID."""
        url = f"https://api.airtable.com/v0/{self.base_id}/{self.scans_table_id}"

        # Extract metadata
        platforms = scan_metadata.get("platforms", [])
        subreddits = scan_metadata.get("subreddits", [])
        keywords = scan_metadata.get("keywords", [])
        time_range = scan_metadata.get("time_range", "")
        min_score = scan_metadata.get("min_score", 5.0)

        # Calculate stats
        fields = {
            "Scan ID": scan_metadata.get("scan_id", ""),
            "Platform": ", ".join(platforms) if platforms else "",
            "Subreddits": ", ".join(subreddits) if subreddits else "",
            "Keywords": ", ".join(keywords) if keywords else "",
            "Time Range": time_range,
            "Posts Fetched": result_count,
            "Results Scored": result_count,
            "Min Score": min_score,
        }

        payload = {"records": [{"fields": fields}]}

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            if data.get("records"):
                return data["records"][0]["id"]
        except Exception as e:
            logger.error(f"Failed to create Scan record: {e}")
            return None

    def _create_pain_points_batch(self, results: list[ScoredResult], scan_record_id: str, requests) -> dict:
        """Create a batch of Pain Point records. Returns {post_id -> record_id}."""
        url = f"https://api.airtable.com/v0/{self.base_id}/{self.pain_points_table_id}"

        records = []
        for result in results:
            fields = {
                "Title": result.title[:255] if result.title else "",
                "Scan": [scan_record_id],  # Linked record
                "Post ID": result.post_id,
                "Platform": result.platform,
                "Source": result.source,
                "Author": result.author,
                "URL": result.url,
                "Body Excerpt": result.body_excerpt[:1000] if result.body_excerpt else "",
                "Composite Score": round(result.composite_score, 3),
                "Frequency": round(result.frequency_score, 3),
                "Intensity": round(result.intensity_score, 3),
                "Specificity": round(result.specificity_score, 3),
                "Recency": round(result.recency_score, 3),
                "Sentiment": round(result.sentiment_score, 3),
                "Signal Count": len(result.signal_phrases),
                "Entities": ", ".join(result.entities) if result.entities else "",
                "Posted At": result.created_at.isoformat(),
            }
            records.append({"fields": fields})

        payload = {"records": records}

        mapping = {}
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            for i, record in enumerate(data.get("records", [])):
                if i < len(results):
                    mapping[results[i].post_id] = record["id"]
        except Exception as e:
            logger.error(f"Failed to create Pain Point batch: {e}")

        return mapping

    def _create_signals_batch(self, pain_points, parent_pain_point_record_id: str, requests) -> int:
        """Create a batch of Signal records linked to a Pain Point."""
        url = f"https://api.airtable.com/v0/{self.base_id}/{self.signals_table_id}"

        records = []
        for pain_point in pain_points:
            fields = {
                "Phrase": pain_point.phrase,
                "Pain Point": [parent_pain_point_record_id],  # Linked record
                "Tier": pain_point.tier.name,
                "Category": pain_point.category,
                "Context": pain_point.context,
            }
            records.append({"fields": fields})

        payload = {"records": records}

        created = 0
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            created = len(data.get("records", []))
        except Exception as e:
            logger.error(f"Failed to create Signal batch: {e}")

        return created
