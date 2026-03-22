"""Pipeline orchestrator: Config -> Fetch -> Detect -> Score -> Store -> Export.

This is the central coordination point. The CLI calls Pipeline.run()
and it handles the full data flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from scopescrape.log import get_logger
from scopescrape.models import RawPost, ScoredResult
from scopescrape.storage import Storage

logger = get_logger(__name__)


class Pipeline:
    """Orchestrates the full scan pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.adapters: dict = {}
        self.storage = Storage(config)

    def run(
        self,
        platforms: list[str],
        queries: dict,
        export_format: str,
        output_file: Path,
        min_score: float = 5.0,
    ):
        """Execute the full pipeline: fetch, detect, score, store, export.

        Args:
            platforms: List of platform keys ("reddit", "hn").
            queries: Dict with "subreddits", "keywords", "limit", "time_range".
            export_format: "json", "csv", or "parquet".
            output_file: Path to write results.
            min_score: Minimum composite score to include.
        """
        logger.info(f"Starting scan: platforms={platforms}")

        # Phase 1: Fetch
        all_posts: list[RawPost] = []
        for platform in platforms:
            adapter = self._get_adapter(platform)
            posts = adapter.fetch(queries)
            all_posts.extend(posts)
            logger.info(f"  {platform}: fetched {len(posts)} posts")

        if not all_posts:
            logger.warning("No posts fetched. Nothing to analyze.")
            return

        # Phase 2: Detect signals
        from scopescrape.signals.detector import SignalDetector

        detector = SignalDetector(self.config)
        all_signals = {}

        for post in all_posts:
            signals = detector.detect(post.full_text, post.id)
            if signals:
                all_signals[post.id] = signals

        logger.info(f"  Signals found in {len(all_signals)}/{len(all_posts)} posts")

        # Phase 3: Score
        from scopescrape.scoring.scorer import Scorer

        scorer = Scorer(self.config)
        scored_results: list[ScoredResult] = []

        for post in all_posts:
            if post.id in all_signals:
                result = scorer.score(post, all_signals[post.id], all_posts)
                if result and result.composite_score >= min_score:
                    scored_results.append(result)

        logger.info(f"  {len(scored_results)} results above threshold ({min_score})")

        # Phase 4: Store
        self.storage.save_results(all_posts, all_signals, scored_results)
        self.storage.cleanup_old()

        # Phase 5: Export
        exporter = self._get_exporter(export_format)
        exporter.export(scored_results, output_file)
        logger.info(f"  Results written to {output_file}")

    def _get_adapter(self, platform: str):
        """Lazy-load and cache platform adapters."""
        if platform not in self.adapters:
            if platform == "reddit":
                from scopescrape.adapters.reddit import RedditAdapter

                self.adapters[platform] = RedditAdapter(self.config)
            elif platform == "hn":
                from scopescrape.adapters.hackernews import HackerNewsAdapter

                self.adapters[platform] = HackerNewsAdapter(self.config)
            elif platform == "github":
                from scopescrape.adapters.github import GitHubAdapter

                self.adapters[platform] = GitHubAdapter(self.config)
            elif platform == "stackoverflow":
                from scopescrape.adapters.stackoverflow import StackOverflowAdapter

                self.adapters[platform] = StackOverflowAdapter(self.config)
            elif platform == "twitter":
                from scopescrape.adapters.twitter import TwitterAdapter

                self.adapters[platform] = TwitterAdapter(self.config)
            elif platform == "producthunt":
                from scopescrape.adapters.producthunt import ProductHuntAdapter

                self.adapters[platform] = ProductHuntAdapter(self.config)
            elif platform == "indiehackers":
                from scopescrape.adapters.indiehackers import IndieHackersAdapter

                self.adapters[platform] = IndieHackersAdapter(self.config)
            else:
                raise ValueError(f"Unknown platform: {platform}")

        return self.adapters[platform]

    def _get_exporter(self, fmt: str):
        """Get the appropriate exporter for the output format."""
        if fmt == "json":
            from scopescrape.export.json_exporter import JSONExporter

            return JSONExporter(self.config)
        elif fmt == "csv":
            from scopescrape.export.csv_exporter import CSVExporter

            return CSVExporter(self.config)
        elif fmt == "parquet":
            from scopescrape.export.parquet_exporter import ParquetExporter

            return ParquetExporter(self.config)
        elif fmt == "airtable":
            from scopescrape.export.airtable_exporter import AirtableExporter

            return AirtableExporter(self.config)
        else:
            raise ValueError(f"Unknown export format: {fmt}")
