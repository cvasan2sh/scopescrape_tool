"""Base exporter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from scopescrape.models import ScoredResult


class BaseExporter(ABC):
    """Abstract base for result exporters."""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def export(self, results: list[ScoredResult], output_file: Path):
        """Write results to the specified file."""
        ...
