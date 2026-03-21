"""Base adapter interface and shared utilities for platform adapters.

Every platform adapter (Reddit, HN, etc.) extends BaseAdapter and
implements fetch(), which returns a list of RawPost objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from scopescrape.log import get_logger
from scopescrape.models import RawPost
from scopescrape.utils import RateLimiter

logger = get_logger(__name__)


class BaseAdapter(ABC):
    """Abstract base class for platform data adapters.

    Subclasses must implement:
        - fetch(queries) -> list[RawPost]
        - platform_name (property)
    """

    def __init__(self, config: dict):
        self.config = config

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Short identifier for this platform (e.g., 'reddit', 'hn')."""
        ...

    @abstractmethod
    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch and normalize posts from the platform.

        Args:
            queries: Dict with keys like 'subreddits', 'keywords',
                     'limit', 'time_range'.

        Returns:
            List of normalized RawPost objects.
        """
        ...

    def fetch_thread(self, post_id: str) -> tuple[Optional[RawPost], list[RawPost]]:
        """Fetch a single thread and its comments.

        Optional override for adapters that support it.

        Returns:
            (top_post, flat_list_of_comments)
        """
        raise NotImplementedError(f"{self.platform_name} does not support fetch_thread")
