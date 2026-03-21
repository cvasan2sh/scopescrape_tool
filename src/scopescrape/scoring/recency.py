"""Recency scoring using exponential time decay.

Recent posts score higher. The decay function uses a configurable
half-life (default 168 hours = 1 week), meaning a post loses half
its recency score every week.

Score range: 0.0 to 10.0
"""

from __future__ import annotations

import math
from datetime import datetime

from scopescrape.log import get_logger
from scopescrape.models import RawPost

logger = get_logger(__name__)


class RecencyScorer:
    """Score posts by how recently they were created."""

    def __init__(self, config: dict):
        self.config = config
        self.half_life_hours = config.get("scoring", {}).get(
            "recency_half_life", 168.0  # 1 week
        )

    def score(self, post: RawPost) -> float:
        """Calculate recency score using exponential decay.

        A post created just now scores 10.0. After one half-life
        period it scores 5.0, after two half-lives 2.5, etc.

        Args:
            post: The post to score.

        Returns:
            Score between 0.0 and 10.0.
        """
        age_hours = post.age_hours
        if age_hours <= 0:
            return 10.0

        # Exponential decay: score = 10 * e^(-ln(2) * age / half_life)
        decay = math.exp(-0.693 * age_hours / self.half_life_hours)
        return round(decay * 10.0, 3)
