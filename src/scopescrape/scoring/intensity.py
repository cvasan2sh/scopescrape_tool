"""Intensity scoring using VADER sentiment and signal phrase tier weighting.

Combines two signals:
  1. VADER compound sentiment (negative sentiment = stronger pain signal)
  2. Signal phrase tier weights (PAIN_POINT tier = highest weight)

Score range: 0.0 to 10.0
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from scopescrape.log import get_logger
from scopescrape.models import PainPoint, RawPost

logger = get_logger(__name__)


class IntensityScorer:
    """Score posts by emotional intensity of detected pain signals."""

    def __init__(self, config: dict):
        self.config = config
        self._vader = SentimentIntensityAnalyzer()

    def score(self, post: RawPost, signals: list[PainPoint]) -> float:
        """Calculate intensity score from sentiment + signal tier weights.

        Args:
            post: The post to score.
            signals: Detected signal phrases for this post.

        Returns:
            Score between 0.0 and 10.0.
        """
        if not signals:
            return 0.0

        # Component 1: VADER sentiment (60% weight)
        sentiment = self.vader_score(post.full_text)
        # Map sentiment: -1.0 (very negative) -> 10.0, +1.0 (very positive) -> 0.0
        # Pain discovery rewards negative sentiment
        sentiment_component = max(0.0, (1.0 - sentiment) * 5.0)

        # Component 2: Signal tier weights (40% weight)
        # Average the tier weights of all detected signals
        tier_weights = [s.tier.weight for s in signals]
        avg_tier_weight = sum(tier_weights) / len(tier_weights)
        tier_component = avg_tier_weight * 10.0

        # Combine: 60% sentiment, 40% tier
        combined = (sentiment_component * 0.6) + (tier_component * 0.4)
        return round(min(10.0, max(0.0, combined)), 3)

    def vader_score(self, text: str) -> float:
        """Get VADER compound sentiment score.

        Returns:
            Float between -1.0 (most negative) and 1.0 (most positive).
        """
        if not text:
            return 0.0
        scores = self._vader.polarity_scores(text)
        return scores["compound"]
