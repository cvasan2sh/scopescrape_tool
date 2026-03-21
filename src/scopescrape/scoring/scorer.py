"""Main scorer: combines all four dimensions into a composite score.

Dimensions:
  1. Frequency  (BM25 relevance)     - default weight 0.25
  2. Intensity  (VADER + tier)       - default weight 0.20
  3. Specificity (NER + length)      - default weight 0.25
  4. Recency    (time decay)         - default weight 0.30

Composite = weighted sum of all four dimensions.
Posts below the min_score threshold are dropped.
"""

from __future__ import annotations

from typing import Optional

from scopescrape.log import get_logger
from scopescrape.models import PainPoint, RawPost, ScoredResult
from scopescrape.scoring.frequency import FrequencyScorer
from scopescrape.scoring.intensity import IntensityScorer
from scopescrape.scoring.recency import RecencyScorer
from scopescrape.scoring.specificity import SpecificityScorer
from scopescrape.utils import truncate

logger = get_logger(__name__)


class Scorer:
    """Multi-dimensional scorer for pain point relevance."""

    def __init__(self, config: dict):
        self.config = config
        scoring_config = config.get("scoring", {})
        weights = scoring_config.get("weights", {})

        self.weights = {
            "frequency": weights.get("frequency", 0.25),
            "intensity": weights.get("intensity", 0.20),
            "specificity": weights.get("specificity", 0.25),
            "recency": weights.get("recency", 0.30),
        }
        self.threshold = scoring_config.get("min_score", 5.0)

        self.frequency_scorer = FrequencyScorer(config)
        self.intensity_scorer = IntensityScorer(config)
        self.specificity_scorer = SpecificityScorer(config)
        self.recency_scorer = RecencyScorer(config)

        self._index_built = False

    def build_index(self, corpus: list[RawPost]):
        """Build the BM25 frequency index from the full corpus.

        Must be called before scoring individual posts.
        """
        self.frequency_scorer.build_index(corpus)
        self._index_built = True
        logger.debug(f"Scorer index built from {len(corpus)} posts")

    def score(
        self,
        post: RawPost,
        signals: list[PainPoint],
        corpus: list[RawPost],
    ) -> Optional[ScoredResult]:
        """Score a post across all four dimensions.

        Args:
            post: The post to score.
            signals: Detected signal phrases for this post.
            corpus: Full list of fetched posts (for frequency baseline).

        Returns:
            ScoredResult if composite >= threshold, else None.
        """
        if not signals:
            return None

        # Build index on first call if not done
        if not self._index_built:
            self.build_index(corpus)

        # Score each dimension
        freq = self.frequency_scorer.score(post)
        intensity = self.intensity_scorer.score(post, signals)
        specificity = self.specificity_scorer.score(post)
        recency = self.recency_scorer.score(post)

        # Weighted composite
        composite = (
            freq * self.weights["frequency"]
            + intensity * self.weights["intensity"]
            + specificity * self.weights["specificity"]
            + recency * self.weights["recency"]
        )

        if composite < self.threshold:
            return None

        # Extract entities and sentiment for metadata
        entities = self.specificity_scorer.extract_entities(post.full_text)
        sentiment = self.intensity_scorer.vader_score(post.full_text)

        return ScoredResult(
            post_id=post.id,
            platform=post.platform,
            title=post.title,
            body_excerpt=truncate(post.body, 200),
            author=post.author,
            source=post.source,
            created_at=post.created_at,
            url=post.url,
            frequency_score=freq,
            intensity_score=intensity,
            specificity_score=specificity,
            recency_score=recency,
            composite_score=round(composite, 3),
            signal_phrases=signals,
            sentiment_score=sentiment,
            entities=entities,
            text_length=len(post.body),
        )
