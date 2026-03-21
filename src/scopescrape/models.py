"""Data models for ScopeScrape.

All core dataclasses and enums used across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class SignalTier(Enum):
    """Signal phrase classification tiers, ordered by explicitness."""

    PAIN_POINT = 1   # "frustrated", "struggle", "bug", "broken"
    EMOTIONAL = 2    # "hate", "love", "wish", "pray"
    COMPARISON = 3   # "vs", "instead of", "alternative to"
    ASK = 4          # "how to", "anyone know", "is there a way"

    @property
    def weight(self) -> float:
        """Default scoring weight per tier. Lower tier number = stronger signal."""
        return {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4}[self.value]


@dataclass
class RawPost:
    """Normalized post format across all platforms.

    Every platform adapter converts its native objects into RawPost
    before passing them into the detection/scoring pipeline.
    """

    id: str
    platform: str                     # "reddit", "hn"
    source: str                       # subreddit name, "hn" feed, etc.
    title: str
    body: str
    author: str
    score: int                        # upvotes or points
    comment_count: int
    url: str
    created_at: datetime
    parent_id: Optional[str] = None   # set for comments
    children_ids: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Title + body concatenated for signal detection."""
        parts = [p for p in (self.title, self.body) if p]
        return " ".join(parts)

    @property
    def age_hours(self) -> float:
        """Hours since post creation."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / 3600


@dataclass
class PainPoint:
    """A single detected signal phrase with surrounding context."""

    phrase: str              # the matched text
    tier: SignalTier
    category: str            # "feature_request", "bug", "workflow", etc.
    position: int            # character offset in full_text
    context: str             # 50 chars before + phrase + 50 chars after
    confidence: float = 1.0  # all signal phrases default to 1.0 in v0.1

    @property
    def context_display(self) -> str:
        """Context string with the matched phrase highlighted in brackets."""
        start = self.context.find(self.phrase)
        if start == -1:
            return self.context
        end = start + len(self.phrase)
        return f"{self.context[:start]}[{self.phrase}]{self.context[end:]}"


@dataclass
class ScoredResult:
    """Final scored pain point ready for export.

    Combines post metadata, the four dimension scores, the composite
    score, and all detected signal phrases into one output record.
    """

    post_id: str
    platform: str
    title: str
    body_excerpt: str
    author: str
    source: str
    created_at: datetime
    url: str

    # Dimension scores (each 0.0 to 10.0)
    frequency_score: float
    intensity_score: float
    specificity_score: float
    recency_score: float

    # Weighted composite
    composite_score: float

    # Metadata
    signal_phrases: list[PainPoint] = field(default_factory=list)
    sentiment_score: float = 0.0     # VADER: -1.0 to 1.0
    entities: list[str] = field(default_factory=list)
    text_length: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON/CSV export."""
        return {
            "post_id": self.post_id,
            "platform": self.platform,
            "title": self.title,
            "body_excerpt": self.body_excerpt,
            "author": self.author,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "url": self.url,
            "frequency_score": round(self.frequency_score, 3),
            "intensity_score": round(self.intensity_score, 3),
            "specificity_score": round(self.specificity_score, 3),
            "recency_score": round(self.recency_score, 3),
            "composite_score": round(self.composite_score, 3),
            "sentiment_score": round(self.sentiment_score, 3),
            "entities": self.entities,
            "signal_count": len(self.signal_phrases),
            "signal_phrases": [
                {
                    "phrase": sp.phrase,
                    "tier": sp.tier.name,
                    "category": sp.category,
                    "context": sp.context,
                }
                for sp in self.signal_phrases
            ],
            "text_length": self.text_length,
        }
