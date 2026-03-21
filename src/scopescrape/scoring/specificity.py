"""Specificity scoring using named entity count and text length.

Posts that name specific tools, companies, or products are more
actionable than vague complaints. Longer posts also tend to be
more detailed and specific.

Score range: 0.0 to 10.0

Note: spaCy NER is optional. If the en_core_web_sm model isn't
installed, we fall back to a regex-based entity approximation.
"""

from __future__ import annotations

import re
from typing import Optional

from scopescrape.log import get_logger
from scopescrape.models import RawPost

logger = get_logger(__name__)

# Fallback: common tool/product name patterns
# Matches capitalized words that look like product names
PRODUCT_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|"  # CamelCase: ClickUp, GitHub
    r"[A-Z]{2,}(?:\.?[a-z]+)?|"            # ACRONYMS: AWS, APIs
    r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b"    # Title Case pairs: Google Docs
)


class SpecificityScorer:
    """Score posts by specificity of content (entities + length)."""

    def __init__(self, config: dict):
        self.config = config
        self._nlp = self._load_spacy()

    def _load_spacy(self):
        """Try to load spaCy model, return None if unavailable."""
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            logger.info(
                "spaCy en_core_web_sm not found. Using regex fallback for entity extraction. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            return None

    def score(self, post: RawPost) -> float:
        """Calculate specificity from entity count and text length.

        Args:
            post: The post to score.

        Returns:
            Score between 0.0 and 10.0.
        """
        text = post.full_text
        if not text:
            return 0.0

        # Component 1: Entity count (60% weight)
        entities = self.extract_entities(text)
        # 0 entities = 0, 1-2 = moderate, 3+ = high
        entity_score = min(10.0, len(entities) * 2.5)

        # Component 2: Text length (40% weight)
        # Short posts (<50 chars) = low, medium (50-500) = moderate, long (500+) = high
        length = len(text)
        if length < 50:
            length_score = 2.0
        elif length < 200:
            length_score = 5.0
        elif length < 500:
            length_score = 7.0
        else:
            length_score = min(10.0, 7.0 + (length - 500) / 500)

        combined = (entity_score * 0.6) + (length_score * 0.4)
        return round(min(10.0, max(0.0, combined)), 3)

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities (products, organizations, tools).

        Uses spaCy if available, falls back to regex pattern matching.

        Returns:
            Deduplicated list of entity strings.
        """
        if not text:
            return []

        if self._nlp is not None:
            return self._extract_spacy(text)
        return self._extract_regex(text)

    def _extract_spacy(self, text: str) -> list[str]:
        """Extract entities using spaCy NER."""
        doc = self._nlp(text[:5000])  # Cap length for performance
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "FAC"):
                entities.add(ent.text.strip())
        return sorted(entities)

    def _extract_regex(self, text: str) -> list[str]:
        """Fallback entity extraction using regex patterns."""
        matches = PRODUCT_PATTERN.findall(text)
        # Filter out common false positives
        stopwords = {
            "The", "This", "That", "What", "When", "Where", "Which",
            "How", "Why", "Who", "Not", "But", "And", "For", "Are",
            "Was", "Has", "Had", "Have", "Can", "Will", "Would",
            "Should", "Could", "May", "Also", "Just", "Some", "Any",
            "All", "Each", "Every", "Most", "Many", "Few", "More",
        }
        entities = set()
        for match in matches:
            if match not in stopwords and len(match) > 1:
                entities.add(match)
        return sorted(entities)
