"""Frequency scoring using intra-corpus signal phrase overlap.

Measures how many signal-related terms are shared across the corpus.
Posts that echo recurring pain themes score higher than one-off complaints.
This is done by extracting signal phrases from each post, counting phrase
frequency across the corpus, and scoring each post based on how many
high-frequency phrases it contains.

Score range: 0.0 to 10.0
"""

from __future__ import annotations

from collections import Counter

from scopescrape.log import get_logger
from scopescrape.models import RawPost

logger = get_logger(__name__)


class FrequencyScorer:
    """Score posts by signal phrase frequency within the corpus."""

    def __init__(self, config: dict):
        self.config = config
        self._signal_phrase_frequencies: Counter = Counter()
        self._corpus_size: int = 0

    def build_index(self, corpus: list[RawPost]):
        """Build signal phrase frequency index from the corpus.

        Extracts signal phrases from each post and counts their
        frequency across all posts. Should be called once with the
        full corpus before scoring individual posts.
        """
        self._corpus_size = len(corpus)
        phrase_counts: Counter = Counter()

        for post in corpus:
            # Extract signal phrases from the post
            signal_phrases = self._extract_signal_phrases(post.full_text)
            for phrase in signal_phrases:
                phrase_counts[phrase.lower()] += 1

        self._signal_phrase_frequencies = phrase_counts
        logger.debug(
            f"Signal phrase frequency index built from {len(corpus)} posts "
            f"with {len(phrase_counts)} unique signal phrases"
        )

    def score(self, post: RawPost) -> float:
        """Score a post based on signal phrase frequency overlap.

        Extracts signal phrases from the post and counts how many
        appear frequently in the corpus. Posts with more high-frequency
        phrases score higher, indicating they echo a recurring theme.

        Returns:
            Score between 0.0 and 10.0.
        """
        if not self._signal_phrase_frequencies:
            return 5.0  # neutral if no index built

        # Extract signal phrases from this post
        post_phrases = self._extract_signal_phrases(post.full_text)
        if not post_phrases:
            return 0.0

        # Count how many of this post's phrases appear frequently in corpus
        # Weight by the frequency: phrases that appear in more posts are worth more
        total_frequency_weight = 0.0
        max_possible_weight = 0.0

        for phrase in set(post_phrases):  # deduplicate within post
            phrase_lower = phrase.lower()
            # Get frequency of this phrase in the corpus
            frequency = self._signal_phrase_frequencies.get(phrase_lower, 0)
            # Weight it: appears in 2+ posts counts as a strong signal
            if frequency >= 2:
                total_frequency_weight += frequency
            max_possible_weight += 1

        if max_possible_weight == 0:
            return 0.0

        # Normalize to 0-10 scale
        # Average frequency across phrases, then scale to 0-10
        # If a post's phrases appear ~2x on average, score = 5.0
        # If they appear ~4x on average, score = 10.0
        avg_frequency = total_frequency_weight / max_possible_weight if max_possible_weight > 0 else 0.0
        # Map: 1.0 freq -> 0.0, 2.0 freq -> 5.0, 4.0+ freq -> 10.0
        normalized = min(10.0, max(0.0, (avg_frequency - 1.0) * 5.0))
        return round(normalized, 3)

    @staticmethod
    def _extract_signal_phrases(text: str) -> list[str]:
        """Extract signal-related terms from text.

        Looks for common pain point keywords and phrases that indicate
        frustration, bugs, feature requests, etc.

        Returns:
            List of signal phrase terms found in the text.
        """
        if not text:
            return []

        text_lower = text.lower()
        # Common signal words/phrases related to pain points
        signal_keywords = [
            "frustrated", "frustrating", "struggling", "struggling",
            "pain point", "dealbreaker", "can't stand", "driving me crazy",
            "broken", "crashing", "doesn't work", "not working", "buggy",
            "unreliable", "waste of time", "too slow", "takes forever",
            "too complicated", "clunky", "bloated", "unusable",
            "hate", "terrible", "awful", "worst", "horrible", "nightmare",
            "wish", "if only", "missing", "feature request",
            "alternative", "switched", "ditched", "moved from",
            "how to", "anyone know", "any recommendations", "looking for",
        ]

        found_phrases = []
        for keyword in signal_keywords:
            if keyword in text_lower:
                found_phrases.append(keyword)

        return found_phrases
