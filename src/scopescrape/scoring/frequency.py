"""Frequency scoring using BM25 relevance.

Measures how relevant a post is to the overall corpus of fetched posts.
Posts that contain terms which appear frequently across the corpus get
higher frequency scores, indicating a recurring theme.

Score range: 0.0 to 10.0
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from scopescrape.log import get_logger
from scopescrape.models import RawPost

logger = get_logger(__name__)


class FrequencyScorer:
    """Score posts by BM25 relevance against the corpus."""

    def __init__(self, config: dict):
        self.config = config
        self._index: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    def build_index(self, corpus: list[RawPost]):
        """Build the BM25 index from all fetched posts.

        Should be called once with the full corpus before scoring
        individual posts.
        """
        self._corpus_tokens = [
            self._tokenize(post.full_text) for post in corpus
        ]

        if not self._corpus_tokens or all(len(t) == 0 for t in self._corpus_tokens):
            logger.warning("Empty corpus for BM25 index")
            self._index = None
            return

        self._index = BM25Okapi(self._corpus_tokens)
        logger.debug(f"BM25 index built with {len(self._corpus_tokens)} documents")

    def score(self, post: RawPost) -> float:
        """Score a single post's frequency relevance.

        Uses the post's signal phrases as the query against the
        BM25 index. Higher score means the post's pain-related
        terms appear frequently across the corpus.

        Returns:
            Score between 0.0 and 10.0.
        """
        if self._index is None:
            return 5.0  # neutral if no index

        query_tokens = self._tokenize(post.full_text)
        if not query_tokens:
            return 0.0

        scores = self._index.get_scores(query_tokens)
        if len(scores) == 0:
            return 0.0

        # Find this post's score in the corpus
        max_score = max(scores) if max(scores) > 0 else 1.0
        # Normalize: find which index this post is at
        # Use the post's own BM25 score relative to max
        post_score = scores[0]  # approximate with first
        for i, tokens in enumerate(self._corpus_tokens):
            if tokens == query_tokens:
                post_score = scores[i]
                break

        # Normalize to 0-10 scale
        normalized = (post_score / max_score) * 10.0 if max_score > 0 else 0.0
        return round(min(10.0, max(0.0, normalized)), 3)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing and minimum length."""
        if not text:
            return []
        return [
            word.lower().strip(".,!?;:\"'()[]{}")
            for word in text.split()
            if len(word) > 2
        ]
