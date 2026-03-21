"""Shared utility functions for ScopeScrape."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Optional


class RateLimiter:
    """Rate limiter with exponential backoff for API calls."""

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 5,
    ):
        self.initial_delay = initial_delay
        self.delay = initial_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.retry_count = 0

    def wait(self):
        """Sleep for the current delay duration."""
        time.sleep(self.delay)

    def on_failure(self):
        """Apply exponential backoff. Raises RuntimeError if max retries exceeded."""
        self.retry_count += 1
        if self.retry_count > self.max_retries:
            raise RuntimeError(
                f"Max retries ({self.max_retries}) exceeded after {self.retry_count} attempts"
            )
        self.delay = min(self.delay * 2, self.max_delay)

    def on_success(self):
        """Reset backoff to initial state."""
        self.delay = self.initial_delay
        self.retry_count = 0


def deduplicate_posts(posts: list, key: str = "id") -> list:
    """Remove duplicate posts by a key field.

    Args:
        posts: List of post objects (dataclasses or dicts).
        key: Attribute or dict key to deduplicate on.

    Returns:
        Deduplicated list preserving original order.
    """
    seen = set()
    unique = []

    for post in posts:
        val = getattr(post, key, None) if hasattr(post, key) else post.get(key)
        if val and val not in seen:
            seen.add(val)
            unique.append(post)

    return unique


def truncate(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if shortened."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_context(text: str, position: int, phrase_length: int, window: int = 50) -> str:
    """Extract a context window around a match position.

    Args:
        text: Full text to extract from.
        position: Start index of the matched phrase.
        phrase_length: Length of the matched phrase.
        window: Characters to include before and after.

    Returns:
        Context string.
    """
    start = max(0, position - window)
    end = min(len(text), position + phrase_length + window)
    return text[start:end]


def generate_post_hash(platform: str, post_id: str) -> str:
    """Generate a deterministic hash for a post across platforms."""
    raw = f"{platform}:{post_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def time_decay(created_at: datetime, half_life_hours: float = 168.0) -> float:
    """Calculate a recency score using exponential decay.

    Score ranges from 0.0 (very old) to 10.0 (just created).

    Args:
        created_at: When the post was created.
        half_life_hours: Hours for score to halve. Default 168 (1 week).

    Returns:
        Recency score between 0.0 and 10.0.
    """
    import math

    age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
    if age_hours < 0:
        age_hours = 0

    decay = math.exp(-0.693 * age_hours / half_life_hours)  # ln(2) = 0.693
    return round(decay * 10.0, 3)


def safe_get(obj: object, attr: str, default: Optional[str] = None):
    """Safely get an attribute from an object (useful for PRAW lazy objects).

    PRAW objects load attributes lazily and may raise exceptions
    for deleted or removed content. This wraps getattr with
    a try/except fallback.
    """
    try:
        value = getattr(obj, attr, default)
        # PRAW sometimes returns None for deleted content attributes
        return value if value is not None else default
    except Exception:
        return default
