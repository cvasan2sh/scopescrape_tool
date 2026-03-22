"""Signal phrase detector for pain point discovery.

Scans text for 60+ signal phrases organized into 4 tiers:
  1. PAIN_POINT  - explicit frustration ("frustrated with", "broken", "bug")
  2. EMOTIONAL   - strong feeling ("hate", "wish", "pray", "love")
  3. COMPARISON  - tool evaluation ("vs", "alternative to", "switched from")
  4. ASK         - seeking help ("how to", "anyone know", "is there a way")

Each match includes the surrounding context (50 chars before/after)
and metadata about which tier and category the phrase belongs to.
"""

from __future__ import annotations

import re
from typing import Optional

from scopescrape.log import get_logger
from scopescrape.models import PainPoint, SignalTier
from scopescrape.utils import extract_context

logger = get_logger(__name__)


# ─── Signal Phrase Taxonomy ───
# Each entry: (phrase, category)
# Categories: emotion, workflow, bug, feature_request, tool_eval, migration, help_seeking

SIGNAL_PHRASES: dict[SignalTier, list[tuple[str, str]]] = {
    SignalTier.PAIN_POINT: [
        # Explicit frustration
        ("frustrated with", "emotion"),
        ("frustrating", "emotion"),
        ("i'm struggling", "emotion"),
        ("i struggle with", "emotion"),
        ("struggling to", "emotion"),
        ("pain point", "emotion"),
        ("dealbreaker", "emotion"),
        ("deal breaker", "emotion"),
        ("can't stand", "emotion"),
        ("driving me crazy", "emotion"),
        ("pulling my hair out", "emotion"),
        ("at my wit's end", "emotion"),
        # Bugs and breakage
        ("broken", "bug"),
        ("keeps crashing", "bug"),
        ("crashes constantly", "bug"),
        ("doesn't work", "bug"),
        ("stopped working", "bug"),
        ("not working", "bug"),
        ("buggy", "bug"),
        ("unreliable", "bug"),
        ("constantly breaks", "bug"),
        # Workflow friction
        ("waste of time", "workflow"),
        ("wasting my time", "workflow"),
        ("too slow", "workflow"),
        ("takes forever", "workflow"),
        ("too complicated", "workflow"),
        ("overly complex", "workflow"),
        ("clunky", "workflow"),
        ("bloated", "workflow"),
        ("unusable", "workflow"),
    ],
    SignalTier.EMOTIONAL: [
        # Negative
        ("i hate", "emotion"),
        ("i despise", "emotion"),
        ("terrible", "emotion"),
        ("awful", "emotion"),
        ("worst", "emotion"),
        ("horrible", "emotion"),
        ("nightmare", "emotion"),
        ("regret", "emotion"),
        # Desire / unmet need
        ("i wish", "feature_request"),
        ("if only", "feature_request"),
        ("would be nice if", "feature_request"),
        ("why can't", "feature_request"),
        ("why doesn't", "feature_request"),
        ("should be able to", "feature_request"),
        ("needs to support", "feature_request"),
        ("missing feature", "feature_request"),
        ("desperately need", "feature_request"),
        ("pray", "emotion"),
    ],
    SignalTier.COMPARISON: [
        # Direct comparison
        ("vs", "tool_eval"),
        ("versus", "tool_eval"),
        ("compared to", "tool_eval"),
        ("better than", "tool_eval"),
        ("worse than", "tool_eval"),
        ("alternative to", "tool_eval"),
        ("alternatives to", "tool_eval"),
        ("replacement for", "tool_eval"),
        ("instead of", "tool_eval"),
        # Migration
        ("switched from", "migration"),
        ("switching from", "migration"),
        ("moved from", "migration"),
        ("moving from", "migration"),
        ("migrated from", "migration"),
        ("migrating from", "migration"),
        ("ditched", "migration"),
        ("dropped", "migration"),
    ],
    SignalTier.ASK: [
        # Help seeking
        ("how to", "help_seeking"),
        ("how do i", "help_seeking"),
        ("how do you", "help_seeking"),
        ("anyone know", "help_seeking"),
        ("does anyone know", "help_seeking"),
        ("any recommendations", "help_seeking"),
        ("can someone recommend", "help_seeking"),
        ("what do you use for", "help_seeking"),
        ("what tool", "help_seeking"),
        ("is there a way", "help_seeking"),
        ("is there a tool", "help_seeking"),
        ("looking for", "help_seeking"),
        ("looking for a tool", "help_seeking"),
        ("need a tool", "help_seeking"),
        ("anyone recommend", "help_seeking"),
        ("suggestions for", "help_seeking"),
        ("best way to", "help_seeking"),
        ("any good", "help_seeking"),
    ],
}


class SignalDetector:
    """Detect signal phrases in text and extract surrounding context."""

    # Patterns for self-promotion detection
    SELF_PROMOTION_TRIGGER = r"\bi\s+(?:built|made|created|designed|developed|founded|launched|built)"
    SELF_PROMOTION_CTA = r"(?:check\s+it\s+out|happy\s+to\s+share|beta|launching|sign\s+up|try\s+it|give\s+it\s+a\s+try|interested)"

    def __init__(self, config: dict, context_window: int = 50):
        self.config = config
        self.context_window = context_window
        self.patterns = self._compile_patterns()
        # Compile self-promotion detection patterns
        self._promo_trigger_pattern = re.compile(self.SELF_PROMOTION_TRIGGER, re.IGNORECASE)
        self._promo_cta_pattern = re.compile(self.SELF_PROMOTION_CTA, re.IGNORECASE)
        logger.debug(
            f"SignalDetector loaded: {sum(len(v) for v in SIGNAL_PHRASES.values())} phrases "
            f"across {len(SIGNAL_PHRASES)} tiers"
        )

    def _compile_patterns(self) -> dict[SignalTier, list[tuple[re.Pattern, str]]]:
        """Compile regex patterns with word boundaries for each phrase.

        Returns a dict mapping each tier to a list of (compiled_pattern, category).
        """
        compiled = {}
        for tier, phrases in SIGNAL_PHRASES.items():
            tier_patterns = []
            for phrase, category in phrases:
                # Word boundary matching, case insensitive
                pattern = re.compile(
                    r"\b" + re.escape(phrase) + r"\b",
                    re.IGNORECASE,
                )
                tier_patterns.append((pattern, category))
            compiled[tier] = tier_patterns
        return compiled

    def detect(self, text: str, post_id: str = "") -> list[PainPoint]:
        """Find all signal phrases in the given text.

        Applies self-promotion detection: if the post appears to be
        promotional (e.g. "I built X, check it out"), reduces confidence
        of all signals by 50% to account for bias.

        Args:
            text: The full text to scan (title + body typically).
            post_id: Optional identifier for logging.

        Returns:
            List of PainPoint objects, one per match. Empty if no signals found.
        """
        if not text or not text.strip():
            return []

        results: list[PainPoint] = []
        seen_positions: set[tuple[int, str]] = set()  # (position, phrase) dedup

        # Check if this post appears to be promotional
        is_promotional = self._is_promotional(text)

        for tier, patterns in self.patterns.items():
            for pattern, category in patterns:
                for match in pattern.finditer(text):
                    phrase = match.group(0)
                    position = match.start()

                    # Skip duplicate matches at the same position
                    key = (position, phrase.lower())
                    if key in seen_positions:
                        continue
                    seen_positions.add(key)

                    context = extract_context(
                        text, position, len(phrase), self.context_window
                    )

                    # Reduce confidence by 50% if post is promotional
                    confidence = 1.0
                    if is_promotional:
                        confidence = 0.5

                    results.append(PainPoint(
                        phrase=phrase,
                        tier=tier,
                        category=category,
                        position=position,
                        context=context,
                        confidence=confidence,
                    ))

        if results:
            logger.debug(f"Post {post_id}: {len(results)} signals detected" +
                        (f" (promotional, confidence halved)" if is_promotional else ""))

        return results

    def _is_promotional(self, text: str) -> bool:
        """Detect if a post appears to be self-promotional.

        Returns True if the post matches the pattern:
        "I built/created/made X" + (call-to-action phrases like
        "check it out", "happy to share", "beta", "launching", "sign up")

        Args:
            text: The text to check.

        Returns:
            True if the post appears promotional, False otherwise.
        """
        # Check if text contains self-promotion trigger phrase
        if not self._promo_trigger_pattern.search(text):
            return False

        # Check if text also contains a call-to-action phrase
        if self._promo_cta_pattern.search(text):
            logger.debug("Detected self-promotional post (I built + CTA)")
            return True

        return False

    def detect_batch(self, posts: list, text_attr: str = "full_text") -> dict[str, list[PainPoint]]:
        """Run detection across a list of posts.

        Args:
            posts: List of RawPost objects.
            text_attr: Attribute name to get text from (default "full_text").

        Returns:
            Dict mapping post_id -> list of PainPoint for posts with signals.
        """
        signals_by_post: dict[str, list[PainPoint]] = {}

        for post in posts:
            text = getattr(post, text_attr, "")
            if not text:
                continue

            signals = self.detect(text, post.id)
            if signals:
                signals_by_post[post.id] = signals

        logger.info(
            f"Signal detection: {len(signals_by_post)} posts with signals "
            f"out of {len(posts)} total"
        )
        return signals_by_post

    @staticmethod
    def get_phrase_count() -> int:
        """Return total number of signal phrases across all tiers."""
        return sum(len(phrases) for phrases in SIGNAL_PHRASES.values())

    @staticmethod
    def get_tier_summary() -> dict[str, int]:
        """Return count of phrases per tier."""
        return {tier.name: len(phrases) for tier, phrases in SIGNAL_PHRASES.items()}
