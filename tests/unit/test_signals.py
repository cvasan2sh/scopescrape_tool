"""Tests for signal phrase detection."""

from scopescrape.models import SignalTier
from scopescrape.signals.detector import SIGNAL_PHRASES, SignalDetector


class TestSignalDetector:
    def setup_method(self):
        self.detector = SignalDetector({})

    def test_phrase_count(self):
        count = SignalDetector.get_phrase_count()
        assert count >= 60, f"Expected 60+ phrases, got {count}"

    def test_tier_summary(self):
        summary = SignalDetector.get_tier_summary()
        assert "PAIN_POINT" in summary
        assert "EMOTIONAL" in summary
        assert "COMPARISON" in summary
        assert "ASK" in summary

    def test_detect_pain_point(self):
        text = "I'm frustrated with every project management tool on the market"
        signals = self.detector.detect(text)
        assert len(signals) >= 1
        phrases = [s.phrase.lower() for s in signals]
        assert "frustrated with" in phrases
        assert signals[0].tier == SignalTier.PAIN_POINT

    def test_detect_emotional(self):
        text = "I wish there was a tool that just works for small teams"
        signals = self.detector.detect(text)
        phrases = [s.phrase.lower() for s in signals]
        assert "i wish" in phrases

    def test_detect_comparison(self):
        text = "Looking for an alternative to Notion that handles databases better"
        signals = self.detector.detect(text)
        phrases = [s.phrase.lower() for s in signals]
        assert "alternative to" in phrases

    def test_detect_ask(self):
        text = "Does anyone know a good tool for scraping Reddit?"
        signals = self.detector.detect(text)
        phrases = [s.phrase.lower() for s in signals]
        assert "does anyone know" in phrases

    def test_detect_multiple_signals(self):
        text = (
            "I'm frustrated with Notion. It's too slow and crashes constantly. "
            "Is there a tool that's a good alternative to it?"
        )
        signals = self.detector.detect(text)
        assert len(signals) >= 3
        tiers = {s.tier for s in signals}
        assert SignalTier.PAIN_POINT in tiers

    def test_detect_case_insensitive(self):
        text = "FRUSTRATED WITH this buggy software"
        signals = self.detector.detect(text)
        assert len(signals) >= 1

    def test_detect_empty_text(self):
        assert self.detector.detect("") == []
        assert self.detector.detect("   ") == []

    def test_detect_no_signals(self):
        text = "The weather is nice today and I had a good lunch."
        signals = self.detector.detect(text)
        assert len(signals) == 0

    def test_context_extraction(self):
        text = "After months of use, I'm frustrated with ClickUp's constant bugs"
        signals = self.detector.detect(text)
        assert len(signals) >= 1
        # Context should include surrounding text
        for s in signals:
            if s.phrase.lower() == "frustrated with":
                assert "ClickUp" in s.context or "months" in s.context

    def test_detect_batch(self, sample_post):
        signals_map = self.detector.detect_batch([sample_post])
        assert sample_post.id in signals_map
        assert len(signals_map[sample_post.id]) >= 1

    def test_word_boundary_matching(self):
        """'vs' should not match inside words like 'canvas' or 'obvious'."""
        text = "I used canvas for the design and it was obvious"
        signals = self.detector.detect(text)
        phrases = [s.phrase.lower() for s in signals]
        assert "vs" not in phrases

    def test_promotional_post_reduces_confidence(self):
        """Detect if post is self-promotional and reduce signal confidence."""
        text = "I built a new tool that solves broken workflows. Check it out and sign up for beta!"
        signals = self.detector.detect(text)
        # Should detect "broken" as a signal
        assert len(signals) >= 1
        # But confidence should be reduced due to promotional nature
        for signal in signals:
            if "broken" in signal.phrase.lower():
                assert signal.confidence == 0.5, f"Expected confidence 0.5 for broken in promo, got {signal.confidence}"

    def test_non_promotional_post_full_confidence(self):
        """Non-promotional posts should have full confidence."""
        text = "I'm frustrated with every project management tool"
        signals = self.detector.detect(text)
        assert len(signals) >= 1
        # Should have full confidence
        for signal in signals:
            if "frustrated" in signal.phrase.lower():
                assert signal.confidence == 1.0

    def test_promotional_without_cta_not_flagged(self):
        """Posts with just 'I built' but no CTA shouldn't be flagged as promotional."""
        text = "I built a tool but it has broken features"
        signals = self.detector.detect(text)
        # Should detect "broken" with full confidence (no CTA present)
        for signal in signals:
            if "broken" in signal.phrase.lower():
                assert signal.confidence == 1.0

    def test_promotional_patterns(self):
        """Test various promotional post patterns."""
        # Each tuple is (text, should_have_reduced_confidence)
        # Note: texts need to contain actual signal phrases to generate signals
        texts = [
            ("I created a tool that's broken, check it out", True),  # Promo + signal
            ("I developed something frustrating and happy to share", True),  # Promo + signal
            ("I designed this but it's buggy and launching soon", True),  # Promo + signal
            ("I built this, sign up today. It's not working", True),  # Promo + signal
            ("This tool is broken", False),  # No "I built" trigger = no reduction
            ("I'm frustrated with this tool", False),  # No CTA = no reduction
        ]

        for text, should_have_reduced_confidence in texts:
            signals = self.detector.detect(text)
            # Check if any signal has reduced confidence (0.5)
            has_reduced_confidence = any(s.confidence == 0.5 for s in signals)
            assert has_reduced_confidence == should_have_reduced_confidence, \
                f"Text '{text}' has_reduced={has_reduced_confidence}, expected {should_have_reduced_confidence}"
