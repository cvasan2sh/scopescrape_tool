"""Tests for data models."""

from datetime import datetime, timedelta

from scopescrape.models import PainPoint, RawPost, ScoredResult, SignalTier


class TestSignalTier:
    def test_tier_values(self):
        assert SignalTier.PAIN_POINT.value == 1
        assert SignalTier.EMOTIONAL.value == 2
        assert SignalTier.COMPARISON.value == 3
        assert SignalTier.ASK.value == 4

    def test_tier_weights(self):
        assert SignalTier.PAIN_POINT.weight == 1.0
        assert SignalTier.ASK.weight == 0.4

    def test_tier_ordering(self):
        """PAIN_POINT should have highest weight, ASK lowest."""
        tiers = list(SignalTier)
        weights = [t.weight for t in tiers]
        assert weights == sorted(weights, reverse=True)


class TestRawPost:
    def test_create_basic_post(self, sample_post):
        assert sample_post.id == "t3_abc123"
        assert sample_post.platform == "reddit"
        assert sample_post.source == "r/saas"
        assert sample_post.score == 127

    def test_full_text_combines_title_and_body(self, sample_post):
        text = sample_post.full_text
        assert sample_post.title in text
        assert sample_post.body in text

    def test_full_text_empty_body(self):
        post = RawPost(
            id="1", platform="hn", source="hn", title="Test",
            body="", author="a", score=0, comment_count=0,
            url="", created_at=datetime.utcnow(),
        )
        assert post.full_text == "Test"

    def test_age_hours(self, sample_post):
        age = sample_post.age_hours
        assert 5.0 < age < 7.0  # created 6 hours ago

    def test_default_optional_fields(self):
        post = RawPost(
            id="1", platform="hn", source="hn", title="T",
            body="B", author="a", score=1, comment_count=0,
            url="", created_at=datetime.utcnow(),
        )
        assert post.parent_id is None
        assert post.children_ids == []


class TestPainPoint:
    def test_create_pain_point(self, sample_pain_point):
        assert sample_pain_point.phrase == "frustrated with"
        assert sample_pain_point.tier == SignalTier.PAIN_POINT
        assert sample_pain_point.confidence == 1.0

    def test_context_display(self):
        pp = PainPoint(
            phrase="struggling",
            tier=SignalTier.PAIN_POINT,
            category="emotion",
            position=10,
            context="I am struggling with this API",
            confidence=1.0,
        )
        display = pp.context_display
        assert "[struggling]" in display
        assert "I am" in display


class TestScoredResult:
    def test_create_scored_result(self, sample_scored_result):
        assert sample_scored_result.composite_score == 7.83
        assert len(sample_scored_result.signal_phrases) == 1
        assert len(sample_scored_result.entities) == 4

    def test_to_dict(self, sample_scored_result):
        d = sample_scored_result.to_dict()
        assert d["post_id"] == "t3_abc123"
        assert d["composite_score"] == 7.83
        assert d["signal_count"] == 1
        assert isinstance(d["created_at"], str)
        assert isinstance(d["signal_phrases"], list)
        assert d["signal_phrases"][0]["tier"] == "PAIN_POINT"

    def test_to_dict_empty_signals(self, sample_post):
        result = ScoredResult(
            post_id=sample_post.id,
            platform="reddit",
            title=sample_post.title,
            body_excerpt="",
            author="a",
            source="r/saas",
            created_at=datetime.utcnow(),
            url="",
            frequency_score=5.0,
            intensity_score=5.0,
            specificity_score=5.0,
            recency_score=5.0,
            composite_score=5.0,
        )
        d = result.to_dict()
        assert d["signal_count"] == 0
        assert d["signal_phrases"] == []
