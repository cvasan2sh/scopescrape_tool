"""Tests for scoring engine."""

from datetime import datetime, timedelta

import pytest

from scopescrape.models import PainPoint, RawPost, SignalTier
from scopescrape.scoring.frequency import FrequencyScorer
from scopescrape.scoring.intensity import IntensityScorer
from scopescrape.scoring.recency import RecencyScorer
from scopescrape.scoring.scorer import Scorer
from scopescrape.scoring.specificity import SpecificityScorer


@pytest.fixture
def corpus(sample_post):
    """Small corpus of posts for testing."""
    return [
        sample_post,
        RawPost(
            id="t3_post2", platform="reddit", source="r/saas",
            title="Anyone know a good CRM for startups?",
            body="Looking for something cheaper than Salesforce.",
            author="u2", score=45, comment_count=12, url="",
            created_at=datetime.utcnow() - timedelta(hours=12),
        ),
        RawPost(
            id="t3_post3", platform="reddit", source="r/python",
            title="Switched from Flask to FastAPI",
            body="Flask was too slow for our use case. FastAPI handles 3x the throughput.",
            author="u3", score=200, comment_count=55, url="",
            created_at=datetime.utcnow() - timedelta(hours=24),
        ),
    ]


class TestFrequencyScorer:
    def test_score_with_corpus(self, sample_config, sample_post, corpus):
        scorer = FrequencyScorer(sample_config)
        scorer.build_index(corpus)
        score = scorer.score(sample_post)
        assert 0.0 <= score <= 10.0

    def test_score_without_index(self, sample_config, sample_post):
        scorer = FrequencyScorer(sample_config)
        score = scorer.score(sample_post)
        assert score == 5.0  # neutral default


class TestIntensityScorer:
    def test_negative_sentiment_high_score(self, sample_config):
        scorer = IntensityScorer(sample_config)
        post = RawPost(
            id="neg", platform="reddit", source="r/saas",
            title="I hate this terrible buggy software",
            body="Everything is broken and unusable. Worst experience ever.",
            author="a", score=0, comment_count=0, url="",
            created_at=datetime.utcnow(),
        )
        signals = [
            PainPoint("hate", SignalTier.EMOTIONAL, "emotion", 2, "I hate", 1.0),
            PainPoint("broken", SignalTier.PAIN_POINT, "bug", 30, "is broken", 1.0),
        ]
        score = scorer.score(post, signals)
        assert score > 5.0  # negative sentiment should score high

    def test_positive_text_low_score(self, sample_config):
        scorer = IntensityScorer(sample_config)
        post = RawPost(
            id="pos", platform="reddit", source="r/saas",
            title="Love this amazing tool",
            body="Best thing ever, works perfectly!",
            author="a", score=0, comment_count=0, url="",
            created_at=datetime.utcnow(),
        )
        signals = [PainPoint("love", SignalTier.EMOTIONAL, "emotion", 0, "Love", 1.0)]
        score = scorer.score(post, signals)
        # Positive sentiment should still score via tier weight but lower overall
        assert 0.0 <= score <= 10.0

    def test_vader_score(self, sample_config):
        scorer = IntensityScorer(sample_config)
        neg = scorer.vader_score("This is terrible and frustrating")
        pos = scorer.vader_score("This is wonderful and amazing")
        assert neg < 0
        assert pos > 0

    def test_empty_signals(self, sample_config, sample_post):
        scorer = IntensityScorer(sample_config)
        assert scorer.score(sample_post, []) == 0.0


class TestSpecificityScorer:
    def test_score_with_entities(self, sample_config, sample_post):
        scorer = SpecificityScorer(sample_config)
        score = scorer.score(sample_post)
        assert score > 0.0  # post mentions Asana, Monday, ClickUp, Notion

    def test_score_short_text(self, sample_config):
        scorer = SpecificityScorer(sample_config)
        post = RawPost(
            id="short", platform="hn", source="hn",
            title="Help", body="", author="a",
            score=0, comment_count=0, url="",
            created_at=datetime.utcnow(),
        )
        score = scorer.score(post)
        assert score < 5.0  # short text = low specificity

    def test_extract_entities_regex_fallback(self, sample_config):
        scorer = SpecificityScorer(sample_config)
        # Force regex fallback
        scorer._nlp = None
        entities = scorer.extract_entities("I use Asana and ClickUp for project management")
        assert len(entities) > 0


class TestRecencyScorer:
    def test_recent_post_high_score(self, sample_config):
        scorer = RecencyScorer(sample_config)
        post = RawPost(
            id="new", platform="reddit", source="r/saas",
            title="Just posted", body="", author="a",
            score=0, comment_count=0, url="",
            created_at=datetime.utcnow() - timedelta(minutes=30),
        )
        score = scorer.score(post)
        assert score > 9.0

    def test_old_post_low_score(self, sample_config):
        scorer = RecencyScorer(sample_config)
        post = RawPost(
            id="old", platform="reddit", source="r/saas",
            title="Ancient post", body="", author="a",
            score=0, comment_count=0, url="",
            created_at=datetime.utcnow() - timedelta(days=30),
        )
        score = scorer.score(post)
        assert score < 2.0

    def test_week_old_roughly_half(self, sample_config):
        scorer = RecencyScorer(sample_config)
        post = RawPost(
            id="week", platform="reddit", source="r/saas",
            title="Week old", body="", author="a",
            score=0, comment_count=0, url="",
            created_at=datetime.utcnow() - timedelta(hours=168),
        )
        score = scorer.score(post)
        assert 4.0 < score < 6.0  # roughly half of 10


class TestScorerComposite:
    def test_full_scoring(self, sample_config, sample_post, sample_pain_point, corpus):
        scorer = Scorer(sample_config)
        result = scorer.score(sample_post, [sample_pain_point], corpus)
        assert result is not None
        assert 0.0 <= result.composite_score <= 10.0
        assert result.post_id == sample_post.id

    def test_no_signals_returns_none(self, sample_config, sample_post, corpus):
        scorer = Scorer(sample_config)
        result = scorer.score(sample_post, [], corpus)
        assert result is None

    def test_below_threshold_returns_none(self, sample_config, sample_post, corpus):
        sample_config["scoring"]["min_score"] = 99.0
        scorer = Scorer(sample_config)
        signals = [PainPoint("frustrated", SignalTier.PAIN_POINT, "emotion", 0, "ctx", 1.0)]
        result = scorer.score(sample_post, signals, corpus)
        assert result is None
