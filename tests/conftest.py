"""Shared pytest fixtures for ScopeScrape tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from scopescrape.models import PainPoint, RawPost, ScoredResult, SignalTier


@pytest.fixture
def sample_config():
    """Minimal valid config dict for testing."""
    return {
        "reddit": {
            "client_id": "test_id",
            "client_secret": "test_secret",
            "user_agent": "ScopeScrape Test v0.1",
            "rate_limit_delay": 0.0,
            "comment_depth": 3,
            "replace_more_limit": 10,
        },
        "hn": {
            "rate_limit_delay": 0.0,
            "comment_depth": 5,
        },
        "twitter": {
            "nitter_instance": "https://nitter.net",
            "fallback_instances": ["https://nitter.unixfederal.com"],
            "rate_limit_delay": 0.0,
        },
        "github": {
            "token": "",
            "rate_limit_delay": 0.0,
        },
        "stackoverflow": {
            "api_key": "",
            "rate_limit_delay": 0.0,
            "max_answers_per_question": 3,
        },
        "producthunt": {
            "token": "",
            "rate_limit_delay": 0.0,
            "max_reviews_per_product": 5,
        },
        "indiehackers": {
            "rate_limit_delay": 0.0,
        },
        "scoring": {
            "weights": {
                "frequency": 0.25,
                "intensity": 0.20,
                "specificity": 0.25,
                "recency": 0.30,
            },
            "min_score": 5.0,
        },
        "storage": {
            "db_path": ":memory:",
            "retention_hours": 48,
            "in_memory": True,
        },
        "scan": {
            "default_limit": 100,
            "default_time_range": "week",
            "default_platforms": ["reddit"],
        },
    }


@pytest.fixture
def sample_post():
    """A realistic RawPost for testing."""
    return RawPost(
        id="t3_abc123",
        platform="reddit",
        source="r/saas",
        title="I'm frustrated with every project management tool on the market",
        body=(
            "I've tried Asana, Monday, ClickUp, and Notion. They all suck at "
            "different things. Asana is too rigid, Monday is bloated, ClickUp "
            "crashes constantly, and Notion is too freeform. Is there anything "
            "that just works for a small team of 5?"
        ),
        author="startup_dev_42",
        score=127,
        comment_count=43,
        url="https://reddit.com/r/saas/comments/abc123/frustrated_with_pm_tools/",
        created_at=datetime.utcnow() - timedelta(hours=6),
    )


@pytest.fixture
def sample_pain_point():
    """A detected signal for testing."""
    return PainPoint(
        phrase="frustrated with",
        tier=SignalTier.PAIN_POINT,
        category="emotion",
        position=4,
        context="I'm [frustrated with] every project management tool",
        confidence=1.0,
    )


@pytest.fixture
def sample_scored_result(sample_post, sample_pain_point):
    """A scored result for testing."""
    return ScoredResult(
        post_id=sample_post.id,
        platform=sample_post.platform,
        title=sample_post.title,
        body_excerpt=sample_post.body[:200],
        author=sample_post.author,
        source=sample_post.source,
        created_at=sample_post.created_at,
        url=sample_post.url,
        frequency_score=7.2,
        intensity_score=8.5,
        specificity_score=6.8,
        recency_score=9.1,
        composite_score=7.83,
        signal_phrases=[sample_pain_point],
        sentiment_score=-0.65,
        entities=["Asana", "Monday", "ClickUp", "Notion"],
        text_length=len(sample_post.body),
    )
