"""Tests for SQLite storage layer."""

from datetime import datetime, timedelta

import pytest

from scopescrape.models import PainPoint, RawPost, ScoredResult, SignalTier
from scopescrape.storage import Storage


@pytest.fixture
def storage(sample_config):
    """In-memory storage instance for testing."""
    s = Storage(sample_config)
    yield s
    s.close()


@pytest.fixture
def old_post():
    """A post created 72 hours ago (older than 48h retention)."""
    return RawPost(
        id="t3_old999",
        platform="reddit",
        source="r/python",
        title="Old post about legacy code",
        body="This is ancient content that should be cleaned up.",
        author="old_user",
        score=5,
        comment_count=1,
        url="https://reddit.com/r/python/comments/old999/",
        created_at=datetime.utcnow() - timedelta(hours=72),
    )


class TestStorageInit:
    def test_creates_tables(self, storage):
        """Tables should exist after init."""
        tables = storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {row[0] for row in tables}
        assert "posts" in table_names
        assert "signals" in table_names
        assert "scores" in table_names

    def test_empty_on_init(self, storage):
        assert storage.count_posts() == 0
        assert storage.count_signals() == 0


class TestSavePosts:
    def test_insert_single_post(self, storage, sample_post):
        count = storage.save_posts([sample_post])
        assert count == 1
        assert storage.count_posts() == 1

    def test_deduplication(self, storage, sample_post):
        storage.save_posts([sample_post])
        count = storage.save_posts([sample_post])
        assert count == 0  # duplicate skipped
        assert storage.count_posts() == 1

    def test_insert_multiple(self, storage, sample_post, old_post):
        count = storage.save_posts([sample_post, old_post])
        assert count == 2
        assert storage.count_posts() == 2


class TestSaveSignals:
    def test_insert_signals(self, storage, sample_post, sample_pain_point):
        storage.save_posts([sample_post])
        count = storage.save_signals({sample_post.id: [sample_pain_point]})
        assert count == 1
        assert storage.count_signals() == 1

    def test_multiple_signals_per_post(self, storage, sample_post):
        storage.save_posts([sample_post])
        signals = [
            PainPoint("frustrated", SignalTier.PAIN_POINT, "emotion", 0, "ctx", 1.0),
            PainPoint("alternative to", SignalTier.COMPARISON, "tool", 50, "ctx2", 1.0),
        ]
        count = storage.save_signals({sample_post.id: signals})
        assert count == 2


class TestSaveScores:
    def test_insert_score(self, storage, sample_post, sample_scored_result):
        storage.save_posts([sample_post])
        count = storage.save_scores([sample_scored_result])
        assert count == 1

    def test_upsert_score(self, storage, sample_post, sample_scored_result):
        storage.save_posts([sample_post])
        storage.save_scores([sample_scored_result])
        # Update the score
        sample_scored_result.composite_score = 9.0
        count = storage.save_scores([sample_scored_result])
        assert count == 1  # replaced


class TestQueryResults:
    def test_query_returns_results(self, storage, sample_post, sample_scored_result):
        storage.save_posts([sample_post])
        storage.save_scores([sample_scored_result])
        results = storage.query_results(min_score=5.0)
        assert len(results) == 1
        assert results[0]["composite_score"] == 7.83

    def test_query_filters_by_min_score(self, storage, sample_post, sample_scored_result):
        storage.save_posts([sample_post])
        storage.save_scores([sample_scored_result])
        results = storage.query_results(min_score=9.0)
        assert len(results) == 0

    def test_query_respects_limit(self, storage):
        # Insert 5 posts with scores
        for i in range(5):
            post = RawPost(
                id=f"t3_{i}", platform="reddit", source="r/test",
                title=f"Post {i}", body="body", author="a", score=i,
                comment_count=0, url="", created_at=datetime.utcnow(),
            )
            storage.save_posts([post])
            storage.save_scores([ScoredResult(
                post_id=f"t3_{i}", platform="reddit", title=f"Post {i}",
                body_excerpt="body", author="a", source="r/test",
                created_at=datetime.utcnow(), url="",
                frequency_score=5.0, intensity_score=5.0,
                specificity_score=5.0, recency_score=5.0,
                composite_score=5.0 + i,
            )])

        results = storage.query_results(min_score=0.0, limit=3)
        assert len(results) == 3


class TestCleanup:
    def test_cleanup_removes_old_posts(self, storage, old_post):
        storage.save_posts([old_post])
        assert storage.count_posts() == 1
        deleted = storage.cleanup_old()
        assert deleted == 1
        assert storage.count_posts() == 0

    def test_cleanup_keeps_recent(self, storage, sample_post):
        storage.save_posts([sample_post])
        deleted = storage.cleanup_old()
        assert deleted == 0
        assert storage.count_posts() == 1


class TestPostExists:
    def test_existing_post(self, storage, sample_post):
        storage.save_posts([sample_post])
        assert storage.post_exists(sample_post.id) is True

    def test_nonexistent_post(self, storage):
        assert storage.post_exists("fake_id") is False
