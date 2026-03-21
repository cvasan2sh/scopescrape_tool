"""Tests for Reddit and HN adapters using mocked HTTP responses."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scopescrape.adapters.hackernews import HackerNewsAdapter
from scopescrape.adapters.reddit import RedditAdapter

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def reddit_fixtures():
    with open(FIXTURES / "reddit_responses.json") as f:
        return json.load(f)


@pytest.fixture
def hn_fixtures():
    with open(FIXTURES / "hn_responses.json") as f:
        return json.load(f)


@pytest.fixture
def reddit_adapter(sample_config):
    """Reddit adapter with rate limiting disabled for tests."""
    sample_config["reddit"]["rate_limit_delay"] = 0.0
    return RedditAdapter(sample_config)


@pytest.fixture
def hn_adapter(sample_config):
    """HN adapter with rate limiting disabled for tests."""
    sample_config["hn"]["rate_limit_delay"] = 0.0
    return HackerNewsAdapter(sample_config)


def _mock_response(data, status_code=200):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.headers = {}
    return resp


# ─── Reddit Adapter Tests ───


class TestRedditNormalization:
    def test_parse_listing(self, reddit_adapter, reddit_fixtures):
        data = reddit_fixtures["search_listing"]
        posts = reddit_adapter._parse_listing(data, source="r/saas")

        # Should skip the deleted post, return 2
        assert len(posts) == 2
        assert posts[0].id == "t3_abc123"
        assert posts[0].platform == "reddit"
        assert posts[0].source == "r/saas"
        assert posts[0].title == "I'm frustrated with every project management tool"
        assert posts[0].score == 127

    def test_deleted_posts_filtered(self, reddit_adapter, reddit_fixtures):
        data = reddit_fixtures["search_listing"]
        posts = reddit_adapter._parse_listing(data, source="r/saas")
        ids = [p.id for p in posts]
        assert "t3_deleted1" not in ids

    def test_normalize_post_basic(self, reddit_adapter):
        raw = {
            "id": "xyz789",
            "subreddit": "python",
            "title": "Test post",
            "selftext": "Body text",
            "author": "user1",
            "score": 10,
            "num_comments": 5,
            "permalink": "/r/python/comments/xyz789/test/",
            "created_utc": 1742630400,
        }
        post = reddit_adapter._normalize_post(raw)
        assert post is not None
        assert post.id == "t3_xyz789"
        assert post.body == "Body text"
        assert "reddit.com" in post.url

    def test_normalize_post_empty_data(self, reddit_adapter):
        assert reddit_adapter._normalize_post({}) is None
        assert reddit_adapter._normalize_post({"id": ""}) is None


class TestRedditCommentFlattening:
    def test_flatten_thread(self, reddit_adapter, reddit_fixtures):
        thread_data = reddit_fixtures["thread_response"]
        # Parse the comment tree from second listing
        children = thread_data[1]["data"]["children"]
        comments = reddit_adapter._flatten_comments(children, source="r/saas")

        # Should get 2 comments (comment1 + reply1), skip deleted and "more"
        assert len(comments) == 2
        assert comments[0].id == "t1_comment1"
        assert comments[1].id == "t1_reply1"
        assert comments[1].parent_id == "t1_comment1"

    def test_deleted_comments_skipped(self, reddit_adapter, reddit_fixtures):
        thread_data = reddit_fixtures["thread_response"]
        children = thread_data[1]["data"]["children"]
        comments = reddit_adapter._flatten_comments(children, source="r/saas")
        ids = [c.id for c in comments]
        assert "t1_deleted_comment" not in ids

    def test_depth_limit(self, reddit_adapter, reddit_fixtures):
        """With depth=0, no comments should be returned."""
        reddit_adapter.comment_depth = 0
        thread_data = reddit_fixtures["thread_response"]
        children = thread_data[1]["data"]["children"]
        comments = reddit_adapter._flatten_comments(children, source="r/saas")
        assert len(comments) == 0


class TestRedditFetch:
    @patch.object(RedditAdapter, "_get_json")
    def test_fetch_subreddits_with_keywords(self, mock_get, reddit_adapter, reddit_fixtures):
        mock_get.return_value = reddit_fixtures["search_listing"]

        queries = {
            "subreddits": ["saas"],
            "keywords": ["pain point"],
            "limit": 100,
            "time_range": "week",
        }
        posts = reddit_adapter.fetch(queries)

        assert len(posts) == 2  # deduped
        mock_get.assert_called_once()

    @patch.object(RedditAdapter, "_get_json")
    def test_fetch_subreddits_no_keywords(self, mock_get, reddit_adapter, reddit_fixtures):
        mock_get.return_value = reddit_fixtures["search_listing"]

        queries = {"subreddits": ["saas"], "limit": 50, "time_range": "week"}
        posts = reddit_adapter.fetch(queries)

        assert len(posts) == 2
        # Should have called _list_subreddit (hot listing) not search
        call_url = mock_get.call_args[0][0]
        assert "/hot.json" in call_url

    @patch.object(RedditAdapter, "_get_json")
    def test_fetch_empty_response(self, mock_get, reddit_adapter):
        mock_get.return_value = {"data": {"children": []}}

        posts = reddit_adapter.fetch({"subreddits": ["empty_sub"], "limit": 10})
        assert posts == []

    @patch.object(RedditAdapter, "_get_json")
    def test_fetch_thread(self, mock_get, reddit_adapter, reddit_fixtures):
        mock_get.return_value = reddit_fixtures["thread_response"]

        top_post, comments = reddit_adapter.fetch_thread("abc123", subreddit="saas")

        assert top_post is not None
        assert top_post.id == "t3_abc123"
        assert len(comments) == 2


class TestRedditRateLimit:
    @patch.object(RedditAdapter, "_create_session")
    def test_429_retry(self, mock_session_fn, sample_config):
        """Should retry on 429 and succeed on second attempt."""
        sample_config["reddit"]["rate_limit_delay"] = 0.0
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"data": {"children": []}}

        mock_session.get.side_effect = [resp_429, resp_200]

        adapter = RedditAdapter(sample_config)
        result = adapter._get_json("https://example.com/test.json")
        assert result == {"data": {"children": []}}
        assert mock_session.get.call_count == 2


# ─── Hacker News Adapter Tests ───


class TestHNNormalization:
    def test_normalize_story(self, hn_adapter, hn_fixtures):
        hit = hn_fixtures["search_stories"]["hits"][0]
        post = hn_adapter._normalize_story(hit)

        assert post is not None
        assert post.id == "hn_39001234"
        assert post.platform == "hn"
        assert post.source == "hn"
        assert post.score == 89
        assert "GummySearch" in post.title

    def test_normalize_story_with_url(self, hn_adapter, hn_fixtures):
        hit = hn_fixtures["search_stories"]["hits"][1]
        post = hn_adapter._normalize_story(hit)
        assert post.url == "https://example.com/reddit-api-changes"

    def test_normalize_story_without_url(self, hn_adapter, hn_fixtures):
        hit = hn_fixtures["search_stories"]["hits"][0]
        post = hn_adapter._normalize_story(hit)
        assert "news.ycombinator.com" in post.url

    def test_normalize_deleted_item(self, hn_adapter, hn_fixtures):
        item = hn_fixtures["deleted_item"]
        result = hn_adapter._normalize_item(item)
        assert result is None


class TestHNCommentFlattening:
    def test_flatten_children(self, hn_adapter, hn_fixtures):
        children = hn_fixtures["item_thread"]["children"]
        comments = hn_adapter._flatten_children(children, depth=0)

        assert len(comments) == 3  # 2 top-level + 1 nested reply
        assert comments[0].id == "hn_39001300"
        assert comments[1].id == "hn_39001400"
        assert comments[1].parent_id == "hn_39001300"

    def test_depth_limit(self, hn_adapter, hn_fixtures):
        hn_adapter.comment_depth = 1
        children = hn_fixtures["item_thread"]["children"]
        comments = hn_adapter._flatten_children(children, depth=0)

        # Should get top-level comments only (depth 0), not nested replies
        assert len(comments) == 2
        assert all("39001300" in c.id or "39001500" in c.id for c in comments)


class TestHNFetch:
    @patch.object(HackerNewsAdapter, "_get_json")
    def test_fetch_with_keywords(self, mock_get, hn_adapter, hn_fixtures):
        # First call: search, then thread fetches for top stories
        mock_get.side_effect = [
            hn_fixtures["search_stories"],      # search
            hn_fixtures["item_thread"],          # thread for story 1 (highest points = 156)
            hn_fixtures["item_thread"],          # thread for story 2
        ]

        queries = {"keywords": ["pain points"], "limit": 50, "time_range": "week"}
        posts = hn_adapter.fetch(queries)

        assert len(posts) > 0
        platforms = {p.platform for p in posts}
        assert platforms == {"hn"}

    @patch.object(HackerNewsAdapter, "_get_json")
    def test_fetch_no_keywords_returns_empty(self, mock_get, hn_adapter):
        posts = hn_adapter.fetch({"limit": 50})
        assert posts == []
        mock_get.assert_not_called()
