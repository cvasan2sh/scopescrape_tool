"""Tests for Reddit, HN, GitHub, Stack Overflow, and Twitter adapters using mocked HTTP responses."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scopescrape.adapters.github import GitHubAdapter
from scopescrape.adapters.hackernews import HackerNewsAdapter
from scopescrape.adapters.indiehackers import IndieHackersAdapter
from scopescrape.adapters.producthunt import ProductHuntAdapter
from scopescrape.adapters.reddit import RedditAdapter
from scopescrape.adapters.stackoverflow import StackOverflowAdapter
from scopescrape.adapters.twitter import TwitterAdapter

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
def twitter_fixtures():
    with open(FIXTURES / "twitter_responses.json") as f:
        return json.load(f)


@pytest.fixture
def github_fixtures():
    with open(FIXTURES / "github_responses.json") as f:
        return json.load(f)


@pytest.fixture
def so_fixtures():
    with open(FIXTURES / "so_responses.json") as f:
        return json.load(f)


@pytest.fixture
def ph_fixtures():
    with open(FIXTURES / "producthunt_responses.json") as f:
        return json.load(f)


@pytest.fixture
def ih_fixtures():
    with open(FIXTURES / "indiehackers_responses.json") as f:
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


@pytest.fixture
def twitter_adapter(sample_config):
    """Twitter adapter with rate limiting disabled for tests."""
    sample_config["twitter"]["rate_limit_delay"] = 0.0
    return TwitterAdapter(sample_config)


@pytest.fixture
def github_adapter(sample_config):
    """GitHub adapter with rate limiting disabled for tests."""
    sample_config["github"]["rate_limit_delay"] = 0.0
    return GitHubAdapter(sample_config)


@pytest.fixture
def so_adapter(sample_config):
    """Stack Overflow adapter with rate limiting disabled for tests."""
    sample_config["stackoverflow"]["rate_limit_delay"] = 0.0
    return StackOverflowAdapter(sample_config)


@pytest.fixture
def ph_adapter(sample_config):
    """Product Hunt adapter with rate limiting disabled for tests."""
    sample_config["producthunt"]["rate_limit_delay"] = 0.0
    return ProductHuntAdapter(sample_config)


@pytest.fixture
def ih_adapter(sample_config):
    """Indie Hackers adapter with rate limiting disabled for tests."""
    sample_config["indiehackers"]["rate_limit_delay"] = 0.0
    return IndieHackersAdapter(sample_config)


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


# ─── Twitter Adapter Tests ───


class TestTwitterNormalization:
    def test_parse_count_k_suffix(self, twitter_adapter):
        """Test parsing engagement counts with K suffix."""
        assert twitter_adapter._parse_count("1.2K") == 1200
        assert twitter_adapter._parse_count("500") == 500
        assert twitter_adapter._parse_count("2.5M") == 2500000

    def test_parse_timestamp_relative_hours(self, twitter_adapter):
        """Test parsing relative timestamps like '2h'."""
        result = twitter_adapter._parse_timestamp("2h")
        assert result is not None
        # Should be approximately 2 hours ago
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        delta = (now - result).total_seconds() / 3600
        assert 1.9 < delta < 2.1

    def test_parse_timestamp_relative_minutes(self, twitter_adapter):
        """Test parsing relative timestamps like '30m'."""
        result = twitter_adapter._parse_timestamp("30m")
        assert result is not None
        from datetime import datetime
        now = datetime.utcnow()
        delta = (now - result).total_seconds() / 60
        assert 29 < delta < 31

    def test_parse_timestamp_relative_days(self, twitter_adapter):
        """Test parsing relative timestamps like '3d'."""
        result = twitter_adapter._parse_timestamp("3d")
        assert result is not None
        from datetime import datetime
        now = datetime.utcnow()
        delta = (now - result).total_seconds() / 86400
        assert 2.9 < delta < 3.1

    def test_parse_timestamp_empty(self, twitter_adapter):
        """Empty timestamp should return now."""
        result = twitter_adapter._parse_timestamp("")
        assert result is not None
        from datetime import datetime
        now = datetime.utcnow()
        delta = (now - result).total_seconds()
        assert delta < 5  # Within 5 seconds


class TestTwitterParsingHTML:
    def test_parse_search_results(self, twitter_adapter, twitter_fixtures):
        html = twitter_fixtures["search_results_html"]
        posts = twitter_adapter._parse_search_results(html, "test query", limit=10)

        assert len(posts) == 3
        assert posts[0].id == "tw_1234567890123456789"
        assert posts[0].platform == "twitter"
        assert posts[0].source == "twitter_search"
        assert "frustrated" in posts[0].body.lower()
        assert posts[0].author == "@user1"
        assert "twitter.com" in posts[0].url

    def test_parse_search_results_empty(self, twitter_adapter, twitter_fixtures):
        html = twitter_fixtures["search_results_empty"]
        posts = twitter_adapter._parse_search_results(html, "test query", limit=10)

        assert len(posts) == 0

    def test_parse_search_results_respects_limit(self, twitter_adapter, twitter_fixtures):
        html = twitter_fixtures["search_results_html"]
        posts = twitter_adapter._parse_search_results(html, "test query", limit=2)

        assert len(posts) == 2

    def test_tweet_engagement_score(self, twitter_adapter, twitter_fixtures):
        """Verify engagement scores are calculated correctly."""
        html = twitter_fixtures["search_results_html"]
        posts = twitter_adapter._parse_search_results(html, "test", limit=10)

        # First tweet: 1200 + 500 + 250 = 1950
        assert posts[0].score == 1950
        # Second tweet: 850 + 320 + 180 = 1350
        assert posts[1].score == 1350
        # Third tweet: 5000 + 2000 + 1500 = 8500
        assert posts[2].score == 8500


class TestTwitterFetch:
    @patch.object(TwitterAdapter, "_get_html")
    def test_fetch_with_keywords(self, mock_get, twitter_adapter, twitter_fixtures):
        mock_get.return_value = twitter_fixtures["search_results_html"]

        queries = {"keywords": ["pain"], "limit": 100, "time_range": "week"}
        posts = twitter_adapter.fetch(queries)

        assert len(posts) == 3
        assert all(p.platform == "twitter" for p in posts)
        mock_get.assert_called_once()

    @patch.object(TwitterAdapter, "_get_html")
    def test_fetch_no_keywords_returns_empty(self, mock_get, twitter_adapter):
        posts = twitter_adapter.fetch({"limit": 50})
        assert posts == []
        mock_get.assert_not_called()

    @patch.object(TwitterAdapter, "_get_html")
    def test_fetch_empty_response(self, mock_get, twitter_adapter, twitter_fixtures):
        mock_get.return_value = twitter_fixtures["search_results_empty"]

        posts = twitter_adapter.fetch({"keywords": ["xyz"], "limit": 50})
        assert posts == []

    @patch.object(TwitterAdapter, "_get_html")
    def test_fetch_multiple_keywords(self, mock_get, twitter_adapter, twitter_fixtures):
        mock_get.return_value = twitter_fixtures["search_results_html"]

        queries = {"keywords": ["pain", "bug"], "limit": 50, "time_range": "week"}
        posts = twitter_adapter.fetch(queries)

        # Should be called once per keyword
        assert mock_get.call_count == 2
        # Results should be deduplicated
        assert len(posts) > 0


class TestTwitterRateLimit:
    @patch.object(TwitterAdapter, "_create_session")
    def test_429_retry(self, mock_session_fn, sample_config):
        """Should retry on 429 and succeed on second attempt."""
        sample_config["twitter"]["rate_limit_delay"] = 0.0
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session

        resp_429 = MagicMock()
        resp_429.status_code = 429

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.text = "<html><body></body></html>"

        mock_session.get.side_effect = [resp_429, resp_200]

        adapter = TwitterAdapter(sample_config)
        result = adapter._get_html("https://example.com/search")
        assert result == "<html><body></body></html>"
        assert mock_session.get.call_count == 2


class TestTwitterThreadFetch:
    def test_fetch_thread_not_supported(self, twitter_adapter):
        """Thread fetching has limited support on Nitter."""
        top_post, comments = twitter_adapter.fetch_thread("tw_123456")
        assert top_post is None
        assert comments == []


# ─── GitHub Adapter Tests ───


class TestGitHubNormalization:
    def test_normalize_item_issue(self, github_adapter, github_fixtures):
        item = github_fixtures["search_results"]["items"][0]
        post = github_adapter._normalize_item(item)

        assert post is not None
        assert post.id.startswith("gh_issue_")
        assert post.platform == "github"
        assert "Performance issues" in post.title
        assert "frustrat" in post.body.lower()

    def test_normalize_item_with_author(self, github_adapter, github_fixtures):
        item = github_fixtures["search_results"]["items"][0]
        post = github_adapter._normalize_item(item)
        assert post.author == "dev_user_1"
        assert post.url == "https://github.com/company/datalib/issues/1001"

    def test_normalize_item_repo_override(self, github_adapter, github_fixtures):
        item = github_fixtures["search_results"]["items"][0]
        post = github_adapter._normalize_item(item, source_override="repo:custom/repo")
        assert post.source == "repo:custom/repo"


class TestGitHubFetch:
    @patch.object(GitHubAdapter, "_get_json")
    def test_fetch_with_keywords(self, mock_get, github_adapter, github_fixtures):
        mock_get.return_value = github_fixtures["search_results"]

        queries = {"keywords": ["performance pain"], "limit": 50}
        posts = github_adapter.fetch(queries)

        assert len(posts) == 2
        assert all(p.platform == "github" for p in posts)

    @patch.object(GitHubAdapter, "_get_json")
    def test_fetch_no_keywords_returns_empty(self, mock_get, github_adapter):
        posts = github_adapter.fetch({"limit": 50})
        assert posts == []
        mock_get.assert_not_called()

    @patch.object(GitHubAdapter, "_get_json")
    def test_fetch_with_repository_filter(self, mock_get, github_adapter, github_fixtures):
        mock_get.return_value = github_fixtures["search_results"]

        queries = {
            "keywords": ["bug"],
            "repositories": ["company/datalib"],
            "limit": 50,
        }
        posts = github_adapter.fetch(queries)

        assert len(posts) == 2
        call_url = mock_get.call_args[0][0]
        assert "search/issues" in call_url


class TestGitHubRateLimit:
    @patch.object(GitHubAdapter, "_create_session")
    def test_403_rate_limit_retry(self, mock_session_fn, sample_config):
        """Should retry on 403 rate limit and succeed on second attempt."""
        sample_config["github"]["rate_limit_delay"] = 0.0
        mock_session = MagicMock()
        mock_session_fn.return_value = mock_session

        resp_403 = MagicMock()
        resp_403.status_code = 403
        resp_403.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1234567890",
        }

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"items": []}

        mock_session.get.side_effect = [resp_403, resp_200]

        adapter = GitHubAdapter(sample_config)
        result = adapter._get_json("https://api.github.com/search/issues", {"q": "test"})
        assert result == {"items": []}
        assert mock_session.get.call_count == 2


# ─── Stack Overflow Adapter Tests ───


class TestStackOverflowNormalization:
    def test_normalize_question(self, so_adapter, so_fixtures):
        item = so_fixtures["search_results"]["items"][0]
        post = so_adapter._normalize_question(item)

        assert post is not None
        assert post.id == "so_q_70123456"
        assert post.platform == "stackoverflow"
        assert "improve slow" in post.title.lower()
        assert post.author == "john_dev"
        assert post.comment_count == 2

    def test_normalize_question_tags_as_source(self, so_adapter, so_fixtures):
        item = so_fixtures["search_results"]["items"][0]
        post = so_adapter._normalize_question(item)
        assert "database" in post.source
        assert "performance" in post.source

    def test_normalize_answer(self, so_adapter, so_fixtures):
        answer_item = so_fixtures["answers_70123456"]["items"][0]
        post = so_adapter._normalize_answer(answer_item, "tags:database,performance", "70123456")

        assert post is not None
        assert post.id == "so_a_70123480"
        assert post.title == ""  # Answers don't have titles
        assert "EXPLAIN" in post.body
        assert post.parent_id == "so_q_70123456"
        assert "stackoverflow.com" in post.url

    def test_normalize_answer_author(self, so_adapter, so_fixtures):
        answer_item = so_fixtures["answers_70123456"]["items"][0]
        post = so_adapter._normalize_answer(answer_item, "tags:test", "70123456")
        assert post.author == "db_expert"


class TestStackOverflowFetch:
    @patch.object(StackOverflowAdapter, "_get_json")
    def test_fetch_with_keywords(self, mock_get, so_adapter, so_fixtures):
        # First call: search questions, then fetch answers for top 2 questions
        mock_get.side_effect = [
            so_fixtures["search_results"],
            so_fixtures["answers_70123456"],
            so_fixtures["answers_70123457"],
        ]

        queries = {"keywords": ["slow queries"], "limit": 50, "time_range": "week"}
        posts = so_adapter.fetch(queries)

        # Should have questions + answers
        assert len(posts) > 2
        assert any(p.id.startswith("so_q_") for p in posts)
        assert any(p.id.startswith("so_a_") for p in posts)

    @patch.object(StackOverflowAdapter, "_get_json")
    def test_fetch_no_keywords_returns_empty(self, mock_get, so_adapter):
        posts = so_adapter.fetch({"limit": 50})
        assert posts == []
        mock_get.assert_not_called()

    @patch.object(StackOverflowAdapter, "_get_json")
    def test_fetch_with_tags(self, mock_get, so_adapter, so_fixtures):
        # Provide side_effect with enough responses for search + 2 answer fetches
        mock_get.side_effect = [
            so_fixtures["search_results"],
            so_fixtures["answers_70123456"],
            so_fixtures["answers_70123457"],
        ]

        queries = {
            "keywords": ["performance"],
            "tags": ["python", "performance"],
            "limit": 50,
        }
        posts = so_adapter.fetch(queries)

        assert len(posts) > 2
        # Verify the first call (search) had the tags parameter
        first_call_params = mock_get.call_args_list[0][0][1]  # Second positional arg
        assert "tagged" in first_call_params


class TestStackOverflowRateLimit:
    @patch.object(StackOverflowAdapter, "_get_json")
    def test_backoff_header_respected(self, mock_get, so_adapter):
        """Should handle backoff header from Stack Exchange API."""
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Backoff": "0"}  # 0 seconds for testing
        resp.json.return_value = {"items": []}

        mock_get.return_value = {"items": []}

        queries = {"keywords": ["test"], "limit": 10}
        posts = so_adapter.fetch(queries)

        assert posts == []


# ─── Product Hunt Adapter Tests ───


class TestProductHuntNormalization:
    def test_normalize_product(self, ph_adapter, ph_fixtures):
        node = ph_fixtures["search_products"]["data"]["products"]["edges"][0]["node"]
        post = ph_adapter._normalize_product(node)

        assert post is not None
        assert post.id.startswith("ph_product_")
        assert post.platform == "producthunt"
        assert post.title == "ScopeScrape"
        assert "pain point" in post.body.lower()
        assert post.score == 245

    def test_normalize_product_with_url(self, ph_adapter, ph_fixtures):
        node = ph_fixtures["search_products"]["data"]["products"]["edges"][0]["node"]
        post = ph_adapter._normalize_product(node)
        assert "producthunt.com/products" in post.url

    def test_normalize_review(self, ph_adapter, ph_fixtures):
        review_node = ph_fixtures["product_reviews"]["data"]["product"]["reviews"]["edges"][0]["node"]
        post = ph_adapter._normalize_review(review_node, "ph_prod_1001", "scopescrape")

        assert post is not None
        assert post.id.startswith("ph_review_")
        assert post.platform == "producthunt"
        assert "saved us" in post.body.lower()
        assert post.author == "jane_entrepreneur"
        assert post.score == 5  # rating

    def test_normalize_review_parent_link(self, ph_adapter, ph_fixtures):
        review_node = ph_fixtures["product_reviews"]["data"]["product"]["reviews"]["edges"][0]["node"]
        post = ph_adapter._normalize_review(review_node, "ph_prod_1001", "scopescrape")
        assert post.parent_id == "ph_product_ph_prod_1001"


class TestProductHuntFetch:
    @patch.object(ProductHuntAdapter, "_post_graphql")
    def test_fetch_with_keywords(self, mock_post, ph_adapter, ph_fixtures):
        # First call: search products, then reviews for top products (min 2)
        mock_post.side_effect = [
            ph_fixtures["search_products"],
            ph_fixtures["product_reviews"],  # reviews for first product
            ph_fixtures["product_reviews"],  # reviews for second product
        ]

        queries = {"keywords": ["pain"], "limit": 50}
        posts = ph_adapter.fetch(queries)

        assert len(posts) > 0
        assert all(p.platform == "producthunt" for p in posts)

    @patch.object(ProductHuntAdapter, "_post_graphql")
    def test_fetch_no_keywords_returns_empty(self, mock_post, ph_adapter):
        posts = ph_adapter.fetch({"limit": 50})
        assert posts == []
        mock_post.assert_not_called()


# ─── Indie Hackers Adapter Tests ───


class TestIndieHackersNormalization:
    def test_normalize_post(self, ih_adapter, ih_fixtures):
        hit = ih_fixtures["search_posts"]["hits"][0]
        post = ih_adapter._normalize_post(hit)

        assert post is not None
        assert post.id == "ih_post_8901"
        assert post.platform == "indiehackers"
        assert post.author == "john_maker"
        assert "pain point" in post.title.lower()
        assert post.score == 142

    def test_normalize_post_url(self, ih_adapter, ih_fixtures):
        hit = ih_fixtures["search_posts"]["hits"][0]
        post = ih_adapter._normalize_post(hit)
        assert "indiehackers.com" in post.url

    def test_parse_timestamp_iso(self, ih_adapter):
        """Test parsing ISO 8601 timestamps."""
        ts_str = "2026-03-20T14:30:00Z"
        result = ih_adapter._parse_timestamp(ts_str)
        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 20

    def test_parse_timestamp_empty(self, ih_adapter):
        """Empty timestamp should return now."""
        result = ih_adapter._parse_timestamp("")
        assert result is not None
        from datetime import datetime
        now = datetime.utcnow()
        delta = (now - result).total_seconds()
        assert delta < 5


class TestIndieHackersFetch:
    @patch.object(IndieHackersAdapter, "_get_json")
    def test_fetch_with_keywords(self, mock_get, ih_adapter, ih_fixtures):
        mock_get.return_value = ih_fixtures["search_posts"]

        queries = {"keywords": ["pain points"], "limit": 50}
        posts = ih_adapter.fetch(queries)

        assert len(posts) == 3
        assert all(p.platform == "indiehackers" for p in posts)

    @patch.object(IndieHackersAdapter, "_get_json")
    def test_fetch_no_keywords_returns_empty(self, mock_get, ih_adapter):
        posts = ih_adapter.fetch({"limit": 50})
        assert posts == []
        mock_get.assert_not_called()

    @patch.object(IndieHackersAdapter, "_get_json")
    def test_fetch_multiple_keywords(self, mock_get, ih_adapter, ih_fixtures):
        mock_get.return_value = ih_fixtures["search_posts"]

        queries = {"keywords": ["pain", "pivot"], "limit": 50}
        posts = ih_adapter.fetch(queries)

        # Called once per keyword
        assert mock_get.call_count == 2
        # Results should be deduplicated
        assert len(posts) > 0
