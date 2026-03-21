"""Reddit adapter using public JSON endpoints (no API key required).

Reddit exposes a public JSON API by appending .json to any URL:
    https://www.reddit.com/r/saas/hot.json
    https://www.reddit.com/r/saas/search.json?q=keyword&t=week
    https://www.reddit.com/r/saas/comments/abc123.json

Rate limits: ~30 requests/minute without auth. We default to a
2-second delay between requests to stay well under that.

Important: Reddit blocks requests without a descriptive User-Agent
and will return 429 if you hit the rate limit.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import requests

from scopescrape.adapters.base import BaseAdapter
from scopescrape.log import get_logger
from scopescrape.models import RawPost
from scopescrape.utils import RateLimiter, deduplicate_posts, truncate

logger = get_logger(__name__)

# Reddit's public JSON API base
BASE_URL = "https://www.reddit.com"

# Time range mapping for Reddit search
TIME_RANGE_MAP = {
    "day": "day",
    "week": "week",
    "month": "month",
    "year": "year",
    "all": "all",
}

# Sort options for subreddit listings
SORT_OPTIONS = ("hot", "new", "top", "rising")


class RedditAdapter(BaseAdapter):
    """Fetch Reddit posts and comments via public JSON endpoints.

    No API key or OAuth required. Uses requests with a polite
    User-Agent and rate limiting.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        reddit_config = config.get("reddit", {})

        self.user_agent = reddit_config.get(
            "user_agent", "ScopeScrape/0.1 (community pain point discovery)"
        )
        self.rate_limiter = RateLimiter(
            initial_delay=reddit_config.get("rate_limit_delay", 2.0),
            max_delay=60.0,
            max_retries=5,
        )
        self.comment_depth = reddit_config.get("comment_depth", 3)
        self.session = self._create_session()

    @property
    def platform_name(self) -> str:
        return "reddit"

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        })
        return s

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a rate-limited GET request and return parsed JSON.

        Handles 429 (rate limit) with exponential backoff and
        retries. Returns None on unrecoverable errors.
        """
        for attempt in range(self.rate_limiter.max_retries + 1):
            self.rate_limiter.wait()

            try:
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code == 200:
                    self.rate_limiter.on_success()
                    return resp.json()

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(
                        f"Rate limited (429). Waiting {retry_after}s "
                        f"(attempt {attempt + 1})"
                    )
                    time.sleep(retry_after)
                    self.rate_limiter.on_failure()
                    continue

                if resp.status_code == 403:
                    logger.error(f"Forbidden (403) for {url}. Subreddit may be private.")
                    return None

                if resp.status_code == 404:
                    logger.error(f"Not found (404): {url}")
                    return None

                logger.warning(f"HTTP {resp.status_code} for {url}")
                self.rate_limiter.on_failure()

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
                self.rate_limiter.on_failure()
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
                return None

        logger.error(f"Max retries exceeded for {url}")
        return None

    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch posts from Reddit based on subreddits and/or keywords.

        Args:
            queries: Dict with optional keys:
                - subreddits: list of subreddit names (without r/ prefix)
                - keywords: list of search terms
                - limit: max posts per source (default 100)
                - time_range: "day", "week", "month", "year" (default "week")

        Returns:
            Deduplicated list of RawPost objects.
        """
        subreddits = queries.get("subreddits", [])
        keywords = queries.get("keywords", [])
        limit = min(queries.get("limit", 100), 100)  # Reddit caps at 100 per request
        time_range = queries.get("time_range", "week")

        all_posts: list[RawPost] = []

        # Strategy 1: Search within specific subreddits
        for sub in subreddits:
            sub = sub.strip().lstrip("r/")
            if keywords:
                for kw in keywords:
                    posts = self._search_subreddit(sub, kw, time_range, limit)
                    all_posts.extend(posts)
            else:
                # No keywords: just fetch top/hot posts
                posts = self._list_subreddit(sub, sort="hot", time_range=time_range, limit=limit)
                all_posts.extend(posts)

        # Strategy 2: Global search with keywords (no specific subreddit)
        if keywords and not subreddits:
            for kw in keywords:
                posts = self._search_global(kw, time_range, limit)
                all_posts.extend(posts)

        # Deduplicate across all sources
        unique = deduplicate_posts(all_posts)
        logger.info(f"Reddit: {len(unique)} unique posts ({len(all_posts)} total fetched)")
        return unique

    def _search_subreddit(
        self, subreddit: str, query: str, time_range: str, limit: int
    ) -> list[RawPost]:
        """Search within a specific subreddit."""
        url = f"{BASE_URL}/r/{subreddit}/search.json"
        params = {
            "q": query,
            "restrict_sr": "on",
            "sort": "relevance",
            "t": TIME_RANGE_MAP.get(time_range, "week"),
            "limit": limit,
            "raw_json": 1,
        }

        logger.debug(f"Searching r/{subreddit} for '{query}'")
        data = self._get_json(url, params)
        if not data:
            return []

        return self._parse_listing(data, source=f"r/{subreddit}")

    def _list_subreddit(
        self, subreddit: str, sort: str = "hot", time_range: str = "week", limit: int = 100
    ) -> list[RawPost]:
        """Fetch posts from a subreddit listing (hot, new, top, rising)."""
        url = f"{BASE_URL}/r/{subreddit}/{sort}.json"
        params = {
            "limit": limit,
            "raw_json": 1,
        }
        if sort == "top":
            params["t"] = TIME_RANGE_MAP.get(time_range, "week")

        logger.debug(f"Listing r/{subreddit}/{sort}")
        data = self._get_json(url, params)
        if not data:
            return []

        return self._parse_listing(data, source=f"r/{subreddit}")

    def _search_global(self, query: str, time_range: str, limit: int) -> list[RawPost]:
        """Search across all of Reddit."""
        url = f"{BASE_URL}/search.json"
        params = {
            "q": query,
            "sort": "relevance",
            "t": TIME_RANGE_MAP.get(time_range, "week"),
            "limit": limit,
            "raw_json": 1,
        }

        logger.debug(f"Global search for '{query}'")
        data = self._get_json(url, params)
        if not data:
            return []

        return self._parse_listing(data, source="global_search")

    def fetch_thread(self, post_id: str, subreddit: str = "") -> tuple[Optional[RawPost], list[RawPost]]:
        """Fetch a thread and its comment tree.

        Args:
            post_id: Reddit post ID (with or without t3_ prefix).
            subreddit: Subreddit name (optional, speeds up lookup).

        Returns:
            (top_post, flat_list_of_comments)
        """
        clean_id = post_id.replace("t3_", "")

        if subreddit:
            url = f"{BASE_URL}/r/{subreddit}/comments/{clean_id}.json"
        else:
            url = f"{BASE_URL}/comments/{clean_id}.json"

        params = {"raw_json": 1, "limit": 200}
        data = self._get_json(url, params)
        if not data or not isinstance(data, list) or len(data) < 2:
            return None, []

        # First element: the post listing
        post_data = data[0].get("data", {}).get("children", [])
        if not post_data:
            return None, []

        top_post = self._normalize_post(post_data[0].get("data", {}))

        # Second element: the comment listing
        comments_data = data[1].get("data", {}).get("children", [])
        comments = self._flatten_comments(comments_data, top_post.source if top_post else "unknown")

        return top_post, comments

    def _parse_listing(self, data: dict, source: str) -> list[RawPost]:
        """Parse a Reddit listing response into RawPost objects."""
        posts = []
        children = data.get("data", {}).get("children", [])

        for child in children:
            if child.get("kind") != "t3":
                continue
            post = self._normalize_post(child.get("data", {}), source_override=source)
            if post:
                posts.append(post)

        return posts

    def _normalize_post(self, data: dict, source_override: str = "") -> Optional[RawPost]:
        """Convert a Reddit post JSON object to a RawPost."""
        if not data or not data.get("id"):
            return None

        # Skip removed/deleted posts
        if data.get("removed_by_category") or data.get("author") == "[deleted]":
            return None

        post_id = f"t3_{data['id']}"
        subreddit = data.get("subreddit", "unknown")
        source = source_override or f"r/{subreddit}"

        # Body: selftext for text posts, empty for link posts
        body = data.get("selftext", "") or ""

        # Timestamp
        created_utc = data.get("created_utc", 0)
        try:
            created_at = datetime.fromtimestamp(created_utc, tz=timezone.utc).replace(tzinfo=None)
        except (ValueError, OSError):
            created_at = datetime.utcnow()

        return RawPost(
            id=post_id,
            platform="reddit",
            source=source,
            title=data.get("title", ""),
            body=body,
            author=data.get("author", "[unknown]"),
            score=data.get("score", 0),
            comment_count=data.get("num_comments", 0),
            url=f"https://reddit.com{data.get('permalink', '')}",
            created_at=created_at,
        )

    def _flatten_comments(
        self, children: list, source: str, depth: int = 0
    ) -> list[RawPost]:
        """Recursively flatten a comment tree into a list of RawPost objects.

        Respects comment_depth config to avoid crawling too deep.
        """
        if depth >= self.comment_depth:
            return []

        comments = []
        for child in children:
            kind = child.get("kind")
            data = child.get("data", {})

            if kind == "more":
                # MoreComments stub. In public API we skip these
                # since expanding them requires separate requests.
                continue

            if kind != "t1":
                continue

            comment = self._normalize_comment(data, source)
            if comment:
                comments.append(comment)

            # Recurse into replies
            replies = data.get("replies")
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                comments.extend(
                    self._flatten_comments(reply_children, source, depth + 1)
                )

        return comments

    def _normalize_comment(self, data: dict, source: str) -> Optional[RawPost]:
        """Convert a Reddit comment JSON object to a RawPost."""
        if not data or not data.get("id"):
            return None

        if data.get("author") == "[deleted]" or data.get("body") in ("[deleted]", "[removed]"):
            return None

        comment_id = f"t1_{data['id']}"
        created_utc = data.get("created_utc", 0)

        try:
            created_at = datetime.fromtimestamp(created_utc, tz=timezone.utc).replace(tzinfo=None)
        except (ValueError, OSError):
            created_at = datetime.utcnow()

        return RawPost(
            id=comment_id,
            platform="reddit",
            source=source,
            title="",  # comments have no title
            body=data.get("body", ""),
            author=data.get("author", "[unknown]"),
            score=data.get("score", 0),
            comment_count=0,
            url=f"https://reddit.com{data.get('permalink', '')}",
            created_at=created_at,
            parent_id=data.get("parent_id"),
        )
