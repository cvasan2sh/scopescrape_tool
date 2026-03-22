"""Indie Hackers adapter using public Algolia search API.

Indie Hackers (indiehackers.com) does not have an official public API,
but their site uses Algolia for search, which is accessible.

Indie Hackers is extremely rich for pain signals:
    - Founders openly discuss failures and frustrations
    - Pivot decisions documented in detail
    - User interview summaries with discovered pain points
    - Posts about grinding through difficult problems

Search endpoints:
    - https://algolia.indiehackers.com/... (Algolia index)

No authentication required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests

from scopescrape.adapters.base import BaseAdapter
from scopescrape.log import get_logger
from scopescrape.models import RawPost
from scopescrape.utils import RateLimiter, deduplicate_posts, truncate

logger = get_logger(__name__)

# Indie Hackers uses Algolia for search (reverse-engineered from their web UI)
ALGOLIA_APP_ID = "2DFCF6E61A"  # IH's Algolia app ID
ALGOLIA_API_KEY = "ea4fc3980bf86b8b2c2bc0b56f62097d"  # Public search key
ALGOLIA_INDEX_NAME = "posts"  # Indie Hackers posts index

ALGOLIA_SEARCH_URL = f"https://{ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/{ALGOLIA_INDEX_NAME}/query"


class IndieHackersAdapter(BaseAdapter):
    """Fetch Indie Hackers posts via Algolia Search API.

    No authentication required. Public posts and discussions accessible.
    Rich source of founder pain points, pivots, and failures.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ih_config = config.get("indiehackers", {})

        self.rate_limiter = RateLimiter(
            initial_delay=ih_config.get("rate_limit_delay", 0.2),
            max_delay=30.0,
            max_retries=3,
        )
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScopeScrape/0.1 (community pain point discovery)",
            "Accept": "application/json",
        })

    @property
    def platform_name(self) -> str:
        return "indiehackers"

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Rate-limited GET with retry logic."""
        if params is None:
            params = {}

        # Add Algolia auth parameters
        params.setdefault("x-algolia-api-key", ALGOLIA_API_KEY)
        params.setdefault("x-algolia-application-id", ALGOLIA_APP_ID)

        for attempt in range(self.rate_limiter.max_retries + 1):
            self.rate_limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    self.rate_limiter.on_success()
                    return resp.json()

                if resp.status_code == 429:
                    logger.warning(
                        f"Indie Hackers rate limited (attempt {attempt + 1})"
                    )
                    self.rate_limiter.on_failure()
                    continue

                logger.warning(f"Indie Hackers HTTP {resp.status_code}")
                return None

            except requests.exceptions.Timeout:
                logger.warning(f"Indie Hackers timeout (attempt {attempt + 1})")
                self.rate_limiter.on_failure()
            except requests.exceptions.RequestException as e:
                logger.error(f"Indie Hackers request failed: {e}")
                return None

        return None

    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch posts from Indie Hackers matching keywords.

        Args:
            queries: Dict with optional keys:
                - keywords: list of search terms
                - limit: max posts to return (default 100)
                - time_range: ignored for IH

        Returns:
            Deduplicated list of RawPost objects.
        """
        keywords = queries.get("keywords", [])
        limit = queries.get("limit", 100)

        if not keywords:
            logger.warning("Indie Hackers adapter requires keywords. Skipping.")
            return []

        all_posts: list[RawPost] = []

        for kw in keywords:
            posts = self._search_posts(kw, limit)
            all_posts.extend(posts)

        unique = deduplicate_posts(all_posts)
        logger.info(f"Indie Hackers: {len(unique)} unique posts ({len(all_posts)} total)")
        return unique

    def _search_posts(self, query: str, limit: int) -> list[RawPost]:
        """Search Indie Hackers posts via Algolia."""
        # Algolia search body must be sent as query parameter JSON
        search_params = {
            "query": query,
            "hitsPerPage": min(limit, 50),
            "typoTolerance": True,
        }

        logger.debug(f"Searching Indie Hackers for '{query}'")
        data = self._get_json(ALGOLIA_SEARCH_URL, search_params)
        if not data:
            return []

        posts = []
        for hit in data.get("hits", []):
            post = self._normalize_post(hit)
            if post:
                posts.append(post)

        # Paginate if needed
        pages_needed = (limit // 50) + (1 if limit % 50 else 0)
        for page in range(1, min(pages_needed, 5)):
            if len(posts) >= limit:
                break

            search_params["page"] = page
            data = self._get_json(ALGOLIA_SEARCH_URL, search_params)
            if not data:
                break

            for hit in data.get("hits", []):
                post = self._normalize_post(hit)
                if post:
                    posts.append(post)

        return posts[:limit]

    def _normalize_post(self, hit: dict) -> Optional[RawPost]:
        """Convert an Algolia search hit to a RawPost."""
        if not hit or not hit.get("objectID"):
            return None

        post_id = hit.get("objectID")
        title = hit.get("title", "")
        body = hit.get("body", "") or ""

        # Indie Hackers posts have rich metadata
        author = hit.get("user_name", "[unknown]")
        url = hit.get("url", f"https://www.indiehackers.com/post/{post_id}")

        # Timestamp handling - might be ISO or Unix timestamp
        created_at_str = hit.get("created_at") or hit.get("publishedAt") or ""
        created_at = self._parse_timestamp(created_at_str)

        # Engagement metrics
        score = hit.get("vote_count", 0) or 0

        # Comments are in the post data
        comment_count = hit.get("comment_count", 0) or 0

        return RawPost(
            id=f"ih_{post_id}",
            platform="indiehackers",
            source="indiehackers",
            title=truncate(title, 300),
            body=truncate(body, 500),
            author=author,
            score=score,
            comment_count=comment_count,
            url=url,
            created_at=created_at,
        )

    @staticmethod
    def _parse_timestamp(ts_str: str) -> datetime:
        """Parse various timestamp formats from Indie Hackers."""
        if not ts_str:
            return datetime.utcnow()

        try:
            # ISO 8601 format
            if "T" in ts_str:
                clean = ts_str.replace("Z", "+00:00")
                return datetime.fromisoformat(clean).replace(tzinfo=None)

            # Unix timestamp
            if ts_str.isdigit():
                return datetime.fromtimestamp(int(ts_str), tz=timezone.utc).replace(
                    tzinfo=None
                )

        except (ValueError, TypeError):
            pass

        logger.debug(f"Could not parse timestamp: {ts_str}")
        return datetime.utcnow()
