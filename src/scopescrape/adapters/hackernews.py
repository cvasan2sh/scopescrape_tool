"""Hacker News adapter using the Algolia Search API.

Algolia provides full-text search over HN stories and comments:
    https://hn.algolia.com/api/v1/search?query=keyword&tags=story
    https://hn.algolia.com/api/v1/items/:id

No authentication required. Rate limits are generous (~10k req/hour).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests

from scopescrape.adapters.base import BaseAdapter
from scopescrape.log import get_logger
from scopescrape.models import RawPost
from scopescrape.utils import RateLimiter, deduplicate_posts

logger = get_logger(__name__)

ALGOLIA_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
ALGOLIA_ITEM_URL = "https://hn.algolia.com/api/v1/items"

# Map our time_range values to Algolia's numericFilters
# created_at_i is a Unix timestamp
SECONDS_MAP = {
    "day": 86400,
    "week": 604800,
    "month": 2592000,
    "year": 31536000,
}


class HackerNewsAdapter(BaseAdapter):
    """Fetch HN stories and comments via Algolia Search API."""

    def __init__(self, config: dict):
        super().__init__(config)
        hn_config = config.get("hn", {})

        self.rate_limiter = RateLimiter(
            initial_delay=hn_config.get("rate_limit_delay", 0.2),
            max_delay=30.0,
            max_retries=3,
        )
        self.comment_depth = hn_config.get("comment_depth", 5)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScopeScrape/0.1 (community pain point discovery)",
            "Accept": "application/json",
        })

    @property
    def platform_name(self) -> str:
        return "hn"

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Rate-limited GET with retry logic."""
        for attempt in range(self.rate_limiter.max_retries + 1):
            self.rate_limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    self.rate_limiter.on_success()
                    return resp.json()

                if resp.status_code == 429:
                    logger.warning(f"HN rate limited (attempt {attempt + 1})")
                    self.rate_limiter.on_failure()
                    continue

                logger.warning(f"HN HTTP {resp.status_code} for {url}")
                return None

            except requests.exceptions.Timeout:
                logger.warning(f"HN timeout (attempt {attempt + 1})")
                self.rate_limiter.on_failure()
            except requests.exceptions.RequestException as e:
                logger.error(f"HN request failed: {e}")
                return None

        return None

    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch HN stories matching keywords.

        Args:
            queries: Dict with optional keys:
                - keywords: list of search terms
                - limit: max stories to return (default 100)
                - time_range: "day", "week", "month", "year"

        Returns:
            Deduplicated list of RawPost objects (stories + their comments).
        """
        keywords = queries.get("keywords", [])
        limit = queries.get("limit", 100)
        time_range = queries.get("time_range", "week")

        if not keywords:
            logger.warning("HN adapter requires keywords for search. Skipping.")
            return []

        all_posts: list[RawPost] = []

        for kw in keywords:
            stories = self._search_stories(kw, time_range, limit)
            all_posts.extend(stories)

            # Fetch comments for top stories (by points)
            top_stories = sorted(stories, key=lambda p: p.score, reverse=True)[:10]
            for story in top_stories:
                _, comments = self.fetch_thread(story.id)
                all_posts.extend(comments)

        unique = deduplicate_posts(all_posts)
        logger.info(f"HN: {len(unique)} unique items ({len(all_posts)} total)")
        return unique

    def _search_stories(self, query: str, time_range: str, limit: int) -> list[RawPost]:
        """Search for HN stories matching a query."""
        params: dict = {
            "query": query,
            "tags": "story",
            "hitsPerPage": min(limit, 50),  # Algolia max per page
        }

        # Time filter
        seconds = SECONDS_MAP.get(time_range)
        if seconds:
            cutoff = int(datetime.utcnow().timestamp()) - seconds
            params["numericFilters"] = f"created_at_i>{cutoff}"

        logger.debug(f"Searching HN stories for '{query}'")
        data = self._get_json(ALGOLIA_SEARCH_URL, params)
        if not data:
            return []

        posts = []
        for hit in data.get("hits", []):
            post = self._normalize_story(hit)
            if post:
                posts.append(post)

        # Paginate if needed
        pages_needed = (limit // 50) + (1 if limit % 50 else 0)
        for page in range(1, min(pages_needed, 5)):  # Cap at 5 pages
            if len(posts) >= limit:
                break
            params["page"] = page
            data = self._get_json(ALGOLIA_SEARCH_URL, params)
            if not data:
                break
            for hit in data.get("hits", []):
                post = self._normalize_story(hit)
                if post:
                    posts.append(post)

        return posts[:limit]

    def fetch_thread(self, story_id: str) -> tuple[Optional[RawPost], list[RawPost]]:
        """Fetch a story and its full comment tree via the items endpoint.

        Args:
            story_id: HN item ID (numeric string).

        Returns:
            (story_post, flat_list_of_comments)
        """
        clean_id = story_id.replace("hn_", "")
        url = f"{ALGOLIA_ITEM_URL}/{clean_id}"

        data = self._get_json(url)
        if not data:
            return None, []

        story = self._normalize_item(data)
        comments = self._flatten_children(data.get("children", []), depth=0)

        return story, comments

    def _flatten_children(self, children: list, depth: int) -> list[RawPost]:
        """Recursively flatten HN comment tree."""
        if depth >= self.comment_depth:
            return []

        results = []
        for child in children:
            if child.get("type") != "comment":
                continue

            comment = self._normalize_item(child)
            if comment:
                results.append(comment)

            # Recurse
            sub_children = child.get("children", [])
            if sub_children:
                results.extend(self._flatten_children(sub_children, depth + 1))

        return results

    def _normalize_story(self, hit: dict) -> Optional[RawPost]:
        """Convert an Algolia search hit to a RawPost."""
        object_id = hit.get("objectID")
        if not object_id:
            return None

        created_at_str = hit.get("created_at", "")
        created_at = self._parse_timestamp(created_at_str)

        # Story text is in story_text or empty for link posts
        body = hit.get("story_text") or ""

        # HN URL: either the linked URL or the HN comments page
        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"

        return RawPost(
            id=f"hn_{object_id}",
            platform="hn",
            source="hn",
            title=hit.get("title", ""),
            body=body,
            author=hit.get("author", "[unknown]"),
            score=hit.get("points", 0) or 0,
            comment_count=hit.get("num_comments", 0) or 0,
            url=url,
            created_at=created_at,
        )

    def _normalize_item(self, item: dict) -> Optional[RawPost]:
        """Convert an Algolia items endpoint response to a RawPost."""
        item_id = item.get("id")
        if not item_id:
            return None

        # Skip dead/deleted items
        if item.get("deleted") or item.get("dead"):
            return None

        created_at_str = item.get("created_at", "")
        created_at = self._parse_timestamp(created_at_str)

        item_type = item.get("type", "story")
        if item_type == "comment":
            title = ""
            body = item.get("text", "") or ""
            parent_id = str(item.get("parent_id", "")) if item.get("parent_id") else None
        else:
            title = item.get("title", "")
            body = item.get("text", "") or ""
            parent_id = None

        url = item.get("url") or f"https://news.ycombinator.com/item?id={item_id}"

        return RawPost(
            id=f"hn_{item_id}",
            platform="hn",
            source="hn",
            title=title,
            body=body,
            author=item.get("author", "[unknown]"),
            score=item.get("points", 0) or 0,
            comment_count=len(item.get("children", [])),
            url=url,
            created_at=created_at,
            parent_id=f"hn_{parent_id}" if parent_id else None,
        )

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """Parse an ISO 8601 timestamp from Algolia."""
        if not ts:
            return datetime.utcnow()
        try:
            # Algolia returns "2026-03-20T14:30:00.000Z"
            clean = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(clean).replace(tzinfo=None)
        except (ValueError, TypeError):
            return datetime.utcnow()
