"""GitHub adapter using the public REST Search API.

GitHub's public REST API allows searching issues and discussions without authentication,
with a rate limit of 60 requests/hour. With a personal access token, the limit increases
to 5000 requests/hour.

Search API documentation:
    GET https://api.github.com/search/issues
    Query format: q=QUERY (use type:issue or type:discussion)

Example queries:
    - "pain in:title,body type:issue" (issues only)
    - "frustrated in:title,body type:discussion" (discussions only)
    - "bug in:title,body repo:owner/repo" (repository-scoped)

Rate limits are returned in X-RateLimit-Remaining and X-RateLimit-Reset headers.
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
from scopescrape.utils import RateLimiter, deduplicate_posts

logger = get_logger(__name__)

# GitHub API base
SEARCH_URL = "https://api.github.com/search/issues"


class GitHubAdapter(BaseAdapter):
    """Fetch GitHub issues and discussions via the REST Search API.

    Supports both authenticated (with token) and unauthenticated requests.
    Unauthenticated: 60 req/hour
    Authenticated: 5000 req/hour (requires GITHUB_TOKEN)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        github_config = config.get("github", {})

        self.token = github_config.get("token", "")
        self.rate_limiter = RateLimiter(
            initial_delay=github_config.get("rate_limit_delay", 1.0),
            max_delay=60.0,
            max_retries=5,
        )
        self.session = self._create_session()

    @property
    def platform_name(self) -> str:
        return "github"

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers."""
        s = requests.Session()
        headers = {
            "User-Agent": "ScopeScrape/0.1 (community pain point discovery)",
            "Accept": "application/vnd.github+json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        s.headers.update(headers)
        return s

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a rate-limited GET request and return parsed JSON.

        Handles 403 (rate limit) with backoff. Returns None on unrecoverable errors.
        GitHub's rate limit is returned in X-RateLimit-Remaining and X-RateLimit-Reset.
        """
        for attempt in range(self.rate_limiter.max_retries + 1):
            self.rate_limiter.wait()

            try:
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code == 200:
                    self.rate_limiter.on_success()
                    return resp.json()

                if resp.status_code == 403:
                    # Rate limit or authentication failure
                    remaining = resp.headers.get("X-RateLimit-Remaining", "0")
                    reset = resp.headers.get("X-RateLimit-Reset", "0")
                    logger.warning(
                        f"GitHub rate limited (403). Remaining: {remaining}, "
                        f"reset: {reset} (attempt {attempt + 1})"
                    )
                    self.rate_limiter.on_failure()
                    continue

                if resp.status_code == 422:
                    # Invalid query or validation error
                    logger.error(f"Invalid GitHub query (422): {resp.json()}")
                    return None

                if resp.status_code == 404:
                    logger.error(f"Not found (404): {url}")
                    return None

                logger.warning(f"GitHub HTTP {resp.status_code} for {url}")
                self.rate_limiter.on_failure()

            except requests.exceptions.Timeout:
                logger.warning(f"GitHub timeout for {url} (attempt {attempt + 1})")
                self.rate_limiter.on_failure()
            except requests.exceptions.RequestException as e:
                logger.error(f"GitHub request failed for {url}: {e}")
                return None

        logger.error(f"Max retries exceeded for {url}")
        return None

    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch posts from GitHub issues and discussions based on keywords.

        Args:
            queries: Dict with optional keys:
                - keywords: list of search terms
                - limit: max results to fetch (default 100)
                - repositories: list of "owner/repo" strings to scope search
                - item_type: "issue" or "discussion" (default: both)

        Returns:
            Deduplicated list of RawPost objects.
        """
        keywords = queries.get("keywords", [])
        limit = min(queries.get("limit", 100), 100)  # GitHub caps per-request at 100
        repositories = queries.get("repositories", [])
        item_type = queries.get("item_type", "both")  # "issue", "discussion", "both"

        if not keywords:
            logger.warning("GitHub adapter requires keywords for search. Skipping.")
            return []

        all_posts: list[RawPost] = []

        # Search for each keyword
        for kw in keywords:
            # Strategy 1: Repository-scoped searches
            if repositories:
                for repo in repositories:
                    posts = self._search_repo(kw, repo, item_type, limit)
                    all_posts.extend(posts)
            else:
                # Strategy 2: Global search
                posts = self._search_global(kw, item_type, limit)
                all_posts.extend(posts)

        unique = deduplicate_posts(all_posts)
        logger.info(f"GitHub: {len(unique)} unique posts ({len(all_posts)} total fetched)")
        return unique

    def _search_global(self, query: str, item_type: str, limit: int) -> list[RawPost]:
        """Search across all of GitHub for issues/discussions matching a query."""
        # Build the GitHub search query
        search_query = f'"{query}" in:title,body'

        if item_type == "issue":
            search_query += " type:issue"
        elif item_type == "discussion":
            search_query += " type:discussion"

        params = {
            "q": search_query,
            "sort": "stars",  # Most starred = most relevant
            "order": "desc",
            "per_page": min(limit, 100),
        }

        logger.debug(f"Searching GitHub globally for '{query}' (type={item_type})")
        data = self._get_json(SEARCH_URL, params)
        if not data:
            return []

        return self._parse_search_results(data)

    def _search_repo(self, query: str, repository: str, item_type: str, limit: int) -> list[RawPost]:
        """Search within a specific repository."""
        search_query = f'"{query}" in:title,body repo:{repository}'

        if item_type == "issue":
            search_query += " type:issue"
        elif item_type == "discussion":
            search_query += " type:discussion"

        params = {
            "q": search_query,
            "sort": "stars",
            "order": "desc",
            "per_page": min(limit, 100),
        }

        logger.debug(f"Searching {repository} for '{query}' (type={item_type})")
        data = self._get_json(SEARCH_URL, params)
        if not data:
            return []

        return self._parse_search_results(data, source_override=f"repo:{repository}")

    def _parse_search_results(self, data: dict, source_override: str = "") -> list[RawPost]:
        """Parse GitHub search API response into RawPost objects."""
        posts = []
        items = data.get("items", [])

        for item in items:
            post = self._normalize_item(item, source_override)
            if post:
                posts.append(post)

        return posts

    def _normalize_item(self, item: dict, source_override: str = "") -> Optional[RawPost]:
        """Convert a GitHub issue/discussion JSON object to a RawPost."""
        if not item or not item.get("id"):
            return None

        # Determine if this is an issue or discussion
        item_type = "discussion" if "discussions" in item.get("url", "") else "issue"

        # Extract key fields
        title = item.get("title", "")
        body = item.get("body", "") or ""
        post_id = item.get("number")
        if not post_id:
            return None

        # Repository name from URL (e.g., "https://api.github.com/repos/owner/repo/issues/123")
        repo_parts = item.get("repository_url", "").split("/")
        source = source_override or f"repo:{'/'.join(repo_parts[-2:])}" if len(repo_parts) >= 2 else "unknown"

        # Timestamp
        created_at_str = item.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            created_at = datetime.utcnow()

        # Author
        author = ""
        if item.get("user"):
            author = item["user"].get("login", "[unknown]")

        return RawPost(
            id=f"gh_{item_type}_{post_id}",
            platform="github",
            source=source,
            title=title,
            body=body,
            author=author,
            score=item.get("reactions", {}).get("total_count", 0) if isinstance(item.get("reactions"), dict) else 0,
            comment_count=item.get("comments", 0),
            url=item.get("html_url", ""),
            created_at=created_at,
        )
