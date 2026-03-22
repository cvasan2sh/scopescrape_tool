"""Stack Overflow adapter using the Stack Exchange API.

The Stack Exchange API provides public access to Stack Overflow data:
    https://api.stackexchange.com/2.3/

No authentication required for basic use. Quotas:
    - Without key: 300 requests/day
    - With key (SO_API_KEY): 10,000 requests/day

API Documentation:
    - Search: /2.3/search/advanced?site=stackoverflow&q=QUERY
    - Questions: /2.3/questions/{ids}
    - Answers: /2.3/questions/{ids}/answers
    - Tags: Use tag:python for tagged searches

The API uses gzip compression by default (transparently handled by requests).
Backoff header indicates when to wait before making more requests.
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

# Stack Exchange API base
SE_BASE = "https://api.stackexchange.com/2.3"
SEARCH_URL = f"{SE_BASE}/search/advanced"
ANSWERS_URL = f"{SE_BASE}/questions/{{ids}}/answers"


class StackOverflowAdapter(BaseAdapter):
    """Fetch Stack Overflow questions and answers via the Stack Exchange API.

    Supports both unauthenticated (300 req/day) and authenticated (10k req/day)
    requests with an SO_API_KEY.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        so_config = config.get("stackoverflow", {})

        self.api_key = so_config.get("api_key", "")
        self.rate_limiter = RateLimiter(
            initial_delay=so_config.get("rate_limit_delay", 0.5),
            max_delay=60.0,
            max_retries=5,
        )
        self.max_answers_per_question = so_config.get("max_answers_per_question", 3)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ScopeScrape/0.1 (community pain point discovery)",
            "Accept": "application/json",
        })

    @property
    def platform_name(self) -> str:
        return "stackoverflow"

    def _get_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a rate-limited GET request and return parsed JSON.

        Stack Exchange API uses gzip compression by default (handled transparently).
        Backoff header (if present) indicates when to make next request.
        Quota remaining is in X-RateLimit-Remaining.
        """
        if params is None:
            params = {}

        # Add site and key parameters
        params.setdefault("site", "stackoverflow")
        if self.api_key:
            params["key"] = self.api_key

        for attempt in range(self.rate_limiter.max_retries + 1):
            self.rate_limiter.wait()

            try:
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code == 200:
                    self.rate_limiter.on_success()

                    # Check for backoff header
                    backoff = resp.headers.get("Backoff")
                    if backoff:
                        wait_time = int(backoff)
                        logger.warning(f"Stack Overflow backoff header: wait {wait_time}s")
                        time.sleep(wait_time)

                    return resp.json()

                if resp.status_code == 400:
                    logger.error(f"Bad request (400) to Stack Exchange: {resp.text}")
                    return None

                if resp.status_code == 429:
                    logger.warning(f"Stack Overflow rate limited (429) (attempt {attempt + 1})")
                    self.rate_limiter.on_failure()
                    continue

                if resp.status_code == 503:
                    logger.warning(f"Stack Exchange unavailable (503) (attempt {attempt + 1})")
                    self.rate_limiter.on_failure()
                    continue

                logger.warning(f"Stack Overflow HTTP {resp.status_code}")
                self.rate_limiter.on_failure()

            except requests.exceptions.Timeout:
                logger.warning(f"Stack Overflow timeout (attempt {attempt + 1})")
                self.rate_limiter.on_failure()
            except requests.exceptions.RequestException as e:
                logger.error(f"Stack Overflow request failed: {e}")
                return None

        logger.error("Max retries exceeded for Stack Overflow")
        return None

    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch questions and answers from Stack Overflow.

        Args:
            queries: Dict with optional keys:
                - keywords: list of search terms
                - limit: max questions to fetch (default 100)
                - tags: list of tags to filter by (e.g., ['python', 'javascript'])
                - time_range: "day", "week", "month" (default "week")

        Returns:
            Deduplicated list of RawPost objects (questions + answers).
        """
        keywords = queries.get("keywords", [])
        limit = min(queries.get("limit", 100), 100)  # SO caps at 100 per page
        tags = queries.get("tags", [])
        time_range = queries.get("time_range", "week")

        if not keywords:
            logger.warning("Stack Overflow adapter requires keywords. Skipping.")
            return []

        all_posts: list[RawPost] = []

        for kw in keywords:
            questions = self._search_questions(kw, tags, time_range, limit)
            all_posts.extend(questions)

            # Fetch top answers for high-scoring questions
            top_questions = sorted(questions, key=lambda p: p.score, reverse=True)[:10]
            for question in top_questions:
                # Extract question ID from the post ID (format: so_q_12345)
                q_id = question.id.split("_")[-1]
                answers = self._fetch_answers(q_id, question.source)
                all_posts.extend(answers)

        unique = deduplicate_posts(all_posts)
        logger.info(f"Stack Overflow: {len(unique)} unique posts ({len(all_posts)} total)")
        return unique

    def _search_questions(self, query: str, tags: list, time_range: str, limit: int) -> list[RawPost]:
        """Search for Stack Overflow questions."""
        # Convert time_range to fromdate parameter (days ago)
        from datetime import datetime, timedelta

        days_map = {"day": 1, "week": 7, "month": 30}
        days_ago = days_map.get(time_range, 7)
        from_date = int((datetime.utcnow() - timedelta(days=days_ago)).timestamp())

        params = {
            "q": query,
            "fromdate": from_date,
            "sort": "votes",
            "order": "desc",
            "pagesize": min(limit, 100),
        }

        # Add tags filter if provided
        if tags:
            params["tagged"] = ";".join(tags)

        logger.debug(f"Searching Stack Overflow for '{query}' (tags={tags})")
        data = self._get_json(SEARCH_URL, params)
        if not data:
            return []

        posts = []
        for item in data.get("items", []):
            post = self._normalize_question(item)
            if post:
                posts.append(post)

        return posts

    def _fetch_answers(self, question_id: str, source: str) -> list[RawPost]:
        """Fetch answers for a specific question."""
        url = ANSWERS_URL.format(ids=question_id)
        params = {
            "sort": "votes",
            "order": "desc",
            "pagesize": self.max_answers_per_question,
        }

        logger.debug(f"Fetching answers for question {question_id}")
        data = self._get_json(url, params)
        if not data:
            return []

        posts = []
        for item in data.get("items", []):
            post = self._normalize_answer(item, source, question_id)
            if post:
                posts.append(post)

        return posts

    def _normalize_question(self, item: dict) -> Optional[RawPost]:
        """Convert a Stack Overflow question JSON to RawPost."""
        if not item or not item.get("question_id"):
            return None

        question_id = item.get("question_id")
        title = item.get("title", "")
        body = item.get("body", "") or ""

        # Author
        author = "[unknown]"
        if item.get("owner"):
            author = item["owner"].get("display_name", "[unknown]")

        # Timestamp
        created_at_ts = item.get("creation_date", 0)
        try:
            created_at = datetime.fromtimestamp(created_at_ts, tz=timezone.utc).replace(tzinfo=None)
        except (ValueError, OSError):
            created_at = datetime.utcnow()

        # Tags as source (useful for filtering)
        tags = item.get("tags", [])
        source = f"tags:{','.join(tags)}" if tags else "stackoverflow"

        return RawPost(
            id=f"so_q_{question_id}",
            platform="stackoverflow",
            source=source,
            title=title,
            body=body,
            author=author,
            score=item.get("score", 0),
            comment_count=item.get("comment_count", 0),
            url=item.get("link", ""),
            created_at=created_at,
        )

    def _normalize_answer(self, item: dict, source: str, question_id: str) -> Optional[RawPost]:
        """Convert a Stack Overflow answer JSON to RawPost."""
        if not item or not item.get("answer_id"):
            return None

        answer_id = item.get("answer_id")
        body = item.get("body", "") or ""

        # Author
        author = "[unknown]"
        if item.get("owner"):
            author = item["owner"].get("display_name", "[unknown]")

        # Timestamp
        created_at_ts = item.get("creation_date", 0)
        try:
            created_at = datetime.fromtimestamp(created_at_ts, tz=timezone.utc).replace(tzinfo=None)
        except (ValueError, OSError):
            created_at = datetime.utcnow()

        # Build answer URL
        answer_url = f"https://stackoverflow.com/questions/{question_id}/_/{answer_id}#${answer_id}"

        return RawPost(
            id=f"so_a_{answer_id}",
            platform="stackoverflow",
            source=source,
            title="",  # Answers don't have titles
            body=body,
            author=author,
            score=item.get("score", 0),
            comment_count=item.get("comment_count", 0),
            url=answer_url,
            created_at=created_at,
            parent_id=f"so_q_{question_id}",
        )
