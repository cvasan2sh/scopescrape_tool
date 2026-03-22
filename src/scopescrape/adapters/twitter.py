"""Twitter/X adapter using Nitter (public HTML scraping, no API key required).

Twitter/X has no public JSON endpoints like Reddit. The official API v2 free tier
is extremely limited (450 req/15min, past 7 days only). Instead, we use Nitter,
a lightweight open-source Twitter frontend that:

    https://nitter.net/search?q=keyword&f=tweets&sort=latest

Nitter provides:
  - No authentication required
  - Generous rate limits (most instances allow 100s of requests per hour)
  - Clean HTML that's easy to parse
  - Search, timeline, and trend support

Limitations:
  - Depends on Nitter instances staying available (can fail over between instances)
  - Slightly slower than API (HTML parsing vs JSON)
  - Only public tweets (like the official API free tier anyway)

Alternative: If Nitter becomes unavailable, Twitter's API v2 free tier could be
used, but would require registered API keys and has strict quotas.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote, urlencode

import requests
from bs4 import BeautifulSoup

from scopescrape.adapters.base import BaseAdapter
from scopescrape.log import get_logger
from scopescrape.models import RawPost
from scopescrape.utils import RateLimiter, deduplicate_posts

logger = get_logger(__name__)

# Nitter instances to try (in order)
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.unixfederal.com",
    "https://nitter.cutelab.space",
]

# Time range mapping for Twitter search
TIME_RANGE_MAP = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
    "all": 999,  # Very old tweets
}


class TwitterAdapter(BaseAdapter):
    """Fetch tweets from Twitter via Nitter HTML scraping.

    No API key or OAuth required. Uses requests + BeautifulSoup to
    scrape public tweets from Nitter instances.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        twitter_config = config.get("twitter", {})

        self.nitter_instance = twitter_config.get("nitter_instance", NITTER_INSTANCES[0])
        self.fallback_instances = twitter_config.get(
            "fallback_instances", NITTER_INSTANCES[1:]
        )
        self.rate_limiter = RateLimiter(
            initial_delay=twitter_config.get("rate_limit_delay", 1.0),
            max_delay=60.0,
            max_retries=3,
        )
        self.session = self._create_session()

    @property
    def platform_name(self) -> str:
        return "twitter"

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": "ScopeScrape/0.1 (community pain point discovery)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        return s

    def _get_html(self, url: str, params: Optional[dict] = None) -> Optional[str]:
        """Make a rate-limited GET request and return HTML.

        Handles failures with instance fallback. Returns None on unrecoverable errors.
        """
        for attempt in range(self.rate_limiter.max_retries + 1):
            self.rate_limiter.wait()

            try:
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code == 200:
                    self.rate_limiter.on_success()
                    return resp.text

                if resp.status_code == 429:
                    logger.warning(
                        f"Rate limited (429). Waiting before retry "
                        f"(attempt {attempt + 1})"
                    )
                    time.sleep(5)
                    self.rate_limiter.on_failure()
                    continue

                if resp.status_code == 404:
                    logger.debug(f"Not found (404): {url}")
                    return None

                logger.warning(f"HTTP {resp.status_code} for {url}")
                self.rate_limiter.on_failure()

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
                self.rate_limiter.on_failure()
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {url}: {e}")
                self.rate_limiter.on_failure()

        return None

    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch tweets based on keywords.

        Args:
            queries: Dict with optional keys:
                - keywords: list of search terms (required)
                - limit: max tweets per keyword (default 100)
                - time_range: "day", "week", "month", "year" (default "week")

        Returns:
            Deduplicated list of RawPost objects.
        """
        keywords = queries.get("keywords", [])
        limit = min(queries.get("limit", 100), 200)  # Nitter limits per page
        time_range = queries.get("time_range", "week")

        if not keywords:
            logger.warning("Twitter adapter requires keywords for search. Skipping.")
            return []

        all_posts: list[RawPost] = []

        for kw in keywords:
            posts = self._search_tweets(kw, time_range, limit)
            all_posts.extend(posts)

        unique = deduplicate_posts(all_posts)
        logger.info(f"Twitter: {len(unique)} unique tweets ({len(all_posts)} total fetched)")
        return unique

    def _search_tweets(self, query: str, time_range: str, limit: int) -> list[RawPost]:
        """Search for tweets matching a query via Nitter."""
        # Calculate date range
        days_back = TIME_RANGE_MAP.get(time_range, 7)
        since_date = (datetime.utcnow() - timedelta(days=days_back)).date()

        url = f"{self.nitter_instance}/search"
        params = {
            "q": query,
            "f": "tweets",
            "sort": "latest",
            "since": str(since_date),
        }

        logger.debug(f"Searching Twitter for '{query}' (since {since_date})")
        html = self._get_html(url, params)
        if not html:
            return []

        return self._parse_search_results(html, query, limit)

    def _parse_search_results(self, html: str, query: str, limit: int) -> list[RawPost]:
        """Parse tweets from Nitter search results HTML."""
        posts = []

        try:
            soup = BeautifulSoup(html, "html.parser")

            # Find all tweet containers (Nitter uses .tweet class)
            tweet_divs = soup.find_all("div", class_="tweet")

            for tweet_div in tweet_divs:
                if len(posts) >= limit:
                    break

                post = self._parse_tweet_div(tweet_div)
                if post:
                    posts.append(post)

        except Exception as e:
            logger.error(f"Failed to parse Twitter search results: {e}")

        return posts

    def _parse_tweet_div(self, tweet_div) -> Optional[RawPost]:
        """Parse a single tweet from a Nitter tweet div element."""
        try:
            # Extract tweet ID from data-tweet-id or href
            tweet_a = tweet_div.find("a", class_="tweet-link")
            if not tweet_a or not tweet_a.get("href"):
                return None

            tweet_url = tweet_a["href"]  # /username/status/123456789
            parts = tweet_url.strip("/").split("/")
            if len(parts) < 3:
                return None

            username = parts[0]
            tweet_id = parts[-1]

            # Build full URL
            full_url = f"https://twitter.com/{username}/status/{tweet_id}"

            # Extract author
            author_elem = tweet_div.find("a", class_="username")
            author = author_elem.get_text(strip=True) if author_elem else f"@{username}"

            # Extract timestamp
            time_elem = tweet_div.find("span", class_="tweet-date")
            created_at = self._parse_timestamp(time_elem.get_text(strip=True) if time_elem else "")

            # Extract tweet text (body)
            text_elem = tweet_div.find("p", class_="tweet-text")
            body = text_elem.get_text(strip=True) if text_elem else ""

            # Extract engagement metrics (likes, replies, retweets)
            stats = tweet_div.find("div", class_="tweet-stats")
            score = 0
            comment_count = 0

            if stats:
                # Parse engagement numbers
                stat_items = stats.find_all("span", class_="stat-count")
                for stat in stat_items:
                    count_text = stat.get_text(strip=True)
                    # Could be likes, replies, or retweets - sum them as engagement
                    try:
                        count = self._parse_count(count_text)
                        score += count
                    except ValueError:
                        pass

            # Nitter doesn't always show reply counts, use score as proxy
            if score == 0:
                score = 1  # Ensure we have a minimum

            return RawPost(
                id=f"tw_{tweet_id}",
                platform="twitter",
                source="twitter_search",
                title="",  # Twitter tweets don't have titles
                body=body,
                author=author,
                score=score,
                comment_count=comment_count,
                url=full_url,
                created_at=created_at,
            )

        except Exception as e:
            logger.debug(f"Failed to parse individual tweet: {e}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse Nitter timestamp formats.

        Nitter uses formats like:
          - "3h" (3 hours ago)
          - "5m" (5 minutes ago)
          - "2d" (2 days ago)
          - "Mar 22" (month/day)
          - "2026-03-22 10:30" (full datetime)
        """
        if not timestamp_str:
            return datetime.utcnow()

        timestamp_str = timestamp_str.lower().strip()
        now = datetime.utcnow()

        # Parse relative times
        if "m" in timestamp_str and "min" not in timestamp_str:
            # "5m" format
            try:
                minutes = int(timestamp_str.replace("m", "").strip())
                return now - timedelta(minutes=minutes)
            except (ValueError, AttributeError):
                pass

        if "h" in timestamp_str:
            # "3h" format
            try:
                hours = int(timestamp_str.replace("h", "").strip())
                return now - timedelta(hours=hours)
            except (ValueError, AttributeError):
                pass

        if "d" in timestamp_str:
            # "2d" format
            try:
                days = int(timestamp_str.replace("d", "").strip())
                return now - timedelta(days=days)
            except (ValueError, AttributeError):
                pass

        # Try to parse as ISO format or other standard formats
        for fmt in [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%b %d %H:%M",
            "%b %d",
        ]:
            try:
                parsed = datetime.strptime(timestamp_str, fmt)
                if "%Y" not in fmt:
                    # If year wasn't in format, assume current year
                    parsed = parsed.replace(year=now.year)
                return parsed
            except ValueError:
                continue

        logger.debug(f"Could not parse timestamp: {timestamp_str}")
        return now

    @staticmethod
    def _parse_count(count_str: str) -> int:
        """Parse engagement count strings like '1.2K', '500', '1M'."""
        count_str = count_str.strip().upper()

        if not count_str:
            return 0

        multipliers = {"K": 1000, "M": 1000000, "B": 1000000000}

        for suffix, multiplier in multipliers.items():
            if suffix in count_str:
                try:
                    num = float(count_str.replace(suffix, "").strip())
                    return int(num * multiplier)
                except ValueError:
                    return 0

        try:
            return int(count_str)
        except ValueError:
            return 0

    def fetch_thread(self, tweet_id: str) -> tuple[Optional[RawPost], list[RawPost]]:
        """Fetch a tweet and its replies.

        Note: Nitter doesn't provide full conversation threading like Twitter's API.
        This returns the tweet and attempts to fetch replies, but with limitations.

        Args:
            tweet_id: Twitter tweet ID (with or without tw_ prefix).

        Returns:
            (top_tweet, flat_list_of_replies)
        """
        clean_id = tweet_id.replace("tw_", "")

        # Try to fetch from a specific tweet URL
        # Note: We need the username but Nitter URLs use numeric IDs
        # For now, we can't reliably reconstruct without more info
        logger.warning(
            "Twitter adapter has limited thread support. "
            "Nitter doesn't expose full conversation trees. "
            "Recommend using official Twitter API v2 for conversation analysis."
        )
        return None, []
