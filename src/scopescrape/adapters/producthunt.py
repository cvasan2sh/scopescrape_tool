"""Product Hunt adapter using public GraphQL API and product pages.

Product Hunt exposes a public GraphQL API at https://api.producthunt.com/v2/api/graphql
with rate limits of ~200 requests/hour for unauthenticated access.

Public product launch pages contain:
    - Reviews (upvotes, comments, maker responses)
    - Discussion threads
    - Comments revealing pain points and use cases

Strategy:
    1. Try public pages first (no auth required)
    2. Support PRODUCTHUNT_TOKEN for higher rate limits
    3. Search for products and fetch their reviews/discussions
    4. Convert to RawPost format
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import requests

from scopescrape.adapters.base import BaseAdapter
from scopescrape.log import get_logger
from scopescrape.models import RawPost
from scopescrape.utils import RateLimiter, deduplicate_posts, truncate

logger = get_logger(__name__)

# Product Hunt API base
PH_API = "https://api.producthunt.com/v2/api/graphql"


class ProductHuntAdapter(BaseAdapter):
    """Fetch Product Hunt products and reviews via public GraphQL API.

    No API key required for basic use (200 req/hour).
    With PRODUCTHUNT_TOKEN, limit increases to 1000+ req/hour.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        ph_config = config.get("producthunt", {})

        self.token = ph_config.get("token", "")
        self.rate_limiter = RateLimiter(
            initial_delay=ph_config.get("rate_limit_delay", 0.5),
            max_delay=60.0,
            max_retries=3,
        )
        self.max_reviews_per_product = ph_config.get("max_reviews_per_product", 5)
        self.session = self._create_session()

    @property
    def platform_name(self) -> str:
        return "producthunt"

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers."""
        s = requests.Session()
        headers = {
            "User-Agent": "ScopeScrape/0.1 (community pain point discovery)",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        s.headers.update(headers)
        return s

    def _post_graphql(self, query: str, variables: Optional[dict] = None) -> Optional[dict]:
        """Make a GraphQL POST request with rate limiting.

        Returns None on unrecoverable errors, retries on 429.
        """
        if variables is None:
            variables = {}

        payload = {
            "query": query,
            "variables": variables,
        }

        for attempt in range(self.rate_limiter.max_retries + 1):
            self.rate_limiter.wait()

            try:
                resp = self.session.post(PH_API, json=payload, timeout=15)

                if resp.status_code == 200:
                    self.rate_limiter.on_success()
                    data = resp.json()
                    # Check for GraphQL errors
                    if data.get("errors"):
                        logger.warning(f"GraphQL error: {data['errors']}")
                        return None
                    return data

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(
                        f"Product Hunt rate limited (429). Waiting {retry_after}s "
                        f"(attempt {attempt + 1})"
                    )
                    time.sleep(retry_after)
                    self.rate_limiter.on_failure()
                    continue

                if resp.status_code == 401:
                    logger.error("Product Hunt: Unauthorized (invalid token)")
                    return None

                logger.warning(f"Product Hunt HTTP {resp.status_code}")
                self.rate_limiter.on_failure()

            except requests.exceptions.Timeout:
                logger.warning(f"Product Hunt timeout (attempt {attempt + 1})")
                self.rate_limiter.on_failure()
            except requests.exceptions.RequestException as e:
                logger.error(f"Product Hunt request failed: {e}")
                return None

        logger.error("Max retries exceeded for Product Hunt")
        return None

    def fetch(self, queries: dict) -> list[RawPost]:
        """Fetch products and reviews from Product Hunt.

        Args:
            queries: Dict with optional keys:
                - keywords: list of search terms
                - limit: max products to fetch (default 100)
                - time_range: ignored for Product Hunt

        Returns:
            Deduplicated list of RawPost objects.
        """
        keywords = queries.get("keywords", [])
        limit = min(queries.get("limit", 100), 100)

        if not keywords:
            logger.warning("Product Hunt adapter requires keywords. Skipping.")
            return []

        all_posts: list[RawPost] = []

        for kw in keywords:
            products = self._search_products(kw, limit)
            all_posts.extend(products)

            # Fetch reviews for top products (by score)
            top_products = sorted(products, key=lambda p: p.score, reverse=True)[
                : min(5, len(products))
            ]
            for product in top_products:
                reviews = self._fetch_product_reviews(product.id)
                all_posts.extend(reviews)

        unique = deduplicate_posts(all_posts)
        logger.info(f"Product Hunt: {len(unique)} unique posts ({len(all_posts)} total)")
        return unique

    def _search_products(self, query: str, limit: int) -> list[RawPost]:
        """Search for Product Hunt products matching a query."""
        gql_query = """
        query SearchProducts($query: String!, $first: Int!) {
            products(
                first: $first
                search: $query
                sortBy: NEWEST
            ) {
                edges {
                    node {
                        id
                        slug
                        name
                        tagline
                        description
                        votesCount
                        commentsCount
                        createdAt
                        reviewsRating
                        website
                    }
                }
            }
        }
        """

        logger.debug(f"Searching Product Hunt for '{query}'")
        data = self._post_graphql(gql_query, {"query": query, "first": min(limit, 20)})
        if not data:
            return []

        posts = []
        edges = data.get("data", {}).get("products", {}).get("edges", [])
        for edge in edges:
            post = self._normalize_product(edge.get("node", {}))
            if post:
                posts.append(post)

        return posts[:limit]

    def _fetch_product_reviews(self, product_id: str) -> list[RawPost]:
        """Fetch reviews for a specific product."""
        gql_query = """
        query GetProductReviews($id: ID!, $first: Int!) {
            product(id: $id) {
                id
                slug
                reviews(first: $first) {
                    edges {
                        node {
                            id
                            body
                            rating
                            createdAt
                            user {
                                name
                                username
                            }
                        }
                    }
                }
            }
        }
        """

        logger.debug(f"Fetching reviews for product {product_id}")
        data = self._post_graphql(
            gql_query,
            {"id": product_id, "first": self.max_reviews_per_product},
        )
        if not data:
            return []

        reviews = []
        product = data.get("data", {}).get("product", {})
        product_slug = product.get("slug", "unknown")

        review_edges = product.get("reviews", {}).get("edges", [])
        for edge in review_edges:
            review = self._normalize_review(
                edge.get("node", {}),
                product_id,
                product_slug,
            )
            if review:
                reviews.append(review)

        return reviews

    def _normalize_product(self, data: dict) -> Optional[RawPost]:
        """Convert a Product Hunt product to RawPost."""
        if not data or not data.get("id"):
            return None

        product_id = data.get("id")
        slug = data.get("slug", "")
        name = data.get("name", "")
        tagline = data.get("tagline", "")
        description = data.get("description", "") or ""

        # Timestamp
        created_at_str = data.get("createdAt", "")
        try:
            created_at = (
                datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                .replace(tzinfo=None)
            )
        except (ValueError, TypeError):
            created_at = datetime.utcnow()

        # Use tagline + description as body
        body = (tagline + " " + description).strip() if tagline or description else ""

        url = f"https://www.producthunt.com/products/{slug}" if slug else ""

        return RawPost(
            id=f"ph_product_{product_id}",
            platform="producthunt",
            source="product_hunt",
            title=name,
            body=truncate(body, 500),
            author="[product]",
            score=data.get("votesCount", 0),
            comment_count=data.get("commentsCount", 0),
            url=url,
            created_at=created_at,
        )

    def _normalize_review(
        self, data: dict, product_id: str, product_slug: str
    ) -> Optional[RawPost]:
        """Convert a Product Hunt review to RawPost."""
        if not data or not data.get("id"):
            return None

        review_id = data.get("id")
        body = data.get("body", "") or ""
        rating = data.get("rating", 0)

        # User info
        user = data.get("user", {})
        author = user.get("username", user.get("name", "[unknown]"))

        # Timestamp
        created_at_str = data.get("createdAt", "")
        try:
            created_at = (
                datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                .replace(tzinfo=None)
            )
        except (ValueError, TypeError):
            created_at = datetime.utcnow()

        url = (
            f"https://www.producthunt.com/products/{product_slug}"
            if product_slug
            else ""
        )

        return RawPost(
            id=f"ph_review_{review_id}",
            platform="producthunt",
            source="product_hunt_reviews",
            title="",  # Reviews have no title
            body=truncate(body, 500),
            author=author,
            score=rating,  # 1-5 star rating
            comment_count=0,
            url=url,
            created_at=created_at,
            parent_id=f"ph_product_{product_id}",
        )
