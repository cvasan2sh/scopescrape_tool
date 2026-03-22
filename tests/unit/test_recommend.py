"""Tests for the ICP-to-subreddit recommendation engine."""

from __future__ import annotations

import pytest

from scopescrape.recommend import Recommender, recommend_for_icp


class TestRecommender:
    """Test the Recommender class."""

    @pytest.fixture
    def recommender(self):
        """Create a Recommender instance."""
        return Recommender()

    def test_initialize(self, recommender):
        """Test that recommender initializes with subreddit taxonomy."""
        assert recommender.subreddits is not None
        assert len(recommender.subreddits) > 80  # Should have 80+ subreddits
        assert any(sub.name == "r/saas" for sub in recommender.subreddits)
        assert any(sub.name == "r/programming" for sub in recommender.subreddits)

    def test_crm_recommendation(self, recommender):
        """Test CRM for real estate agents ICP."""
        result = recommender.recommend("CRM for real estate agents")

        assert result.icp_summary == "CRM for real estate agents"
        assert len(result.subreddits) > 0
        assert len(result.keywords) > 0
        assert "reddit" in result.platforms

        # Should recommend real estate subreddits
        sub_names = [sub["name"] for sub in result.subreddits]
        assert any("realestate" in name.lower() for name in sub_names)

        # Should include CRM-related keywords
        assert any("crm" in kw.lower() for kw in result.keywords)

        # Should be top-ranked (rel > 0.3)
        top_sub = result.subreddits[0]
        assert top_sub["relevance"] >= 0.2

    def test_developer_tool_recommendation(self, recommender):
        """Test developer tool ICP."""
        result = recommender.recommend("API testing tool for developers")

        assert len(result.subreddits) > 0
        sub_names = [sub["name"] for sub in result.subreddits]

        # Should include developer-focused subreddits
        assert any(
            name in ["r/programming", "r/webdev", "r/devops", "r/node"] for name in sub_names
        )

        # Should recommend developer platforms
        assert "github" in result.platforms or "stackoverflow" in result.platforms

    def test_marketing_tool_recommendation(self, recommender):
        """Test marketing tool ICP."""
        result = recommender.recommend("Email marketing automation for B2B companies")

        assert len(result.subreddits) > 0
        assert len(result.keywords) > 0

        sub_names = [sub["name"] for sub in result.subreddits]

        # Should include marketing subreddits
        assert any("marketing" in name.lower() for name in sub_names)

        # Should recommend Product Hunt and Twitter
        assert "producthunt" in result.platforms

        # Should include marketing keywords
        assert any(
            kw in [k.lower() for k in result.keywords]
            for kw in ["email", "marketing", "automation"]
        )

    def test_saas_product_recommendation(self, recommender):
        """Test generic SaaS product ICP."""
        result = recommender.recommend("SaaS project management tool")

        assert len(result.subreddits) > 0
        sub_names = [sub["name"] for sub in result.subreddits]

        # Should include SaaS and PM subreddits
        assert any(name in ["r/saas", "r/startups", "r/productivity"] for name in sub_names)

        # Should recommend multiple platforms
        assert len(result.platforms) >= 2

    def test_empty_icp(self, recommender):
        """Test that empty ICP returns empty result."""
        result = recommender.recommend("")

        assert result.icp_summary == ""
        assert result.subreddits == []
        assert result.keywords == []
        assert result.platforms == []

    def test_whitespace_only_icp(self, recommender):
        """Test that whitespace-only ICP returns empty result."""
        result = recommender.recommend("   ")

        assert result.subreddits == []

    def test_keyword_generation(self, recommender):
        """Test keyword generation from ICP text."""
        keywords = recommender._generate_keywords("crm for real estate agents")

        assert len(keywords) > 0
        assert "crm" in [kw.lower() for kw in keywords]
        assert "real" in [kw.lower() for kw in keywords]

        # Should include pain-point phrases
        assert any("frustrated" in kw.lower() for kw in keywords)

    def test_competitor_mapping(self, recommender):
        """Test that competitor mapping is used."""
        keywords = recommender._generate_keywords("crm tool")

        # Should include known CRM competitors
        competitor_matches = [
            kw for kw in keywords if kw.lower() in ["salesforce", "hubspot", "pipedrive"]
        ]
        assert len(competitor_matches) > 0

    def test_relevance_scoring(self, recommender):
        """Test that subreddit scoring works."""
        keywords = ["marketing", "automation", "email", "saas"]
        scored = recommender._score_subreddits(keywords)

        assert len(scored) > 0

        # Should rank marketing-focused subs high
        top_names = [sub["name"] for sub in scored[:5]]
        assert any("marketing" in name.lower() for name in top_names)

        # Relevance should be between 0 and 1
        for sub in scored:
            assert 0 <= sub["relevance"] <= 1

    def test_platform_recommendation_dev_tool(self, recommender):
        """Test platform recommendation for dev tools."""
        keywords = [
            {"name": "r/programming", "relevance": 0.9},
            {"name": "r/webdev", "relevance": 0.8},
        ]
        platforms = recommender._recommend_platforms("api testing tool for developers", keywords)

        assert "github" in platforms
        assert "stackoverflow" in platforms

    def test_platform_recommendation_marketing(self, recommender):
        """Test platform recommendation for marketing tools."""
        keywords = [{"name": "r/marketing", "relevance": 0.9}]
        platforms = recommender._recommend_platforms("email marketing for saas", keywords)

        assert "producthunt" in platforms
        assert "twitter" in platforms

    def test_platform_recommendation_indie(self, recommender):
        """Test platform recommendation for indie/startup tools."""
        keywords = [{"name": "r/indiehackers", "relevance": 0.9}]
        platforms = recommender._recommend_platforms("bootstrapped saas starter kit", keywords)

        assert "indiehackers" in platforms

    def test_subreddit_categories(self, recommender):
        """Test that subreddit taxonomy covers key categories."""
        categories = {sub.category for sub in recommender.subreddits}

        assert "SaaS" in categories
        assert "Developer Tools" in categories
        assert "AI/ML" in categories
        assert "Marketing" in categories
        assert "E-commerce" in categories

    def test_relevance_score_range(self, recommender):
        """Test that relevance scores are properly formatted."""
        result = recommender.recommend("CRM software for B2B sales teams")

        for sub in result.subreddits:
            # Should be rounded to 2 decimal places
            assert isinstance(sub["relevance"], float)
            assert 0 <= sub["relevance"] <= 1.0

    def test_reason_field_populated(self, recommender):
        """Test that recommendation reasons are provided."""
        result = recommender.recommend("project management tool for teams")

        for sub in result.subreddits:
            assert "reason" in sub
            assert len(sub["reason"]) > 0
            assert "Matches" in sub["reason"] or "matches" in sub["reason"]


class TestRecommendForICP:
    """Test the convenience function."""

    def test_convenience_function(self):
        """Test the recommend_for_icp convenience function."""
        result = recommend_for_icp("CRM for startups")

        assert result is not None
        assert len(result.subreddits) > 0
        assert len(result.keywords) > 0
        assert result.icp_summary == "CRM for startups"

    def test_returns_correct_type(self):
        """Test that function returns RecommendationResult."""
        from scopescrape.recommend import RecommendationResult

        result = recommend_for_icp("AI writing assistant")
        assert isinstance(result, RecommendationResult)
