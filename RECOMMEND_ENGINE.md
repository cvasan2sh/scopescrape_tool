# ICP-to-Subreddit Recommendation Engine

## Overview

The recommendation engine helps users discover the most relevant subreddits, keywords, and platforms for their Ideal Customer Profile (ICP) without requiring any external API calls or ML models.

## Features

### 1. Smart Subreddit Recommendations
- 95+ curated subreddits organized by industry/niche
- TF-IDF-style keyword matching for relevance scoring
- Ranked by relevance with explanations of why each subreddit matches

**Categories include:**
- SaaS/Software (r/saas, r/startups, r/indiehackers, etc.)
- Developer Tools (r/programming, r/webdev, r/devops, r/reactjs, etc.)
- AI/ML (r/artificial, r/MachineLearning, r/ChatGPT, etc.)
- Marketing (r/marketing, r/SEO, r/emailmarketing, etc.)
- E-commerce (r/ecommerce, r/shopify, r/dropship, etc.)
- Finance/Fintech (r/fintech, r/investing, r/CryptoCurrency, etc.)
- Healthcare (r/healthIT, r/medicine, r/nursing, etc.)
- Real Estate (r/realestate, r/RealEstateInvesting, etc.)
- And more...

### 2. Pain-Oriented Keyword Generation
Generated keywords include:
- Base terminology extracted from ICP
- Pain-oriented phrases: "frustrated with X", "struggling with X"
- Competitor alternatives: "alternative to X"
- Solution-oriented: "best X for Y"
- Known competitors for common niches (Salesforce, HubSpot, Shopify, etc.)

### 3. Platform Recommendations
Based on ICP characteristics:
- **Developer tools** → GitHub, Stack Overflow, Hacker News, Reddit
- **Marketing tools** → Product Hunt, Twitter, Reddit
- **SaaS products** → All platforms
- **Indie/Bootstrapped** → Indie Hackers, Product Hunt
- **General tech** → Hacker News

### 4. Zero External Dependencies
- No API calls required
- No LLM usage
- Pure local mapping system
- Fast performance (< 100ms per recommendation)

## Usage

### CLI Command

```bash
# Get recommendations for an ICP
scopescrape recommend --icp "CRM for real estate agents"

# Output includes:
# - Recommended subreddits with relevance scores
# - Keywords tailored to the ICP
# - Recommended platforms beyond Reddit
```

### CLI Integration with Scan

```bash
# Use ICP to auto-populate scan parameters
scopescrape scan --icp "Email marketing for B2B SaaS"

# Auto-populates:
# - Top 5 recommended subreddits
# - Top 5 recommended keywords
# - Most relevant platform
```

### Web API

```python
# POST /api/recommend
{
  "icp": "CRM for real estate agents"
}

# Response
{
  "subreddits": [
    {
      "name": "r/realestate",
      "relevance": 0.75,
      "reason": "Matches 3 keywords: real, agents, estate"
    },
    ...
  ],
  "keywords": ["crm", "real", "estate", "agents", ...],
  "platforms": ["reddit", "producthunt", ...],
  "icp_summary": "CRM for real estate agents"
}
```

### Python Library

```python
from scopescrape.recommend import recommend_for_icp, Recommender

# Quick recommendation
result = recommend_for_icp("AI writing assistant for marketers")

print(result.subreddits)    # List of recommended subreddits
print(result.keywords)      # Generated keywords
print(result.platforms)     # Recommended platforms
print(result.icp_summary)   # Parsed ICP summary

# Or use the class directly
recommender = Recommender()
result = recommender.recommend("Your ICP description")
```

## Architecture

### Core Classes

**`Subreddit`** - Metadata for a single subreddit:
- `name` - Subreddit name (e.g., "r/saas")
- `category` - Industry/niche category
- `typical_audience` - Target audience description
- `relevance_keywords` - Keywords that map to this subreddit

**`Recommender`** - Main engine:
- `recommend(icp_text: str)` - Generate recommendations
- `_generate_keywords()` - Extract and expand keywords from ICP
- `_score_subreddits()` - Rank subreddits by relevance
- `_recommend_platforms()` - Suggest cross-platform targets

**`RecommendationResult`** - Output structure:
- `subreddits` - List of recommended subreddits with scores
- `keywords` - Generated keywords for search
- `platforms` - Recommended platforms
- `icp_summary` - Normalized ICP description

### Scoring Algorithm

1. **Keyword Extraction** - Split ICP into base terms, remove stop words
2. **Expansion** - Add pain-phrases, competitors, role-specific terms
3. **Matching** - For each subreddit, count keyword overlaps
4. **Normalization** - Normalize score by subreddit taxonomy size
5. **Ranking** - Sort by relevance score (0.0 to 1.0)
6. **Explanation** - Generate reason string explaining matches

## Examples

### Example 1: CRM for Real Estate
```bash
$ scopescrape recommend --icp "CRM for real estate agents"

Output:
- r/realestate (75% relevance)
- r/RealEstateInvesting (67% relevance)
- r/SaaS_Sales (40% relevance)
- Keywords: crm, real, estate, agents, frustrated with crm, ...
- Platforms: reddit
```

### Example 2: Email Marketing Automation
```bash
$ scopescrape recommend --icp "Email marketing automation for B2B SaaS"

Output:
- r/emailmarketing (100% relevance)
- r/marketing (100% relevance)
- r/GrowMyBusiness (100% relevance)
- Keywords: email, marketing, automation, saas, ...
- Platforms: reddit, producthunt, twitter, hn
```

### Example 3: API Testing Tool
```bash
$ scopescrape recommend --icp "API testing tool for developers"

Output:
- r/programming (80%+ relevance)
- r/webdev (75%+ relevance)
- r/devops (70%+ relevance)
- Keywords: api, testing, developer, tools, ...
- Platforms: reddit, github, stackoverflow, hn
```

## Testing

All recommendation engine functionality is tested with 18 unit tests:

```bash
pytest tests/unit/test_recommend.py -v

# Tests cover:
# - Subreddit taxonomy initialization
# - ICP-based recommendations
# - Keyword generation with pain-oriented phrases
# - Competitor mapping
# - Relevance scoring
# - Platform recommendations by niche
# - Edge cases (empty input, whitespace, etc.)
```

All tests pass successfully.

## Files

- `/src/scopescrape/recommend.py` - Core recommendation engine (400+ lines)
- `/src/scopescrape/cli.py` - Updated with `recommend` command and `--icp` flag
- `/src/scopescrape/web.py` - Updated with `/api/recommend` endpoint
- `/tests/unit/test_recommend.py` - Comprehensive test suite (280+ lines)

## Future Enhancements

Potential additions (without breaking local-first design):
- Community-contributed taxonomy expansions
- Industry-specific keyword templates
- Subreddit subscriber count considerations
- Time-decay for recommendations (trending niches)
- Custom competitor mappings per user
- Recommendation confidence intervals
