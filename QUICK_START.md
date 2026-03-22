# ICP Recommendation Engine - Quick Start Guide

## Installation

The recommendation engine is built into ScopeScrape. No additional installation needed.

## Quick Examples

### 1. Get Recommendations for Your ICP

```bash
scopescrape recommend --icp "CRM for real estate agents"
```

Output shows:
- Top recommended subreddits with relevance scores
- Keywords to use when searching
- Platforms to focus on

### 2. Use ICP to Auto-Populate a Scan

```bash
scopescrape scan --icp "Email marketing automation for B2B companies"
```

This automatically:
- Selects the 5 most relevant subreddits
- Generates 5 targeted keywords
- Chooses the best platform to scan

### 3. Use the Python API

```python
from scopescrape.recommend import recommend_for_icp

result = recommend_for_icp("AI writing assistant for content creators")

# Access recommendations
for sub in result.subreddits[:3]:
    print(f"{sub['name']}: {sub['relevance']*100:.0f}% relevance")

print("Keywords:", ', '.join(result.keywords[:5]))
print("Platforms:", ', '.join(result.platforms))
```

### 4. Use the Web API

```bash
curl -X POST http://localhost:8888/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"icp": "Project management tool for remote teams"}'
```

## What Gets Recommended

### Subreddits
- 95+ curated subreddits across 15+ industries
- Ranked by relevance to your ICP
- Each recommendation explains why it matches

### Keywords
- Extracted from your ICP
- Pain-point oriented: "frustrated with X", "struggling with X"
- Competitor-aware: "alternative to Salesforce", etc.
- Solution focused: "best X for Y"

### Platforms
- **Developer tools**: Reddit, GitHub, Stack Overflow, Hacker News
- **Marketing tools**: Reddit, Product Hunt, Twitter
- **SaaS products**: All platforms
- **Indie products**: Indie Hackers, Product Hunt

## Examples by Product Type

### SaaS Product
```bash
scopescrape recommend --icp "Project management tool for remote teams"

# Recommends: r/saas, r/remotejobs, r/productivity
# Platforms: reddit, producthunt, twitter
```

### Developer Tool
```bash
scopescrape recommend --icp "API testing tool for developers"

# Recommends: r/programming, r/webdev, r/devops
# Platforms: reddit, github, stackoverflow, hn
```

### Marketing Product
```bash
scopescrape recommend --icp "Email marketing automation for SaaS"

# Recommends: r/emailmarketing, r/marketing, r/SaaS_Sales
# Platforms: reddit, producthunt, twitter, hn
```

### E-commerce Product
```bash
scopescrape recommend --icp "Inventory management for Shopify stores"

# Recommends: r/shopify, r/ecommerce, r/dropship
# Platforms: reddit, producthunt
```

### Real Estate Product
```bash
scopescrape recommend --icp "CRM for real estate agents"

# Recommends: r/realestate, r/RealEstateInvesting
# Platforms: reddit
```

## Tips

1. **Be Specific**: More detail in your ICP produces better recommendations
   - ✓ "CRM for small real estate teams (2-5 people)"
   - ✗ "CRM"

2. **Include the Audience**: Mention who your customers are
   - ✓ "Email marketing tool for e-commerce brands"
   - ✗ "Email marketing tool"

3. **Mention the Problem**: What pain point are you solving?
   - ✓ "API monitoring tool for DevOps teams managing microservices"
   - ✗ "API tool"

4. **Use with Scan**: Combine recommendations with automated scanning
   - First: `scopescrape recommend --icp "your description"`
   - Then: `scopescrape scan --icp "your description"`

## Command Reference

```bash
# Get recommendations
scopescrape recommend --icp "YOUR ICP DESCRIPTION"

# Scan with auto-populated recommendations
scopescrape scan --icp "YOUR ICP DESCRIPTION"

# Override recommendations if needed
scopescrape scan --icp "YOUR ICP" --subreddits "custom1,custom2" --keywords "custom,keywords"

# Dry run to see what would be scanned
scopescrape scan --icp "YOUR ICP" --dry-run

# Start web UI (with /api/recommend endpoint)
scopescrape web --port 8888
```

## What's Being Measured

The recommendations are based on:
- **Industry category** - What industry are you in?
- **Product type** - SaaS, developer tool, marketing tool, etc.
- **Target audience** - Who are your customers?
- **Problem being solved** - What pain points does your product address?

No external APIs or machine learning - pure local mapping based on curated community data.

## Need Help?

```bash
# See full command options
scopescrape recommend --help
scopescrape scan --help

# View the recommendation engine code
cat src/scopescrape/recommend.py

# Run tests
pytest tests/unit/test_recommend.py -v
```

That's it! The recommendation engine helps you discover where your customers hang out and what they're talking about.
