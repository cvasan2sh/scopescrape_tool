# ScopeScrape

Open-source community pain point discovery tool. Scans Reddit and Hacker News for user frustrations, scores them across four dimensions, and exports actionable results.

**No API key required.** ScopeScrape uses Reddit's public JSON endpoints and HN's Algolia Search API.

## Install

```bash
pip install -e .
```

Optional extras:
```bash
pip install -e ".[nlp]"       # spaCy NER for better entity extraction
pip install -e ".[parquet]"   # Parquet export support
pip install -e ".[all]"       # Everything
```

## Quick start

```bash
# Scan r/saas for pain points about project management tools
python -m scopescrape scan \
  --subreddits saas \
  --keywords "frustrated,alternative to,struggling with" \
  --output json \
  --output-file results.json

# Scan Hacker News instead
python -m scopescrape scan \
  --keywords "pain point,broken,switched from" \
  --platforms hn \
  --output csv

# Check available platforms
python -m scopescrape platforms

# View current config (credentials masked)
python -m scopescrape config
```

## How it works

```
Fetch (Reddit/HN) -> Detect signals (67 phrases, 4 tiers) -> Score (4 dimensions) -> Export
```

**Signal detection:** 67 signal phrases organized into 4 tiers.

| Tier | Examples | Weight |
|------|----------|--------|
| PAIN_POINT | "frustrated with", "crashes constantly", "too slow" | 1.0 |
| EMOTIONAL | "i hate", "i wish", "nightmare", "missing feature" | 0.8 |
| COMPARISON | "alternative to", "switched from", "vs", "better than" | 0.6 |
| ASK | "anyone know", "looking for a tool", "how do i" | 0.4 |

**Scoring:** Each post is scored on four dimensions (0-10 each), then combined into a weighted composite.

| Dimension | Method | Default weight |
|-----------|--------|----------------|
| Frequency | BM25 relevance across corpus | 0.25 |
| Intensity | VADER sentiment + signal tier | 0.20 |
| Specificity | Named entity count + text length | 0.25 |
| Recency | Exponential time decay (1-week half-life) | 0.30 |

Posts below the `min_score` threshold (default 5.0) are dropped.

## CLI options

```
scopescrape scan [OPTIONS]

  --subreddits TEXT     Comma-separated subreddit names (e.g., saas,startups)
  --keywords TEXT       Comma-separated search terms
  --platforms [reddit|hn|all]
  --time-range [day|week|month|year]
  --limit INTEGER       Max posts to analyze (default 100)
  --min-score FLOAT     Minimum composite score (default 5.0)
  --output [json|csv|parquet]
  --output-file PATH    Destination file
  --dry-run             Show plan without executing
```

## Configuration

Copy `config/config.yaml.example` to `~/.scopescrape/config.yaml` or `./config.yaml`. All settings have sensible defaults. Environment variables override config file values.

## Project structure

```
src/scopescrape/
  cli.py              Click CLI
  config.py           YAML + env var config loader
  models.py           RawPost, PainPoint, ScoredResult dataclasses
  storage.py          SQLite persistence
  pipeline.py         Orchestrator
  adapters/
    reddit.py         Public JSON endpoint adapter
    hackernews.py     Algolia Search API adapter
  signals/
    detector.py       67-phrase signal taxonomy
  scoring/
    scorer.py         Composite scorer
    frequency.py      BM25 relevance
    intensity.py      VADER sentiment
    specificity.py    NER + text length
    recency.py        Time decay
  export/
    json_exporter.py
    csv_exporter.py
    parquet_exporter.py
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Known issues (v0.1)

See [BACKLOG.md](BACKLOG.md) for the full product backlog. Key items:

- **B-001:** Entity extractor is noisy without spaCy. Capitalized common words get tagged as entities.
- **B-002:** Frequency scorer returns 10.0 for nearly all posts when they come from the same search query.
- **B-003:** "broken" triggers false positives in promotional beta-feedback posts.

## License

MIT. See [LICENSE](LICENSE).

## Links

- Website: https://scopescrape.earnedconviction.com
- Blog: https://scopescrape.earnedconviction.com/blog
- Author: [Siva Pentakota](https://earnedconviction.com)
