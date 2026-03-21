# ScopeScrape Product Backlog

Last updated: March 22, 2026
Source: v0.1 first-run analysis (119 results from r/saas)

---

## Priority 1: Fix from first run

### B-001: Entity extractor produces too much noise without spaCy
**Severity:** High
**Found in:** First run analysis
**Problem:** The regex fallback entity extractor matches any capitalized word. Results include "You", "They", "Happy", "Curious", "Then", "Built" as entities. These are not products or tools.
**Root cause:** `PRODUCT_PATTERN` in `specificity.py` matches CamelCase and Title Case words without enough filtering. The stopword list misses common English words that happen to appear at sentence starts.
**Fix options:**
  - (a) Expand the stopword set to 200+ words (quick fix, diminishing returns)
  - (b) Add a curated product/tool dictionary and only match known names
  - (c) Install spaCy `en_core_web_sm` (35 MB) for real NER. This is the intended path but requires `pip install scopescrape[nlp]` and Python < 3.14 until spaCy ships 3.14 wheels
  - (d) Use a lightweight NER alternative that supports 3.14 (e.g., flair, stanza)
**Recommendation:** (a) as immediate patch, (c) as default once spaCy supports 3.14

### B-002: Frequency scorer returns 10.0 for nearly all posts
**Severity:** Medium
**Found in:** First run analysis (all top 15 posts had frequency_score=10.0)
**Problem:** BM25 scores all posts as maximally relevant because they were all fetched using the same search query. The query terms ("frustrated", "alternative to") appear in every result by definition. BM25 is comparing posts against themselves.
**Root cause:** The BM25 index is built from the same posts that are being scored. Every document in the corpus matches the query well because the search already filtered for relevance.
**Fix options:**
  - (a) Change frequency scoring to measure intra-corpus term overlap: how many other posts in the corpus share the same pain-related terms? Posts that echo a recurring theme score higher than one-off complaints
  - (b) Use TF-IDF instead of BM25, with IDF computed from a larger background corpus (e.g., 10k random Reddit posts)
  - (c) Score frequency by literal signal phrase overlap: if "frustrated with" appears in 30/160 posts, that phrase's frequency score is 30/160 = 1.875 on a 0-10 scale. Posts with high-frequency phrases score higher
  - (d) Deprecate frequency as a dimension in v0.1 and redistribute its weight to the other three dimensions
**Recommendation:** (c) for v0.1 patch, (b) for v0.2

### B-003: "broken" false positive in promotional posts
**Severity:** Low
**Found in:** Post t3_1rxxrcu ("tell me what's broken or missing")
**Problem:** The word "broken" triggered a PAIN_POINT/bug signal in a promotional post. The author was inviting beta feedback, not reporting a bug. Context: "I'm genuinely looking for people to test it and tell me what's broken or missing."
**Fix options:**
  - (a) Add negation detection: if "broken" is preceded by "what's" or "if anything is", reduce confidence
  - (b) Add a promotional post filter: if the post contains "I built", "launching", "beta", "sign up" in combination, flag it as self-promotion and reduce all signal scores by 50%
  - (c) Leave as-is for v0.1. The scoring system still ranks genuine complaints higher because promotional posts have positive sentiment (VADER), which lowers the intensity score
**Recommendation:** (c) for now, (b) for v0.2

---

## Priority 2: Improvements for v0.1.x

### B-004: Add comment fetching to Reddit adapter
**Status:** Not started
**Value:** High
**Problem:** v0.1 only analyzes post titles and bodies. Comments often contain the richest pain signals ("I switched from X because...", "The worst part about Y is..."). The adapter already has `fetch_thread()` implemented but the pipeline does not call it.
**Fix:** After fetching posts, call `fetch_thread()` for the top N posts (by upvotes) and run signal detection on comments too. Requires rate limit budget management since each thread fetch is 1 request.

### B-005: Add progress bar for long scans
**Status:** Not started
**Value:** Medium
**Problem:** A scan of 3 subreddits with keywords takes 30+ seconds. The user sees log lines but no progress indication.
**Fix:** Use Rich progress bar in the pipeline. Show: fetching [1/3 subreddits], detecting [50/160 posts], scoring [100/160 posts].

### B-006: Dedup by content similarity, not just post ID
**Status:** Not started
**Value:** Medium
**Problem:** Cross-posted content (same text, different subreddit) gets scored separately. The current dedup is by post ID only.
**Fix:** Add a content hash (first 500 chars, lowercased, stripped) and dedup on that before scoring.

### B-007: Add --subreddits-only mode (no keywords)
**Status:** Partially done
**Value:** Low
**Problem:** Running without keywords currently fetches the hot listing. Some users want to scan ALL recent posts in a subreddit for pain signals, not just search results.
**Fix:** Add a `--sort` flag (hot/new/top/rising) and when no keywords are given, fetch the listing and run signal detection on everything.

---

## Priority 3: v0.2 roadmap

### B-008: LLM-assisted signal classification
Pipe detected signals through an LLM (Claude or GPT) to classify whether the signal is genuine pain vs sarcasm, humor, or promotional language. Use structured output to return confidence and category.

### B-009: Hacker News adapter improvements
Add "Show HN" and "Ask HN" specific filters. Weight Show HN comments differently (they tend to contain direct product feedback).

### B-010: GitHub Discussions adapter
Implement the adapter pattern for GitHub Discussions via GraphQL API. Target repos with active community discussions (e.g., framework repos where users report pain points).

### B-011: Background corpus for better frequency scoring
Maintain a rolling 30-day background corpus per subreddit. Score frequency against this background to identify genuinely trending pain points vs one-time complaints.

### B-012: Self-promotion filter
Classify posts as self-promotional using heuristics: author post history, presence of links to own domain, "I built" + "check it out" patterns. Reduce signal confidence for these posts.

### B-013: Dashboard (Streamlit or similar)
Build a simple web UI for browsing results, filtering by tier/category/score, and drilling into individual posts.

---

## Completed

- [x] Phase A: Foundation (models, config, storage, CLI)
- [x] Phase B: Reddit public JSON adapter + HN Algolia adapter
- [x] Phase C: Signal detector (67 phrases, 4 tiers)
- [x] Phase D: Scoring engine (frequency, intensity, specificity, recency)
- [x] Phase E: Export layer (JSON, CSV, Parquet)
- [x] First live run: 160 posts fetched, 119 scored from r/saas
