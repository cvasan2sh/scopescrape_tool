"""ScopeScrape Web App — FastAPI backend.

Wraps the pipeline in a local web server with Server-Sent Events
for real-time scan progress. Serves a single-page frontend from
the bundled static/ directory.

Start via CLI:
    scopescrape web
    scopescrape web --port 8899
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scopescrape.config import load_config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="ScopeScrape", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"

# In-memory scan storage (local use only)
_scans: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    subreddits: str = ""          # comma-separated
    keywords: str = ""            # comma-separated
    platforms: str = "reddit"     # "reddit", "hn", "all"
    time_range: str = "week"      # "day", "week", "month", "year"
    limit: int = 100
    min_score: float = 5.0


class ScanStatus(BaseModel):
    scan_id: str
    status: str                   # "running", "completed", "failed"
    progress: str = ""
    result_count: int = 0
    results: list[dict] | None = None
    error: str | None = None
    started_at: str = ""
    finished_at: str | None = None


class RecommendRequest(BaseModel):
    icp: str


class RecommendResponse(BaseModel):
    subreddits: list[dict]
    keywords: list[str]
    platforms: list[str]
    icp_summary: str


# ---------------------------------------------------------------------------
# Capture pipeline logs as progress messages
# ---------------------------------------------------------------------------

class ProgressCapture(logging.Handler):
    """Intercepts scopescrape log messages to report scan progress."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord):
        if record.name.startswith("scopescrape"):
            self.messages.append(record.getMessage())


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/api/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Get ICP-to-subreddit recommendations."""
    if not req.icp or not req.icp.strip():
        raise HTTPException(400, "ICP description cannot be empty")

    from scopescrape.recommend import recommend_for_icp

    result = recommend_for_icp(req.icp)

    return RecommendResponse(
        subreddits=result.subreddits,
        keywords=result.keywords,
        platforms=result.platforms,
        icp_summary=result.icp_summary,
    )


@app.post("/api/scan", response_model=ScanStatus)
async def start_scan(req: ScanRequest):
    """Kick off a scan in a background thread."""
    if not req.subreddits and not req.keywords:
        raise HTTPException(400, "Provide at least one of subreddits or keywords")

    scan_id = uuid.uuid4().hex[:12]
    _scans[scan_id] = {
        "status": "running",
        "progress": "Initializing scan...",
        "results": None,
        "result_count": 0,
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": None,
    }

    # Run the scan in a thread so we don't block the server
    asyncio.get_event_loop().run_in_executor(
        None, _run_scan, scan_id, req
    )

    return ScanStatus(scan_id=scan_id, status="running", started_at=_scans[scan_id]["started_at"])


@app.get("/api/scan/{scan_id}", response_model=ScanStatus)
async def get_scan(scan_id: str):
    """Poll scan status and results."""
    if scan_id not in _scans:
        raise HTTPException(404, "Scan not found")

    s = _scans[scan_id]
    return ScanStatus(
        scan_id=scan_id,
        status=s["status"],
        progress=s["progress"],
        result_count=s["result_count"],
        results=s["results"] if s["status"] == "completed" else None,
        error=s["error"],
        started_at=s["started_at"],
        finished_at=s["finished_at"],
    )


@app.get("/api/scans")
async def list_scans():
    """List all scans in this session."""
    out = []
    for sid, s in _scans.items():
        out.append({
            "scan_id": sid,
            "status": s["status"],
            "result_count": s["result_count"],
            "started_at": s["started_at"],
            "finished_at": s["finished_at"],
        })
    return out


@app.post("/api/upload-results")
async def upload_results(payload: dict):
    """Upload a previously exported results.json for exploration."""
    results = payload.get("results", [])
    if not results:
        raise HTTPException(400, "No results in payload")

    scan_id = f"upload-{uuid.uuid4().hex[:8]}"
    _scans[scan_id] = {
        "status": "completed",
        "progress": "Uploaded",
        "results": results,
        "result_count": len(results),
        "error": None,
        "started_at": datetime.utcnow().isoformat(),
        "finished_at": datetime.utcnow().isoformat(),
    }
    return {"scan_id": scan_id, "result_count": len(results)}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>ScopeScrape</h1><p>Static files not found.</p>"


# ---------------------------------------------------------------------------
# Pipeline runner (runs in thread)
# ---------------------------------------------------------------------------

def _run_scan(scan_id: str, req: ScanRequest):
    """Execute the pipeline synchronously in a background thread."""
    handler = ProgressCapture()
    root_logger = logging.getLogger("scopescrape")
    root_logger.addHandler(handler)

    try:
        config = load_config()

        # Build queries dict
        queries: dict[str, Any] = {
            "limit": req.limit,
            "time_range": req.time_range,
        }
        if req.subreddits:
            queries["subreddits"] = [s.strip() for s in req.subreddits.split(",") if s.strip()]
        if req.keywords:
            queries["keywords"] = [k.strip() for k in req.keywords.split(",") if k.strip()]

        # Resolve platforms
        if req.platforms == "all":
            platforms = ["reddit", "hn"]
        else:
            platforms = [req.platforms]

        _scans[scan_id]["progress"] = f"Fetching from {', '.join(platforms)}..."

        # --- Phase 1: Fetch ---
        from scopescrape.models import RawPost
        from scopescrape.pipeline import Pipeline

        pipeline = Pipeline(config)
        all_posts: list[RawPost] = []

        for platform in platforms:
            adapter = pipeline._get_adapter(platform)
            posts = adapter.fetch(queries)
            all_posts.extend(posts)
            _scans[scan_id]["progress"] = f"Fetched {len(all_posts)} posts so far..."

        if not all_posts:
            _scans[scan_id]["status"] = "completed"
            _scans[scan_id]["progress"] = "No posts found"
            _scans[scan_id]["results"] = []
            _scans[scan_id]["finished_at"] = datetime.utcnow().isoformat()
            return

        # --- Phase 2: Detect ---
        _scans[scan_id]["progress"] = f"Detecting signals in {len(all_posts)} posts..."

        from scopescrape.signals.detector import SignalDetector

        detector = SignalDetector(config)
        all_signals = {}
        for post in all_posts:
            signals = detector.detect(post.full_text, post.id)
            if signals:
                all_signals[post.id] = signals

        _scans[scan_id]["progress"] = f"Signals found in {len(all_signals)}/{len(all_posts)} posts. Scoring..."

        # --- Phase 3: Score ---
        from scopescrape.scoring.scorer import Scorer

        scorer = Scorer(config)
        scored_results = []

        for post in all_posts:
            if post.id in all_signals:
                result = scorer.score(post, all_signals[post.id], all_posts)
                if result and result.composite_score >= req.min_score:
                    scored_results.append(result)

        # Sort by composite score descending
        scored_results.sort(key=lambda r: r.composite_score, reverse=True)

        # Convert to dicts for JSON response
        result_dicts = [r.to_dict() for r in scored_results]

        _scans[scan_id]["status"] = "completed"
        _scans[scan_id]["progress"] = f"Done. {len(result_dicts)} results above threshold."
        _scans[scan_id]["results"] = result_dicts
        _scans[scan_id]["result_count"] = len(result_dicts)
        _scans[scan_id]["finished_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        _scans[scan_id]["status"] = "failed"
        _scans[scan_id]["error"] = str(e)
        _scans[scan_id]["progress"] = f"Error: {e}"
        _scans[scan_id]["finished_at"] = datetime.utcnow().isoformat()

    finally:
        root_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Launcher (called from CLI)
# ---------------------------------------------------------------------------

def start_server(host: str = "127.0.0.1", port: int = 8888):
    """Start the uvicorn server."""
    import uvicorn

    print(f"\n  ScopeScrape Web UI → http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
