"""SQLite storage layer for ScopeScrape.

Handles schema creation, post/signal/score persistence,
deduplication, and time-based cleanup for Reddit API compliance.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from scopescrape.models import PainPoint, RawPost, ScoredResult, SignalTier

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT,
    body TEXT,
    author TEXT,
    score INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    url TEXT,
    created_at DATETIME,
    parent_id TEXT,
    children_ids TEXT,
    inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL,
    phrase TEXT NOT NULL,
    tier TEXT NOT NULL,
    category TEXT,
    position INTEGER,
    context TEXT,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT UNIQUE NOT NULL,
    frequency_score REAL,
    intensity_score REAL,
    specificity_score REAL,
    recency_score REAL,
    composite_score REAL,
    sentiment_score REAL,
    entities TEXT,
    text_length INTEGER,
    FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(platform);
CREATE INDEX IF NOT EXISTS idx_scores_composite ON scores(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_signals_post_id ON signals(post_id);
"""


class Storage:
    """SQLite-backed storage for posts, signals, and scores.

    Supports both file-based and in-memory databases.
    """

    def __init__(self, config: dict):
        storage_config = config.get("storage", {})
        self.retention_hours = storage_config.get("retention_hours", 48)

        if storage_config.get("in_memory", False):
            self.db_path = ":memory:"
        else:
            self.db_path = str(Path(storage_config.get("db_path", "data.db")).expanduser())
            # Ensure parent directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_schema(self):
        """Create tables and indexes if they don't exist."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def save_posts(self, posts: list[RawPost]) -> int:
        """Insert posts, skipping duplicates. Returns count of new inserts."""
        inserted = 0
        for post in posts:
            try:
                self.conn.execute(
                    """INSERT OR IGNORE INTO posts
                    (id, platform, source, title, body, author, score,
                     comment_count, url, created_at, parent_id, children_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        post.id,
                        post.platform,
                        post.source,
                        post.title,
                        post.body,
                        post.author,
                        post.score,
                        post.comment_count,
                        post.url,
                        post.created_at.isoformat(),
                        post.parent_id,
                        json.dumps(post.children_ids),
                    ),
                )
                if self.conn.execute("SELECT changes()").fetchone()[0] > 0:
                    inserted += 1
            except sqlite3.Error:
                continue

        self.conn.commit()
        return inserted

    def save_signals(self, signals_by_post: dict[str, list[PainPoint]]) -> int:
        """Insert signal detections. Returns total count inserted."""
        count = 0
        for post_id, signals in signals_by_post.items():
            for signal in signals:
                self.conn.execute(
                    """INSERT INTO signals
                    (post_id, phrase, tier, category, position, context, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        post_id,
                        signal.phrase,
                        signal.tier.name,
                        signal.category,
                        signal.position,
                        signal.context,
                        signal.confidence,
                    ),
                )
                count += 1

        self.conn.commit()
        return count

    def save_scores(self, results: list[ScoredResult]) -> int:
        """Insert or update score records. Returns count saved."""
        count = 0
        for result in results:
            self.conn.execute(
                """INSERT OR REPLACE INTO scores
                (post_id, frequency_score, intensity_score, specificity_score,
                 recency_score, composite_score, sentiment_score, entities, text_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.post_id,
                    result.frequency_score,
                    result.intensity_score,
                    result.specificity_score,
                    result.recency_score,
                    result.composite_score,
                    result.sentiment_score,
                    json.dumps(result.entities),
                    result.text_length,
                ),
            )
            count += 1

        self.conn.commit()
        return count

    def save_results(
        self,
        posts: list[RawPost],
        signals: dict[str, list[PainPoint]],
        results: list[ScoredResult],
    ):
        """Convenience method: save posts, signals, and scores in one call."""
        self.save_posts(posts)
        self.save_signals(signals)
        self.save_scores(results)

    def cleanup_old(self) -> int:
        """Delete posts older than retention_hours. Returns count deleted."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
        cursor = self.conn.execute(
            "DELETE FROM posts WHERE created_at < ?", (cutoff.isoformat(),)
        )
        self.conn.commit()
        return cursor.rowcount

    def query_results(self, min_score: float = 0.0, limit: int = 100) -> list[dict]:
        """Query scored results joined with post metadata.

        Returns list of dicts sorted by composite_score descending.
        """
        rows = self.conn.execute(
            """SELECT p.id, p.platform, p.source, p.title, p.body, p.author,
                      p.score, p.url, p.created_at,
                      s.frequency_score, s.intensity_score, s.specificity_score,
                      s.recency_score, s.composite_score, s.sentiment_score,
                      s.entities, s.text_length
               FROM scores s
               JOIN posts p ON s.post_id = p.id
               WHERE s.composite_score >= ?
               ORDER BY s.composite_score DESC
               LIMIT ?""",
            (min_score, limit),
        ).fetchall()

        return [dict(row) for row in rows]

    def post_exists(self, post_id: str) -> bool:
        """Check if a post is already stored."""
        row = self.conn.execute("SELECT 1 FROM posts WHERE id = ?", (post_id,)).fetchone()
        return row is not None

    def count_posts(self) -> int:
        """Return total post count."""
        return self.conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]

    def count_signals(self) -> int:
        """Return total signal count."""
        return self.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
